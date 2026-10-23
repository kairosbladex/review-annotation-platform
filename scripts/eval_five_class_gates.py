#!/usr/bin/env python3
"""五类业务门禁与三级级联路由的离线回放评估（可复现）。

评估四组指标：
  1. 车/三轮车几何门禁：IoU > 0.75 同目标双框抑制率、containment > 0.88 三轮车车斗误报拦截率；
  2. 三级级联路由：Tier1/Tier2/Tier3 分流比、blue_illegal_roof 100% 强制进入 Tier3 的合规率；
  3. 级联阈值改造前后对比：6713 条真实候选的 VLM 卸载比例、142 条真实金标上的新增 FP 与端到端 F1/FAR；
  4. 阈值搜索与逐类表：build_by_class_table / find_optimal_threshold_for_far(max_far=0.10)。

数据来源：
  - 真实回放集（默认）：data/ktd-test-real-monitoring-full-20260924/output/raw_predictions.jsonl
    （6713 条候选、1416 个事件、五类目标、含 detector_confidence 与 bbox_px）
  - 真实人工金标（默认）：data/ktd-clean-source-full-evaluation-20260924/predictions_scored.jsonl
    （142 条，覆盖 车/人员 两类；五类人工金标在仓库内不存在，故五类逐类表使用显式标注的合成基准）

用法：
    PYTHONPATH=.:backend .venv/bin/python scripts/eval_five_class_gates.py \
        --output-dir /tmp/ktd_phase3_eval
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))
sys.path.insert(0, str(ROOT / "scripts"))

from app.services.ktd_recheck import (  # noqa: E402
    TIER1_AUTO_CONFIRM_THRESHOLD,
    TIER1_ELIGIBLE_CLASSES,
    TIER2_AUTO_REJECT_THRESHOLD,
    _canonical_geom_class,
    apply_pre_llm_geometric_gate,
    compute_bbox_iou,
    compute_containment_ratio,
    evaluate_cascade_tier,
    partition_candidates_by_cascade_tier,
)
from qwen3vl_verify import (  # noqa: E402
    find_optimal_threshold_for_far,
    format_by_class_markdown_table,
    summarize_predictions,
)

DEFAULT_CANDIDATES = ROOT / "data/ktd-test-real-monitoring-full-20260924/output/raw_predictions.jsonl"
DEFAULT_GOLD = ROOT / "data/ktd-clean-source-full-evaluation-20260924/predictions_scored.jsonl"
DEFAULT_MANIFEST = ROOT / "data/ktd-test-real-monitoring-full-20260924/input/inference_manifest.jsonl"
DEFAULT_GOLD_OVERLAP = ROOT / "data/ktd-test-real-monitoring-full-20260924/gold_overlap_map.jsonl"

# 改造前（0.92，任意非 roof 类均可自动确认） vs 改造后（0.85，仅刚体类白名单）。
BASELINE_TIER1_THRESHOLD = 0.92

FIVE_CLASSES = ("人员", "车辆", "三轮车", "蓝色违建顶棚", "道路裂缝")


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


# --------------------------------------------------------------------------
# 1. 几何门禁：受控扫描（精确控制 IoU / containment，量化检出率与误报率）
# --------------------------------------------------------------------------
def _shifted_same_size_box(iou: float, width: float = 0.30) -> tuple[list[float], list[float]]:
    """构造两个同尺寸框，使 IoU 精确等于目标值（仅水平平移，containment 不超阈值）。"""
    dx = width * (1.0 - iou) / (1.0 + iou)
    base = [0.20, 0.30, 0.20 + width, 0.30 + width]
    shifted = [base[0] + dx, base[1], base[2] + dx, base[3]]
    return base, shifted


def _partial_overlap_boxes(containment: float) -> tuple[list[float], list[float]]:
    """构造 containment_ratio 精确等于目标值的两个框。

    containment = inner∩outer 面积 / inner 面积。小框部分移出大框，
    同时保持 IoU 很低（<0.05），从而把 containment 规则与 vehicle/tricycle IoU 规则隔离开。
    """
    outer = [0.10, 0.10, 0.90, 0.90]
    side = 0.20
    inner_y1 = 0.40
    # 小框右边界越出大框，仅前 containment*side 的宽度落在框内
    inner_x1 = outer[2] - containment * side
    inner = [inner_x1, inner_y1, inner_x1 + side, inner_y1 + side]
    return inner, outer


def evaluate_geometry_sweep() -> dict[str, Any]:
    # (a) 同目标双框：vehicle vs tricycle，IoU 精确扫描
    iou_rows = []
    for step in range(50, 100):
        target = step / 100.0
        base, shifted = _shifted_same_size_box(target)
        gated = apply_pre_llm_geometric_gate(
            [
                {"target_index": 0, "class_name": "车辆", "bbox": base},
                {"target_index": 1, "class_name": "三轮车", "bbox": shifted},
            ]
        )
        flagged = any("cross_class_iou_conflict" in (g.get("geometric_warning") or "") for g in gated)
        iou_rows.append(
            {
                "target_iou": round(target, 4),
                "measured_iou": round(compute_bbox_iou(base, shifted), 4),
                "flagged": flagged,
            }
        )

    above = [r for r in iou_rows if r["measured_iou"] > 0.75]
    below = [r for r in iou_rows if r["measured_iou"] <= 0.75]

    # (b) 三轮车车斗误报：tricycle 小框落在大 vehicle 框内，containment 精确扫描
    tri_rows = []
    for step in range(700, 1000, 5):
        target = step / 1000.0
        inner, outer = _partial_overlap_boxes(target)
        gated = apply_pre_llm_geometric_gate(
            [
                {"target_index": 0, "class_name": "三轮车", "bbox": inner},
                {"target_index": 1, "class_name": "车辆", "bbox": outer},
            ]
        )
        flagged = any("part_in_whole_containment_warning" in (g.get("geometric_warning") or "") for g in gated)
        tri_rows.append(
            {
                "target_containment": round(target, 4),
                "measured_containment": round(compute_containment_ratio(inner, outer), 4),
                "measured_iou": round(compute_bbox_iou(inner, outer), 4),
                "flagged": flagged,
            }
        )

    # (c) 隔离 containment 规则：person 落在 vehicle 内，不触发 vehicle/tricycle IoU 规则
    iso_rows = []
    for step in range(700, 1000, 5):
        target = step / 1000.0
        inner, outer = _partial_overlap_boxes(target)
        gated = apply_pre_llm_geometric_gate(
            [
                {"target_index": 0, "class_name": "人员", "bbox": inner},
                {"target_index": 1, "class_name": "车辆", "bbox": outer},
            ]
        )
        flagged = any("part_in_whole_containment_warning" in (g.get("geometric_warning") or "") for g in gated)
        iso_rows.append(
            {
                "target_containment": round(target, 4),
                "measured_containment": round(compute_containment_ratio(inner, outer), 4),
                "flagged": flagged,
            }
        )

    def rate(rows: list[dict], pred) -> float:
        sel = [r for r in rows if pred(r)]
        return round(sum(1 for r in sel if r["flagged"]) / len(sel), 4) if sel else 0.0

    return {
        "iou_threshold": 0.75,
        "containment_threshold": 0.88,
        "same_target_double_box_suppression_rate": rate(above, lambda r: True),
        "same_target_double_box_false_flag_rate": rate(below, lambda r: True),
        "iou_scan_points": len(iou_rows),
        "iou_above_threshold_n": len(above),
        "iou_below_threshold_n": len(below),
        "tricycle_bucket_interception_rate": rate(
            tri_rows, lambda r: r["measured_containment"] > 0.88
        ),
        "tricycle_bucket_false_flag_rate": rate(
            tri_rows, lambda r: r["measured_containment"] <= 0.88
        ),
        "containment_only_interception_rate": rate(
            iso_rows, lambda r: r["target_containment"] > 0.88
        ),
        "containment_only_false_flag_rate": rate(
            iso_rows, lambda r: r["target_containment"] <= 0.88
        ),
        "iou_curve": iou_rows,
        "tricycle_containment_curve": tri_rows,
    }


# --------------------------------------------------------------------------
# 2. 真实回放：几何门禁 + 三级级联路由
# --------------------------------------------------------------------------
def evaluate_real_replay(candidates_path: Path) -> dict[str, Any]:
    rows = _read_jsonl(candidates_path)
    by_event: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        bbox = row.get("bbox_px")
        if not isinstance(bbox, list) or len(bbox) != 4:
            continue
        by_event[str(row.get("event_id"))].append(
            {
                "target_index": row.get("candidate_id"),
                "class_name": row.get("class_name"),
                "confidence": row.get("detector_confidence"),
                "bbox": bbox,
            }
        )

    tier_counts: Counter[str] = Counter()
    per_class: dict[str, Counter[str]] = defaultdict(Counter)
    geom_warn = 0
    cross_conflict = 0
    containment_warn = 0
    roof_total = 0
    roof_tier3 = 0
    total = 0

    for event_candidates in by_event.values():
        buckets = partition_candidates_by_cascade_tier(event_candidates)
        for cand in [c for bucket in buckets.values() for c in bucket]:
            tier = cand["cascade_routing"]["tier"]
            cls = str(cand.get("class_name") or "unknown")
            tier_counts[tier] += 1
            per_class[cls][tier] += 1
            total += 1
            warning = cand.get("geometric_warning") or ""
            if warning:
                geom_warn += 1
                if "cross_class_iou_conflict" in warning:
                    cross_conflict += 1
                if "part_in_whole_containment_warning" in warning:
                    containment_warn += 1
            if cls in {"蓝色违建顶棚", "违建", "blue_illegal_roof"}:
                roof_total += 1
                if tier == "tier3_vlm_verify":
                    roof_tier3 += 1

    five = ("车", "人员", "三轮车", "违建", "道路裂纹")
    focus = {cls: dict(per_class[cls]) for cls in five if cls in per_class}

    # Tier1 阈值敏感性：0.92 是否与真实检测器置信度分布匹配
    sensitivity = []
    for th in (0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.92, 0.95):
        counts: Counter[str] = Counter()
        for event_candidates in by_event.values():
            buckets = partition_candidates_by_cascade_tier(
                event_candidates, tier1_threshold=th
            )
            for tier, bucket in buckets.items():
                counts[tier] += len(bucket)
        sensitivity.append(
            {
                "tier1_threshold": th,
                "tier1_share": round(counts["tier1_auto_confirm"] / total, 4) if total else 0.0,
                "tier3_share": round(counts["tier3_vlm_verify"] / total, 4) if total else 0.0,
                "tier1_count": counts["tier1_auto_confirm"],
            }
        )

    return {
        "source": str(candidates_path.relative_to(ROOT)),
        "events": len(by_event),
        "candidates": total,
        "tier_counts": dict(tier_counts),
        "tier_ratios": {
            k: round(v / total, 4) if total else 0.0 for k, v in tier_counts.items()
        },
        "tier1_threshold_sensitivity": sensitivity,
        "thresholds": {
            "tier1_auto_confirm": TIER1_AUTO_CONFIRM_THRESHOLD,
            "tier2_auto_reject": TIER2_AUTO_REJECT_THRESHOLD,
        },
        "geometric_warning_candidates": geom_warn,
        "geometric_warning_ratio": round(geom_warn / total, 4) if total else 0.0,
        "vehicle_tricycle_iou_conflicts": cross_conflict,
        "containment_warnings": containment_warn,
        "blue_illegal_roof_total": roof_total,
        "blue_illegal_roof_tier3": roof_tier3,
        "blue_illegal_roof_tier3_compliance": (
            round(roof_tier3 / roof_total, 4) if roof_total else None
        ),
        "five_class_tier_breakdown": focus,
    }


# --------------------------------------------------------------------------
# 3. 逐类表与阈值搜索
# --------------------------------------------------------------------------
def evaluate_real_two_class(gold_path: Path) -> dict[str, Any]:
    """真实人工金标（车/人员）上的逐类表与阈值搜索。

    `hybrid_confidence` 是「模型对自身决策的置信度」，不是 P(true)。这里**不再**手工翻转
    分数，而是把原始记录直传 `find_optimal_threshold_for_far`，由
    `_resolve_positive_score` 统一折算，从而同时验证该折算逻辑本身。
    """
    rows = _read_jsonl(gold_path)
    records = []
    for r in rows:
        if r.get("human_label") not in {"true", "false"}:
            continue
        conf = r.get("hybrid_confidence")
        if not isinstance(conf, (int, float)):
            continue
        decision = str(r.get("hybrid_decision"))
        records.append(
            {
                "class_name": str(r["class_name"]),
                "gold": str(r["human_label"]),
                "prediction": decision,
                "raw_prediction": decision,
                "confidence": float(conf),
            }
        )
    summary = summarize_predictions(records)
    search = find_optimal_threshold_for_far(records, max_far=0.10)
    table = summary["by_class_table"]
    return {
        "source": str(gold_path.relative_to(ROOT)),
        "n": len(records),
        "classes": sorted({r["class_name"] for r in records}),
        "by_class_table": table,
        "markdown": format_by_class_markdown_table(summary),
        "threshold_search": search,
        "overall": {
            "f1": round(
                (2 * summary["overall"]["confusion"]["tp"])
                / max(
                    1,
                    2 * summary["overall"]["confusion"]["tp"]
                    + summary["overall"]["confusion"]["fp"]
                    + summary["overall"]["confusion"]["fn"],
                ),
                4,
            ),
            "far": summary["overall"]["far"],
            "frr": summary["overall"]["frr"],
            "coverage": summary["overall"]["coverage"],
        },
    }


def _gate_real_candidates(candidates_path: Path) -> list[dict[str, Any]]:
    """Load the real replay snapshot and apply the pre-LLM geometric gate per event."""
    rows = _read_jsonl(candidates_path)
    by_event: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        bbox = row.get("bbox_px")
        if not isinstance(bbox, list) or len(bbox) != 4:
            continue
        by_event[str(row.get("event_id"))].append(
            {
                "target_index": row.get("candidate_id"),
                "class_name": row.get("class_name"),
                "confidence": row.get("detector_confidence"),
                "bbox": bbox,
            }
        )
    gated: list[dict[str, Any]] = []
    for event_candidates in by_event.values():
        gated.extend(apply_pre_llm_geometric_gate(event_candidates))
    return gated


def evaluate_cascade_comparison(
    candidates_path: Path, *, thresholds: tuple[float, ...] = (BASELINE_TIER1_THRESHOLD, TIER1_AUTO_CONFIRM_THRESHOLD)
) -> dict[str, Any]:
    """对比改造前后 6713 条真实候选的 VLM 卸载比例与 Tier1 白名单护栏。"""
    gated = _gate_real_candidates(candidates_path)
    total = len(gated)
    runs: list[dict[str, Any]] = []
    for th in thresholds:
        counts: Counter[str] = Counter()
        per_class: dict[str, Counter[str]] = defaultdict(Counter)
        for cand in gated:
            routing = evaluate_cascade_tier(cand, tier1_threshold=th)
            counts[routing["tier"]] += 1
            per_class[str(cand.get("class_name") or "unknown")][routing["tier"]] += 1
        offloaded = counts["tier1_auto_confirm"] + counts["tier2_auto_reject"]
        runs.append(
            {
                "tier1_threshold": th,
                "tier1_count": counts["tier1_auto_confirm"],
                "tier2_count": counts["tier2_auto_reject"],
                "tier3_count": counts["tier3_vlm_verify"],
                "tier1_share": round(counts["tier1_auto_confirm"] / total, 4) if total else 0.0,
                "tier2_share": round(counts["tier2_auto_reject"] / total, 4) if total else 0.0,
                "vlm_share": round(counts["tier3_vlm_verify"] / total, 4) if total else 0.0,
                "vlm_offload_share": round(offloaded / total, 4) if total else 0.0,
                "geometric_gate_direct_intercept_share": 0.0,
                "per_class": {cls: dict(counter) for cls, counter in sorted(per_class.items())},
            }
        )

    # 护栏价值：若无 Tier1 白名单，多少非刚体类高分候选会被错误自动确认。
    guard = Counter()
    for cand in gated:
        cls_canon = _canonical_geom_class(cand.get("class_name"))
        confidence = cand.get("confidence")
        if cls_canon in TIER1_ELIGIBLE_CLASSES or bool(cand.get("geometric_warning")):
            continue
        if isinstance(confidence, (int, float)) and confidence >= TIER1_AUTO_CONFIRM_THRESHOLD:
            guard[cls_canon] += 1

    return {
        "source": str(candidates_path.relative_to(ROOT)),
        "candidates": total,
        "runs": runs,
        "tier1_allowlist": sorted(TIER1_ELIGIBLE_CLASSES),
        "non_allowlist_candidates_blocked_at_new_threshold": dict(sorted(guard.items())),
        "non_allowlist_blocked_total": sum(guard.values()),
    }


def evaluate_gold_cascade_replay(
    manifest_path: Path,
    overlap_path: Path,
    gold_path: Path,
    *,
    thresholds: tuple[float, ...] = (BASELINE_TIER1_THRESHOLD, TIER1_AUTO_CONFIRM_THRESHOLD),
) -> dict[str, Any]:
    """142 条真实人工金标上的端到端级联回放（Tier3 回退到真实 VLM 决策）。

    金标候选通过 `gold_overlap_map.jsonl` 关联到真实监控快照的 `detector_confidence`，
    因此可以用真实检测器分数验证「Tier1 自动确认的误报增量」。
    """
    manifest = {row["candidate_id"]: row for row in _read_jsonl(manifest_path)}
    overlap = _read_jsonl(overlap_path)
    vlm = {str(row.get("candidate_id")): row for row in _read_jsonl(gold_path)}

    gold_rows: list[dict[str, Any]] = []
    missing = 0
    for item in overlap:
        source = manifest.get(str(item.get("full_candidate_id")))
        if source is None:
            missing += 1
            continue
        gold_rows.append(
            {
                **source,
                "human_label": str(item.get("human_label")),
                "legacy_candidate_id": item.get("legacy_candidate_id"),
            }
        )

    by_event: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in gold_rows:
        by_event[str(row.get("event_id"))].append(row)

    runs: list[dict[str, Any]] = []
    for th in thresholds:
        decisions: list[dict[str, Any]] = []
        tier_counts: Counter[str] = Counter()
        tier1_fp = tier1_tp = 0
        for event_rows in by_event.values():
            gated = apply_pre_llm_geometric_gate(
                [
                    {
                        "target_index": row["candidate_id"],
                        "class_name": row.get("class_name"),
                        "confidence": row.get("detector_confidence"),
                        "bbox": row.get("bbox_px"),
                    }
                    for row in event_rows
                ]
            )
            for cand, source in zip(gated, event_rows):
                routing = evaluate_cascade_tier(cand, tier1_threshold=th)
                tier_counts[routing["tier"]] += 1
                if routing["tier"] == "tier1_auto_confirm":
                    if source["human_label"] == "false":
                        tier1_fp += 1
                    else:
                        tier1_tp += 1
                    prediction = "true"
                elif routing["tier"] == "tier2_auto_reject":
                    prediction = "false"
                else:
                    fallback = vlm.get(str(source.get("legacy_candidate_id")))
                    prediction = str((fallback or {}).get("hybrid_decision") or "uncertain")
                decisions.append(
                    {
                        "class_name": str(source.get("class_name")),
                        "gold": source["human_label"],
                        "prediction": prediction,
                        "parse_error": None,
                    }
                )
        summary = summarize_predictions(decisions)
        confusion = summary["overall"]["confusion"]
        runs.append(
            {
                "tier1_threshold": th,
                "n": len(decisions),
                "tier1_auto_confirm": tier_counts["tier1_auto_confirm"],
                "tier2_auto_reject": tier_counts["tier2_auto_reject"],
                "tier3_vlm_verify": tier_counts["tier3_vlm_verify"],
                "tier1_true_positive": tier1_tp,
                "tier1_false_positive": tier1_fp,
                "tier1_precision": round(tier1_tp / (tier1_tp + tier1_fp), 4)
                if (tier1_tp + tier1_fp)
                else None,
                "accuracy": round(summary["overall"]["accuracy"], 4),
                "selective_accuracy": round(summary["overall"]["selective_accuracy"], 4),
                "f1": round(
                    (2 * confusion["tp"]) / max(1, 2 * confusion["tp"] + confusion["fp"] + confusion["fn"]),
                    4,
                ),
                "far": summary["overall"]["far"],
                "frr": summary["overall"]["frr"],
                "manual_review_rate": round(summary["overall"]["manual_review_rate"], 4),
                "confusion": confusion,
            }
        )

    return {
        "gold_source": str(overlap_path.relative_to(ROOT)),
        "detector_score_source": str(manifest_path.relative_to(ROOT)),
        "vlm_decision_source": str(gold_path.relative_to(ROOT)),
        "gold_candidates": len(gold_rows),
        "unmatched": missing,
        "runs": runs,
    }


def evaluate_five_class_benchmark(seed: int = 20261023) -> dict[str, Any]:
    """五类合成基准（显式标注为 proxy）：验证逐类表与阈值搜索在五类上的可运行性。

    说明：仓库内不存在覆盖五类的人工金标预测文件，因此本表数值来自受控合成基准，
    仅用于验证指标管线与阈值搜索逻辑，不代表真实模型精度。
    """
    rng = random.Random(seed)
    # 每类的（真阳性样本数，假阳性样本数，正类分数均值，负类分数均值）
    profile = {
        "人员": (120, 45, 0.86, 0.28),
        "车辆": (150, 60, 0.88, 0.25),
        "三轮车": (90, 50, 0.78, 0.34),
        "蓝色违建顶棚": (70, 55, 0.74, 0.38),
        "道路裂缝": (60, 60, 0.69, 0.42),
    }
    records: list[dict[str, Any]] = []
    for cls, (n_pos, n_neg, mu_pos, mu_neg) in profile.items():
        for _ in range(n_pos):
            score = min(0.999, max(0.001, rng.gauss(mu_pos, 0.13)))
            records.append(
                {"class_name": cls, "gold": "true", "score": round(score, 3),
                 "prediction": "true" if score >= 0.5 else "false"}
            )
        for _ in range(n_neg):
            score = min(0.999, max(0.001, rng.gauss(mu_neg, 0.13)))
            records.append(
                {"class_name": cls, "gold": "false", "score": round(score, 3),
                 "prediction": "true" if score >= 0.5 else "false"}
            )
    summary = summarize_predictions(records)
    search = find_optimal_threshold_for_far(records, max_far=0.10)
    return {
        "source": "constructed_benchmark",
        "note": "合成基准 proxy：仅验证指标管线与阈值搜索，不代表真实模型精度",
        "seed": seed,
        "n": len(records),
        "by_class_table": summary["by_class_table"],
        "markdown": format_by_class_markdown_table(summary),
        "threshold_search": search,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidates", type=Path, default=DEFAULT_CANDIDATES)
    parser.add_argument("--gold", type=Path, default=DEFAULT_GOLD)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--gold-overlap", type=Path, default=DEFAULT_GOLD_OVERLAP)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    report = {
        "generated_at": "deterministic",
        "geometry_gate": evaluate_geometry_sweep(),
        "real_replay": evaluate_real_replay(args.candidates),
        "cascade_before_after": evaluate_cascade_comparison(args.candidates),
        "gold_cascade_replay": evaluate_gold_cascade_replay(
            args.manifest,
            args.gold_overlap,
            args.gold,
            thresholds=(0.75, 0.80, 0.82, 0.85, 0.92),
        ),
        "real_two_class_table": evaluate_real_two_class(args.gold),
        "five_class_benchmark": evaluate_five_class_benchmark(),
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "five_class_gate_eval.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    geo = report["geometry_gate"]
    rep = report["real_replay"]
    cmp_runs = {run["tier1_threshold"]: run for run in report["cascade_before_after"]["runs"]}
    gold_runs = {run["tier1_threshold"]: run for run in report["gold_cascade_replay"]["runs"]}
    base = cmp_runs[BASELINE_TIER1_THRESHOLD]
    new = cmp_runs[TIER1_AUTO_CONFIRM_THRESHOLD]
    gold_base = gold_runs[BASELINE_TIER1_THRESHOLD]
    gold_new = gold_runs[TIER1_AUTO_CONFIRM_THRESHOLD]
    md = [
        "# 五类业务门禁与三级级联路由离线回放评估",
        "",
        "## 1. 几何门禁（受控扫描，精确控制 IoU / containment）",
        "",
        "| 指标 | 数值 |",
        "|---|---:|",
        f"| 同目标双框抑制率（IoU > 0.75） | {geo['same_target_double_box_suppression_rate']:.2%} |",
        f"| 同目标双框误抑制率（IoU <= 0.75） | {geo['same_target_double_box_false_flag_rate']:.2%} |",
        f"| 三轮车车斗误报拦截率（containment > 0.88） | {geo['tricycle_bucket_interception_rate']:.2%} |",
        f"| 三轮车车斗误拦截率（containment <= 0.88） | {geo['tricycle_bucket_false_flag_rate']:.2%} |",
        f"| containment 规则隔离检出率（person-in-vehicle） | {geo['containment_only_interception_rate']:.2%} |",
        f"| containment 规则隔离误报率 | {geo['containment_only_false_flag_rate']:.2%} |",
        "",
        "## 2. 三级级联路由（真实回放）",
        "",
        f"数据源：`{rep['source']}`，{rep['events']} 个事件 / {rep['candidates']} 个候选。",
        "",
        "| Tier | 数量 | 占比 |",
        "|---|---:|---:|",
    ]
    for tier in ("tier1_auto_confirm", "tier2_auto_reject", "tier3_vlm_verify"):
        md.append(
            f"| {tier} | {rep['tier_counts'].get(tier, 0)} | "
            f"{rep['tier_ratios'].get(tier, 0.0):.2%} |"
        )
    md += [
        "",
        f"- 几何冲突告警候选：{rep['geometric_warning_candidates']}"
        f"（{rep['geometric_warning_ratio']:.2%}），其中车/三轮车 IoU 冲突 "
        f"{rep['vehicle_tricycle_iou_conflicts']}，包含关系告警 {rep['containment_warnings']}。",
        f"- blue_illegal_roof 强制 Tier3 合规率："
        f"{(rep['blue_illegal_roof_tier3_compliance'] or 0):.2%}"
        f"（{rep['blue_illegal_roof_tier3']}/{rep['blue_illegal_roof_total']}）。",
        "",
        "## 3. 级联阈值改造前后对比（真实数据）",
        "",
        f"数据源：`{report['cascade_before_after']['source']}`，"
        f"{report['cascade_before_after']['candidates']} 条真实候选；"
        f"金标回放 {report['gold_cascade_replay']['gold_candidates']} 条（Tier3 回退真实 VLM 决策）。",
        "",
        "| 指标 | 改造前（Tier1=0.92） | 改造后（Tier1=0.85，刚体类白名单） |",
        "|---|---:|---:|",
        f"| Tier1 自动确认 | {base['tier1_count']}（{base['tier1_share']:.2%}） | {new['tier1_count']}（{new['tier1_share']:.2%}） |",
        f"| Tier2 自动拒识 | {base['tier2_count']}（{base['tier2_share']:.2%}） | {new['tier2_count']}（{new['tier2_share']:.2%}） |",
        f"| L20 VLM 调用 | {base['tier3_count']}（{base['vlm_share']:.2%}） | {new['tier3_count']}（{new['vlm_share']:.2%}） |",
        f"| VLM 卸载比例（Tier1+Tier2） | {base['vlm_offload_share']:.2%} | {new['vlm_offload_share']:.2%} |",
        f"| 金标 Tier1 自动确认数 | {gold_base['tier1_auto_confirm']} | {gold_new['tier1_auto_confirm']} |",
        f"| 金标 Tier1 新增误报 FP | {gold_base['tier1_false_positive']} | {gold_new['tier1_false_positive']} |",
        f"| 金标端到端 F1 | {gold_base['f1']:.4f} | {gold_new['f1']:.4f} |",
        f"| 金标端到端 FAR | {gold_base['far']} | {gold_new['far']} |",
        f"| 金标端到端准确率 | {gold_base['accuracy']:.4f} | {gold_new['accuracy']:.4f} |",
        "",
        f"- Tier1 白名单：{', '.join(report['cascade_before_after']['tier1_allowlist'])}；"
        f"在新阈值下被白名单拦下的非刚体类高分候选 "
        f"{report['cascade_before_after']['non_allowlist_blocked_total']} 条"
        f"（{report['cascade_before_after']['non_allowlist_candidates_blocked_at_new_threshold']}）。",
        "- 几何门禁直接拦截占比：0%（当前门禁只把冲突候选升级到 Tier3，不做直接裁决）。",
        "",
        "### 3.1 Tier1 阈值 ↔ 金标误报权衡（142 条真实人工金标）",
        "",
        "| Tier1 阈值 | 金标自动确认数 | 新增 FP | 端到端 F1 | 端到端 FAR |",
        "|---:|---:|---:|---:|---:|",
        *[
            f"| {run['tier1_threshold']:.2f} | {run['tier1_auto_confirm']} | "
            f"{run['tier1_false_positive']} | {run['f1']:.4f} | "
            f"{run['far'] if run['far'] is None else round(run['far'], 4)} |"
            for run in report["gold_cascade_replay"]["runs"]
        ],
        "",
        "## 4. 逐类表与阈值搜索",
        "",
        "### 4.1 真实人工金标（车/人员）",
        "",
        f"数据源：`{report['real_two_class_table']['source']}`，n={report['real_two_class_table']['n']}。",
        "",
        report["real_two_class_table"]["markdown"],
        "",
        f"FAR 约束搜索（max_far=0.10）推荐阈值："
        f"**{report['real_two_class_table']['threshold_search']['recommended_threshold']}**，"
        f"best F1={report['real_two_class_table']['threshold_search']['best_metrics']['f1']:.4f}，"
        f"FAR={report['real_two_class_table']['threshold_search']['best_metrics']['far']}。",
        "",
        "### 4.2 五类合成基准（proxy，非真实精度）",
        "",
        report["five_class_benchmark"]["markdown"],
        "",
        f"FAR 约束搜索（max_far=0.10）推荐阈值："
        f"**{report['five_class_benchmark']['threshold_search']['recommended_threshold']}**，"
        f"best F1={report['five_class_benchmark']['threshold_search']['best_metrics']['f1']:.4f}，"
        f"FAR={report['five_class_benchmark']['threshold_search']['best_metrics']['far']}。",
    ]
    (args.output_dir / "five_class_gate_eval.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print("\n".join(md))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
