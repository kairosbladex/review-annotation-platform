#!/usr/bin/env python3
"""五类业务门禁与三级级联路由的离线回放评估（可复现）。

评估四组指标：
  1. 车/三轮车几何门禁：IoU > 0.75 同目标双框抑制率、containment > 0.88 三轮车车斗误报拦截率；
  2. 三级级联路由：Tier1/Tier2/Tier3 分流比、blue_illegal_roof 100% 强制进入 Tier3 的合规率；
  3. 级联阈值改造前后对比：6713 条真实候选的 VLM 卸载比例、142 条真实金标上的新增 FP 与端到端 F1/FAR；
  4. 阈值搜索与逐类表：build_by_class_table / find_optimal_threshold_for_far(max_far=0.10)。
  5. Tier 2.5（Jev-Omni Fast Gate）：二级卸载率（含假框拒识）、FAR/FRR、平均单框耗时下降比例。

数据来源：
  - 真实回放集（默认）：data/ktd-test-real-monitoring-full-20260924/output/raw_predictions.jsonl
    （6713 条候选、1416 个事件、五类目标、含 detector_confidence / bbox_px / VLM decision / elapsed_seconds）
  - 真实人工金标（默认）：data/ktd-clean-source-full-evaluation-20260924/predictions_scored.jsonl
    （142 条，覆盖 车/人员 两类；五类人工金标在仓库内不存在，故五类逐类表使用显式标注的合成基准）

用法：
    PYTHONPATH=.:backend .venv/bin/python scripts/eval_five_class_gates.py \
        --output-dir /tmp/ktd_phase3_eval

    # 真机部署 Jev-Omni 后，用真实概率重跑同一评估（不依赖代理校准）
    PYTHONPATH=.:backend .venv/bin/python scripts/eval_five_class_gates.py \
        --output-dir /tmp/ktd_jev_eval --jev-omni-probs /tmp/jev_omni_probs.jsonl
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))
sys.path.insert(0, str(ROOT / "scripts"))

from app.services.ktd_recheck import (  # noqa: E402
    JEV_OMNI_CONFIRM_THRESHOLD,
    JEV_OMNI_REJECT_THRESHOLD,
    TIER1_AUTO_CONFIRM_THRESHOLD,
    TIER1_ELIGIBLE_CLASSES,
    TIER2_AUTO_REJECT_THRESHOLD,
    _canonical_geom_class,
    apply_pre_llm_geometric_gate,
    compute_bbox_iou,
    compute_containment_ratio,
    evaluate_cascade_tier,
    evaluate_jev_omni_tier,
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


# --------------------------------------------------------------------------
# 5. Tier 2.5（Jev-Omni Fast Gate）离线收益评估
# --------------------------------------------------------------------------
JEV_OMNI_PROXY_NOTE = (
    "本机无 NVIDIA GPU、未下载 12B 权重：Tier 2.5 的 P(true) 为**代理口径**，不是 Jev-Omni "
    "真实输出。主口径用同一候选的 8B 多模态决策置信度折算（与 Tier 3 伪标签同源，精确率是上界）；"
    "下界用 142 条人工金标拟合的 detector_confidence -> P(true) Platt 校准（与 Tier 3 独立）。"
    "真机部署后用 --jev-omni-probs 指向真实概率 JSONL 重跑同一评估，结果应落在这两个界之间。"
)


def _sigmoid(value: float) -> float:
    if value >= 0:
        return 1.0 / (1.0 + math.exp(-value))
    exp_value = math.exp(value)
    return exp_value / (1.0 + exp_value)


def fit_platt_calibration(pairs: Iterable[tuple[float, int]]) -> dict[str, float]:
    """在真实金标上拟合 ``P(true) = sigmoid(a * detector_confidence + b)``。

    用牛顿法（IRLS）解二维 logistic 回归，确定性、无第三方依赖。
    """
    data = [(float(score), int(label)) for score, label in pairs]
    if len(data) < 2 or len({label for _, label in data}) < 2:
        raise ValueError("Platt 校准至少需要两个类别各一个样本")
    a, b = 2.0, -1.0
    for _ in range(100):
        grad_a = grad_b = 0.0
        hess_aa = hess_ab = hess_bb = 0.0
        for score, label in data:
            probability = _sigmoid(a * score + b)
            residual = probability - label
            grad_a += residual * score
            grad_b += residual
            weight = max(probability * (1.0 - probability), 1e-9)
            hess_aa += weight * score * score
            hess_ab += weight * score
            hess_bb += weight
        determinant = hess_aa * hess_bb - hess_ab * hess_ab
        if abs(determinant) < 1e-12:
            break
        step_a = (hess_bb * grad_a - hess_ab * grad_b) / determinant
        step_b = (-hess_ab * grad_a + hess_aa * grad_b) / determinant
        a -= step_a
        b -= step_b
        if abs(step_a) < 1e-9 and abs(step_b) < 1e-9:
            break
    return {"a": a, "b": b, "n": len(data)}


def _rank_auc(pairs: list[tuple[float, int]]) -> float | None:
    positives = [score for score, label in pairs if label == 1]
    negatives = [score for score, label in pairs if label == 0]
    if not positives or not negatives:
        return None
    wins = 0.0
    for positive in positives:
        for negative in negatives:
            if positive > negative:
                wins += 1.0
            elif positive == negative:
                wins += 0.5
    return round(wins / (len(positives) * len(negatives)), 4)


def _load_prob_overrides(path: Path | None) -> dict[str, float]:
    if path is None:
        return {}
    overrides: dict[str, float] = {}
    for row in _read_jsonl(path):
        candidate_id = row.get("candidate_id") or row.get("target_index")
        value = row.get("p_true", row.get("probability", row.get("score")))
        if candidate_id is None or not isinstance(value, (int, float)):
            continue
        probability = float(value)
        if 0.0 <= probability <= 1.0:
            overrides[str(candidate_id)] = probability
    return overrides


def _gold_rows_with_detector_scores(
    manifest_path: Path, overlap_path: Path
) -> tuple[list[dict[str, Any]], int]:
    """142 条人工金标 join 到真实监控快照的检测器分数（与 gold_cascade_replay 同口径）。"""
    manifest = {row["candidate_id"]: row for row in _read_jsonl(manifest_path)}
    rows: list[dict[str, Any]] = []
    missing = 0
    for item in _read_jsonl(overlap_path):
        source = manifest.get(str(item.get("full_candidate_id")))
        if source is None:
            missing += 1
            continue
        rows.append(
            {
                **source,
                "human_label": str(item.get("human_label")),
                "legacy_candidate_id": item.get("legacy_candidate_id"),
            }
        )
    return rows, missing


def _multimodal_probability(row: dict[str, Any]) -> float:
    """把同一候选的多模态复核决策+置信度折算成 ``P(true)`` 代理。

    说明：``qwen3-vl:8b`` 的 ``decision``/``confidence`` 与 Jev-Omni 属于同一类
    「看图判真假」的多模态验证器，比检测器分数更接近 Jev-Omni 的校准概率形态；
    但它与 Tier 3 的伪标签同源，因此**拒识精确率是上界**，必须配合
    ``detector_platt`` 下界一起解读。
    """
    decision = str(row.get("decision", row.get("hybrid_decision")) or "").strip().lower()
    confidence = row.get("confidence", row.get("hybrid_confidence"))
    if not isinstance(confidence, (int, float)) or isinstance(confidence, bool):
        return 0.5
    value = min(1.0, max(0.0, float(confidence)))
    if decision in {"true", "confirmed", "keep"}:
        return value
    if decision in {"false", "rejected", "reject"}:
        return 1.0 - value
    return 0.5


def evaluate_jev_omni_tier25(
    candidates_path: Path,
    manifest_path: Path,
    overlap_path: Path,
    gold_path: Path,
    *,
    confirm_threshold: float = JEV_OMNI_CONFIRM_THRESHOLD,
    reject_threshold: float = JEV_OMNI_REJECT_THRESHOLD,
    jev_omni_latency_ms: float = 900.0,
    probs_path: Path | None = None,
    prob_source: str = "multimodal_proxy",
) -> dict[str, Any]:
    """评估接入 Tier 2.5 后的二级卸载率、FAR/FRR 与平均单框耗时变化。

    概率来源三档：

    * ``real``（``--jev-omni-probs``）：真机 Jev-Omni 输出，唯一非代理口径；
    * ``multimodal_proxy``（默认）：用同一候选的 8B 多模态决策置信度折算，
      形态最接近 12B 校准分类器，但拒识精确率是**上界**；
    * ``detector_platt``：142 条金标上拟合的 ``detector_confidence -> P(true)``，
      与 Tier 3 伪标签完全独立，是**保守下界**（当前数据下 Tier 2.5 不触发）。

    主口径用 ``multimodal_proxy``，同时输出 ``detector_platt`` 下界，避免用单一
    代理高估收益。Tier 3 耗时取真实候选的实测 ``elapsed_seconds``；Tier 2.5 耗时
    用 ``jev_omni_latency_ms`` 参数化（真机部署后可用实测值替换）。
    """
    if prob_source not in {"multimodal_proxy", "detector_platt", "real"}:
        raise ValueError("prob_source 必须是 multimodal_proxy / detector_platt / real")
    if prob_source == "real" and probs_path is None:
        raise ValueError("prob_source=real 需要提供 --jev-omni-probs")

    overrides = _load_prob_overrides(probs_path)
    gold_rows, gold_missing = _gold_rows_with_detector_scores(manifest_path, overlap_path)
    gold_vlm = {str(row.get("candidate_id")): row for row in _read_jsonl(gold_path)}
    for row in gold_rows:
        legacy = gold_vlm.get(str(row.get("legacy_candidate_id"))) or {}
        row["hybrid_decision"] = legacy.get("hybrid_decision")
        row["hybrid_confidence"] = legacy.get("hybrid_confidence")
        row["decision"] = legacy.get("hybrid_decision")
        row["confidence"] = legacy.get("hybrid_confidence")

    calibration_pairs = [
        (float(row["detector_confidence"]), 1 if row["human_label"] == "true" else 0)
        for row in gold_rows
        if isinstance(row.get("detector_confidence"), (int, float))
    ]
    calibration = fit_platt_calibration(calibration_pairs)

    def detector_probability(row: dict[str, Any]) -> float:
        confidence = row.get("detector_confidence")
        if not isinstance(confidence, (int, float)) or isinstance(confidence, bool):
            return 0.5
        return _sigmoid(calibration["a"] * float(confidence) + calibration["b"])

    def make_probability(source: str):
        def probability_of(row: dict[str, Any]) -> float:
            key = str(row.get("candidate_id"))
            if key in overrides:
                return overrides[key]
            if source == "detector_platt":
                return detector_probability(row)
            return _multimodal_probability(row)

        return probability_of

    rows = _read_jsonl(candidates_path)
    by_event: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        bbox = row.get("bbox_px")
        if isinstance(bbox, list) and len(bbox) == 4:
            by_event[str(row.get("event_id"))].append(row)

    vlm_false_ids = {
        str(row.get("candidate_id")) for row in rows if str(row.get("decision")) == "false"
    }
    vlm_true_ids = {
        str(row.get("candidate_id")) for row in rows if str(row.get("decision")) == "true"
    }

    def replay_candidates(probability_of) -> dict[str, Any]:
        counts: Counter[str] = Counter()
        base_counts: Counter[str] = Counter()
        per_class: dict[str, Counter[str]] = defaultdict(Counter)
        vlm_seconds_before = vlm_seconds_after = tier25_seconds = 0.0
        rejected_rows: list[dict[str, Any]] = []
        confirmed_rows: list[dict[str, Any]] = []
        tier3_before_ids: set[str] = set()
        total = 0

        for event_rows in by_event.values():
            gated = apply_pre_llm_geometric_gate(
                [
                    {
                        "target_index": row.get("candidate_id"),
                        "class_name": row.get("class_name"),
                        "confidence": row.get("detector_confidence"),
                        "bbox": row.get("bbox_px"),
                    }
                    for row in event_rows
                ]
            )
            for candidate, row in zip(gated, event_rows):
                total += 1
                cls = str(row.get("class_name") or "unknown")
                elapsed = float(row.get("elapsed_seconds") or 0.0)
                routing = evaluate_cascade_tier(candidate)
                tier = routing["tier"]
                base_counts[tier] += 1
                if routing["requires_vlm"]:
                    tier3_before_ids.add(str(row.get("candidate_id")))
                    vlm_seconds_before += elapsed
                    tier25 = evaluate_jev_omni_tier(
                        candidate,
                        p_true=probability_of(row),
                        confirm_threshold=confirm_threshold,
                        reject_threshold=reject_threshold,
                    )
                    tier = tier25["tier"]
                    if tier25["requires_vlm"]:
                        vlm_seconds_after += elapsed
                    else:
                        tier25_seconds += jev_omni_latency_ms / 1000.0
                        if tier25["auto_label"] == "false":
                            rejected_rows.append(row)
                        else:
                            confirmed_rows.append(row)
                counts[tier] += 1
                per_class[cls][tier] += 1

        def share(value: int) -> float:
            return round(value / total, 4) if total else 0.0

        tier1 = counts["tier1_auto_confirm"]
        tier2 = counts["tier2_auto_reject"]
        tier25_confirm = counts["tier2_5_jev_omni_confirm"]
        tier25_reject = counts["tier2_5_jev_omni_reject"]
        tier3 = counts["tier3_vlm_verify"]
        tier3_before = base_counts["tier3_vlm_verify"]
        rejected_false = sum(1 for row in rejected_rows if str(row.get("candidate_id")) in vlm_false_ids)
        rejected_true = sum(1 for row in rejected_rows if str(row.get("candidate_id")) in vlm_true_ids)
        false_box_total = len(vlm_false_ids & tier3_before_ids)
        before_seconds = vlm_seconds_before
        after_seconds = vlm_seconds_after + tier25_seconds

        return {
            "candidates": total,
            "events": len(by_event),
            "before": {
                "tier1_auto_confirm": tier1,
                "tier2_auto_reject": tier2,
                "tier3_vlm_verify": tier3_before,
                "vlm_share": share(tier3_before),
                "vlm_offload_share": share(tier1 + tier2),
            },
            "after": {
                "tier1_auto_confirm": tier1,
                "tier2_auto_reject": tier2,
                "tier2_5_jev_omni_confirm": tier25_confirm,
                "tier2_5_jev_omni_reject": tier25_reject,
                "tier3_vlm_verify": tier3,
                "vlm_share": share(tier3),
                "vlm_offload_share": share(total - tier3),
                "second_stage_offload_share": share(tier25_confirm + tier25_reject),
                "second_stage_offload_within_previous_tier3": round(
                    (tier25_confirm + tier25_reject) / tier3_before, 4
                )
                if tier3_before
                else 0.0,
            },
            "false_box_reject": {
                "rejected_candidates": tier25_reject,
                "vlm_false_total_among_previous_tier3": false_box_total,
                "rejected_and_vlm_false": rejected_false,
                "rejected_but_vlm_true": rejected_true,
                "false_box_reject_rate": round(rejected_false / false_box_total, 4)
                if false_box_total
                else 0.0,
                "reject_precision_vs_vlm": round(rejected_false / len(rejected_rows), 4)
                if rejected_rows
                else 0.0,
            },
            "confirmed": {
                "confirmed_candidates": tier25_confirm,
                "confirmed_and_vlm_true": sum(
                    1 for row in confirmed_rows if str(row.get("candidate_id")) in vlm_true_ids
                ),
                "confirmed_but_vlm_false": sum(
                    1 for row in confirmed_rows if str(row.get("candidate_id")) in vlm_false_ids
                ),
            },
            "per_class": {
                cls: dict(counter)
                for cls, counter in sorted(per_class.items(), key=lambda item: -sum(item[1].values()))
            },
            "latency": {
                "vlm_measured_mean_ms": round(vlm_seconds_before / tier3_before * 1000.0, 1)
                if tier3_before
                else 0.0,
                "jev_omni_assumed_mean_ms": jev_omni_latency_ms,
                "before_total_seconds": round(before_seconds, 1),
                "after_total_seconds": round(after_seconds, 1),
                "before_mean_ms_per_box": round(before_seconds / total * 1000.0, 1) if total else 0.0,
                "after_mean_ms_per_box": round(after_seconds / total * 1000.0, 1) if total else 0.0,
                "mean_latency_drop_ratio": round(1.0 - after_seconds / before_seconds, 4)
                if before_seconds
                else 0.0,
            },
        }

    primary = replay_candidates(make_probability(prob_source))
    detector_bound = (
        replay_candidates(make_probability("detector_platt"))
        if prob_source != "detector_platt"
        else None
    )

    # ---------------- 142 条人工金标 ----------------
    gold_probability = make_probability("detector_platt" if prob_source == "detector_platt" else "multimodal_proxy")

    def replay_gold(*, use_tier25: bool) -> dict[str, Any]:
        decisions: list[dict[str, Any]] = []
        tier_counts: Counter[str] = Counter()
        reject_true = reject_false = confirm_true = confirm_false = 0
        for event_rows in _group_by_event(gold_rows).values():
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
            for candidate, source in zip(gated, event_rows):
                routing = evaluate_cascade_tier(candidate)
                if routing["requires_vlm"] and use_tier25:
                    tier25 = evaluate_jev_omni_tier(
                        candidate,
                        p_true=gold_probability(source),
                        confirm_threshold=confirm_threshold,
                        reject_threshold=reject_threshold,
                    )
                    tier_counts[tier25["tier"]] += 1
                    if tier25["tier"] == "tier2_5_jev_omni_reject":
                        if source["human_label"] == "false":
                            reject_false += 1
                        else:
                            reject_true += 1
                        prediction = "false"
                    elif tier25["tier"] == "tier2_5_jev_omni_confirm":
                        if source["human_label"] == "true":
                            confirm_true += 1
                        else:
                            confirm_false += 1
                        prediction = "true"
                    else:
                        prediction = _gold_vlm_prediction(gold_path, source)
                elif routing["tier"] == "tier1_auto_confirm":
                    tier_counts[routing["tier"]] += 1
                    prediction = "true"
                elif routing["tier"] == "tier2_auto_reject":
                    tier_counts[routing["tier"]] += 1
                    prediction = "false"
                else:
                    tier_counts[routing["tier"]] += 1
                    prediction = _gold_vlm_prediction(gold_path, source)
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
        return {
            "n": len(decisions),
            "tier_counts": {key: value for key, value in tier_counts.items() if value},
            "accuracy": round(summary["overall"]["accuracy"], 4),
            "f1": round(
                (2 * confusion["tp"]) / max(1, 2 * confusion["tp"] + confusion["fp"] + confusion["fn"]),
                4,
            ),
            "far": summary["overall"]["far"],
            "frr": summary["overall"]["frr"],
            "manual_review_rate": round(summary["overall"]["manual_review_rate"], 4),
            "confusion": confusion,
            "tier2_5_reject_true_negative": reject_false,
            "tier2_5_reject_false_negative": reject_true,
            "tier2_5_confirm_true_positive": confirm_true,
            "tier2_5_confirm_false_positive": confirm_false,
        }

    gold_before = replay_gold(use_tier25=False)
    gold_after = replay_gold(use_tier25=True)

    return {
        "note": JEV_OMNI_PROXY_NOTE,
        "probability_source": (
            f"real:{probs_path.relative_to(ROOT)}" if probs_path is not None else prob_source
        ),
        "calibration": {
            **calibration,
            "auc_detector_score": _rank_auc(calibration_pairs),
            "positives": sum(1 for _, label in calibration_pairs if label == 1),
            "negatives": sum(1 for _, label in calibration_pairs if label == 0),
        },
        "thresholds": {
            "tier1_auto_confirm": TIER1_AUTO_CONFIRM_THRESHOLD,
            "tier2_auto_reject": TIER2_AUTO_REJECT_THRESHOLD,
            "jev_omni_confirm": confirm_threshold,
            "jev_omni_reject": reject_threshold,
            "jev_omni_latency_ms": jev_omni_latency_ms,
        },
        "replay": {
            "source": str(candidates_path.relative_to(ROOT)),
            **primary,
        },
        "replay_detector_bound": detector_bound,
        "gold": {
            "gold_source": str(overlap_path.relative_to(ROOT)),
            "detector_score_source": str(manifest_path.relative_to(ROOT)),
            "candidates": len(gold_rows),
            "unmatched": gold_missing,
            "before": gold_before,
            "after": gold_after,
            "delta": {
                "f1": round(gold_after["f1"] - gold_before["f1"], 4),
                "far": _delta(gold_after["far"], gold_before["far"]),
                "frr": _delta(gold_after["frr"], gold_before["frr"]),
                "manual_review_rate": round(
                    gold_after["manual_review_rate"] - gold_before["manual_review_rate"], 4
                ),
            },
        },
    }

def _delta(after: Any, before: Any) -> float | None:
    if isinstance(after, (int, float)) and isinstance(before, (int, float)):
        return round(float(after) - float(before), 4)
    return None


def _group_by_event(rows: Iterable[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row.get("event_id"))].append(row)
    return grouped


def _gold_vlm_prediction(gold_path: Path, source: dict[str, Any]) -> str:
    """Tier 3 回退：用真实 VLM 决策（与 gold_cascade_replay 同口径）。"""
    cache = globals().get("_GOLD_VLM_CACHE")
    if cache is None or cache[0] != str(gold_path):
        mapping = {
            str(row.get("candidate_id")): str(row.get("hybrid_decision") or "uncertain")
            for row in _read_jsonl(gold_path)
        }
        cache = (str(gold_path), mapping)
        globals()["_GOLD_VLM_CACHE"] = cache
    return cache[1].get(str(source.get("legacy_candidate_id")), "uncertain")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidates", type=Path, default=DEFAULT_CANDIDATES)
    parser.add_argument("--gold", type=Path, default=DEFAULT_GOLD)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--gold-overlap", type=Path, default=DEFAULT_GOLD_OVERLAP)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--jev-omni-probs",
        type=Path,
        default=None,
        help="真实 Jev-Omni 概率 JSONL（candidate_id + p_true）；缺省用 142 条金标的 Platt 校准代理",
    )
    parser.add_argument("--jev-omni-confirm-threshold", type=float, default=JEV_OMNI_CONFIRM_THRESHOLD)
    parser.add_argument("--jev-omni-reject-threshold", type=float, default=JEV_OMNI_REJECT_THRESHOLD)
    parser.add_argument(
        "--jev-omni-prob-source",
        choices=("multimodal_proxy", "detector_platt", "real"),
        default="multimodal_proxy",
        help="概率来源：multimodal_proxy（默认，上界）/ detector_platt（下界）/ real（真机概率）",
    )
    parser.add_argument(
        "--jev-omni-latency-ms",
        type=float,
        default=900.0,
        help="Jev-Omni 单框耗时假设（毫秒）。L20 BF16 实测前先用 900ms 保守估计。",
    )
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
        "jev_omni_tier25": evaluate_jev_omni_tier25(
            args.candidates,
            args.manifest,
            args.gold_overlap,
            args.gold,
            confirm_threshold=args.jev_omni_confirm_threshold,
            reject_threshold=args.jev_omni_reject_threshold,
            jev_omni_latency_ms=args.jev_omni_latency_ms,
            probs_path=args.jev_omni_probs,
            prob_source=args.jev_omni_prob_source,
        ),
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
    jev = report["jev_omni_tier25"]
    rj = jev["replay"]
    rq = rj["false_box_reject"]
    gj = jev["gold"]

    def _fmt(value: Any, *, signed: bool = False) -> str:
        if value is None:
            return "—"
        if isinstance(value, (int, float)):
            return f"{value:+.4f}" if signed else f"{value:.4f}"
        return str(value)
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
        "",
        "## 5. Tier 2.5（Jev-Omni Fast Gate）接入前后对比",
        "",
        f"> {jev['note']}",
        "",
        f"概率来源：`{jev['probability_source']}`；校准参数 a={jev['calibration']['a']:.4f}、"
        f"b={jev['calibration']['b']:.4f}（142 条金标：{jev['calibration']['positives']} 正 / "
        f"{jev['calibration']['negatives']} 负，检测器分数 AUC={jev['calibration']['auc_detector_score']}）。",
        "",
        f"阈值：Tier1={jev['thresholds']['tier1_auto_confirm']}、Tier2={jev['thresholds']['tier2_auto_reject']}、"
        f"Jev-Omni 确认={jev['thresholds']['jev_omni_confirm']}、Jev-Omni 拒识={jev['thresholds']['jev_omni_reject']}。",
        "",
        "### 5.1 6713 条真实候选的分流与卸载",
        "",
        "| 指标 | 接入前（Tier1+Tier2） | 接入 Tier 2.5 后 |",
        "|---|---:|---:|",
        f"| Tier1 自动确认 | {rj['before']['tier1_auto_confirm']} | {rj['after']['tier1_auto_confirm']} |",
        f"| Tier2 自动拒识 | {rj['before']['tier2_auto_reject']} | {rj['after']['tier2_auto_reject']} |",
        f"| Tier2.5 自动确认 | — | {rj['after']['tier2_5_jev_omni_confirm']} |",
        f"| Tier2.5 自动拒识（假框） | — | {rj['after']['tier2_5_jev_omni_reject']} |",
        f"| L20 Qwen3-VL 调用 | {rj['before']['tier3_vlm_verify']}（{rj['before']['vlm_share']:.2%}） | "
        f"{rj['after']['tier3_vlm_verify']}（{rj['after']['vlm_share']:.2%}） |",
        f"| VLM 卸载比例 | {rj['before']['vlm_offload_share']:.2%} | {rj['after']['vlm_offload_share']:.2%} |",
        f"| 二级卸载率（Tier 2.5 占全量） | — | {rj['after']['second_stage_offload_share']:.2%} |",
        f"| 二级卸载率（占原 Tier3） | — | {rj['after']['second_stage_offload_within_previous_tier3']:.2%} |",
        "",
        "### 5.1.1 代理口径敏感性（收益区间）",
        "",
        "| 概率口径 | Tier2.5 确认 | Tier2.5 拒识 | VLM 卸载比例 | 平均单框耗时下降 |",
        "|---|---:|---:|---:|---:|",
        f"| `{jev['probability_source']}`（主口径） | {rj['after']['tier2_5_jev_omni_confirm']} | "
        f"{rj['after']['tier2_5_jev_omni_reject']} | {rj['after']['vlm_offload_share']:.2%} | "
        f"{rj['latency']['mean_latency_drop_ratio']:.2%} |",
        *(
            [
                f"| `detector_platt`（独立下界） | {db['after']['tier2_5_jev_omni_confirm']} | "
                f"{db['after']['tier2_5_jev_omni_reject']} | {db['after']['vlm_offload_share']:.2%} | "
                f"{db['latency']['mean_latency_drop_ratio']:.2%} |"
            ]
            if (db := report["jev_omni_tier25"]["replay_detector_bound"])
            else []
        ),
        "",
        "- 主口径用 8B 多模态决策置信度折算，形态接近 12B 校准分类器，但**与 Tier 3 伪标签同源**，"
        "因此确认/拒识的精确率是**上界**。",
        "- `detector_platt` 只用检测器分数（与 Tier 3 独立），是**下界**：当前数据下检测器分数"
        "经 Platt 校准后落在 [0.16, 0.93]，Tier 2.5 在 0.12/0.88 双阈值下不触发。",
        "- 真实收益必须用真机 Jev-Omni 概率（`--jev-omni-probs`）重跑后落在这两个界之间。",
        "",
        "### 5.2 假框拒识质量（以真实 VLM decision 为伪标签）",
        "",
        "| 指标 | 数值 |",
        "|---|---:|",
        f"| 原 Tier3 中 VLM 判为 false 的候选 | {rq['vlm_false_total_among_previous_tier3']} |",
        f"| 被 Tier 2.5 拒识且 VLM 亦判 false | {rq['rejected_and_vlm_false']} |",
        f"| 被 Tier 2.5 拒识但 VLM 判 true（误拒风险） | {rq['rejected_but_vlm_true']} |",
        f"| 假框拒识率（拒识覆盖的 VLM-false 占比） | {rq['false_box_reject_rate']:.2%} |",
        f"| 拒识精确率（拒识中 VLM 亦判 false 占比） | {rq['reject_precision_vs_vlm']:.2%} |",
        "",
        "### 5.3 平均单框耗时",
        "",
        "| 指标 | 数值 |",
        "|---|---:|",
        f"| L20 Qwen3-VL 实测单框均值 | {rj['latency']['vlm_measured_mean_ms']:.1f} ms |",
        f"| Jev-Omni 单框耗时（假设值） | {rj['latency']['jev_omni_assumed_mean_ms']:.1f} ms |",
        f"| 接入前平均单框耗时 | {rj['latency']['before_mean_ms_per_box']:.1f} ms |",
        f"| 接入后平均单框耗时 | {rj['latency']['after_mean_ms_per_box']:.1f} ms |",
        f"| 平均单框耗时下降比例 | **{rj['latency']['mean_latency_drop_ratio']:.2%}** |",
        f"| 全量总耗时 | {rj['latency']['before_total_seconds']:.1f}s → {rj['latency']['after_total_seconds']:.1f}s |",
        "",
        "### 5.4 142 条人工金标的端到端影响",
        "",
        "| 指标 | 接入前 | 接入 Tier 2.5 后 | 变化 |",
        "|---|---:|---:|---:|",
        f"| 端到端 F1 | {gj['before']['f1']:.4f} | {gj['after']['f1']:.4f} | {gj['delta']['f1']:+.4f} |",
        f"| FAR | {_fmt(gj['before']['far'])} | {_fmt(gj['after']['far'])} | {_fmt(gj['delta']['far'], signed=True)} |",
        f"| FRR | {_fmt(gj['before']['frr'])} | {_fmt(gj['after']['frr'])} | {_fmt(gj['delta']['frr'], signed=True)} |",
        f"| 准确率 | {gj['before']['accuracy']:.4f} | {gj['after']['accuracy']:.4f} | "
        f"{gj['after']['accuracy'] - gj['before']['accuracy']:+.4f} |",
        f"| 人工复核率 | {gj['before']['manual_review_rate']:.4f} | {gj['after']['manual_review_rate']:.4f} | "
        f"{gj['delta']['manual_review_rate']:+.4f} |",
        "",
        f"- Tier 2.5 在金标上拒识：{gj['after']['tier2_5_reject_true_negative']} 个真负样本、"
        f"{gj['after']['tier2_5_reject_false_negative']} 个正样本（误拒）。",
        f"- Tier 2.5 在金标上确认：{gj['after']['tier2_5_confirm_true_positive']} 个真阳样本、"
        f"{gj['after']['tier2_5_confirm_false_positive']} 个负样本（误确认）。",
        "",
        "### 5.5 逐类分流（接入 Tier 2.5 后）",
        "",
        "| 类别 | Tier1 | Tier2 | Tier2.5 确认 | Tier2.5 拒识 | Tier3 |",
        "|---|---:|---:|---:|---:|---:|",
        *[
            f"| {cls} | {counts.get('tier1_auto_confirm', 0)} | {counts.get('tier2_auto_reject', 0)} | "
            f"{counts.get('tier2_5_jev_omni_confirm', 0)} | {counts.get('tier2_5_jev_omni_reject', 0)} | "
            f"{counts.get('tier3_vlm_verify', 0)} |"
            for cls, counts in rj["per_class"].items()
        ],
    ]
    (args.output_dir / "five_class_gate_eval.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print("\n".join(md))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
