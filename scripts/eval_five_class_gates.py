#!/usr/bin/env python3
"""五类业务门禁与三级级联路由的离线回放评估（可复现）。

评估三组指标：
  1. 车/三轮车几何门禁：IoU > 0.75 同目标双框抑制率、containment > 0.88 三轮车车斗误报拦截率；
  2. 三级级联路由：Tier1/Tier2/Tier3 分流比、blue_illegal_roof 100% 强制进入 Tier3 的合规率；
  3. 阈值搜索与逐类表：build_by_class_table / find_optimal_threshold_for_far(max_far=0.10)。

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
    TIER2_AUTO_REJECT_THRESHOLD,
    apply_pre_llm_geometric_gate,
    compute_bbox_iou,
    compute_containment_ratio,
    partition_candidates_by_cascade_tier,
)
from qwen3vl_verify import (  # noqa: E402
    find_optimal_threshold_for_far,
    format_by_class_markdown_table,
    summarize_predictions,
)

DEFAULT_CANDIDATES = ROOT / "data/ktd-test-real-monitoring-full-20260924/output/raw_predictions.jsonl"
DEFAULT_GOLD = ROOT / "data/ktd-clean-source-full-evaluation-20260924/predictions_scored.jsonl"

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

    `hybrid_confidence` 是「模型对自身决策的置信度」，不是 P(true)。阈值搜索要求
    score 语义为 P(true)，因此对 false 决策取 1-confidence 做翻转；uncertain 记 0.5。
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
        if decision == "true":
            score_true = float(conf)
        elif decision == "false":
            score_true = 1.0 - float(conf)
        else:
            score_true = 0.5
        records.append(
            {
                "class_name": str(r["class_name"]),
                "gold": str(r["human_label"]),
                "score": round(score_true, 6),
                "prediction": decision,
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
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    report = {
        "generated_at": "deterministic",
        "geometry_gate": evaluate_geometry_sweep(),
        "real_replay": evaluate_real_replay(args.candidates),
        "real_two_class_table": evaluate_real_two_class(args.gold),
        "five_class_benchmark": evaluate_five_class_benchmark(),
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "five_class_gate_eval.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    geo = report["geometry_gate"]
    rep = report["real_replay"]
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
        "## 3. 逐类表与阈值搜索",
        "",
        "### 3.1 真实人工金标（车/人员）",
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
        "### 3.2 五类合成基准（proxy，非真实精度）",
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
