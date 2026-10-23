"""142 条金标四组同分母对比（A/B/C/D）的口径回归。

锁定三件事：
1. ``uncertain`` 在自动裁决口径下按「未确认」计为负例，分母恒为样本总数；
2. FAR / FRR 的分母是真实的负/正样本数，不是「已裁决」子集；
3. ``evaluate_gold_four_way`` 在最小合成夹具上能跑通四组，且 Tier 2.5 真的被触发。
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))

from eval_five_class_gates import (  # noqa: E402
    _four_way_metrics,
    evaluate_gold_four_way,
)


def test_four_way_metrics_treats_uncertain_as_not_confirmed():
    decisions = [
        {"gold": "true", "prediction": "true"},
        {"gold": "true", "prediction": "uncertain"},
        {"gold": "false", "prediction": "false"},
        {"gold": "false", "prediction": "uncertain"},
    ]

    metrics = _four_way_metrics(decisions, latencies_ms=[10.0, 20.0, 30.0, 40.0])

    assert metrics["n"] == 4
    assert (metrics["tp"], metrics["fp"], metrics["tn"], metrics["fn"]) == (1, 0, 2, 1)
    assert metrics["manual_review_n"] == 2
    assert metrics["manual_review_rate"] == 0.5
    # FAR / FRR 的分母是真实负/正样本数（2 / 2），不是已裁决子集
    assert metrics["far"] == 0.0
    assert metrics["frr"] == 0.5
    assert metrics["accuracy"] == 0.5
    assert metrics["mean_latency_ms"] == 25.0


def test_four_way_metrics_keeps_same_denominator_across_groups():
    rows = [
        {"gold": "true", "prediction": "true"},
        {"gold": "false", "prediction": "true"},
        {"gold": "false", "prediction": "false"},
    ]
    metrics = _four_way_metrics(rows, latencies_ms=[])
    assert metrics["n"] == 3
    assert metrics["precision"] == 0.5
    assert metrics["recall"] == 1.0
    assert metrics["f1"] == round(2 * 0.5 * 1.0 / 1.5, 4)
    assert metrics["mean_latency_ms"] == 0.0


def _write_jsonl(path: Path, rows: list[dict]) -> Path:
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8"
    )
    return path


def test_evaluate_gold_four_way_runs_cascade_with_real_probabilities(tmp_path):
    # 一个事件两个候选：一个高置信刚体（走 Tier1），一个低分候选（走 Tier2.5）
    manifest_rows = [
        {
            "candidate_id": "evt1:000",
            "event_id": "evt1",
            "class_name": "车辆",
            "detector_confidence": 0.93,
            "bbox_px": [10.0, 10.0, 60.0, 60.0],
            "frame_size": [960, 720],
        },
        {
            "candidate_id": "evt1:001",
            "event_id": "evt1",
            "class_name": "车辆",
            "detector_confidence": 0.40,
            "bbox_px": [300.0, 300.0, 420.0, 420.0],
            "frame_size": [960, 720],
        },
    ]
    overlap_rows = [
        {
            "full_candidate_id": "evt1:000",
            "legacy_candidate_id": "P001",
            "event_id": "evt1",
            "class_name": "车辆",
            "human_label": "true",
        },
        {
            "full_candidate_id": "evt1:001",
            "legacy_candidate_id": "P002",
            "event_id": "evt1",
            "class_name": "车辆",
            "human_label": "false",
        },
    ]
    gold_rows = [
        {"candidate_id": "P001", "hybrid_decision": "true", "hybrid_confidence": 0.9},
        {"candidate_id": "P002", "hybrid_decision": "false", "hybrid_confidence": 0.8},
    ]
    prob_rows = [
        {"candidate_id": "evt1:000", "p_true": 0.95, "p_false": 0.05, "latency_ms": 200.0},
        {"candidate_id": "evt1:001", "p_true": 0.005, "p_false": 0.995, "latency_ms": 210.0},
    ]

    report = evaluate_gold_four_way(
        _write_jsonl(tmp_path / "manifest.jsonl", manifest_rows),
        _write_jsonl(tmp_path / "overlap.jsonl", overlap_rows),
        _write_jsonl(tmp_path / "gold.jsonl", gold_rows),
        probs_path=_write_jsonl(tmp_path / "probs.jsonl", prob_rows),
        vlm_log_path=None,
    )

    assert report["samples"] == 2
    assert report["missing_jev_probs"] == 0
    assert set(report["groups"]) == {
        "A_detector_only",
        "B_jev_omni_only",
        "C_qwen3vl_only",
        "D_cascade_fused",
    }
    assert all(block["n"] == 2 for block in report["groups"].values())
    # 组 A 只按 0.25 门槛：两个候选都被接受 → 1 个 FP
    assert report["groups"]["A_detector_only"]["tp"] == 1
    assert report["groups"]["A_detector_only"]["fp"] == 1
    # 组 D：Tier1 收下第一个，Tier2.5 用真机 p_true=0.005（≤ 拒识阈值 0.01）拒识第二个
    assert report["cascade_tier_counts"]["tier1_auto_confirm"] == 1
    assert report["cascade_tier_counts"]["tier2_5_jev_omni_reject"] == 1
    assert report["groups"]["D_cascade_fused"]["tp"] == 1
    assert report["groups"]["D_cascade_fused"]["tn"] == 1
