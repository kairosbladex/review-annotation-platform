from __future__ import annotations

from app.services.gold_selection import (
    GoldSelectionConfig,
    select_gold_review_candidates,
    summarize_gold_selection,
)


def _row(
    index: int,
    *,
    class_name: str = "车",
    image_id: str | None = None,
    model_label: str = "uncertain",
    reason: str = "视觉证据不足",
    source_group: str | None = None,
    source_split: str | None = None,
    is_blind_source: bool | int = False,
    human_confirmed: bool | int = False,
    bbox: list[float] | None = None,
) -> dict:
    return {
        "image_id": image_id or f"image-{index}",
        "candidate_index": 0,
        "class_name": class_name,
        "model_label": model_label,
        "model_confidence": 0.5,
        "confidence": 0.7,
        "bbox": bbox or [0.1, 0.1, 0.2, 0.2],
        "reason": reason,
        "source_group": source_group,
        "source_split": source_split,
        "is_blind_source": is_blind_source,
        "human_confirmed": human_confirmed,
        "status": "succeeded",
        "evidence": {
            "red_overlay_occlusion": False,
            "background_only": False,
        },
    }


def test_selector_does_not_treat_evidence_field_names_as_known_false_positive_signals():
    rows = [_row(1, reason="普通候选", model_label="true")]

    selected = select_gold_review_candidates(rows, target_count=1)

    assert len(selected) == 1
    assert "known_false_positive" not in selected[0]["selection_reason_codes"]
    assert selected[0]["human_label"] is None
    assert selected[0]["presence"] is None
    assert selected[0]["scene"] is None
    assert summarize_gold_selection(selected)["human_fields_prefilled"] == 0


def test_selector_keeps_all_policy_hits_even_when_the_target_is_smaller():
    rows = []
    for index in range(3):
        rows.append({
            **_row(index, model_label="true"),
            "policy": {
                "source": "policy",
                "reason_code": "rooftop_vehicle_rule",
                "final_label": "false",
                "scene": {"label": "rooftop", "confidence": 0.95},
            },
        })

    selected = select_gold_review_candidates(rows, target_count=2)

    assert len(selected) == 3
    assert {row["selection_bucket"] for row in selected} == {"policy_rule_hit"}
    assert all(row["human_label"] is None for row in selected)


def test_selector_fills_disjoint_quota_buckets_and_reports_shortfalls():
    rows = [
        _row(1, source_group="blind-a", source_split="locked_test", is_blind_source=1),
        _row(2, source_group="blind-b", source_split="locked_test", is_blind_source=True),
        _row(3, class_name="卡车"),
        _row(4, class_name="卡车"),
        _row(5, reason="红框遮挡背景"),
        _row(6, bbox=[0.0, 0.0, 0.02, 0.02]),
        _row(7, bbox=[0.01, 0.01, 0.03, 0.03]),
        _row(8, human_confirmed=1),
    ]
    config = GoldSelectionConfig(
        target_count=5,
        informative_quota=1,
        known_false_positive_quota=1,
        high_risk_quota=1,
        blind_quota=1,
    )

    selected = select_gold_review_candidates(rows, config=config)
    summary = summarize_gold_selection(selected, config=config)

    assert len(selected) == 5
    assert all(row["human_confirmed"] is False for row in selected)
    assert {row["selection_bucket"] for row in selected[:4]} == {
        "independent_blind_source",
        "high_risk_class",
        "known_false_positive",
        "informative",
    }
    assert summary["requested_quotas"]["independent_blind_source"] == 1
    assert summary["observed_coverage"]["independent_blind_source"] >= 1
    assert summary["quota_shortfalls"]["independent_blind_source"] == 0
    assert summary["target_shortfall"] == 0


def test_selector_diversifies_independent_source_groups_and_exposes_blind_shortfall():
    rows = [
        _row(1, source_group="same", source_split="locked_test"),
        _row(2, source_group="same", source_split="locked_test"),
        _row(3, source_group="other", source_split="locked_test"),
    ]
    config = GoldSelectionConfig(
        target_count=2,
        informative_quota=0,
        known_false_positive_quota=0,
        high_risk_quota=0,
        blind_quota=4,
    )

    selected = select_gold_review_candidates(rows, config=config)
    summary = summarize_gold_selection(selected, config=config)

    assert {row["source_group"] for row in selected} == {"same", "other"}
    assert summary["quota_shortfalls"]["independent_blind_source"] == 2
    assert summary["target_shortfall"] == 0
