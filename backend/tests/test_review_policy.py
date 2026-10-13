from __future__ import annotations

import pytest

from app.services.review_policy import (
    BACKGROUND_COUNTEREVIDENCE_REASON_CODE,
    RED_OVERLAY_ELIGIBLE_CLASSES,
    ROOFTOP_FORBIDDEN_CLASSES,
    apply_review_policy,
    normalize_scene,
)


def _scene(label: str = "ground", confidence: float = 0.95) -> dict:
    return {"label": label, "confidence": confidence, "reason": "test scene"}


def _red_evidence(**overrides) -> dict:
    evidence = {
        "red_overlay_occlusion": True,
        "overlay_obscures": True,
        "background_only": False,
        "target_visible": "occluded",
        "occlusion_type": "red_overlay",
    }
    evidence.update(overrides)
    return evidence


@pytest.mark.parametrize("class_name", sorted(RED_OVERLAY_ELIGIBLE_CLASSES))
def test_red_overlay_exception_accepts_every_vehicle_class_and_person(class_name):
    decision = apply_review_policy(
        class_name=class_name,
        model_label="uncertain",
        model_confidence=0.42,
        scene=_scene(),
        evidence=_red_evidence(),
    )

    assert decision.label == "true"
    assert decision.reason_code == "red_overlay_occlusion_exempt"
    assert decision.source == "policy"
    assert decision.model_label == "uncertain"


def test_background_counterevidence_never_auto_accepts_a_red_overlay_candidate():
    decision = apply_review_policy(
        class_name="车",
        model_label="true",
        model_confidence=0.95,
        scene=_scene(),
        evidence=_red_evidence(background_only=True),
    )

    assert decision.label == "uncertain"
    assert decision.reason_code == BACKGROUND_COUNTEREVIDENCE_REASON_CODE
    assert decision.source == "manual_review"
    assert decision.model_label == "true"


def test_non_vehicle_non_person_does_not_trigger_red_overlay_exception():
    decision = apply_review_policy(
        class_name="房屋",
        model_label="uncertain",
        scene=_scene(),
        evidence=_red_evidence(),
    )

    assert decision.label == "uncertain"
    assert decision.reason_code == "manual_review"
    assert decision.reason_code != "red_overlay_occlusion_exempt"


@pytest.mark.parametrize("class_name", sorted(ROOFTOP_FORBIDDEN_CLASSES))
def test_high_confidence_rooftop_forbidden_vehicle_is_rejected_before_red_overlay(class_name):
    decision = apply_review_policy(
        class_name=class_name,
        model_label="true",
        model_confidence=0.95,
        scene=_scene("rooftop", 0.8),
        evidence=_red_evidence(),
    )

    assert decision.label == "false"
    assert decision.reason_code == "rooftop_vehicle_rule"
    assert decision.source == "policy"


@pytest.mark.parametrize("class_name", ["公交车", "两轮车", "人员"])
def test_rooftop_allowed_classes_do_not_use_forbidden_vehicle_rule(class_name):
    decision = apply_review_policy(
        class_name=class_name,
        model_label="false",
        model_confidence=0.95,
        scene=_scene("rooftop", 0.95),
        evidence=_red_evidence(),
    )

    assert decision.label == "true"
    assert decision.reason_code == "red_overlay_occlusion_exempt"


@pytest.mark.parametrize(
    "scene",
    [None, {"label": "unknown", "confidence": 1.0, "reason": "ambiguous"}, {"label": "rooftop", "confidence": 0.79}],
)
def test_missing_or_low_confidence_rooftop_scene_keeps_uncertain_for_uncertain_model(scene):
    decision = apply_review_policy(
        class_name="车",
        model_label="uncertain",
        scene=scene,
    )

    assert decision.label == "uncertain"
    assert decision.source == "manual_review"
    assert decision.scene["label"] in {"unknown", "rooftop"}


def test_old_model_response_without_scene_defaults_to_unknown():
    scene = normalize_scene(None)

    assert scene == {"label": "unknown", "confidence": 0.0, "reason": ""}


def test_invalid_model_confidence_is_not_kept_as_a_numeric_ticket():
    decision = apply_review_policy(
        class_name="车",
        model_label="uncertain",
        model_confidence=float("nan"),
    )

    assert decision.model_confidence is None
