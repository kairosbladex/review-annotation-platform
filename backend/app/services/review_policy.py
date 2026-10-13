"""Shared, deterministic policy for aerial candidate rechecks.

The model is an evidence provider.  This module owns the small amount of
business logic that must be identical in the API auto-annotation path and in
the offline KTD recheck runner.  It deliberately keeps ``uncertain`` as a
first-class result: absent or malformed evidence never becomes a positive or
negative label by accident.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Literal, Mapping


TRUE_LABEL = "true"
FALSE_LABEL = "false"
UNCERTAIN_LABEL = "uncertain"
REVIEW_LABELS = frozenset({TRUE_LABEL, FALSE_LABEL, UNCERTAIN_LABEL})

VEHICLE_CLASSES = frozenset({"车", "两轮车", "三轮车", "卡车", "公交车"})
ROOFTOP_FORBIDDEN_CLASSES = frozenset({"车", "卡车", "三轮车"})
PERSON_CLASS = "人员"
RED_OVERLAY_ELIGIBLE_CLASSES = VEHICLE_CLASSES | {PERSON_CLASS}

ROOFTOP_SCENE = "rooftop"
GROUND_SCENE = "ground"
UNKNOWN_SCENE = "unknown"
SCENE_LABELS = frozenset({ROOFTOP_SCENE, GROUND_SCENE, UNKNOWN_SCENE})
DEFAULT_ROOFTOP_CONFIDENCE_THRESHOLD = 0.8
BACKGROUND_COUNTEREVIDENCE_REASON_CODE = "red_overlay_background_counterevidence"

_SCENE_ALIASES = {
    "rooftop": ROOFTOP_SCENE,
    "roof": ROOFTOP_SCENE,
    "屋顶": ROOFTOP_SCENE,
    "屋面": ROOFTOP_SCENE,
    "ground": GROUND_SCENE,
    "ground_level": GROUND_SCENE,
    "地面": GROUND_SCENE,
    "道路": GROUND_SCENE,
    "unknown": UNKNOWN_SCENE,
    "不确定": UNKNOWN_SCENE,
    "未知": UNKNOWN_SCENE,
}

_TRUE_STRINGS = frozenset({"true", "1", "yes", "y", "是"})
_FALSE_STRINGS = frozenset({"false", "0", "no", "n", "否"})
_BACKGROUND_TARGET_VALUES = frozenset({
    "background",
    "background_only",
    "empty",
    "none",
    "no",
    "not_visible",
    "false",
    "背景",
    "空",
    "无",
})
_VISIBLE_TARGET_VALUES = frozenset({
    "true",
    "yes",
    "partial",
    "occluded",
    "部分可见",
    "遮挡",
})
_RED_OCCLUSION_TYPES = frozenset({
    "red_overlay",
    "red_frame",
    "red_box",
    "red_rectangle",
    "红框",
    "红色框体",
})


def _coerce_bool(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)) and value in {0, 1}:
        return bool(value)
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in _TRUE_STRINGS:
            return True
        if normalized in _FALSE_STRINGS:
            return False
    return None


def _coerce_confidence(value: Any) -> float:
    if isinstance(value, bool):
        return 0.0
    try:
        confidence = float(value)
    except (TypeError, ValueError):
        return 0.0
    return confidence if 0.0 <= confidence <= 1.0 else 0.0


def normalize_label(value: Any) -> str:
    """Normalize model/product aliases without inventing a label."""
    if value is None:
        return UNCERTAIN_LABEL
    normalized = str(value).strip().lower()
    aliases = {
        "keep": TRUE_LABEL,
        "confirmed": TRUE_LABEL,
        "accept": TRUE_LABEL,
        "reject": FALSE_LABEL,
        "rejected": FALSE_LABEL,
        "drop": FALSE_LABEL,
        "manual_review": UNCERTAIN_LABEL,
        "doubtful": UNCERTAIN_LABEL,
        "unknown": UNCERTAIN_LABEL,
    }
    normalized = aliases.get(normalized, normalized)
    return normalized if normalized in REVIEW_LABELS else UNCERTAIN_LABEL


def normalize_scene(value: Any = None, *, confidence: Any = None, reason: Any = None) -> dict[str, Any]:
    """Return a stable scene object; omitted/invalid scene is ``unknown``."""
    if isinstance(value, Mapping):
        raw_label = value.get("label", value.get("scene", value.get("name")))
        if confidence is None:
            confidence = value.get("confidence", value.get("score"))
        if reason is None:
            reason = value.get("reason", "")
    else:
        raw_label = value
    label = _SCENE_ALIASES.get(str(raw_label).strip().lower(), UNKNOWN_SCENE) if raw_label is not None else UNKNOWN_SCENE
    confidence_value = _coerce_confidence(confidence)
    reason_value = str(reason or "")[:1000]
    if label == UNKNOWN_SCENE:
        confidence_value = min(confidence_value, 0.0)
    return {"label": label, "confidence": confidence_value, "reason": reason_value}


def normalize_overlay_evidence(
    evidence: Mapping[str, Any] | None = None,
    *,
    red_overlay_occlusion: Any = None,
    overlay_obscures: Any = None,
    background_only: Any = None,
    target_visible: Any = None,
    occlusion_type: Any = None,
) -> dict[str, Any]:
    """Normalize optional model evidence used by the red-overlay exception.

    ``red_overlay_occlusion`` is intentionally explicit.  A generic
    ``overlay_obscures`` flag can describe tree/building occlusion and is not
    enough on its own to trigger the business rule unless the model also says
    the occluder is the red review frame.
    """
    source: Mapping[str, Any] = evidence if isinstance(evidence, Mapping) else {}

    def first(*keys: str, direct: Any = None) -> Any:
        if direct is not None:
            return direct
        for key in keys:
            if key in source and source[key] is not None:
                return source[key]
        return None

    nested = source.get("evidence")
    if isinstance(nested, Mapping):
        merged = dict(nested)
        merged.update({key: value for key, value in source.items() if key not in merged})
        source = merged

    red_flag = _coerce_bool(first(
        "red_overlay_occlusion",
        "red_frame_occlusion",
        "red_overlay",
        "red_frame",
        direct=red_overlay_occlusion,
    ))
    obscures_flag = _coerce_bool(first("overlay_obscures", "overlay_covers_target", direct=overlay_obscures))
    background_flag = _coerce_bool(first(
        "background_only",
        "empty_box",
        "box_background_only",
        "explicit_background",
        "no_target",
        direct=background_only,
    ))
    visible_value = first("target_visible", "target_evidence", direct=target_visible)
    visible_text = str(visible_value).strip().lower() if visible_value is not None else None
    type_value = first("occlusion_type", "overlay_type", direct=occlusion_type)
    type_text = str(type_value).strip().lower() if type_value is not None else None
    if red_flag is None and obscures_flag is True and type_text in _RED_OCCLUSION_TYPES:
        red_flag = True
    if background_flag is None and visible_text in _BACKGROUND_TARGET_VALUES:
        background_flag = True
    return {
        "red_overlay_occlusion": red_flag,
        "overlay_obscures": obscures_flag,
        "background_only": background_flag,
        "target_visible": visible_text,
        "occlusion_type": type_text,
        "reason": str(first("reason", "evidence_reason", direct=None) or "")[:1000],
    }


@dataclass(frozen=True)
class PolicyDecision:
    label: Literal["true", "false", "uncertain"]
    reason_code: str
    source: Literal["model", "policy", "manual_review"]
    model_label: Literal["true", "false", "uncertain"]
    model_confidence: float | None
    scene: dict[str, Any]
    evidence: dict[str, Any]

    @property
    def applied(self) -> bool:
        return self.source == "policy"

    def as_dict(self) -> dict[str, Any]:
        return {
            "final_label": self.label,
            "reason_code": self.reason_code,
            "source": self.source,
            "applied": self.applied,
            "model_label": self.model_label,
            "model_confidence": self.model_confidence,
            "scene": self.scene,
            "evidence": self.evidence,
        }


def apply_review_policy(
    *,
    class_name: Any,
    model_label: Any,
    scene: Any = None,
    evidence: Mapping[str, Any] | None = None,
    red_overlay_occlusion: Any = None,
    overlay_obscures: Any = None,
    background_only: Any = None,
    target_visible: Any = None,
    occlusion_type: Any = None,
    model_confidence: Any = None,
    rooftop_confidence_threshold: float = DEFAULT_ROOFTOP_CONFIDENCE_THRESHOLD,
) -> PolicyDecision:
    """Apply the shared rule order and return a fully auditable decision."""
    normalized_class = str(class_name or "").strip()
    normalized_model_label = normalize_label(model_label)
    normalized_scene = normalize_scene(scene)
    evidence_value = normalize_overlay_evidence(
        evidence,
        red_overlay_occlusion=red_overlay_occlusion,
        overlay_obscures=overlay_obscures,
        background_only=background_only,
        target_visible=target_visible,
        occlusion_type=occlusion_type,
    )
    try:
        threshold = float(rooftop_confidence_threshold)
    except (TypeError, ValueError):
        threshold = DEFAULT_ROOFTOP_CONFIDENCE_THRESHOLD
    threshold = min(max(threshold, 0.0), 1.0)
    confidence_value: float | None
    try:
        candidate_confidence = float(model_confidence) if model_confidence is not None else None
    except (TypeError, ValueError):
        candidate_confidence = None
    confidence_value = (
        candidate_confidence
        if candidate_confidence is not None
        and math.isfinite(candidate_confidence)
        and 0.0 <= candidate_confidence <= 1.0
        else None
    )

    # This rule deliberately runs first: it is an explicit business constraint
    # and must win over a red frame that happens to cover the same candidate.
    if (
        normalized_scene["label"] == ROOFTOP_SCENE
        and normalized_scene["confidence"] >= threshold
        and normalized_class in ROOFTOP_FORBIDDEN_CLASSES
    ):
        return PolicyDecision(
            label=FALSE_LABEL,
            reason_code="rooftop_vehicle_rule",
            source="policy",
            model_label=normalized_model_label,
            model_confidence=confidence_value,
            scene=normalized_scene,
            evidence=evidence_value,
        )

    red_overlay_confirmed = evidence_value["red_overlay_occlusion"] is True
    background_counterevidence = evidence_value["background_only"] is True
    if evidence_value["target_visible"] in _BACKGROUND_TARGET_VALUES:
        background_counterevidence = True
    if (
        normalized_class in RED_OVERLAY_ELIGIBLE_CLASSES
        and red_overlay_confirmed
        and background_counterevidence
    ):
        return PolicyDecision(
            label=UNCERTAIN_LABEL,
            reason_code=BACKGROUND_COUNTEREVIDENCE_REASON_CODE,
            source="manual_review",
            model_label=normalized_model_label,
            model_confidence=confidence_value,
            scene=normalized_scene,
            evidence=evidence_value,
        )
    if (
        normalized_class in RED_OVERLAY_ELIGIBLE_CLASSES
        and red_overlay_confirmed
        and not background_counterevidence
    ):
        return PolicyDecision(
            label=TRUE_LABEL,
            reason_code="red_overlay_occlusion_exempt",
            source="policy",
            model_label=normalized_model_label,
            model_confidence=confidence_value,
            scene=normalized_scene,
            evidence=evidence_value,
        )

    if normalized_model_label in {TRUE_LABEL, FALSE_LABEL}:
        return PolicyDecision(
            label=normalized_model_label,
            reason_code="model_consensus",
            source="model",
            model_label=normalized_model_label,
            model_confidence=confidence_value,
            scene=normalized_scene,
            evidence=evidence_value,
        )

    return PolicyDecision(
        label=UNCERTAIN_LABEL,
        reason_code="manual_review",
        source="manual_review",
        model_label=normalized_model_label,
        model_confidence=confidence_value,
        scene=normalized_scene,
        evidence=evidence_value,
    )


__all__ = [
    "BACKGROUND_COUNTEREVIDENCE_REASON_CODE",
    "DEFAULT_ROOFTOP_CONFIDENCE_THRESHOLD",
    "FALSE_LABEL",
    "GROUND_SCENE",
    "PERSON_CLASS",
    "PolicyDecision",
    "RED_OVERLAY_ELIGIBLE_CLASSES",
    "REVIEW_LABELS",
    "ROOFTOP_FORBIDDEN_CLASSES",
    "ROOFTOP_SCENE",
    "TRUE_LABEL",
    "UNCERTAIN_LABEL",
    "UNKNOWN_SCENE",
    "VEHICLE_CLASSES",
    "apply_review_policy",
    "normalize_label",
    "normalize_overlay_evidence",
    "normalize_scene",
]
