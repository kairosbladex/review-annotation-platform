"""Deterministic, human-empty queue selection for KTD gold review.

The selector never manufactures a gold label. It only ranks candidates for
human review, keeps policy hits auditable, and exposes shortfalls when an
independent blind-source pool is not available.
"""

from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass, replace
from typing import Any, Iterable, Mapping, Sequence


POLICY_REASON_CODES = frozenset({"rooftop_vehicle_rule", "red_overlay_occlusion_exempt"})
HIGH_RISK_CLASSES = frozenset({"人员", "卡车", "三轮车", "公交车"})
KNOWN_FALSE_POSITIVE_TERMS = (
    "red_overlay",
    "red frame",
    "red box",
    "background",
    "empty box",
    "shadow",
    "text",
    "reflection",
    "road marking",
    "roof equipment",
    "红框",
    "背景",
    "空框",
    "阴影",
    "文字",
    "反光",
    "道路标线",
    "屋顶设施",
)
BLIND_SPLITS = frozenset({"blind", "blind_test", "holdout", "locked", "locked_test"})


@dataclass(frozen=True)
class GoldSelectionConfig:
    target_count: int = 240
    informative_quota: int = 120
    known_false_positive_quota: int = 40
    high_risk_quota: int = 40
    blind_quota: int = 40


DEFAULT_CONFIG = GoldSelectionConfig()


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _float(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _flag(value: Any) -> bool:
    """Coerce persisted JSON/SQLite boolean forms without treating text loosely."""
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return value == 1
    if isinstance(value, str):
        return value.strip().lower() in {"true", "1", "yes", "y", "是"}
    return False


def _candidate_key(row: Mapping[str, Any]) -> tuple[str, int]:
    image_key = str(row.get("image_sha256") or row.get("image_id") or row.get("file_path") or "")
    try:
        candidate_index = int(row.get("candidate_index", -1))
    except (TypeError, ValueError):
        candidate_index = -1
    return image_key, candidate_index


def _policy(row: Mapping[str, Any]) -> Mapping[str, Any]:
    direct = row.get("policy")
    if isinstance(direct, Mapping):
        return direct
    return _mapping(_mapping(row.get("decision")).get("policy"))


def _model_scene(row: Mapping[str, Any]) -> Mapping[str, Any]:
    value = row.get("scene")
    if isinstance(value, Mapping):
        return value
    policy_scene = _policy(row).get("scene")
    if isinstance(policy_scene, Mapping):
        return policy_scene
    decision_scene = _mapping(row.get("decision")).get("scene")
    return decision_scene if isinstance(decision_scene, Mapping) else {}


def _model_evidence(row: Mapping[str, Any]) -> Mapping[str, Any]:
    value = row.get("evidence")
    if isinstance(value, Mapping):
        return value
    policy_evidence = _policy(row).get("evidence")
    if isinstance(policy_evidence, Mapping):
        return policy_evidence
    decision_evidence = _mapping(row.get("decision")).get("evidence")
    return decision_evidence if isinstance(decision_evidence, Mapping) else {}


def _source_group(row: Mapping[str, Any]) -> str:
    explicit = row.get("source_group")
    if explicit not in (None, ""):
        return str(explicit)
    source_row_ids = row.get("source_row_ids")
    if isinstance(source_row_ids, Sequence) and not isinstance(source_row_ids, (str, bytes)) and source_row_ids:
        return f"source_row:{source_row_ids[0]}"
    image_id = row.get("image_id")
    return f"image:{image_id}" if image_id not in (None, "") else "unknown"


def _source_split(row: Mapping[str, Any]) -> str:
    for key in ("source_split", "dataset_split", "split"):
        value = row.get(key)
        if value not in (None, ""):
            return str(value).strip().lower()
    return ""


def _is_blind_source(row: Mapping[str, Any]) -> bool:
    if _flag(row.get("is_blind_source")) or _flag(row.get("blind_holdout")):
        return True
    if _flag(row.get("source_independent")):
        return True
    return _source_split(row) in BLIND_SPLITS


def _bbox_features(row: Mapping[str, Any]) -> tuple[float | None, bool]:
    bbox = row.get("bbox")
    if not isinstance(bbox, Sequence) or isinstance(bbox, (str, bytes)) or len(bbox) != 4:
        return None, False
    try:
        x1, y1, x2, y2 = (float(value) for value in bbox)
    except (TypeError, ValueError):
        return None, False
    if not all(math.isfinite(value) for value in (x1, y1, x2, y2)) or x2 <= x1 or y2 <= y1:
        return None, False
    area = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    edge = x1 <= 0.01 or y1 <= 0.01 or x2 >= 0.99 or y2 >= 0.99
    return area, edge


def _information_profile(row: Mapping[str, Any]) -> tuple[float, list[str]]:
    score = 0.0
    reasons: list[str] = []
    decision = _mapping(row.get("decision"))
    policy = _policy(row)
    model_label = str(row.get("model_label") or decision.get("model_label") or policy.get("model_label") or "")
    if str(row.get("auto_label") or "") == "uncertain" or model_label == "uncertain":
        score += 6.0
        reasons.append("uncertain")

    votes = decision.get("votes")
    vote_labels: set[str] = set()
    vote_confidences: list[float] = []
    if isinstance(votes, Sequence) and not isinstance(votes, (str, bytes)):
        for vote in votes:
            if not isinstance(vote, Mapping):
                continue
            label = str(vote.get("label") or "")
            if label:
                vote_labels.add(label)
            confidence = _float(vote.get("confidence"))
            if confidence is not None:
                vote_confidences.append(confidence)
    if len(vote_labels) > 1:
        score += 4.0
        reasons.append("model_disagreement")
    if "uncertain" in vote_labels:
        score += 1.5
        reasons.append("uncertain_vote")

    model_confidence = _float(row.get("model_confidence"))
    if model_confidence is None:
        model_confidence = _float(policy.get("model_confidence"))
    if model_confidence is None and vote_confidences:
        model_confidence = min(vote_confidences)
    if model_confidence is not None and model_confidence < 0.65:
        score += 1.0 + (0.65 - max(0.0, model_confidence)) * 4.0
        reasons.append("low_model_confidence")

    detector_confidence = _float(row.get("confidence"))
    if detector_confidence is not None and detector_confidence < 0.4:
        score += 2.0
        reasons.append("low_detector_confidence")

    area, edge = _bbox_features(row)
    if area is not None and area < 0.0025:
        score += 3.0
        reasons.append("small_target")
    elif area is not None and area < 0.01:
        score += 1.0
        reasons.append("medium_small_target")
    if edge:
        score += 2.0
        reasons.append("edge_box")
    if _flag(row.get("needs_manual_review")):
        score += 2.0
        reasons.append("source_conflict")
    first_pass = row.get("first_pass_label")
    second_pass = row.get("second_pass_label")
    if first_pass not in (None, "") and second_pass not in (None, "") and first_pass != second_pass:
        score += 3.0
        reasons.append("pass_disagreement")
    return score, reasons


def _known_false_positive(row: Mapping[str, Any]) -> bool:
    def text_values(value: Any) -> Iterable[str]:
        if isinstance(value, Mapping):
            for nested in value.values():
                yield from text_values(nested)
        elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
            for nested in value:
                yield from text_values(nested)
        elif isinstance(value, str):
            yield value

    # Search evidence/reasons, not serialized field names. For example,
    # ``background_only: false`` must not itself become a known-FP signal.
    relevant = {
        key: row.get(key)
        for key in (
            "class_name",
            "reason",
            "reason_code",
            "model_label",
            "policy",
            "evidence",
            "model_evidence",
            "decision",
        )
        if key in row
    }
    text = " ".join(text_values(relevant)).lower()
    return any(term in text for term in KNOWN_FALSE_POSITIVE_TERMS)


def _is_policy_hit(row: Mapping[str, Any]) -> bool:
    policy = _policy(row)
    return policy.get("source") == "policy" and str(policy.get("reason_code") or "") in POLICY_REASON_CODES


def _all_reasons(row: Mapping[str, Any]) -> list[str]:
    reasons: list[str] = []
    policy = _policy(row)
    reason_code = str(policy.get("reason_code") or "")
    if _is_policy_hit(row):
        reasons.append(reason_code)
    _, information_reasons = _information_profile(row)
    reasons.extend(information_reasons)
    if _known_false_positive(row):
        reasons.append("known_false_positive")
    if str(row.get("class_name") or "") in HIGH_RISK_CLASSES:
        reasons.append("high_risk_class")
    if _is_blind_source(row):
        reasons.append("independent_blind_source")
    return list(dict.fromkeys(reasons))


def _rank(rows: Iterable[Mapping[str, Any]]) -> list[Mapping[str, Any]]:
    return sorted(
        rows,
        key=lambda row: (-_information_profile(row)[0], _candidate_key(row)),
    )


def _diverse_blind_rank(rows: Iterable[Mapping[str, Any]]) -> list[Mapping[str, Any]]:
    grouped: dict[str, list[Mapping[str, Any]]] = {}
    for row in _rank(rows):
        grouped.setdefault(_source_group(row), []).append(row)
    ordered: list[Mapping[str, Any]] = []
    while grouped:
        for group in sorted(tuple(grouped)):
            ordered.append(grouped[group].pop(0))
            if not grouped[group]:
                del grouped[group]
    return ordered


def _gold_record(
    row: Mapping[str, Any],
    *,
    selection_bucket: str,
    queue_rank: int,
) -> dict[str, Any]:
    policy = dict(_policy(row))
    model_scene = dict(_model_scene(row))
    model_evidence = dict(_model_evidence(row))
    output = dict(row)
    output.update({
        "queue_rank": queue_rank,
        "selection_bucket": selection_bucket,
        "selection_reason_codes": _all_reasons(row),
        "selection_score": round(_information_profile(row)[0], 6),
        "source_group": _source_group(row),
        "source_split": _source_split(row) or None,
        "is_blind_source": _is_blind_source(row),
        "model_label": row.get("model_label") or _mapping(row.get("decision")).get("model_label") or policy.get("model_label"),
        "policy": policy,
        "model_scene": model_scene,
        "model_evidence": model_evidence,
        "human_confirmed": False,
        "human_label": None,
        "presence": None,
        "class_match": None,
        "box_fit": None,
        "scene": None,
        "occlusion_type": None,
        "overlay_obscures": None,
        "reason_code": None,
    })
    return output


def select_gold_review_candidates(
    rows: Iterable[Mapping[str, Any]],
    *,
    config: GoldSelectionConfig | None = None,
    target_count: int | None = None,
) -> list[dict[str, Any]]:
    """Return a deterministic review queue with every human field unset."""
    config = config or DEFAULT_CONFIG
    if target_count is not None:
        config = replace(config, target_count=target_count)
    if min(
        config.target_count,
        config.informative_quota,
        config.known_false_positive_quota,
        config.high_risk_quota,
        config.blind_quota,
    ) < 0:
        raise ValueError("gold selection counts must be non-negative")

    unique: dict[tuple[str, int], Mapping[str, Any]] = {}
    for row in rows:
        if not isinstance(row, Mapping) or _flag(row.get("human_confirmed")):
            continue
        if row.get("status") not in (None, "succeeded"):
            continue
        unique.setdefault(_candidate_key(row), row)
    candidates = list(unique.values())

    selected: dict[tuple[str, int], tuple[Mapping[str, Any], str]] = {}
    mandatory = _rank(row for row in candidates if _is_policy_hit(row))
    for row in mandatory:
        selected[_candidate_key(row)] = (row, "policy_rule_hit")
    effective_target = max(config.target_count, len(mandatory))

    def add(pool: Iterable[Mapping[str, Any]], bucket: str, quota: int) -> None:
        added = 0
        for row in pool:
            if len(selected) >= effective_target or added >= quota:
                break
            key = _candidate_key(row)
            if key in selected:
                continue
            selected[key] = (row, bucket)
            added += 1

    add(
        _diverse_blind_rank(row for row in candidates if _is_blind_source(row)),
        "independent_blind_source",
        config.blind_quota,
    )
    add(
        _rank(row for row in candidates if str(row.get("class_name") or "") in HIGH_RISK_CLASSES),
        "high_risk_class",
        config.high_risk_quota,
    )
    add(
        _rank(row for row in candidates if _known_false_positive(row)),
        "known_false_positive",
        config.known_false_positive_quota,
    )
    add(
        _rank(row for row in candidates if _information_profile(row)[1]),
        "informative",
        config.informative_quota,
    )
    add(_rank(candidates), "information_rank_fill", effective_target)

    return [
        _gold_record(row, selection_bucket=bucket, queue_rank=index)
        for index, (row, bucket) in enumerate(selected.values(), start=1)
    ]


def summarize_gold_selection(
    rows: Sequence[Mapping[str, Any]],
    *,
    config: GoldSelectionConfig | None = None,
) -> dict[str, Any]:
    config = config or DEFAULT_CONFIG
    buckets = Counter(str(row.get("selection_bucket") or "unknown") for row in rows)
    reasons = Counter(
        str(reason)
        for row in rows
        for reason in (row.get("selection_reason_codes") or [])
    )
    independent_blind_rows = sum(_flag(row.get("is_blind_source")) for row in rows)
    high_risk_rows = sum(str(row.get("class_name") or "") in HIGH_RISK_CLASSES for row in rows)
    known_false_positive_rows = sum(_known_false_positive(row) for row in rows)
    informative_rows = sum(bool(_information_profile(row)[1]) for row in rows)
    requested_quotas = {
        "independent_blind_source": config.blind_quota,
        "high_risk_class": config.high_risk_quota,
        "known_false_positive": config.known_false_positive_quota,
        "informative": config.informative_quota,
    }
    observed_coverage = {
        "independent_blind_source": independent_blind_rows,
        "high_risk_class": high_risk_rows,
        "known_false_positive": known_false_positive_rows,
        "informative": informative_rows,
    }
    return {
        "rows": len(rows),
        "requested_target_count": config.target_count,
        "policy_rule_hits": buckets["policy_rule_hit"],
        "independent_blind_rows": independent_blind_rows,
        "requested_quotas": requested_quotas,
        "observed_coverage": observed_coverage,
        "quota_shortfalls": {
            key: max(0, requested - observed_coverage[key])
            for key, requested in requested_quotas.items()
        },
        "target_shortfall": max(0, config.target_count - len(rows)),
        "by_selection_bucket": dict(sorted(buckets.items())),
        "by_reason_code": dict(sorted(reasons.items())),
        "human_fields_prefilled": sum(
            any(row.get(field) is not None for field in (
                "human_label",
                "presence",
                "class_match",
                "box_fit",
                "scene",
                "occlusion_type",
                "overlay_obscures",
                "reason_code",
            ))
            for row in rows
        ),
    }


build_gold_review_manifest = select_gold_review_candidates


__all__ = [
    "DEFAULT_CONFIG",
    "GoldSelectionConfig",
    "HIGH_RISK_CLASSES",
    "POLICY_REASON_CODES",
    "build_gold_review_manifest",
    "select_gold_review_candidates",
    "summarize_gold_selection",
]
