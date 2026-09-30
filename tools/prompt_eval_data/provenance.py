"""Source lineage, leakage-group, and clean-pixel readiness for prompt-gold."""

from __future__ import annotations

from collections import defaultdict
from datetime import datetime, timezone
import json
import math
import re
from typing import Any

from .common import sha256_text


_METADATA_GROUP_FIELDS = (
    ("task", "source_task_id"),
    ("task", "task_id"),
    ("video", "source_video_id"),
    ("video", "video_id"),
    ("track", "source_track_id"),
    ("track", "track_id"),
    ("event", "event_id"),
)
_FORMAL_ANCHOR_KIND = "task"
_PIXEL_STATUSES = {
    "clean": "clean",
    "verified_clean": "clean",
    "overlay": "overlay",
    "overlay_present": "overlay",
    "overlay_detected": "overlay",
    "unknown": "unknown",
    "unverified": "unknown",
    "unverified_overlay_risk": "unknown",
}
_CLEAN_REQUIRED_FIELDS = (
    "clean_object_key",
    "clean_object_version",
    "clean_sha256",
    "clean_width",
    "clean_height",
    "annotated_object_key",
    "annotated_object_version",
    "annotated_sha256",
    "annotated_width",
    "annotated_height",
    "overlay_check_method",
    "overlay_check_version",
    "overlay_evidence_sha256",
    "mapping_method",
    "mapping_version",
    "geometry_transform",
    "source_system",
    "source_table",
    "source_row_id",
)
_FRAME_IDENTITY_FIELDS = ("capture_id", "frame_id", "frame_timestamp")
_SHA256_RE = re.compile(r"[0-9a-f]{64}")
_URL_OR_QUERY_RE = re.compile(
    r"://|^//|[?#]|(?:^|[&;])[^&;=\s]+=[^&;\s]*(?:[&;]|$)",
    re.IGNORECASE,
)
_URI_SCHEME_RE = re.compile(r"^[a-z][a-z0-9+.-]*:", re.IGNORECASE)
_TEMPORAL_FIELDS = (
    "metadata.source_pixels.frame_timestamp",
    "metadata.frame_timestamp",
    "metadata.event_time",
)
_SAFE_METADATA_FIELDS = {
    "annex_id",
    "busi_id",
    "capture_id",
    "event_id",
    "event_time",
    "frame_id",
    "frame_timestamp",
    "source_event_id",
    "source_task_id",
    "source_track_id",
    "source_video_id",
    "task_id",
    "terminal_id",
    "track_id",
    "video_id",
}
_SAFE_SOURCE_PIXEL_FIELDS = set(_CLEAN_REQUIRED_FIELDS) | {
    "capture_id",
    "frame_id",
    "frame_timestamp",
    "overlay_status",
}


def _scalar(value: Any) -> str | None:
    if value is None or isinstance(value, (dict, list, tuple, set, bool)):
        return None
    text = str(value).strip()
    return text or None


def _safe_external_scalar(value: Any) -> str | None:
    """Return a provenance scalar only when it cannot carry a URL or query."""
    text = _scalar(value)
    if text is None or (isinstance(value, float) and not math.isfinite(value)):
        return None
    if _URI_SCHEME_RE.match(text) or _URL_OR_QUERY_RE.search(text):
        return None
    return text


def _is_synthetic_annex_event(row: dict[str, Any], event_id: str) -> bool:
    if event_id.startswith("ktd-annex-"):
        return True
    metadata = row.get("metadata") if isinstance(row.get("metadata"), dict) else {}
    annex_id = _scalar(metadata.get("annex_id"))
    return bool(annex_id and event_id == f"ktd-annex-{annex_id}")


def _safe_metadata(metadata: Any) -> dict[str, Any] | None:
    if not isinstance(metadata, dict):
        return None
    safe = {}
    for key in sorted(_SAFE_METADATA_FIELDS & metadata.keys()):
        value = metadata[key]
        if _safe_external_scalar(value) is not None:
            safe[key] = value.strip() if isinstance(value, str) else value
    source_pixels = metadata.get("source_pixels")
    if isinstance(source_pixels, dict):
        pixel_safe = {}
        for key in sorted(_SAFE_SOURCE_PIXEL_FIELDS & source_pixels.keys()):
            value = source_pixels[key]
            if key == "geometry_transform" and value == {"type": "identity"}:
                pixel_safe[key] = value
            elif _safe_external_scalar(value) is not None:
                pixel_safe[key] = value.strip() if isinstance(value, str) else value
        if pixel_safe:
            safe["source_pixels"] = pixel_safe
    return safe


def source_record(row: dict[str, Any]) -> dict[str, Any]:
    event_id = _safe_external_scalar(row.get("source_event_id"))
    if event_id is None:
        event_semantics = "missing"
    elif _is_synthetic_annex_event(row, event_id):
        event_semantics = "synthetic_import_row_id"
    else:
        event_semantics = "upstream_event_id"
    return {
        "candidate_id": row["candidate_id"],
        "sample_id": row["sample_id"],
        "model_run_id": row["model_run_id"],
        "target_index": row["target_index"],
        "source_type": row.get("source_type"),
        "source_service": row.get("source_service"),
        "source_event_id": event_id,
        "source_event_semantics": event_semantics,
        "session_id": _safe_external_scalar(row.get("session_id")),
        "capture_time": row.get("capture_time"),
        "camera_id": row.get("camera_id"),
        "rule_code": row.get("rule_code"),
        "metadata": _safe_metadata(row.get("metadata")),
        "metadata_sha256": (
            sha256_text(str(row["metadata_json"]))
            if row.get("metadata_json") not in (None, "")
            else None
        ),
        "metadata_error": row.get("metadata_error"),
        "asset": {
            "bucket": row.get("asset_bucket"),
            "kind": row.get("asset_kind"),
            "object_key": row.get("object_key"),
            "declared_sha256": row.get("asset_sha256"),
        },
    }


def _group_identities(row: dict[str, Any]) -> list[tuple[str, str, str]]:
    metadata = row.get("metadata") if isinstance(row.get("metadata"), dict) else {}
    identities = []
    for kind, field in _METADATA_GROUP_FIELDS:
        value = _safe_external_scalar(metadata.get(field))
        if value and not (kind == "event" and _is_synthetic_annex_event(row, value)):
            identities.append((kind, value, f"metadata.{field}"))
    session_id = _safe_external_scalar(row.get("session_id"))
    if session_id:
        identities.append(("session", session_id, "samples.session_id"))
    event_id = _safe_external_scalar(row.get("source_event_id"))
    if event_id and not _is_synthetic_annex_event(row, event_id):
        identities.append(("event", event_id, "samples.source_event_id"))
    return identities


def _pixel_payload(row: dict[str, Any]) -> dict[str, Any] | None:
    metadata = row.get("metadata") if isinstance(row.get("metadata"), dict) else {}
    nested = metadata.get("source_pixels")
    payload = nested if isinstance(nested, dict) else metadata
    raw_status = _safe_external_scalar(payload.get("overlay_status"))
    if raw_status is None:
        return None
    status = _PIXEL_STATUSES.get(raw_status.lower())
    if status is None:
        status = "unknown"
    return {**payload, "overlay_status": status, "declared_overlay_status": raw_status}


def _clean_validation_errors(
    payload: dict[str, Any],
    *,
    object_key: str,
    image_sha256: str,
    width: int,
    height: int,
    declared_asset_sha256: Any,
) -> list[str]:
    errors = [
        f"missing_{field}"
        for field in _CLEAN_REQUIRED_FIELDS
        if _safe_external_scalar(payload.get(field)) is None and field != "geometry_transform"
    ]
    if payload.get("geometry_transform") in (None, ""):
        errors.append("missing_geometry_transform")
    if not any(_safe_external_scalar(payload.get(field)) is not None for field in _FRAME_IDENTITY_FIELDS):
        errors.append("missing_capture_or_frame_identity")
    if _safe_external_scalar(payload.get("clean_object_key")) != object_key:
        errors.append("clean_object_key_not_exported_object")
    if _safe_external_scalar(payload.get("clean_sha256")) != image_sha256:
        errors.append("clean_sha256_mismatch")
    try:
        if int(payload.get("clean_width")) != width or int(payload.get("clean_height")) != height:
            errors.append("clean_dimensions_mismatch")
    except (OverflowError, TypeError, ValueError):
        errors.append("clean_dimensions_invalid")
    if _safe_external_scalar(payload.get("annotated_object_key")) != object_key:
        errors.append("non_identity_annotated_mapping_not_supported")
    if _safe_external_scalar(payload.get("annotated_sha256")) != image_sha256:
        errors.append("annotated_sha256_not_verified_exported_bytes")
    try:
        if int(payload.get("annotated_width")) != width or int(payload.get("annotated_height")) != height:
            errors.append("annotated_dimensions_mismatch")
    except (OverflowError, TypeError, ValueError):
        errors.append("annotated_dimensions_invalid")
    if _safe_external_scalar(declared_asset_sha256) != image_sha256:
        errors.append("asset_sha256_not_verified_exported_bytes")
    if _safe_external_scalar(payload.get("clean_object_version")) != _safe_external_scalar(
        payload.get("annotated_object_version")
    ):
        errors.append("identity_object_version_mismatch")
    if _safe_external_scalar(payload.get("mapping_method")) not in {"identity", "same_asset"}:
        errors.append("unsupported_identity_mapping_method")
    evidence_sha = _safe_external_scalar(payload.get("overlay_evidence_sha256"))
    if evidence_sha is not None and not _SHA256_RE.fullmatch(evidence_sha):
        errors.append("overlay_evidence_sha256_invalid")
    transform = payload.get("geometry_transform")
    if transform not in ("identity", {"type": "identity"}):
        errors.append("non_identity_geometry_not_supported")
    return sorted(set(errors))


def _clean_declaration_sha(payload: dict[str, Any]) -> str:
    identity_fields = set(_CLEAN_REQUIRED_FIELDS) | set(_FRAME_IDENTITY_FIELDS)
    identity_fields.discard("source_row_id")
    normalized = {field: payload.get(field) for field in sorted(identity_fields)}
    return sha256_text(
        json.dumps(normalized, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    )


def _source_pixels(
    rows: list[dict[str, Any]], *, object_key: str, image_sha256: str, width: int, height: int
) -> dict[str, Any]:
    declarations = []
    declared_samples: set[str] = set()
    clean_declaration_shas: set[str] = set()
    for row in sorted(rows, key=lambda item: item["candidate_id"]):
        payload = _pixel_payload(row)
        if payload is None:
            continue
        declared_samples.add(row["sample_id"])
        errors = []
        if payload["overlay_status"] == "clean":
            errors = _clean_validation_errors(
                payload,
                object_key=object_key,
                image_sha256=image_sha256,
                width=width,
                height=height,
                declared_asset_sha256=row.get("asset_sha256"),
            )
            clean_declaration_shas.add(_clean_declaration_sha(payload))
        declarations.append(
            {
                "candidate_id": row["candidate_id"],
                "sample_id": row["sample_id"],
                "declared_status": payload["declared_overlay_status"],
                "normalized_status": payload["overlay_status"],
                "validation_errors": errors,
                "declaration_sha256": _clean_declaration_sha(payload),
            }
        )

    normalized = {item["normalized_status"] for item in declarations}
    clean_errors = sorted(
        {error for item in declarations if item["normalized_status"] == "clean" for error in item["validation_errors"]}
    )
    expected_samples = {row["sample_id"] for row in rows}
    missing_samples = sorted(expected_samples - declared_samples)
    clean_errors.extend(
        f"missing_source_pixel_declaration_for_sample:{sample_id}"
        for sample_id in missing_samples
    )
    if len(clean_declaration_shas) > 1:
        clean_errors.append("conflicting_clean_source_declarations")
    clean_errors = sorted(set(clean_errors))
    if "overlay" in normalized:
        status = "overlay"
        blockers = ["overlay_declared"]
    elif "unknown" in normalized:
        status = "unknown"
        blockers = ["source_pixel_status_unknown"]
    elif "clean" in normalized and not clean_errors:
        status = "clean"
        blockers = []
    elif "clean" in normalized:
        status = "unknown"
        blockers = clean_errors
    else:
        status = "unknown"
        blockers = ["missing_source_pixel_declaration"]
    return {
        "status": status,
        "exported_object_key": object_key,
        "exported_sha256": image_sha256,
        "declarations": declarations,
        "blocking_reasons": blockers,
        "formal_input_allowed": status == "clean",
    }


def _timestamp(value: Any) -> tuple[str | None, str | None]:
    raw = _scalar(value)
    if raw is None:
        return None, None
    candidate = raw[:-1] + "+00:00" if raw.endswith(("Z", "z")) else raw
    try:
        parsed = datetime.fromisoformat(candidate)
    except ValueError:
        return None, "invalid_iso8601"
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        return None, "timezone_missing"
    normalized = parsed.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")
    return normalized, None


def _temporal_provenance(rows: list[dict[str, Any]]) -> dict[str, Any]:
    evidence = []
    for row in sorted(rows, key=lambda item: item["candidate_id"]):
        metadata = row.get("metadata") if isinstance(row.get("metadata"), dict) else {}
        pixel_metadata = metadata.get("source_pixels")
        pixel_metadata = pixel_metadata if isinstance(pixel_metadata, dict) else {}
        for field, value in (
            ("metadata.event_time", metadata.get("event_time")),
            ("metadata.frame_timestamp", metadata.get("frame_timestamp")),
            ("metadata.source_pixels.frame_timestamp", pixel_metadata.get("frame_timestamp")),
        ):
            raw = _safe_external_scalar(value)
            normalized, error = _timestamp(raw)
            if raw:
                evidence.append(
                    {
                        "sample_id": row["sample_id"],
                        "field": field,
                        "value": raw,
                        "normalized_value": normalized,
                        "validation_error": error,
                    }
                )
    unique = {
        (item["sample_id"], item["field"], item["value"], item["normalized_value"], item["validation_error"])
        for item in evidence
    }
    evidence = [
        {
            "sample_id": sample_id,
            "field": field,
            "value": value,
            "normalized_value": normalized_value,
            "validation_error": validation_error,
        }
        for sample_id, field, value, normalized_value, validation_error in sorted(unique)
    ]
    invalid = [item for item in evidence if item["validation_error"]]
    selected_field = next(
        (field for field in _TEMPORAL_FIELDS if any(item["field"] == field for item in evidence)),
        None,
    )
    selected_values = {
        item["normalized_value"]
        for item in evidence
        if item["field"] == selected_field and item["validation_error"] is None
    }
    conflicting_fields = sorted(
        field
        for field in _TEMPORAL_FIELDS
        if len(
            {
                item["normalized_value"]
                for item in evidence
                if item["field"] == field and item["validation_error"] is None
            }
        )
        > 1
    )
    expected_samples = {row["sample_id"] for row in rows}
    selected_samples = {
        item["sample_id"]
        for item in evidence
        if item["field"] == selected_field and item["validation_error"] is None
    }
    missing_samples = sorted(expected_samples - selected_samples)
    if invalid:
        status = "invalid_upstream_time"
    elif conflicting_fields:
        status = "conflicting_upstream_times"
    elif selected_field is None:
        status = "temporal_provenance_unavailable"
    elif missing_samples:
        status = "incomplete_upstream_time"
    elif len(selected_values) == 1:
        status = "upstream_time_available"
    else:
        status = "conflicting_upstream_times"
    return {
        "status": status,
        "selected_field": selected_field,
        "selected_value": next(iter(selected_values)) if len(selected_values) == 1 else None,
        "conflicting_fields": conflicting_fields,
        "missing_samples": missing_samples,
        "evidence": evidence,
    }


def _box_source_pixels(
    image_provenance: dict[str, Any], rows: list[dict[str, Any]]
) -> dict[str, Any]:
    aliases = {(row["candidate_id"], row["sample_id"]) for row in rows}
    return {
        **image_provenance,
        "declarations": [
            declaration
            for declaration in image_provenance["declarations"]
            if (declaration["candidate_id"], declaration["sample_id"]) in aliases
        ],
    }


def _box_temporal(
    image_provenance: dict[str, Any], rows: list[dict[str, Any]]
) -> dict[str, Any]:
    sample_ids = {row["sample_id"] for row in rows}
    return {
        **image_provenance,
        "missing_samples": [
            sample_id
            for sample_id in image_provenance["missing_samples"]
            if sample_id in sample_ids
        ],
        "evidence": [
            item
            for item in image_provenance["evidence"]
            if item["sample_id"] in sample_ids
        ],
    }


class _DisjointSet:
    def __init__(self) -> None:
        self.parent: dict[str, str] = {}

    def find(self, item: str) -> str:
        self.parent.setdefault(item, item)
        if self.parent[item] != item:
            self.parent[item] = self.find(self.parent[item])
        return self.parent[item]

    def union(self, left: str, right: str) -> None:
        left_root, right_root = self.find(left), self.find(right)
        if left_root == right_root:
            return
        low, high = sorted((left_root, right_root))
        self.parent[high] = low


def attach_provenance(
    boxes: dict[str, dict[str, Any]],
    *,
    source_rows: list[dict[str, Any]] | None = None,
    image_sha_by_object_key: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Attach deterministic provenance and connected source grouping to boxes."""
    rows_by_image: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    identities_by_image: dict[str, set[tuple[str, str, str]]] = defaultdict(set)
    dsu = _DisjointSet()

    for canonical_id, box in boxes.items():
        bbox = box.get("bbox_px")
        if (
            not isinstance(bbox, (list, tuple))
            or len(bbox) != 4
            or any(not isinstance(value, int) or isinstance(value, bool) for value in bbox)
        ):
            raise ValueError(f"{canonical_id}: bbox_px must contain four integers")
        if bbox[2] <= bbox[0] or bbox[3] <= bbox[1]:
            raise ValueError(f"{canonical_id}: bbox_px must have positive width and height")
        image_sha = box["image_sha256"]
        for row in box["rows"]:
            rows_by_image[image_sha][row["candidate_id"]] = row

    graph_rows = source_rows
    if graph_rows is None:
        graph_rows = [row for rows in rows_by_image.values() for row in rows.values()]
    image_sha_by_object_key = image_sha_by_object_key or {}
    for row in graph_rows:
        identities = _group_identities(row)
        tokens = sorted({f"{kind}:{value}" for kind, value, _ in identities})
        for token in tokens:
            dsu.find(token)
        for token in tokens[1:]:
            dsu.union(tokens[0], token)
        image_sha = image_sha_by_object_key.get(row.get("object_key"))
        if image_sha:
            identities_by_image[image_sha].update(identities)

    # Preserve the legacy direct-call behavior when no all-row image mapping is supplied.
    for image_sha, rows in rows_by_image.items():
        for row in rows.values():
            identities_by_image[image_sha].update(_group_identities(row))

    for identities in identities_by_image.values():
        tokens = sorted({f"{kind}:{value}" for kind, value, _ in identities})
        for token in tokens:
            dsu.find(token)
        for token in tokens[1:]:
            dsu.union(tokens[0], token)

    members_by_root: dict[str, set[str]] = defaultdict(set)
    for token in sorted(dsu.parent):
        members_by_root[dsu.find(token)].add(token)

    grouping_by_image = {}
    for image_sha in sorted(rows_by_image):
        identities = sorted(identities_by_image[image_sha])
        if identities:
            first_token = f"{identities[0][0]}:{identities[0][1]}"
            component = sorted(members_by_root[dsu.find(first_token)])
            anchors = [
                token for token in component if token.startswith(f"{_FORMAL_ANCHOR_KIND}:")
            ]
            anchor_kind = _FORMAL_ANCHOR_KIND if anchors else None
        else:
            component = []
            anchors = []
            anchor_kind = None
        if len(anchors) == 1:
            split_key = "source-group:" + sha256_text(anchors[0])
            grouping_by_image[image_sha] = {
                "status": "source_group_available",
                "split_key": split_key,
                "identities": [
                    {"kind": kind, "value": value, "origin": origin}
                    for kind, value, origin in identities
                ],
                "component_identities": component,
                "anchor_identities": anchors,
                "anchor_kind": anchor_kind,
                "basis": "unique_task_anchor_in_connected_component",
            }
        else:
            grouping_by_image[image_sha] = {
                "status": "source_group_conflict" if len(anchors) > 1 else "grouping_unavailable",
                "split_key": image_sha,
                "identities": [
                    {"kind": kind, "value": value, "origin": origin}
                    for kind, value, origin in identities
                ],
                "component_identities": component,
                "anchor_identities": anchors,
                "anchor_kind": anchor_kind,
                "basis": (
                    "conflicting_task_anchors_image_sha256_fallback"
                    if anchors
                    else "missing_task_anchor_image_sha256_fallback"
                ),
            }

    pixel_status_by_image = {}
    temporal_status_by_image = {}
    for box in boxes.values():
        image_sha = box["image_sha256"]
        if image_sha not in pixel_status_by_image:
            pixel_status_by_image[image_sha] = _source_pixels(
                list(rows_by_image[image_sha].values()),
                object_key=box["image_object_key"],
                image_sha256=image_sha,
                width=box["image_width"],
                height=box["image_height"],
            )
            temporal_status_by_image[image_sha] = _temporal_provenance(
                list(rows_by_image[image_sha].values())
            )
        box["provenance"] = {
            "source_records": [source_record(row) for row in box["rows"]],
            "grouping": grouping_by_image[image_sha],
            "source_pixels": _box_source_pixels(
                pixel_status_by_image[image_sha], box["rows"]
            ),
            "temporal": _box_temporal(temporal_status_by_image[image_sha], box["rows"]),
        }

    image_count = len(rows_by_image)
    grouping_available = sum(
        item["status"] == "source_group_available" for item in grouping_by_image.values()
    )
    grouping_conflicts = sum(
        item["status"] == "source_group_conflict" for item in grouping_by_image.values()
    )
    grouping_unavailable = sum(
        item["status"] == "grouping_unavailable" for item in grouping_by_image.values()
    )
    pixel_counts = {
        status: sum(item["status"] == status for item in pixel_status_by_image.values())
        for status in ("clean", "overlay", "unknown")
    }
    temporal_available = sum(
        item["status"] == "upstream_time_available" for item in temporal_status_by_image.values()
    )
    temporal_conflicts = sum(
        item["status"] == "conflicting_upstream_times" for item in temporal_status_by_image.values()
    )
    temporal_invalid = sum(
        item["status"] == "invalid_upstream_time" for item in temporal_status_by_image.values()
    )
    temporal_incomplete = sum(
        item["status"] == "incomplete_upstream_time" for item in temporal_status_by_image.values()
    )
    temporal_unavailable = sum(
        item["status"] == "temporal_provenance_unavailable"
        for item in temporal_status_by_image.values()
    )
    metadata_error_images = sum(
        any(row.get("metadata_error") for row in rows.values()) for rows in rows_by_image.values()
    )
    blockers = []
    if image_count == 0:
        blockers.append("no_images")
    if grouping_unavailable:
        blockers.append("grouping_unavailable")
    if grouping_conflicts:
        blockers.append("source_group_conflict")
    if pixel_counts["overlay"]:
        blockers.append("overlay_source_pixels")
    if pixel_counts["unknown"]:
        blockers.append("source_pixel_status_unknown")
    if temporal_available != image_count:
        blockers.append("temporal_provenance_unavailable_or_conflicting")
    if metadata_error_images:
        blockers.append("metadata_parse_errors")
    return {
        "source_group_images": grouping_available,
        "grouping_unavailable_images": grouping_unavailable,
        "source_group_conflict_images": grouping_conflicts,
        "source_groups": len(
            {item["split_key"] for item in grouping_by_image.values() if item["status"] == "source_group_available"}
        ),
        "source_pixels_clean_images": pixel_counts["clean"],
        "source_pixels_overlay_images": pixel_counts["overlay"],
        "source_pixels_unknown_images": pixel_counts["unknown"],
        "upstream_time_images": temporal_available,
        "temporal_provenance_unavailable_images": temporal_unavailable,
        "temporal_provenance_incomplete_images": temporal_incomplete,
        "temporal_provenance_conflict_images": temporal_conflicts,
        "temporal_provenance_invalid_images": temporal_invalid,
        "metadata_error_images": metadata_error_images,
        "formal_training_ready": not blockers,
        "formal_training_blockers": blockers,
    }
