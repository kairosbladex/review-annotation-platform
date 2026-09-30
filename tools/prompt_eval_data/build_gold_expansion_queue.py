"""Build a deterministic human-gold expansion queue from a frozen snapshot.

The queue contains model inputs only. It deliberately omits all prior model
decisions so that the annotator is not primed by teacher or service output.
Locked-test images and already labeled or quarantined canonical boxes are
excluded by default. Source pixels must carry explicit clean provenance.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import sys
import tempfile
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.prompt_eval_data.splits import LOCKED_TEST, SELECTION, assign_split  # noqa: E402


SIZE_BINS = ("tiny_lt16", "small_16_47", "medium_48_95", "large_ge96")
HUMAN_LABELS = ("false", "true")
BLOCKED_SOURCE_STATUSES = ("unknown", "overlay")
_GROUP_ANCHOR_PRECEDENCE = ("task", "session", "video", "track")
_GROUP_METADATA_FIELDS = (
    ("task", "source_task_id"),
    ("task", "task_id"),
    ("video", "source_video_id"),
    ("video", "video_id"),
    ("track", "source_track_id"),
    ("track", "track_id"),
    ("event", "event_id"),
)
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
_TEMPORAL_FIELDS = (
    "metadata.source_pixels.frame_timestamp",
    "metadata.frame_timestamp",
    "metadata.event_time",
)
_SHA256_RE = re.compile(r"[0-9a-f]{64}")
_QUEUE_SOURCE_FILES = (
    "inference_records.jsonl",
    "selection_gold.jsonl",
    "selection_inputs.jsonl",
    "quarantine.jsonl",
    "locked_test_inputs.jsonl",
)


def _parse_jsonl_bytes(data: bytes, name: str) -> list[dict[str, Any]]:
    rows = []
    for line_number, line in enumerate(data.decode("utf-8").splitlines(), 1):
        if not line.strip():
            continue
        row = json.loads(line)
        if not isinstance(row, dict):
            raise ValueError(f"{name}:{line_number}: row must be an object")
        rows.append(row)
    return rows


def _sha256sums(data: bytes) -> dict[str, str]:
    sums = {}
    for line_number, line in enumerate(data.decode("utf-8").splitlines(), 1):
        try:
            sha256, name = line.split("  ", 1)
        except ValueError:
            raise ValueError(f"SHA256SUMS:{line_number}: malformed line") from None
        if not _SHA256_RE.fullmatch(sha256) or not name:
            raise ValueError(f"SHA256SUMS:{line_number}: malformed entry")
        if name in sums:
            raise ValueError(f"SHA256SUMS: duplicate entry: {name}")
        sums[name] = sha256
    return sums


def _validate_snapshot(
    snapshot: Path,
) -> tuple[dict[str, Any], bytes, dict[str, bytes]]:
    manifest_path = snapshot / "snapshot_manifest.json"
    if not manifest_path.is_file():
        raise ValueError("snapshot_manifest.json is required")
    manifest_bytes = manifest_path.read_bytes()
    manifest = json.loads(manifest_bytes)
    if not isinstance(manifest, dict):
        raise ValueError("snapshot manifest must be an object")
    schema_version = manifest.get("schema_version")
    if schema_version not in {"prompt-gold/1", "prompt-gold/2"}:
        raise ValueError(f"unsupported snapshot schema: {schema_version!r}")
    if manifest.get("snapshot_id") != snapshot.name:
        raise ValueError("snapshot ID does not match directory name")

    entries: dict[str, dict[str, Any]] = {}
    for entry in manifest.get("files", []):
        if not isinstance(entry, dict) or not isinstance(entry.get("path"), str):
            raise ValueError("invalid snapshot manifest file entry")
        name = entry["path"]
        if name in entries:
            raise ValueError(f"duplicate snapshot manifest file entry: {name}")
        entries[name] = entry

    required = [*_QUEUE_SOURCE_FILES, "SHA256SUMS"]
    if schema_version == "prompt-gold/2":
        required.append("source_readiness.json")
    verified = {}
    for name in required:
        entry = entries.get(name)
        path = snapshot / name
        if entry is None or not path.is_file():
            raise ValueError(f"required snapshot member missing: {name}")
        data = path.read_bytes()
        if (
            entry.get("sha256") != hashlib.sha256(data).hexdigest()
            or entry.get("bytes") != len(data)
        ):
            raise ValueError(f"snapshot member hash mismatch: {name}")
        verified[name] = data

    declared_sums = _sha256sums(verified["SHA256SUMS"])
    manifest_sums = {
        name: entry.get("sha256") for name, entry in entries.items() if name != "SHA256SUMS"
    }
    if declared_sums != manifest_sums:
        raise ValueError("SHA256SUMS does not match snapshot manifest")

    if schema_version == "prompt-gold/2":
        snapshot_id = manifest.get("snapshot_id")
        if not isinstance(snapshot_id, str) or not snapshot_id.startswith("pg2-"):
            raise ValueError("prompt-gold/2 snapshot ID must be content-addressed")
        identity_payload = "".join(
            f"{name}:{sha256}\n" for name, sha256 in sorted(declared_sums.items())
        ).encode("utf-8")
        expected_snapshot_id = "pg2-" + hashlib.sha256(identity_payload).hexdigest()[:12]
        if snapshot_id != expected_snapshot_id:
            raise ValueError("prompt-gold/2 snapshot ID does not match member hashes")
        readiness = json.loads(verified["source_readiness.json"])
        if readiness != manifest.get("readiness"):
            raise ValueError("source readiness does not match snapshot manifest")
    return manifest, manifest_bytes, verified


def _size_bin(row: dict[str, Any]) -> str:
    bbox = row["bbox_px"]
    if not isinstance(bbox, list) or len(bbox) != 4:
        raise ValueError("bbox_px must contain four coordinates")
    width = int(bbox[2]) - int(bbox[0])
    height = int(bbox[3]) - int(bbox[1])
    if width <= 0 or height <= 0:
        raise ValueError("bbox_px must have positive width and height")
    long_edge = max(width, height)
    if long_edge < 16:
        return SIZE_BINS[0]
    if long_edge < 48:
        return SIZE_BINS[1]
    if long_edge < 96:
        return SIZE_BINS[2]
    return SIZE_BINS[3]


def _scalar(value: Any) -> str | None:
    if value is None or isinstance(value, (dict, list, tuple, set, bool)):
        return None
    text = str(value).strip()
    return text or None


def _provenance_error(row: dict[str, Any], section: str, detail: str) -> ValueError:
    return ValueError(
        f"{row.get('canonical_id')}: {section} provenance is not validated: {detail}"
    )


def _alias_source_records(row: dict[str, Any]) -> list[dict[str, Any]]:
    aliases = row.get("aliases")
    provenance = row.get("provenance")
    records = provenance.get("source_records") if isinstance(provenance, dict) else None
    if not isinstance(aliases, list) or not aliases or not all(
        isinstance(alias, dict) for alias in aliases
    ):
        raise _provenance_error(row, "alias", "aliases must be a non-empty object list")
    if not isinstance(records, list) or not records or not all(
        isinstance(record, dict) for record in records
    ):
        raise _provenance_error(
            row, "alias", "source_records must be a non-empty object list"
        )
    if row.get("alias_count") != len(aliases):
        raise _provenance_error(row, "alias", "alias_count does not match aliases")

    def identities(items: list[dict[str, Any]], name: str) -> set[tuple[str, str]]:
        pairs = []
        for item in items:
            candidate_id = item.get("candidate_id")
            sample_id = item.get("sample_id")
            if not isinstance(candidate_id, str) or not candidate_id:
                raise _provenance_error(row, "alias", f"{name} has no candidate_id")
            if not isinstance(sample_id, str) or not sample_id:
                raise _provenance_error(row, "alias", f"{name} has no sample_id")
            pairs.append((candidate_id, sample_id))
        if len(set(pairs)) != len(pairs):
            raise _provenance_error(row, "alias", f"{name} has duplicate identities")
        return set(pairs)

    if identities(aliases, "aliases") != identities(records, "source_records"):
        raise _provenance_error(row, "alias", "source_records do not cover every alias")
    return records


def _clean_declaration_sha(payload: dict[str, Any]) -> str:
    identity_fields = set(_CLEAN_REQUIRED_FIELDS) | set(_FRAME_IDENTITY_FIELDS)
    identity_fields.discard("source_row_id")
    normalized = {field: payload.get(field) for field in sorted(identity_fields)}
    encoded = json.dumps(
        normalized, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    )
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _validate_positive_dimension(
    row: dict[str, Any], payload: dict[str, Any], field: str, expected: Any
) -> None:
    value = payload.get(field)
    if isinstance(value, bool):
        raise _provenance_error(row, "clean source", f"{field} is not an integer")
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        raise _provenance_error(row, "clean source", f"{field} is not an integer") from None
    if parsed <= 0 or parsed != expected:
        raise _provenance_error(row, "clean source", f"{field} does not match exported pixels")


def _source_pixel_status(row: dict[str, Any]) -> str:
    provenance = row.get("provenance")
    source_pixels = provenance.get("source_pixels") if isinstance(provenance, dict) else None
    status = source_pixels.get("status") if isinstance(source_pixels, dict) else None
    if status == "clean":
        declarations = source_pixels.get("declarations")
        if (
            not isinstance(source_pixels, dict)
            or source_pixels.get("formal_input_allowed") is not True
            or source_pixels.get("blocking_reasons") != []
            or source_pixels.get("exported_object_key") != row.get("image_object_key")
            or source_pixels.get("exported_sha256") != row.get("image_sha256")
            or not isinstance(declarations, list)
            or not declarations
            or not all(isinstance(item, dict) for item in declarations)
        ):
            raise _provenance_error(row, "clean source", "aggregate claim is incomplete")

        records = _alias_source_records(row)
        declared_by_alias: dict[tuple[str, str], dict[str, Any]] = {}
        for declaration in declarations:
            identity = (declaration.get("candidate_id"), declaration.get("sample_id"))
            if not all(isinstance(value, str) and value for value in identity):
                raise _provenance_error(
                    row, "clean source", "declaration has no alias identity"
                )
            if identity in declared_by_alias:
                raise _provenance_error(
                    row, "clean source", "duplicate declaration for one alias"
                )
            declared_by_alias[identity] = declaration

        record_identities = {
            (record["candidate_id"], record["sample_id"]) for record in records
        }
        if set(declared_by_alias) != record_identities:
            raise _provenance_error(
                row, "clean source", "declarations do not cover every alias"
            )

        declaration_hashes = set()
        for record in records:
            identity = (record["candidate_id"], record["sample_id"])
            metadata = record.get("metadata")
            payload = metadata.get("source_pixels") if isinstance(metadata, dict) else None
            if not isinstance(payload, dict):
                raise _provenance_error(
                    row, "clean source", "alias has no source_pixels declaration"
                )
            missing = [
                field
                for field in _CLEAN_REQUIRED_FIELDS
                if field != "geometry_transform" and _scalar(payload.get(field)) is None
            ]
            if payload.get("geometry_transform") in (None, ""):
                missing.append("geometry_transform")
            if missing:
                raise _provenance_error(
                    row, "clean source", f"declaration missing {','.join(sorted(missing))}"
                )
            if not any(_scalar(payload.get(field)) for field in _FRAME_IDENTITY_FIELDS):
                raise _provenance_error(
                    row, "clean source", "declaration has no capture or frame identity"
                )

            raw_status = _scalar(payload.get("overlay_status"))
            if raw_status is None or raw_status.lower() not in {"clean", "verified_clean"}:
                raise _provenance_error(row, "clean source", "overlay status is not clean")
            image_sha = row.get("image_sha256")
            image_key = row.get("image_object_key")
            if not isinstance(image_sha, str) or not _SHA256_RE.fullmatch(image_sha):
                raise _provenance_error(row, "clean source", "exported SHA-256 is invalid")
            for field in ("clean_sha256", "annotated_sha256", "overlay_evidence_sha256"):
                value = _scalar(payload.get(field))
                if value is None or not _SHA256_RE.fullmatch(value):
                    raise _provenance_error(row, "clean source", f"{field} is invalid")
            if _scalar(payload.get("clean_object_key")) != image_key:
                raise _provenance_error(
                    row, "clean source", "clean object is not the exported object"
                )
            if _scalar(payload.get("annotated_object_key")) != image_key:
                raise _provenance_error(
                    row, "clean source", "annotated object is not the exported identity object"
                )
            if _scalar(payload.get("clean_sha256")) != image_sha:
                raise _provenance_error(row, "clean source", "clean SHA-256 does not match")
            if _scalar(payload.get("annotated_sha256")) != image_sha:
                raise _provenance_error(
                    row, "clean source", "annotated SHA-256 does not match"
                )
            asset = record.get("asset")
            if not isinstance(asset, dict) or _scalar(asset.get("declared_sha256")) != image_sha:
                raise _provenance_error(
                    row, "clean source", "asset SHA-256 does not match exported bytes"
                )
            _validate_positive_dimension(row, payload, "clean_width", row.get("image_width"))
            _validate_positive_dimension(row, payload, "clean_height", row.get("image_height"))
            _validate_positive_dimension(
                row, payload, "annotated_width", row.get("image_width")
            )
            _validate_positive_dimension(
                row, payload, "annotated_height", row.get("image_height")
            )
            if _scalar(payload.get("clean_object_version")) != _scalar(
                payload.get("annotated_object_version")
            ):
                raise _provenance_error(
                    row, "clean source", "identity object versions do not match"
                )
            if _scalar(payload.get("mapping_method")) not in {"identity", "same_asset"}:
                raise _provenance_error(
                    row, "clean source", "only a verified identity mapping is supported"
                )
            if payload.get("geometry_transform") not in (
                "identity",
                {"type": "identity"},
            ):
                raise _provenance_error(
                    row, "clean source", "geometry transform is not identity"
                )

            declaration_sha = _clean_declaration_sha(payload)
            declaration_hashes.add(declaration_sha)
            declaration = declared_by_alias[identity]
            if (
                declaration.get("declared_status") != raw_status
                or declaration.get("normalized_status") != "clean"
                or declaration.get("validation_errors") != []
                or declaration.get("declaration_sha256") != declaration_sha
            ):
                raise _provenance_error(
                    row, "clean source", "aggregate declaration does not match source metadata"
                )
        if len(declaration_hashes) != 1:
            raise _provenance_error(
                row, "clean source", "alias declarations disagree on pixel identity"
            )
        return "clean"
    if status == "overlay":
        return "overlay"
    return "unknown"


def _provenance_status(row: dict[str, Any], section: str, field: str, default: str) -> str:
    provenance = row.get("provenance")
    value = provenance.get(section) if isinstance(provenance, dict) else None
    status = value.get(field) if isinstance(value, dict) else None
    return status if isinstance(status, str) and status else default


def _normalize_timestamp(value: Any) -> tuple[str | None, str | None]:
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


def _temporal_value(record: dict[str, Any], field: str) -> Any:
    metadata = record.get("metadata")
    if not isinstance(metadata, dict):
        return None
    if field == "metadata.source_pixels.frame_timestamp":
        source_pixels = metadata.get("source_pixels")
        return source_pixels.get("frame_timestamp") if isinstance(source_pixels, dict) else None
    return metadata.get(field.removeprefix("metadata."))


def _temporal_status(row: dict[str, Any]) -> str:
    provenance = row.get("provenance")
    temporal = provenance.get("temporal") if isinstance(provenance, dict) else None
    status = temporal.get("status") if isinstance(temporal, dict) else None
    if status == "upstream_time_available":
        if not isinstance(temporal, dict):
            raise _provenance_error(row, "upstream time", "claim is not an object")
        records = _alias_source_records(row)
        expected_samples = {record["sample_id"] for record in records}
        actual_evidence: set[tuple[str, str, str, str | None, str | None]] = set()
        for record in records:
            for field in _TEMPORAL_FIELDS:
                raw = _scalar(_temporal_value(record, field))
                if raw is None:
                    continue
                normalized, error = _normalize_timestamp(raw)
                actual_evidence.add(
                    (record["sample_id"], field, raw, normalized, error)
                )
        selected_field = next(
            (
                field
                for field in _TEMPORAL_FIELDS
                if any(item[1] == field for item in actual_evidence)
            ),
            None,
        )
        selected_items = {
            item for item in actual_evidence if item[1] == selected_field
        }
        selected_samples = {
            item[0] for item in selected_items if item[4] is None
        }
        selected_values = {item[3] for item in selected_items if item[4] is None}
        if (
            selected_field is None
            or any(item[4] is not None for item in actual_evidence)
            or selected_samples != expected_samples
            or len(selected_values) != 1
        ):
            raise _provenance_error(
                row,
                "upstream time",
                "source metadata is invalid, incomplete, or conflicting",
            )
        selected_value = next(iter(selected_values))
        normalized_selected, selected_error = _normalize_timestamp(
            temporal.get("selected_value")
        )
        if (
            temporal.get("selected_field") != selected_field
            or selected_error is not None
            or normalized_selected != selected_value
            or temporal.get("selected_value") != selected_value
            or temporal.get("missing_samples") != []
        ):
            raise _provenance_error(
                row, "upstream time", "selected time does not match source metadata"
            )

        evidence = temporal.get("evidence")
        if not isinstance(evidence, list) or not evidence or not all(
            isinstance(item, dict) for item in evidence
        ):
            raise _provenance_error(row, "upstream time", "evidence is missing")
        claimed_evidence = {
            (
                item.get("sample_id"),
                item.get("field"),
                item.get("value"),
                item.get("normalized_value"),
                item.get("validation_error"),
            )
            for item in evidence
        }
        if len(claimed_evidence) != len(evidence) or claimed_evidence != actual_evidence:
            raise _provenance_error(
                row, "upstream time", "evidence does not match alias metadata"
            )
        return status
    return status if isinstance(status, str) and status else "temporal_provenance_unavailable"


def _is_synthetic_annex_event(record: dict[str, Any], event_id: str) -> bool:
    if event_id.startswith("ktd-annex-"):
        return True
    metadata = record.get("metadata")
    annex_id = _scalar(metadata.get("annex_id")) if isinstance(metadata, dict) else None
    return bool(annex_id and event_id == f"ktd-annex-{annex_id}")


def _source_group_tokens(record: dict[str, Any]) -> set[str]:
    metadata = record.get("metadata")
    metadata = metadata if isinstance(metadata, dict) else {}
    tokens = set()
    for kind, field in _GROUP_METADATA_FIELDS:
        value = _scalar(metadata.get(field))
        if value and not (kind == "event" and _is_synthetic_annex_event(record, value)):
            tokens.add(f"{kind}:{value}")
    session_id = _scalar(record.get("session_id"))
    if session_id:
        tokens.add(f"session:{session_id}")
    event_id = _scalar(record.get("source_event_id"))
    if event_id and not _is_synthetic_annex_event(record, event_id):
        tokens.add(f"event:{event_id}")
    return tokens


def _derive_source_groups(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    identities_by_image: defaultdict[str, set[str]] = defaultdict(set)
    for row in rows:
        image_sha = row["image_sha256"]
        identities_by_image[image_sha]
        provenance = row.get("provenance")
        records = provenance.get("source_records") if isinstance(provenance, dict) else None
        if isinstance(records, list):
            for record in records:
                if isinstance(record, dict):
                    identities_by_image[image_sha].update(_source_group_tokens(record))

    parent: dict[str, str] = {}

    def find(item: str) -> str:
        parent.setdefault(item, item)
        if parent[item] != item:
            parent[item] = find(parent[item])
        return parent[item]

    def union(left: str, right: str) -> None:
        left_root, right_root = find(left), find(right)
        if left_root == right_root:
            return
        low, high = sorted((left_root, right_root))
        parent[high] = low

    for identities in identities_by_image.values():
        ordered = sorted(identities)
        for identity in ordered:
            find(identity)
        for identity in ordered[1:]:
            union(ordered[0], identity)

    components: defaultdict[str, set[str]] = defaultdict(set)
    for identity in sorted(parent):
        components[find(identity)].add(identity)

    derived = {}
    for image_sha, identities in identities_by_image.items():
        if identities:
            component = sorted(components[find(sorted(identities)[0])])
        else:
            component = []
        anchors_by_kind = {
            kind: [token for token in component if token.startswith(f"{kind}:")]
            for kind in _GROUP_ANCHOR_PRECEDENCE
        }
        anchor_kind = next(
            (kind for kind in _GROUP_ANCHOR_PRECEDENCE if anchors_by_kind[kind]),
            None,
        )
        anchors = anchors_by_kind[anchor_kind] if anchor_kind else []
        if len(anchors) == 1:
            split_key = "source-group:" + hashlib.sha256(
                anchors[0].encode("utf-8")
            ).hexdigest()
            status = "source_group_available"
        else:
            split_key = image_sha
            status = "source_group_conflict" if anchors else "grouping_unavailable"
        derived[image_sha] = {
            "status": status,
            "split_key": split_key,
            "component_identities": component,
            "anchor_identities": anchors,
            "anchor_kind": anchor_kind,
        }
    return derived


def _split_key(row: dict[str, Any], derived: dict[str, Any]) -> str:
    provenance = row.get("provenance")
    grouping = provenance.get("grouping") if isinstance(provenance, dict) else None
    if not isinstance(grouping, dict) or grouping.get("status") != "source_group_available":
        return row["image_sha256"]

    _alias_source_records(row)
    anchor_kind = grouping.get("anchor_kind")
    anchors = grouping.get("anchor_identities")
    component = grouping.get("component_identities")
    if (
        derived.get("status") != "source_group_available"
        or anchor_kind not in _GROUP_ANCHOR_PRECEDENCE
        or anchor_kind != derived.get("anchor_kind")
        or not isinstance(anchors, list)
        or anchors != derived.get("anchor_identities")
        or len(anchors) != 1
        or not isinstance(component, list)
        or component != derived.get("component_identities")
        or grouping.get("split_key") != derived.get("split_key")
    ):
        raise _provenance_error(
            row,
            "source group",
            "anchor/component/key do not match source-record identities",
        )
    anchor_prefix, separator, anchor_value = anchors[0].partition(":")
    if separator != ":" or anchor_prefix != anchor_kind or not anchor_value:
        raise _provenance_error(row, "source group", "anchor kind or value is invalid")
    return derived["split_key"]


def _order(row: dict[str, Any]) -> tuple[str, int, str, str]:
    seed = hashlib.sha256(f"gold-expansion-v2|{row['canonical_id']}".encode("utf-8")).hexdigest()
    return (row["class_name"], SIZE_BINS.index(row["size_bin"]), seed, row["canonical_id"])


def _index_by_canonical_id(rows: list[dict[str, Any]], name: str) -> dict[str, dict[str, Any]]:
    indexed: dict[str, dict[str, Any]] = {}
    for row in rows:
        canonical_id = row.get("canonical_id")
        if not isinstance(canonical_id, str) or not canonical_id:
            raise ValueError(f"{name}: row without canonical_id")
        if canonical_id in indexed:
            raise ValueError(f"{name}: duplicate canonical_id {canonical_id}")
        indexed[canonical_id] = row
    return indexed


def _selection_gold_coverage(
    selection_gold: list[dict[str, Any]],
    selection_inputs: list[dict[str, Any]],
    class_names: list[str],
) -> dict[str, Any]:
    gold_by_id = _index_by_canonical_id(selection_gold, "selection_gold.jsonl")
    inputs_by_id = _index_by_canonical_id(selection_inputs, "selection_inputs.jsonl")
    if set(gold_by_id) != set(inputs_by_id):
        missing_inputs = sorted(set(gold_by_id) - set(inputs_by_id))
        missing_gold = sorted(set(inputs_by_id) - set(gold_by_id))
        raise ValueError(
            "selection gold/input canonical_id mismatch: "
            f"missing_inputs={len(missing_inputs)} missing_gold={len(missing_gold)}"
        )

    box_counts: Counter[tuple[str, str, str]] = Counter()
    image_sets: defaultdict[tuple[str, str, str], set[str]] = defaultdict(set)
    label_counts: Counter[str] = Counter()
    size_counts: Counter[str] = Counter()
    all_images: set[str] = set()
    for canonical_id, gold in gold_by_id.items():
        label = gold.get("label")
        if label not in HUMAN_LABELS:
            raise ValueError(f"selection_gold.jsonl: {canonical_id} has invalid label {label!r}")
        input_row = inputs_by_id[canonical_id]
        if (gold.get("class_id"), gold.get("class_name")) != (
            input_row.get("class_id"),
            input_row.get("class_name"),
        ):
            raise ValueError(f"selection gold/input class mismatch for {canonical_id}")
        size_bin = _size_bin(input_row)
        key = (gold["class_name"], label, size_bin)
        box_counts[key] += 1
        image_sha256 = input_row["image_sha256"]
        image_sets[key].add(image_sha256)
        all_images.add(image_sha256)
        label_counts[label] += 1
        size_counts[size_bin] += 1

    cells = []
    gaps = []
    for class_name in class_names:
        for label in HUMAN_LABELS:
            for size_bin in SIZE_BINS:
                key = (class_name, label, size_bin)
                cell = {
                    "class_name": class_name,
                    "label": label,
                    "size_bin": size_bin,
                    "boxes": box_counts[key],
                    "images": len(image_sets[key]),
                }
                cells.append(cell)
                if cell["boxes"] == 0:
                    gaps.append({"class_name": class_name, "label": label, "size_bin": size_bin})

    return {
        "boxes": len(selection_gold),
        "images": len(all_images),
        "by_label": {label: label_counts[label] for label in HUMAN_LABELS},
        "by_size_bin": {size_bin: size_counts[size_bin] for size_bin in SIZE_BINS},
        "class_label_size_cells": cells,
        "gap_definition": "zero human-gold boxes in a class_name x label x original-pixel-long-edge cell",
        "gap_cells": len(gaps),
        "gaps": gaps,
    }


def _planned_split_key(row: dict[str, Any], derived_group: dict[str, Any]) -> str:
    if derived_group.get("status") == "source_group_available":
        return derived_group["split_key"]
    return row["image_sha256"]


def _candidate_item(
    row: dict[str, Any], derived_group: dict[str, Any]
) -> dict[str, Any]:
    split_group_key = _split_key(row, derived_group)
    item = {
        "canonical_id": row["canonical_id"],
        "class_id": row["class_id"],
        "class_name": row["class_name"],
        "bbox_norm": row["bbox_norm"],
        "bbox_px": row["bbox_px"],
        "image_sha256": row["image_sha256"],
        "image_object_key": row["image_object_key"],
        "image_width": row["image_width"],
        "image_height": row["image_height"],
        "split_group_key": split_group_key,
    }
    item["size_bin"] = _size_bin(item)
    item["planned_split"] = assign_split(split_group_key)
    item["source_pixel_status"] = _source_pixel_status(row)
    item["source_grouping_status"] = _provenance_status(
        row, "grouping", "status", "grouping_unavailable"
    )
    item["temporal_provenance_status"] = _temporal_status(row)
    return item


def build_queue(
    snapshot: Path,
    *,
    max_per_class: int = 20,
    max_per_image: int = 2,
    _validated_snapshot: tuple[dict[str, Any], bytes, dict[str, bytes]] | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    if max_per_class < 1 or max_per_image < 1:
        raise ValueError("queue limits must be positive")
    validated_snapshot = _validated_snapshot or _validate_snapshot(snapshot)
    manifest, _, verified = validated_snapshot
    inference = _parse_jsonl_bytes(
        verified["inference_records.jsonl"], "inference_records.jsonl"
    )
    selection_gold = _parse_jsonl_bytes(
        verified["selection_gold.jsonl"], "selection_gold.jsonl"
    )
    selection_inputs = _parse_jsonl_bytes(
        verified["selection_inputs.jsonl"], "selection_inputs.jsonl"
    )
    quarantine = _parse_jsonl_bytes(
        verified["quarantine.jsonl"], "quarantine.jsonl"
    )
    locked_inputs = _parse_jsonl_bytes(
        verified["locked_test_inputs.jsonl"], "locked_test_inputs.jsonl"
    )

    _index_by_canonical_id(quarantine, "quarantine.jsonl")
    _index_by_canonical_id(locked_inputs, "locked_test_inputs.jsonl")
    selection_shas = {row["image_sha256"] for row in selection_inputs}
    locked_shas = {row["image_sha256"] for row in locked_inputs}
    if selection_shas & locked_shas:
        raise ValueError("selection and locked image_sha256 overlap")
    selection_groups = {
        row.get("split_group_key", row["image_sha256"]) for row in selection_inputs
    }
    locked_groups = {
        row.get("split_group_key", row["image_sha256"]) for row in locked_inputs
    }
    if selection_groups & locked_groups:
        raise ValueError("selection and locked split_group_key overlap")
    for name, rows, expected_split in (
        ("selection_inputs.jsonl", selection_inputs, SELECTION),
        ("locked_test_inputs.jsonl", locked_inputs, LOCKED_TEST),
    ):
        for row in rows:
            split_key = row.get("split_group_key", row["image_sha256"])
            if not isinstance(split_key, str) or not split_key:
                raise ValueError(f"{name}: empty split_group_key")
            if assign_split(split_key) != expected_split:
                raise ValueError(f"{name}: row assigned to the wrong split")
    selection_ids = {row["canonical_id"] for row in selection_gold}
    selection_input_ids = {row["canonical_id"] for row in selection_inputs}
    quarantine_ids = {row["canonical_id"] for row in quarantine}
    locked_ids = {row["canonical_id"] for row in locked_inputs}
    excluded_ids = selection_ids | selection_input_ids | quarantine_ids | locked_ids
    existing_gold_images = selection_shas | locked_shas
    quarantine_images = {row["image_sha256"] for row in quarantine}
    excluded_images = existing_gold_images | quarantine_images
    existing_gold_groups = selection_groups | locked_groups
    derived_groups = _derive_source_groups(inference)

    class_names = sorted(
        {row["class_name"] for row in inference} | {row["class_name"] for row in selection_gold}
    )
    coverage = _selection_gold_coverage(selection_gold, selection_inputs, class_names)
    covered_class_sizes = {
        (cell["class_name"], cell["size_bin"])
        for cell in coverage["class_label_size_cells"]
        if cell["boxes"] > 0
    }

    eligible: list[dict[str, Any]] = []
    blocked_sources: list[dict[str, Any]] = []
    candidate_status_counts: Counter[str] = Counter()
    candidate_grouping_counts: Counter[str] = Counter()
    candidate_temporal_counts: Counter[str] = Counter()
    candidate_split_counts: Counter[str] = Counter()
    candidate_boxes_after_exclusion = 0
    planned_locked_images: set[str] = set()
    planned_locked_boxes = 0
    candidate_selection_boxes = 0
    existing_group_images: set[str] = set()
    existing_group_boxes = 0
    seen_selection_ids: set[str] = set()
    for row in inference:
        if row["canonical_id"] in excluded_ids or row["image_sha256"] in excluded_images:
            continue
        candidate_boxes_after_exclusion += 1
        derived_group = derived_groups[row["image_sha256"]]
        planned_split_key = _planned_split_key(row, derived_group)
        planned_split = assign_split(planned_split_key)
        candidate_split_counts[planned_split] += 1
        if planned_split == LOCKED_TEST:
            planned_locked_boxes += 1
            planned_locked_images.add(row["image_sha256"])
            continue
        candidate_selection_boxes += 1
        if planned_split_key in existing_gold_groups:
            existing_group_boxes += 1
            existing_group_images.add(row["image_sha256"])
            continue
        canonical_id = row.get("canonical_id")
        if not isinstance(canonical_id, str) or not canonical_id:
            raise ValueError("inference_records.jsonl: selection row without canonical_id")
        if canonical_id in seen_selection_ids:
            raise ValueError(
                f"inference_records.jsonl: duplicate canonical_id {canonical_id} in selection"
            )
        seen_selection_ids.add(canonical_id)
        item = _candidate_item(row, derived_group)
        candidate_status_counts[item["source_pixel_status"]] += 1
        candidate_grouping_counts[item["source_grouping_status"]] += 1
        candidate_temporal_counts[item["temporal_provenance_status"]] += 1
        blockers = []
        if manifest["schema_version"] != "prompt-gold/2":
            blockers.append("snapshot_schema_not_prompt_gold_2")
        if item["source_pixel_status"] != "clean":
            blockers.append(f"source_pixels_{item['source_pixel_status']}")
        if item["source_grouping_status"] != "source_group_available":
            blockers.append("source_grouping_unavailable")
        if item["temporal_provenance_status"] != "upstream_time_available":
            blockers.append("temporal_provenance_unavailable_or_conflicting")
        if blockers:
            blocked_sources.append(
                {
                    **item,
                    "blocked_reasons": blockers,
                    "label": None,
                }
            )
            continue
        eligible.append(item)

    buckets: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in sorted(eligible, key=_order):
        buckets[(row["class_name"], row["size_bin"])].append(row)

    selected: list[dict[str, Any]] = []
    image_counts: Counter[str] = Counter()
    selected_groups: set[str] = set(existing_gold_groups)
    newly_selected_groups: set[str] = set()
    class_counts: Counter[str] = Counter()
    bucket_keys = sorted(
        buckets,
        key=lambda key: (
            key[0],
            SIZE_BINS.index(key[1]),
        ),
    )
    for is_covered in (False, True):
        priority_keys = [
            key for key in bucket_keys if ((key[0], key[1]) in covered_class_sizes) is is_covered
        ]
        while True:
            added = False
            for class_name, size_bin in priority_keys:
                if class_counts[class_name] >= max_per_class:
                    continue
                bucket = buckets[(class_name, size_bin)]
                while bucket and (
                    image_counts[bucket[0]["image_sha256"]] >= max_per_image
                    or bucket[0]["split_group_key"] in selected_groups
                ):
                    bucket.pop(0)
                if not bucket:
                    continue
                row = bucket.pop(0)
                image_counts[row["image_sha256"]] += 1
                selected_groups.add(row["split_group_key"])
                newly_selected_groups.add(row["split_group_key"])
                selected.append({**row, "queue": "gold_expansion", "label": None})
                class_counts[class_name] += 1
                added = True
            if not added:
                break

    selected.sort(key=_order)
    blocked_sources.sort(key=_order)
    blocked_counts = Counter(row["source_pixel_status"] for row in blocked_sources)
    blocked_reason_counts = Counter(
        reason for row in blocked_sources for reason in row["blocked_reasons"]
    )
    summary = {
        "schema_version": "gold-expansion-queue/2",
        "snapshot_id": manifest["snapshot_id"],
        "snapshot_schema_version": manifest["schema_version"],
        "selection_policy": {
            "max_per_class": max_per_class,
            "max_per_image": max_per_image,
            "max_per_source_group": 1,
            "size_bins": list(SIZE_BINS),
            "size_measure": "max(bbox_px width, bbox_px height)",
            "coverage_priority": "uncovered class_name x pixel_size_bin first; pending true/false remains unknown",
            "ordering": "class, pixel_size_bin, sha256(gold-expansion-v2|canonical_id), canonical_id",
            "planned_split": "assign_split(provenance.grouping.split_key); image_sha256 only as explicit fallback",
            "locked_test": "always excluded before any row-level queue or blocker output",
            "existing_gold_images": "selection_inputs and locked_test_inputs image_sha256 excluded",
            "formal_source_gate": (
                "hash-verified prompt-gold/2; validated alias-complete clean evidence, "
                "one source-group anchor, and alias-complete timezone-aware upstream time"
            ),
            "snapshot_integrity": "manifest schema/id and every queue dependency sha256/bytes verified",
            "model_priors_in_queue": False,
        },
        "pending_label_policy": {
            "label_state": "unknown_until_human_review",
            "can_prebalance_true_false": False,
            "required_follow_up": "rebuild coverage after human labeling, then sample remaining deficits",
        },
        "selection_human_gold_coverage": coverage,
        "management_only": {
            "contains_human_label_aggregates": True,
            "must_not_be_included_in_blind_annotation_package": True,
            "blind_annotation_files": ["queue.jsonl"],
        },
        "candidate_boxes_after_existing_image_exclusion": candidate_boxes_after_exclusion,
        "candidate_selection_boxes_after_locked_exclusion": candidate_selection_boxes,
        "candidate_by_planned_split": {
            split: candidate_split_counts[split] for split in sorted(candidate_split_counts)
        },
        "candidate_by_source_pixel_status": {
            status: candidate_status_counts[status] for status in ("clean", *BLOCKED_SOURCE_STATUSES)
        },
        "candidate_by_source_grouping_status": dict(sorted(candidate_grouping_counts.items())),
        "candidate_by_temporal_provenance_status": dict(sorted(candidate_temporal_counts.items())),
        "blocked_source_boxes": len(blocked_sources),
        "blocked_source_images": len({row["image_sha256"] for row in blocked_sources}),
        "blocked_source_by_status": {
            status: blocked_counts[status] for status in BLOCKED_SOURCE_STATUSES
        },
        "blocked_source_by_reason": dict(sorted(blocked_reason_counts.items())),
        "planned_locked_boxes_excluded": planned_locked_boxes,
        "planned_locked_images_excluded": len(planned_locked_images),
        "existing_source_group_boxes_excluded": existing_group_boxes,
        "existing_source_group_images_excluded": len(existing_group_images),
        "eligible_unlabeled_boxes": len(eligible),
        "eligible_source_groups": len({row["split_group_key"] for row in eligible}),
        "eligible_by_class": dict(sorted(Counter(row["class_name"] for row in eligible).items())),
        "selected_boxes": len(selected),
        "selected_images": len({row["image_sha256"] for row in selected}),
        "selected_source_groups": len(newly_selected_groups),
        "selected_by_class": dict(sorted(Counter(row["class_name"] for row in selected).items())),
        "selected_by_size_bin": {
            size_bin: sum(row["size_bin"] == size_bin for row in selected) for size_bin in SIZE_BINS
        },
        "existing_gold_boxes": len(selection_gold) + len(locked_inputs),
        "selection_gold_boxes": len(selection_gold),
        "locked_test_input_boxes_excluded": len(locked_inputs),
        "quarantine_boxes": len(quarantine),
        "existing_gold_images_excluded": len(existing_gold_images),
        "quarantine_images_excluded": len(quarantine_images),
    }
    return selected, blocked_sources, summary


def _jsonl_bytes(rows: list[dict[str, Any]]) -> bytes:
    return "".join(
        json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n"
        for row in rows
    ).encode("utf-8")


def _json_bytes(value: dict[str, Any]) -> bytes:
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n").encode(
        "utf-8"
    )


def _readme_bytes(summary: dict[str, Any]) -> bytes:
    coverage = summary["selection_human_gold_coverage"]
    return (
        "# Human Gold Expansion Queue: Source Gated\n\n"
        "This is a preparation or blocker artifact, not a gold-label file.\n\n"
        "## Result\n\n"
        f"- Source snapshot: `{summary['snapshot_id']}` "
        f"(`{summary['snapshot_schema_version']}`)\n"
        f"- Formal blind-label queue: **{summary['selected_boxes']} boxes**\n"
        f"- Public source-recovery blockers: **{summary['blocked_source_boxes']} boxes / "
        f"{summary['blocked_source_images']} images**\n"
        f"- Planned locked candidates excluded before row output: "
        f"**{summary['planned_locked_boxes_excluded']} boxes / "
        f"{summary['planned_locked_images_excluded']} images**\n"
        f"- Existing selection human gold: **{coverage['boxes']} boxes / "
        f"{coverage['images']} images**\n\n"
        "## Audience Boundary\n\n"
        "- `queue.jsonl` is the only blind-annotator package. Its labels remain null.\n"
        "- `blocked_sources.jsonl` is for source recovery and must not be labeled.\n"
        "- `summary.json` is management-only because it contains aggregate human-label coverage.\n"
        "- Planned locked rows never appear in any row-level output.\n\n"
        "A formal queue row requires a content-addressed prompt-gold/2 snapshot, "
        "alias-complete clean identity-pixel evidence, one deterministic non-event source "
        "anchor, and alias-complete timezone-aware upstream time. Until those gates pass, "
        "do not train again or claim an accuracy increase.\n"
    ).encode("utf-8")


def _artifact_files(
    snapshot: Path,
    rows: list[dict[str, Any]],
    blocked_sources: list[dict[str, Any]],
    summary: dict[str, Any],
    *,
    snapshot_manifest_bytes: bytes | None = None,
) -> dict[str, bytes]:
    files = {
        "README.md": _readme_bytes(summary),
        "queue.jsonl": _jsonl_bytes(rows),
        "blocked_sources.jsonl": _jsonl_bytes(blocked_sources),
        "summary.json": _json_bytes(summary),
    }
    if snapshot_manifest_bytes is None:
        snapshot_manifest_bytes = (snapshot / "snapshot_manifest.json").read_bytes()
    snapshot_manifest = json.loads(snapshot_manifest_bytes)
    source_paths = {*_QUEUE_SOURCE_FILES, "SHA256SUMS"}
    if snapshot_manifest.get("schema_version") == "prompt-gold/2":
        source_paths.add("source_readiness.json")
    source_entries = [
        entry
        for entry in snapshot_manifest["files"]
        if entry.get("path") in source_paths
    ]
    artifact_manifest = {
        "schema_version": "gold-expansion-artifact/1",
        "generator": "tools.prompt_eval_data.build_gold_expansion_queue",
        "source_snapshot": {
            "snapshot_id": snapshot_manifest["snapshot_id"],
            "schema_version": snapshot_manifest["schema_version"],
            "manifest_sha256": hashlib.sha256(snapshot_manifest_bytes).hexdigest(),
            "members": sorted(source_entries, key=lambda entry: entry["path"]),
        },
        "policy": summary["selection_policy"],
        "files": [
            {
                "path": name,
                "sha256": hashlib.sha256(payload).hexdigest(),
                "bytes": len(payload),
                "audience": (
                    "blind_annotator"
                    if name == "queue.jsonl"
                    else "management_only"
                    if name == "summary.json"
                    else "all_operators"
                    if name == "README.md"
                    else "source_recovery"
                ),
            }
            for name, payload in sorted(files.items())
        ],
    }
    files["artifact_manifest.json"] = _json_bytes(artifact_manifest)
    return files


def _publish_atomic(output_dir: Path, files: dict[str, bytes]) -> None:
    if output_dir.exists():
        raise FileExistsError(f"refusing to replace existing output: {output_dir}")
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(prefix=f".{output_dir.name}.", dir=str(output_dir.parent))
    )
    try:
        for name, payload in sorted(files.items()):
            (temporary / name).write_bytes(payload)
        temporary.rename(output_dir)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-per-class", type=int, default=20)
    parser.add_argument("--max-per-image", type=int, default=2)
    args = parser.parse_args()
    if args.max_per_class < 1 or args.max_per_image < 1:
        raise ValueError("queue limits must be positive")
    if not args.snapshot.is_dir():
        raise FileNotFoundError(args.snapshot)
    if args.output_dir.exists():
        raise FileExistsError(f"refusing to replace existing output: {args.output_dir}")
    validated_snapshot = _validate_snapshot(args.snapshot)
    rows, blocked_sources, summary = build_queue(
        args.snapshot,
        max_per_class=args.max_per_class,
        max_per_image=args.max_per_image,
        _validated_snapshot=validated_snapshot,
    )
    _publish_atomic(
        args.output_dir,
        _artifact_files(
            args.snapshot,
            rows,
            blocked_sources,
            summary,
            snapshot_manifest_bytes=validated_snapshot[1],
        ),
    )
    print(json.dumps(summary, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
