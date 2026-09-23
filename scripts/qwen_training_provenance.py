"""Strict provenance checks for formal Qwen candidate-verifier training."""

from __future__ import annotations

import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence


_HEX_SHA256 = set("0123456789abcdef")
_VERDICT_TO_LABEL = {"CONFIRMED": "true", "REJECTED": "false"}


def _resolve_file(
    value: Any,
    *,
    roots: Sequence[Path],
    dataset_dir: Path,
    purpose: str,
) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise ValueError("missing declared " + purpose + " path")
    declared = Path(value)
    candidates: list[Path] = []
    if declared.is_absolute():
        candidates.append(declared)
    else:
        candidates.extend(root / declared for root in roots)
        if purpose == "source image":
            candidates.extend(
                [
                    dataset_dir / "source_images" / declared.name,
                    dataset_dir / declared.name,
                ]
            )
    seen: set[str] = set()
    for candidate in candidates:
        resolved = candidate.expanduser().resolve()
        if str(resolved) in seen:
            continue
        seen.add(str(resolved))
        if resolved.is_file():
            return resolved
    raise ValueError("declared " + purpose + " file is unavailable: " + value)


def _read_source_annotations(path: Path) -> dict[tuple[str, str], dict[str, Any]]:
    by_candidate: dict[tuple[str, str], dict[str, Any]] = {}
    with path.open(encoding="utf-8") as stream:
        for line_no, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    "invalid source annotation JSONL at "
                    + str(path)
                    + ":"
                    + str(line_no)
                ) from exc
            key = (str(row.get("sample_id", "")), str(row.get("target_index", "")))
            if not all(key) or key in by_candidate:
                raise ValueError("duplicate or incomplete source annotation key: " + repr(key))
            by_candidate[key] = row
    return by_candidate


def _assistant_verdict(row: Mapping[str, Any]) -> str:
    turns = row.get("conversations")
    if not isinstance(turns, list):
        raise ValueError("training row has no conversations")
    for turn in reversed(turns):
        if not isinstance(turn, Mapping) or turn.get("from") not in {"gpt", "assistant"}:
            continue
        value = turn.get("value")
        try:
            parsed = json.loads(str(value))
        except (TypeError, json.JSONDecodeError) as exc:
            raise ValueError("assistant target is not a JSON verdict") from exc
        decision = parsed.get("decision") if isinstance(parsed, Mapping) else None
        if decision not in _VERDICT_TO_LABEL:
            raise ValueError("assistant target is not CONFIRMED or REJECTED")
        return str(decision)
    raise ValueError("training row has no assistant target")


def _same_bbox(left: Any, right: Any) -> bool:
    if not isinstance(left, (list, tuple)) or not isinstance(right, (list, tuple)):
        return False
    if len(left) != 4 or len(right) != 4:
        return False
    try:
        return all(
            math.isclose(float(a), float(b), rel_tol=0.0, abs_tol=1e-6)
            for a, b in zip(left, right)
        )
    except (TypeError, ValueError):
        return False


def validate_human_gold_dataset(
    rows: Sequence[Mapping[str, Any]],
    *,
    dataset_path: Path,
    provenance_root: Path,
    allow_duplicates: bool = False,
    allow_conflicts: bool = False,
) -> dict[str, Any]:
    """Fail closed unless every target traces to a binary human annotation."""
    if not rows:
        raise ValueError("formal training dataset is empty")
    dataset_path = dataset_path.resolve()
    dataset_dir = dataset_path.parent
    roots = (provenance_root.resolve(), dataset_dir, dataset_dir.parent)
    source_cache: dict[Path, dict[tuple[str, str], dict[str, Any]]] = {}
    image_hash_cache: dict[Path, str] = {}
    candidate_groups: dict[
        tuple[str, str, tuple[float, ...]], list[dict[str, Any]]
    ] = defaultdict(list)
    image_shas: set[str] = set()
    source_group_ids: set[str] = set()
    class_counts: Counter[str] = Counter()
    label_counts: Counter[str] = Counter()

    for row_number, row in enumerate(rows, start=1):
        meta = row.get("meta")
        if not isinstance(meta, Mapping):
            raise ValueError("row " + str(row_number) + " has no metadata")
        if meta.get("label_source") != "human_gold" or meta.get("label_field") != "human_label":
            raise ValueError("row " + str(row_number) + " is not declared human gold")

        required = (
            "labeled_by",
            "image_id",
            "candidate_index",
            "class_name",
            "bbox",
            "image_sha256",
            "image_object_key",
            "source_file",
            "source_image",
        )
        missing = [key for key in required if meta.get(key) in (None, "")]
        if missing:
            raise ValueError(
                "row "
                + str(row_number)
                + " lacks required provenance fields: "
                + ", ".join(missing)
            )

        digest = str(meta["image_sha256"]).lower()
        if len(digest) != 64 or any(char not in _HEX_SHA256 for char in digest):
            raise ValueError("row " + str(row_number) + " has an invalid image SHA-256")
        source_image = _resolve_file(
            meta["source_image"],
            roots=roots,
            dataset_dir=dataset_dir,
            purpose="source image",
        )
        if source_image not in image_hash_cache:
            image_hash_cache[source_image] = hashlib.sha256(
                source_image.read_bytes()
            ).hexdigest()
        actual_digest = image_hash_cache[source_image]
        if actual_digest != digest:
            raise ValueError(
                "row "
                + str(row_number)
                + " source image SHA-256 mismatch: "
                + str(meta["image_id"])
            )

        source_file = _resolve_file(
            meta["source_file"],
            roots=roots,
            dataset_dir=dataset_dir,
            purpose="source annotation",
        )
        if source_file not in source_cache:
            source_cache[source_file] = _read_source_annotations(source_file)
        source_key = (str(meta["image_id"]), str(meta["candidate_index"]))
        source = source_cache[source_file].get(source_key)
        if source is None:
            raise ValueError(
                "row "
                + str(row_number)
                + " does not resolve to a source annotation: "
                + repr(source_key)
            )

        human_label = source.get("human_label")
        if human_label not in {"true", "false"}:
            raise ValueError("source annotation is not a binary human label")
        if not source.get("labeled_by") or not source.get("labeled_at"):
            raise ValueError("source annotation lacks a human actor or timestamp")
        source_group_id = source.get("source_event_id")
        if source_group_id in (None, ""):
            raise ValueError("source annotation lacks a task/event group identifier")
        if str(source.get("labeled_by")) != str(meta["labeled_by"]):
            raise ValueError("human annotator differs from the source annotation")
        if str(source.get("class_name")) != str(meta["class_name"]):
            raise ValueError("candidate class differs from the source annotation")
        if str(source.get("image_object_key")) != str(meta["image_object_key"]):
            raise ValueError("source image object key differs from the annotation")
        if not _same_bbox(source.get("bbox"), meta.get("bbox")):
            raise ValueError("candidate box differs from the source annotation")
        if _assistant_verdict(row) != (
            "CONFIRMED" if human_label == "true" else "REJECTED"
        ):
            raise ValueError("training target differs from the human annotation")

        image_paths = row.get("image")
        if not isinstance(image_paths, list) or len(image_paths) != 2:
            raise ValueError("row " + str(row_number) + " must contain context and ROI views")
        for image_path in image_paths:
            _resolve_file(
                image_path,
                roots=(dataset_dir,),
                dataset_dir=dataset_dir,
                purpose="training view",
            )

        bbox_key = tuple(round(float(value), 6) for value in meta["bbox"])
        candidate_key = (digest, str(meta["class_name"]), bbox_key)
        candidate_groups[candidate_key].append(
            {
                "row_id": str(row.get("id", "")),
                "label": str(human_label),
                "image_id": str(meta["image_id"]),
                "candidate_index": str(meta["candidate_index"]),
            }
        )
        image_shas.add(digest)
        source_group_ids.add(str(source_group_id))
        class_counts[str(meta["class_name"])] += 1
        label_counts[str(human_label)] += 1

    duplicate_groups = [
        (key, members)
        for key, members in candidate_groups.items()
        if len(members) > 1
    ]
    conflict_groups = [
        (key, members)
        for key, members in duplicate_groups
        if len({member["label"] for member in members}) > 1
    ]
    if duplicate_groups and not allow_duplicates:
        raise ValueError(
            "formal training data contains "
            + str(len(duplicate_groups))
            + " duplicate image/class/box candidate group(s); deduplicate first"
        )
    if conflict_groups and not allow_conflicts:
        raise ValueError(
            "formal training data contains "
            + str(len(conflict_groups))
            + " conflicting human-label group(s); exclude them for adjudication"
        )

    return {
        "rows": len(rows),
        "human_gold_rows": len(rows),
        "unique_image_sha256": len(image_shas),
        "class_counts": dict(sorted(class_counts.items())),
        "label_counts": dict(sorted(label_counts.items())),
        "source_annotation_files": len(source_cache),
        "source_annotation_sha256": [
            {
                "path": str(path),
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }
            for path in sorted(source_cache)
        ],
        "image_sha256s": image_shas,
        "source_group_ids": source_group_ids,
        "duplicate_groups": len(duplicate_groups),
        "duplicate_extra_rows": sum(len(members) - 1 for _, members in duplicate_groups),
        "conflict_groups": len(conflict_groups),
    }


def validate_disjoint_image_splits(
    train_summary: Mapping[str, Any],
    val_summary: Mapping[str, Any],
) -> None:
    overlap = set(train_summary["image_sha256s"]) & set(val_summary["image_sha256s"])
    if overlap:
        raise ValueError(
            "train and validation sets share "
            + str(len(overlap))
            + " source image SHA-256 group(s)"
        )
    train_groups = set(train_summary.get("source_group_ids", ()))
    val_groups = set(val_summary.get("source_group_ids", ()))
    if not train_groups or not val_groups:
        raise ValueError("train and validation require source event/task group identifiers")
    group_overlap = train_groups & val_groups
    if group_overlap:
        raise ValueError(
            "train and validation share "
            + str(len(group_overlap))
            + " source event/task group(s)"
        )


def deduplicate_human_gold_rows(
    rows: Sequence[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    """Collapse consistent duplicate targets and exclude every conflicting copy."""
    groups: dict[tuple[str, str, tuple[float, ...]], list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        meta = row.get("meta")
        if not isinstance(meta, Mapping):
            raise ValueError("cannot deduplicate a row without metadata")
        bbox = meta.get("bbox")
        if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
            raise ValueError("cannot deduplicate a row without a four-coordinate box")
        key = (
            str(meta.get("image_sha256", "")),
            str(meta.get("class_name", "")),
            tuple(round(float(value), 6) for value in bbox),
        )
        groups[key].append(row)

    canonical: list[dict[str, Any]] = []
    duplicate_manifest: list[dict[str, Any]] = []
    conflict_rows: list[dict[str, Any]] = []
    for (digest, class_name, bbox), members in sorted(
        groups.items(), key=lambda item: item[0]
    ):
        labels = [_VERDICT_TO_LABEL[_assistant_verdict(row)] for row in members]
        label_set = set(labels)
        if len(label_set) > 1:
            conflict_rows.extend(dict(row) for row in members)
            duplicate_manifest.append(
                {
                    "image_sha256": digest,
                    "class_name": class_name,
                    "bbox": list(bbox),
                    "status": "conflicting_human_labels",
                    "labels": sorted(label_set),
                    "row_ids": [str(row.get("id", "")) for row in members],
                    "source_candidates": [
                        {
                            "image_id": str(row.get("meta", {}).get("image_id", "")),
                            "candidate_index": str(
                                row.get("meta", {}).get("candidate_index", "")
                            ),
                        }
                        for row in members
                    ],
                }
            )
            continue

        canonical.append(dict(members[0]))
        if len(members) > 1:
            duplicate_manifest.append(
                {
                    "image_sha256": digest,
                    "class_name": class_name,
                    "bbox": list(bbox),
                    "status": "consistent_duplicate_collapsed",
                    "label": next(iter(label_set)),
                    "canonical_row_id": str(members[0].get("id", "")),
                    "duplicate_row_ids": [
                        str(row.get("id", "")) for row in members[1:]
                    ],
                }
            )
    return canonical, duplicate_manifest, conflict_rows


def validate_supervision_lengths(
    *,
    sequence_length: int,
    prompt_length: int,
    max_length: int,
    row_id: str,
) -> int:
    """Refuse silent truncation; return the number of supervised answer tokens."""
    if sequence_length > max_length:
        raise ValueError(
            "row "
            + row_id
            + " exceeds max_length; refusing to truncate the multimodal prompt/answer"
        )
    answer_length = sequence_length - prompt_length
    if prompt_length < 0 or answer_length <= 0:
        raise ValueError("row " + row_id + " has no supervised assistant answer tokens")
    return answer_length
