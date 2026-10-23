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
_CLEAN_SOURCE_VIEW_SCHEMA = "annotated_locator_plus_clean_original_v1"
_CLEAN_SOURCE_VIEW_ROLES = (
    "annotated_context",
    "annotated_roi",
    "clean_context",
    "clean_roi",
)
_CLEAN_SOURCE_VIEW_ROLES_2VIEW = (
    "annotated_context",
    "clean_roi",
)
_ALLOWED_CLEAN_SOURCE_ROLE_SETS = (
    _CLEAN_SOURCE_VIEW_ROLES,
    _CLEAN_SOURCE_VIEW_ROLES_2VIEW,
)


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


def _read_candidate_rows(path: Path, *, purpose: str) -> dict[str, dict[str, Any]]:
    by_candidate: dict[str, dict[str, Any]] = {}
    with path.open(encoding="utf-8") as stream:
        for line_no, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid {purpose} JSONL at {path}:{line_no}") from exc
            candidate_id = str(row.get("candidate_id") or "")
            if not candidate_id or candidate_id in by_candidate:
                raise ValueError(f"duplicate or incomplete {purpose} candidate_id: {candidate_id!r}")
            by_candidate[candidate_id] = row
    return by_candidate


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _valid_sha256(value: Any) -> bool:
    digest = str(value or "").lower()
    return len(digest) == 64 and not any(char not in _HEX_SHA256 for char in digest)


def _human_prompt(row: Mapping[str, Any]) -> str:
    turns = row.get("conversations")
    if not isinstance(turns, list):
        raise ValueError("training row has no conversations")
    for turn in turns:
        if isinstance(turn, Mapping) and turn.get("from") in {"human", "user"}:
            value = turn.get("value")
            if not isinstance(value, str) or not value:
                raise ValueError("training row has no human prompt text")
            return value
    raise ValueError("training row has no human prompt")


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
    candidate_pair_cache: dict[Path, dict[str, dict[str, Any]]] = {}
    gold_cache: dict[Path, dict[str, dict[str, Any]]] = {}
    provenance_files: set[Path] = set()
    image_hash_cache: dict[Path, str] = {}
    candidate_groups: dict[
        tuple[str, str, tuple[float, ...]], list[dict[str, Any]]
    ] = defaultdict(list)
    image_shas: set[str] = set()
    view_shas: set[str] = set()
    annotated_frame_shas: set[str] = set()
    clean_frame_shas: set[str] = set()
    source_group_ids: set[str] = set()
    source_task_ids: set[str] = set()
    source_event_ids: set[str] = set()
    view_schemas: set[str] = set()
    class_counts: Counter[str] = Counter()
    label_counts: Counter[str] = Counter()

    for row_number, row in enumerate(rows, start=1):
        meta = row.get("meta")
        if not isinstance(meta, Mapping):
            raise ValueError("row " + str(row_number) + " has no metadata")
        if meta.get("label_source") != "human_gold" or meta.get("label_field") != "human_label":
            raise ValueError("row " + str(row_number) + " is not declared human gold")

        if meta.get("view_schema") == _CLEAN_SOURCE_VIEW_SCHEMA:
            required = (
                "candidate_id",
                "image_id",
                "candidate_index",
                "class_name",
                "bbox",
                "image_sha256",
                "image_object_key",
                "source_image",
                "candidate_pairs_file",
                "candidate_pairs_sha256",
                "gold_file",
                "gold_sha256",
                "view_generation",
                "annotated_frame",
                "annotated_frame_sha256",
                "annotated_image_size",
                "normalized_review_bbox",
                "clean_frame",
                "clean_frame_sha256",
                "clean_frame_size",
                "normalized_clean_bbox",
                "prompt_template_sha256",
                "prompt_sha256",
                "source_task_id",
                "event_id",
            )
            missing = [key for key in required if meta.get(key) in (None, "")]
            if missing:
                raise ValueError(
                    "row "
                    + str(row_number)
                    + " lacks required clean-source provenance fields: "
                    + ", ".join(missing)
                )
            candidate_id = str(meta["candidate_id"])
            if str(meta["image_id"]) != candidate_id:
                raise ValueError("clean-source candidate_id and image_id differ")

            pairs_file = _resolve_file(
                meta["candidate_pairs_file"],
                roots=roots,
                dataset_dir=dataset_dir,
                purpose="candidate pairs",
            )
            gold_file = _resolve_file(
                meta["gold_file"],
                roots=roots,
                dataset_dir=dataset_dir,
                purpose="gold labels",
            )
            provenance_files.update((pairs_file, gold_file))
            for source_path, sha_field, purpose in (
                (pairs_file, "candidate_pairs_sha256", "candidate pairs"),
                (gold_file, "gold_sha256", "gold labels"),
            ):
                declared_sha = str(meta[sha_field]).lower()
                if not _valid_sha256(declared_sha):
                    raise ValueError(f"row {row_number} has an invalid {purpose} SHA-256")
                if _file_sha256(source_path) != declared_sha:
                    raise ValueError(f"row {row_number} {purpose} SHA-256 mismatch")
            if pairs_file not in candidate_pair_cache:
                candidate_pair_cache[pairs_file] = _read_candidate_rows(
                    pairs_file, purpose="candidate pairs"
                )
            if gold_file not in gold_cache:
                gold_cache[gold_file] = _read_candidate_rows(
                    gold_file, purpose="gold labels"
                )
            pair = candidate_pair_cache[pairs_file].get(candidate_id)
            gold = gold_cache[gold_file].get(candidate_id)
            if pair is None or gold is None:
                raise ValueError(
                    f"row {row_number} does not resolve to candidate/gold lineage: {candidate_id}"
                )
            for field in ("human_label", "class_name", "partition", "event_id"):
                if str(pair.get(field) or "") != str(gold.get(field) or ""):
                    raise ValueError(
                        f"candidate {candidate_id} differs between pair and gold for {field}"
                    )
            human_label = gold.get("human_label")
            if human_label not in {"true", "false"}:
                raise ValueError("gold source is not a binary human label")
            if not pair.get("has_clean_source") or pair.get("clean_match_status") not in {
                "strong",
                "usable",
            }:
                raise ValueError("clean-source training row is not strong or usable")
            expected_meta = {
                "class_name": pair.get("class_name"),
                "event_id": pair.get("event_id"),
                "source_task_id": pair.get("source_task_id"),
                "partition": pair.get("partition"),
                "human_label": human_label,
            }
            for field, expected in expected_meta.items():
                if str(meta.get(field) or "") != str(expected or ""):
                    raise ValueError(
                        f"clean-source metadata differs from provenance for {field}"
                    )
            if not _same_bbox(pair.get("bbox_px"), meta.get("bbox")):
                raise ValueError("candidate box differs from the candidate-pair source")
            if _assistant_verdict(row) != (
                "CONFIRMED" if human_label == "true" else "REJECTED"
            ):
                raise ValueError("training target differs from the human annotation")

            image_paths = row.get("image")
            roles = meta.get("view_roles")
            declared_view_shas = meta.get("view_sha256")
            role_tuple = tuple(roles) if isinstance(roles, (list, tuple)) else ()
            if role_tuple not in _ALLOWED_CLEAN_SOURCE_ROLE_SETS:
                raise ValueError("clean-source view_roles are missing or out of order")
            if not isinstance(image_paths, list) or len(image_paths) != len(role_tuple):
                raise ValueError(
                    "row " + str(row_number) + " must contain ordered training views matching view_roles"
                )
            if not isinstance(declared_view_shas, Mapping):
                raise ValueError("clean-source view_sha256 must map every view role")
            if set(declared_view_shas) != set(role_tuple):
                raise ValueError("clean-source view_sha256 roles are incomplete")
            actual_view_shas: dict[str, str] = {}
            for role, image_path in zip(role_tuple, image_paths):
                view = _resolve_file(
                    image_path,
                    roots=(dataset_dir,),
                    dataset_dir=dataset_dir,
                    purpose="training view",
                )
                if view not in image_hash_cache:
                    image_hash_cache[view] = _file_sha256(view)
                actual = image_hash_cache[view]
                declared = str(declared_view_shas[role]).lower()
                if not _valid_sha256(declared) or actual != declared:
                    raise ValueError(
                        f"row {row_number} {role} training-view SHA-256 mismatch"
                    )
                actual_view_shas[role] = actual
            if str(meta["source_image"]) != str(image_paths[0]):
                raise ValueError("source_image must be the annotated context view")
            if str(meta["image_sha256"]).lower() != actual_view_shas["annotated_context"]:
                raise ValueError("source image SHA-256 differs from annotated context")
            prompt = _human_prompt(row)
            prompt_sha256 = str(meta["prompt_sha256"]).lower()
            prompt_template_sha256 = str(meta["prompt_template_sha256"]).lower()
            if not _valid_sha256(prompt_sha256) or hashlib.sha256(
                prompt.encode("utf-8")
            ).hexdigest() != prompt_sha256:
                raise ValueError(f"row {row_number} prompt SHA-256 mismatch")
            if not _valid_sha256(prompt_template_sha256):
                raise ValueError(f"row {row_number} has an invalid prompt-template SHA-256")
            dataset_manifest_path = dataset_dir / "dataset_manifest.json"
            if dataset_manifest_path.is_file():
                dataset_manifest = json.loads(
                    dataset_manifest_path.read_text(encoding="utf-8")
                )
                contract = dataset_manifest.get("deployment_contract")
                if not isinstance(contract, Mapping):
                    raise ValueError("dataset manifest lacks deployment_contract")
                template = contract.get("prompt_template")
                if not isinstance(template, str) or hashlib.sha256(
                    template.encode("utf-8")
                ).hexdigest() != prompt_template_sha256:
                    raise ValueError("prompt-template SHA-256 differs from dataset contract")
                if contract.get("view_schema") != _CLEAN_SOURCE_VIEW_SCHEMA or tuple(
                    contract.get("view_roles") or ()
                ) != role_tuple:
                    raise ValueError("view contract differs from clean-source schema")
                if contract.get("view_generation") != "service_crop_v1_lossless_png":
                    raise ValueError("view-generation contract differs from clean-source schema")
                if meta.get("view_generation") != contract.get("view_generation"):
                    raise ValueError("row view generation differs from dataset contract")
                if not isinstance(contract.get("crop_spec"), Mapping):
                    raise ValueError("dataset contract lacks crop_spec")

            annotated_frame = _resolve_file(
                meta["annotated_frame"],
                roots=(dataset_dir,),
                dataset_dir=dataset_dir,
                purpose="annotated frame",
            )
            if annotated_frame not in image_hash_cache:
                image_hash_cache[annotated_frame] = _file_sha256(annotated_frame)
            annotated_frame_digest = str(meta["annotated_frame_sha256"]).lower()
            if (
                not _valid_sha256(annotated_frame_digest)
                or image_hash_cache[annotated_frame] != annotated_frame_digest
            ):
                raise ValueError(f"row {row_number} annotated-frame SHA-256 mismatch")

            clean_frame = _resolve_file(
                meta["clean_frame"],
                roots=(dataset_dir,),
                dataset_dir=dataset_dir,
                purpose="clean frame",
            )
            if clean_frame not in image_hash_cache:
                image_hash_cache[clean_frame] = _file_sha256(clean_frame)
            clean_frame_digest = str(meta["clean_frame_sha256"]).lower()
            if not _valid_sha256(clean_frame_digest) or image_hash_cache[clean_frame] != clean_frame_digest:
                raise ValueError(f"row {row_number} clean-frame SHA-256 mismatch")

            digest = actual_view_shas["annotated_context"]
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
            view_shas.update(actual_view_shas.values())
            annotated_frame_shas.add(annotated_frame_digest)
            clean_frame_shas.add(clean_frame_digest)
            source_group_ids.add(str(meta["event_id"]))
            source_event_ids.add(str(meta["event_id"]))
            source_task_ids.add(str(meta["source_task_id"]))
            view_schemas.add(_CLEAN_SOURCE_VIEW_SCHEMA)
            class_counts[str(meta["class_name"])] += 1
            label_counts[str(human_label)] += 1
            continue

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
        source_event_ids.add(str(source_group_id))
        view_schemas.add("context_plus_roi_v1")
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
        "source_annotation_files": len(set(source_cache) | provenance_files),
        "source_annotation_sha256": [
            {
                "path": str(path),
                "sha256": _file_sha256(path),
            }
            for path in sorted(set(source_cache) | provenance_files)
        ],
        "view_schemas": sorted(view_schemas),
        "image_sha256s": image_shas,
        "view_sha256s": view_shas,
        "annotated_frame_sha256s": annotated_frame_shas,
        "clean_frame_sha256s": clean_frame_shas,
        "source_group_ids": source_group_ids,
        "source_event_ids": source_event_ids,
        "source_task_ids": source_task_ids,
        "duplicate_groups": len(duplicate_groups),
        "duplicate_extra_rows": sum(len(members) - 1 for _, members in duplicate_groups),
        "conflict_groups": len(conflict_groups),
    }


def validate_disjoint_image_splits(
    train_summary: Mapping[str, Any],
    val_summary: Mapping[str, Any],
) -> None:
    checks = (
        ("image_sha256s", "source image SHA-256 group"),
        ("view_sha256s", "training-view SHA-256"),
        ("annotated_frame_sha256s", "annotated-frame SHA-256"),
        ("clean_frame_sha256s", "clean-frame SHA-256"),
        ("source_task_ids", "source task"),
        ("source_event_ids", "source event"),
    )
    for field, label in checks:
        train_values = set(train_summary.get(field, ()))
        val_values = set(val_summary.get(field, ()))
        overlap = train_values & val_values
        if overlap:
            raise ValueError(
                "train and validation sets share "
                + str(len(overlap))
                + " "
                + label
                + "(s)"
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

    clean_schema = _CLEAN_SOURCE_VIEW_SCHEMA
    if clean_schema in set(train_summary.get("view_schemas", ())) | set(
        val_summary.get("view_schemas", ())
    ):
        for field, label in (
            ("annotated_frame_sha256s", "annotated-frame SHA-256"),
            ("clean_frame_sha256s", "clean-frame SHA-256"),
            ("source_task_ids", "source task"),
            ("source_event_ids", "source event"),
        ):
            if not train_summary.get(field) or not val_summary.get(field):
                raise ValueError(
                    "clean-source train and validation require " + label + " lineage"
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


_QUANTIZATION_CONTRACT = {
    "load_in_4bit": True,
    "bnb_4bit_quant_type": "nf4",
    "bnb_4bit_use_double_quant": True,
    "bnb_4bit_compute_dtype": "bfloat16",
}
_STRICT_OUTPUT_CONTRACT = {
    "json_object_keys": ["decision"],
    "allowed_decisions": ["CONFIRMED", "REJECTED"],
    "invalid_output": "uncertain",
}

def _training_stratum(row: dict[str, Any]) -> str:
    meta = row.get("meta")
    if not isinstance(meta, dict) or not meta.get("class_name"):
        raise ValueError("balanced sampling requires class_name metadata")
    label = meta.get("human_label")
    if label not in {"true", "false"}:
        conversations = row.get("conversations")
        if not isinstance(conversations, list) or not conversations:
            raise ValueError("balanced sampling requires a binary human target")
        try:
            target = json.loads(str(conversations[-1]["value"]))
        except (KeyError, TypeError, json.JSONDecodeError) as exc:
            raise ValueError("balanced sampling requires a JSON assistant target") from exc
        label = {"CONFIRMED": "true", "REJECTED": "false"}.get(target.get("decision"))
    if label not in {"true", "false"}:
        raise ValueError("balanced sampling requires a binary human target")
    return str(meta["class_name"]) + "|" + str(label)


def _balanced_sampling_manifest(
    rows: list[dict[str, Any]],
    seed: int,
    *,
    negative_oversample_ratio: float = 1.0,
) -> tuple[list[float], dict[str, Any]]:
    if not negative_oversample_ratio > 0:
        raise ValueError("negative_oversample_ratio must be positive")
    strata = [_training_stratum(row) for row in rows]
    counts = Counter(strata)
    raw = {
        stratum: (len(rows) / (len(counts) * count))
        * (negative_oversample_ratio if stratum.endswith("|false") else 1.0)
        for stratum, count in counts.items()
    }
    scale = len(rows) / sum(raw[stratum] for stratum in strata) if rows else 1.0
    per_stratum = {stratum: raw[stratum] * scale for stratum in raw}
    weights = [per_stratum[stratum] for stratum in strata]
    return weights, {
        "strategy": "deterministic_weighted_replacement_by_class_and_human_label",
        "seed": seed,
        "replacement": True,
        "negative_oversample_ratio": negative_oversample_ratio,
        "samples_per_epoch": len(rows),
        "validation_sampling": "natural_original_partition",
        "stratum_counts": dict(sorted(counts.items())),
        "per_row_weight_by_stratum": dict(sorted(per_stratum.items())),
    }


def _capped_sampling_manifest(
    rows: list[dict[str, Any]],
    seed: int,
    cap: float,
    *,
    negative_oversample_ratio: float = 1.0,
) -> tuple[list[float], dict[str, Any]]:
    """等权抽样的折中：每个“类别|标签”分组的放大倍数不超过 cap，再整体归一化到平均权重 1。

    cap 越小越接近自然比例，越大越接近完全等权（balanced_stratum）。
    """
    if not cap > 0:
        raise ValueError("stratum weight cap must be positive")
    if not negative_oversample_ratio > 0:
        raise ValueError("negative_oversample_ratio must be positive")
    strata = [_training_stratum(row) for row in rows]
    counts = Counter(strata)
    raw = {
        stratum: min(len(rows) / (len(counts) * count), cap)
        * (negative_oversample_ratio if stratum.endswith("|false") else 1.0)
        for stratum, count in counts.items()
    }
    scale = len(rows) / sum(raw[stratum] for stratum in strata)
    per_stratum = {stratum: raw[stratum] * scale for stratum in raw}
    weights = [per_stratum[stratum] for stratum in strata]
    return weights, {
        "strategy": "deterministic_weighted_replacement_by_class_and_human_label_capped",
        "seed": seed,
        "replacement": True,
        "stratum_weight_cap": cap,
        "negative_oversample_ratio": negative_oversample_ratio,
        "samples_per_epoch": len(rows),
        "validation_sampling": "natural_original_partition",
        "stratum_counts": dict(sorted(counts.items())),
        "per_row_weight_by_stratum": dict(sorted(per_stratum.items())),
    }


def _load_dataset_contract(
    train_path: Path,
    val_path: Path,
    train_rows: list[dict[str, Any]],
    val_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    clean_schema = "annotated_locator_plus_clean_original_v1"
    schemas = {
        str(row.get("meta", {}).get("view_schema") or "")
        for row in train_rows + val_rows
    }
    if clean_schema not in schemas:
        schemas.discard("")
        return {
            "view_schema": sorted(schemas) or ["legacy_context_plus_roi"],
            "strict_output": dict(_STRICT_OUTPUT_CONTRACT),
            "processor": None,
            "quantization": dict(_QUANTIZATION_CONTRACT),
            "contract_source": "derived_from_legacy_dataset",
        }
    if schemas != {clean_schema}:
        raise ValueError(
            "clean-source train and validation must contain only the four-view schema"
        )
    if train_path.resolve().parent != val_path.resolve().parent:
        raise ValueError("clean-source train and validation files must share a dataset manifest")
    manifest_path = train_path.resolve().parent / "dataset_manifest.json"
    if not manifest_path.is_file():
        raise ValueError("clean-source dataset is missing dataset_manifest.json")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    contract = manifest.get("deployment_contract")
    if not isinstance(contract, dict):
        raise ValueError("clean-source dataset manifest lacks deployment_contract")
    template = contract.get("prompt_template")
    template_sha = contract.get("prompt_template_sha256")
    if not isinstance(template, str) or hashlib.sha256(template.encode("utf-8")).hexdigest() != template_sha:
        raise ValueError("dataset prompt template SHA-256 is invalid")
    if contract.get("view_schema") != clean_schema:
        raise ValueError("dataset view schema differs from clean-source training schema")
    if contract.get("strict_output") != _STRICT_OUTPUT_CONTRACT:
        raise ValueError("dataset strict-output contract is unsupported")
    if contract.get("quantization") != _QUANTIZATION_CONTRACT:
        raise ValueError("dataset quantization contract is unsupported")
    for split, path, rows in (
        ("train", train_path, train_rows),
        ("validation", val_path, val_rows),
    ):
        declared = manifest.get("outputs", {}).get(split, {})
        if declared.get("sha256") != _file_sha256(path) or declared.get("rows") != len(rows):
            raise ValueError(f"{split} file differs from dataset manifest")
        for row in rows:
            meta = row.get("meta", {})
            if meta.get("view_schema") != clean_schema:
                raise ValueError("row view schema differs from the clean-source contract")
            if meta.get("prompt_template_sha256") != template_sha:
                raise ValueError("row prompt template differs from dataset manifest")
            conversations = row.get("conversations")
            prompt = None
            if isinstance(conversations, list):
                for turn in conversations:
                    if isinstance(turn, dict) and turn.get("from") in {"human", "user"}:
                        prompt = turn.get("value")
                        break
            expected_prompt = template.replace(
                "{class_name}", str(meta.get("class_name") or "")
            )
            if prompt != expected_prompt:
                raise ValueError(
                    "row human prompt differs from the rendered deployment template"
                )
    return {**contract, "contract_source": str(manifest_path)}


def _configure_processor(processor: Any, min_pixels: int, max_pixels: int) -> None:
    """Apply and verify the pixel budget across dict and SizeDict processors."""
    image_processor = processor.image_processor
    size = getattr(image_processor, "size", None)
    configured_min = False
    configured_max = False

    if hasattr(image_processor, "min_pixels"):
        image_processor.min_pixels = min_pixels
        configured_min = True
    if hasattr(image_processor, "max_pixels"):
        image_processor.max_pixels = max_pixels
        configured_max = True

    if size is not None:
        if hasattr(size, "shortest_edge"):
            size.shortest_edge = min_pixels
            configured_min = True
        elif hasattr(size, "__setitem__"):
            size["shortest_edge"] = min_pixels
            configured_min = True
        if hasattr(size, "longest_edge"):
            size.longest_edge = max_pixels
            configured_max = True
        elif hasattr(size, "__setitem__"):
            size["longest_edge"] = max_pixels
            configured_max = True

    observed_min = getattr(image_processor, "min_pixels", None)
    observed_max = getattr(image_processor, "max_pixels", None)
    if observed_min is None and size is not None:
        observed_min = getattr(size, "shortest_edge", None)
        if observed_min is None and hasattr(size, "get"):
            observed_min = size.get("shortest_edge")
    if observed_max is None and size is not None:
        observed_max = getattr(size, "longest_edge", None)
        if observed_max is None and hasattr(size, "get"):
            observed_max = size.get("longest_edge")

    if not configured_min or not configured_max:
        raise RuntimeError("image processor exposes no supported pixel-budget fields")
    if observed_min != min_pixels or observed_max != max_pixels:
        raise RuntimeError(
            "image processor pixel budget differs after configuration: "
            f"min={observed_min!r}, max={observed_max!r}"
        )
