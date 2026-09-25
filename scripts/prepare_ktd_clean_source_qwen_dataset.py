#!/usr/bin/env python3
"""Build a self-contained four-view Qwen verifier dataset from reviewed KTD pairs."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from collections import Counter
from pathlib import Path
from typing import Any, Iterable


VIEW_SCHEMA = "annotated_locator_plus_clean_original_v1"
VIEW_ROLES = (
    "annotated_context",
    "annotated_roi",
    "clean_context",
    "clean_roi",
)
ELIGIBLE_MATCH_STATUSES = {"strong", "usable"}
PARTITION_OUTPUTS = {"train": "train.jsonl", "validation": "validation.jsonl"}
PROCESSOR_MIN_PIXELS = 28 * 28 * 64
PROCESSOR_MAX_PIXELS = 28 * 28 * 256
PROMPT_TEMPLATE = (
    "<image>\n图1：带初审定位信息的上下文图。\n"
    "<image>\n图2：带初审定位信息的候选区域图。\n"
    "<image>\n图3：同一现场的清晰无标注原图上下文。\n"
    "<image>\n图4：从清晰原图投影得到的候选区域。\n"
    "初审类别：{class_name}。结合四张图判断红框所指候选是否确为该类别。"
    "只能输出一个 JSON 对象，不得输出解释或 Markdown："
    '{"decision":"CONFIRMED"} 或 {"decision":"REJECTED"}。'
)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as stream:
        for line_no, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSONL at {path}:{line_no}") from exc
            if not isinstance(value, dict):
                raise ValueError(f"JSONL row is not an object at {path}:{line_no}")
            rows.append(value)
    return rows


def _write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _indexed(rows: list[dict[str, Any]], *, name: str) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for row in rows:
        candidate_id = str(row.get("candidate_id") or "")
        if not candidate_id or candidate_id in result:
            raise ValueError(f"{name} contains a duplicate or empty candidate_id: {candidate_id!r}")
        result[candidate_id] = row
    return result


def _validate_gold(pair: dict[str, Any], gold: dict[str, Any]) -> None:
    candidate_id = str(pair["candidate_id"])
    for field in ("human_label", "class_name", "partition", "event_id"):
        if str(pair.get(field) or "") != str(gold.get(field) or ""):
            raise ValueError(f"candidate {candidate_id} differs between pair and gold for {field}")
    if pair.get("human_label") not in {"true", "false"}:
        raise ValueError(f"candidate {candidate_id} has no binary human gold label")
    if pair.get("partition") not in PARTITION_OUTPUTS:
        raise ValueError(f"candidate {candidate_id} has unsupported partition {pair.get('partition')!r}")


def _copy_verified_view(
    *,
    source_root: Path,
    output_dir: Path,
    candidate_id: str,
    role: str,
    declared_path: Any,
    declared_sha256: Any,
) -> tuple[str, str]:
    if not isinstance(declared_path, str) or not declared_path:
        raise ValueError(f"candidate {candidate_id} is missing {role}")
    source = (source_root / declared_path).resolve()
    if not source.is_file():
        raise ValueError(f"candidate {candidate_id} {role} file is unavailable: {source}")
    actual_sha256 = _sha256(source)
    if actual_sha256 != str(declared_sha256 or "").lower():
        raise ValueError(f"candidate {candidate_id} {role} SHA-256 mismatch")
    suffix = source.suffix.lower() or ".jpg"
    destination = output_dir / "media" / role / f"{candidate_id}{suffix}"
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)
    if _sha256(destination) != actual_sha256:
        raise RuntimeError(f"copied {role} SHA-256 mismatch for candidate {candidate_id}")
    return str(destination.relative_to(output_dir)), actual_sha256


def _copy_clean_frame(
    *,
    source_root: Path,
    output_dir: Path,
    candidate_id: str,
    declared_path: Any,
) -> tuple[str, str]:
    if not isinstance(declared_path, str) or not declared_path:
        raise ValueError(f"candidate {candidate_id} is missing clean_frame")
    source = (source_root / declared_path).resolve()
    if not source.is_file():
        raise ValueError(f"candidate {candidate_id} clean frame is unavailable: {source}")
    digest = _sha256(source)
    suffix = source.suffix.lower() or ".jpg"
    destination = output_dir / "clean_frames" / f"{digest}{suffix}"
    destination.parent.mkdir(parents=True, exist_ok=True)
    if not destination.exists():
        shutil.copy2(source, destination)
    if _sha256(destination) != digest:
        raise RuntimeError(f"copied clean-frame SHA-256 mismatch for candidate {candidate_id}")
    return str(destination.relative_to(output_dir)), digest


def _training_prompt(class_name: str) -> str:
    return PROMPT_TEMPLATE.replace("{class_name}", class_name)


def build_dataset(
    *,
    candidate_pairs_path: Path,
    gold_path: Path,
    output_dir: Path,
) -> dict[str, Any]:
    candidate_pairs_path = candidate_pairs_path.expanduser().resolve()
    gold_path = gold_path.expanduser().resolve()
    if not candidate_pairs_path.is_file():
        raise FileNotFoundError(candidate_pairs_path)
    if not gold_path.is_file():
        raise FileNotFoundError(gold_path)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"output directory is not empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    pair_rows = _read_jsonl(candidate_pairs_path)
    gold_rows = _read_jsonl(gold_path)
    pairs = _indexed(pair_rows, name="candidate pairs")
    gold = _indexed(gold_rows, name="gold labels")
    if set(pairs) != set(gold):
        missing_gold = sorted(set(pairs) - set(gold))
        unexpected_gold = sorted(set(gold) - set(pairs))
        raise ValueError(
            "candidate/gold ID sets differ; missing_gold="
            + repr(missing_gold[:5])
            + ", unexpected_gold="
            + repr(unexpected_gold[:5])
        )

    provenance_dir = output_dir / "provenance"
    provenance_dir.mkdir(parents=True, exist_ok=True)
    copied_pairs = provenance_dir / "candidate_pairs.jsonl"
    copied_gold = provenance_dir / "gold_labels.jsonl"
    shutil.copy2(candidate_pairs_path, copied_pairs)
    shutil.copy2(gold_path, copied_gold)
    input_shas = {
        "candidate_pairs": _sha256(candidate_pairs_path),
        "gold_labels": _sha256(gold_path),
    }
    if _sha256(copied_pairs) != input_shas["candidate_pairs"] or _sha256(copied_gold) != input_shas["gold_labels"]:
        raise RuntimeError("copied provenance input SHA-256 mismatch")

    output_rows: dict[str, list[dict[str, Any]]] = {partition: [] for partition in PARTITION_OUTPUTS}
    lineage_rows: list[dict[str, Any]] = []
    source_root = candidate_pairs_path.parent
    class_counts: Counter[str] = Counter()
    label_counts: Counter[str] = Counter()
    match_counts: Counter[str] = Counter()
    source_kind_counts: Counter[str] = Counter()

    for candidate_id in sorted(pairs):
        pair = pairs[candidate_id]
        gold_row = gold[candidate_id]
        _validate_gold(pair, gold_row)
        eligible = bool(pair.get("has_clean_source")) and pair.get("clean_match_status") in ELIGIBLE_MATCH_STATUSES
        lineage: dict[str, Any] = {
            "candidate_id": candidate_id,
            "partition": pair["partition"],
            "event_id": pair["event_id"],
            "source_task_id": pair.get("source_task_id"),
            "class_name": pair["class_name"],
            "human_label": gold_row["human_label"],
            "eligible": eligible,
            "exclusion_reason": None if eligible else "missing_or_unusable_clean_source",
            "clean_match_status": pair.get("clean_match_status"),
            "clean_source_kind": pair.get("clean_source_kind"),
            "candidate_pairs_input_sha256": input_shas["candidate_pairs"],
            "gold_labels_input_sha256": input_shas["gold_labels"],
        }
        if not eligible:
            lineage_rows.append(lineage)
            continue
        if not pair.get("source_task_id") or not pair.get("event_id"):
            raise ValueError(f"candidate {candidate_id} lacks task/event lineage")

        image_paths: list[str] = []
        view_shas: dict[str, str] = {}
        for role in VIEW_ROLES:
            relative_path, digest = _copy_verified_view(
                source_root=source_root,
                output_dir=output_dir,
                candidate_id=candidate_id,
                role=role,
                declared_path=pair.get(role),
                declared_sha256=pair.get(role + "_sha256"),
            )
            image_paths.append(relative_path)
            view_shas[role] = digest
        clean_frame_path, clean_frame_sha256 = _copy_clean_frame(
            source_root=source_root,
            output_dir=output_dir,
            candidate_id=candidate_id,
            declared_path=pair.get("clean_frame"),
        )
        decision = "CONFIRMED" if gold_row["human_label"] == "true" else "REJECTED"
        prompt = _training_prompt(str(pair["class_name"]))
        prompt_template_sha256 = hashlib.sha256(PROMPT_TEMPLATE.encode("utf-8")).hexdigest()
        prompt_sha256 = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
        training_row = {
            "id": candidate_id,
            "image": image_paths,
            "conversations": [
                {"from": "human", "value": prompt},
                {"from": "gpt", "value": json.dumps({"decision": decision}, ensure_ascii=False, separators=(",", ":"))},
            ],
            "meta": {
                "view_schema": VIEW_SCHEMA,
                "view_roles": list(VIEW_ROLES),
                "view_sha256": view_shas,
                "prompt_template_sha256": prompt_template_sha256,
                "prompt_sha256": prompt_sha256,
                "label_source": "human_gold",
                "label_field": "human_label",
                "human_label": gold_row["human_label"],
                "image_id": candidate_id,
                "candidate_id": candidate_id,
                "candidate_index": "0",
                "class_name": pair["class_name"],
                "bbox": pair["bbox_px"],
                "partition": pair["partition"],
                "event_id": pair["event_id"],
                "source_event_id": pair["event_id"],
                "source_task_id": pair["source_task_id"],
                "image_sha256": view_shas["annotated_context"],
                "image_object_key": str(pair.get("annotated_context") or ""),
                "source_image": image_paths[0],
                "candidate_pairs_file": "provenance/candidate_pairs.jsonl",
                "candidate_pairs_sha256": input_shas["candidate_pairs"],
                "gold_file": "provenance/gold_labels.jsonl",
                "gold_sha256": input_shas["gold_labels"],
                "clean_frame": clean_frame_path,
                "clean_frame_sha256": clean_frame_sha256,
                "clean_match_status": pair["clean_match_status"],
                "clean_source_kind": pair.get("clean_source_kind"),
                "projected_clean_bbox_px": pair.get("projected_clean_bbox_px"),
                "projected_clean_polygon_px": pair.get("projected_clean_polygon_px"),
            },
        }
        output_rows[str(pair["partition"])].append(training_row)
        lineage.update(
            view_schema=VIEW_SCHEMA,
            views=[{"role": role, "path": path, "sha256": view_shas[role]} for role, path in zip(VIEW_ROLES, image_paths)],
            clean_frame=clean_frame_path,
            clean_frame_sha256=clean_frame_sha256,
        )
        lineage_rows.append(lineage)
        class_counts[str(pair["class_name"])] += 1
        label_counts[str(gold_row["human_label"])] += 1
        match_counts[str(pair["clean_match_status"])] += 1
        source_kind_counts[str(pair.get("clean_source_kind"))] += 1

    output_files: dict[str, dict[str, Any]] = {}
    for partition, filename in PARTITION_OUTPUTS.items():
        path = output_dir / filename
        _write_jsonl(path, output_rows[partition])
        output_files[partition] = {
            "path": filename,
            "rows": len(output_rows[partition]),
            "sha256": _sha256(path),
        }
    lineage_path = output_dir / "lineage_manifest.jsonl"
    _write_jsonl(lineage_path, lineage_rows)
    output_files["lineage"] = {
        "path": lineage_path.name,
        "rows": len(lineage_rows),
        "sha256": _sha256(lineage_path),
    }

    train = output_rows["train"]
    validation = output_rows["validation"]
    isolation = {
        "clean_frame_sha256_overlap": sorted(
            {row["meta"]["clean_frame_sha256"] for row in train}
            & {row["meta"]["clean_frame_sha256"] for row in validation}
        ),
        "source_task_id_overlap": sorted(
            {row["meta"]["source_task_id"] for row in train}
            & {row["meta"]["source_task_id"] for row in validation}
        ),
        "event_id_overlap": sorted(
            {row["meta"]["event_id"] for row in train}
            & {row["meta"]["event_id"] for row in validation}
        ),
    }
    if any(isolation.values()):
        raise ValueError(f"declared train/validation partitions are not isolated: {isolation}")

    summary = {
        "schema_version": 1,
        "view_schema": VIEW_SCHEMA,
        "deployment_contract": {
            "view_schema": VIEW_SCHEMA,
            "view_roles": list(VIEW_ROLES),
            "prompt_template": PROMPT_TEMPLATE,
            "prompt_template_sha256": hashlib.sha256(PROMPT_TEMPLATE.encode("utf-8")).hexdigest(),
            "strict_output": {
                "json_object_keys": ["decision"],
                "allowed_decisions": ["CONFIRMED", "REJECTED"],
                "invalid_output": "uncertain",
            },
            "processor": {
                "min_pixels": PROCESSOR_MIN_PIXELS,
                "max_pixels": PROCESSOR_MAX_PIXELS,
                "max_sequence_length": 2048,
            },
            "quantization": {
                "load_in_4bit": True,
                "bnb_4bit_quant_type": "nf4",
                "bnb_4bit_use_double_quant": True,
                "bnb_4bit_compute_dtype": "bfloat16",
            },
        },
        "source": {
            "candidate_pairs": {"path": str(candidate_pairs_path), "sha256": input_shas["candidate_pairs"], "rows": len(pair_rows)},
            "gold_labels": {"path": str(gold_path), "sha256": input_shas["gold_labels"], "rows": len(gold_rows)},
        },
        "selection_policy": "has_clean_source=true and clean_match_status in {strong,usable}; preserve source partition",
        "limitations": [
            "normalized gold_labels.jsonl does not retain labeled_by or labeled_at; this manifest pins its SHA-256 but does not independently re-attest reviewer identity or timestamp"
        ],
        "input_candidates": len(pair_rows),
        "eligible_candidates": len(train) + len(validation),
        "excluded_candidates": len(pair_rows) - len(train) - len(validation),
        "partition_rows": {"train": len(train), "validation": len(validation)},
        "class_counts": dict(sorted(class_counts.items())),
        "label_counts": dict(sorted(label_counts.items())),
        "clean_match_status_counts": dict(sorted(match_counts.items())),
        "clean_source_kind_counts": dict(sorted(source_kind_counts.items())),
        "unique_clean_frames": len({row["meta"]["clean_frame_sha256"] for row in train + validation}),
        "unique_source_tasks": len({row["meta"]["source_task_id"] for row in train + validation}),
        "unique_events": len({row["meta"]["event_id"] for row in train + validation}),
        "isolation": isolation,
        "outputs": output_files,
    }
    summary_path = output_dir / "dataset_manifest.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-pairs", type=Path, required=True)
    parser.add_argument("--gold", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    summary = build_dataset(
        candidate_pairs_path=args.candidate_pairs,
        gold_path=args.gold,
        output_dir=args.output_dir,
    )
    print(json.dumps(summary, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
