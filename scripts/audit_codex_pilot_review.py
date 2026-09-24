#!/usr/bin/env python3
"""Audit a Codex pilot human-review export and freeze eligible human gold."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import statistics
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping, Sequence


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_no, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not raw.strip():
            continue
        try:
            row = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise ValueError(f"invalid JSONL at {path}:{line_no}") from exc
        if not isinstance(row, dict):
            raise ValueError(f"JSONL row is not an object at {path}:{line_no}")
        rows.append(row)
    return rows


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.write_text(
        "".join(json.dumps(dict(row), ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _parse_timestamp(value: Any, candidate_id: str) -> datetime:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{candidate_id} has no reviewed_at timestamp")
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError(f"{candidate_id} has an invalid reviewed_at timestamp") from exc


def _unique_by_candidate(
    rows: Sequence[Mapping[str, Any]], *, source_name: str
) -> dict[str, Mapping[str, Any]]:
    result: dict[str, Mapping[str, Any]] = {}
    for row in rows:
        candidate_id = str(row.get("candidate_id") or "")
        if not candidate_id or candidate_id in result:
            raise ValueError(f"{source_name} has a missing or duplicate candidate_id")
        result[candidate_id] = row
    return result


def _require_equal(candidate_id: str, field: str, actual: Any, expected: Any) -> None:
    if actual != expected:
        raise ValueError(
            f"{candidate_id} {field} differs from the frozen pilot: "
            f"{actual!r} != {expected!r}"
        )


def _comparison_status(suggestion: str, action: str, label: Any) -> str:
    if action == "confirm":
        return "confirmed_binary_suggestion"
    if suggestion == "uncertain" and label in {"true", "false"}:
        return "human_resolved_uncertain"
    if suggestion in {"true", "false"} and label in {"true", "false"}:
        return "human_corrected_binary"
    if action == "uncertain":
        return "human_kept_uncertain"
    return "human_flagged_box_issue"


def _local_source_metadata(
    manifest: Mapping[str, Any], source_root: Path | None
) -> dict[str, Any]:
    fallback = {
        "source_task_id": None,
        "source_event_group": str(manifest["source_group"]),
        "image_object_key": str(manifest["source_original"]),
        "canonical_source_image": None,
        "detail_file": None,
    }
    if source_root is None:
        return fallback
    source_image = (source_root / str(manifest["source_original"])).resolve()
    if not source_image.is_file():
        raise ValueError(f"declared source image is unavailable: {source_image}")
    if _sha256(source_image) != str(manifest["original_sha256"]):
        raise ValueError(f"declared source image SHA-256 mismatch: {source_image}")
    detail_file = source_image.parent / "detail.json"
    if not detail_file.is_file():
        return {**fallback, "canonical_source_image": str(source_image)}
    detail = json.loads(detail_file.read_text(encoding="utf-8"))
    data = detail.get("data") if isinstance(detail, Mapping) else None
    if not isinstance(data, Mapping):
        raise ValueError(f"detail file has no data object: {detail_file}")
    task_id = str(data.get("taskId") or "")
    images = data.get("warningImages") or data.get("suspectWarningImages")
    if not task_id or not isinstance(images, list) or not images:
        raise ValueError(f"detail file lacks task/image lineage: {detail_file}")
    file_path = str(images[0].get("filePath") or "")
    marker = "/minio-src/"
    object_key = file_path.split(marker, 1)[1] if marker in file_path else file_path
    if not object_key:
        raise ValueError(f"detail file has no image object key: {detail_file}")
    return {
        "source_task_id": task_id,
        "source_event_group": task_id,
        "image_object_key": object_key,
        "canonical_source_image": str(source_image),
        "detail_file": str(detail_file),
    }


def audit_review_export(
    *,
    review_export: Path,
    pilot_root: Path,
    output_dir: Path,
    reviewer_id: str,
    expected_count: int | None = None,
    source_root: Path | None = None,
) -> dict[str, Any]:
    if not reviewer_id.strip():
        raise ValueError("reviewer_id must be non-empty")
    review_export = review_export.resolve()
    pilot_root = pilot_root.resolve()
    manifest_path = pilot_root / "pilot_manifest.jsonl"
    prelabel_path = pilot_root / "codex_prelabels.jsonl"
    pilot_summary_path = pilot_root / "summary.json"
    for path in (review_export, manifest_path, prelabel_path, pilot_summary_path):
        if not path.is_file():
            raise FileNotFoundError(path)

    review_rows = _read_jsonl(review_export)
    manifest_rows = _read_jsonl(manifest_path)
    prelabel_rows = _read_jsonl(prelabel_path)
    if expected_count is not None and len(review_rows) != expected_count:
        raise ValueError(
            f"review export has {len(review_rows)} rows; expected {expected_count}"
        )
    if not review_rows:
        raise ValueError("review export is empty")

    review_by_id = _unique_by_candidate(review_rows, source_name="review export")
    manifest_by_id = _unique_by_candidate(manifest_rows, source_name="pilot manifest")
    prelabel_by_id = _unique_by_candidate(prelabel_rows, source_name="Codex prelabels")
    expected_ids = set(manifest_by_id)
    if set(review_by_id) != expected_ids or set(prelabel_by_id) != expected_ids:
        raise ValueError("candidate IDs differ across review, manifest, and prelabels")

    batch_ids = {str(row.get("batch_id") or "") for row in review_rows}
    prelabel_batch_ids = {str(row.get("batch_id") or "") for row in prelabel_rows}
    if len(batch_ids) != 1 or batch_ids != prelabel_batch_ids or "" in batch_ids:
        raise ValueError("batch_id differs across review and Codex prelabels")
    batch_id = next(iter(batch_ids))

    human_gold: list[dict[str, Any]] = []
    training_candidates: list[dict[str, Any]] = []
    training_admission: list[dict[str, Any]] = []
    hard_negatives: list[dict[str, Any]] = []
    qwen_training_fragment: list[dict[str, Any]] = []
    exclusions: list[dict[str, Any]] = []
    comparisons: list[dict[str, Any]] = []
    timestamps: list[datetime] = []

    for manifest in manifest_rows:
        candidate_id = str(manifest["candidate_id"])
        review = review_by_id[candidate_id]
        prelabel = prelabel_by_id[candidate_id]
        _require_equal(candidate_id, "batch_id", review.get("batch_id"), batch_id)
        _require_equal(
            candidate_id,
            "source_image_sha256",
            review.get("source_image_sha256"),
            manifest.get("original_sha256"),
        )
        _require_equal(
            candidate_id,
            "candidate_bbox_px",
            review.get("candidate_bbox_px"),
            manifest.get("bbox_px"),
        )
        _require_equal(
            candidate_id,
            "candidate_class",
            review.get("candidate_class"),
            manifest.get("class_name"),
        )
        _require_equal(
            candidate_id,
            "source_group",
            review.get("source_group"),
            manifest.get("source_group"),
        )
        _require_equal(
            candidate_id,
            "codex_suggestion",
            review.get("codex_suggestion"),
            prelabel.get("codex_suggestion"),
        )
        _require_equal(
            candidate_id,
            "codex_evidence",
            review.get("codex_evidence"),
            prelabel.get("visual_evidence"),
        )
        if review.get("human_reviewed") is not True:
            raise ValueError(f"{candidate_id} is not marked human_reviewed")
        if review.get("label_source") != "human_review_pending_audit":
            raise ValueError(f"{candidate_id} has an unexpected review label_source")
        if review.get("is_training_eligible") is not False:
            raise ValueError(f"{candidate_id} was prematurely marked training eligible")

        action = str(review.get("human_action") or "")
        label = review.get("human_label")
        suggestion = str(prelabel.get("codex_suggestion") or "")
        if action == "confirm":
            if suggestion not in {"true", "false"} or label != suggestion:
                raise ValueError(f"{candidate_id} has an invalid confirm action")
        elif action in {"true", "false"}:
            if label != action:
                raise ValueError(f"{candidate_id} action and human_label differ")
        elif action == "uncertain":
            if label != "uncertain":
                raise ValueError(f"{candidate_id} uncertain action has a wrong label")
        elif action == "issue":
            if label is not None:
                raise ValueError(f"{candidate_id} box issue must not have a binary label")
        else:
            raise ValueError(f"{candidate_id} has an unsupported human_action: {action!r}")

        reviewed_at = _parse_timestamp(review.get("reviewed_at"), candidate_id)
        timestamps.append(reviewed_at)
        full_original = (pilot_root / str(manifest["full_original_copy"])).resolve()
        if not full_original.is_file():
            raise ValueError(f"{candidate_id} original image copy is unavailable")
        if _sha256(full_original) != str(manifest["original_sha256"]):
            raise ValueError(f"{candidate_id} original image SHA-256 mismatch")
        for view_name, expected_digest in dict(manifest["derived_sha256"]).items():
            view_path = (pilot_root / str(manifest["media"][view_name])).resolve()
            if not view_path.is_file() or _sha256(view_path) != expected_digest:
                raise ValueError(f"{candidate_id} {view_name} view SHA-256 mismatch")

        eligible = label in {"true", "false"} and action != "issue"
        source_metadata = _local_source_metadata(manifest, source_root)
        comparison = {
            "batch_id": batch_id,
            "candidate_id": candidate_id,
            "codex_suggestion": suggestion,
            "human_action": action,
            "human_label": label,
            "human_note": str(review.get("human_comment") or ""),
            "comparison_status": _comparison_status(suggestion, action, label),
            "reviewed_at": str(review["reviewed_at"]),
        }
        comparisons.append(comparison)

        common = {
            "schema_version": 1,
            "batch_id": batch_id,
            "candidate_id": candidate_id,
            "sample_id": str(manifest["event_id"]),
            "source_event_id": source_metadata["source_event_group"],
            "source_task_id": source_metadata["source_task_id"],
            "source_capture_group": str(manifest["source_group"]),
            "target_index": candidate_id,
            "class_name": str(manifest["class_name"]),
            "bbox": list(manifest["bbox_norm"]),
            "bbox_px": list(manifest["bbox_px"]),
            "image_size": list(manifest["image_size"]),
            "image_sha256": str(manifest["original_sha256"]),
            "image_object_key": source_metadata["image_object_key"],
            "source_image": str(full_original),
            "canonical_source_image": source_metadata["canonical_source_image"],
            "source_detail_file": source_metadata["detail_file"],
            "context_image": str(
                (pilot_root / str(manifest["media"]["context"])).resolve()
            ),
            "roi_image": str(
                (pilot_root / str(manifest["media"]["roi"])).resolve()
            ),
            "human_label": label,
            "human_note": str(review.get("human_comment") or ""),
            "labeled_by": reviewer_id,
            "reviewer_identity_status": "session_attributed_workspace_account",
            "labeled_at": str(review["reviewed_at"]),
            "human_action": action,
            "label_source": "human_gold" if eligible else "human_review_excluded",
            "codex_suggestion": suggestion,
            "codex_prompt_version": prelabel.get("prompt_version"),
            "codex_model": prelabel.get("codex_model"),
            "codex_box_fit": prelabel.get("box_fit"),
            "source_row_ordinals": list(manifest.get("source_row_ordinals", [])),
            "split": "train_candidate_single_source_group",
            "training_task": "candidate_roi_binary_verifier",
            "training_eligible": eligible,
            "qwen_training_eligible": eligible,
            "detector_training_eligible": False,
            "detector_training_reason": (
                "candidate truth was reviewed; detector box quality and image-level "
                "annotation completeness were not independently adjudicated"
            ),
            "validation_eligible": False,
            "locked_test_eligible": False,
        }
        training_admission.append(
            {
                "batch_id": batch_id,
                "candidate_id": candidate_id,
                "human_label": label,
                "qwen_training_eligible": eligible,
                "qwen_reason": (
                    "binary human confirmation with verified image/box lineage"
                    if eligible
                    else "human result is uncertain or the candidate box is invalid"
                ),
                "detector_training_eligible": False,
                "detector_reason": common["detector_training_reason"],
                "validation_eligible": False,
                "locked_test_eligible": False,
                "source_event_id": common["source_event_id"],
                "source_capture_group": common["source_capture_group"],
            }
        )
        if eligible:
            human_gold.append(common)
            training_candidates.append(common)
            decision = "CONFIRMED" if label == "true" else "REJECTED"
            prompt = (
                "<image>\n<image>\n"
                "第一张图是候选框上下文，第二张图是候选区域放大图。"
                f"候选类别：{common['class_name']}。"
                "请判断候选区域内是否存在且符合该类别的目标。"
                '只输出 JSON：{"decision":"CONFIRMED"} 或 '
                '{"decision":"REJECTED"}。'
            )
            qwen_training_fragment.append(
                {
                    "id": candidate_id,
                    "image": [
                        f"media/{candidate_id}-context.jpg",
                        f"media/{candidate_id}-roi.jpg",
                    ],
                    "conversations": [
                        {"from": "human", "value": prompt},
                        {
                            "from": "gpt",
                            "value": json.dumps({"decision": decision}),
                        },
                    ],
                    "meta": {
                        "label_source": "human_gold",
                        "label_field": "human_label",
                        "labeled_by": reviewer_id,
                        "source_run_id": batch_id,
                        "image_id": common["sample_id"],
                        "image_object_key": common["image_object_key"],
                        "image_sha256": common["image_sha256"],
                        "candidate_index": candidate_id,
                        "class_name": common["class_name"],
                        "bbox": common["bbox"],
                        "source_image": f"source_images/{common['sample_id']}.jpg",
                        "source_file": "human_gold_annotations.jsonl",
                    },
                }
            )
            if label == "false":
                hard_negatives.append(common)
        else:
            exclusions.append(
                {
                    **common,
                    "exclusion_reason": (
                        "human_uncertain" if action == "uncertain" else "box_issue"
                    ),
                }
            )

    action_times = sorted(timestamps)
    action_gaps = [
        (right - left).total_seconds()
        for left, right in zip(action_times, action_times[1:])
    ]
    comparison_counts = Counter(row["comparison_status"] for row in comparisons)
    binary_comparisons = [
        row
        for row in comparisons
        if row["codex_suggestion"] in {"true", "false"}
    ]
    binary_agreements = sum(
        row["codex_suggestion"] == row["human_label"]
        for row in binary_comparisons
    )
    source_groups = sorted({row["source_event_id"] for row in training_candidates})
    summary = {
        "schema_version": 1,
        "batch_id": batch_id,
        "audit_status": "passed",
        "reviewer_id": reviewer_id,
        "reviewer_identity_status": "session_attributed_workspace_account",
        "reviewer_attribution": "workspace user who submitted the export in the current Codex conversation",
        "review_export": str(review_export),
        "review_export_sha256": _sha256(review_export),
        "pilot_manifest_sha256": _sha256(manifest_path),
        "codex_prelabels_sha256": _sha256(prelabel_path),
        "reviewed_rows": len(review_rows),
        "unique_candidates": len(review_by_id),
        "human_gold_rows": len(human_gold),
        "excluded_rows": len(exclusions),
        "qwen_training_candidate_rows": len(training_candidates),
        "qwen_training_fragment_rows": len(qwen_training_fragment),
        "detector_training_candidate_rows": 0,
        "hard_negative_rows": len(hard_negatives),
        "human_labels": dict(sorted(Counter(row["human_label"] for row in human_gold).items())),
        "human_actions": dict(sorted(Counter(str(row["human_action"]) for row in review_rows).items())),
        "codex_suggestions": dict(sorted(Counter(str(row["codex_suggestion"]) for row in review_rows).items())),
        "comparison": dict(sorted(comparison_counts.items())),
        "workflow_comparison": {
            "codex_binary_suggestions": len(binary_comparisons),
            "human_agreed_with_codex_binary": binary_agreements,
            "human_reversed_codex_binary": len(binary_comparisons) - binary_agreements,
            "codex_uncertain_resolved_by_human": comparison_counts.get(
                "human_resolved_uncertain", 0
            ),
            "three_state_exact_match_rows": comparison_counts.get(
                "confirmed_binary_suggestion", 0
            ),
            "three_state_exact_match_rate": round(
                comparison_counts.get("confirmed_binary_suggestion", 0)
                / len(review_rows),
                6,
            ),
            "accuracy_claim": False,
        },
        "human_notes_nonempty": sum(
            bool(str(row.get("human_comment") or "").strip()) for row in review_rows
        ),
        "codex_box_fit": dict(
            sorted(Counter(str(row.get("box_fit")) for row in prelabel_rows).items())
        ),
        "source_images": len({row["image_sha256"] for row in human_gold}),
        "source_groups": source_groups,
        "source_capture_groups": sorted(
            {row["source_capture_group"] for row in training_candidates}
        ),
        "classes": dict(sorted(Counter(row["class_name"] for row in human_gold).items())),
        "first_saved_action_at": action_times[0].isoformat(),
        "last_saved_action_at": action_times[-1].isoformat(),
        "saved_action_window_seconds": round(
            (action_times[-1] - action_times[0]).total_seconds(), 3
        ),
        "median_inter_action_gap_seconds": (
            round(statistics.median(action_gaps), 3)
            if action_gaps
            else None
        ),
        "training_partition": "train_candidate_single_source_group",
        "accuracy_evaluated": False,
        "limitations": [
            "All candidates belong to one source capture group and must remain in one split.",
            "This pilot contains vehicle candidates only and has no person-class coverage.",
            "The saved action window is not an active-time measurement.",
            "Codex/human agreement on this workflow pilot is not an independent accuracy result.",
        ],
    }
    input_receipt = {
        "schema_version": 1,
        "batch_id": batch_id,
        "files": [
            {"role": "human_review_export", "path": str(review_export), "sha256": _sha256(review_export)},
            {"role": "pilot_manifest", "path": str(manifest_path), "sha256": _sha256(manifest_path)},
            {"role": "codex_prelabels", "path": str(prelabel_path), "sha256": _sha256(prelabel_path)},
            {"role": "pilot_summary", "path": str(pilot_summary_path), "sha256": _sha256(pilot_summary_path)},
        ],
        "source_images": [
            {
                "sample_id": row["sample_id"],
                "path": row["canonical_source_image"] or row["source_image"],
                "sha256": row["image_sha256"],
            }
            for row in {
                row["sample_id"]: row for row in training_candidates
            }.values()
        ],
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    media_dir = output_dir / "media"
    source_image_dir = output_dir / "source_images"
    media_dir.mkdir(exist_ok=True)
    source_image_dir.mkdir(exist_ok=True)
    copied_source_ids: set[str] = set()
    for row in training_candidates:
        candidate_id = str(row["candidate_id"])
        for view_name, source_field in (
            ("context", "context_image"),
            ("roi", "roi_image"),
        ):
            source_path = Path(str(row[source_field]))
            destination = media_dir / f"{candidate_id}-{view_name}.jpg"
            shutil.copyfile(source_path, destination)
            if _sha256(destination) != _sha256(source_path):
                raise AssertionError(f"copied {view_name} differs for {candidate_id}")
        sample_id = str(row["sample_id"])
        if sample_id not in copied_source_ids:
            source_path = Path(
                str(row["canonical_source_image"] or row["source_image"])
            )
            destination = source_image_dir / f"{sample_id}.jpg"
            shutil.copyfile(source_path, destination)
            if _sha256(destination) != str(row["image_sha256"]):
                raise AssertionError(f"copied source image differs for {sample_id}")
            copied_source_ids.add(sample_id)
    copied_export = output_dir / "source_human_review_export.jsonl"
    shutil.copyfile(review_export, copied_export)
    if _sha256(copied_export) != summary["review_export_sha256"]:
        raise AssertionError("copied review export hash differs from the submitted file")
    _write_jsonl(output_dir / "human_gold_annotations.jsonl", human_gold)
    _write_jsonl(output_dir / "training_candidates.jsonl", training_candidates)
    _write_jsonl(output_dir / "training_admission.jsonl", training_admission)
    _write_jsonl(output_dir / "hard_negative.jsonl", hard_negatives)
    _write_jsonl(
        output_dir / "qwen_training_fragment.jsonl", qwen_training_fragment
    )
    _write_jsonl(output_dir / "excluded_from_training.jsonl", exclusions)
    _write_jsonl(output_dir / "codex_human_comparison.jsonl", comparisons)
    (output_dir / "input_receipt.json").write_text(
        json.dumps(input_receipt, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (output_dir / "README.md").write_text(
        "# KTD Codex 试批人工确认审计\n\n"
        f"- 批次：`{batch_id}`\n"
        f"- 已审计人工二态标签：{len(human_gold)} 条\n"
        f"- 排除：{len(exclusions)} 条\n"
        f"- 人工标签：{summary['human_labels']}\n"
        f"- Codex/人工对比：{summary['comparison']}\n"
        f"- Qwen 候选复核训练准入：{len(training_candidates)} 条\n"
        f"- 自包含 Qwen 训练片段：{len(qwen_training_fragment)} 条\n"
        "- 检测器训练准入：0 条；本次人工动作没有独立确认框质量与整图标注完整性。\n"
        "- 整个来源任务组冻结为训练候选，不拆到验证集或盲测集。\n"
        "- 本试批不是独立准确率基准。\n",
        encoding="utf-8",
    )
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--review-export", type=Path, required=True)
    parser.add_argument("--pilot-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--reviewer-id", required=True)
    parser.add_argument("--expected-count", type=int)
    parser.add_argument(
        "--source-root",
        type=Path,
        help="Optional root used to verify canonical images and local detail.json lineage.",
    )
    args = parser.parse_args()
    summary = audit_review_export(
        review_export=args.review_export,
        pilot_root=args.pilot_root,
        output_dir=args.output_dir,
        reviewer_id=args.reviewer_id,
        expected_count=args.expected_count,
        source_root=args.source_root,
    )
    print(json.dumps(summary, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
