"""Canonical box construction, alias merging, gold/quarantine determination.

Implements contract sections 2-3. No voting anywhere: conflicting human labels
or label/note suspected conflicts go to quarantine with full evidence.
"""

from __future__ import annotations

import math
from typing import Any

from . import CLASS_NAME_TO_ID
from .common import sha256_text
from .priors import prior_key
from .provenance import attach_provenance, source_record

# Contract section 3.2: deterministic suspicion token lists (heuristic sentinel,
# only routes rows to human review, never decides a label).
SUSPECT_TOKENS_TRUE = ("不是", "并非", "误报", "误检", "没有", "无此", "假目标")
SUSPECT_TOKENS_FALSE = ("确实有", "确认为真", "是真的")


def pixel_bbox(bbox_norm: list[float], width: int, height: int) -> list[int]:
    """floor(norm * size + 0.5), clamped to [0, size] — contract section 2."""

    def _px(value: float, size: int) -> int:
        return max(0, min(size, int(math.floor(value * size + 0.5))))

    return [
        _px(bbox_norm[0], width),
        _px(bbox_norm[1], height),
        _px(bbox_norm[2], width),
        _px(bbox_norm[3], height),
    ]


def canonical_id(image_sha256: str, class_id: int, bbox_px: list[int]) -> str:
    return sha256_text(f"{image_sha256}|{class_id}|{bbox_px[0]}|{bbox_px[1]}|{bbox_px[2]}|{bbox_px[3]}")


def suspect_note(label: str, note: str) -> list[str]:
    """Return matched suspicion tokens for a (label, note) pair; empty = clean."""
    if not note:
        return []
    tokens = SUSPECT_TOKENS_TRUE if label == "true" else SUSPECT_TOKENS_FALSE if label == "false" else ()
    return [token for token in tokens if token in note]


def build_canonical(
    rows: list[dict[str, Any]],
    images: dict[str, dict],
    image_errors: dict[str, str],
    gpt_priors: dict | None = None,
    qwen_priors: dict | None = None,
) -> dict[str, Any]:
    """Group DB rows into canonical boxes and decide gold / quarantine / unlabeled.

    Returns {"boxes": {canonical_id: box}, "authoring": [...], "counts": {...}}.
    Each box carries aliases, priors, and a verdict: gold | quarantine | unlabeled.
    """
    gpt_priors = gpt_priors or {}
    qwen_priors = qwen_priors or {}
    boxes: dict[str, dict[str, Any]] = {}
    authoring: list[dict[str, Any]] = []
    counts = {
        "raw_candidates": len(rows),
        "excluded_unknown_class": 0,
        "excluded_image_unavailable": 0,
        "image_errors": len(image_errors),
    }

    for row in rows:
        class_id = CLASS_NAME_TO_ID.get(row["class_name"])
        ann = row["annotation"]
        if class_id is None:
            counts["excluded_unknown_class"] += 1
            if ann:
                authoring.append(_authoring_row(row, ann, None, None, "unknown_class"))
            continue
        image = images.get(row["object_key"] or "")
        if image is None:
            counts["excluded_image_unavailable"] += 1
            if ann:
                authoring.append(_authoring_row(row, ann, class_id, None, "image_unavailable"))
            continue

        bbox_px = pixel_bbox(row["bbox_norm"], image["width"], image["height"])
        cid = canonical_id(image["sha256"], class_id, bbox_px)
        box = boxes.get(cid)
        if box is None:
            box = boxes[cid] = {
                "canonical_id": cid,
                "class_id": class_id,
                "class_name": row["class_name"],
                "image_sha256": image["sha256"],
                "image_object_key": row["object_key"],
                "image_width": image["width"],
                "image_height": image["height"],
                "bbox_px": bbox_px,
                "rows": [],
            }
        box["rows"].append(row)
        if ann:
            authoring.append(_authoring_row(row, ann, class_id, cid, None))

    result_boxes = {}
    for cid in sorted(boxes):
        box = boxes[cid]
        box["rows"].sort(key=lambda r: r["candidate_id"])
        verdict, detail = _verdict(box)
        box["verdict"] = verdict
        box.update(detail)
        result_boxes[cid] = box
        _attach_priors(box, gpt_priors, qwen_priors)

    counts.update(
        attach_provenance(
            result_boxes,
            source_rows=rows,
            image_sha_by_object_key={
                object_key: image["sha256"] for object_key, image in images.items()
            },
        )
    )

    # Fill authoring inclusion flags now that verdicts are known.
    gold_cids = {cid for cid, b in result_boxes.items() if b["verdict"] == "gold"}
    reason_by_cid = {
        cid: b["quarantine_reasons"][0]
        for cid, b in result_boxes.items()
        if b["verdict"] == "quarantine"
    }
    for entry in authoring:
        cid = entry["canonical_id"]
        if cid is None:
            continue  # exclusion_reason already set above
        if entry["source"] == "auto_annotation":
            entry["included_in_gold"] = False
            entry["exclusion_reason"] = "auto_annotation_applied"
        elif cid in gold_cids:
            entry["included_in_gold"] = True
        else:
            entry["included_in_gold"] = False
            entry["exclusion_reason"] = reason_by_cid.get(cid, "not_in_gold")

    counts["canonical_boxes"] = len(result_boxes)
    counts["gold_records"] = len(gold_cids)
    counts["quarantine_records"] = sum(1 for b in result_boxes.values() if b["verdict"] == "quarantine")
    counts["quarantine_label_conflict"] = sum(
        1 for b in result_boxes.values() if b["verdict"] == "quarantine" and "label_conflict" in b["quarantine_reasons"]
    )
    counts["quarantine_note_label_suspect"] = sum(
        1
        for b in result_boxes.values()
        if b["verdict"] == "quarantine" and "note_label_suspect" in b["quarantine_reasons"]
    )
    counts["unlabeled_canonical"] = sum(1 for b in result_boxes.values() if b["verdict"] == "unlabeled")
    counts["authoring_rows"] = len(authoring)
    counts["authoring_human"] = sum(1 for a in authoring if a["source"] == "human")
    counts["authoring_auto_excluded"] = sum(1 for a in authoring if a["source"] == "auto_annotation")
    counts["prior_gpt_matched"] = sum(1 for b in result_boxes.values() if b["prior_gpt"])
    counts["prior_qwen_matched"] = sum(1 for b in result_boxes.values() if b["prior_qwen"])
    labeled_ats = [a["labeled_at"] for a in authoring if a["source"] == "human" and a["labeled_at"]]
    counts["max_labeled_at"] = max(labeled_ats) if labeled_ats else None
    return {"boxes": result_boxes, "authoring": sorted(authoring, key=lambda a: a["annotation_id"]), "counts": counts}


def _authoring_row(row, ann, class_id, cid, exclusion_reason) -> dict[str, Any]:
    return {
        "annotation_id": ann["annotation_id"],
        "candidate_id": row["candidate_id"],
        "sample_id": row["sample_id"],
        "canonical_id": cid,
        "class_id": class_id,
        "class_name": row["class_name"],
        "label": ann["label"],
        "note": ann["note"],
        "reviewer": ann["reviewer"],
        "reviewer_id": ann["labeled_by"],
        "revision": ann["revision"],
        "labeled_at": ann["labeled_at"],
        "source": "human" if ann["labeled_by"] else "auto_annotation",
        "included_in_gold": False,
        "exclusion_reason": exclusion_reason,
        "provenance": source_record(row),
    }


def _verdict(box) -> tuple[str, dict[str, Any]]:
    """gold | quarantine | unlabeled, per contract section 3. Never votes."""
    human = [
        (row, row["annotation"])
        for row in box["rows"]
        if row["annotation"] and row["annotation"]["labeled_by"]
    ]
    aliases = [
        {
            "candidate_id": row["candidate_id"],
            "sample_id": row["sample_id"],
            "model_run_id": row["model_run_id"],
            "target_index": row["target_index"],
            "confidence": row["confidence"],
        }
        for row in box["rows"]
    ]
    base = {"aliases": aliases}
    if not human:
        return "unlabeled", base

    reasons = []
    labels = {ann["label"] for _, ann in human}
    if len(labels) > 1:
        reasons.append("label_conflict")
    suspect_hits = []
    for row, ann in human:
        hits = suspect_note(ann["label"], ann["note"])
        if hits:
            suspect_hits.append({"candidate_id": row["candidate_id"], "tokens": hits})
    if suspect_hits:
        reasons.append("note_label_suspect")
    if reasons:
        evidence = [
            {
                "candidate_id": row["candidate_id"],
                "sample_id": row["sample_id"],
                "label": row["annotation"]["label"],
                "note": row["annotation"]["note"],
                "reviewer": row["annotation"]["reviewer"],
                "reviewer_id": row["annotation"]["labeled_by"],
                "revision": row["annotation"]["revision"],
                "labeled_at": row["annotation"]["labeled_at"],
                "source": "human" if row["annotation"]["labeled_by"] else "auto_annotation",
            }
            for row in box["rows"]
            if row["annotation"]
        ]
        return "quarantine", {
            **base,
            "quarantine_reasons": reasons,
            "suspect_hits": suspect_hits,
            "evidence": evidence,
        }

    primary_row, primary_ann = human[0]  # rows pre-sorted by candidate_id
    return "gold", {
        **base,
        "label": primary_ann["label"],
        "note": primary_ann["note"],
        "reviewer": primary_ann["reviewer"],
        "reviewer_id": primary_ann["labeled_by"],
        "revision": primary_ann["revision"],
        "labeled_at": primary_ann["labeled_at"],
        "source": {
            "candidate_id": primary_row["candidate_id"],
            "sample_id": primary_row["sample_id"],
            "model_run_id": primary_row["model_run_id"],
            "source_event_id": primary_row["source_event_id"],
            "rule_code": primary_row["rule_code"],
            "provenance": source_record(primary_row),
        },
    }


def _attach_priors(box, gpt_priors, qwen_priors) -> None:
    primary = box["rows"][0]
    key = prior_key(box["image_object_key"], box["class_name"], primary["bbox_norm"])
    gpt = gpt_priors.get(key)
    qwen = qwen_priors.get(key)
    box["prior_gpt"] = {"passes": gpt} if gpt else None
    box["prior_qwen"] = {"attempts": qwen} if qwen else None
    box["prior_yolo"] = {
        "decision": primary["yolo_decision"],
        "raw_decision": primary["raw_decision"],
        "reason_code": primary["reason_code"],
        "confidence": primary["confidence"],
    }
    confidences = [row["confidence"] for row in box["rows"] if row["confidence"] is not None]
    box["detector_confidence"] = max(confidences) if confidences else None
    box["bbox_norm"] = primary["bbox_norm"]
