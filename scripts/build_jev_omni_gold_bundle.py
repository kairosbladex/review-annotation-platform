#!/usr/bin/env python3
"""把 142 条真实人工金标打包成 Jev-Omni 真机推理所需的双视角合图 + 决策清单。

产物（默认 ``/tmp/jev_omni_gold_eval_bundle``）：

* ``composites/<candidate_id>.jpg``：1024×512 双视角合图（左 annotated_context、
  右 clean_roi，8px 中性灰分隔条），由
  ``build_jev_omni_review_views`` + ``build_jev_omni_composite_image`` 生成；
* ``manifest.jsonl``：每行一条样本的 ``(state, question, options)`` 与元数据，
  远端推理脚本只依赖本文件即可跑，不需要引入后端代码。

视图来源有两档，逐样本记录在 ``view_source``：

1. ``review_views_from_original_frame``：``event_annotated_frames/<event_id>.jpg``
   存在且分辨率与 ``inference_manifest.jsonl`` 的 ``frame_size`` **完全一致**时，
   用原始检测帧 + 归一化 bbox 现场重建两个视角 —— 与线上
   ``auto_annotation_service._process_item`` 的调用路径逐像素一致；
2. ``materialized_views``：原始帧缺失时，回退到数据集里已经落盘的
   ``annotated_context`` / ``clean_roi``（同一套裁剪管线的产物），只做合图。

用法：
    PYTHONPATH=.:backend .venv/bin/python scripts/build_jev_omni_gold_bundle.py \
        --output-dir /tmp/jev_omni_gold_eval_bundle
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))

from PIL import Image  # noqa: E402

from app.adapters.jev_omni_classifier import (  # noqa: E402
    DEFAULT_CANVAS_WIDTH,
    DEFAULT_TARGET_HEIGHT,
    build_jev_omni_composite_image,
    build_jev_omni_decision_query,
    build_jev_omni_review_views,
)
from app.services.ktd_recheck import apply_pre_llm_geometric_gate  # noqa: E402

DEFAULT_GOLD_ROOT = ROOT / "data/ktd-clean-source-full-evaluation-20260924"
DEFAULT_OVERLAP = ROOT / "data/ktd-test-real-monitoring-full-20260924/gold_overlap_map.jsonl"
DEFAULT_MANIFEST = ROOT / "data/ktd-test-real-monitoring-full-20260924/input/inference_manifest.jsonl"

VIEW_SOURCE_REBUILT = "review_views_from_original_frame"
VIEW_SOURCE_MATERIALIZED = "materialized_views"
VIEW_SOURCE_ANNOTATED_ROI = "materialized_views_annotated_roi_fallback"


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _safe_name(candidate_id: str) -> str:
    return candidate_id.replace(":", "_").replace("/", "_")


def _normalized_bbox(bbox_px: list[float], frame_size: list[int]) -> list[float]:
    width, height = float(frame_size[0]), float(frame_size[1])
    x1, y1, x2, y2 = (float(v) for v in bbox_px)
    return [x1 / width, y1 / height, x2 / width, y2 / height]


def _geometric_warnings(rows: list[dict[str, Any]]) -> dict[str, bool | None]:
    """按事件跑一次几何门禁，取回每个候选的 ``geometric_warning``（与线上同口径）。"""
    by_event: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_event[str(row.get("event_id"))].append(row)
    warnings: dict[str, bool | None] = {}
    for event_rows in by_event.values():
        gated = apply_pre_llm_geometric_gate(
            [
                {
                    "target_index": row["candidate_id"],
                    "class_name": row.get("class_name"),
                    "confidence": row.get("detector_confidence"),
                    "bbox": row.get("bbox_px"),
                }
                for row in event_rows
            ]
        )
        for candidate, source in zip(gated, event_rows):
            warnings[str(source["candidate_id"])] = candidate.get("geometric_warning")
    return warnings


def build_bundle(
    *,
    output_dir: Path,
    gold_root: Path,
    overlap_path: Path,
    manifest_path: Path,
    jpeg_quality: int = 92,
    roi_profile: str = "legacy",
    prompt_variant: str = "legacy",
) -> dict[str, Any]:
    manifest = {str(row["candidate_id"]): row for row in _read_jsonl(manifest_path)}
    overlap = _read_jsonl(overlap_path)
    frames_dir = gold_root / "event_annotated_frames"

    resolved: list[dict[str, Any]] = []
    missing = 0
    for item in overlap:
        source = manifest.get(str(item.get("full_candidate_id")))
        if source is None:
            missing += 1
            continue
        resolved.append({**source, **{k: item[k] for k in ("legacy_candidate_id", "human_label") if k in item}})

    warnings = _geometric_warnings(resolved)

    composites_dir = output_dir / "composites"
    composites_dir.mkdir(parents=True, exist_ok=True)

    records: list[dict[str, Any]] = []
    counts: dict[str, int] = {VIEW_SOURCE_REBUILT: 0, VIEW_SOURCE_MATERIALIZED: 0, VIEW_SOURCE_ANNOTATED_ROI: 0}
    frame_skipped: list[str] = []

    for row in resolved:
        candidate_id = str(row["candidate_id"])
        legacy_id = str(row["legacy_candidate_id"])
        frame_path = frames_dir / f"{row['event_id']}.jpg"
        view_source = VIEW_SOURCE_MATERIALIZED
        context_img = roi_img = None

        if frame_path.exists():
            with Image.open(frame_path) as frame:
                frame = frame.convert("RGB")
                if list(frame.size) == list(row["frame_size"]):
                    context_img, roi_img = build_jev_omni_review_views(
                        frame,
                        _normalized_bbox(row["bbox_px"], row["frame_size"]),
                        class_name=row.get("class_name"),
                        roi_profile=roi_profile,
                    )
                    view_source = VIEW_SOURCE_REBUILT
                else:
                    frame_skipped.append(f"{candidate_id}:{frame.size}!={row['frame_size']}")
        else:
            frame_skipped.append(f"{candidate_id}:frame-missing")

        if context_img is None:
            context_path = gold_root / "annotated_context" / f"{legacy_id}.jpg"
            roi_path = gold_root / "clean_roi" / f"{legacy_id}.jpg"
            if not roi_path.exists():
                # 16 条人员样本 `has_clean_source=false`：没有 clean_roi，只有带角标的
                # annotated_roi，仍能提供候选框内的局部视图。
                roi_path = gold_root / "annotated_roi" / f"{legacy_id}.jpg"
                view_source = VIEW_SOURCE_ANNOTATED_ROI
            if not context_path.exists() or not roi_path.exists():
                raise FileNotFoundError(f"候选 {candidate_id} 既无原始帧也无落盘视角")
            context_img = Image.open(context_path).convert("RGB")
            roi_img = Image.open(roi_path).convert("RGB")

        composite = build_jev_omni_composite_image(context_img, roi_img)
        if composite.size != (DEFAULT_CANVAS_WIDTH, DEFAULT_TARGET_HEIGHT):
            raise ValueError(f"合图画幅异常：{candidate_id} -> {composite.size}")

        target = composites_dir / f"{_safe_name(candidate_id)}.jpg"
        composite.save(target, format="JPEG", quality=jpeg_quality, optimize=True)
        digest = hashlib.sha256(target.read_bytes()).hexdigest()

        meta = {
            "candidate_id": candidate_id,
            "class_name": row.get("class_name"),
            "detector_confidence": row.get("detector_confidence"),
            "geometric_warning": warnings.get(candidate_id),
        }
        query = build_jev_omni_decision_query(
            row.get("class_name"),
            meta,
            prompt_variant=prompt_variant,
        )

        records.append(
            {
                "candidate_id": candidate_id,
                "legacy_candidate_id": legacy_id,
                "event_id": row.get("event_id"),
                "class_name": row.get("class_name"),
                "human_label": row.get("human_label"),
                "detector_confidence": row.get("detector_confidence"),
                "geometric_warning": warnings.get(candidate_id),
                "view_source": view_source,
                "composite": str(target.relative_to(output_dir)),
                "composite_sha256": digest,
                "composite_size": list(composite.size),
                "state": query["state"],
                "question": query["question"],
                "options": query["options"],
            }
        )
        counts[view_source] += 1

    manifest_path_out = output_dir / "manifest.jsonl"
    with manifest_path_out.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    summary = {
        "gold_source": str(overlap_path.relative_to(ROOT)),
        "detector_score_source": str(manifest_path.relative_to(ROOT)),
        "samples": len(records),
        "unmatched": missing,
        "view_source": counts,
        "frame_fallback_reasons": frame_skipped,
        "canvas": [DEFAULT_CANVAS_WIDTH, DEFAULT_TARGET_HEIGHT],
        "class_counts": {
            cls: sum(1 for r in records if r["class_name"] == cls)
            for cls in sorted({str(r["class_name"]) for r in records})
        },
        "label_counts": {
            label: sum(1 for r in records if r["human_label"] == label)
            for label in sorted({str(r["human_label"]) for r in records})
        },
    }
    (output_dir / "bundle_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--gold-root", type=Path, default=DEFAULT_GOLD_ROOT)
    parser.add_argument("--gold-overlap", type=Path, default=DEFAULT_OVERLAP)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--roi-profile", choices=("legacy", "adaptive"), default="legacy")
    parser.add_argument("--prompt-variant", choices=("legacy", "category_guidelines"), default="legacy")
    args = parser.parse_args()

    summary = build_bundle(
        output_dir=args.output_dir,
        gold_root=args.gold_root,
        overlap_path=args.gold_overlap,
        manifest_path=args.manifest,
        roi_profile=args.roi_profile,
        prompt_variant=args.prompt_variant,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
