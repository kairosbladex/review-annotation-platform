#!/usr/bin/env python3
"""Build a self-contained four-view Qwen verifier dataset from reviewed KTD pairs."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Sequence

import cv2
import numpy as np
from PIL import Image


VIEW_SCHEMA = "annotated_locator_plus_clean_original_v1"
VIEW_ROLES_4VIEW = (
    "annotated_context",
    "annotated_roi",
    "clean_context",
    "clean_roi",
)
VIEW_ROLES_2VIEW = (
    "annotated_context",
    "clean_roi",
)
VIEW_ROLES = VIEW_ROLES_4VIEW
VALID_VIEW_MODES = ("2view", "4view")
VALID_BBOX_STYLES = ("corner_ticks", "solid_box", "none")
NEUTRAL_GRAY_RGB = (230, 230, 230)
ELIGIBLE_MATCH_STATUSES = {"strong", "usable"}
PARTITION_OUTPUTS = {"train": "train.jsonl", "validation": "validation.jsonl"}
PROCESSOR_MIN_PIXELS = 28 * 28 * 64
PROCESSOR_MAX_PIXELS = 28 * 28 * 256
PROCESSOR_MIN_PIXELS_2VIEW = 28 * 28 * 32
PROCESSOR_MAX_PIXELS_2VIEW = 28 * 28 * 128
NORMALIZED_BBOX_MARGIN = 0.05

CATEGORY_ADAPTIVE_SPECS: dict[str, dict[str, Any]] = {
    "person": {
        "aliases": ("person", "pedestrian", "human", "人员", "行人", "人"),
        "bbox_margin": 0.20,
        "roi_scale": 5.0,
        "context_scale": 14.0,
        "min_context_scale": 5.0,
        "clamp_context_to_image": False,
        "clean_roi_upscale": "cv2.INTER_LANCZOS4",
    },
    "road_crack": {
        "aliases": ("road_crack", "crack", "pavement_crack", "道路裂缝", "裂缝", "路面裂缝"),
        "bbox_margin": 0.15,
        "roi_scale": 3.5,
        "context_scale": 10.0,
        "min_context_scale": 3.5,
        "clamp_context_to_image": False,
        "clean_roi_upscale": "cv2.INTER_CUBIC",
    },
    "blue_illegal_roof": {
        "aliases": (
            "blue_illegal_roof",
            "illegal_roof",
            "blue_roof",
            "蓝色违建顶棚",
            "违建",
            "蓝色顶棚",
            "蓝顶违建",
        ),
        "bbox_margin": 0.15,
        "roi_scale": 4.0,
        "context_scale": 12.0,
        "min_context_scale": 3.0,
        "clamp_context_to_image": True,
        "clean_roi_upscale": "cv2.INTER_LANCZOS4",
    },
    "tricycle": {
        "aliases": ("tricycle", "three_wheeler", "三轮车"),
        "bbox_margin": 0.12,
        "roi_scale": 4.0,
        "context_scale": 12.0,
        "min_context_scale": 4.0,
        "clamp_context_to_image": False,
        "clean_roi_upscale": "cv2.INTER_LANCZOS4",
    },
    "vehicle": {
        "aliases": ("vehicle", "car", "truck", "van", "bus", "车辆", "机动车", "车", "汽车"),
        "bbox_margin": 0.12,
        "roi_scale": 4.0,
        "context_scale": 12.0,
        "min_context_scale": 4.0,
        "clamp_context_to_image": False,
        "clean_roi_upscale": "cv2.INTER_LANCZOS4",
    },
}

DEFAULT_CATEGORY_SPEC: dict[str, Any] = {
    "canonical_category": "default",
    "bbox_margin": NORMALIZED_BBOX_MARGIN,
    "roi_scale": 4.0,
    "context_scale": 12.0,
    "min_context_scale": 3.0,
    "clamp_context_to_image": False,
    "clean_roi_upscale": "cv2.INTER_LANCZOS4",
}

CROP_SPEC = {
    "annotated_context": {"factor": 12.0, "min_side": 320, "output_size": 768},
    "annotated_roi": {"factor": 4.0, "min_side": 96, "output_size": 512},
    "clean_context": {"factor": 12.0, "min_side": 320, "output_size": 768},
    "clean_roi": {"factor": 4.0, "min_side": 96, "output_size": 512},
    "padding": "viewport_shift_then_cv2.BORDER_CONSTANT_rgb_230_230_230",
    "padding_color_rgb": list(NEUTRAL_GRAY_RGB),
    "bbox_style_default": "corner_ticks",
    "downscale": "cv2.INTER_AREA",
    "upscale": "cv2.INTER_LANCZOS4",
    "category_adaptive_specs": {
        key: {k: v for k, v in spec.items() if k != "aliases"}
        for key, spec in CATEGORY_ADAPTIVE_SPECS.items()
    },
}
PROMPT_TEMPLATE_4VIEW = (
    "<image>\n图1：带初审定位信息的上下文图。\n"
    "<image>\n图2：带初审定位信息的候选区域图。\n"
    "<image>\n图3：同一现场的清晰无标注原图上下文。\n"
    "<image>\n图4：从清晰原图投影得到的候选区域。\n"
    "初审类别：{class_name}。结合四张图判断红框所指候选是否确为该类别。"
    "只能输出一个 JSON 对象，不得输出解释或 Markdown："
    '{"decision":"CONFIRMED"} 或 {"decision":"REJECTED"}。'
)
PROMPT_TEMPLATE_2VIEW = (
    "<image>\n图1：带初审定位角标的全局上下文图。\n"
    "<image>\n图2：从清晰无标注原图裁剪得到的纯净候选区域细节图。\n"
    "初审类别：{class_name}。结合两张图判断定位角标所指候选是否确为该类别。"
    "只能输出一个 JSON 对象，不得输出解释或 Markdown："
    '{"decision":"CONFIRMED"} 或 {"decision":"REJECTED"}。'
)
PROMPT_TEMPLATE = PROMPT_TEMPLATE_4VIEW


def get_category_crop_spec(class_name: str | None) -> dict[str, Any]:
    """Resolve category-adaptive crop margin and scale configuration."""
    normalized = str(class_name or "").strip().lower()
    if not normalized:
        return dict(DEFAULT_CATEGORY_SPEC)
    for canonical, spec in CATEGORY_ADAPTIVE_SPECS.items():
        aliases = tuple(str(alias).lower() for alias in spec["aliases"])
        if normalized in aliases:
            return {
                "canonical_category": canonical,
                **{k: v for k, v in spec.items() if k != "aliases"},
            }
    for canonical, spec in CATEGORY_ADAPTIVE_SPECS.items():
        aliases = tuple(str(alias).lower() for alias in spec["aliases"])
        if any(alias in normalized for alias in aliases if len(alias) >= 2):
            return {
                "canonical_category": canonical,
                **{k: v for k, v in spec.items() if k != "aliases"},
            }
    return dict(DEFAULT_CATEGORY_SPEC)


def get_processor_pixel_budget(view_mode: str = "4view") -> dict[str, int]:
    """Return processor min/max pixel budget adjusted for 2view or 4view mode."""
    if view_mode == "2view":
        return {
            "min_pixels": PROCESSOR_MIN_PIXELS_2VIEW,
            "max_pixels": PROCESSOR_MAX_PIXELS_2VIEW,
            "max_sequence_length": 2048,
        }
    if view_mode == "4view":
        return {
            "min_pixels": PROCESSOR_MIN_PIXELS,
            "max_pixels": PROCESSOR_MAX_PIXELS,
            "max_sequence_length": 2048,
        }
    raise ValueError(f"unsupported view_mode: {view_mode!r}")


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


def _validated_size(value: Any, *, candidate_id: str, field: str) -> tuple[int, int]:
    if (
        not isinstance(value, (list, tuple))
        or len(value) != 2
        or isinstance(value[0], bool)
        or isinstance(value[1], bool)
    ):
        raise ValueError(f"candidate {candidate_id} has invalid {field}")
    try:
        width, height = int(value[0]), int(value[1])
    except (TypeError, ValueError) as exc:
        raise ValueError(f"candidate {candidate_id} has invalid {field}") from exc
    if width <= 0 or height <= 0:
        raise ValueError(f"candidate {candidate_id} has invalid {field}")
    return width, height


def _normalized_bbox(
    value: Any,
    *,
    image_size: tuple[int, int],
    candidate_id: str,
    field: str,
    class_name: str | None = None,
) -> tuple[float, float, float, float]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence) or len(value) != 4:
        raise ValueError(f"candidate {candidate_id} has invalid {field}")
    try:
        x1, y1, x2, y2 = (float(part) for part in value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"candidate {candidate_id} has invalid {field}") from exc
    if not all(np.isfinite(part) for part in (x1, y1, x2, y2)):
        raise ValueError(f"candidate {candidate_id} has invalid {field}")
    width, height = image_size
    normalized = (x1 / width, y1 / height, x2 / width, y2 / height)
    nx1, ny1, nx2, ny2 = normalized
    if not (nx1 < nx2 and ny1 < ny2):
        raise ValueError(f"candidate {candidate_id} has invalid {field}")
    margin = float(get_category_crop_spec(class_name)["bbox_margin"])
    if not (
        -margin <= nx1
        and -margin <= ny1
        and nx2 <= 1.0 + margin
        and ny2 <= 1.0 + margin
        and nx2 > 0.0
        and ny2 > 0.0
        and nx1 < 1.0
        and ny1 < 1.0
    ):
        raise ValueError(f"candidate {candidate_id} has invalid {field}")
    return normalized


def draw_bbox_overlay(
    image: np.ndarray | Image.Image,
    bbox_px: tuple[float, float, float, float],
    *,
    style: str = "corner_ticks",
    color: tuple[int, int, int] = (255, 0, 0),
    thickness: int = 2,
    expand_px: int = 2,
) -> np.ndarray:
    """Draw non-occluding L-shaped corner ticks or a solid box around bbox_px."""
    if style not in VALID_BBOX_STYLES:
        raise ValueError(f"unsupported bbox style: {style!r}")
    canvas = np.asarray(image.convert("RGB") if isinstance(image, Image.Image) else image).copy()
    if style == "none":
        return canvas
    height, width = canvas.shape[:2]
    x1, y1, x2, y2 = bbox_px
    if style == "solid_box":
        bx1 = int(np.clip(round(x1), 0, max(0, width - 1)))
        by1 = int(np.clip(round(y1), 0, max(0, height - 1)))
        bx2 = int(np.clip(round(x2), 0, max(0, width - 1)))
        by2 = int(np.clip(round(y2), 0, max(0, height - 1)))
        cv2.rectangle(canvas, (bx1, by1), (bx2, by2), color, thickness=thickness)
        return canvas

    # corner_ticks: expand 2px outward and draw 4 L-shaped corners (20% of side, clamped to [6, 24] px)
    bx1 = int(np.clip(round(x1) - expand_px, 0, max(0, width - 1)))
    by1 = int(np.clip(round(y1) - expand_px, 0, max(0, height - 1)))
    bx2 = int(np.clip(round(x2) + expand_px, 0, max(0, width - 1)))
    by2 = int(np.clip(round(y2) + expand_px, 0, max(0, height - 1)))
    bw = max(1, bx2 - bx1)
    bh = max(1, by2 - by1)
    tick_w = int(np.clip(round(bw * 0.20), 6, 24))
    tick_h = int(np.clip(round(bh * 0.20), 6, 24))
    # Prevent ticks from crossing opposite corners on very tiny boxes while keeping open mid-edges
    if bw > 2 * tick_w + 2:
        eff_w = tick_w
    else:
        eff_w = max(2, (bw - 2) // 2) if bw >= 6 else max(1, bw // 3)
    if bh > 2 * tick_h + 2:
        eff_h = tick_h
    else:
        eff_h = max(2, (bh - 2) // 2) if bh >= 6 else max(1, bh // 3)

    # Top-left L
    cv2.line(canvas, (bx1, by1), (min(width - 1, bx1 + eff_w), by1), color, thickness=thickness)
    cv2.line(canvas, (bx1, by1), (bx1, min(height - 1, by1 + eff_h)), color, thickness=thickness)
    # Top-right L
    cv2.line(canvas, (bx2, by1), (max(0, bx2 - eff_w), by1), color, thickness=thickness)
    cv2.line(canvas, (bx2, by1), (bx2, min(height - 1, by1 + eff_h)), color, thickness=thickness)
    # Bottom-left L
    cv2.line(canvas, (bx1, by2), (min(width - 1, bx1 + eff_w), by2), color, thickness=thickness)
    cv2.line(canvas, (bx1, by2), (bx1, max(0, by2 - eff_h)), color, thickness=thickness)
    # Bottom-right L
    cv2.line(canvas, (bx2, by2), (max(0, bx2 - eff_w), by2), color, thickness=thickness)
    cv2.line(canvas, (bx2, by2), (bx2, max(0, by2 - eff_h)), color, thickness=thickness)
    return canvas


def _crop_square_with_metadata(
    image: Image.Image,
    bbox: tuple[float, float, float, float],
    *,
    factor: float,
    min_side: int,
    output_size: int,
    class_name: str | None = None,
    role: str = "clean_roi",
    bbox_style: str = "none",
) -> tuple[Image.Image, dict[str, Any]]:
    array = np.asarray(image.convert("RGB"))
    width, height = image.size
    x1, y1, x2, y2 = (
        bbox[0] * width,
        bbox[1] * height,
        bbox[2] * width,
        bbox[3] * height,
    )
    box_w = max(1.0, x2 - x1)
    box_h = max(1.0, y2 - y1)
    center_x = (x1 + x2) / 2.0
    center_y = (y1 + y2) / 2.0

    cat_spec = get_category_crop_spec(class_name)
    effective_factor = float(factor)
    is_context = "context" in role
    if is_context and cat_spec.get("clamp_context_to_image"):
        min_ctx = float(cat_spec.get("min_context_scale", 3.0))
        effective_factor = max(effective_factor, min_ctx)
        min_required_side = max(box_w, box_h) * min_ctx
        max_image_side = float(max(width, height))
        desired_side = max(float(min_side), max(box_w, box_h) * effective_factor)
        side = min(max(desired_side, min_required_side), max_image_side)
    else:
        side = max(float(min_side), max(box_w, box_h) * effective_factor)

    side_int = max(8, int(round(side)))
    left = int(round(center_x - side_int / 2.0))
    top = int(round(center_y - side_int / 2.0))

    # 1. Prioritize shifting viewport within original image bounds while keeping target inside viewport
    if side_int <= width:
        if left < 0:
            left = 0
        elif left + side_int > width:
            left = width - side_int
    else:
        # Center image inside oversized viewport
        left = int(round((width - side_int) / 2.0))

    if side_int <= height:
        if top < 0:
            top = 0
        elif top + side_int > height:
            top = height - side_int
    else:
        top = int(round((height - side_int) / 2.0))

    # Ensure shifted viewport still contains target bbox if target is inside image
    if 0 <= x1 and x2 <= width and side_int >= box_w:
        if left > int(math.floor(x1)) if "math" in globals() else left > int(np.floor(x1)):
            left = max(0, int(np.floor(x1)))
        if left + side_int < int(np.ceil(x2)):
            left = min(max(0, width - side_int), int(np.ceil(x2)) - side_int)
    if 0 <= y1 and y2 <= height and side_int >= box_h:
        if top > int(np.floor(y1)):
            top = max(0, int(np.floor(y1)))
        if top + side_int < int(np.ceil(y2)):
            top = min(max(0, height - side_int), int(np.ceil(y2)) - side_int)

    right = left + side_int
    bottom = top + side_int
    pad_top = max(0, -top)
    pad_left = max(0, -left)
    pad_bottom = max(0, bottom - array.shape[0])
    pad_right = max(0, right - array.shape[1])
    crop = array[
        max(0, top):min(array.shape[0], bottom),
        max(0, left):min(array.shape[1], right),
    ]
    if crop.size == 0:
        raise ValueError("generated view crop is empty")
    if any((pad_top, pad_left, pad_bottom, pad_right)):
        crop = cv2.copyMakeBorder(
            crop,
            pad_top,
            pad_bottom,
            pad_left,
            pad_right,
            cv2.BORDER_CONSTANT,
            value=NEUTRAL_GRAY_RGB,
        )

    upscale_method = (
        cv2.INTER_CUBIC
        if role == "clean_roi" and cat_spec.get("clean_roi_upscale") == "cv2.INTER_CUBIC"
        else cv2.INTER_LANCZOS4
    )
    interpolation = upscale_method if max(crop.shape[:2]) < output_size else cv2.INTER_AREA
    resized = cv2.resize(crop, (output_size, output_size), interpolation=interpolation)

    scale = output_size / float(side_int)
    shifted_bbox = (
        float(np.clip((x1 - left) * scale, 0.0, float(output_size))),
        float(np.clip((y1 - top) * scale, 0.0, float(output_size))),
        float(np.clip((x2 - left) * scale, 0.0, float(output_size))),
        float(np.clip((y2 - top) * scale, 0.0, float(output_size))),
    )
    if bbox_style != "none" and role.startswith("annotated_"):
        resized = draw_bbox_overlay(resized, shifted_bbox, style=bbox_style)

    meta = {
        "viewport": [left, top, right, bottom],
        "padding": [pad_top, pad_bottom, pad_left, pad_right],
        "shifted_bbox_px": [round(v, 2) for v in shifted_bbox],
        "effective_factor": effective_factor,
    }
    return Image.fromarray(resized), meta


def _crop_square(
    image: Image.Image,
    bbox: tuple[float, float, float, float],
    *,
    factor: float,
    min_side: int,
    output_size: int,
    class_name: str | None = None,
    role: str = "clean_roi",
    bbox_style: str = "none",
) -> Image.Image:
    cropped, _ = _crop_square_with_metadata(
        image,
        bbox,
        factor=factor,
        min_side=min_side,
        output_size=output_size,
        class_name=class_name,
        role=role,
        bbox_style=bbox_style,
    )
    return cropped


def _copy_source_frame(
    *,
    source: Path,
    output_dir: Path,
    kind: str,
    expected_size: tuple[int, int],
    candidate_id: str,
) -> tuple[str, str]:
    source = source.resolve()
    if not source.is_file():
        raise ValueError(f"candidate {candidate_id} {kind} frame is unavailable: {source}")
    with Image.open(source) as image:
        if image.size != expected_size:
            raise ValueError(f"candidate {candidate_id} {kind} frame size mismatch")
        image.verify()
    digest = _sha256(source)
    suffix = source.suffix.lower() or ".jpg"
    destination = output_dir / "source_frames" / kind / f"{digest}{suffix}"
    destination.parent.mkdir(parents=True, exist_ok=True)
    if not destination.exists():
        shutil.copy2(source, destination)
    if _sha256(destination) != digest:
        raise RuntimeError(f"copied {kind}-frame SHA-256 mismatch for candidate {candidate_id}")
    return str(destination.relative_to(output_dir)), digest


def _save_generated_views(
    *,
    annotated_source: Path,
    clean_source: Path,
    annotated_bbox: tuple[float, float, float, float],
    clean_bbox: tuple[float, float, float, float],
    output_dir: Path,
    candidate_id: str,
    class_name: str | None = None,
    view_mode: str = "4view",
    bbox_style: str = "corner_ticks",
) -> tuple[list[str], dict[str, str], dict[str, Any]]:
    if view_mode not in VALID_VIEW_MODES:
        raise ValueError(f"unsupported view_mode: {view_mode!r}")
    if bbox_style not in VALID_BBOX_STYLES:
        raise ValueError(f"unsupported bbox_style: {bbox_style!r}")
    cat_spec = get_category_crop_spec(class_name)
    context_scale = float(cat_spec["context_scale"])
    roi_scale = float(cat_spec["roi_scale"])
    active_roles = VIEW_ROLES_2VIEW if view_mode == "2view" else VIEW_ROLES_4VIEW

    role_configs = {
        "annotated_context": (annotated_source, annotated_bbox, context_scale, 320, 768, bbox_style),
        "annotated_roi": (annotated_source, annotated_bbox, roi_scale, 96, 512, bbox_style),
        "clean_context": (clean_source, clean_bbox, context_scale, 320, 768, "none"),
        "clean_roi": (clean_source, clean_bbox, roi_scale, 96, 512, "none"),
    }

    image_paths: list[str] = []
    view_shas: dict[str, str] = {}
    view_crop_meta: dict[str, Any] = {}
    with Image.open(annotated_source) as annotated_image, Image.open(clean_source) as clean_image:
        source_map = {
            annotated_source: annotated_image,
            clean_source: clean_image,
        }
        for role in active_roles:
            src_path, bbox, factor, min_side, out_size, style = role_configs[role]
            view, meta = _crop_square_with_metadata(
                source_map[src_path],
                bbox,
                factor=factor,
                min_side=min_side,
                output_size=out_size,
                class_name=class_name,
                role=role,
                bbox_style=style,
            )
            destination = output_dir / "media" / role / f"{candidate_id}.png"
            destination.parent.mkdir(parents=True, exist_ok=True)
            view.save(destination, format="PNG")
            relative = str(destination.relative_to(output_dir))
            image_paths.append(relative)
            view_shas[role] = _sha256(destination)
            view_crop_meta[role] = meta
    return image_paths, view_shas, view_crop_meta


def _training_prompt(class_name: str, *, view_mode: str = "4view") -> str:
    template = PROMPT_TEMPLATE_2VIEW if view_mode == "2view" else PROMPT_TEMPLATE_4VIEW
    return template.replace("{class_name}", class_name)


def build_dataset(
    *,
    candidate_pairs_path: Path,
    gold_path: Path,
    output_dir: Path,
    view_mode: str = "4view",
    bbox_style: str = "corner_ticks",
) -> dict[str, Any]:
    if view_mode not in VALID_VIEW_MODES:
        raise ValueError(f"unsupported view_mode: {view_mode!r}")
    if bbox_style not in VALID_BBOX_STYLES:
        raise ValueError(f"unsupported bbox_style: {bbox_style!r}")
    active_roles = VIEW_ROLES_2VIEW if view_mode == "2view" else VIEW_ROLES_4VIEW
    active_prompt_template = PROMPT_TEMPLATE_2VIEW if view_mode == "2view" else PROMPT_TEMPLATE_4VIEW
    processor_budget = get_processor_pixel_budget(view_mode)

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

        annotated_size = _validated_size(
            pair.get("annotated_image_size"), candidate_id=candidate_id, field="annotated_image_size"
        )
        clean_size = _validated_size(
            pair.get("clean_frame_size"), candidate_id=candidate_id, field="clean_frame_size"
        )
        # 旧数据按事件共用截图；新增样本可在 pair 中显式给出自己的截图路径
        annotated_relative = pair.get("annotated_frame") or str(Path("event_annotated_frames") / f"{pair['event_id']}.jpg")
        annotated_source = source_root / annotated_relative
        clean_source_value = pair.get("clean_frame")
        if not isinstance(clean_source_value, str) or not clean_source_value:
            raise ValueError(f"candidate {candidate_id} is missing clean_frame")
        clean_source = source_root / clean_source_value
        annotated_frame_path, annotated_frame_sha256 = _copy_source_frame(
            source=annotated_source,
            output_dir=output_dir,
            kind="annotated",
            expected_size=annotated_size,
            candidate_id=candidate_id,
        )
        clean_frame_path, clean_frame_sha256 = _copy_source_frame(
            source=clean_source,
            output_dir=output_dir,
            kind="clean",
            expected_size=clean_size,
            candidate_id=candidate_id,
        )
        class_name_str = str(pair["class_name"])
        cat_spec = get_category_crop_spec(class_name_str)
        normalized_review_bbox = _normalized_bbox(
            pair.get("bbox_px"),
            image_size=annotated_size,
            candidate_id=candidate_id,
            field="bbox_px",
            class_name=class_name_str,
        )
        normalized_clean_bbox = _normalized_bbox(
            pair.get("projected_clean_bbox_px"),
            image_size=clean_size,
            candidate_id=candidate_id,
            field="projected_clean_bbox_px",
            class_name=class_name_str,
        )
        image_paths, view_shas, view_crop_meta = _save_generated_views(
            annotated_source=annotated_source,
            clean_source=clean_source,
            annotated_bbox=normalized_review_bbox,
            clean_bbox=normalized_clean_bbox,
            output_dir=output_dir,
            candidate_id=candidate_id,
            class_name=class_name_str,
            view_mode=view_mode,
            bbox_style=bbox_style,
        )
        decision = "CONFIRMED" if gold_row["human_label"] == "true" else "REJECTED"
        prompt = _training_prompt(class_name_str, view_mode=view_mode)
        prompt_template_sha256 = hashlib.sha256(active_prompt_template.encode("utf-8")).hexdigest()
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
                "view_mode": view_mode,
                "bbox_style": bbox_style,
                "view_roles": list(active_roles),
                "view_sha256": view_shas,
                "view_crop_meta": view_crop_meta,
                "category_crop_spec": cat_spec,
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
                "image_object_key": annotated_relative,
                "source_image": image_paths[0],
                "view_generation": "service_crop_v1_lossless_png",
                "annotated_frame": annotated_frame_path,
                "annotated_frame_sha256": annotated_frame_sha256,
                "annotated_image_size": list(annotated_size),
                "clean_frame_size": list(clean_size),
                "normalized_review_bbox": list(normalized_review_bbox),
                "normalized_clean_bbox": list(normalized_clean_bbox),
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
            view_mode=view_mode,
            bbox_style=bbox_style,
            views=[{"role": role, "path": path, "sha256": view_shas[role]} for role, path in zip(active_roles, image_paths)],
            annotated_frame=annotated_frame_path,
            annotated_frame_sha256=annotated_frame_sha256,
            clean_frame=clean_frame_path,
            clean_frame_sha256=clean_frame_sha256,
            view_generation="service_crop_v1_lossless_png",
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
        "annotated_frame_sha256_overlap": sorted(
            {row["meta"]["annotated_frame_sha256"] for row in train}
            & {row["meta"]["annotated_frame_sha256"] for row in validation}
        ),
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
        "view_mode": view_mode,
        "bbox_style": bbox_style,
        "deployment_contract": {
            "view_schema": VIEW_SCHEMA,
            "view_mode": view_mode,
            "bbox_style": bbox_style,
            "view_roles": list(active_roles),
            "prompt_template": active_prompt_template,
            "crop_spec": CROP_SPEC,
            "view_generation": "service_crop_v1_lossless_png",
            "normalized_bbox_margin": NORMALIZED_BBOX_MARGIN,
            "prompt_template_sha256": hashlib.sha256(active_prompt_template.encode("utf-8")).hexdigest(),
            "strict_output": {
                "json_object_keys": ["decision"],
                "allowed_decisions": ["CONFIRMED", "REJECTED"],
                "invalid_output": "uncertain",
            },
            "processor": processor_budget,
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
        "unique_annotated_frames": len({row["meta"]["annotated_frame_sha256"] for row in train + validation}),
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
    parser.add_argument(
        "--view-mode",
        choices=VALID_VIEW_MODES,
        default="2view",
        help="Multi-view mode: 2view (annotated_context + clean_roi, default) or 4view (legacy compatibility).",
    )
    parser.add_argument(
        "--bbox-style",
        choices=VALID_BBOX_STYLES,
        default="corner_ticks",
        help="Bounding box drawing style on annotated views: corner_ticks (default), solid_box, or none.",
    )
    args = parser.parse_args()
    summary = build_dataset(
        candidate_pairs_path=args.candidate_pairs,
        gold_path=args.gold,
        output_dir=args.output_dir,
        view_mode=args.view_mode,
        bbox_style=args.bbox_style,
    )
    print(json.dumps(summary, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
