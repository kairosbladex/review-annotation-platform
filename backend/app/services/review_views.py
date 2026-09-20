"""Deterministic visual views used as auxiliary evidence during recheck."""

from __future__ import annotations

import hashlib
from io import BytesIO
from typing import Any, Mapping, Sequence

import numpy as np
from PIL import Image, ImageFilter


def _open_rgb(image_bytes: bytes) -> Image.Image:
    try:
        return Image.open(BytesIO(image_bytes)).convert("RGB")
    except Exception as exc:
        raise ValueError("image cannot be decoded") from exc


def _encode_jpeg(image: Image.Image, *, quality: int = 88) -> bytes:
    output = BytesIO()
    image.save(output, format="JPEG", quality=quality, optimize=True)
    return output.getvalue()


def normalized_bbox(candidate: Mapping[str, Any]) -> tuple[float, float, float, float] | None:
    value = candidate.get("bbox") if isinstance(candidate, Mapping) else None
    if not isinstance(value, (list, tuple)) or len(value) != 4:
        return None
    try:
        x1, y1, x2, y2 = (float(item) for item in value)
    except (TypeError, ValueError):
        return None
    if not (0.0 <= x1 < x2 <= 1.0 and 0.0 <= y1 < y2 <= 1.0):
        return None
    return x1, y1, x2, y2


def overlay_bytes(image_bytes: bytes, candidates: Sequence[Mapping[str, Any]]) -> bytes:
    """Draw numbered red candidate boxes without mutating the source bytes."""
    from PIL import ImageDraw

    image = _open_rgb(image_bytes)
    draw = ImageDraw.Draw(image)
    width, height = image.size
    line_width = max(2, min(width, height) // 300)
    for position, candidate in enumerate(candidates, start=1):
        bbox = normalized_bbox(candidate)
        if bbox is None:
            continue
        x1, y1, x2, y2 = bbox
        left = max(0, min(width - 1, round(x1 * width)))
        top = max(0, min(height - 1, round(y1 * height)))
        right = max(left + 1, min(width - 1, round(x2 * width)))
        bottom = max(top + 1, min(height - 1, round(y2 * height)))
        draw.rectangle((left, top, right, bottom), outline=(238, 56, 68), width=line_width)
        label = str(candidate.get("target_index", position - 1))
        text_top = max(0, top - 14)
        label_width = max(16, 7 * len(label) + 6)
        draw.rectangle((left, text_top, min(width - 1, left + label_width), top), fill=(238, 56, 68))
        draw.text((left + 3, text_top + 1), label, fill=(255, 255, 255))
    return _encode_jpeg(image)


def _red_mask(image: Image.Image) -> np.ndarray:
    array = np.asarray(image, dtype=np.uint8)
    red = array[:, :, 0].astype(np.int16)
    green = array[:, :, 1].astype(np.int16)
    blue = array[:, :, 2].astype(np.int16)
    # Burned-in review frames are saturated red and have a clear channel gap.
    return (red >= 145) & (red >= green + 45) & (red >= blue + 45)


def masked_bytes(image_bytes: bytes) -> bytes:
    """Mask saturated red review pixels as an auxiliary diagnostic view.

    This is deliberately not treated as a clean source image.  OpenCV
    inpainting is used when available; a median-filter fallback keeps local
    test environments usable without changing the original image.
    """
    image = _open_rgb(image_bytes)
    mask = _red_mask(image)
    if not bool(mask.any()):
        return _encode_jpeg(image)
    expanded = Image.fromarray((mask.astype(np.uint8) * 255), mode="L")
    expanded = expanded.filter(ImageFilter.MaxFilter(5))
    expanded_array = np.asarray(expanded) > 0
    try:
        import cv2

        source = np.asarray(image)[:, :, ::-1]
        repaired = cv2.inpaint(source, expanded_array.astype(np.uint8) * 255, 3, cv2.INPAINT_TELEA)
        return _encode_jpeg(Image.fromarray(repaired[:, :, ::-1]))
    except Exception:
        softened = image.filter(ImageFilter.MedianFilter(5))
        return _encode_jpeg(Image.composite(softened, image, expanded))


def roi_context_bytes(
    image_bytes: bytes,
    candidate: Mapping[str, Any],
    *,
    padding_ratio: float = 0.45,
    minimum_padding: float = 0.04,
) -> bytes:
    """Crop one candidate with surrounding context, without drawing a box."""
    image = _open_rgb(image_bytes)
    bbox = normalized_bbox(candidate)
    if bbox is None:
        raise ValueError("candidate bbox is not normalized")
    x1, y1, x2, y2 = bbox
    width, height = image.size
    box_width = x2 - x1
    box_height = y2 - y1
    pad_x = max(box_width * padding_ratio, minimum_padding)
    pad_y = max(box_height * padding_ratio, minimum_padding)
    left = max(0.0, x1 - pad_x)
    top = max(0.0, y1 - pad_y)
    right = min(1.0, x2 + pad_x)
    bottom = min(1.0, y2 + pad_y)
    pixel_box = (
        max(0, round(left * width)),
        max(0, round(top * height)),
        min(width, max(1, round(right * width))),
        min(height, max(1, round(bottom * height))),
    )
    if pixel_box[2] <= pixel_box[0] or pixel_box[3] <= pixel_box[1]:
        raise ValueError("candidate context crop is empty")
    return _encode_jpeg(image.crop(pixel_box), quality=90)


def build_visual_views(
    image_bytes: bytes,
    candidates: Sequence[Mapping[str, Any]],
) -> dict[str, bytes]:
    """Build auxiliary views while allowing invalid optional views to fail open."""
    views: dict[str, bytes] = {}
    try:
        views["overlay"] = overlay_bytes(image_bytes, candidates)
    except ValueError:
        return views
    try:
        views["masked"] = masked_bytes(image_bytes)
    except ValueError:
        pass
    for position, candidate in enumerate(candidates):
        target_index = candidate.get("target_index", position) if isinstance(candidate, Mapping) else position
        try:
            views[f"roi_{target_index}"] = roi_context_bytes(image_bytes, candidate)
        except ValueError:
            continue
    return views


def view_sha256(views: Mapping[str, bytes]) -> dict[str, str]:
    return {name: hashlib.sha256(value).hexdigest() for name, value in sorted(views.items())}


__all__ = [
    "build_visual_views",
    "masked_bytes",
    "normalized_bbox",
    "overlay_bytes",
    "roi_context_bytes",
    "view_sha256",
]
