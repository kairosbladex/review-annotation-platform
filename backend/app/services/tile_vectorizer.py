"""Offline tiling and patch embedding extraction.

The module deliberately keeps model loading out of the indexing path.  A caller
provides an encoder implementing :class:`FeatureEncoder`; this makes the same
pipeline usable with RemoteCLIP/DINO based services in production and with a
small deterministic encoder in an offline test.  The output records contain
only JSON-safe metadata and a normalised vector, so they can be passed directly
to a vector-store adapter.
"""

from __future__ import annotations

import hashlib
import io
import math
from dataclasses import dataclass, field
from datetime import date, datetime
from pathlib import Path
from typing import Any, Iterable, Mapping, Protocol, Sequence

import numpy as np
from PIL import Image


class TileVectorizationError(ValueError):
    """Raised when an image, tile configuration, or encoder output is invalid."""


def _as_rgb_image(source: Any) -> Image.Image:
    """Decode the supported image inputs without leaking open file handles."""

    if isinstance(source, Image.Image):
        return source.convert("RGB")
    if isinstance(source, (bytes, bytearray, memoryview)):
        try:
            with Image.open(io.BytesIO(bytes(source))) as image:
                return image.convert("RGB")
        except Exception as exc:  # pragma: no cover - Pillow supplies varied errors
            raise TileVectorizationError("无法解码图像 bytes") from exc
    if isinstance(source, (str, Path)):
        try:
            with Image.open(source) as image:
                return image.convert("RGB")
        except Exception as exc:  # pragma: no cover - Pillow supplies varied errors
            raise TileVectorizationError(f"无法读取图像: {source}") from exc
    read = getattr(source, "read", None)
    if callable(read):
        return _as_rgb_image(read())
    raise TileVectorizationError("图像输入必须是 PIL.Image、bytes、路径或可读文件对象")


def _json_safe(value: Any) -> Any:
    """Convert common metadata values to a stable JSON-safe representation."""

    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise TileVectorizationError("元数据不能包含 NaN 或无穷值")
        return value
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    if isinstance(value, np.generic):
        return _json_safe(value.item())
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    raise TileVectorizationError(f"不支持的元数据类型: {type(value).__name__}")


@dataclass(frozen=True, slots=True)
class TileGridConfig:
    """Regular grid configuration.

    Coordinates use half-open pixel boxes ``[x1, y1, x2, y2)``.  The final
    start position on each axis is shifted to the image edge so a large image
    has complete edge tiles rather than a narrow uncovered strip.
    """

    tile_size: int = 512
    overlap: float = 0.20

    def __post_init__(self) -> None:
        if isinstance(self.tile_size, bool) or not isinstance(self.tile_size, int) or self.tile_size <= 0:
            raise ValueError("tile_size 必须是正整数")
        if not isinstance(self.overlap, (int, float)) or not math.isfinite(float(self.overlap)):
            raise ValueError("overlap 必须是有限数")
        if not 0 <= float(self.overlap) < 1:
            raise ValueError("overlap 必须满足 0 <= overlap < 1")

    @property
    def stride(self) -> int:
        return max(1, round(self.tile_size * (1.0 - float(self.overlap))))


@dataclass(frozen=True, slots=True)
class TileRecord:
    """A source tile and its location in the original image."""

    image_id: str
    tile_index: int
    bbox_on_orig: tuple[int, int, int, int]
    image: Image.Image = field(repr=False, compare=False)
    capture_time: str | None = None
    resolution: Any = None

    def __post_init__(self) -> None:
        if not self.image_id:
            raise ValueError("image_id 不能为空")
        if self.tile_index < 0:
            raise ValueError("tile_index 不能为负数")
        if len(self.bbox_on_orig) != 4:
            raise ValueError("bbox_on_orig 必须包含四个坐标")
        x1, y1, x2, y2 = self.bbox_on_orig
        if not (0 <= x1 < x2 and 0 <= y1 < y2):
            raise ValueError("bbox_on_orig 必须是合法半开像素框")
        if self.image.size != (x2 - x1, y2 - y1):
            raise ValueError("tile image 尺寸必须与 bbox_on_orig 一致")

    @property
    def record_id(self) -> str:
        return f"{self.image_id}:{self.tile_index}"

    def payload(self) -> dict[str, Any]:
        return {
            "image_id": self.image_id,
            "tile_index": self.tile_index,
            "bbox_on_orig": list(self.bbox_on_orig),
            "capture_time": self.capture_time,
            "resolution": _json_safe(self.resolution),
            "tile_size": [self.image.width, self.image.height],
        }


@dataclass(frozen=True, slots=True)
class TileVectorRecord:
    """A normalised embedding plus the source tile metadata."""

    id: str
    vector: tuple[float, ...]
    payload: Mapping[str, Any]

    @property
    def dimension(self) -> int:
        return len(self.vector)


class FeatureEncoder(Protocol):
    """Small seam shared by RemoteCLIP/DINO services and test encoders."""

    dimension: int

    def encode_images(self, images: Sequence[Image.Image]) -> Any:
        ...

    def encode_text(self, text: str) -> Any:
        ...


def _normalise_vectors(value: Any, *, expected_rows: int, expected_dimension: int | None = None) -> np.ndarray:
    array = np.asarray(value, dtype=np.float32)
    if array.ndim == 1:
        if expected_rows != 1:
            raise TileVectorizationError("encoder 返回的向量数量与 batch 不一致")
        array = array.reshape(1, -1)
    if array.ndim != 2 or array.shape[0] != expected_rows or array.shape[1] == 0:
        raise TileVectorizationError("encoder 必须返回形状为 [batch, dimension] 的二维向量")
    if expected_dimension is not None and array.shape[1] != expected_dimension:
        raise TileVectorizationError(
            f"encoder 维度不匹配: expected={expected_dimension}, actual={array.shape[1]}"
        )
    if not np.isfinite(array).all():
        raise TileVectorizationError("encoder 返回了 NaN 或无穷向量")
    norms = np.linalg.norm(array, axis=1, keepdims=True)
    if np.any(norms <= 1e-12):
        raise TileVectorizationError("encoder 返回了零向量")
    return array / norms


class TileVectorizer:
    """Generate regular tiles and batch-encode them."""

    def __init__(self, encoder: FeatureEncoder, config: TileGridConfig | None = None, *, batch_size: int = 32):
        if not callable(getattr(encoder, "encode_images", None)) and not callable(
            getattr(encoder, "encode_image", None)
        ):
            raise TypeError("encoder 必须提供 encode_images 或 encode_image")
        if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size <= 0:
            raise ValueError("batch_size 必须是正整数")
        self.encoder = encoder
        self.config = config or TileGridConfig()
        self.batch_size = batch_size
        dimension = getattr(encoder, "dimension", None)
        if dimension is not None and (not isinstance(dimension, int) or dimension <= 0):
            raise ValueError("encoder.dimension 必须是正整数")
        self.dimension = dimension

    def iter_tiles(
        self,
        source: Any,
        *,
        image_id: str,
        capture_time: str | datetime | date | None = None,
        resolution: Any = None,
    ) -> Iterable[TileRecord]:
        image = _as_rgb_image(source)
        if not image_id:
            raise ValueError("image_id 不能为空")
        capture = capture_time.isoformat() if isinstance(capture_time, (datetime, date)) else capture_time
        width, height = image.size
        x_starts = self._axis_starts(width)
        y_starts = self._axis_starts(height)
        tile_index = 0
        for y1 in y_starts:
            for x1 in x_starts:
                x2, y2 = min(width, x1 + self.config.tile_size), min(height, y1 + self.config.tile_size)
                tile = image.crop((x1, y1, x2, y2)).copy()
                yield TileRecord(
                    image_id=image_id,
                    tile_index=tile_index,
                    bbox_on_orig=(x1, y1, x2, y2),
                    image=tile,
                    capture_time=capture,
                    resolution=_json_safe(resolution),
                )
                tile_index += 1

    def vectorize(
        self,
        source: Any,
        *,
        image_id: str,
        capture_time: str | datetime | date | None = None,
        resolution: Any = None,
    ) -> list[TileVectorRecord]:
        tiles = list(
            self.iter_tiles(
                source,
                image_id=image_id,
                capture_time=capture_time,
                resolution=resolution,
            )
        )
        records: list[TileVectorRecord] = []
        for offset in range(0, len(tiles), self.batch_size):
            batch = tiles[offset : offset + self.batch_size]
            vectors = self._encode_images([tile.image for tile in batch])
            for tile, vector in zip(batch, vectors, strict=True):
                records.append(
                    TileVectorRecord(
                        id=tile.record_id,
                        vector=tuple(float(item) for item in vector),
                        payload=tile.payload(),
                    )
                )
        return records

    def vectorize_many(self, sources: Iterable[tuple[Any, str]], **metadata: Any) -> list[TileVectorRecord]:
        records: list[TileVectorRecord] = []
        for source, image_id in sources:
            records.extend(self.vectorize(source, image_id=image_id, **metadata))
        return records

    # Explicit aliases keep the public vocabulary close to the indexing
    # requirement while retaining the iterator/array distinction above.
    def extract_tiles(self, source: Any, **metadata: Any) -> Iterable[TileRecord]:
        return self.iter_tiles(source, **metadata)

    def build_index(self, source: Any, *, image_id: str, vector_store: Any, **metadata: Any) -> int:
        records = self.vectorize(source, image_id=image_id, **metadata)
        upsert = getattr(vector_store, "upsert", None)
        if not callable(upsert):
            raise TypeError("vector_store 必须提供 upsert")
        return int(upsert(records))

    def _axis_starts(self, length: int) -> list[int]:
        size = self.config.tile_size
        if length <= size:
            return [0]
        maximum = length - size
        starts = list(range(0, maximum + 1, self.config.stride))
        if starts[-1] != maximum:
            starts.append(maximum)
        return starts

    def _encode_images(self, images: Sequence[Image.Image]) -> np.ndarray:
        batch_encoder = getattr(self.encoder, "encode_images", None)
        if callable(batch_encoder):
            raw = batch_encoder(images)
        else:
            single_encoder = getattr(self.encoder, "encode_image")
            raw = [single_encoder(image) for image in images]
        return _normalise_vectors(raw, expected_rows=len(images), expected_dimension=self.dimension)


class HashFeatureEncoder:
    """Deterministic CPU fallback for smoke tests and local index dry-runs.

    It is intentionally not presented as a semantic model.  Production callers
    should inject a RemoteCLIP or another calibrated cross-modal encoder.
    """

    def __init__(self, dimension: int = 512):
        if isinstance(dimension, bool) or not isinstance(dimension, int) or dimension <= 0:
            raise ValueError("dimension 必须是正整数")
        self.dimension = dimension

    def _digest_vector(self, payload: bytes) -> np.ndarray:
        output = bytearray()
        counter = 0
        while len(output) < self.dimension * 4:
            output.extend(hashlib.sha256(payload + counter.to_bytes(4, "big")).digest())
            counter += 1
        values = np.frombuffer(bytes(output[: self.dimension * 4]), dtype=np.uint32).astype(np.float32)
        return values / np.float32(2**32)

    def encode_images(self, images: Sequence[Image.Image]) -> np.ndarray:
        rows: list[np.ndarray] = []
        for image in images:
            rgb = _as_rgb_image(image).resize((32, 32))
            pixels = np.asarray(rgb, dtype=np.uint8)
            stats = np.concatenate([pixels.mean(axis=(0, 1)), pixels.std(axis=(0, 1))]).astype(np.float32)
            digest = self._digest_vector(pixels.tobytes())
            digest[: min(len(stats), self.dimension)] += stats[: self.dimension]
            rows.append(digest)
        return np.stack(rows)

    def encode_text(self, text: str) -> np.ndarray:
        if not isinstance(text, str) or not text.strip():
            raise ValueError("text query 不能为空")
        return self._digest_vector(text.strip().encode("utf-8"))


class RemoteCLIPEncoder:
    """Lazy HuggingFace RemoteCLIP adapter.

    The optional ``transformers``/``torch`` dependencies are imported only on
    the first call, keeping the backend and its offline tests lightweight.
    """

    def __init__(self, model_id: str, *, dimension: int = 512, device: str | None = None):
        if not model_id:
            raise ValueError("model_id 不能为空")
        self.model_id = model_id
        self.dimension = dimension
        self.device = device
        self._processor: Any = None
        self._model: Any = None
        self._torch: Any = None

    def _load(self) -> None:
        if self._model is not None:
            return
        try:
            import torch  # type: ignore[import-not-found]
            from transformers import AutoModel, AutoProcessor  # type: ignore[import-not-found]
        except Exception as exc:  # pragma: no cover - optional production dependency
            raise TileVectorizationError("RemoteCLIP 需要安装 torch 与 transformers") from exc
        self._torch = torch
        self._processor = AutoProcessor.from_pretrained(self.model_id)
        self._model = AutoModel.from_pretrained(self.model_id)
        if self.device:
            self._model = self._model.to(self.device)
        self._model.eval()

    def encode_images(self, images: Sequence[Image.Image]) -> np.ndarray:
        self._load()
        inputs = self._processor(images=list(images), return_tensors="pt")
        if self.device:
            inputs = {key: value.to(self.device) for key, value in inputs.items()}
        with self._torch.no_grad():
            features = self._model.get_image_features(**inputs)
        return features.detach().cpu().numpy()

    def encode_text(self, text: str) -> np.ndarray:
        self._load()
        inputs = self._processor(text=[text], return_tensors="pt", padding=True)
        if self.device:
            inputs = {key: value.to(self.device) for key, value in inputs.items()}
        with self._torch.no_grad():
            features = self._model.get_text_features(**inputs)
        return features.detach().cpu().numpy()[0]


__all__ = [
    "FeatureEncoder",
    "HashFeatureEncoder",
    "RemoteCLIPEncoder",
    "TileGridConfig",
    "TileRecord",
    "TileVectorRecord",
    "TileVectorizationError",
    "TileVectorizer",
]
