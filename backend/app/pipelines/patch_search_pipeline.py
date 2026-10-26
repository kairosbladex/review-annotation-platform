"""Online text/image patch retrieval and bounded stage-2 refinement."""

from __future__ import annotations

import inspect
import math
import time
from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping, Protocol, Sequence

import numpy as np

from ..adapters.vector_store import SearchHit, VectorStore
from ..services.tile_vectorizer import _as_rgb_image


class PatchSearchError(ValueError):
    """Raised for invalid search input or an unusable query embedding."""


class QueryEncoder(Protocol):
    def encode_text(self, text: str) -> Any:
        ...

    def encode_image(self, image: Any) -> Any:
        ...


class CandidateRefiner(Protocol):
    def refine(self, image: Any, candidate: "PatchCandidate", **kwargs: Any) -> Any:
        ...


def _unit_query(value: Any, dimension: int | None = None) -> np.ndarray:
    array = np.asarray(value, dtype=np.float32)
    if array.ndim == 2:
        if array.shape[0] != 1:
            raise PatchSearchError("query encoder 必须返回一个向量")
        array = array[0]
    if array.ndim != 1 or array.size == 0:
        raise PatchSearchError("query encoder 必须返回一维非空向量")
    if dimension is not None and array.size != dimension:
        raise PatchSearchError(f"query 向量维度不匹配: expected={dimension}, actual={array.size}")
    if not np.isfinite(array).all():
        raise PatchSearchError("query 向量不能包含 NaN 或无穷值")
    norm = float(np.linalg.norm(array))
    if norm <= 1e-12:
        raise PatchSearchError("query 不能编码为零向量")
    return array / norm


@dataclass(slots=True)
class PatchCandidate:
    id: str
    score: float
    payload: dict[str, Any]
    image_id: str | None = None
    tile_index: int | None = None
    bbox_on_orig: tuple[int, int, int, int] | None = None
    refinement_status: str = "not_requested"
    refinement: dict[str, Any] | None = None
    global_boxes: list[dict[str, Any]] = field(default_factory=list)
    error: str | None = None

    @classmethod
    def from_hit(cls, hit: SearchHit | Mapping[str, Any]) -> "PatchCandidate":
        if isinstance(hit, SearchHit):
            identifier, score, payload = hit.id, hit.score, dict(hit.payload)
        elif isinstance(hit, Mapping):
            identifier = str(hit.get("id") or hit.get("record_id") or "")
            score = float(hit.get("score", 0.0))
            payload = dict(hit.get("payload") or hit.get("metadata") or {})
        else:
            raise TypeError("ANN hit 类型不受支持")
        if not identifier:
            raise PatchSearchError("ANN hit 缺少 id")
        if not math.isfinite(score):
            raise PatchSearchError("ANN hit score 必须是有限数")
        raw_bbox = payload.get("bbox_on_orig")
        bbox = _parse_bbox(raw_bbox) if raw_bbox is not None else None
        tile_index = payload.get("tile_index")
        return cls(
            id=identifier,
            score=score,
            payload=payload,
            image_id=str(payload["image_id"]) if payload.get("image_id") is not None else None,
            tile_index=int(tile_index) if isinstance(tile_index, (int, float)) else None,
            bbox_on_orig=bbox,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "score": self.score,
            "image_id": self.image_id,
            "tile_index": self.tile_index,
            "bbox_on_orig": list(self.bbox_on_orig) if self.bbox_on_orig else None,
            "payload": dict(self.payload),
            "refinement_status": self.refinement_status,
            "refinement": self.refinement,
            "global_boxes": list(self.global_boxes),
            "error": self.error,
        }

    def __getitem__(self, key: str) -> Any:
        return self.to_dict()[key]


@dataclass(slots=True)
class PatchSearchResponse:
    query_type: str
    candidates: list[PatchCandidate]
    status: str
    timings_ms: dict[str, float]
    verified: bool = False
    evidence: dict[str, Any] = field(default_factory=dict)

    @property
    def results(self) -> list[PatchCandidate]:
        return self.candidates

    def to_dict(self) -> dict[str, Any]:
        return {
            "query_type": self.query_type,
            "status": self.status,
            "verified": self.verified,
            "evidence": dict(self.evidence),
            "timings_ms": dict(self.timings_ms),
            "candidates": [candidate.to_dict() for candidate in self.candidates],
            "results": [candidate.to_dict() for candidate in self.candidates],
        }

    def __getitem__(self, key: str) -> Any:
        return self.to_dict()[key]

    def get(self, key: str, default: Any = None) -> Any:
        return self.to_dict().get(key, default)


class PatchSearchPipeline:
    """Run query encoding, ANN retrieval, and an optional bounded refinement.

    ``image_loader`` is intentionally injected.  It may read an original image
    from object storage in production or return an in-memory fixture in tests;
    the vector store never receives image bytes.  Refinement failures preserve
    the ANN candidate and are represented in its status instead of becoming a
    false negative.
    """

    def __init__(
        self,
        encoder: QueryEncoder,
        vector_store: VectorStore,
        *,
        refiner: CandidateRefiner | Any | None = None,
        image_loader: Any | None = None,
        max_top_k: int = 200,
        max_refine_candidates: int = 50,
    ):
        if not callable(getattr(encoder, "encode_text", None)):
            raise TypeError("query encoder 必须提供 encode_text")
        if not callable(getattr(encoder, "encode_image", None)) and not callable(
            getattr(encoder, "encode_images", None)
        ):
            raise TypeError("query encoder 必须提供 encode_image 或 encode_images")
        if max_top_k <= 0 or max_refine_candidates <= 0:
            raise ValueError("候选上限必须为正整数")
        self.encoder = encoder
        self.vector_store = vector_store
        self.refiner = refiner
        self.image_loader = image_loader
        self.max_top_k = max_top_k
        self.max_refine_candidates = min(max_refine_candidates, max_top_k)

    def search(
        self,
        query: str | Any | None = None,
        *,
        text: str | None = None,
        image: Any | None = None,
        top_k: int = 50,
        filters: Mapping[str, Any] | None = None,
        refine: bool = True,
        class_name: str | None = None,
        timeout_ms: float | None = 1500.0,
    ) -> PatchSearchResponse:
        if query is None:
            if text is not None and image is not None:
                raise PatchSearchError("text 与 image 只能选择一个")
            query = text if text is not None else image
        elif text is not None or image is not None:
            raise PatchSearchError("query 与 text/image 不能同时提供")
        if query is None:
            raise PatchSearchError("必须提供 text 或 image query")
        if isinstance(top_k, bool) or not isinstance(top_k, int) or top_k <= 0 or top_k > self.max_top_k:
            raise PatchSearchError(f"top_k 必须在 1..{self.max_top_k} 范围内")
        if timeout_ms is not None and (not math.isfinite(float(timeout_ms)) or timeout_ms <= 0):
            raise PatchSearchError("timeout_ms 必须为正数或 None")
        started = time.perf_counter()
        deadline = started + float(timeout_ms) / 1000.0 if timeout_ms is not None else None
        query_type, encoded = self._encode_query(query)
        encoded_ms = _elapsed_ms(started)
        ann_started = time.perf_counter()
        hits = self.vector_store.search(encoded, limit=top_k, filters=filters)
        ann_ms = _elapsed_ms(ann_started)
        candidates = [PatchCandidate.from_hit(hit) for hit in hits]
        timings = {
            "query_encoding_ms": round(encoded_ms, 3),
            "ann_ms": round(ann_ms, 3),
            "refinement_ms": 0.0,
            "total_ms": 0.0,
        }

        if not refine or self.refiner is None:
            for candidate in candidates:
                candidate.refinement_status = "not_requested" if not refine else "refiner_unavailable"
            timings["total_ms"] = round(_elapsed_ms(started), 3)
            return PatchSearchResponse(
                query_type=query_type,
                candidates=candidates,
                status="ann_only",
                timings_ms=timings,
                verified=False,
                evidence={"ann_candidates": len(candidates), "refined": 0, "verified": False},
            )

        refinement_started = time.perf_counter()
        refinement_count = 0
        deadline_hit = False
        failed_count = 0
        for candidate_index, candidate in enumerate(candidates[: self.max_refine_candidates]):
            if deadline is not None and time.perf_counter() >= deadline:
                deadline_hit = True
                candidate.refinement_status = "deadline_exceeded"
                for remaining in candidates[candidate_index + 1 :]:
                    remaining.refinement_status = "deadline_exceeded"
                break
            try:
                image = self._load_candidate_image(candidate)
                raw = self._invoke_refiner(image, candidate, class_name=class_name)
                candidate.refinement = _normalise_refinement(raw)
                candidate.global_boxes = _map_refinement_boxes(candidate.refinement, candidate)
                accepted = candidate.refinement.get("accepted")
                candidate.refinement_status = (
                    "accepted" if accepted is True else "rejected" if accepted is False else "refined"
                )
                refinement_count += 1
            except Exception as exc:  # refinement must never discard ANN evidence
                candidate.refinement_status = "refinement_failed"
                candidate.error = type(exc).__name__
                failed_count += 1
        if len(candidates) > self.max_refine_candidates:
            for candidate in candidates[self.max_refine_candidates :]:
                candidate.refinement_status = "refinement_skipped_limit"
        timings["refinement_ms"] = round(_elapsed_ms(refinement_started), 3)
        timings["total_ms"] = round(_elapsed_ms(started), 3)
        status = "partial" if deadline_hit or failed_count else "completed"
        return PatchSearchResponse(
            query_type=query_type,
            candidates=candidates,
            status=status,
            timings_ms=timings,
            verified=False,
            evidence={
                "ann_candidates": len(candidates),
                "refined": refinement_count,
                "refinement_failed": failed_count,
                "deadline_exceeded": deadline_hit,
                "verified": False,
            },
        )

    def search_text(self, text: str, **kwargs: Any) -> PatchSearchResponse:
        return self.search(text, **kwargs)

    def search_image(self, image: Any, **kwargs: Any) -> PatchSearchResponse:
        return self.search(image, **kwargs)

    def _encode_query(self, query: str | Any) -> tuple[str, np.ndarray]:
        if isinstance(query, str):
            if not query.strip():
                raise PatchSearchError("text query 不能为空")
            raw = self.encoder.encode_text(query)
            query_type = "text"
        else:
            image = _as_rgb_image(query)
            encoder = getattr(self.encoder, "encode_image", None)
            if callable(encoder):
                raw = encoder(image)
            else:
                raw = self.encoder.encode_images([image])
            query_type = "image"
        return query_type, _unit_query(raw, getattr(self.vector_store, "dimension", None))

    def _load_candidate_image(self, candidate: PatchCandidate) -> Any:
        if self.image_loader is None:
            embedded = candidate.payload.get("_image")
            if embedded is not None:
                return embedded
            raise PatchSearchError("refinement 需要注入 image_loader")
        if candidate.image_id is None:
            raise PatchSearchError("候选缺少 image_id，无法加载原图")
        if isinstance(self.image_loader, Mapping):
            return self.image_loader[candidate.image_id]
        return self.image_loader(candidate.image_id)

    def _invoke_refiner(self, image: Any, candidate: PatchCandidate, *, class_name: str | None) -> Any:
        method = getattr(self.refiner, "refine", self.refiner)
        if not callable(method):
            raise PatchSearchError("refiner 必须提供 refine 方法或可调用对象")
        try:
            signature = inspect.signature(method)
            parameters = signature.parameters
            accepts_kwargs = any(parameter.kind == inspect.Parameter.VAR_KEYWORD for parameter in parameters.values())
        except (TypeError, ValueError):
            accepts_kwargs = True
            parameters = {}
        kwargs = {"class_name": class_name} if accepts_kwargs or "class_name" in parameters else {}
        positional = [
            parameter
            for parameter in parameters.values()
            if parameter.kind in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)
        ]
        if len(positional) == 1:
            return method(candidate, **kwargs)
        return method(image, candidate, **kwargs)


class JevOmniRefiner:
    """Adapter from the existing Jev-Omni double-view classifier to this pipeline."""

    def __init__(self, adapter: Any, *, confirm_threshold: float = 0.91, class_name: str | None = None):
        if not callable(getattr(adapter, "decide", None)):
            raise TypeError("Jev-Omni adapter 必须提供 decide")
        if not 0 <= confirm_threshold <= 1:
            raise ValueError("confirm_threshold 必须在 [0, 1]")
        self.adapter = adapter
        self.confirm_threshold = confirm_threshold
        self.default_class_name = class_name

    def refine(self, image: Any, candidate: PatchCandidate, *, class_name: str | None = None) -> dict[str, Any]:
        from ..adapters.jev_omni_classifier import build_jev_omni_review_views

        original = _as_rgb_image(image)
        bbox = candidate.bbox_on_orig
        if bbox is None:
            raise PatchSearchError("Jev-Omni 精检候选缺少 bbox_on_orig")
        width, height = original.size
        normalised_bbox = [bbox[0] / width, bbox[1] / height, bbox[2] / width, bbox[3] / height]
        context, roi = build_jev_omni_review_views(
            original, normalised_bbox, class_name=class_name or self.default_class_name
        )
        decision = self.adapter.decide(
            context,
            roi,
            class_name or self.default_class_name,
            {"candidate_id": candidate.id, "detector_score": candidate.score},
        )
        p_true = float(decision.get("p_true", 0.0))
        return {
            **decision,
            "accepted": p_true >= self.confirm_threshold,
            "bbox": list(bbox),
            "bbox_space": "orig",
            "verified": True,
        }


def _parse_bbox(value: Any) -> tuple[int, int, int, int]:
    if not isinstance(value, (list, tuple)) or len(value) != 4:
        raise PatchSearchError("候选 bbox_on_orig 必须包含四个坐标")
    numbers = tuple(float(item) for item in value)
    if not all(math.isfinite(item) for item in numbers):
        raise PatchSearchError("候选 bbox 含非有限坐标")
    x1, y1, x2, y2 = (round(item) for item in numbers)
    if x1 < 0 or y1 < 0 or x2 <= x1 or y2 <= y1:
        raise PatchSearchError("候选 bbox_on_orig 非法")
    return x1, y1, x2, y2


def _normalise_refinement(raw: Any) -> dict[str, Any]:
    if raw is None:
        return {"accepted": None}
    if isinstance(raw, bool):
        return {"accepted": raw}
    if isinstance(raw, Mapping):
        return dict(raw)
    raise PatchSearchError("refiner 返回值必须是 mapping、bool 或 None")


def _map_refinement_boxes(refinement: Mapping[str, Any], candidate: PatchCandidate) -> list[dict[str, Any]]:
    raw_boxes = refinement.get("boxes") or refinement.get("detections") or []
    if isinstance(raw_boxes, Mapping):
        raw_boxes = [raw_boxes]
    if not isinstance(raw_boxes, Sequence) or isinstance(raw_boxes, (str, bytes, bytearray)):
        return []
    mapped: list[dict[str, Any]] = []
    for raw in raw_boxes:
        if isinstance(raw, Mapping):
            bbox = raw.get("bbox")
            item = dict(raw)
        else:
            bbox = raw
            item = {}
        if bbox is None:
            continue
        try:
            item["bbox"] = list(_map_bbox(bbox, candidate, refinement.get("bbox_space")))
        except (TypeError, ValueError, PatchSearchError):
            continue
        mapped.append(item)
    if not mapped and refinement.get("bbox") is not None:
        try:
            mapped.append({"bbox": list(_map_bbox(refinement["bbox"], candidate, refinement.get("bbox_space")))})
        except (TypeError, ValueError, PatchSearchError):
            pass
    return mapped


def _map_bbox(value: Any, candidate: PatchCandidate, space: Any = None) -> tuple[int, int, int, int]:
    if not isinstance(value, (list, tuple)) or len(value) != 4:
        raise ValueError("bbox 必须包含四个坐标")
    coords = tuple(float(item) for item in value)
    if not all(math.isfinite(item) for item in coords):
        raise ValueError("bbox 含非有限坐标")
    if space == "orig":
        return _clip_bbox(coords, candidate.bbox_on_orig)
    tile_bbox = candidate.bbox_on_orig
    if tile_bbox is None:
        raise PatchSearchError("候选缺少 tile bbox")
    tx1, ty1, tx2, ty2 = tile_bbox
    tile_size = candidate.payload.get("tile_size")
    if isinstance(tile_size, (list, tuple)) and len(tile_size) == 2:
        tw, th = float(tile_size[0]), float(tile_size[1])
    else:
        tw, th = float(tx2 - tx1), float(ty2 - ty1)
    if space == "normalized" or (space is None and all(0.0 <= item <= 1.0 for item in coords)):
        local = (coords[0] * tw, coords[1] * th, coords[2] * tw, coords[3] * th)
    else:
        local = coords
    return _clip_bbox((tx1 + local[0], ty1 + local[1], tx1 + local[2], ty1 + local[3]), tile_bbox)


def _clip_bbox(coords: Sequence[float], bounds: tuple[int, int, int, int] | None) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = coords
    if bounds is not None:
        bx1, by1, bx2, by2 = bounds
        x1, y1, x2, y2 = max(bx1, x1), max(by1, y1), min(bx2, x2), min(by2, y2)
    result = (round(x1), round(y1), round(x2), round(y2))
    if result[2] <= result[0] or result[3] <= result[1]:
        raise ValueError("bbox 映射后为空")
    return result


def _elapsed_ms(started: float) -> float:
    return (time.perf_counter() - started) * 1000.0


__all__ = [
    "CandidateRefiner",
    "JevOmniRefiner",
    "PatchCandidate",
    "PatchSearchError",
    "PatchSearchPipeline",
    "PatchSearchResponse",
    "QueryEncoder",
]
