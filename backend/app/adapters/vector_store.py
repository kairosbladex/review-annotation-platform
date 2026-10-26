"""Vector-store seam for patch retrieval.

``InMemoryVectorStore`` is the offline reference implementation.  The Qdrant
adapter is optional and imported lazily, so a backend worker can run all
contract tests without installing a vector database client.  Both adapters
expose the same cosine/top-k semantics and JSON-safe patch payloads.
"""

from __future__ import annotations

import copy
import json
import math
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Protocol, Sequence

import numpy as np


class VectorStoreError(RuntimeError):
    """Base class for vector-store failures."""


class VectorStoreUnavailableError(VectorStoreError):
    """Raised when an optional remote vector-store dependency is unavailable."""


@dataclass(frozen=True, slots=True)
class SearchHit:
    id: str
    score: float
    payload: Mapping[str, Any]
    vector: tuple[float, ...] | None = None

    def __getitem__(self, key: str) -> Any:
        if key == "id":
            return self.id
        if key == "score":
            return self.score
        if key == "payload":
            return self.payload
        if key == "vector":
            return self.vector
        raise KeyError(key)


class VectorStore(Protocol):
    dimension: int

    def upsert(self, records: Iterable[Any]) -> int:
        ...

    def search(self, vector: Any, *, limit: int = 50, filters: Mapping[str, Any] | None = None) -> list[SearchHit]:
        ...

    def delete(self, ids: Iterable[str]) -> int:
        ...


def _normalise(value: Any, dimension: int | None = None) -> np.ndarray:
    array = np.asarray(value, dtype=np.float32)
    if array.ndim != 1 or array.size == 0:
        raise ValueError("向量必须是一维非空数组")
    if dimension is not None and array.size != dimension:
        raise ValueError(f"向量维度不匹配: expected={dimension}, actual={array.size}")
    if not np.isfinite(array).all():
        raise ValueError("向量不能包含 NaN 或无穷值")
    norm = float(np.linalg.norm(array))
    if norm <= 1e-12:
        raise ValueError("向量不能是零向量")
    return array / norm


def _safe_payload(payload: Mapping[str, Any] | None) -> dict[str, Any]:
    if payload is None:
        return {}
    if not isinstance(payload, Mapping):
        raise TypeError("payload 必须是 mapping")
    try:
        # A round trip prevents object references and catches unsupported values
        # before a remote adapter receives them.
        return json.loads(json.dumps(dict(payload), ensure_ascii=False, allow_nan=False))
    except (TypeError, ValueError) as exc:
        raise ValueError("payload 必须是 JSON-safe 数据") from exc


def _record_parts(record: Any) -> tuple[str, Any, Mapping[str, Any]]:
    if hasattr(record, "id") and hasattr(record, "vector"):
        identifier = getattr(record, "id")
        vector = getattr(record, "vector")
        payload = getattr(record, "payload", {})
    elif isinstance(record, Mapping):
        identifier = record.get("id") or record.get("record_id")
        vector = record.get("vector") if "vector" in record else record.get("embedding")
        payload = record.get("payload") if "payload" in record else record.get("metadata")
        payload = {} if payload is None else payload
    else:
        raise TypeError("record 必须是带 id/vector/payload 的对象或 mapping")
    if not isinstance(identifier, str) or not identifier:
        raise ValueError("record id 必须是非空字符串")
    return identifier, vector, payload


class InMemoryVectorStore:
    """Reference cosine ANN substitute used by offline and acceptance tests.

    The implementation scans the compact numpy matrix.  It has the same
    bounded ``limit`` and filter contract as the Qdrant HNSW adapter, which
    lets callers validate the whole pipeline before provisioning a database.
    """

    metric = "cosine"
    index_type = "flat-offline-reference"

    def __init__(self, dimension: int, *, hnsw_m: int = 16, ef_construct: int = 100, max_limit: int = 1000):
        if isinstance(dimension, bool) or not isinstance(dimension, int) or dimension <= 0:
            raise ValueError("dimension 必须是正整数")
        if hnsw_m <= 0 or ef_construct <= 0 or max_limit <= 0:
            raise ValueError("HNSW 参数与 max_limit 必须为正整数")
        self.dimension = dimension
        self.hnsw_m = hnsw_m
        self.ef_construct = ef_construct
        self.max_limit = max_limit
        self._records: OrderedDict[str, tuple[np.ndarray, dict[str, Any]]] = OrderedDict()
        self._ids: list[str] = []
        self._matrix = np.empty((0, dimension), dtype=np.float32)
        self._payloads: list[dict[str, Any]] = []

    def upsert(self, records: Iterable[Any]) -> int:
        pending: list[tuple[str, np.ndarray, dict[str, Any]]] = []
        for record in records:
            identifier, vector, payload = _record_parts(record)
            pending.append((identifier, _normalise(vector, self.dimension), _safe_payload(payload)))
        for identifier, vector, payload in pending:
            self._records[identifier] = (vector, payload)
        if pending:
            self._rebuild_matrix()
        return len(pending)

    def _rebuild_matrix(self) -> None:
        self._ids = list(self._records)
        self._payloads = [payload for _vector, payload in self._records.values()]
        self._matrix = (
            np.stack([vector for vector, _payload in self._records.values()]).astype(np.float32, copy=False)
            if self._records
            else np.empty((0, self.dimension), dtype=np.float32)
        )

    def search(
        self,
        vector: Any,
        *,
        limit: int = 50,
        filters: Mapping[str, Any] | None = None,
    ) -> list[SearchHit]:
        if isinstance(limit, bool) or not isinstance(limit, int) or limit <= 0 or limit > self.max_limit:
            raise ValueError(f"limit 必须在 1..{self.max_limit} 范围内")
        query = _normalise(vector, self.dimension)
        if filters is not None and not isinstance(filters, Mapping):
            raise TypeError("filters 必须是 mapping")
        if not self._ids:
            return []
        scores = self._matrix @ query
        if filters:
            eligible = [index for index, payload in enumerate(self._payloads) if _matches(payload, filters)]
        else:
            if len(self._ids) > limit:
                # Avoid sorting all 100k scores in the common unfiltered path;
                # only the bounded ANN result set needs an exact ordering.
                eligible = np.argpartition(-scores, limit - 1)[:limit].tolist()
            else:
                eligible = list(range(len(self._ids)))
        # Stable argsort preserves insertion order for ties, making test runs
        # and pagination deterministic across Python/numpy versions.
        ranked = sorted(eligible, key=lambda index: (-float(scores[index]), index))[:limit]
        return [
            SearchHit(
                id=self._ids[index],
                score=float(scores[index]),
                payload=copy.deepcopy(self._payloads[index]),
                vector=tuple(float(item) for item in self._matrix[index]),
            )
            for index in ranked
        ]

    def delete(self, ids: Iterable[str]) -> int:
        deleted = 0
        for identifier in ids:
            if identifier in self._records:
                del self._records[identifier]
                deleted += 1
        if deleted:
            self._rebuild_matrix()
        return deleted

    def count(self) -> int:
        return len(self._records)


def _matches(payload: Mapping[str, Any], filters: Mapping[str, Any]) -> bool:
    for key, expected in filters.items():
        actual = payload.get(key)
        if isinstance(expected, (set, frozenset, tuple, list)):
            if actual not in expected:
                return False
        elif actual != expected:
            return False
    return True


class QdrantVectorStore:
    """Qdrant HNSW adapter with lazy optional dependency loading."""

    metric = "cosine"
    index_type = "hnsw"

    def __init__(
        self,
        collection_name: str,
        dimension: int,
        *,
        client: Any | None = None,
        url: str | None = None,
        api_key: str | None = None,
        hnsw_m: int = 16,
        ef_construct: int = 100,
        recreate: bool = False,
    ):
        if not collection_name:
            raise ValueError("collection_name 不能为空")
        if dimension <= 0:
            raise ValueError("dimension 必须为正整数")
        self.collection_name = collection_name
        self.dimension = dimension
        self.hnsw_m = hnsw_m
        self.ef_construct = ef_construct
        if client is None:
            try:
                from qdrant_client import QdrantClient  # type: ignore[import-not-found]
            except Exception as exc:  # pragma: no cover - optional dependency
                raise VectorStoreUnavailableError("Qdrant 适配器需要安装 qdrant-client") from exc
            client = QdrantClient(url=url, api_key=api_key) if url else QdrantClient(location=":memory:")
        self.client = client
        self._ensure_collection(recreate=recreate)

    def _ensure_collection(self, *, recreate: bool) -> None:
        try:
            from qdrant_client.http import models  # type: ignore[import-not-found]
        except Exception as exc:  # pragma: no cover
            raise VectorStoreUnavailableError("Qdrant models 不可用") from exc
        if recreate:
            try:
                self.client.delete_collection(self.collection_name)
            except Exception:
                pass
        exists = False
        try:
            exists = bool(self.client.collection_exists(self.collection_name))
        except AttributeError:
            try:
                self.client.get_collection(self.collection_name)
                exists = True
            except Exception:
                exists = False
        if not exists:
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=models.VectorParams(size=self.dimension, distance=models.Distance.COSINE),
                hnsw_config=models.HnswConfigDiff(m=self.hnsw_m, ef_construct=self.ef_construct),
            )

    def upsert(self, records: Iterable[Any]) -> int:
        try:
            from qdrant_client.http import models  # type: ignore[import-not-found]
        except Exception as exc:  # pragma: no cover
            raise VectorStoreUnavailableError("Qdrant models 不可用") from exc
        points = []
        for record in records:
            identifier, vector, payload = _record_parts(record)
            normalised = _normalise(vector, self.dimension)
            # Qdrant accepts UUIDs or unsigned integers as point IDs.  Keep
            # the human-readable patch ID in payload while using a stable UUID
            # for the database key.
            import uuid

            qdrant_id = str(uuid.uuid5(uuid.NAMESPACE_URL, "review-annotation-platform:" + identifier))
            safe_payload = _safe_payload(payload)
            safe_payload.setdefault("record_id", identifier)
            points.append(
                models.PointStruct(
                    id=qdrant_id,
                    vector=normalised.tolist(),
                    payload=safe_payload,
                )
            )
        if points:
            self.client.upsert(collection_name=self.collection_name, points=points, wait=True)
        return len(points)

    def search(
        self,
        vector: Any,
        *,
        limit: int = 50,
        filters: Mapping[str, Any] | None = None,
    ) -> list[SearchHit]:
        if limit <= 0:
            raise ValueError("limit 必须为正整数")
        query = _normalise(vector, self.dimension).tolist()
        query_filter = self._filter(filters)
        if hasattr(self.client, "query_points"):
            response = self.client.query_points(
                collection_name=self.collection_name,
                query=query,
                query_filter=query_filter,
                limit=limit,
                with_payload=True,
                with_vectors=False,
            )
            points = getattr(response, "points", response)
        else:  # qdrant-client < 1.10
            points = self.client.search(
                collection_name=self.collection_name,
                query_vector=query,
                query_filter=query_filter,
                limit=limit,
                with_payload=True,
                with_vectors=False,
            )
        return [
            SearchHit(
                id=str((point.payload or {}).get("record_id") or point.id),
                score=float(point.score),
                payload=_safe_payload(point.payload or {}),
            )
            for point in points
        ]

    def _filter(self, filters: Mapping[str, Any] | None) -> Any:
        if not filters:
            return None
        try:
            from qdrant_client.http import models  # type: ignore[import-not-found]
        except Exception as exc:  # pragma: no cover
            raise VectorStoreUnavailableError("Qdrant models 不可用") from exc
        conditions = []
        for key, value in filters.items():
            if isinstance(value, (list, tuple, set, frozenset)):
                conditions.append(models.FieldCondition(key=key, match=models.MatchAny(any=list(value))))
            else:
                conditions.append(models.FieldCondition(key=key, match=models.MatchValue(value=value)))
        return models.Filter(must=conditions)

    def delete(self, ids: Iterable[str]) -> int:
        identifiers = list(ids)
        if not identifiers:
            return 0
        try:
            from qdrant_client.http import models  # type: ignore[import-not-found]
        except Exception as exc:  # pragma: no cover
            raise VectorStoreUnavailableError("Qdrant models 不可用") from exc
        import uuid

        point_ids = [
            str(uuid.uuid5(uuid.NAMESPACE_URL, "review-annotation-platform:" + identifier))
            for identifier in identifiers
        ]
        self.client.delete(
            collection_name=self.collection_name,
            points_selector=models.PointIdsList(points=point_ids),
            wait=True,
        )
        return len(identifiers)


__all__ = [
    "InMemoryVectorStore",
    "QdrantVectorStore",
    "SearchHit",
    "VectorStore",
    "VectorStoreError",
    "VectorStoreUnavailableError",
]
