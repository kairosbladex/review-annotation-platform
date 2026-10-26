from __future__ import annotations

import numpy as np
import pytest

from app.adapters.vector_store import InMemoryVectorStore
from app.services.tile_vectorizer import TileVectorRecord


def _record(identifier: str, vector, image_id: str = "a"):
    return TileVectorRecord(identifier, tuple(vector), {"image_id": image_id, "kind": "roof"})


def test_in_memory_store_normalizes_vectors_and_returns_cosine_ranked_hits():
    store = InMemoryVectorStore(2, max_limit=10)
    assert store.upsert([
        _record("east", [10, 0]),
        _record("diagonal", [1, 1]),
        _record("north", [0, 3]),
    ]) == 3

    hits = store.search([2, 0], limit=3)
    assert [hit.id for hit in hits] == ["east", "diagonal", "north"]
    assert hits[0].score == pytest.approx(1.0)
    assert hits[1].score == pytest.approx(2**-0.5)
    # Returned payloads are copies, so callers cannot mutate the index.
    hits[0].payload["image_id"] = "mutated"
    assert store.search([1, 0], limit=1)[0].payload["image_id"] == "a"


def test_store_filters_payload_and_delete_is_idempotent():
    store = InMemoryVectorStore(2)
    store.upsert([_record("a-1", [1, 0], "a"), _record("b-1", [1, 0], "b")])

    assert [hit.id for hit in store.search([1, 0], filters={"image_id": "b"})] == ["b-1"]
    assert [hit.id for hit in store.search([1, 0], filters={"image_id": ["a", "c"]})] == ["a-1"]
    assert store.delete(["a-1", "missing", "a-1"]) == 1
    assert store.count() == 1
    assert store.delete([]) == 0


@pytest.mark.parametrize(
    "operation",
    [
        lambda store: store.upsert([_record("bad", [1, 2, 3])]),
        lambda store: store.search([0, 0]),
        lambda store: store.search([1, 0], limit=0),
        lambda store: store.search([1, 0], limit=1001),
    ],
)
def test_store_rejects_invalid_dimensions_zero_queries_and_limits(operation):
    with pytest.raises(ValueError):
        operation(InMemoryVectorStore(2))


def test_upsert_accepts_mapping_records_and_replaces_existing_id():
    store = InMemoryVectorStore(2)
    assert store.upsert([{"id": "same", "embedding": np.array([1, 0]), "metadata": {"v": 1}}]) == 1
    assert store.upsert([{"id": "same", "vector": [0, 1], "payload": {"v": 2}}]) == 1
    hit = store.search([0, 1], limit=1)[0]
    assert hit.id == "same"
    assert hit.payload == {"v": 2}
