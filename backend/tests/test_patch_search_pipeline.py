from __future__ import annotations

from PIL import Image
import pytest

from app.adapters.vector_store import InMemoryVectorStore
from app.pipelines.patch_search_pipeline import PatchSearchError, PatchSearchPipeline
from app.services.tile_vectorizer import TileVectorRecord


class QueryEncoder:
    def encode_text(self, text):
        return [1.0, 0.0, 0.0, 0.0] if "棚" in text else [0.0, 1.0, 0.0, 0.0]

    def encode_image(self, image):
        return [0.0, 1.0, 0.0, 0.0]


def _store():
    store = InMemoryVectorStore(4)
    store.upsert([
        TileVectorRecord(
            "img-1:0",
            (1, 0, 0, 0),
            {"image_id": "img-1", "tile_index": 0, "bbox_on_orig": [100, 200, 612, 712], "tile_size": [512, 512]},
        ),
        TileVectorRecord(
            "img-2:0",
            (0, 1, 0, 0),
            {"image_id": "img-2", "tile_index": 0, "bbox_on_orig": [0, 0, 512, 512], "tile_size": [512, 512]},
        ),
    ])
    return store


def test_text_search_returns_ann_candidates_with_bounded_timings():
    pipeline = PatchSearchPipeline(QueryEncoder(), _store())

    response = pipeline.search_text("蓝色彩钢棚", top_k=1, refine=False)

    assert response.query_type == "text"
    assert response.status == "ann_only"
    assert response.verified is False
    assert [candidate.id for candidate in response.candidates] == ["img-1:0"]
    assert response.candidates[0].refinement_status == "not_requested"
    assert response.timings_ms["total_ms"] >= response.timings_ms["ann_ms"]
    assert response.evidence["ann_candidates"] == 1


def test_image_query_and_refinement_map_normalized_tile_box_to_original_pixels():
    calls = []

    class Refiner:
        def refine(self, image, candidate, *, class_name=None):
            calls.append((image.size, candidate.id, class_name))
            return {"accepted": True, "bbox_space": "normalized", "boxes": [{"bbox": [0.25, 0.5, 0.75, 1.0]}]}

    pipeline = PatchSearchPipeline(
        QueryEncoder(),
        _store(),
        refiner=Refiner(),
        image_loader=lambda image_id: Image.new("RGB", (1_024, 1_024), "white"),
    )
    response = pipeline.search_image(Image.new("RGB", (32, 32)), top_k=1, class_name="彩钢棚")

    candidate = response.candidates[0]
    assert response.status == "completed"
    assert candidate.refinement_status == "accepted"
    assert candidate.global_boxes == [{"bbox": [128, 256, 384, 512]}]
    assert calls == [((1_024, 1_024), "img-2:0", "彩钢棚")]


def test_refinement_failure_keeps_ann_candidate_and_marks_error():
    class BrokenRefiner:
        def refine(self, image, candidate, **kwargs):
            raise RuntimeError("provider secret must not escape")

    pipeline = PatchSearchPipeline(
        QueryEncoder(),
        _store(),
        refiner=BrokenRefiner(),
        image_loader=lambda image_id: Image.new("RGB", (1_024, 1_024)),
    )
    response = pipeline.search_text("蓝色彩钢棚", top_k=1)

    assert response.status == "partial"
    assert response.candidates[0].id == "img-1:0"
    assert response.candidates[0].refinement_status == "refinement_failed"
    assert response.candidates[0].error == "RuntimeError"
    assert "secret" not in str(response.to_dict())


@pytest.mark.parametrize("query", ["", "   "])
def test_invalid_query_and_top_k_fail_before_store_work(query):
    class Store:
        dimension = 4

        def search(self, *args, **kwargs):
            raise AssertionError("store should not be called")

    pipeline = PatchSearchPipeline(QueryEncoder(), Store())
    with pytest.raises(PatchSearchError):
        pipeline.search_text(query)
    with pytest.raises(PatchSearchError):
        pipeline.search_text("valid", top_k=0)


def test_filter_is_passed_to_ann_and_missing_loader_is_explicit_refinement_failure():
    pipeline = PatchSearchPipeline(QueryEncoder(), _store(), refiner=lambda image, candidate: {"accepted": True})
    response = pipeline.search_text("蓝色彩钢棚", filters={"image_id": "img-1"}, top_k=2)

    assert [candidate.id for candidate in response.candidates] == ["img-1:0"]
    assert response.status == "partial"
    assert response.candidates[0].refinement_status == "refinement_failed"
