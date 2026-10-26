from __future__ import annotations

from datetime import datetime, timezone

import numpy as np
import pytest
from PIL import Image

from app.services.tile_vectorizer import (
    HashFeatureEncoder,
    TileGridConfig,
    TileVectorizationError,
    TileVectorizer,
)


class RecordingEncoder:
    dimension = 3

    def __init__(self):
        self.batches = []

    def encode_images(self, images):
        self.batches.append([image.size for image in images])
        # Deliberately non-unit vectors: the vectorizer owns normalization.
        return np.tile(np.array([3.0, 4.0, 0.0], dtype=np.float32), (len(images), 1))


def test_grid_uses_overlap_and_shifts_last_tile_to_image_edge():
    encoder = RecordingEncoder()
    vectorizer = TileVectorizer(encoder, TileGridConfig(tile_size=512, overlap=0.2), batch_size=8)
    image = Image.new("RGB", (1_000, 700), "white")

    tiles = list(vectorizer.iter_tiles(image, image_id="orthophoto-1"))

    # stride=round(512*.8)=410; x starts are 0, 410, 488 and y starts 0,188.
    assert [tile.bbox_on_orig for tile in tiles] == [
        (0, 0, 512, 512),
        (410, 0, 922, 512),
        (488, 0, 1000, 512),
        (0, 188, 512, 700),
        (410, 188, 922, 700),
        (488, 188, 1000, 700),
    ]
    assert all(tile.image.size == (x2 - x1, y2 - y1) for tile in tiles for x1, y1, x2, y2 in [tile.bbox_on_orig])


def test_vectorize_batches_and_normalizes_with_json_safe_payload():
    encoder = RecordingEncoder()
    vectorizer = TileVectorizer(encoder, TileGridConfig(tile_size=4, overlap=0), batch_size=2)
    image = Image.new("RGB", (8, 4), "#123456")

    records = vectorizer.vectorize(
        image,
        image_id="img-1",
        capture_time=datetime(2026, 10, 26, 1, 2, 3, tzinfo=timezone.utc),
        resolution=np.float32(0.25),
    )

    assert len(records) == 2
    assert encoder.batches == [[(4, 4), (4, 4)]]
    assert records[0].id == "img-1:0"
    assert records[0].vector == pytest.approx((0.6, 0.8, 0.0), abs=1e-6)
    assert records[0].payload["bbox_on_orig"] == [0, 0, 4, 4]
    assert records[0].payload["capture_time"] == "2026-10-26T01:02:03+00:00"
    assert records[0].payload["resolution"] == 0.25


@pytest.mark.parametrize("kwargs", [{"tile_size": 0}, {"tile_size": 8, "overlap": -0.1}, {"tile_size": 8, "overlap": 1}])
def test_invalid_grid_config_is_rejected(kwargs):
    with pytest.raises(ValueError):
        TileGridConfig(**kwargs)


def test_invalid_encoder_output_and_zero_vectors_fail_closed():
    class BadEncoder:
        dimension = 2

        def encode_images(self, images):
            return np.zeros((len(images), 2), dtype=np.float32)

    vectorizer = TileVectorizer(BadEncoder(), TileGridConfig(tile_size=8))
    with pytest.raises(TileVectorizationError, match="零向量"):
        vectorizer.vectorize(Image.new("RGB", (8, 8)), image_id="img")


def test_hash_encoder_is_deterministic_and_supports_text_queries():
    encoder = HashFeatureEncoder(dimension=8)
    image = Image.new("RGB", (12, 10), "#abcdef")
    first = encoder.encode_images([image])[0]
    second = encoder.encode_images([image])[0]
    text_a = encoder.encode_text("蓝色彩钢棚")
    text_b = encoder.encode_text("蓝色彩钢棚")

    assert first.shape == (8,)
    assert np.array_equal(first, second)
    assert np.array_equal(text_a, text_b)
    assert np.isfinite(first).all() and np.isfinite(text_a).all()
