"""Tests for tools/prompt_eval_data/crop_dataset (contract section 12).

Requires Pillow + pytest (isolated env):
    .venv-crop/bin/python -m pytest backend/tests/test_crop_dataset.py -q
"""

from __future__ import annotations

import io
import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.prompt_eval_data.common import sha256_bytes  # noqa: E402
from tools.prompt_eval_data.crop_dataset import (  # noqa: E402
    EXCLUDED,
    TRAIN,
    VAL,
    assign_train_val,
    build_crop_dataset,
    padded_crop_box,
    render_crop,
)

PIL = pytest.importorskip("PIL", reason="crop_dataset requires Pillow")
from PIL import Image  # noqa: E402

MIN_EDGE = 64  # small value keeps fixture images tiny


def _image_for_split(size: tuple[int, int], target: str) -> bytes:
    """Solid/quadrant PNG whose sha256 lands in the requested split bucket."""
    for nonce in range(20000):
        im = Image.new("RGB", size, (30, 30, 30))
        # left half red / right half blue; nonce pixel perturbs the sha.
        for x in range(size[0]):
            for y in range(size[1]):
                im.putpixel((x, y), (200, 10, 10) if x < size[0] // 2 else (10, 10, 200))
        im.putpixel((0, 0), (nonce % 256, (nonce // 256) % 256, (nonce // 65536) % 256))
        buf = io.BytesIO()
        im.save(buf, "PNG")
        data = buf.getvalue()
        if assign_train_val(sha256_bytes(data)) == target:
            return data
    raise AssertionError(f"no fixture image found for split {target}")


def _record(cid: str, image: dict, bbox: list[int], class_id: int = 8, teacher: str | None = "true") -> dict:
    return {
        "canonical_id": cid,
        "class_id": class_id,
        "class_name": "车",
        "bbox_px": bbox,
        "detector_confidence": 0.5,
        "image_sha256": image["sha"],
        "image_object_key": image["key"],
        "image_width": image["width"],
        "image_height": image["height"],
        "prior_gpt": (
            {"passes": [{"label": teacher, "model": "gpt-5.6-luna", "run_id": "fixture-run", "status": "succeeded"}]}
            if teacher
            else None
        ),
    }


def _gold(cid: str, label: str) -> dict:
    return {"canonical_id": cid, "class_id": 8, "class_name": "车", "label": label}


def build_fixture(tmp_path: Path) -> tuple[Path, Path, dict]:
    images_root = tmp_path / "images"
    images = {}
    for name, size, target in [
        ("train", (96, 72), TRAIN),
        ("val", (64, 64), VAL),
        ("locked", (48, 48), EXCLUDED),
    ]:
        data = _image_for_split(size, target)
        key = f"fix/{name}.png"
        (images_root / "fix").mkdir(parents=True, exist_ok=True)
        (images_root / key).write_bytes(data)
        images[name] = {"sha": sha256_bytes(data), "key": key, "width": size[0], "height": size[1], "bytes": data}

    inference = [
        _record("cid-a", images["train"], [10, 10, 30, 26], teacher="true"),    # human true + teacher
        _record("cid-b", images["train"], [40, 40, 48, 50], teacher="false"),   # teacher only
        _record("cid-c", images["train"], [2, 60, 10, 70], teacher=None),       # no labels -> skipped
        _record("cid-d", images["val"], [5, 5, 20, 20], teacher="uncertain"),   # human false; teacher dropped (val)
        _record("cid-e", images["locked"], [5, 5, 15, 15], teacher="true"),     # locked image -> fully excluded
    ]
    gold = [_gold("cid-a", "true"), _gold("cid-d", "false"), _gold("cid-e", "true")]

    snapshot = tmp_path / "snapshot"
    snapshot.mkdir()
    (snapshot / "inference_records.jsonl").write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in inference), encoding="utf-8"
    )
    (snapshot / "gold_records.jsonl").write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in gold), encoding="utf-8"
    )
    (snapshot / "snapshot_manifest.json").write_text(json.dumps({"snapshot_id": "pg1-fixture"}), encoding="utf-8")
    return snapshot, images_root, images


def _rows(dataset_dir: Path, name: str) -> list[dict]:
    path = dataset_dir / name
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def test_assign_train_val_buckets() -> None:
    # rule: mod5==0 -> excluded_locked, ==4 -> val, else train; image-level salt
    from tools.prompt_eval_data.common import sha256_text

    seen = set()
    for i in range(200):
        digest = int(sha256_text(f"prompt-gold-split-v1|fake-{i}"), 16) % 5
        seen.add(assign_train_val(f"fake-{i}"))
        expected = EXCLUDED if digest == 0 else (VAL if digest == 4 else TRAIN)
        assert assign_train_val(f"fake-{i}") == expected
    assert seen == {TRAIN, VAL, EXCLUDED}


def test_padded_crop_box_math() -> None:
    assert padded_crop_box([10, 10, 30, 26], 100, 100, 0.25) == (5, 6, 35, 30)
    # clamped at image edges
    assert padded_crop_box([0, 0, 8, 8], 100, 100, 0.5) == (0, 0, 12, 12)
    assert padded_crop_box([92, 92, 100, 100], 100, 100, 0.5) == (88, 88, 100, 100)


def test_render_crop_upscales_small_boxes() -> None:
    images_root_fixture = Image.new("RGB", (96, 72), (200, 10, 10))
    buf = io.BytesIO()
    images_root_fixture.save(buf, "PNG")
    out = render_crop(buf.getvalue(), (10, 10, 30, 26), MIN_EDGE)
    with Image.open(io.BytesIO(out)) as crop:
        # 20x16 crop -> long edge 20 upscaled to 64 => (64, 51)
        assert max(crop.size) == MIN_EDGE
        assert crop.size[0] / crop.size[1] == pytest.approx(20 / 16, rel=0.06)
        assert crop.format == "JPEG"
        r, _, b = crop.convert("RGB").getpixel((crop.size[0] // 2, crop.size[1] // 2))
        assert r > 150 and b < 90  # still the red half after pad+resize


def test_render_crop_keeps_large_boxes() -> None:
    im = Image.new("RGB", (300, 200), (10, 10, 200))
    buf = io.BytesIO()
    im.save(buf, "PNG")
    out = render_crop(buf.getvalue(), (10, 10, 200, 150), MIN_EDGE)
    with Image.open(io.BytesIO(out)) as crop:
        assert crop.size == (190, 140)  # no upscale when long edge >= MIN_EDGE


def test_build_dataset_isolation_and_labels(tmp_path: Path) -> None:
    snapshot, images_root, images = build_fixture(tmp_path)
    out_root = tmp_path / "out"
    manifest = build_crop_dataset(str(snapshot), str(out_root), image_root=str(images_root), min_long_edge=MIN_EDGE, workers=2)
    dataset = Path(manifest["dataset_dir"])

    human = {r["canonical_id"]: r for r in _rows(dataset, "labels_human.jsonl")}
    teacher = {r["canonical_id"]: r for r in _rows(dataset, "labels_teacher_gpt.jsonl")}

    # human labels: only gold on non-excluded images; never carries teacher fields
    assert set(human) == {"cid-a", "cid-d"}
    assert human["cid-a"]["label"] == "true" and human["cid-a"]["split"] == TRAIN
    assert human["cid-d"]["label"] == "false" and human["cid-d"]["split"] == VAL
    assert all("teacher_label" not in r for r in human.values())

    # teacher labels: train images only; val-image teacher dropped; never carries human label
    assert set(teacher) == {"cid-a", "cid-b"}
    assert all(r["split"] == TRAIN for r in teacher.values())
    assert all("label" not in r for r in teacher.values())
    assert manifest["counts"]["teacher_skipped_val_image"] == 1

    # crops: written for labeled non-excluded boxes only
    assert (dataset / human["cid-a"]["crop_path"]).exists()
    assert (dataset / teacher["cid-b"]["crop_path"]).exists()
    assert not list(dataset.glob("crops/excluded_locked/*"))
    assert human["cid-a"]["crop_path"].startswith("crops/train/")
    assert human["cid-d"]["crop_path"].startswith("crops/val/")

    # train/val image sets are disjoint
    train_sha = {r["image_sha256"] for r in human.values() if r["split"] == TRAIN}
    val_sha = {r["image_sha256"] for r in human.values() if r["split"] == VAL}
    assert train_sha.isdisjoint(val_sha)
    assert images["locked"]["sha"] not in train_sha | val_sha

    # counters agree with rows
    counts = manifest["counts"]
    assert counts["human_train"] == 1 and counts["human_val"] == 1
    assert counts["teacher_train"] == 2
    assert counts["boxes_excluded_locked"] == 1
    assert counts["boxes_no_label"] == 1
    assert counts["crops_written"] == 3

    # SHA256SUMS covers every payload file
    sums = (dataset / "SHA256SUMS").read_text(encoding="utf-8")
    for rel in ("labels_human.jsonl", "labels_teacher_gpt.jsonl", human["cid-a"]["crop_path"]):
        assert rel in sums


def test_build_dataset_deterministic(tmp_path: Path) -> None:
    snapshot, images_root, _ = build_fixture(tmp_path)
    m1 = build_crop_dataset(str(snapshot), str(tmp_path / "out1"), image_root=str(images_root), min_long_edge=MIN_EDGE, workers=1)
    m2 = build_crop_dataset(str(snapshot), str(tmp_path / "out2"), image_root=str(images_root), min_long_edge=MIN_EDGE, workers=1)
    assert m1["dataset_id"] == m2["dataset_id"]
    sums1 = (Path(m1["dataset_dir"]) / "SHA256SUMS").read_text(encoding="utf-8")
    sums2 = (Path(m2["dataset_dir"]) / "SHA256SUMS").read_text(encoding="utf-8")
    assert sums1 == sums2  # byte-identical payload, independent output roots

    # rebuild into the same root: idempotent, keeps existing dir
    m3 = build_crop_dataset(str(snapshot), str(tmp_path / "out1"), image_root=str(images_root), min_long_edge=MIN_EDGE, workers=1)
    assert m3["dataset_id"] == m1["dataset_id"]
