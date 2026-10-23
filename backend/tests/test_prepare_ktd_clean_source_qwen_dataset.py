from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys

import pytest
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))

from prepare_ktd_clean_source_qwen_dataset import (
    NEUTRAL_GRAY_RGB,
    VIEW_ROLES_2VIEW,
    VIEW_SCHEMA,
    _crop_square_with_metadata,
    build_dataset,
    draw_bbox_overlay,
    get_category_crop_spec,
    get_processor_pixel_budget,
)
from qwen_training_provenance import (
    validate_disjoint_image_splits,
    validate_human_gold_dataset,
)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def _source_fixture(tmp_path: Path) -> tuple[Path, Path]:
    source = tmp_path / "source"
    source.mkdir()
    pairs: list[dict] = []
    gold: list[dict] = []
    specs = (
        ("train-1", "train", "event-train", "task-train", "车", "true", "strong", True),
        ("val-1", "validation", "event-val", "task-val", "人员", "false", "usable", True),
        ("excluded", "train", "event-missing", "task-missing", "人员", "false", None, False),
    )
    for candidate_id, partition, event_id, task_id, class_name, label, status, has_clean in specs:
        row = {
            "candidate_id": candidate_id,
            "partition": partition,
            "event_id": event_id,
            "source_task_id": task_id,
            "class_name": class_name,
            "human_label": label,
            "bbox_px": [10, 12, 30, 40],
            "annotated_image_size": [80, 64],
            "has_clean_source": has_clean,
            "clean_match_status": status,
            "clean_source_kind": "primary" if has_clean else None,
        }
        annotated_dir = source / "event_annotated_frames"
        annotated_dir.mkdir(exist_ok=True)
        annotated_frame = annotated_dir / f"{event_id}.jpg"
        color_seed = sum(candidate_id.encode("utf-8")) % 200
        Image.new("RGB", (80, 64), (50 + color_seed % 100, 20, 10)).save(annotated_frame)
        clean_frame = source / f"{candidate_id}-clean-frame.jpg"
        Image.new("RGB", (100, 80), (10, 55 + color_seed % 120, 30)).save(clean_frame)
        row["clean_frame"] = clean_frame.name
        row["clean_frame_size"] = [100, 80]
        row["projected_clean_bbox_px"] = [12, 10, 44, 42]
        pairs.append(row)
        gold.append(
            {
                "candidate_id": candidate_id,
                "partition": partition,
                "event_id": event_id,
                "class_name": class_name,
                "human_label": label,
            }
        )
    pairs_path = source / "candidate_pairs.jsonl"
    gold_path = source / "gold_labels.jsonl"
    _write_jsonl(pairs_path, pairs)
    _write_jsonl(gold_path, gold)
    return pairs_path, gold_path


def _read(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def test_builds_self_contained_four_view_dataset_and_preserves_partition(tmp_path: Path):
    pairs, gold = _source_fixture(tmp_path)
    output = tmp_path / "output"

    summary = build_dataset(candidate_pairs_path=pairs, gold_path=gold, output_dir=output)

    train = _read(output / "train.jsonl")
    validation = _read(output / "validation.jsonl")
    lineage = _read(output / "lineage_manifest.jsonl")
    assert summary["eligible_candidates"] == 2
    assert summary["excluded_candidates"] == 1
    contract = summary["deployment_contract"]
    assert contract["view_roles"] == [
        "annotated_context",
        "annotated_roi",
        "clean_context",
        "clean_roi",
    ]
    assert contract["view_generation"] == "service_crop_v1_lossless_png"
    assert contract["crop_spec"]["clean_roi"]["output_size"] == 512
    assert contract["strict_output"]["allowed_decisions"] == [
        "CONFIRMED",
        "REJECTED",
    ]
    assert contract["processor"] == {
        "min_pixels": 50176,
        "max_pixels": 200704,
        "max_sequence_length": 2048,
    }
    assert len(contract["prompt_template_sha256"]) == 64
    assert summary["partition_rows"] == {"train": 1, "validation": 1}
    assert summary["isolation"] == {
        "annotated_frame_sha256_overlap": [],
        "clean_frame_sha256_overlap": [],
        "source_task_id_overlap": [],
        "event_id_overlap": [],
    }
    assert train[0]["meta"]["partition"] == "train"
    assert validation[0]["meta"]["partition"] == "validation"
    assert train[0]["meta"]["view_schema"] == VIEW_SCHEMA
    assert len(train[0]["image"]) == 4
    assert all(path.endswith(".png") for path in train[0]["image"])
    assert all((output / path).is_file() for path in train[0]["image"])
    assert [Image.open(output / path).size for path in train[0]["image"]] == [
        (768, 768), (512, 512), (768, 768), (512, 512)
    ]
    assert (output / train[0]["meta"]["annotated_frame"]).is_file()
    assert sum(row["eligible"] for row in lineage) == 2
    assert next(row for row in lineage if row["candidate_id"] == "excluded")["exclusion_reason"]


def test_four_view_provenance_hashes_and_split_isolation_are_enforced(tmp_path: Path):
    pairs, gold = _source_fixture(tmp_path)
    output = tmp_path / "output"
    build_dataset(candidate_pairs_path=pairs, gold_path=gold, output_dir=output)
    train_rows = _read(output / "train.jsonl")
    val_rows = _read(output / "validation.jsonl")

    train_summary = validate_human_gold_dataset(
        train_rows, dataset_path=output / "train.jsonl", provenance_root=output
    )
    val_summary = validate_human_gold_dataset(
        val_rows, dataset_path=output / "validation.jsonl", provenance_root=output
    )
    validate_disjoint_image_splits(train_summary, val_summary)
    assert train_summary["view_schemas"] == [VIEW_SCHEMA]
    assert train_summary["unique_image_sha256"] == 1
    assert len(train_summary["view_sha256s"]) == 4
    assert len(train_summary["annotated_frame_sha256s"]) == 1
    assert len(train_summary["clean_frame_sha256s"]) == 1

    (output / train_rows[0]["image"][2]).write_bytes(b"tampered")
    with pytest.raises(ValueError, match="clean_context training-view SHA-256 mismatch"):
        validate_human_gold_dataset(
            train_rows, dataset_path=output / "train.jsonl", provenance_root=output
        )


def test_rejects_candidate_gold_mismatch(tmp_path: Path):
    pairs, gold = _source_fixture(tmp_path)
    rows = _read(gold)
    rows[0]["human_label"] = "false"
    _write_jsonl(gold, rows)

    with pytest.raises(ValueError, match="differs between pair and gold"):
        build_dataset(candidate_pairs_path=pairs, gold_path=gold, output_dir=tmp_path / "output")


def test_category_adaptive_crop_specs_match_five_target_categories():
    person = get_category_crop_spec("人员")
    assert person["canonical_category"] == "person"
    assert person["bbox_margin"] == 0.20
    assert person["roi_scale"] == 5.0
    assert person["context_scale"] == 14.0

    crack = get_category_crop_spec("道路裂缝")
    assert crack["canonical_category"] == "road_crack"
    assert crack["bbox_margin"] == 0.15
    assert crack["roi_scale"] == 3.5
    assert crack["context_scale"] == 10.0
    assert crack["clean_roi_upscale"] == "cv2.INTER_CUBIC"

    roof = get_category_crop_spec("蓝色违建顶棚")
    assert roof["canonical_category"] == "blue_illegal_roof"
    assert roof["bbox_margin"] == 0.15
    assert roof["min_context_scale"] == 3.0
    assert roof["clamp_context_to_image"] is True

    tricycle = get_category_crop_spec("三轮车")
    vehicle = get_category_crop_spec("vehicle")
    assert tricycle["bbox_margin"] == vehicle["bbox_margin"] == 0.12
    assert tricycle["roi_scale"] == vehicle["roi_scale"] == 4.0
    assert tricycle["context_scale"] == vehicle["context_scale"] == 12.0


def test_corner_ticks_leaves_mid_edges_unoccluded():
    import numpy as np

    blank = np.full((120, 120, 3), 50, dtype=np.uint8)
    drawn = draw_bbox_overlay(blank, (30, 30, 90, 90), style="corner_ticks", color=(255, 0, 0))
    # Corner (28, 28) is red
    assert tuple(int(c) for c in drawn[28, 28]) == (255, 0, 0)
    # Mid-top edge (28, 60) and mid-left edge (60, 28) remain untouched background (50, 50, 50)
    assert tuple(int(c) for c in drawn[28, 60]) == (50, 50, 50)
    assert tuple(int(c) for c in drawn[60, 28]) == (50, 50, 50)


def test_edge_crop_shifts_viewport_or_pads_constant_neutral_gray_without_mirror_artifacts():
    import numpy as np

    # Create a 64x64 dark image with a bright stripe near the left edge
    arr = np.full((64, 64, 3), 20, dtype=np.uint8)
    arr[:, 2:5] = (200, 10, 10)
    image = Image.fromarray(arr)
    # Request oversized crop (min_side=128 > 64) so padding is required
    cropped, meta = _crop_square_with_metadata(
        image,
        (2 / 64, 10 / 64, 12 / 64, 20 / 64),
        factor=3.5,
        min_side=128,
        output_size=128,
        class_name="道路裂缝",
        role="clean_roi",
        bbox_style="none",
    )
    cropped_arr = np.asarray(cropped)
    assert any(p > 0 for p in meta["padding"])
    # Top-left padded corner must be constant neutral gray (230, 230, 230), never reflected pixels
    assert tuple(int(c) for c in cropped_arr[0, 0]) == NEUTRAL_GRAY_RGB


def _gray_fill_ratio(view: Image.Image) -> float:
    import numpy as np

    array = np.asarray(view)
    return float(np.all(array == np.array(NEUTRAL_GRAY_RGB, dtype=np.uint8), axis=-1).mean())


def test_large_target_roi_viewport_converges_on_image_short_side_without_gray_padding():
    """回归 F5：700x400 违建顶棚在 1920x1080 帧内按 4x 放大后曾达 2800px 视口，灰边 73%。"""
    import numpy as np

    frame_w, frame_h = 1920, 1080
    frame = Image.fromarray(np.full((frame_h, frame_w, 3), 90, dtype=np.uint8))
    x1, y1, x2, y2 = 610, 340, 1310, 740
    bbox = (x1 / frame_w, y1 / frame_h, x2 / frame_w, y2 / frame_h)

    view, meta = _crop_square_with_metadata(
        frame,
        bbox,
        factor=4.0,
        min_side=96,
        output_size=512,
        class_name="蓝色违建顶棚",
        role="annotated_roi",
        bbox_style="corner_ticks",
    )

    assert meta["roi_side_clamped"] is True
    assert meta["desired_side"] == 2800
    assert meta["viewport_side"] == frame_h
    assert meta["padding"] == [0, 0, 0, 0]
    assert _gray_fill_ratio(view) <= 0.15
    # The target box plus its corner-tick margin stays fully inside the viewport.
    sx1, sy1, sx2, sy2 = meta["shifted_bbox_px"]
    assert 0 <= sx1 < sx2 <= 512
    assert 0 <= sy1 < sy2 <= 512
    assert sx1 >= 4 and sy1 >= 4 and sx2 <= 508 and sy2 <= 508


def test_roi_viewport_never_exceeds_the_image_long_side_for_oversized_targets():
    import numpy as np

    frame_w, frame_h = 1920, 1080
    frame = Image.fromarray(np.full((frame_h, frame_w, 3), 90, dtype=np.uint8))
    # A crack spanning almost the full frame width cannot fit the short side.
    bbox = (60 / frame_w, 500 / frame_h, 1860 / frame_w, 600 / frame_h)
    view, meta = _crop_square_with_metadata(
        frame,
        bbox,
        factor=3.5,
        min_side=96,
        output_size=512,
        class_name="道路裂缝",
        role="clean_roi",
        bbox_style="none",
    )
    assert meta["viewport_side"] <= frame_w
    assert meta["viewport_side"] == frame_w
    assert meta["padding"] == [0, 0, 0, 0] or meta["padding"][0] == meta["padding"][1]
    # The full-width target is still contained horizontally (1800 of 1920 px -> ~480 of 512).
    sx1, _, sx2, _ = meta["shifted_bbox_px"]
    assert 0 <= sx1 < sx2 <= 512
    assert (sx2 - sx1) >= 470


def test_small_target_roi_keeps_requested_zoom_instead_of_being_clamped():
    import numpy as np

    frame = Image.fromarray(np.full((1080, 1920, 3), 90, dtype=np.uint8))
    bbox = (100 / 1920, 100 / 1080, 200 / 1920, 200 / 1080)
    _, meta = _crop_square_with_metadata(
        frame,
        bbox,
        factor=5.0,
        min_side=96,
        output_size=512,
        class_name="人员",
        role="annotated_roi",
        bbox_style="corner_ticks",
    )
    assert meta["roi_side_clamped"] is False
    assert meta["viewport_side"] == 500
    assert meta["padding"] == [0, 0, 0, 0]


def test_roof_context_view_is_never_tighter_than_its_roi_view():
    import numpy as np

    frame = Image.fromarray(np.full((1080, 1920, 3), 90, dtype=np.uint8))
    bbox = (610 / 1920, 340 / 1080, 1310 / 1920, 740 / 1080)
    _, context_meta = _crop_square_with_metadata(
        frame,
        bbox,
        factor=12.0,
        min_side=320,
        output_size=768,
        class_name="蓝色违建顶棚",
        role="annotated_context",
        bbox_style="corner_ticks",
    )
    _, roi_meta = _crop_square_with_metadata(
        frame,
        bbox,
        factor=4.0,
        min_side=96,
        output_size=512,
        class_name="蓝色违建顶棚",
        role="annotated_roi",
        bbox_style="corner_ticks",
    )
    assert context_meta["viewport_side"] >= roi_meta["viewport_side"]


def test_large_roof_in_real_monitoring_frame_is_not_padded():
    """真实监控帧（960x720）里的 700x400 违建顶棚：视口收敛到短边 720，灰边归零。"""
    import numpy as np

    frame = Image.fromarray(np.full((720, 960, 3), 90, dtype=np.uint8))
    bbox = (100 / 960, 150 / 720, 800 / 960, 550 / 720)
    view, meta = _crop_square_with_metadata(
        frame,
        bbox,
        factor=4.0,
        min_side=96,
        output_size=512,
        class_name="蓝色违建顶棚",
        role="annotated_roi",
        bbox_style="corner_ticks",
    )
    assert meta["viewport_side"] == 720
    assert meta["padding"] == [0, 0, 0, 0]
    assert _gray_fill_ratio(view) == 0.0
    sx1, sy1, sx2, sy2 = meta["shifted_bbox_px"]
    # 8px tick margin (2px offset + stroke) must stay inside the 512px output.
    assert 0 < sx1 and 0 < sy1 and sx2 < 512 and sy2 < 512


def test_builds_2view_dataset_and_passes_provenance_validation(tmp_path: Path):
    pairs, gold = _source_fixture(tmp_path)
    output = tmp_path / "output-2view"

    summary = build_dataset(
        candidate_pairs_path=pairs,
        gold_path=gold,
        output_dir=output,
        view_mode="2view",
        bbox_style="corner_ticks",
    )

    train = _read(output / "train.jsonl")
    validation = _read(output / "validation.jsonl")
    contract = summary["deployment_contract"]
    assert contract["view_mode"] == "2view"
    assert contract["bbox_style"] == "corner_ticks"
    assert contract["view_roles"] == list(VIEW_ROLES_2VIEW)
    assert contract["processor"] == get_processor_pixel_budget("2view")
    assert len(train[0]["image"]) == 2
    assert [Image.open(output / path).size for path in train[0]["image"]] == [
        (768, 768),
        (512, 512),
    ]
    train_summary = validate_human_gold_dataset(
        train, dataset_path=output / "train.jsonl", provenance_root=output
    )
    val_summary = validate_human_gold_dataset(
        validation, dataset_path=output / "validation.jsonl", provenance_root=output
    )
    validate_disjoint_image_splits(train_summary, val_summary)
    assert len(train_summary["view_sha256s"]) == 2
