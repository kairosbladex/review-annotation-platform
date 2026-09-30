"""Direct tests for the provenance-gated human-gold expansion queue."""

from __future__ import annotations

import copy
import hashlib
import json
import sys
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.prompt_eval_data.build_gold_expansion_queue import (  # noqa: E402
    SIZE_BINS,
    _clean_declaration_sha,
    _size_bin,
    build_queue,
    main,
)
from tools.prompt_eval_data.splits import LOCKED_TEST, SELECTION, assign_split  # noqa: E402


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def _sha_for_split(prefix: str, split: str) -> str:
    for nonce in range(100):
        value = hashlib.sha256(f"{prefix}-{nonce}".encode("utf-8")).hexdigest()
        if assign_split(value) == split:
            return value
    raise AssertionError(f"could not find {split} sha for {prefix}")


def _task_anchor_for_split(prefix: str, split: str) -> tuple[str, str]:
    for nonce in range(100):
        value = f"{prefix}-{nonce}"
        identity = f"task:{value}"
        split_key = "source-group:" + hashlib.sha256(identity.encode("utf-8")).hexdigest()
        if assign_split(split_key) == split:
            return value, split_key
    raise AssertionError(f"could not find {split} task anchor for {prefix}")


def _set_task_group(row: dict, anchor_value: str) -> str:
    anchor = f"task:{anchor_value}"
    split_key = "source-group:" + hashlib.sha256(anchor.encode("utf-8")).hexdigest()
    row["provenance"]["source_records"][0]["metadata"]["task_id"] = anchor_value
    row["provenance"]["grouping"] = {
        "status": "source_group_available",
        "split_key": split_key,
        "identities": [
            {"kind": "task", "value": anchor_value, "origin": "metadata.task_id"}
        ],
        "component_identities": [anchor],
        "anchor_identities": [anchor],
        "anchor_kind": "task",
        "basis": "highest_scope_unique_anchor_in_connected_component",
    }
    return split_key


def _input(
    canonical_id: str,
    image_sha256: str,
    bbox_px: list[int],
    *,
    class_id: int = 8,
    class_name: str = "car",
) -> dict:
    return {
        "canonical_id": canonical_id,
        "class_id": class_id,
        "class_name": class_name,
        "bbox_norm": [0.0, 0.0, 0.1, 0.1],
        "bbox_px": bbox_px,
        "image_sha256": image_sha256,
        "image_object_key": f"images/{image_sha256}.jpg",
        "image_width": 640,
        "image_height": 480,
        "detector_confidence": 0.5,
    }


def _inference(
    canonical_id: str,
    image_sha256: str,
    bbox_px: list[int],
    *,
    source_status: str | None,
    class_id: int = 8,
    class_name: str = "car",
) -> dict:
    candidate_id = f"candidate-{canonical_id}"
    sample_id = f"sample-{canonical_id}"
    image_object_key = f"images/{image_sha256}.jpg"
    anchor_value, _ = _task_anchor_for_split(canonical_id, assign_split(image_sha256))
    metadata = {
        "task_id": anchor_value,
        "event_time": "2026-09-03T02:00:00Z",
    }
    source_record = {
        "candidate_id": candidate_id,
        "sample_id": sample_id,
        "model_run_id": f"run-{canonical_id}",
        "target_index": 0,
        "source_event_id": None,
        "source_event_semantics": "missing",
        "session_id": None,
        "metadata": metadata,
        "asset": {
            "object_key": image_object_key,
            "declared_sha256": image_sha256,
        },
    }
    row = {
        **_input(
            canonical_id,
            image_sha256,
            bbox_px,
            class_id=class_id,
            class_name=class_name,
        ),
        "prior_gpt": {"passes": [{"label": "false"}]},
        "prior_qwen": {"attempts": [{"label": "true"}]},
        "prior_yolo": {"decision": "uncertain"},
        "alias_count": 1,
        "aliases": [
            {
                "candidate_id": candidate_id,
                "sample_id": sample_id,
                "model_run_id": f"run-{canonical_id}",
                "target_index": 0,
            }
        ],
        "has_human_gold": False,
    }
    row["provenance"] = {
        "source_records": [source_record],
        "temporal": {
            "status": "upstream_time_available",
            "selected_field": "metadata.event_time",
            "selected_value": "2026-09-03T02:00:00Z",
            "missing_samples": [],
            "evidence": [
                {
                    "sample_id": sample_id,
                    "field": "metadata.event_time",
                    "value": "2026-09-03T02:00:00Z",
                    "normalized_value": "2026-09-03T02:00:00Z",
                    "validation_error": None,
                }
            ],
        },
    }
    _set_task_group(row, anchor_value)
    if source_status == "clean":
        source_payload = {
            "overlay_status": "clean",
            "clean_object_key": image_object_key,
            "clean_object_version": "fixture-v1",
            "clean_sha256": image_sha256,
            "clean_width": 640,
            "clean_height": 480,
            "annotated_object_key": image_object_key,
            "annotated_object_version": "fixture-v1",
            "annotated_sha256": image_sha256,
            "annotated_width": 640,
            "annotated_height": 480,
            "overlay_check_method": "fixture-human-review",
            "overlay_check_version": "1",
            "overlay_evidence_sha256": hashlib.sha256(
                f"overlay-{canonical_id}".encode("utf-8")
            ).hexdigest(),
            "mapping_method": "same_asset",
            "mapping_version": "1",
            "geometry_transform": "identity",
            "source_system": "fixture",
            "source_table": "samples",
            "source_row_id": sample_id,
            "frame_id": f"frame-{canonical_id}",
        }
        metadata["source_pixels"] = source_payload
        row["provenance"]["source_pixels"] = {
            "status": "clean",
            "exported_object_key": image_object_key,
            "exported_sha256": image_sha256,
            "formal_input_allowed": True,
            "blocking_reasons": [],
            "declarations": [
                {
                    "candidate_id": candidate_id,
                    "sample_id": sample_id,
                    "declared_status": "clean",
                    "normalized_status": "clean",
                    "validation_errors": [],
                    "declaration_sha256": _clean_declaration_sha(source_payload),
                }
            ],
        }
    elif source_status == "overlay":
        metadata["source_pixels"] = {"overlay_status": "overlay"}
        row["provenance"]["source_pixels"] = {
            "status": "overlay",
            "formal_input_allowed": False,
            "blocking_reasons": ["overlay_declared"],
            "declarations": [
                {
                    "candidate_id": candidate_id,
                    "sample_id": sample_id,
                    "declared_status": "overlay",
                    "normalized_status": "overlay",
                    "validation_errors": [],
                }
            ],
        }
    else:
        row["provenance"]["source_pixels"] = {
            "status": "unknown",
            "formal_input_allowed": False,
            "blocking_reasons": ["missing_source_pixel_declaration"],
            "declarations": [],
        }
    return row


def _refresh_clean_declaration(row: dict, index: int = 0) -> None:
    record = row["provenance"]["source_records"][index]
    payload = record["metadata"]["source_pixels"]
    declaration = row["provenance"]["source_pixels"]["declarations"][index]
    declaration["declared_status"] = payload["overlay_status"]
    declaration["declaration_sha256"] = _clean_declaration_sha(payload)


def _add_alias(
    row: dict,
    suffix: str,
    *,
    include_pixel_declaration: bool = True,
    include_time: bool = True,
) -> None:
    first_alias = row["aliases"][0]
    first_record = row["provenance"]["source_records"][0]
    candidate_id = f"{first_alias['candidate_id']}-{suffix}"
    sample_id = f"{first_alias['sample_id']}-{suffix}"
    alias = copy.deepcopy(first_alias)
    alias.update({"candidate_id": candidate_id, "sample_id": sample_id})
    record = copy.deepcopy(first_record)
    record.update({"candidate_id": candidate_id, "sample_id": sample_id})
    if not include_time:
        record["metadata"].pop("event_time", None)
    source_pixels = record["metadata"].get("source_pixels")
    if isinstance(source_pixels, dict):
        source_pixels["source_row_id"] = sample_id
    row["aliases"].append(alias)
    row["alias_count"] = len(row["aliases"])
    row["provenance"]["source_records"].append(record)
    if include_pixel_declaration and isinstance(source_pixels, dict):
        row["provenance"]["source_pixels"]["declarations"].append(
            {
                "candidate_id": candidate_id,
                "sample_id": sample_id,
                "declared_status": source_pixels["overlay_status"],
                "normalized_status": "clean",
                "validation_errors": [],
                "declaration_sha256": _clean_declaration_sha(source_pixels),
            }
        )
    if include_time:
        value = record["metadata"]["event_time"]
        row["provenance"]["temporal"]["evidence"].append(
            {
                "sample_id": sample_id,
                "field": "metadata.event_time",
                "value": value,
                "normalized_value": value,
                "validation_error": None,
            }
        )


def _make_snapshot(
    root: Path,
    *,
    inference: list[dict],
    selection_inputs: list[dict] | None = None,
    selection_gold: list[dict] | None = None,
    quarantine: list[dict] | None = None,
    locked_inputs: list[dict] | None = None,
) -> Path:
    readiness = {"formal_training_ready": True, "blockers": []}
    encode_jsonl = lambda rows: "".join(  # noqa: E731 - compact fixture encoder
        json.dumps(row, ensure_ascii=False) + "\n"
        for row in sorted(rows, key=lambda item: item.get("canonical_id", ""))
    ).encode("utf-8")
    files = {
        "inference_records.jsonl": encode_jsonl(inference),
        "selection_inputs.jsonl": encode_jsonl(selection_inputs or []),
        "selection_gold.jsonl": encode_jsonl(selection_gold or []),
        "quarantine.jsonl": encode_jsonl(quarantine or []),
        "locked_test_inputs.jsonl": encode_jsonl(locked_inputs or []),
        "source_readiness.json": (
            json.dumps(readiness, sort_keys=True) + "\n"
        ).encode("utf-8"),
        # The queue builder may verify these hashes but must not parse the content.
        "gold_records.jsonl": b"not-json\n",
        "locked_test_gold.jsonl": b"not-json\n",
    }
    snapshot_payload = "".join(
        f"{name}:{hashlib.sha256(payload).hexdigest()}\n"
        for name, payload in sorted(files.items())
    ).encode("utf-8")
    snapshot_id = "pg2-" + hashlib.sha256(snapshot_payload).hexdigest()[:12]
    root = root.parent / snapshot_id
    root.mkdir(parents=True)
    files["SHA256SUMS"] = "".join(
        f"{hashlib.sha256(payload).hexdigest()}  {name}\n"
        for name, payload in sorted(files.items())
    ).encode("utf-8")
    for name, payload in files.items():
        (root / name).write_bytes(payload)
    manifest = {
        "schema_version": "prompt-gold/2",
        "snapshot_id": snapshot_id,
        "readiness": readiness,
        "files": [
            {
                "path": name,
                "sha256": hashlib.sha256(payload).hexdigest(),
                "bytes": len(payload),
            }
            for name, payload in sorted(files.items())
        ],
    }
    (root / "snapshot_manifest.json").write_text(
        json.dumps(manifest, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return root


@pytest.mark.parametrize(
    ("bbox_px", "expected"),
    [
        ([0, 0, 15, 1], "tiny_lt16"),
        ([0, 0, 16, 1], "small_16_47"),
        ([0, 0, 47, 1], "small_16_47"),
        ([0, 0, 48, 1], "medium_48_95"),
        ([0, 0, 95, 1], "medium_48_95"),
        ([0, 0, 96, 1], "large_ge96"),
        ([0, 0, 1, 96], "large_ge96"),
    ],
)
def test_size_bin_uses_original_pixel_long_edge(bbox_px: list[int], expected: str) -> None:
    assert _size_bin({"bbox_px": bbox_px, "bbox_norm": [0.0, 0.0, 0.01, 0.01]}) == expected


@pytest.mark.parametrize("bbox_px", ([0, 0, 0, 1], [4, 0, 3, 1], [0, 4, 1, 3]))
def test_size_bin_rejects_non_positive_boxes(bbox_px: list[int]) -> None:
    with pytest.raises(ValueError, match="bbox_px must have positive width and height"):
        _size_bin({"bbox_px": bbox_px})


def test_build_queue_gates_provenance_and_reports_gold_gaps_deterministically(tmp_path: Path) -> None:
    existing_sha = _sha_for_split("existing", SELECTION)
    quarantined_sha = _sha_for_split("quarantined", SELECTION)
    prior_locked_sha = _sha_for_split("prior-locked", LOCKED_TEST)
    future_locked_sha = _sha_for_split("future-locked", LOCKED_TEST)
    clean_tiny_sha = _sha_for_split("clean-tiny", SELECTION)
    clean_small_sha = _sha_for_split("clean-small", SELECTION)
    clean_medium_sha = _sha_for_split("clean-medium", SELECTION)
    unknown_sha = _sha_for_split("unknown", SELECTION)
    overlay_sha = _sha_for_split("overlay", SELECTION)

    selection_inputs = [_input("gold", existing_sha, [0, 0, 48, 1])]
    selection_gold = [
        {"canonical_id": "gold", "class_id": 8, "class_name": "car", "label": "false"}
    ]
    locked_inputs = [_input("old-locked", prior_locked_sha, [0, 0, 20, 1])]
    quarantine = [
        {
            "canonical_id": "quarantined",
            "class_id": 8,
            "class_name": "car",
            "bbox_px": [0, 0, 20, 1],
            "image_sha256": quarantined_sha,
        }
    ]
    inference = [
        _inference("same-gold-image", existing_sha, [0, 0, 15, 1], source_status="clean"),
        _inference("same-quarantine-image", quarantined_sha, [0, 0, 16, 1], source_status="clean"),
        _inference("same-locked-image", prior_locked_sha, [0, 0, 48, 1], source_status="clean"),
        _inference("clean-tiny", clean_tiny_sha, [0, 0, 15, 1], source_status="clean"),
        _inference("clean-small", clean_small_sha, [0, 0, 16, 1], source_status="clean"),
        _inference(
            "clean-medium",
            clean_medium_sha,
            [0, 0, 48, 1],
            source_status="clean",
            class_id=6,
            class_name="truck",
        ),
        _inference("future-locked", future_locked_sha, [0, 0, 96, 1], source_status="clean"),
        _inference("unknown", unknown_sha, [0, 0, 20, 1], source_status=None),
        _inference("overlay", overlay_sha, [0, 0, 20, 1], source_status="overlay"),
    ]
    snapshot = _make_snapshot(
        tmp_path / "snapshot",
        inference=inference,
        selection_inputs=selection_inputs,
        selection_gold=selection_gold,
        quarantine=quarantine,
        locked_inputs=locked_inputs,
    )

    selected, blocked, summary = build_queue(snapshot, max_per_class=10)
    assert {row["canonical_id"] for row in selected} == {
        "clean-tiny",
        "clean-small",
        "clean-medium",
    }
    assert [row["size_bin"] for row in selected if row["class_name"] == "car"] == [
        "tiny_lt16",
        "small_16_47",
    ]
    assert all(row["planned_split"] == SELECTION for row in selected)
    assert all(row["source_pixel_status"] == "clean" and row["label"] is None for row in selected)
    assert not any(
        key in row
        for row in selected
        for key in ("prior_gpt", "prior_qwen", "prior_yolo", "aliases", "has_human_gold")
    )

    assert {row["canonical_id"]: row["blocked_reasons"] for row in blocked} == {
        "overlay": ["source_pixels_overlay"],
        "unknown": ["source_pixels_unknown"],
    }
    assert summary["blocked_source_by_status"] == {"unknown": 1, "overlay": 1}
    assert summary["planned_locked_boxes_excluded"] == 1
    assert summary["pending_label_policy"]["can_prebalance_true_false"] is False

    coverage = summary["selection_human_gold_coverage"]
    assert coverage["boxes"] == 1
    assert coverage["by_label"] == {"false": 1, "true": 0}
    assert coverage["by_size_bin"] == {
        "tiny_lt16": 0,
        "small_16_47": 0,
        "medium_48_95": 1,
        "large_ge96": 0,
    }
    cells = {
        (row["class_name"], row["label"], row["size_bin"]): row
        for row in coverage["class_label_size_cells"]
    }
    assert cells[("car", "false", "medium_48_95")] == {
        "class_name": "car",
        "label": "false",
        "size_bin": "medium_48_95",
        "boxes": 1,
        "images": 1,
    }
    assert coverage["gap_cells"] == 15  # two classes x two labels x four bins, one occupied

    first = (selected, blocked, summary)
    reversed_snapshot = _make_snapshot(
        tmp_path / "reversed" / "snapshot",
        inference=list(reversed(inference)),
        selection_inputs=selection_inputs,
        selection_gold=selection_gold,
        quarantine=quarantine,
        locked_inputs=locked_inputs,
    )
    assert build_queue(reversed_snapshot, max_per_class=10) == first

def test_current_v1_snapshot_is_empty_and_blocked_without_clean_provenance() -> None:
    snapshot = REPO_ROOT / "data/prompt-eval/snapshots/pg1-bc8c818184d8"

    selected, blocked, summary = build_queue(snapshot)

    assert selected == []
    assert len(blocked) == 2273
    assert summary["candidate_by_source_pixel_status"] == {
        "clean": 0,
        "unknown": 2273,
        "overlay": 0,
    }
    assert summary["blocked_source_by_status"] == {"unknown": 2273, "overlay": 0}
    assert summary["planned_locked_boxes_excluded"] == 500
    assert summary["planned_locked_images_excluded"] == 104
    assert summary["selection_human_gold_coverage"]["boxes"] == 438
    assert summary["selection_human_gold_coverage"]["gap_cells"] == 46


def test_main_writes_complete_artifact_and_refuses_existing_output(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    unknown_sha = _sha_for_split("main-unknown", SELECTION)
    snapshot = _make_snapshot(
        tmp_path / "snapshot",
        inference=[_inference("unknown", unknown_sha, [0, 0, 20, 1], source_status=None)],
    )
    output = tmp_path / "new-output"
    monkeypatch.setattr(
        sys,
        "argv",
        ["build_gold_expansion_queue.py", "--snapshot", str(snapshot), "--output-dir", str(output)],
    )

    assert main() == 0
    assert (output / "queue.jsonl").read_text(encoding="utf-8") == ""
    blocked = [json.loads(line) for line in (output / "blocked_sources.jsonl").read_text().splitlines()]
    assert [row["canonical_id"] for row in blocked] == ["unknown"]
    assert json.loads((output / "summary.json").read_text())["blocked_source_boxes"] == 1
    artifact_manifest = json.loads((output / "artifact_manifest.json").read_text())
    assert artifact_manifest["schema_version"] == "gold-expansion-artifact/1"
    assert artifact_manifest["source_snapshot"]["manifest_sha256"] == hashlib.sha256(
        (snapshot / "snapshot_manifest.json").read_bytes()
    ).hexdigest()
    assert {entry["path"] for entry in artifact_manifest["files"]} == {
        "README.md",
        "blocked_sources.jsonl",
        "queue.jsonl",
        "summary.json",
    }
    assert {
        entry["path"]: entry["audience"] for entry in artifact_manifest["files"]
    } == {
        "README.md": "all_operators",
        "blocked_sources.jsonl": "source_recovery",
        "queue.jsonl": "blind_annotator",
        "summary.json": "management_only",
    }
    assert "SHA256SUMS" in {
        entry["path"] for entry in artifact_manifest["source_snapshot"]["members"]
    }
    for entry in artifact_manifest["files"]:
        payload = (output / entry["path"]).read_bytes()
        assert entry["sha256"] == hashlib.sha256(payload).hexdigest()
        assert entry["bytes"] == len(payload)

    old_v2 = tmp_path / "gold-expansion-20260904-v2"
    old_v2.mkdir()
    sentinel = old_v2 / "queue.jsonl"
    sentinel.write_text("existing-v2\n", encoding="utf-8")
    monkeypatch.setattr(
        sys,
        "argv",
        ["build_gold_expansion_queue.py", "--snapshot", str(snapshot), "--output-dir", str(old_v2)],
    )
    with pytest.raises(FileExistsError, match="refusing to replace existing output"):
        main()
    assert sentinel.read_text(encoding="utf-8") == "existing-v2\n"


def test_build_queue_rejects_mismatched_selection_views(tmp_path: Path) -> None:
    sha = _sha_for_split("mismatch", SELECTION)
    snapshot = _make_snapshot(
        tmp_path / "snapshot",
        inference=[],
        selection_inputs=[_input("input-only", sha, [0, 0, 20, 1])],
        selection_gold=[
            {"canonical_id": "gold-only", "class_id": 8, "class_name": "car", "label": "true"}
        ],
    )

    with pytest.raises(ValueError, match="selection gold/input canonical_id mismatch"):
        build_queue(snapshot)


def test_build_queue_rejects_duplicate_inference_canonical_ids(tmp_path: Path) -> None:
    first_sha = _sha_for_split("duplicate-first", SELECTION)
    second_sha = _sha_for_split("duplicate-second", SELECTION)
    snapshot = _make_snapshot(
        tmp_path / "snapshot",
        inference=[
            _inference("duplicate", first_sha, [0, 0, 20, 1], source_status="clean"),
            _inference("duplicate", second_sha, [0, 0, 20, 1], source_status="clean"),
        ],
    )

    with pytest.raises(ValueError, match="inference_records.jsonl: duplicate canonical_id"):
        build_queue(snapshot)


@pytest.mark.parametrize("overlap_field", ("image_sha256", "split_group_key"))
def test_build_queue_rejects_selection_locked_input_overlap(
    tmp_path: Path, overlap_field: str
) -> None:
    selection_sha = _sha_for_split("split-selection", SELECTION)
    locked_sha = selection_sha if overlap_field == "image_sha256" else _sha_for_split(
        "split-locked", LOCKED_TEST
    )
    selection = _input("selection-gold", selection_sha, [0, 0, 20, 1])
    locked = _input("locked-gold", locked_sha, [0, 0, 20, 1])
    selection["split_group_key"] = "source-group:shared" if overlap_field == "split_group_key" else "source-group:selection"
    locked["split_group_key"] = "source-group:shared" if overlap_field == "split_group_key" else "source-group:locked"
    snapshot = _make_snapshot(
        tmp_path / "snapshot",
        inference=[],
        selection_inputs=[selection],
        selection_gold=[
            {"canonical_id": "selection-gold", "class_id": 8, "class_name": "car", "label": "true"}
        ],
        locked_inputs=[locked],
    )

    with pytest.raises(ValueError, match=f"selection and locked {overlap_field} overlap"):
        build_queue(snapshot)


def test_build_queue_rejects_snapshot_member_hash_mismatch(tmp_path: Path) -> None:
    sha = _sha_for_split("tampered", SELECTION)
    snapshot = _make_snapshot(
        tmp_path / "snapshot",
        inference=[_inference("candidate", sha, [0, 0, 20, 1], source_status="clean")],
    )
    with (snapshot / "inference_records.jsonl").open("a", encoding="utf-8") as fh:
        fh.write("{}\n")

    with pytest.raises(ValueError, match="snapshot member hash mismatch"):
        build_queue(snapshot)


def test_build_queue_rejects_sha256sums_manifest_disagreement(tmp_path: Path) -> None:
    sha = _sha_for_split("bad-sums", SELECTION)
    snapshot = _make_snapshot(
        tmp_path / "snapshot",
        inference=[_inference("candidate", sha, [0, 0, 20, 1], source_status="clean")],
    )
    sums_path = snapshot / "SHA256SUMS"
    sums = sums_path.read_text(encoding="utf-8").replace(
        hashlib.sha256((snapshot / "inference_records.jsonl").read_bytes()).hexdigest(),
        "0" * 64,
        1,
    )
    sums_path.write_text(sums, encoding="utf-8")
    manifest_path = snapshot / "snapshot_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    sums_payload = sums_path.read_bytes()
    sums_entry = next(entry for entry in manifest["files"] if entry["path"] == "SHA256SUMS")
    sums_entry.update(
        {"sha256": hashlib.sha256(sums_payload).hexdigest(), "bytes": len(sums_payload)}
    )
    manifest_path.write_text(json.dumps(manifest, sort_keys=True) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="SHA256SUMS does not match snapshot manifest"):
        build_queue(snapshot)


def test_build_queue_rejects_non_content_addressed_pg2_id(tmp_path: Path) -> None:
    sha = _sha_for_split("bad-id", SELECTION)
    snapshot = _make_snapshot(
        tmp_path / "snapshot",
        inference=[_inference("candidate", sha, [0, 0, 20, 1], source_status="clean")],
    )
    bad_snapshot = snapshot.with_name("pg2-000000000000")
    manifest_path = snapshot / "snapshot_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["snapshot_id"] = bad_snapshot.name
    manifest_path.write_text(json.dumps(manifest, sort_keys=True) + "\n", encoding="utf-8")
    snapshot.rename(bad_snapshot)

    with pytest.raises(ValueError, match="snapshot ID does not match member hashes"):
        build_queue(bad_snapshot)


def test_build_queue_parses_the_same_bytes_that_were_hash_verified(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    sha = _sha_for_split("verified-bytes", SELECTION)
    snapshot = _make_snapshot(
        tmp_path / "snapshot",
        inference=[_inference("candidate", sha, [0, 0, 20, 1], source_status="clean")],
    )
    target = snapshot / "inference_records.jsonl"
    original_read_bytes = Path.read_bytes
    target_reads = 0

    def read_bytes_once(path: Path) -> bytes:
        nonlocal target_reads
        payload = original_read_bytes(path)
        if path == target:
            target_reads += 1
            path.write_text("not-json\n", encoding="utf-8")
        return payload

    monkeypatch.setattr(Path, "read_bytes", read_bytes_once)

    selected, _, _ = build_queue(snapshot)

    assert [row["canonical_id"] for row in selected] == ["candidate"]
    assert target_reads == 1


@pytest.mark.parametrize(
    ("target_view", "actual_split"),
    (("selection", LOCKED_TEST), ("locked", SELECTION)),
)
def test_build_queue_rejects_input_rows_assigned_to_the_wrong_split(
    tmp_path: Path, target_view: str, actual_split: str
) -> None:
    sha = _sha_for_split(f"wrong-{target_view}", actual_split)
    input_row = _input("gold", sha, [0, 0, 20, 1])
    kwargs = {
        "selection_inputs": [input_row] if target_view == "selection" else [],
        "selection_gold": (
            [{"canonical_id": "gold", "class_id": 8, "class_name": "car", "label": "true"}]
            if target_view == "selection"
            else []
        ),
        "locked_inputs": [input_row] if target_view == "locked" else [],
    }
    snapshot = _make_snapshot(tmp_path / "snapshot", inference=[], **kwargs)

    expected_name = (
        "selection_inputs.jsonl" if target_view == "selection" else "locked_test_inputs.jsonl"
    )
    with pytest.raises(ValueError, match=f"{expected_name}: row assigned"):
        build_queue(snapshot)


def test_build_queue_rejects_available_group_without_split_key(tmp_path: Path) -> None:
    sha = _sha_for_split("missing-key", SELECTION)
    row = _inference("missing-key", sha, [0, 0, 20, 1], source_status="clean")
    del row["provenance"]["grouping"]["split_key"]
    snapshot = _make_snapshot(tmp_path / "snapshot", inference=[row])

    with pytest.raises(ValueError, match="source group provenance is not validated"):
        build_queue(snapshot)


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("clean_object_key", "images/other.jpg"),
        ("clean_sha256", "b" * 64),
        ("clean_width", 641),
        ("annotated_object_version", "fixture-v2"),
        ("mapping_method", "affine"),
        ("geometry_transform", {"type": "affine"}),
        ("overlay_evidence_sha256", "not-a-sha"),
    ),
)
def test_build_queue_rejects_self_consistent_but_invalid_clean_pixel_claims(
    tmp_path: Path, field: str, value: object
) -> None:
    sha = _sha_for_split(f"bad-pixel-{field}", SELECTION)
    row = _inference("bad-pixel", sha, [0, 0, 20, 1], source_status="clean")
    row["provenance"]["source_records"][0]["metadata"]["source_pixels"][field] = value
    _refresh_clean_declaration(row)
    snapshot = _make_snapshot(tmp_path / "snapshot", inference=[row])

    with pytest.raises(ValueError, match="clean source provenance is not validated"):
        build_queue(snapshot)


def test_build_queue_requires_clean_pixel_evidence_for_every_alias(tmp_path: Path) -> None:
    sha = _sha_for_split("pixel-alias-gap", SELECTION)
    row = _inference("pixel-alias-gap", sha, [0, 0, 20, 1], source_status="clean")
    _add_alias(row, "missing-pixel", include_pixel_declaration=False)
    snapshot = _make_snapshot(tmp_path / "snapshot", inference=[row])

    with pytest.raises(ValueError, match="declarations do not cover every alias"):
        build_queue(snapshot)


def test_build_queue_rejects_bare_clean_status_without_validation_evidence(tmp_path: Path) -> None:
    sha = _sha_for_split("bare-clean", SELECTION)
    row = _inference("bare-clean", sha, [0, 0, 20, 1], source_status="clean")
    row["provenance"]["source_pixels"] = {"status": "clean"}
    snapshot = _make_snapshot(tmp_path / "snapshot", inference=[row])

    with pytest.raises(ValueError, match="clean source provenance is not validated"):
        build_queue(snapshot)


def test_build_queue_rejects_bare_temporal_status_without_evidence(tmp_path: Path) -> None:
    sha = _sha_for_split("bare-time", SELECTION)
    row = _inference("bare-time", sha, [0, 0, 20, 1], source_status="overlay")
    row["provenance"]["temporal"] = {"status": "upstream_time_available"}
    snapshot = _make_snapshot(tmp_path / "snapshot", inference=[row])

    with pytest.raises(ValueError, match="upstream time provenance is not validated"):
        build_queue(snapshot)


@pytest.mark.parametrize(
    "claimed_time",
    ("definitely-not-iso8601", "2026-09-03T02:00:00"),
)
def test_build_queue_reparses_self_asserted_upstream_time(
    tmp_path: Path, claimed_time: str
) -> None:
    sha = _sha_for_split("bad-time", SELECTION)
    row = _inference("bad-time", sha, [0, 0, 20, 1], source_status="overlay")
    row["provenance"]["source_records"][0]["metadata"]["event_time"] = claimed_time
    temporal = row["provenance"]["temporal"]
    temporal["selected_value"] = claimed_time
    temporal["evidence"][0].update(
        {"value": claimed_time, "normalized_value": claimed_time, "validation_error": None}
    )
    snapshot = _make_snapshot(tmp_path / "snapshot", inference=[row])

    with pytest.raises(ValueError, match="upstream time provenance is not validated"):
        build_queue(snapshot)


def test_build_queue_requires_upstream_time_for_every_alias(tmp_path: Path) -> None:
    sha = _sha_for_split("time-alias-gap", SELECTION)
    row = _inference("time-alias-gap", sha, [0, 0, 20, 1], source_status="clean")
    _add_alias(row, "missing-time", include_time=False)
    snapshot = _make_snapshot(tmp_path / "snapshot", inference=[row])

    with pytest.raises(ValueError, match="upstream time provenance is not validated"):
        build_queue(snapshot)


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("split_key", "source-group:" + "0" * 64),
        ("anchor_kind", "event"),
        ("anchor_identities", ["task:invented"]),
        ("component_identities", ["task:invented"]),
    ),
)
def test_build_queue_rejects_self_asserted_source_group_claims(
    tmp_path: Path, field: str, value: object
) -> None:
    sha = _sha_for_split(f"bad-group-{field}", SELECTION)
    row = _inference("bad-group", sha, [0, 0, 20, 1], source_status="clean")
    row["provenance"]["grouping"][field] = value
    snapshot = _make_snapshot(tmp_path / "snapshot", inference=[row])

    with pytest.raises(ValueError, match="source group provenance is not validated"):
        build_queue(snapshot)


def test_v1_snapshot_cannot_produce_formal_queue_rows(tmp_path: Path) -> None:
    sha = _sha_for_split("legacy-clean", SELECTION)
    snapshot = _make_snapshot(
        tmp_path / "snapshot",
        inference=[_inference("legacy-clean", sha, [0, 0, 20, 1], source_status="clean")],
    )
    manifest_path = snapshot / "snapshot_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["schema_version"] = "prompt-gold/1"
    manifest_path.write_text(json.dumps(manifest, sort_keys=True) + "\n", encoding="utf-8")

    selected, blocked, _ = build_queue(snapshot)

    assert selected == []
    assert blocked[0]["blocked_reasons"] == ["snapshot_schema_not_prompt_gold_2"]


def test_queue_uses_source_group_split_and_blocks_missing_group(tmp_path: Path) -> None:
    image_sha = _sha_for_split("image-selection", SELECTION)
    grouped = _inference("grouped", image_sha, [0, 0, 20, 1], source_status="clean")
    locked_anchor, _ = _task_anchor_for_split("group-locked", LOCKED_TEST)
    _set_task_group(grouped, locked_anchor)

    missing_group_sha = _sha_for_split("missing-group", SELECTION)
    missing_group = _inference(
        "missing-group", missing_group_sha, [0, 0, 20, 1], source_status="clean"
    )
    missing_group["provenance"]["grouping"] = {
        "status": "grouping_unavailable",
        "split_key": missing_group_sha,
    }
    snapshot = _make_snapshot(
        tmp_path / "snapshot",
        inference=[grouped, missing_group],
    )

    selected, blocked, summary = build_queue(snapshot)

    assert selected == []
    assert summary["planned_locked_boxes_excluded"] == 1
    assert blocked[0]["canonical_id"] == "missing-group"
    assert blocked[0]["blocked_reasons"] == ["source_grouping_unavailable"]


def test_planned_locked_candidate_is_not_exposed_in_blocked_inventory(tmp_path: Path) -> None:
    locked_sha = _sha_for_split("locked-overlay", LOCKED_TEST)
    snapshot = _make_snapshot(
        tmp_path / "snapshot",
        inference=[
            _inference("locked-overlay", locked_sha, [0, 0, 20, 1], source_status="overlay")
        ],
    )

    selected, blocked, summary = build_queue(snapshot)

    assert selected == []
    assert blocked == []
    assert summary["planned_locked_boxes_excluded"] == 1
    assert summary["planned_locked_images_excluded"] == 1


def test_planned_locked_candidate_is_excluded_before_provenance_validation(
    tmp_path: Path,
) -> None:
    image_sha = _sha_for_split("locked-poison", SELECTION)
    row = _inference(
        "LOCKED-CANONICAL-ID-MUST-STAY-PRIVATE",
        image_sha,
        [0, 0, 20, 1],
        source_status="clean",
    )
    locked_anchor, _ = _task_anchor_for_split("locked-poison-group", LOCKED_TEST)
    _set_task_group(row, locked_anchor)
    row["aliases"] = []
    row["provenance"]["source_pixels"] = {"status": "clean"}
    row["provenance"]["temporal"] = {"status": "upstream_time_available"}
    snapshot = _make_snapshot(tmp_path / "snapshot", inference=[row])

    selected, blocked, summary = build_queue(snapshot)

    public_payload = json.dumps([selected, blocked, summary], ensure_ascii=False)
    assert selected == [] and blocked == []
    assert summary["planned_locked_boxes_excluded"] == 1
    assert "LOCKED-CANONICAL-ID-MUST-STAY-PRIVATE" not in public_payload


def test_existing_selection_source_group_is_excluded_across_images(tmp_path: Path) -> None:
    existing_sha = _sha_for_split("existing-group-image", SELECTION)
    candidate_sha = _sha_for_split("new-group-image", SELECTION)
    anchor, split_key = _task_anchor_for_split("existing-source-group", SELECTION)
    existing_input = _input("gold", existing_sha, [0, 0, 20, 1])
    existing_input["split_group_key"] = split_key
    candidate = _inference(
        "same-source-new-image", candidate_sha, [0, 0, 20, 1], source_status="clean"
    )
    _set_task_group(candidate, anchor)
    snapshot = _make_snapshot(
        tmp_path / "snapshot",
        inference=[candidate],
        selection_inputs=[existing_input],
        selection_gold=[
            {"canonical_id": "gold", "class_id": 8, "class_name": "car", "label": "true"}
        ],
    )

    selected, blocked, summary = build_queue(snapshot)

    assert selected == [] and blocked == []
    assert summary["existing_source_group_boxes_excluded"] == 1
    assert summary["existing_source_group_images_excluded"] == 1


def test_queue_limits_candidates_per_source_group(tmp_path: Path) -> None:
    first_sha = _sha_for_split("group-first", SELECTION)
    second_sha = _sha_for_split("group-second", SELECTION)
    first = _inference("first", first_sha, [0, 0, 20, 1], source_status="clean")
    second = _inference("second", second_sha, [0, 0, 21, 1], source_status="clean")
    shared_anchor, _ = _task_anchor_for_split("one-task", SELECTION)
    for row in (first, second):
        _set_task_group(row, shared_anchor)
    snapshot = _make_snapshot(tmp_path / "snapshot", inference=[first, second])

    selected, _, summary = build_queue(snapshot, max_per_class=10, max_per_image=10)

    assert len(selected) == 1
    assert summary["selected_source_groups"] == 1
    assert summary["eligible_source_groups"] == 1


def test_uncovered_class_size_wins_shared_group_globally(tmp_path: Path) -> None:
    existing_sha = _sha_for_split("global-existing", SELECTION)
    covered_sha = _sha_for_split("global-covered", SELECTION)
    uncovered_sha = _sha_for_split("global-uncovered", SELECTION)
    shared_anchor, _ = _task_anchor_for_split("global-shared", SELECTION)
    covered = _inference(
        "a-covered",
        covered_sha,
        [0, 0, 48, 1],
        source_status="clean",
        class_id=1,
        class_name="aaa",
    )
    uncovered = _inference(
        "z-uncovered",
        uncovered_sha,
        [0, 0, 96, 1],
        source_status="clean",
        class_id=2,
        class_name="zzz",
    )
    _set_task_group(covered, shared_anchor)
    _set_task_group(uncovered, shared_anchor)
    snapshot = _make_snapshot(
        tmp_path / "snapshot",
        inference=[covered, uncovered],
        selection_inputs=[
            _input(
                "gold-covered",
                existing_sha,
                [0, 0, 48, 1],
                class_id=1,
                class_name="aaa",
            )
        ],
        selection_gold=[
            {
                "canonical_id": "gold-covered",
                "class_id": 1,
                "class_name": "aaa",
                "label": "true",
            }
        ],
    )

    selected, _, _ = build_queue(snapshot, max_per_class=1)

    assert [row["canonical_id"] for row in selected] == ["z-uncovered"]


def test_queue_prioritizes_uncovered_size_cells(tmp_path: Path) -> None:
    existing_sha = _sha_for_split("covered-existing", SELECTION)
    covered_sha = _sha_for_split("covered-candidate", SELECTION)
    gap_sha = _sha_for_split("gap-candidate", SELECTION)
    snapshot = _make_snapshot(
        tmp_path / "snapshot",
        inference=[
            _inference("covered-medium", covered_sha, [0, 0, 48, 1], source_status="clean"),
            _inference("gap-large", gap_sha, [0, 0, 96, 1], source_status="clean"),
        ],
        selection_inputs=[_input("gold-medium", existing_sha, [0, 0, 48, 1])],
        selection_gold=[
            {"canonical_id": "gold-medium", "class_id": 8, "class_name": "car", "label": "true"}
        ],
    )

    selected, _, _ = build_queue(snapshot, max_per_class=1)

    assert [row["canonical_id"] for row in selected] == ["gap-large"]


def test_size_bins_match_training_metric_contract() -> None:
    assert SIZE_BINS == ("tiny_lt16", "small_16_47", "medium_48_95", "large_ge96")


def test_summary_label_aggregates_are_management_only() -> None:
    summary = json.loads(
        (
            REPO_ROOT
            / "data/prompt-eval/gold-expansion-20260904-v3-source-gated/summary.json"
        ).read_text(encoding="utf-8")
    )
    assert summary["management_only"] == {
        "contains_human_label_aggregates": True,
        "must_not_be_included_in_blind_annotation_package": True,
        "blind_annotation_files": ["queue.jsonl"],
    }


def test_superseded_v2_artifact_has_no_human_or_machine_queue() -> None:
    root = REPO_ROOT / "data/prompt-eval/gold-expansion-20260904-v2"
    summary = json.loads((root / "summary.json").read_text(encoding="utf-8"))
    readme = (root / "README.md").read_text(encoding="utf-8")

    assert summary["artifact_status"] == "superseded_do_not_use"
    assert summary["machine_usable"] is False
    assert not (root / "queue.jsonl").exists()
    assert not (root / "platform-map.jsonl").exists()
    assert "DO NOT LABEL OR CONSUME" in readme
    assert "must be reviewed" not in readme
