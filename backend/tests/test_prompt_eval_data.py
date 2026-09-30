"""Tests for tools/prompt_eval_data (contract: docs/PROMPT_GOLD_CONTRACT.md).

Self-contained: builds a tiny SQLite fixture and PNG images in tmp_path.
Run: PYTHONPATH=backend python3 -m pytest backend/tests/test_prompt_eval_data.py -q
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
import struct
import sys
import zlib
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.prompt_eval_data import CLASS_MAP, CLASS_NAME_TO_ID  # noqa: E402
from tools.prompt_eval_data.auditor import audit  # noqa: E402
from tools.prompt_eval_data.canon import canonical_id, pixel_bbox, suspect_note  # noqa: E402
from tools.prompt_eval_data.exporter import (  # noqa: E402
    SNAPSHOT_IDENTITY_FILE,
    build_snapshot,
    write_snapshot,
)
from tools.prompt_eval_data.scorer import (  # noqa: E402
    extract_decisions,
    judge,
    load_jsonl,
    normalize_decision,
    score,
    wilson_interval,
)
from tools.prompt_eval_data.splits import assign_split  # noqa: E402

EXPECTED_CLASSES = [
    "三轮车", "两轮车", "人员", "光伏板", "光斑", "公交车", "卡车", "没戴头盔", "车", "违建", "车牌",
]

W, H = 100, 80


def make_png(width: int = W, height: int = H, seed: int = 0) -> bytes:
    def chunk(tag: bytes, data: bytes) -> bytes:
        return struct.pack(">I", len(data)) + tag + data + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)

    ihdr = chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0))
    row = bytes(((x + seed) % 256 for x in range(width * 3)))
    raw = b"".join(b"\x00" + row for _ in range(height))
    return b"\x89PNG\r\n\x1a\n" + ihdr + chunk(b"IDAT", zlib.compress(raw)) + chunk(b"IEND", b"")


def verified_clean_payload(object_key: str, image_sha: str, frame_id: str) -> dict:
    return {
        "overlay_status": "clean",
        "clean_object_key": object_key,
        "clean_object_version": "fixture-v1",
        "clean_sha256": image_sha,
        "clean_width": W,
        "clean_height": H,
        "annotated_object_key": object_key,
        "annotated_object_version": "fixture-v1",
        "annotated_sha256": image_sha,
        "annotated_width": W,
        "annotated_height": H,
        "overlay_check_method": "fixture-human-review",
        "overlay_check_version": "1",
        "overlay_evidence_sha256": hashlib.sha256(b"fixture-overlay-evidence").hexdigest(),
        "mapping_method": "same_asset",
        "mapping_version": "1",
        "geometry_transform": "identity",
        "source_system": "fixture",
        "source_table": "samples",
        "source_row_id": frame_id,
        "frame_id": frame_id,
    }


def build_fixture(tmp_path: Path) -> tuple[Path, Path]:
    db_path = tmp_path / "fixture.sqlite3"
    images = tmp_path / "images"
    con = sqlite3.connect(db_path)
    con.executescript(
        """
        CREATE TABLE samples (id VARCHAR(32) PRIMARY KEY, source_type VARCHAR(40),
                              source_event_id VARCHAR(255), source_service VARCHAR(120),
                              session_id VARCHAR(120), capture_time DATETIME,
                              camera_id VARCHAR(120), rule_code VARCHAR(120),
                              metadata_json TEXT);
        CREATE TABLE assets (id VARCHAR(32) PRIMARY KEY, sample_id VARCHAR(32), kind VARCHAR(40),
                             bucket VARCHAR(255), object_key VARCHAR(1024), sha256 VARCHAR(128));
        CREATE TABLE candidates (id VARCHAR(32) PRIMARY KEY, sample_id VARCHAR(32), model_run_id VARCHAR(32),
                                 target_index INTEGER, class_id INTEGER, class_name VARCHAR(160),
                                 confidence FLOAT, x1 FLOAT, y1 FLOAT, x2 FLOAT, y2 FLOAT,
                                 raw_decision VARCHAR(40), yolo_decision VARCHAR(40), reason_code VARCHAR(120));
        CREATE TABLE annotations (id VARCHAR(32) PRIMARY KEY, candidate_id VARCHAR(32), label VARCHAR(10),
                                  note TEXT, labeled_by VARCHAR(32), revision INTEGER, labeled_at DATETIME);
        CREATE TABLE users (id VARCHAR(32) PRIMARY KEY, username VARCHAR(120));
        """
    )
    con.execute("INSERT INTO users VALUES ('u-admin', 'admin')")

    # imgA (aliased twice): 车 box human-true on both aliases + 车牌 box human-true.
    # imgB: 人员 box with conflicting human labels. imgC: 车 true with suspicious note.
    # imgD: 光伏板 auto-labeled only + 违建 human-false. imgE: image missing.
    object_keys = {"s1": "a/1.png", "s2": "a/1.png", "s3": "b/2.png", "s4": "b/2.png",
                   "s5": "c/3.png", "s6": "d/4.png", "s7": "e/5.png"}
    for i, (sample, key) in enumerate(sorted(object_keys.items())):
        con.execute(
            "INSERT INTO samples VALUES (?, 'review_service', ?, 'fixture', NULL, "
            "'2026-09-03 10:00:00', NULL, 'ktd-test', '{}')",
            (sample, f"event-{sample}"),
        )
        con.execute(
            "INSERT INTO assets VALUES (?, ?, 'original', 'minio-src', ?, NULL)",
            (f"asset-{sample}", sample, key),
        )
        if key != "e/5.png":
            target = images / key
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(make_png(seed=i))

    box = [0.125, 0.0, 0.5, 1.0]
    rows = [
        # (candidate, sample, class, label, note, labeled_by, bbox)
        ("c-a1", "s1", "车", "true", "清晰车辆", "u-admin", box),
        ("c-a2", "s2", "车", "true", "清晰车辆", "u-admin", box),  # alias of c-a1, agrees
        ("c-a3", "s1", "车牌", "true", "蓝牌", "u-admin", [0.3, 0.3, 0.4, 0.4]),
        ("c-b1", "s3", "人员", "true", "有人", "u-admin", box),
        ("c-b2", "s4", "人员", "false", "树影", "u-admin", box),  # label conflict with c-b1
        ("c-c1", "s5", "车", "true", "这里没有车，只是树影", "u-admin", box),  # note suspect
        ("c-d1", "s6", "光伏板", "true", "auto annotation consensus: x=true", None, box),  # auto -> excluded
        ("c-d2", "s6", "违建", "false", "临时棚非违建", "u-admin", [0.1, 0.1, 0.2, 0.2]),
        ("c-e1", "s7", "车", "true", "图没了", "u-admin", box),  # image unavailable
        ("c-f1", "s6", "卡车", None, None, None, [0.6, 0.6, 0.7, 0.7]),  # never labeled
    ]
    for i, (cid, sample, cls, label, note, by, bbox) in enumerate(rows):
        con.execute(
            "INSERT INTO candidates VALUES (?, ?, 'run-1', 0, NULL, ?, 0.9, ?, ?, ?, ?, 'uncertain', 'uncertain', 'model_conflict')",
            (cid, sample, cls, *bbox),
        )
        if label is not None:
            con.execute(
                "INSERT INTO annotations VALUES (?, ?, ?, ?, ?, 1, '2026-09-03 10:00:00')",
                (f"ann-{cid}", cid, label, note, by),
            )
    con.commit()
    con.close()
    return db_path, images


def make_inputs(tmp_path: Path) -> dict:
    """Fixture DB + images + prior files; returns build_snapshot kwargs (stable paths)."""
    db_path, images = build_fixture(tmp_path)
    gpt = tmp_path / "gpt.jsonl"
    gpt.write_text(
        json.dumps({
            "run_id": "gpt-run", "file_path": "a/1.png", "class_name": "车",
            "bbox": [0.125, 0.0, 0.5, 1.0], "status": "succeeded", "auto_label": "true",
            "decision": {"consensus": "true"}, "provider": "gpt", "model": "gpt-x",
        }) + "\n",
        encoding="utf-8",
    )
    qwen = tmp_path / "qwen.jsonl"
    qwen.write_text(
        json.dumps({
            "source_run_id": "qwen-run", "file_path": "a/1.png", "class_name": "车",
            "bbox": [0.125, 0.0, 0.5, 1.0], "status": "succeeded", "model": "qwen3-vl:8b",
            "qwen_label": "true", "qwen_decision": "CONFIRMED", "qwen_confidence_score": 0.9,
        }) + "\n",
        encoding="utf-8",
    )
    return {
        "db_path": str(db_path),
        "image_root": str(images),
        "prior_gpt_paths": [str(gpt)],
        "prior_qwen_paths": [str(qwen)],
    }


def export(tmp_path: Path, root_name: str = "snapshots"):
    inputs = make_inputs(tmp_path)
    snapshot_id, files, manifest = build_snapshot(**inputs)
    target = write_snapshot(files, str(tmp_path / root_name), snapshot_id)
    return snapshot_id, files, manifest, target


def read_jsonl(data: bytes):
    return [json.loads(line) for line in data.decode("utf-8").splitlines() if line.strip()]


def test_class_map_fixed_order_and_plate_appended():
    assert [name for _, name in CLASS_MAP] == EXPECTED_CLASSES
    assert [cid for cid, _ in CLASS_MAP] == list(range(11))
    assert CLASS_NAME_TO_ID["车牌"] == 10


def test_pixel_bbox_rounding_rule():
    assert pixel_bbox([0.125, 0.0, 0.5, 1.0], 100, 80) == [13, 0, 50, 80]
    assert pixel_bbox([0.0, 0.0, 1.0, 1.0], 100, 80) == [0, 0, 100, 80]


def test_canonical_id_stable():
    a = canonical_id("ab" * 32, 8, [13, 0, 50, 80])
    b = canonical_id("ab" * 32, 8, [13, 0, 50, 80])
    c = canonical_id("ab" * 32, 8, [13, 0, 50, 81])
    assert a == b and a != c and len(a) == 64


def test_export_funnel_alias_merge_quarantine(tmp_path):
    snapshot_id, files, manifest, _ = export(tmp_path)
    assert snapshot_id.startswith("pg2-")
    assert manifest["schema_version"] == "prompt-gold/2"
    counts = manifest["counts"]
    assert counts["raw_candidates"] == 10
    assert counts["excluded_image_unavailable"] == 1
    # c-a1/c-a2 merge into one box, c-b1/c-b2 merge into one -> 7 canonical
    assert counts["canonical_boxes"] == 7
    # gold: 车(imgA merged), 车牌(imgA), 违建(imgD) = 3
    assert counts["gold_records"] == 3
    # quarantine: label_conflict (人员) + note_label_suspect (车 imgC) = 2
    assert counts["quarantine_records"] == 2
    assert counts["quarantine_label_conflict"] == 1
    assert counts["quarantine_note_label_suspect"] == 1
    # unlabeled: 光伏板 (auto only) + 卡车 (no annotation) = 2
    assert counts["unlabeled_canonical"] == 2
    assert counts["authoring_auto_excluded"] == 1
    assert counts["source_pixels_clean_images"] == 0
    assert counts["source_pixels_unknown_images"] == 4
    assert manifest["readiness"]["formal_training_ready"] is False
    assert json.loads(files["source_readiness.json"]) == manifest["readiness"]

    gold = read_jsonl(files["gold_records.jsonl"])
    by_class = {row["class_name"]: row for row in gold}
    merged = by_class["车"]
    assert merged["label"] == "true"
    assert len(merged["aliases"]) == 2  # c-a1 + c-a2 merged, single gold row
    assert merged["reviewer"] == "admin" and merged["revision"] == 1
    assert by_class["车牌"]["class_id"] == 10
    assert by_class["违建"]["label"] == "false"
    # priors attached to gold but kept in their own fields
    assert merged["prior_gpt"]["passes"][0]["label"] == "true"
    assert merged["prior_qwen"]["attempts"][0]["decision"] == "CONFIRMED"
    assert merged["prior_yolo"]["raw_decision"] == "uncertain"
    assert counts["prior_gpt_matched"] >= 1 and counts["prior_qwen_matched"] >= 1
    assert merged["provenance"]["grouping"]["status"] == "grouping_unavailable"
    assert merged["provenance"]["source_pixels"]["status"] == "unknown"
    assert merged["source"]["provenance"]["source_event_semantics"] == "upstream_event_id"

    quarantine = read_jsonl(files["quarantine.jsonl"])
    reasons = {tuple(sorted(row["quarantine_reasons"])) for row in quarantine}
    assert ("label_conflict",) in reasons and ("note_label_suspect",) in reasons
    conflict = next(r for r in quarantine if r["quarantine_reasons"] == ["label_conflict"])
    labels = {e["label"] for e in conflict["evidence"]}
    assert labels == {"true", "false"}  # no voting: both sides preserved

    authoring = read_jsonl(files["authoring.jsonl"])
    auto = next(a for a in authoring if a["source"] == "auto_annotation")
    assert auto["exclusion_reason"] == "auto_annotation_applied" and not auto["included_in_gold"]
    missing = next(a for a in authoring if a["candidate_id"] == "c-e1")
    assert missing["exclusion_reason"] == "image_unavailable" and missing["canonical_id"] is None
    conflict_author = [a for a in authoring if a["candidate_id"] in {"c-b1", "c-b2"}]
    assert all(a["exclusion_reason"] == "label_conflict" for a in conflict_author)


def test_split_disjoint_and_inputs_free_of_gold(tmp_path):
    _, files, manifest, target = export(tmp_path)
    sel_inputs = read_jsonl(files["selection_inputs.jsonl"])
    lock_inputs = read_jsonl(files["locked_test_inputs.jsonl"])
    assert {r["image_sha256"] for r in sel_inputs} & {r["image_sha256"] for r in lock_inputs} == set()
    forbidden = {"label", "note", "reviewer", "reviewer_id", "labeled_at",
                 "prior_yolo", "prior_gpt", "prior_qwen", "aliases"}
    for row in sel_inputs + lock_inputs:
        assert not (set(row) & forbidden), f"input leak: {set(row) & forbidden}"
    assert len(sel_inputs) + len(lock_inputs) == manifest["counts"]["gold_records"]
    # auditor agrees, end to end
    results = audit(str(target))
    assert results, "auditor returned no checks"
    assert all(ok for _, ok, _ in results), [r for r in results if not r[1]]
    protected_results = audit(str(target), include_protected_gold=True)
    assert all(ok for _, ok, _ in protected_results), [r for r in protected_results if not r[1]]


def test_auditor_does_not_open_locked_gold_by_default(tmp_path):
    _, _, _, target = export(tmp_path)
    poison = b"not-json-and-must-not-be-parsed\n"
    protected_names = {"gold_records.jsonl", "locked_test_gold.jsonl"}
    for name in protected_names:
        (target / name).write_bytes(poison)

    sums_path = target / "SHA256SUMS"
    lines = []
    for line in sums_path.read_text(encoding="utf-8").splitlines():
        sha, name = line.split("  ", 1)
        if name in protected_names:
            sha = hashlib.sha256(poison).hexdigest()
        lines.append(f"{sha}  {name}")
    sums_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    manifest_path = target / "snapshot_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    for entry in manifest["files"]:
        if entry["path"] in protected_names:
            entry["sha256"] = hashlib.sha256(poison).hexdigest()
            entry["bytes"] = len(poison)
        elif entry["path"] == "SHA256SUMS":
            payload = sums_path.read_bytes()
            entry["sha256"] = hashlib.sha256(payload).hexdigest()
            entry["bytes"] = len(payload)
    sums = {
        name: sha
        for sha, name in (
            line.split("  ", 1)
            for line in sums_path.read_text(encoding="utf-8").splitlines()
        )
    }
    snapshot_id = "pg2-" + hashlib.sha256(
        "".join(f"{name}:{sums[name]}\n" for name in sorted(sums)).encode()
    ).hexdigest()[:12]
    manifest["snapshot_id"] = snapshot_id
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    renamed_target = target.parent / snapshot_id
    target.rename(renamed_target)
    target = renamed_target

    results = audit(str(target))

    assert all(ok for _, ok, _ in results), [result for result in results if not result[1]]
    assert (
        "protected_label_content_skipped",
        True,
        "not opened; recorded hashes compared only",
    ) in results


def test_export_byte_identical_across_runs(tmp_path):
    inputs = make_inputs(tmp_path)  # one fixture, stable source paths
    snap1, files1, _ = build_snapshot(**inputs)
    snap2, files2, _ = build_snapshot(**inputs)
    assert snap1 == snap2
    assert set(files1) == set(files2)
    for path in files1:
        assert files1[path] == files2[path], f"{path} differs between runs"
    # writing to different output roots does not change bytes
    target1 = write_snapshot(files1, str(tmp_path / "out1"), snap1)
    target2 = write_snapshot(files2, str(tmp_path / "out2"), snap2)
    for path in files1:
        assert (target1 / path).read_bytes() == (target2 / path).read_bytes()


def test_snapshot_identity_changes_when_only_funnel_counts_change(tmp_path):
    inputs = make_inputs(tmp_path)
    baseline_id, baseline_files, _ = build_snapshot(**inputs)
    con = sqlite3.connect(inputs["db_path"])
    con.execute(
        "INSERT INTO candidates VALUES "
        "('c-unknown-extra', 's6', 'run-extra', 0, NULL, 'UNKNOWN', 0.9, "
        "0.1, 0.1, 0.2, 0.2, 'uncertain', 'uncertain', 'model_conflict')"
    )
    con.commit()
    con.close()

    changed_id, changed_files, changed_manifest = build_snapshot(**inputs)

    assert changed_id != baseline_id
    unchanged_members = set(baseline_files) - {
        SNAPSHOT_IDENTITY_FILE,
        "SHA256SUMS",
        "snapshot_manifest.json",
    }
    assert all(baseline_files[name] == changed_files[name] for name in unchanged_members)
    identity = json.loads(changed_files[SNAPSHOT_IDENTITY_FILE])
    assert identity == {
        key: changed_manifest[key]
        for key in (
            "schema_version",
            "class_map",
            "source",
            "counts",
            "determinism",
            "readiness",
        )
    }
    assert identity["counts"]["raw_candidates"] == 11
    assert identity["counts"]["excluded_unknown_class"] == 1


def test_write_snapshot_is_idempotent_but_rejects_same_id_with_different_bytes(tmp_path):
    snapshot_id, files, _ = build_snapshot(**make_inputs(tmp_path))
    output_root = tmp_path / "out"
    target = write_snapshot(files, str(output_root), snapshot_id)

    assert write_snapshot(files, str(output_root), snapshot_id) == target
    conflicting_files = dict(files)
    conflicting_files["source_readiness.json"] += b"\n"
    with pytest.raises(FileExistsError, match="different content"):
        write_snapshot(conflicting_files, str(output_root), snapshot_id)


def test_assign_split_deterministic():
    assert assign_split("ab" * 32) == assign_split("ab" * 32)
    assert assign_split("ab" * 32) in {"selection", "locked_test"}


def test_synthetic_annex_event_is_not_treated_as_leakage_group(tmp_path):
    inputs = make_inputs(tmp_path)
    con = sqlite3.connect(inputs["db_path"])
    con.execute(
        "UPDATE samples SET source_event_id=?, metadata_json=? WHERE id='s6'",
        ("ktd-annex-annex-6", json.dumps({"annex_id": "annex-6"})),
    )
    con.commit()
    con.close()

    _, files, manifest = build_snapshot(**inputs)
    inference = read_jsonl(files["inference_records.jsonl"])
    rows = [row for row in inference if row["image_object_key"] == "d/4.png"]
    assert rows
    assert all(row["provenance"]["grouping"]["status"] == "grouping_unavailable" for row in rows)
    record = rows[0]["provenance"]["source_records"][0]
    assert record["source_event_semantics"] == "synthetic_import_row_id"
    assert manifest["counts"]["grouping_unavailable_images"] == 4


def test_synthetic_annex_metadata_event_is_not_treated_as_group_evidence(tmp_path):
    inputs = make_inputs(tmp_path)
    con = sqlite3.connect(inputs["db_path"])
    con.execute(
        "UPDATE samples SET source_event_id=NULL, metadata_json=? WHERE id='s6'",
        (json.dumps({"annex_id": "annex-6", "event_id": "ktd-annex-annex-6"}),),
    )
    con.commit()
    con.close()

    _, files, _ = build_snapshot(**inputs)
    row = next(
        row
        for row in read_jsonl(files["inference_records.jsonl"])
        if row["image_object_key"] == "d/4.png"
    )

    assert row["provenance"]["grouping"]["component_identities"] == []
    assert row["provenance"]["grouping"]["status"] == "grouping_unavailable"


def test_event_only_identity_is_not_a_formal_source_group(tmp_path):
    inputs = make_inputs(tmp_path)

    _, files, manifest = build_snapshot(**inputs)

    inference = read_jsonl(files["inference_records.jsonl"])
    assert inference
    assert all(
        row["provenance"]["grouping"]["status"] == "grouping_unavailable"
        for row in inference
    )
    assert manifest["readiness"]["formal_training_ready"] is False
    assert "grouping_unavailable" in manifest["readiness"]["blockers"]


def test_session_video_and_track_are_connection_evidence_not_formal_anchors(tmp_path):
    inputs = make_inputs(tmp_path)
    con = sqlite3.connect(inputs["db_path"])
    con.execute(
        "UPDATE samples SET session_id=?, metadata_json=? WHERE id='s6'",
        (
            "session-only",
            json.dumps({"video_id": "video-only", "track_id": "track-only"}),
        ),
    )
    con.commit()
    con.close()

    _, files, _ = build_snapshot(**inputs)
    row = next(
        row
        for row in read_jsonl(files["inference_records.jsonl"])
        if row["image_object_key"] == "d/4.png"
    )
    grouping = row["provenance"]["grouping"]

    assert grouping["status"] == "grouping_unavailable"
    assert grouping["anchor_identities"] == []
    assert {identity.split(":", 1)[0] for identity in grouping["component_identities"]} == {
        "event",
        "session",
        "track",
        "video",
    }


def test_task_split_key_is_stable_when_lower_scope_ids_are_added(tmp_path):
    inputs = make_inputs(tmp_path)
    con = sqlite3.connect(inputs["db_path"])
    con.execute(
        "UPDATE samples SET metadata_json=? WHERE id='s6'",
        (json.dumps({"task_id": "task-stable"}),),
    )
    con.commit()

    _, baseline_files, _ = build_snapshot(**inputs)
    baseline = next(
        row
        for row in read_jsonl(baseline_files["inference_records.jsonl"])
        if row["image_object_key"] == "d/4.png"
    )

    con.execute(
        "UPDATE samples SET metadata_json=? WHERE id='s6'",
        (
            json.dumps(
                {
                    "task_id": "task-stable",
                    "video_id": "video-later",
                    "track_id": "track-later",
                    "event_id": "event-later",
                }
            ),
        ),
    )
    con.commit()
    con.execute("UPDATE samples SET session_id='session-later' WHERE id='s6'")
    con.commit()
    con.close()

    _, enriched_files, _ = build_snapshot(**inputs)
    enriched = next(
        row
        for row in read_jsonl(enriched_files["inference_records.jsonl"])
        if row["image_object_key"] == "d/4.png"
    )

    assert enriched["provenance"]["grouping"]["split_key"] == baseline["provenance"]["grouping"]["split_key"]
    assert enriched["provenance"]["grouping"]["anchor_identities"] == ["task:task-stable"]


def test_connected_task_ids_conflict_and_grouping_counts_are_exclusive(tmp_path):
    inputs = make_inputs(tmp_path)
    con = sqlite3.connect(inputs["db_path"])
    con.execute(
        "UPDATE samples SET session_id='shared-session', metadata_json=? WHERE id='s1'",
        (json.dumps({"task_id": "task-a"}),),
    )
    con.execute(
        "UPDATE samples SET session_id='shared-session', metadata_json=? WHERE id='s6'",
        (json.dumps({"task_id": "task-b"}),),
    )
    con.commit()
    con.close()

    _, files, manifest = build_snapshot(**inputs)
    inference = read_jsonl(files["inference_records.jsonl"])
    conflicted = {
        row["image_object_key"]
        for row in inference
        if row["provenance"]["grouping"]["status"] == "source_group_conflict"
    }
    counts = manifest["counts"]

    assert conflicted == {"a/1.png", "d/4.png"}
    assert counts["source_group_images"] == 0
    assert counts["grouping_unavailable_images"] == 2
    assert counts["source_group_conflict_images"] == 2
    assert (
        counts["source_group_images"]
        + counts["grouping_unavailable_images"]
        + counts["source_group_conflict_images"]
        == counts["unique_image_sha256"]
    )
    assert "grouping_unavailable" in manifest["readiness"]["blockers"]
    assert "source_group_conflict" in manifest["readiness"]["blockers"]


def test_excluded_source_row_still_connects_task_components(tmp_path):
    inputs = make_inputs(tmp_path)
    con = sqlite3.connect(inputs["db_path"])
    con.execute(
        "UPDATE samples SET metadata_json=? WHERE id='s1'",
        (json.dumps({"task_id": "task-a", "video_id": "video-bridge"}),),
    )
    con.execute(
        "UPDATE samples SET metadata_json=? WHERE id='s2'",
        (json.dumps({"task_id": "task-a"}),),
    )
    con.execute(
        "UPDATE samples SET metadata_json=? WHERE id='s6'",
        (json.dumps({"task_id": "task-b", "track_id": "track-bridge"}),),
    )
    # s7 has no readable image and is excluded from canonical rows, but its
    # upstream identities still prove that the two included task anchors connect.
    con.execute(
        "UPDATE samples SET metadata_json=? WHERE id='s7'",
        (json.dumps({"video_id": "video-bridge", "track_id": "track-bridge"}),),
    )
    con.commit()
    con.close()

    _, files, manifest = build_snapshot(**inputs)
    inference = read_jsonl(files["inference_records.jsonl"])
    bridged = [
        row
        for row in inference
        if row["image_object_key"] in {"a/1.png", "d/4.png"}
    ]

    assert bridged
    assert all(
        row["provenance"]["grouping"]["status"] == "source_group_conflict"
        for row in bridged
    )
    assert all(
        row["provenance"]["grouping"]["anchor_identities"]
        == ["task:task-a", "task:task-b"]
        for row in bridged
    )
    assert manifest["counts"]["source_group_conflict_images"] == 2


def test_rounded_zero_area_bbox_is_rejected(tmp_path):
    inputs = make_inputs(tmp_path)
    con = sqlite3.connect(inputs["db_path"])
    con.execute(
        "UPDATE candidates SET x1=0.100, y1=0.200, x2=0.104, y2=0.300 WHERE id='c-f1'"
    )
    con.commit()
    con.close()

    with pytest.raises(ValueError, match="bbox_px must have positive width and height"):
        build_snapshot(**inputs)


def test_clean_pixel_status_requires_hash_dimensions_and_identity_mapping(tmp_path):
    inputs = make_inputs(tmp_path)
    image_path = Path(inputs["image_root"]) / "d/4.png"
    image_sha = hashlib.sha256(image_path.read_bytes()).hexdigest()
    source_pixels = verified_clean_payload("d/4.png", image_sha, "frame-s6")
    con = sqlite3.connect(inputs["db_path"])
    con.execute("UPDATE assets SET sha256=? WHERE sample_id='s6'", (image_sha,))
    con.execute(
        "UPDATE samples SET metadata_json=? WHERE id='s6'",
        (json.dumps({"event_time": "2026-09-03T10:00:00Z", "source_pixels": source_pixels}),),
    )
    con.commit()
    con.close()

    _, files, manifest = build_snapshot(**inputs)
    inference = read_jsonl(files["inference_records.jsonl"])
    rows = [row for row in inference if row["image_object_key"] == "d/4.png"]
    assert rows
    assert all(row["provenance"]["source_pixels"]["status"] == "clean" for row in rows)
    assert manifest["counts"]["source_pixels_clean_images"] == 1
    assert manifest["counts"]["source_pixels_unknown_images"] == 3
    assert manifest["counts"]["upstream_time_images"] == 1


def test_clean_status_requires_consistent_declarations_for_every_alias_sample(tmp_path):
    inputs = make_inputs(tmp_path)
    image_path = Path(inputs["image_root"]) / "a/1.png"
    image_sha = hashlib.sha256(image_path.read_bytes()).hexdigest()
    con = sqlite3.connect(inputs["db_path"])
    con.execute("UPDATE assets SET sha256=? WHERE sample_id='s1'", (image_sha,))
    con.execute(
        "UPDATE samples SET metadata_json=? WHERE id='s1'",
        (
            json.dumps(
                {
                    "event_time": "2026-09-03T10:00:00Z",
                    "source_pixels": verified_clean_payload("a/1.png", image_sha, "frame-a"),
                }
            ),
        ),
    )
    con.commit()
    con.close()

    _, files, _ = build_snapshot(**inputs)
    rows = [
        row
        for row in read_jsonl(files["inference_records.jsonl"])
        if row["image_object_key"] == "a/1.png"
    ]

    assert rows
    assert all(row["provenance"]["source_pixels"]["status"] == "unknown" for row in rows)
    assert all(
        "missing_source_pixel_declaration_for_sample:s2"
        in row["provenance"]["source_pixels"]["blocking_reasons"]
        for row in rows
    )


def test_clean_status_requires_scalar_frame_identity(tmp_path):
    inputs = make_inputs(tmp_path)
    image_path = Path(inputs["image_root"]) / "d/4.png"
    image_sha = hashlib.sha256(image_path.read_bytes()).hexdigest()
    source_pixels = verified_clean_payload("d/4.png", image_sha, "frame-s6")
    source_pixels["frame_id"] = False
    con = sqlite3.connect(inputs["db_path"])
    con.execute("UPDATE assets SET sha256=? WHERE sample_id='s6'", (image_sha,))
    con.execute(
        "UPDATE samples SET metadata_json=? WHERE id='s6'",
        (
            json.dumps(
                {
                    "event_time": "2026-09-03T10:00:00Z",
                    "source_pixels": source_pixels,
                    "task_id": "task-frame-check",
                }
            ),
        ),
    )
    con.commit()
    con.close()

    _, files, _ = build_snapshot(**inputs)
    row = next(
        row
        for row in read_jsonl(files["inference_records.jsonl"])
        if row["image_object_key"] == "d/4.png"
    )

    assert row["provenance"]["source_pixels"]["status"] == "unknown"
    assert (
        "missing_capture_or_frame_identity"
        in row["provenance"]["source_pixels"]["blocking_reasons"]
    )


def test_upstream_time_must_cover_every_alias_sample(tmp_path):
    inputs = make_inputs(tmp_path)
    con = sqlite3.connect(inputs["db_path"])
    con.execute(
        "UPDATE samples SET metadata_json=? WHERE id='s1'",
        (json.dumps({"event_time": "2026-09-03T10:00:00Z"}),),
    )
    con.commit()
    con.close()

    _, files, _ = build_snapshot(**inputs)
    row = next(
        row
        for row in read_jsonl(files["inference_records.jsonl"])
        if row["image_object_key"] == "a/1.png"
    )

    assert row["provenance"]["temporal"]["status"] == "incomplete_upstream_time"
    assert row["provenance"]["temporal"]["missing_samples"] == ["s2"]


def test_conflict_in_non_selected_temporal_field_blocks_formal_time(tmp_path):
    inputs = make_inputs(tmp_path)
    common_frame_time = "2026-09-03T09:00:00Z"
    con = sqlite3.connect(inputs["db_path"])
    for sample_id, event_time in (
        ("s1", "2026-09-03T10:00:00Z"),
        ("s2", "2026-09-03T11:00:00Z"),
    ):
        con.execute(
            "UPDATE samples SET metadata_json=? WHERE id=?",
            (
                json.dumps(
                    {
                        "event_time": event_time,
                        "source_pixels": {"frame_timestamp": common_frame_time},
                    }
                ),
                sample_id,
            ),
        )
    con.commit()

    _, conflicted_files, _ = build_snapshot(**inputs)
    conflicted = next(
        row
        for row in read_jsonl(conflicted_files["inference_records.jsonl"])
        if row["image_object_key"] == "a/1.png" and row["class_name"] == "车"
    )
    assert conflicted["provenance"]["temporal"]["selected_field"] == (
        "metadata.source_pixels.frame_timestamp"
    )
    assert conflicted["provenance"]["temporal"]["status"] == "conflicting_upstream_times"
    assert conflicted["provenance"]["temporal"]["conflicting_fields"] == [
        "metadata.event_time"
    ]

    # Different timestamp kinds may describe different instants. Each kind only
    # needs to agree across aliases; kinds are not forced to equal one another.
    con.execute(
        "UPDATE samples SET metadata_json=? WHERE id='s2'",
        (
            json.dumps(
                {
                    "event_time": "2026-09-03T10:00:00Z",
                    "source_pixels": {"frame_timestamp": common_frame_time},
                }
            ),
        ),
    )
    con.commit()
    con.close()
    _, consistent_files, _ = build_snapshot(**inputs)
    consistent = next(
        row
        for row in read_jsonl(consistent_files["inference_records.jsonl"])
        if row["image_object_key"] == "a/1.png" and row["class_name"] == "车"
    )
    assert consistent["provenance"]["temporal"]["status"] == "upstream_time_available"
    assert consistent["provenance"]["temporal"]["conflicting_fields"] == []


def test_self_declared_clean_and_invalid_time_do_not_pass_formal_gates(tmp_path):
    inputs = make_inputs(tmp_path)
    image_path = Path(inputs["image_root"]) / "d/4.png"
    image_sha = hashlib.sha256(image_path.read_bytes()).hexdigest()
    source_pixels = {
        "overlay_status": "clean",
        "clean_object_key": "d/4.png",
        "clean_sha256": image_sha,
        "clean_width": W,
        "clean_height": H,
        "annotated_object_key": "d/4.png",
        "annotated_sha256": image_sha,
        "mapping_method": "arbitrary-claim",
        "mapping_version": "1",
        "geometry_transform": "identity",
        "source_system": "fixture",
        "source_table": "samples",
        "source_row_id": "s6",
        "frame_id": "frame-s6",
    }
    con = sqlite3.connect(inputs["db_path"])
    con.execute(
        "UPDATE samples SET metadata_json=? WHERE id='s6'",
        (json.dumps({"event_time": "not-a-timestamp", "source_pixels": source_pixels}),),
    )
    con.commit()
    con.close()

    _, files, manifest = build_snapshot(**inputs)
    row = next(
        row
        for row in read_jsonl(files["inference_records.jsonl"])
        if row["image_object_key"] == "d/4.png"
    )

    assert row["provenance"]["source_pixels"]["status"] == "unknown"
    assert row["provenance"]["source_pixels"]["formal_input_allowed"] is False
    assert row["provenance"]["temporal"]["status"] == "invalid_upstream_time"
    assert manifest["readiness"]["formal_training_ready"] is False


def test_inference_provenance_allowlists_metadata_and_hashes_raw_source(tmp_path):
    inputs = make_inputs(tmp_path)
    metadata = {
        "task_id": "task-safe",
        "event_time": "2026-09-03T10:00:00+08:00",
        "source_task_id": {"token": "SECRET-NESTED-MARKER"},
        "label": "SECRET-LABEL-MARKER",
        "token": "SECRET-TOKEN-MARKER",
        "signed_url": "https://example.invalid/SECRET-URL-MARKER",
        "source_file_path": "https://example.invalid/image?token=SECRET-SOURCE-FILE-MARKER",
    }
    con = sqlite3.connect(inputs["db_path"])
    con.execute(
        "UPDATE samples SET metadata_json=? WHERE id='s6'",
        (json.dumps(metadata, sort_keys=True),),
    )
    con.commit()
    con.close()

    _, files, _ = build_snapshot(**inputs)
    inference_bytes = files["inference_records.jsonl"]
    assert b"SECRET-LABEL-MARKER" not in inference_bytes
    assert b"SECRET-TOKEN-MARKER" not in inference_bytes
    assert b"SECRET-URL-MARKER" not in inference_bytes
    assert b"SECRET-SOURCE-FILE-MARKER" not in inference_bytes
    assert b"SECRET-NESTED-MARKER" not in inference_bytes

    row = next(
        row
        for row in read_jsonl(inference_bytes)
        if row["image_object_key"] == "d/4.png"
    )
    record = row["provenance"]["source_records"][0]
    assert record["metadata"] == {
        "event_time": "2026-09-03T10:00:00+08:00",
        "task_id": "task-safe",
    }
    assert record["metadata_sha256"] == hashlib.sha256(
        json.dumps(metadata, sort_keys=True).encode("utf-8")
    ).hexdigest()
    assert "metadata_json" not in record


def test_url_or_query_like_task_identity_is_not_exported_or_used_as_anchor(tmp_path):
    inputs = make_inputs(tmp_path)
    metadata = {
        "task_id": "https://example.invalid/task?token=SECRET-GROUP-MARKER#fragment",
        "source_task_id": float("nan"),
        "source_pixels": {
            "overlay_status": "https://example.invalid/status?token=SECRET-STATUS-MARKER"
        },
    }
    con = sqlite3.connect(inputs["db_path"])
    con.execute(
        "UPDATE samples SET metadata_json=? WHERE id='s6'",
        (json.dumps(metadata),),
    )
    con.commit()
    con.close()

    _, files, _ = build_snapshot(**inputs)
    inference_bytes = files["inference_records.jsonl"]
    assert b"SECRET-GROUP-MARKER" not in inference_bytes
    assert b"SECRET-STATUS-MARKER" not in inference_bytes
    row = next(
        row
        for row in read_jsonl(inference_bytes)
        if row["image_object_key"] == "d/4.png"
    )
    grouping = row["provenance"]["grouping"]

    assert row["provenance"]["source_records"][0]["metadata"] == {}
    assert grouping["status"] == "grouping_unavailable"
    assert grouping["anchor_identities"] == []
    assert all("nan" not in identity.lower() for identity in grouping["component_identities"])
    assert row["provenance"]["source_pixels"]["status"] == "unknown"


def test_uri_scheme_task_identity_is_not_exported_or_used_as_anchor(tmp_path):
    inputs = make_inputs(tmp_path)
    marker = "data:text/plain,SECRET-DATA-URL-MARKER"
    con = sqlite3.connect(inputs["db_path"])
    con.execute(
        "UPDATE samples SET metadata_json=? WHERE id='s6'",
        (json.dumps({"task_id": marker}),),
    )
    con.commit()
    con.close()

    _, files, _ = build_snapshot(**inputs)
    assert marker.encode() not in files["inference_records.jsonl"]
    row = next(
        row
        for row in read_jsonl(files["inference_records.jsonl"])
        if row["image_object_key"] == "d/4.png"
    )
    assert row["provenance"]["source_records"][0]["metadata"] == {}
    assert row["provenance"]["grouping"]["status"] == "grouping_unavailable"
    assert row["provenance"]["grouping"]["anchor_identities"] == []


def test_same_upstream_group_keeps_different_images_in_one_split(tmp_path):
    inputs = make_inputs(tmp_path)
    con = sqlite3.connect(inputs["db_path"])
    con.execute(
        "UPDATE samples SET metadata_json=? WHERE id IN ('s1', 's2', 's6')",
        (json.dumps({"task_id": "task-one"}),),
    )
    con.commit()
    con.close()

    _, files, _ = build_snapshot(**inputs)
    gold = read_jsonl(files["gold_records.jsonl"])
    grouped = [row for row in gold if row["image_object_key"] in {"a/1.png", "d/4.png"}]
    assert len({row["split"] for row in grouped}) == 1
    assert len({row["provenance"]["grouping"]["split_key"] for row in grouped}) == 1


def test_suspect_note_tokens():
    assert suspect_note("true", "这不是车") == ["不是"]
    assert suspect_note("false", "临时棚影") == []  # no false-token present
    assert suspect_note("false", "确实有目标") == ["确实有"]
    assert suspect_note("true", "清晰车牌") == []


def _gold(cid, label, class_id=8):
    return {"canonical_id": cid, "class_id": class_id, "class_name": "车", "label": label}


def test_scorer_buckets_and_exec_separation():
    gold = [_gold("g1", "true"), _gold("g2", "true"), _gold("g3", "false"), _gold("g4", "false"),
            _gold("g5", "true"), _gold("g6", "false"), _gold("g7", "true", class_id=10)]
    decisions = [
        {"canonical_id": "g1", "execution_status": "ok", "decision": "true"},    # correct_accept
        {"canonical_id": "g2", "execution_status": "ok", "decision": "false"},   # dangerous_miss
        {"canonical_id": "g3", "execution_status": "ok", "decision": "false"},   # correct_reject
        {"canonical_id": "g4", "execution_status": "ok", "decision": "true"},    # false_accept
        {"canonical_id": "g5", "execution_status": "ok", "decision": "uncertain"},  # uncertain_on_true
        {"canonical_id": "g6", "execution_status": "technical_failure", "decision": "true"},  # -> null+keep
        {"canonical_id": "g7", "execution_status": "protocol_error", "decision": None},
        {"canonical_id": "zz", "execution_status": "ok", "decision": "true"},    # unmatched
    ]
    report, details = score(decisions, gold)
    t = report["totals"]
    assert t["correct_accept"] == 1 and t["dangerous_miss"] == 1
    assert t["correct_reject"] == 1 and t["false_accept"] == 1
    assert t["uncertain_on_true"] == 1 and t["uncertain"] == 1
    assert t["technical_failure"] == 1 and t["protocol_failure"] == 1
    assert t["unmatched_decisions"] == 1
    assert report["scope"] == "candidate_box" and "recall" in report["scope_note"]
    # execution failure => decision null + keep, never judged as a call
    d6 = next(d for d in details if d["canonical_id"] == "g6")
    assert d6["decision"] is None and d6["action"] == "keep" and d6["bucket"] == "technical_failure"
    # invalid decision value degrades to protocol failure
    status, decision = normalize_decision({"execution_status": "ok", "decision": "yes"})
    assert (status, decision) == ("protocol_error", None)
    # per-class denominators and rates (class 8: g1,g2,g5 true; g3,g4,g6 false)
    per_class = {row["class_id"]: row for row in report["per_class"]}
    assert per_class[8]["gold_true"] == 3 and per_class[8]["gold_false"] == 3
    assert per_class[8]["dangerous_miss_rate"]["n"] == 3
    assert per_class[10]["gold_true"] == 1
    assert report["rates"]["dangerous_miss_rate"]["k"] == 1
    lo, hi = wilson_interval(1, 4)
    assert 0.0 <= lo <= 0.25 <= hi <= 1.0
    assert judge("true", "ok", "false") == "dangerous_miss"
    assert judge("false", "ok", "false") == "correct_reject"


def test_scorer_unparseable_lines_and_extract(tmp_path):
    bad_file = tmp_path / "bad.jsonl"
    bad_file.write_text('{"canonical_id": "g1", "decision": "true"}\nnot-json\n', encoding="utf-8")
    rows, bad = load_jsonl(str(bad_file))
    assert len(rows) == 1 and bad == 1

    inference = [
        {"canonical_id": "c1", "prior_yolo": {"raw_decision": "confirmed"}, "prior_gpt": None, "prior_qwen": None},
        {"canonical_id": "c2", "prior_yolo": {"raw_decision": "rejected"}, "prior_gpt": None, "prior_qwen": None},
        {"canonical_id": "c3", "prior_yolo": {"raw_decision": "uncertain"}, "prior_gpt": None, "prior_qwen": None},
        {"canonical_id": "c4", "prior_yolo": None,
         "prior_gpt": {"passes": [{"run_id": "r", "label": "true", "status": "succeeded", "model": "m"}]},
         "prior_qwen": None},
        {"canonical_id": "c5", "prior_yolo": None, "prior_gpt": None,
         "prior_qwen": {"attempts": [{"run_id": "r", "decision": "UNCERTAIN", "status": "failed", "model": "q"}]}},
    ]
    yolo = {r["canonical_id"]: r for r in extract_decisions(inference, "yolo")}
    assert yolo["c1"]["decision"] == "true" and yolo["c2"]["decision"] == "false"
    assert yolo["c3"]["decision"] == "uncertain"
    gpt = {r["canonical_id"]: r for r in extract_decisions(inference, "gpt")}
    assert gpt["c4"]["decision"] == "true" and gpt["c4"]["execution_status"] == "ok"
    qwen = {r["canonical_id"]: r for r in extract_decisions(inference, "qwen")}
    assert qwen["c5"]["execution_status"] == "technical_failure" and qwen["c5"]["decision"] is None
