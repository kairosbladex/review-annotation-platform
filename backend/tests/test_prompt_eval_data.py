"""Tests for tools/prompt_eval_data (contract: docs/PROMPT_GOLD_CONTRACT.md).

Self-contained: builds a tiny SQLite fixture and PNG images in tmp_path.
Run: PYTHONPATH=backend python3 -m pytest backend/tests/test_prompt_eval_data.py -q
"""

from __future__ import annotations

import json
import sqlite3
import struct
import sys
import zlib
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.prompt_eval_data import CLASS_MAP, CLASS_NAME_TO_ID  # noqa: E402
from tools.prompt_eval_data.auditor import audit  # noqa: E402
from tools.prompt_eval_data.canon import canonical_id, pixel_bbox, suspect_note  # noqa: E402
from tools.prompt_eval_data.exporter import build_snapshot, write_snapshot  # noqa: E402
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


def build_fixture(tmp_path: Path) -> tuple[Path, Path]:
    db_path = tmp_path / "fixture.sqlite3"
    images = tmp_path / "images"
    con = sqlite3.connect(db_path)
    con.executescript(
        """
        CREATE TABLE samples (id VARCHAR(32) PRIMARY KEY, source_event_id VARCHAR(255),
                              rule_code VARCHAR(120));
        CREATE TABLE assets (id VARCHAR(32) PRIMARY KEY, sample_id VARCHAR(32), kind VARCHAR(40),
                             bucket VARCHAR(255), object_key VARCHAR(1024));
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
        con.execute("INSERT INTO samples VALUES (?, ?, ?)", (sample, f"event-{sample}", "ktd-test"))
        con.execute("INSERT INTO assets VALUES (?, ?, 'original', 'minio-src', ?)", (f"asset-{sample}", sample, key))
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


def test_assign_split_deterministic():
    assert assign_split("ab" * 32) == assign_split("ab" * 32)
    assert assign_split("ab" * 32) in {"selection", "locked_test"}


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
