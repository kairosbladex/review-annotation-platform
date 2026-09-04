"""Tests for tools/prompt_eval_data/paired_scorer (contract section 13).

Self-contained: builds a tiny snapshot fixture (manifest + selection_gold +
selection_inputs) and synthetic v1.0.0/v1.1.0 final decisions in tmp_path.
Run: PYTHONPATH=backend python3 -m pytest backend/tests/test_paired_scorer.py -q
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.prompt_eval_data import CLASS_NAME_TO_ID  # noqa: E402
from tools.prompt_eval_data.common import jdumps  # noqa: E402
from tools.prompt_eval_data.paired_scorer import (  # noqa: E402
    CALLS_NOTE,
    PairedScorerError,
    build_error_packet,
    enrich_from_attempts,
    load_decisions,
    load_gold,
    main,
    paired_score,
    verify_run_terminal,
    verify_snapshot,
)

GOLD_ROWS = [
    # canonical_id, class_name, label
    ("c01", "车", "true"),
    ("c02", "车", "true"),
    ("c03", "车", "false"),
    ("c04", "光伏板", "true"),
    ("c05", "光伏板", "false"),
    ("c06", "违建", "true"),
    ("c07", "违建", "false"),
    ("c08", "人员", "true"),
]
GOLD_IDS = {row[0] for row in GOLD_ROWS}


def _gold_fixture(tmp_path: Path) -> Path:
    snap = tmp_path / "pg1-testfixture"
    snap.mkdir()
    gold_lines = [
        jdumps({
            "canonical_id": cid,
            "class_id": CLASS_NAME_TO_ID[name],
            "class_name": name,
            "label": label,
        })
        for cid, name, label in GOLD_ROWS
    ]
    (snap / "selection_gold.jsonl").write_text("\n".join(gold_lines) + "\n", encoding="utf-8")
    inputs_lines = [
        jdumps({"canonical_id": cid, "image_sha256": f"imgsha-{cid}", "image_object_key": f"key/{cid}.jpg"})
        for cid, _, _ in GOLD_ROWS
    ]
    (snap / "selection_inputs.jsonl").write_text("\n".join(inputs_lines) + "\n", encoding="utf-8")
    files = []
    for name in ("selection_gold.jsonl", "selection_inputs.jsonl"):
        data = (snap / name).read_bytes()
        files.append({"path": name, "sha256": hashlib.sha256(data).hexdigest(), "bytes": len(data)})
    manifest = {"schema_version": "prompt-gold/1", "snapshot_id": snap.name, "files": files}
    (snap / "snapshot_manifest.json").write_text(jdumps(manifest) + "\n", encoding="utf-8")
    return snap


def _decision(cid: str, version: str, decision: str | None, status: str = "succeeded",
              class_name: str | None = None, run_id: str = "run-v10") -> dict:
    row = {
        "canonical_id": cid,
        "execution_status": status,
        "decision": decision,
        "prompt_version": f"ktd.x.baseline.{version}",
        "run_id": run_id,
        "safety_action": "keep" if decision in (None, "uncertain") else "auto_ok" if decision == "true" else "auto_reject",
    }
    if class_name is not None:
        row["class_name"] = class_name
    return row


def _write_decisions(path: Path, rows: list[dict]) -> Path:
    path.write_text("".join(jdumps(row) + "\n" for row in rows), encoding="utf-8")
    return path


def _v10_rows() -> list[dict]:
    name_by_cid = {cid: name for cid, name, _ in GOLD_ROWS}
    decisions = {
        "c01": ("true", "succeeded"),          # correct_accept both
        "c02": ("false", "succeeded"),         # dangerous_miss in v10 -> fixed in v11
        "c03": ("true", "succeeded"),          # false_accept in v10 -> recovered in v11
        "c04": ("uncertain", "succeeded"),     # uncertain_on_true -> resolved in v11
        "c05": (None, "technical_failure"),    # technical_failure both
        "c06": ("true", "succeeded"),          # correct_accept both
        "c07": ("false", "succeeded"),         # correct_reject both
        "c08": ("uncertain", "succeeded"),     # uncertain_on_true, stays uncertain
    }
    return [_decision(cid, "v1.0.0", dec, status, name_by_cid[cid]) for cid, (dec, status) in decisions.items()]


def _v11_rows() -> list[dict]:
    name_by_cid = {cid: name for cid, name, _ in GOLD_ROWS}
    decisions = {
        "c01": ("true", "succeeded"),          # correct_accept
        "c02": ("true", "succeeded"),          # improved: dangerous_miss -> correct_accept
        "c03": ("false", "succeeded"),         # recovered correct_reject
        "c04": ("true", "succeeded"),          # resolved uncertain -> correct_accept
        "c05": (None, "technical_failure"),    # still technical_failure
        "c06": ("uncertain", "succeeded"),     # regressed: correct_accept -> uncertain_on_true
        "c07": ("false", "succeeded"),         # correct_reject
        "c08": ("uncertain", "succeeded"),     # still uncertain_on_true
    }
    return [_decision(cid, "v1.1.0", dec, status, name_by_cid[cid], run_id="run-v11") for cid, (dec, status) in decisions.items()]


def _paired(tmp_path: Path):
    snap = _gold_fixture(tmp_path)
    gold = load_gold(snap / "selection_gold.jsonl")
    v10 = _write_decisions(tmp_path / "v10.jsonl", _v10_rows())
    v11 = _write_decisions(tmp_path / "v11.jsonl", _v11_rows())
    d10, meta10 = load_decisions(v10, "v10", GOLD_IDS)
    d11, meta11 = load_decisions(v11, "v11", GOLD_IDS)
    return gold, d10, d11, meta10, meta11


def test_paired_score_buckets_transitions_and_gate(tmp_path):
    gold, d10, d11, meta10, meta11 = _paired(tmp_path)
    report, packet = paired_score(gold, d10, d11, meta10, meta11)

    v10, v11 = report["v10"], report["v11"]
    assert v10["gold_true"] == 5 and v10["gold_false"] == 3
    # v1.0 buckets: c01/c06 accept, c02 miss, c03 false_accept, c04+c08 uncertain_on_true,
    # c05 technical_failure, c07 correct_reject
    assert v10["correct_accept"] == 2 and v10["dangerous_miss"] == 1
    assert v10["false_accept"] == 1 and v10["correct_reject"] == 1
    assert v10["uncertain_on_true"] == 2 and v10["uncertain_on_false"] == 0
    assert v10["technical_failure"] == 1 and v10["protocol_failure"] == 0
    # v1.1 buckets: c01/c02/c04 accept, c03+c07 reject, c06+c08 uncertain_on_true, c05 tech
    assert v11["correct_accept"] == 3 and v11["dangerous_miss"] == 0
    assert v11["false_accept"] == 0 and v11["correct_reject"] == 2
    assert v11["uncertain_on_true"] == 2 and v11["technical_failure"] == 1
    # conservation: every bucket sums to the gold record count on both sides
    for side in (v10, v11):
        total = sum(side[key] for key in (
            "correct_accept", "dangerous_miss", "correct_reject", "false_accept",
            "uncertain_on_true", "uncertain_on_false", "protocol_failure", "technical_failure",
        ))
        assert total == len(GOLD_ROWS)
    # per-class denominators and rates
    car10 = next(row for row in v10["per_class"] if row["class_name"] == "车")
    assert car10["gold_true"] == 2 and car10["gold_false"] == 1
    assert car10["dangerous_miss_rate"]["n"] == 2 and car10["dangerous_miss_rate"]["k"] == 1
    assert car10["false_accept_rate"]["n"] == 1 and car10["false_accept_rate"]["k"] == 1
    person11 = next(row for row in v11["per_class"] if row["class_name"] == "人员")
    assert person11["gold_true"] == 1 and person11["gold_false"] == 0
    assert person11["false_accept_rate"]["value"] is None  # denominator 0, never fabricated

    paired = report["paired"]
    assert paired["transition_matrix"]["dangerous_miss"] == {"correct_accept": 1}
    assert paired["transition_matrix"]["false_accept"] == {"correct_reject": 1}
    assert paired["transition_matrix"]["correct_accept"] == {"correct_accept": 1, "uncertain_on_true": 1}
    assert paired["improved"] == ["c02", "c03", "c04"]
    assert paired["regressed"] == ["c06"]
    assert paired["new_dangerous_miss"] == []
    assert paired["recovered_correct_reject"] == ["c03"]
    assert paired["uncertain"]["resolved_uncertain"] == ["c04"]
    assert paired["uncertain"]["newly_uncertain"] == ["c06"]

    gate = report["gate"]
    assert gate["passed"] is True and gate["verdict"] == "improved"
    assert gate["checks"]["v11_dangerous_miss_zero"] is True

    cvb = report["calls_vs_boxes"]
    assert cvb["calls_total"] == 16 and cvb["unique_boxes"] == 8
    assert "876" in CALLS_NOTE and "438" in CALLS_NOTE

    # packet: only rows with a non-correct bucket on at least one side
    packet_ids = [row["canonical_id"] for row in packet]
    assert packet_ids == ["c02", "c03", "c04", "c05", "c06", "c08"]
    row = packet[0]
    assert set(row) == {"schema_version", "canonical_id", "class_name", "image", "v10", "v11", "error_buckets"}
    assert "label" not in row and "gold" not in row and "note" not in row and "reviewer" not in row


def test_technical_failure_and_protocol_error_are_keep_not_true_false(tmp_path):
    gold, d10, d11, meta10, meta11 = _paired(tmp_path)
    # c05 technical_failure carries no decision; force a contradictory row: status ok + junk decision
    d10["c05"] = {"canonical_id": "c05", "execution_status": "ok", "decision": "junk",
                  "class_name": "光伏板", "prompt_version": "ktd.x.baseline.v1.0.0", "safety_action": None}
    report, _ = paired_score(gold, d10, d11, meta10, meta11)
    assert report["v10"]["protocol_failure"] == 1
    assert report["v10"]["technical_failure"] == 0
    # protocol failure is never counted as a true/false call
    assert report["v10"]["correct_accept"] + report["v10"]["dangerous_miss"] == 3
    # succeeded is accepted as an alias of ok; technical_failure keeps decision=None
    assert d10["c01"]["execution_status"] == "ok"
    assert d10["c05"]["decision"] is None


def test_duplicate_missing_extra_fail(tmp_path):
    snap = _gold_fixture(tmp_path)
    gold = load_gold(snap / "selection_gold.jsonl")
    dup = _v10_rows() + [_decision("c01", "v1.0.0", "true")]
    with pytest.raises(PairedScorerError, match="duplicate"):
        load_decisions(_write_decisions(tmp_path / "dup.jsonl", dup), "v10", GOLD_IDS)
    missing = [row for row in _v10_rows() if row["canonical_id"] != "c01"]
    with pytest.raises(PairedScorerError, match="missing 1"):
        load_decisions(_write_decisions(tmp_path / "missing.jsonl", missing), "v10", GOLD_IDS)
    extra = _v10_rows() + [_decision("c99", "v1.0.0", "true")]
    with pytest.raises(PairedScorerError, match="extra 1"):
        load_decisions(_write_decisions(tmp_path / "extra.jsonl", extra), "v10", GOLD_IDS)
    bad_line = tmp_path / "bad.jsonl"
    bad_line.write_text(jdumps(_v10_rows()[0]) + "\n{not json\n", encoding="utf-8")
    with pytest.raises(PairedScorerError, match="not valid JSON"):
        load_decisions(bad_line, "v10", GOLD_IDS)


def test_version_and_run_id_mismatch_fail(tmp_path):
    wrong_version = [_decision(row["canonical_id"], "v1.1.0", "true") for row in _v10_rows()]
    with pytest.raises(PairedScorerError, match="version mismatch"):
        load_decisions(_write_decisions(tmp_path / "wrong.jsonl", wrong_version), "v10", GOLD_IDS)
    no_version = [_decision(r[0], "v1.0.0", "true") for r in GOLD_ROWS]
    for row in no_version:
        del row["prompt_version"]
    with pytest.raises(PairedScorerError, match="version mismatch"):
        load_decisions(_write_decisions(tmp_path / "noversion.jsonl", no_version), "v10", GOLD_IDS)
    multi_run = _v10_rows()
    multi_run[0]["run_id"] = "run-other"
    with pytest.raises(PairedScorerError, match="multiple run_id"):
        load_decisions(_write_decisions(tmp_path / "multirun.jsonl", multi_run), "v10", GOLD_IDS)


def test_class_name_mismatch_fails(tmp_path):
    gold, d10, d11, meta10, meta11 = _paired(tmp_path)
    d10["c01"]["class_name"] = "卡车"
    with pytest.raises(PairedScorerError, match="class_name"):
        paired_score(gold, d10, d11, meta10, meta11)


def test_gold_integrity_checks(tmp_path):
    snap = _gold_fixture(tmp_path)
    gold = load_gold(snap / "selection_gold.jsonl")
    assert len(gold) == 8
    rows = [jdumps({"canonical_id": "c01", "class_id": 999, "class_name": "车", "label": "true"})]
    bad = tmp_path / "badgold.jsonl"
    bad.write_text("\n".join(rows) + "\n", encoding="utf-8")
    with pytest.raises(PairedScorerError, match="class mismatch"):
        load_gold(bad)


def test_snapshot_verification(tmp_path):
    snap = _gold_fixture(tmp_path)
    info = verify_snapshot(snap, expected_manifest_sha=None)
    assert info["snapshot_id"] == snap.name
    with pytest.raises(PairedScorerError, match="manifest SHA mismatch"):
        verify_snapshot(snap, expected_manifest_sha="0" * 64)
    # tamper with the gold file after manifest was written
    (snap / "selection_gold.jsonl").write_text("{}\n", encoding="utf-8")
    with pytest.raises(PairedScorerError, match="SHA mismatch"):
        verify_snapshot(snap, expected_manifest_sha=None)


def test_run_terminal_verification(tmp_path):
    running = tmp_path / "running.json"
    running.write_text(jdumps({"run_id": "r1", "status": "running"}) + "\n", encoding="utf-8")
    with pytest.raises(PairedScorerError, match="not terminal"):
        verify_run_terminal(running, None)
    ok = tmp_path / "ok.json"
    ok.write_text(
        jdumps({"run_id": "r1", "status": "succeeded", "lineage": {"snapshot_sha": "abc"}}) + "\n",
        encoding="utf-8",
    )
    assert verify_run_terminal(ok, "abc")["run_id"] == "r1"
    with pytest.raises(PairedScorerError, match="snapshot SHA mismatch"):
        verify_run_terminal(ok, "def")


def test_gate_regressed_on_new_dangerous_miss(tmp_path):
    gold, d10, d11, meta10, meta11 = _paired(tmp_path)
    d11["c06"] = {"canonical_id": "c06", "execution_status": "ok", "decision": "false",
                  "class_name": "违建", "prompt_version": "ktd.x.baseline.v1.1.0", "safety_action": "auto_reject"}
    report, packet = paired_score(gold, d10, d11, meta10, meta11)
    assert report["v11"]["dangerous_miss"] == 1
    assert report["gate"]["passed"] is False and report["gate"]["verdict"] == "regressed"
    assert report["paired"]["new_dangerous_miss"] == ["c06"]
    assert "保留 v1.0.0" in report["gate"]["recommendation"]


def test_gate_tie_keeps_v10(tmp_path):
    gold, d10, d11, meta10, meta11 = _paired(tmp_path)
    # make v1.1 identical to v1.0 (same buckets, including the dangerous_miss) -> not passed
    # (dm not zero) is impossible for a tie; instead mirror with the dangerous row fixed in both:
    d10["c02"] = {"canonical_id": "c02", "execution_status": "ok", "decision": "true",
                  "class_name": "车", "prompt_version": "ktd.x.baseline.v1.0.0", "safety_action": "auto_ok"}
    d11["c02"] = dict(d10["c02"], prompt_version="ktd.x.baseline.v1.1.0")
    d11["c03"] = {"canonical_id": "c03", "execution_status": "ok", "decision": "true",
                  "class_name": "车", "prompt_version": "ktd.x.baseline.v1.1.0", "safety_action": "auto_ok"}
    d11["c04"] = {"canonical_id": "c04", "execution_status": "ok", "decision": "uncertain",
                  "class_name": "光伏板", "prompt_version": "ktd.x.baseline.v1.1.0", "safety_action": "keep"}
    d11["c06"] = {"canonical_id": "c06", "execution_status": "ok", "decision": "true",
                  "class_name": "违建", "prompt_version": "ktd.x.baseline.v1.1.0", "safety_action": "auto_ok"}
    report, _ = paired_score(gold, d10, d11, meta10, meta11)
    assert report["v10"]["dangerous_miss"] == 0 and report["v11"]["dangerous_miss"] == 0
    assert report["gate"]["passed"] is True
    assert report["gate"]["verdict"] == "tie_keep_v1.0"
    assert "不得写成提升" in report["gate"]["recommendation"]
    assert report["paired"]["improved"] == [] and report["paired"]["regressed"] == []


def test_enrich_from_attempts_latest_wins(tmp_path):
    gold, d10, d11, meta10, meta11 = _paired(tmp_path)
    manifest = tmp_path / "prompt_manifest.json"
    manifest.write_text(jdumps([{"prompt_key": "baseline_x", "version": "ktd.x.baseline.v1.0.0"}]) + "\n", encoding="utf-8")
    attempts = tmp_path / "attempts.jsonl"
    rows = [
        {"record_key": "c02", "prompt_key": "baseline_x", "attempt_no": 1, "attempt_id": "a1",
         "parsed_result": {"target_present": "no", "class_match": "not_applicable",
                           "scene_match": "not_applicable", "reason_codes": ["target_absent"]}},
        {"record_key": "c02", "prompt_key": "baseline_x", "attempt_no": 2, "attempt_id": "a2",
         "parsed_result": {"target_present": "yes", "class_match": "no",
                           "scene_match": "not_applicable", "reason_codes": ["class_mismatch"]}},
        {"record_key": "c99", "prompt_key": "baseline_x", "attempt_no": 1, "attempt_id": "a9",
         "parsed_result": {"target_present": "yes"}},
    ]
    _write_decisions(attempts, rows)
    enrich_from_attempts(d10, attempts, manifest, "v10")
    assert d10["c02"]["axes"]["class_match"] == "no"  # attempt 2 wins
    assert d10["c02"]["reason_codes"] == ["class_mismatch"]
    assert "c99" not in d10
    packet = build_error_packet(gold, {"c02": {"image_sha256": "s", "image_object_key": "k"}}, d10, d11,
                                {cid: "dangerous_miss" for cid in GOLD_IDS},
                                {cid: "correct_accept" for cid in GOLD_IDS})
    c02 = next(row for row in packet if row["canonical_id"] == "c02")
    assert c02["v10"]["axes"] == {"target_present": "yes", "class_match": "no", "scene_match": "not_applicable"}
    assert c02["v10"]["reason_codes"] == ["class_mismatch"]
    assert c02["v11"]["axes"] is None


def test_cli_end_to_end_deterministic(tmp_path, capsys):
    snap = _gold_fixture(tmp_path)
    manifest_sha = hashlib.sha256((snap / "snapshot_manifest.json").read_bytes()).hexdigest()
    v10 = _write_decisions(tmp_path / "v10.jsonl", _v10_rows())
    v11 = _write_decisions(tmp_path / "v11.jsonl", _v11_rows())
    report10 = tmp_path / "report10.json"
    report11 = tmp_path / "report11.json"
    report10.write_text(
        jdumps({"run_id": "run-v10", "status": "succeeded", "lineage": {"snapshot_sha": manifest_sha}}) + "\n",
        encoding="utf-8",
    )
    report11.write_text(
        jdumps({"run_id": "run-v11", "status": "partial_failed", "lineage": {"snapshot_sha": manifest_sha}}) + "\n",
        encoding="utf-8",
    )
    out_json, out_md, out_packet = (tmp_path / "o.json", tmp_path / "o.md", tmp_path / "o.jsonl")
    argv = [
        "score", "--v10", str(v10), "--v11", str(v11),
        "--report-v10", str(report10), "--report-v11", str(report11),
        "--snapshot-dir", str(snap), "--expect-manifest-sha", manifest_sha,
        "--report-json", str(out_json), "--report-md", str(out_md), "--error-packet", str(out_packet),
    ]
    assert main(argv) == 0
    first = (out_json.read_bytes(), out_md.read_bytes(), out_packet.read_bytes())
    assert main(argv) == 0
    assert first == (out_json.read_bytes(), out_md.read_bytes(), out_packet.read_bytes())
    report = json.loads(out_json.read_text(encoding="utf-8"))
    assert report["schema_version"] == "prompt-gold-paired-score/1"
    assert report["scope"] == "candidate_box"
    assert report["gate"]["verdict"] == "improved"
    assert report["snapshot"]["manifest_sha256"] == manifest_sha
    md = out_md.read_text(encoding="utf-8")
    assert "candidate_box" in md and "门禁" in md and "迁移矩阵" in md
    # non-terminal run -> exit 2, no outputs written
    report11.write_text(jdumps({"run_id": "run-v11", "status": "running"}) + "\n", encoding="utf-8")
    assert main(argv) == 2
    assert "PAIRED SCORE FAILED" in capsys.readouterr().err


def test_cli_run_id_mismatch_fails(tmp_path):
    snap = _gold_fixture(tmp_path)
    v10 = _write_decisions(tmp_path / "v10.jsonl", _v10_rows())
    v11 = _write_decisions(tmp_path / "v11.jsonl", _v11_rows())
    report10 = tmp_path / "report10.json"
    report11 = tmp_path / "report11.json"
    report10.write_text(jdumps({"run_id": "run-OTHER", "status": "succeeded"}) + "\n", encoding="utf-8")
    report11.write_text(jdumps({"run_id": "run-v11", "status": "succeeded"}) + "\n", encoding="utf-8")
    rc = main([
        "score", "--v10", str(v10), "--v11", str(v11),
        "--report-v10", str(report10), "--report-v11", str(report11),
        "--snapshot-dir", str(snap), "--expect-manifest-sha",
        hashlib.sha256((snap / "snapshot_manifest.json").read_bytes()).hexdigest(),
        "--report-json", str(tmp_path / "o.json"), "--report-md", str(tmp_path / "o.md"),
        "--error-packet", str(tmp_path / "o.jsonl"),
    ])
    assert rc == 2


def test_real_snapshot_gold_loads():
    """The fixed snapshot must parse under the strict paired rules (438 ids)."""
    snap = REPO_ROOT / "data/prompt-eval/snapshots/pg1-bc8c818184d8"
    info = verify_snapshot(snap)  # default pins the published manifest SHA
    assert info["snapshot_id"] == "pg1-bc8c818184d8"
    gold = load_gold(info["gold_path"])
    assert len(gold) == 438
    assert sum(1 for row in gold.values() if row["label"] == "true") + sum(
        1 for row in gold.values() if row["label"] == "false"
    ) == 438
