"""Tests for tools/prompt_eval_data/service_attempts (deployed-service adapter).

Run: PYTHONPATH=backend python3 -m pytest backend/tests/test_service_attempts.py -q
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.prompt_eval_data.scorer import score  # noqa: E402
from tools.prompt_eval_data.service_attempts import (  # noqa: E402
    convert_attempts,
    load_alias_map,
    main,
)


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")


def _gold_fixture(tmp_path: Path) -> Path:
    """Two gold canonicals: c-true (aliases A1, A2), c-false (alias B1). One orphan candidate Z9."""
    rows = [
        {
            "canonical_id": "c-true",
            "class_id": 8,
            "class_name": "车",
            "label": "true",
            "aliases": [
                {"candidate_id": "A1", "sample_id": "s1", "model_run_id": "r1", "target_index": 0},
                {"candidate_id": "A2", "sample_id": "s2", "model_run_id": "r2", "target_index": 0},
            ],
        },
        {
            "canonical_id": "c-false",
            "class_id": 2,
            "class_name": "人员",
            "label": "false",
            "aliases": [
                {"candidate_id": "B1", "sample_id": "s3", "model_run_id": "r1", "target_index": 1},
            ],
        },
    ]
    path = tmp_path / "gold_records.jsonl"
    _write_jsonl(path, rows)
    return path


def _attempt(status: str, started: str, attempt_no: int, candidate_ids: list[str], results: list[dict] | None):
    rec = {
        "attempt_no": attempt_no,
        "status": status,
        "started_at": started,
        "candidate_ids": candidate_ids,
        "response": {"results": results} if results is not None else None,
    }
    return rec


def test_load_alias_map(tmp_path: Path) -> None:
    gold = _gold_fixture(tmp_path)
    alias_map = load_alias_map(str(gold))
    assert alias_map == {"A1": "c-true", "A2": "c-true", "B1": "c-false"}


def test_convert_final_decision_view(tmp_path: Path) -> None:
    gold = _gold_fixture(tmp_path)
    attempts = tmp_path / "attempts.jsonl"
    _write_jsonl(
        attempts,
        [
            _attempt(
                "succeeded",
                "2026-09-03T07:00:00+00:00",
                1,
                ["A1", "B1", "Z9"],
                [
                    {"decision": "confirmed", "yolo_decision": "uncertain", "businessAction": "keep"},
                    {"decision": "rejected", "yolo_decision": "rejected", "businessAction": "candidate_drop"},
                    {"decision": "confirmed", "yolo_decision": "confirmed", "businessAction": "keep"},
                ],
            )
        ],
    )
    rows, stats = convert_attempts(str(attempts), load_alias_map(str(gold)), source="decision")
    assert [r["canonical_id"] for r in rows] == ["c-false", "c-true"]
    by_cid = {r["canonical_id"]: r for r in rows}
    assert by_cid["c-true"]["decision"] == "true"  # confirmed → true
    assert by_cid["c-true"]["execution_status"] == "ok"
    assert by_cid["c-false"]["decision"] == "false"  # rejected → false
    assert stats["no_gold_alias"] == 1  # Z9 skipped
    assert stats["execution_ok"] == 2

    gold_rows = [json.loads(line) for line in gold.read_text(encoding="utf-8").splitlines()]
    report, details = score(rows, gold_rows)
    assert report["totals"]["correct_accept"] == 1
    assert report["totals"]["correct_reject"] == 1
    assert report["totals"]["unmatched_decisions"] == 0


def test_convert_raw_yolo_view_keeps_uncertain(tmp_path: Path) -> None:
    gold = _gold_fixture(tmp_path)
    attempts = tmp_path / "attempts.jsonl"
    _write_jsonl(
        attempts,
        [
            _attempt(
                "succeeded",
                "2026-09-03T07:00:00+00:00",
                1,
                ["A1"],
                [{"decision": "confirmed", "yolo_decision": "uncertain", "businessAction": "keep"}],
            )
        ],
    )
    rows, stats = convert_attempts(str(attempts), load_alias_map(str(gold)), source="yolo_decision")
    assert rows[0]["decision"] == "uncertain"
    assert stats["decision_uncertain"] == 1
    gold_rows = [json.loads(line) for line in gold.read_text(encoding="utf-8").splitlines()]
    report, _ = score(rows, gold_rows)
    assert report["totals"]["uncertain_on_true"] == 1
    assert report["totals"]["unscored_gold"] == 1  # c-false has no decision


def test_failed_attempt_and_bad_result_statuses(tmp_path: Path) -> None:
    gold = _gold_fixture(tmp_path)
    attempts = tmp_path / "attempts.jsonl"
    _write_jsonl(
        attempts,
        [
            _attempt("failed", "2026-09-03T07:00:00+00:00", 1, ["A1"], None),
            _attempt(
                "succeeded",
                "2026-09-03T07:01:00+00:00",
                1,
                ["B1", "A2"],
                [
                    {"decision": None, "errorCode": "TIMEOUT"},
                    {"decision": "bogus-value"},
                ],
            ),
        ],
    )
    rows, stats = convert_attempts(str(attempts), load_alias_map(str(gold)), source="decision")
    by_cid = {r["candidate_id"]: r for r in rows}
    assert by_cid["A1"]["execution_status"] == "technical_failure"  # attempt failed
    assert by_cid["B1"]["execution_status"] == "technical_failure"  # errorCode set
    assert by_cid["A2"]["execution_status"] == "protocol_error"  # unmapped decision value
    assert all(r["decision"] is None for r in rows)
    assert stats["technical_failure"] == 2
    assert stats["protocol_error"] == 1
    gold_rows = [json.loads(line) for line in gold.read_text(encoding="utf-8").splitlines()]
    report, _ = score(rows, gold_rows)
    # A1 and A2 share canonical c-true; the scorer keeps the first emitted row (A1),
    # so A2's protocol_error surfaces only as a duplicate, not as a judged bucket.
    assert report["totals"]["technical_failure"] == 2
    assert report["totals"]["duplicate_decisions"] == 1


def test_retry_latest_attempt_wins(tmp_path: Path) -> None:
    gold = _gold_fixture(tmp_path)
    attempts = tmp_path / "attempts.jsonl"
    _write_jsonl(
        attempts,
        [
            _attempt(
                "failed", "2026-09-03T07:00:00+00:00", 1, ["B1"], None
            ),
            _attempt(
                "succeeded",
                "2026-09-03T07:05:00+00:00",
                2,
                ["B1"],
                [{"decision": "rejected", "yolo_decision": "rejected"}],
            ),
        ],
    )
    rows, stats = convert_attempts(str(attempts), load_alias_map(str(gold)), source="decision")
    assert len(rows) == 1
    assert rows[0]["execution_status"] == "ok"
    assert rows[0]["decision"] == "false"
    assert stats["retry_overrides"] == 1


def test_canonical_conflict_counted(tmp_path: Path) -> None:
    gold = _gold_fixture(tmp_path)
    attempts = tmp_path / "attempts.jsonl"
    _write_jsonl(
        attempts,
        [
            _attempt(
                "succeeded",
                "2026-09-03T07:00:00+00:00",
                1,
                ["A1", "A2"],
                [
                    {"decision": "confirmed"},
                    {"decision": "rejected"},
                ],
            )
        ],
    )
    rows, stats = convert_attempts(str(attempts), load_alias_map(str(gold)), source="decision")
    assert stats["canonical_multi_alias"] == 1
    assert stats["canonical_conflicts"] == 1
    # Deterministic emission order: sorted by (canonical_id, candidate_id) → A1 first.
    assert [r["candidate_id"] for r in rows] == ["A1", "A2"]
    gold_rows = [json.loads(line) for line in gold.read_text(encoding="utf-8").splitlines()]
    report, _ = score(rows, gold_rows)
    assert report["totals"]["duplicate_decisions"] == 1  # scorer keeps first per canonical


def test_cli_writes_outputs(tmp_path: Path) -> None:
    gold = _gold_fixture(tmp_path)
    attempts = tmp_path / "attempts.jsonl"
    _write_jsonl(
        attempts,
        [
            _attempt(
                "succeeded",
                "2026-09-03T07:00:00+00:00",
                1,
                ["A1"],
                [{"decision": "corrected", "businessAction": "keep_manual_review"}],
            )
        ],
    )
    out = tmp_path / "decisions.jsonl"
    stats_path = tmp_path / "stats.json"
    rc = main(
        [
            "--attempts",
            str(attempts),
            "--gold",
            str(gold),
            "--source",
            "decision",
            "--output",
            str(out),
            "--stats",
            str(stats_path),
        ]
    )
    assert rc == 0
    emitted = [json.loads(line) for line in out.read_text(encoding="utf-8").splitlines()]
    assert emitted[0]["decision"] == "true"  # corrected → true
    stats = json.loads(stats_path.read_text(encoding="utf-8"))
    assert stats["emitted_decisions"] == 1
    assert stats["alias_map_candidates"] == 3
