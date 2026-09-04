"""Deployed review-service attempts → scorer decisions adapter (contract section 8).

Converts prefill/service attempt logs (e.g. ``data/ktd-8188-kimi-prefill-20260903/attempts.jsonl``
produced against the GPU review service ``/api/v1/review-agent/review``) into the unified
scorer decisions format, joining ``candidate_id`` → ``canonical_id`` via the aliases of a
prompt-gold snapshot's ``gold_records.jsonl``.

Two decision sources exist in the service response and must not be conflated:

- ``decision`` — the *final applied business verdict*. On the deployed 8188 service every
  ``uncertain`` from the verifier is converted to ``confirmed`` ("keep by product design"),
  so this view measures what the platform actually applies in production.
- ``yolo_decision`` — the verifier's *raw three-state output* (``confirmed`` / ``corrected`` /
  ``rejected`` / ``uncertain``). This view measures the verifier's real discrimination.

Both are mapped through the scorer's YOLO vocabulary:
``confirmed/corrected → true``, ``rejected → false``, ``uncertain → uncertain``.

Usage:
    python3 -m tools.prompt_eval_data.service_attempts \
        --attempts attempts.jsonl --gold gold_records.jsonl \
        --source decision --output decisions.jsonl [--stats stats.json]
"""

from __future__ import annotations

import argparse
import json
import sys

from .common import jdumps

ADAPTER_SCHEMA_VERSION = "prompt-gold-service-attempts/1"

#: Fields in a service result entry that may serve as the decision source.
SOURCES = ("decision", "yolo_decision")

#: Service vocabulary → scorer three-state (identical to scorer._YOLO_MAP).
_SERVICE_MAP = {"confirmed": "true", "corrected": "true", "rejected": "false", "uncertain": "uncertain"}


def load_alias_map(gold_records_path: str) -> dict[str, str]:
    """candidate_id → canonical_id from every alias of every gold record."""
    alias_map: dict[str, str] = {}
    with open(gold_records_path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            for alias in row.get("aliases") or []:
                cid = alias.get("candidate_id")
                if cid:
                    alias_map[cid] = row["canonical_id"]
    return alias_map


def _result_status(result: dict | None, source: str) -> tuple[str, str | None]:
    """(execution_status, decision) for one service result entry."""
    if result is None:
        return "protocol_error", None
    if result.get("errorCode"):
        return "technical_failure", None
    raw = result.get(source)
    decision = _SERVICE_MAP.get(str(raw)) if raw is not None else None
    if decision is None:
        return "protocol_error", None
    return "ok", decision


def convert_attempts(
    attempts_path: str,
    alias_map: dict[str, str],
    source: str = "decision",
) -> tuple[list[dict], dict]:
    """Convert one attempts.jsonl into scorer decisions.

    Retry semantics: when the same candidate_id appears in several attempt rows
    (resumed prefill), the row with the latest (started_at, attempt_no) wins.
    Emitted rows are sorted by (canonical_id, candidate_id) for determinism.
    Candidates without a gold alias are skipped and counted (they simply have no
    human label to score against).
    """
    if source not in SOURCES:
        raise ValueError(f"unknown source: {source!r} (expected one of {SOURCES})")

    stats = {
        "schema_version": ADAPTER_SCHEMA_VERSION,
        "attempts_path": attempts_path,
        "source": source,
        "attempt_rows": 0,
        "attempt_succeeded": 0,
        "attempt_failed": 0,
        "result_entries": 0,
        "candidates_seen": 0,
        "no_gold_alias": 0,
        "retry_overrides": 0,
        "execution_ok": 0,
        "protocol_error": 0,
        "technical_failure": 0,
        "decision_true": 0,
        "decision_false": 0,
        "decision_uncertain": 0,
    }
    # candidate_id -> (sort_key, decision_row); latest attempt wins.
    by_candidate: dict[str, tuple[tuple, dict]] = {}

    with open(attempts_path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            stats["attempt_rows"] += 1
            attempt_ok = rec.get("status") == "succeeded"
            stats["attempt_succeeded" if attempt_ok else "attempt_failed"] += 1
            sort_key = (str(rec.get("started_at") or ""), int(rec.get("attempt_no") or 0))
            candidate_ids = rec.get("candidate_ids") or []
            results = (rec.get("response") or {}).get("results") or []
            for idx, candidate_id in enumerate(candidate_ids):
                stats["candidates_seen"] += 1
                stats["result_entries"] += 1
                canonical_id = alias_map.get(candidate_id)
                if canonical_id is None:
                    stats["no_gold_alias"] += 1
                    continue
                if attempt_ok:
                    result = results[idx] if idx < len(results) else None
                    status, decision = _result_status(result, source)
                else:
                    status, decision = "technical_failure", None
                stats["execution_ok" if status == "ok" else status] += 1
                if decision:
                    stats[f"decision_{decision}"] += 1
                row = {
                    "canonical_id": canonical_id,
                    "execution_status": status,
                    "decision": decision,
                    "model": "deployed-service",
                    "candidate_id": candidate_id,
                    "source_field": source,
                    "raw_value": (results[idx] or {}).get(source) if attempt_ok and idx < len(results) else None,
                    "business_action": (results[idx] or {}).get("businessAction") if attempt_ok and idx < len(results) else None,
                }
                prev = by_candidate.get(candidate_id)
                if prev is not None:
                    stats["retry_overrides"] += 1
                if prev is None or sort_key >= prev[0]:
                    by_candidate[candidate_id] = (sort_key, row)

    rows = [entry[1] for _, entry in sorted(by_candidate.items(), key=lambda kv: (kv[1][1]["canonical_id"], kv[0]))]

    # Same canonical judged through several aliases (e.g. duplicated model runs):
    # the scorer keeps the first emitted row; count and quantify disagreements here.
    by_canonical: dict[str, set] = {}
    for row in rows:
        by_canonical.setdefault(row["canonical_id"], set()).add(
            (row["execution_status"], row["decision"])
        )
    alias_counts: dict[str, int] = {}
    for row in rows:
        alias_counts[row["canonical_id"]] = alias_counts.get(row["canonical_id"], 0) + 1
    stats["emitted_decisions"] = len(rows)
    stats["unique_canonicals"] = len(by_canonical)
    stats["canonical_multi_alias"] = sum(1 for n in alias_counts.values() if n > 1)
    stats["canonical_conflicts"] = sum(
        1 for cid, n in alias_counts.items() if n > 1 and len(by_canonical[cid]) > 1
    )
    return rows, stats


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Convert deployed-service attempts into scorer decisions.")
    parser.add_argument("--attempts", required=True, help="attempts.jsonl from the prefill/service run")
    parser.add_argument("--gold", required=True, help="snapshot gold_records.jsonl (alias source)")
    parser.add_argument("--source", default="decision", choices=SOURCES)
    parser.add_argument("--output", default=None, help="decisions JSONL path (default: stdout)")
    parser.add_argument("--stats", default=None, help="optional stats JSON path")
    args = parser.parse_args(argv)

    alias_map = load_alias_map(args.gold)
    rows, stats = convert_attempts(args.attempts, alias_map, args.source)
    stats["alias_map_candidates"] = len(alias_map)

    out_text = "".join(jdumps(row) + "\n" for row in rows)
    if args.output:
        with open(args.output, "w", encoding="utf-8") as fh:
            fh.write(out_text)
    else:
        sys.stdout.write(out_text)
    stats_text = jdumps(stats) + "\n"
    if args.stats:
        with open(args.stats, "w", encoding="utf-8") as fh:
            fh.write(stats_text)
    else:
        sys.stderr.write(stats_text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
