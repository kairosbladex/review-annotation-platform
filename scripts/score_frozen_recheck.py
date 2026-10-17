#!/usr/bin/env python3
"""Score existing JSONL predictions against frozen tasks and human labels offline.

Only aggregate counts and file hashes leave this tool. No models, images,
credentials, databases or network clients are loaded. Missing and failed boxes
remain in the frozen denominator; neither becomes a negative prediction.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import sys

from score_luna_annotated import FALSE, TRUE, ratio, score


def require(condition: bool) -> None:
    if not condition:
        raise ValueError("INPUT_CONTRACT_ERROR")


def named(value) -> bool:
    return isinstance(value, str) and bool(value.strip())


def valid_box(value) -> bool:
    return (isinstance(value, list) and len(value) == 4
            and all(type(n) in (int, float) and math.isfinite(n) for n in value)
            and 0 <= value[0] < value[2] <= 1 and 0 <= value[1] < value[3] <= 1)


def bind_inputs(source: list, labels: list) -> tuple[dict, dict]:
    require(isinstance(source, list) and isinstance(labels, list))
    events, candidates, gold = {}, {}, {}
    for task in source:
        require(isinstance(task, dict) and named(task.get("event_id")))
        event = task["event_id"]
        require(event not in events and isinstance(task.get("candidates"), list))
        ids = []
        for index, candidate in enumerate(task["candidates"]):
            require(isinstance(candidate, dict) and named(candidate.get("candidate_id")))
            cid = candidate["candidate_id"]
            require(cid not in candidates and type(candidate.get("index")) is int
                    and candidate["index"] == index and named(candidate.get("class_name"))
                    and valid_box(candidate.get("bbox")))
            candidates[cid] = (event, candidate)
            ids.append(cid)
        events[event] = ids
    for row in labels:
        require(isinstance(row, dict) and named(row.get("candidate_id")))
        cid = row["candidate_id"]
        require(cid not in gold and cid in candidates)
        event, candidate = candidates[cid]
        require(row.get("event_id") == event and row.get("class_name") == candidate["class_name"]
                and valid_box(row.get("bbox")) and row["bbox"] == candidate["bbox"]
                and named(row.get("group")) and type(row.get("scorable")) is bool)
        require((row["scorable"] and row.get("gold") in TRUE | FALSE)
                or (not row["scorable"] and row.get("gold") == "看不清"))
        gold[cid] = row
    require(set(gold) == set(candidates))
    return events, gold


def read_decisions(records: list, events: dict) -> tuple[dict, set, dict]:
    raw, failed, seen = {}, set(), set()
    failed_events = invalid_events = 0
    for record in records:
        require(isinstance(record, dict) and named(record.get("event_id")))
        event = record["event_id"]
        require(event in events and event not in seen)
        ids = events[event]
        require(record.get("candidate_ids") == ids)
        seen.add(event)
        if record.get("error"):
            failed.update(ids)
            failed_events += 1
            continue
        parsed = record.get("parsed")
        results = parsed.get("results") if isinstance(parsed, dict) else None
        valid = isinstance(results, list)
        mapped = {}
        if valid:
            for result in results:
                if not isinstance(result, dict):
                    valid = False
                    break
                index, decision = result.get("target_index"), result.get("decision")
                if (type(index) is not int or not 0 <= index < len(ids) or index in mapped
                        or decision not in ("true", "false", "uncertain")):
                    valid = False
                    break
                mapped[index] = decision
        if not valid:
            failed.update(ids)
            failed_events += 1
            invalid_events += 1
        else:
            raw.update({ids[index]: decision for index, decision in mapped.items()})
    return raw, failed, {"present_events": len(seen), "missing_events": len(events) - len(seen),
                         "failed_events": failed_events, "invalid_response_events": invalid_events}


def aggregate(rows: dict, raw: dict, failed: set, policy: str) -> dict:
    labels = {cid: row["gold"] in TRUE for cid, row in rows.items() if row["scorable"]}
    predictions = {
        cid: ("uncertain" if policy == "manual" and decision == "uncertain"
              else "confirmed" if decision == "true" else "rejected")
        for cid, decision in raw.items() if cid in labels
    }
    # Reuse the existing fixed-label confusion accounting; split its absent
    # predictions into missing results and explicitly failed executions.
    result = score(labels, predictions)
    failed_count = len(set(labels) & failed)
    count_keys = ("total", "available", "right", "wrong", "manual", "true_targets",
                  "non_targets", "wrongly_rejected", "wrongly_confirmed", "decided")
    counts = {key: result[key] for key in count_keys}
    counts.update(
        missing=result["missing"] - failed_count, failed=failed_count,
        excluded_gold=len(rows) - len(labels),
        raw_uncertain=sum(raw.get(cid) == "uncertain" for cid in labels),
        accuracy_decided=result["accuracy_decided"],
        coverage=ratio(result["decided"], result["total"]),
        response_coverage=ratio(result["available"], result["total"]),
        correct_fraction_total=ratio(result["right"], result["total"]),
        false_rejection_rate_total=ratio(result["wrongly_rejected"], result["true_targets"]),
        false_acceptance_rate_total=ratio(result["wrongly_confirmed"], result["non_targets"]),
    )
    return counts


def score_frozen(source: list, labels: list, records: list) -> dict:
    events, gold = bind_inputs(source, labels)
    raw, failed, execution = read_decisions(records, events)
    report = {
        "schema_version": "frozen-recheck-score-v1",
        "dataset": {"events": len(events), "candidates": len(gold),
                    "scorable": sum(row["scorable"] for row in gold.values()),
                    "excluded_gold": sum(not row["scorable"] for row in gold.values())},
        "execution": execution,
        "metric_definitions": {
            "partition": "total = right + wrong + manual + missing + failed",
            "coverage": "decided / total frozen scorable labels",
            "response_coverage": "available valid decisions / total frozen scorable labels",
            "accuracy_decided": "right / decided; conditional accuracy, not full coverage",
            "correct_fraction_total": "right / total frozen scorable labels",
            "online": "only true is confirmed; false and uncertain are rejected",
            "manual": "uncertain is referred; missing and failed remain separate",
            "binding": "file/candidate alignment only; not trusted original-frame provenance",
        },
        "policies": {},
    }
    for policy in ("online", "manual"):
        buckets = {"all": aggregate(gold, raw, failed, policy)}
        for output_key, field in (("groups", "group"), ("classes", "class_name")):
            buckets[output_key] = {
                name: aggregate({cid: row for cid, row in gold.items() if row[field] == name}, raw, failed, policy)
                for name in sorted({row[field] for row in gold.values()})
            }
        report["policies"][policy] = buckets
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("source", "labels", "predictions"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--out", type=Path, help="Optional aggregate-only JSON report")
    args = parser.parse_args()
    try:
        paths = {name: getattr(args, name) for name in ("source", "labels", "predictions")}
        require(args.out is None or args.out.resolve() not in {path.resolve() for path in paths.values()})
        require(all(path.stat().st_size <= 25_000_000 for path in paths.values()))
        frozen = {name: path.read_bytes() for name, path in paths.items()}
        report = score_frozen(json.loads(frozen["source"]), json.loads(frozen["labels"]),
                              [json.loads(line) for line in frozen["predictions"].splitlines() if line.strip()])
        report["input_sha256"] = {name: hashlib.sha256(data).hexdigest() for name, data in frozen.items()}
        report["implementation_sha256"] = {
            name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
            for name in ("score_frozen_recheck.py", "score_luna_annotated.py")
        }
        report["generated_at"] = datetime.now(timezone.utc).isoformat()
        encoded = json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
        if args.out:
            args.out.parent.mkdir(parents=True, exist_ok=True)
            args.out.write_text(encoded, encoding="utf-8")
        print(encoded, end="")
    except (ValueError, TypeError, KeyError, OSError):
        print("INPUT_CONTRACT_ERROR", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
