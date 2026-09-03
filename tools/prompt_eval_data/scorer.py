"""Unified prompt scorer — contract section 8.

Final three-state decisions are kept separate from execution status:
technical failures and invalid JSON/protocol errors force decision=null and the
record is kept for review ("keep"), never judged as a true/false call.

Usage:
    python3 -m tools.prompt_eval_data.scorer score --decisions D.jsonl --gold G.jsonl [--details-out D2.jsonl]
    python3 -m tools.prompt_eval_data.scorer extract --inference inference_records.jsonl --prior yolo|gpt|qwen
"""

from __future__ import annotations

import argparse
import json
import sys

from . import CLASS_ID_TO_NAME, SCORE_SCHEMA_VERSION
from .common import jdumps

EXECUTION_STATUSES = {"ok", "protocol_error", "technical_failure"}
DECISIONS = {"true", "false", "uncertain"}

SCOPE_NOTE = (
    "候选框级评判：只衡量模型已产生候选框的判定质量；未覆盖漏检目标，"
    "不能代表整图 recall/mAP。"
)

_YOLO_MAP = {"confirmed": "true", "corrected": "true", "rejected": "false", "uncertain": "uncertain"}
_QWEN_MAP = {"CONFIRMED": "true", "REJECTED": "false", "UNCERTAIN": "uncertain"}


def wilson_interval(k: int, n: int, z: float = 1.96) -> list[float]:
    if n == 0:
        return [0.0, 0.0]
    p = k / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    margin = z * ((p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5) / denom
    return [round(max(0.0, center - margin), 6), round(min(1.0, center + margin), 6)]


def _rate(k: int, n: int) -> dict:
    return {"value": round(k / n, 6) if n else None, "ci95": wilson_interval(k, n), "k": k, "n": n}


def load_jsonl(path: str) -> tuple[list[dict], int]:
    """Parse a JSONL decisions/gold file; bad lines are counted, not fatal."""
    rows, bad = [], 0
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                bad += 1
    return rows, bad


def normalize_decision(record: dict) -> tuple[str, str | None]:
    """(execution_status, decision). Invalid decision values => protocol_error."""
    status = str(record.get("execution_status") or "ok")
    if status not in EXECUTION_STATUSES:
        status = "protocol_error"
    decision = record.get("decision")
    if status != "ok":
        return status, None  # technical failure / bad protocol => null + keep
    if decision is None:
        return "protocol_error", None
    decision = str(decision).lower()
    if decision not in DECISIONS:
        return "protocol_error", None
    return "ok", decision


_BUCKET_KEYS = [
    "correct_accept",
    "dangerous_miss",
    "correct_reject",
    "false_accept",
    "uncertain_on_true",
    "uncertain_on_false",
    "protocol_failure",
    "technical_failure",
]


def _empty_buckets() -> dict:
    return {key: 0 for key in _BUCKET_KEYS}


def judge(gold_label: str, status: str, decision: str | None) -> str:
    if status == "protocol_error":
        return "protocol_failure"
    if status == "technical_failure":
        return "technical_failure"
    if decision == "uncertain":
        return "uncertain_on_true" if gold_label == "true" else "uncertain_on_false"
    if gold_label == "true":
        return "correct_accept" if decision == "true" else "dangerous_miss"
    return "false_accept" if decision == "true" else "correct_reject"


def score(decision_rows: list[dict], gold_rows: list[dict], unparseable: int = 0) -> tuple[dict, list[dict]]:
    gold_by_cid = {row["canonical_id"]: row for row in gold_rows}
    decisions_by_cid = {}
    for row in decision_rows:
        cid = row.get("canonical_id")
        if cid:
            decisions_by_cid.setdefault(cid, row)  # first wins: deterministic

    totals = _empty_buckets()
    per_class: dict[int, dict] = {}
    details = []
    unmatched = 0
    for cid, row in decisions_by_cid.items():
        if cid not in gold_by_cid:
            unmatched += 1
    unscored = 0
    for cid in sorted(gold_by_cid):
        gold = gold_by_cid[cid]
        label = gold["label"]
        class_id = gold["class_id"]
        bucket_row = per_class.setdefault(
            class_id,
            {"class_id": class_id, "class_name": CLASS_ID_TO_NAME.get(class_id, str(class_id)),
             "gold_true": 0, "gold_false": 0, **_empty_buckets()},
        )
        bucket_row["gold_true" if label == "true" else "gold_false"] += 1
        decision_row = decisions_by_cid.get(cid)
        if decision_row is None:
            unscored += 1
            continue
        status, decision = normalize_decision(decision_row)
        bucket = judge(label, status, decision)
        totals[bucket] += 1
        bucket_row[bucket] += 1
        details.append(
            {
                "canonical_id": cid,
                "class_id": class_id,
                "gold_label": label,
                "execution_status": status,
                "decision": decision,
                "action": "keep" if decision is None else ("review" if decision == "uncertain" else "act"),
                "bucket": bucket,
            }
        )

    gold_true = sum(r["gold_true"] for r in per_class.values())
    gold_false = sum(r["gold_false"] for r in per_class.values())
    for row in per_class.values():
        row["dangerous_miss_rate"] = _rate(row["dangerous_miss"], row["gold_true"])
        row["false_accept_rate"] = _rate(row["false_accept"], row["gold_false"])

    report = {
        "schema_version": SCORE_SCHEMA_VERSION,
        "scope": "candidate_box",
        "scope_note": SCOPE_NOTE,
        "totals": {
            "gold_records": len(gold_rows),
            "gold_true": gold_true,
            "gold_false": gold_false,
            "decision_records": len(decision_rows),
            "matched_decisions": len(decision_rows) - unmatched - (len(decision_rows) - len(decisions_by_cid)),
            "unmatched_decisions": unmatched,
            "duplicate_decisions": len(decision_rows) - len(decisions_by_cid),
            "unscored_gold": unscored,
            "unparseable_lines": unparseable,
            **totals,
            "uncertain": totals["uncertain_on_true"] + totals["uncertain_on_false"],
        },
        "rates": {
            "dangerous_miss_rate": _rate(totals["dangerous_miss"], gold_true),
            "false_accept_rate": _rate(totals["false_accept"], gold_false),
            "uncertain_rate": _rate(
                totals["uncertain_on_true"] + totals["uncertain_on_false"], gold_true + gold_false
            ),
        },
        "per_class": [per_class[cid] for cid in sorted(per_class)],
    }
    return report, details


def extract_decisions(inference_rows: list[dict], prior: str) -> list[dict]:
    """Turn a prior conclusion field into scorer decisions (contract section 8)."""
    out = []
    for row in inference_rows:
        cid = row["canonical_id"]
        if prior == "yolo":
            info = row.get("prior_yolo") or {}
            raw = info.get("raw_decision") or info.get("decision")
            if raw is None:
                continue
            decision = _YOLO_MAP.get(str(raw))
            out.append(
                {
                    "canonical_id": cid,
                    "execution_status": "ok" if decision else "protocol_error",
                    "decision": decision,
                    "model": "yolo_prior",
                    "raw_value": raw,
                }
            )
        elif prior == "gpt":
            info = row.get("prior_gpt") or {}
            passes = info.get("passes") or []
            if not passes:
                continue
            latest = passes[-1]
            label = latest.get("label")
            decision = str(label).lower() if label is not None else None
            ok = latest.get("status") == "succeeded" and decision in DECISIONS
            out.append(
                {
                    "canonical_id": cid,
                    "execution_status": "ok" if ok else "technical_failure" if latest.get("status") != "succeeded" else "protocol_error",
                    "decision": decision if ok else None,
                    "model": latest.get("model") or "gpt_prior",
                    "run_id": latest.get("run_id"),
                }
            )
        elif prior == "qwen":
            info = row.get("prior_qwen") or {}
            attempts = info.get("attempts") or []
            if not attempts:
                continue
            latest = attempts[-1]
            decision = _QWEN_MAP.get(str(latest.get("decision")))
            ok = latest.get("status") == "succeeded" and decision is not None
            out.append(
                {
                    "canonical_id": cid,
                    "execution_status": "ok" if ok else "technical_failure",
                    "decision": decision if ok else None,
                    "model": latest.get("model") or "qwen_prior",
                    "run_id": latest.get("run_id"),
                }
            )
        else:
            raise ValueError(f"unknown prior: {prior}")
    out.sort(key=lambda r: r["canonical_id"])
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Unified prompt-gold scorer (candidate-box scope).")
    sub = parser.add_subparsers(dest="command", required=True)
    p_score = sub.add_parser("score", help="score decisions against gold")
    p_score.add_argument("--decisions", required=True)
    p_score.add_argument("--gold", required=True)
    p_score.add_argument("--output", default=None, help="report JSON path (default: stdout)")
    p_score.add_argument("--details-out", default=None, help="optional per-record judgments JSONL")
    p_extract = sub.add_parser("extract", help="extract decisions from inference_records priors")
    p_extract.add_argument("--inference", required=True)
    p_extract.add_argument("--prior", required=True, choices=["yolo", "gpt", "qwen"])
    p_extract.add_argument("--output", default=None)
    args = parser.parse_args(argv)

    if args.command == "extract":
        inference_rows, _ = load_jsonl(args.inference)
        decisions = extract_decisions(inference_rows, args.prior)
        text = "\n".join(jdumps(row) for row in decisions) + "\n"
        if args.output:
            with open(args.output, "w", encoding="utf-8") as fh:
                fh.write(text)
        else:
            sys.stdout.write(text)
        return 0

    decision_rows, bad = load_jsonl(args.decisions)
    gold_rows, gold_bad = load_jsonl(args.gold)
    report, details = score(decision_rows, gold_rows, unparseable=bad + gold_bad)
    report["inputs"] = {"decisions": args.decisions, "gold": args.gold}
    text = jdumps(report) + "\n"
    if args.output:
        with open(args.output, "w", encoding="utf-8") as fh:
            fh.write(text)
    else:
        sys.stdout.write(text)
    if args.details_out:
        with open(args.details_out, "w", encoding="utf-8") as fh:
            fh.write("\n".join(jdumps(row) for row in details) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
