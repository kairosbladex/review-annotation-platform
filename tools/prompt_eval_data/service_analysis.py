"""Failure-mode analysis for deployed review-service attempts.

Aggregates one prefill/service ``attempts.jsonl`` into a quantified failure-mode
report (JSON): field distributions, templated-reasoning buckets, per-class score
breakdown passthrough, and latency stats. Pure stdlib, deterministic.

Usage:
    python3 -m tools.prompt_eval_data.service_analysis \
        --attempts attempts.jsonl --score 8188-final.score-gold.json \
        --output analysis.json
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter

from .common import jdumps

ANALYSIS_SCHEMA_VERSION = "prompt-gold-service-analysis/1"

_DIGITS = re.compile(r"\d+(?:\.\d+)?")
_QUOTED = re.compile(r"'[^']*'")
# Bracketed verifier tag, e.g. "[YOLOE复核]".
_TAG = re.compile(r"^\[[^\]]*\]")


def reasoning_template(text: str | None) -> str:
    """Collapse a templated reasoning string to its rule template (digits/class names out)."""
    if not text:
        return "(empty)"
    tpl = _DIGITS.sub("N", str(text))
    tpl = _QUOTED.sub("'X'", tpl)
    tpl = _TAG.sub("[TAG]", tpl)
    return tpl.strip()


def _percentile(sorted_vals: list[float], q: float) -> float | None:
    if not sorted_vals:
        return None
    idx = min(len(sorted_vals) - 1, max(0, int(round(q * (len(sorted_vals) - 1)))))
    return round(sorted_vals[idx], 1)


def analyze_attempts(attempts_path: str) -> dict:
    fields: dict[str, Counter] = {
        "decision": Counter(),
        "yolo_decision": Counter(),
        "llm_decision": Counter(),
        "businessAction": Counter(),
        "reasonCode": Counter(),
        "errorCode": Counter(),
    }
    reasoning_by_decision: dict[str, Counter] = {}
    durations: list[float] = []
    status_counts: Counter = Counter()
    images = 0
    candidates = 0

    with open(attempts_path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            images += 1
            status_counts[str(rec.get("status"))] += 1
            if rec.get("duration_ms") is not None:
                durations.append(float(rec["duration_ms"]))
            for result in (rec.get("response") or {}).get("results") or []:
                candidates += 1
                for key in fields:
                    fields[key][str(result.get(key))] += 1
                yolo = str(result.get("yolo_decision"))
                bucket = reasoning_by_decision.setdefault(yolo, Counter())
                bucket[reasoning_template(result.get("reasoning"))] += 1

    durations.sort()
    return {
        "images": images,
        "candidates": candidates,
        "attempt_status": dict(status_counts),
        "fields": {k: dict(v.most_common()) for k, v in fields.items()},
        "reasoning_templates": {
            k: [{"template": t, "count": c} for t, c in v.most_common(15)]
            for k, v in sorted(reasoning_by_decision.items())
        },
        "latency_ms": {
            "p50": _percentile(durations, 0.50),
            "p95": _percentile(durations, 0.95),
            "max": durations[-1] if durations else None,
        },
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Failure-mode analysis of deployed-service attempts.")
    parser.add_argument("--attempts", required=True)
    parser.add_argument("--score", default=None, help="optional scorer report JSON to embed per-class table")
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)

    report = {
        "schema_version": ANALYSIS_SCHEMA_VERSION,
        "attempts_path": args.attempts,
        **analyze_attempts(args.attempts),
    }
    if args.score:
        with open(args.score, encoding="utf-8") as fh:
            score_report = json.load(fh)
        report["score_scope"] = score_report.get("scope")
        report["score_totals"] = score_report.get("totals")
        report["score_rates"] = score_report.get("rates")
        report["per_class"] = score_report.get("per_class")
    with open(args.output, "w", encoding="utf-8") as fh:
        fh.write(jdumps(report) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
