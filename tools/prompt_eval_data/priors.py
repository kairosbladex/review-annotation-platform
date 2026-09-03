"""Attach prior GPT/Qwen conclusions from external artifact JSONL files.

Priors are keyed by (object_key, class_name, bbox rounded to 6 decimals) — the
same join validated against ktd-gpt-recheck-20260902, ktd-uncertain-recheck-20260903
and ktd-qwen-shadow-20260903. They never feed gold labels (contract section 4).
"""

from __future__ import annotations

import json
from typing import Any, Iterable

PriorKey = tuple[str, str, float, float, float, float]


def prior_key(object_key: str, class_name: str, bbox_norm: Iterable[float]) -> PriorKey:
    rounded = tuple(round(float(v), 6) for v in bbox_norm)
    return (object_key, class_name, *rounded)  # type: ignore[return-value]


def load_gpt_priors(paths: list[str]) -> dict[PriorKey, list[dict[str, Any]]]:
    """One entry per recheck pass, keyed by candidate box. Sorted by run_id."""
    index: dict[PriorKey, list[dict[str, Any]]] = {}
    for path in paths:
        with open(path, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                decision = row.get("decision")
                label = None
                if isinstance(decision, dict):
                    label = decision.get("consensus")
                if label is None:
                    label = row.get("auto_label")
                entry = {
                    "run_id": row.get("run_id"),
                    "run_stage": row.get("run_stage") or "first_pass",
                    "label": label,
                    "status": row.get("status"),
                    "provider": row.get("provider"),
                    "model": row.get("model"),
                }
                key = prior_key(row["file_path"], row["class_name"], row["bbox"])
                index.setdefault(key, []).append(entry)
    for entries in index.values():
        entries.sort(key=lambda e: (str(e["run_id"]), str(e["run_stage"])))
    return index


def load_qwen_priors(paths: list[str]) -> dict[PriorKey, list[dict[str, Any]]]:
    index: dict[PriorKey, list[dict[str, Any]]] = {}
    for path in paths:
        with open(path, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                entry = {
                    "run_id": row.get("source_run_id"),
                    "model": row.get("model"),
                    "label": row.get("qwen_label"),
                    "decision": row.get("qwen_decision"),
                    "confidence": row.get("qwen_confidence_score"),
                    "status": row.get("status"),
                }
                key = prior_key(row["file_path"], row["class_name"], row["bbox"])
                index.setdefault(key, []).append(entry)
    for entries in index.values():
        entries.sort(key=lambda e: (str(e["run_id"]), str(e["model"])))
    return index
