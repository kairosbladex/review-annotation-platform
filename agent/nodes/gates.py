"""Gate nodes: human approval (LangGraph interrupt) and dataset audit."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from langgraph.types import interrupt

from agent.core.registry import register
from agent.core.state import RunState


@register("approval_gate")
def approval_gate(state: RunState) -> RunState:
    """Pause the run until a human resumes it with a decision.

    CLI:  agent/cli.py resume <pipeline> --run-id <id> \
              --decision '{"approved": true, "decided_by": "...", "note": "..."}'

    Rejection stops the chain (the run fails with the reviewer's note);
    nothing downstream of a rejected gate ever executes.
    """
    params = state["params"]
    gate_name = str(params.get("gate_name", "approval"))
    message = str(params.get("gate_message", "approve to continue"))
    decision = interrupt(
        {
            "gate": gate_name,
            "message": message,
            "run_id": state["run_id"],
            "metrics": state.get("metrics", {}),
        }
    )
    decision = decision or {}
    approved = bool(decision.get("approved"))
    state["gates"][gate_name] = {
        "passed": approved,
        "note": str(decision.get("note", "")),
        "decided_by": str(decision.get("decided_by", "")),
    }
    if not approved:
        raise RuntimeError(
            f"gate {gate_name} rejected by {decision.get('decided_by', '?')}: "
            f"{decision.get('note', '')}"
        )
    return state


def _count_labels_per_class(labels_dir: Path) -> dict[int, int]:
    counts: dict[int, int] = {}
    for label_file in sorted(labels_dir.rglob("*.txt")):
        for line in label_file.read_text(encoding="utf-8").splitlines():
            fields = line.split()
            if fields:
                class_id = int(fields[0])
                counts[class_id] = counts.get(class_id, 0) + 1
    return counts


@register("dataset_audit")
def dataset_audit(state: RunState) -> RunState:
    """Codify the KTD formal-training readiness checks as a gate.

    Checks (docs/KTD_FORMAL_TRAINING_READINESS_20260903.md):
      1. zero-positive classes      (classes.txt vs per-class label counts)
      2. cross-split leakage        (same image sha256 in train and val)
      3. label bounds               (YOLO rows within normalized bounds)
      4. gold manifest non-empty    (params.recheck_dir + require_gold)
      5. task/weight match          (params.confirm_task must be explicit)

    params.on_block: "fail" (default) raises on blockers; "continue" only
    records them in state["gates"]["dataset_audit"].
    """
    params = state["params"]
    dataset_dir = params.get("dataset_dir")
    if dataset_dir:
        root = Path(str(dataset_dir).format(run_id=state["run_id"]))
    else:
        candidates = [a for a in state["artifacts"] if a["kind"] == "dataset"]
        if not candidates:
            raise RuntimeError("dataset_audit: no dataset_dir param and no dataset artifact")
        root = Path(candidates[-1]["uri"][len("file://"):])
    blockers: list[str] = []
    notes: list[str] = []

    classes: dict[int, str] = {}
    classes_file = root / "classes.txt"
    if not classes_file.exists():
        raise RuntimeError(f"dataset_audit: missing {classes_file}")
    for line in classes_file.read_text(encoding="utf-8").splitlines():
        if line.strip():
            class_id, name = line.split("\t", 1)
            classes[int(class_id)] = name

    counts = _count_labels_per_class(root / "labels")
    zero_positive = [name for class_id, name in sorted(classes.items()) if counts.get(class_id, 0) == 0]
    if zero_positive:
        blockers.append(f"zero-positive classes: {zero_positive}")

    split_hashes: dict[str, dict[str, str]] = {}
    for split in ("train", "val"):
        split_hashes[split] = {}
        for image in sorted((root / "images" / split).glob("*")):
            if image.is_file():
                digest = hashlib.sha256(image.read_bytes()).hexdigest()
                split_hashes[split][digest] = image.name
    leaked = set(split_hashes["train"]) & set(split_hashes["val"])
    if leaked:
        blockers.append(
            f"train/val content leakage: {len(leaked)} duplicate image(s) across splits"
        )
    if not any(split_hashes.values()):
        blockers.append("no local images found; content-leakage check is unverifiable")

    try:
        from scripts.prepare_ktd_gpu_dataset import _validate_label_file

        for label_file in sorted((root / "labels").rglob("*.txt")):
            _validate_label_file(label_file, len(classes))
    except Exception as exc:
        blockers.append(f"label validation failed: {exc}")

    if params.get("require_gold"):
        recheck_dir = params.get("recheck_dir")
        gold = Path(str(recheck_dir)) / "training_manifest.jsonl" if recheck_dir else None
        gold_rows = (
            sum(1 for line in gold.read_text(encoding="utf-8").splitlines() if line.strip())
            if gold and gold.exists()
            else 0
        )
        if gold_rows == 0:
            blockers.append("gold training_manifest.jsonl is empty or missing")
        notes.append(f"gold rows: {gold_rows}")

    if not params.get("confirm_task"):
        blockers.append("task/weight match not confirmed (set params.confirm_task)")
    else:
        notes.append(f"confirmed task: {params['confirm_task']}")

    passed = not blockers
    state["gates"]["dataset_audit"] = {
        "passed": passed,
        "blockers": blockers,
        "notes": notes,
        "dataset_dir": str(root),
    }
    if blockers and str(params.get("on_block", "fail")) == "fail":
        raise RuntimeError(f"dataset_audit blockers: {json.dumps(blockers, ensure_ascii=False)}")
    return state
