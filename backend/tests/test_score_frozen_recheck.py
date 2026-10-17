"""Exercise the offline scorer's CLI with synthetic files, without model calls."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import os
import subprocess
import sys

import pytest


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts" / "score_frozen_recheck.py"


def frozen_files(tmp_path):
    # Independent oracle: 9 scorable boxes, 5 positives and 4 negatives.
    specs = {
        "event-one": [
            ("a", "人员", "真目标"), ("b", "人员", "误检"),
            ("c", "人员", "真目标"), ("d", "车", "非车"),
            ("e", "人员", "真目标"), ("f", "车", "看不清"),
            ("g", "车", "非车"), ("h", "车", "真车"),
        ],
        "event-failed": [("i", "人员", "误检")],
        "event-absent": [("j", "人员", "真目标")],
    }
    source, labels = [], []
    for event, entries in specs.items():
        candidates = []
        for index, (cid, cls, gold) in enumerate(entries):
            bbox = [0.1, 0.2, 0.3, 0.4]
            candidates.append(dict(index=index, candidate_id=cid, class_name=cls, bbox=bbox))
            labels.append(dict(event_id=event, candidate_id=cid, class_name=cls, bbox=bbox,
                               gold=gold, scorable=gold != "看不清",
                               group="person" if cls == "人员" else "vehicle"))
        source.append(dict(event_id=event, image="private-image-reference", candidates=candidates))
    # Deliberately out of result order. e is absent; f has no evaluable truth.
    results = [dict(target_index=i, decision=d) for i, d in
               [(7, "false"), (2, "uncertain"), (1, "true"), (6, "uncertain"),
                (0, "true"), (3, "false"), (5, "true")]]
    predictions = [
        dict(event_id="event-one", candidate_ids=list("abcdefgh"), parsed={"results": results}),
        dict(event_id="event-failed", candidate_ids=["i"], error="private-error-detail",
             parsed={"results": [dict(target_index=0, decision="false")]}),
    ]
    paths = {name: tmp_path / name for name in ("tasks.json", "labels.json", "predictions.jsonl")}
    paths["tasks.json"].write_text(json.dumps(source))
    paths["labels.json"].write_text(json.dumps(labels))
    paths["predictions.jsonl"].write_text("\n".join(json.dumps(row) for row in predictions))
    return paths, source, labels, predictions


def invoke(paths):
    return subprocess.run(
        [sys.executable, str(SCRIPT), "--source", str(paths["tasks.json"]),
         "--labels", str(paths["labels.json"]), "--predictions", str(paths["predictions.jsonl"])],
        cwd=ROOT, capture_output=True, text=True, timeout=10,
        env={"PATH": os.defpath, "PYTHONDONTWRITEBYTECODE": "1"},
    )


def test_fixed_denominators_failure_uncertainty_and_confusion_are_separate(tmp_path):
    paths, _, _, _ = frozen_files(tmp_path)
    result = invoke(paths)
    assert result.returncode == 0, result.stderr
    report = json.loads(result.stdout)
    assert report["dataset"] == {"events": 3, "candidates": 10, "scorable": 9, "excluded_gold": 1}
    online = report["policies"]["online"]["all"]
    manual = report["policies"]["manual"]["all"]
    keys = ["total", "right", "wrong", "manual", "missing", "failed", "raw_uncertain",
            "wrongly_rejected", "wrongly_confirmed", "true_targets", "non_targets"]
    assert [online[k] for k in keys] == [9, 3, 3, 0, 2, 1, 2, 2, 1, 5, 4]
    assert [manual[k] for k in keys] == [9, 2, 2, 2, 2, 1, 2, 1, 1, 5, 4]
    assert online["coverage"] == pytest.approx(6 / 9)
    assert manual["coverage"] == pytest.approx(4 / 9)
    assert manual["response_coverage"] == pytest.approx(6 / 9)
    assert online["accuracy_decided"] == 0.5
    assert online["correct_fraction_total"] == pytest.approx(3 / 9)
    assert online["false_rejection_rate_total"] == 2 / 5
    assert online["false_acceptance_rate_total"] == 1 / 4
    person = report["policies"]["online"]["groups"]["person"]
    vehicle = report["policies"]["manual"]["classes"]["车"]
    assert [person[k] for k in ["total", "right", "wrong", "missing", "failed"]] == [6, 1, 2, 2, 1]
    assert [vehicle[k] for k in ["total", "right", "wrong", "manual"]] == [3, 1, 1, 1]
    assert report["execution"] == {"present_events": 2, "missing_events": 1,
                                    "failed_events": 1, "invalid_response_events": 0}
    for role, name in [("source", "tasks.json"), ("labels", "labels.json"), ("predictions", "predictions.jsonl")]:
        assert report["input_sha256"][role] == hashlib.sha256(paths[name].read_bytes()).hexdigest()
    assert "private-image-reference" not in result.stdout
    assert "private-error-detail" not in result.stdout
    assert "event-one" not in result.stdout


def test_empty_predictions_keep_all_nine_boxes_in_denominator(tmp_path):
    paths, _, _, _ = frozen_files(tmp_path)
    paths["predictions.jsonl"].write_text("")
    result = invoke(paths)
    assert result.returncode == 0, result.stderr
    score = json.loads(result.stdout)["policies"]["online"]["all"]
    assert [score[k] for k in ["total", "missing", "failed", "right", "wrong"]] == [9, 9, 0, 0, 0]
    assert score["coverage"] == 0
    assert score["accuracy_decided"] is None


@pytest.mark.parametrize("bad_result", [
    {"target_index": True, "decision": "true"},
    {"target_index": 8, "decision": "false"},
    {"target_index": -1, "decision": "false"},
    {"target_index": "4", "decision": "true"},
    {"target_index": 4, "decision": "maybe"},
    {"target_index": 0, "decision": "true"},  # Duplicate of the real result.
])
def test_invalid_response_is_failure_without_false_labels(tmp_path, bad_result):
    paths, _, _, predictions = frozen_files(tmp_path)
    predictions[0]["parsed"]["results"].append(bad_result)
    paths["predictions.jsonl"].write_text("\n".join(json.dumps(r) for r in predictions))
    result = invoke(paths)
    assert result.returncode == 0, result.stderr
    report = json.loads(result.stdout)
    score = report["policies"]["online"]["all"]
    assert [score[k] for k in ["total", "right", "wrong", "failed", "missing"]] == [9, 0, 0, 8, 1]
    assert report["execution"]["invalid_response_events"] == 1


@pytest.mark.parametrize("problem", ["missing_label", "duplicate_label", "label_geometry", "candidate_order", "duplicate_event"])
def test_broken_source_binding_is_rejected_without_private_details(tmp_path, problem):
    paths, _, labels, predictions = frozen_files(tmp_path)
    if problem == "missing_label":
        labels.pop()
    elif problem == "duplicate_label":
        labels.append(labels[0])
    elif problem == "label_geometry":
        labels[0]["bbox"] = [0, 0, 0.1, 0.1]
    elif problem == "candidate_order":
        predictions[0]["candidate_ids"].reverse()
    else:
        predictions.append(predictions[0])
    paths["labels.json"].write_text(json.dumps(labels))
    paths["predictions.jsonl"].write_text("\n".join(json.dumps(r) for r in predictions))
    result = invoke(paths)
    assert result.returncode == 2
    assert result.stdout == ""
    assert result.stderr.strip() == "INPUT_CONTRACT_ERROR"
