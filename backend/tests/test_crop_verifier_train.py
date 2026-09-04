"""Tests for tools/crop_verifier/train.py data plumbing (no torch required for these).

Run: .venv-crop/bin/python -m pytest backend/tests/test_crop_verifier_train.py -q
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.crop_verifier.train import build_samples, confusion_metrics  # noqa: E402


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")


def _dataset(tmp_path: Path) -> Path:
    human = [
        {"canonical_id": "c1", "class_id": 8, "crop_path": "crops/train/c1.jpg", "label": "true", "split": "train"},
        {"canonical_id": "c2", "class_id": 3, "crop_path": "crops/val/c2.jpg", "label": "false", "split": "val"},
        {"canonical_id": "c3", "class_id": 9, "crop_path": "crops/val/c3.jpg", "label": "true", "split": "val"},
    ]
    teacher = [
        {"canonical_id": "c1", "class_id": 8, "crop_path": "crops/train/c1.jpg", "teacher_label": "false"},  # human overrides
        {"canonical_id": "c4", "class_id": 5, "crop_path": "crops/train/c4.jpg", "teacher_label": "true"},
        {"canonical_id": "c5", "class_id": 1, "crop_path": "crops/train/c5.jpg", "teacher_label": "uncertain"},  # excluded
    ]
    _write_jsonl(tmp_path / "labels_human.jsonl", human)
    _write_jsonl(tmp_path / "labels_teacher_gpt.jsonl", teacher)
    return tmp_path


def test_build_samples_human_overrides_teacher(tmp_path: Path) -> None:
    dataset = _dataset(tmp_path)
    train, val = build_samples(dataset)
    by_cid = {s["canonical_id"]: s for s in train}
    assert by_cid["c1"]["y"] == 1 and by_cid["c1"]["source"] == "human"  # human true wins over teacher false
    assert by_cid["c4"]["y"] == 1 and by_cid["c4"]["source"] == "teacher_gpt"
    assert "c5" not in by_cid  # uncertain teacher excluded
    assert [s["canonical_id"] for s in val] == ["c2", "c3"]  # sorted, val only
    assert all(s["source"] == "human" for s in val)


def test_confusion_metrics_buckets() -> None:
    samples = [{"y": 1}, {"y": 1}, {"y": 0}, {"y": 0}]
    preds = [
        {"idx": 0, "prob_true": 0.9, "y": 1},   # correct accept
        {"idx": 1, "prob_true": 0.2, "y": 1},   # dangerous miss
        {"idx": 2, "prob_true": 0.1, "y": 0},   # correct reject
        {"idx": 3, "prob_true": 0.8, "y": 0},   # false accept
    ]
    m = confusion_metrics(preds, samples)
    assert m["correct_accept"] == 1 and m["dangerous_miss"] == 1
    assert m["correct_reject"] == 1 and m["false_accept"] == 1
    assert m["dangerous_miss_rate"]["value"] == 0.5
    assert m["false_accept_rate"]["value"] == 0.5
    assert m["accuracy"]["value"] == 0.5
    # Wilson CI shape matches the scorer's convention
    assert m["dangerous_miss_rate"]["ci95"][0] <= 0.5 <= m["dangerous_miss_rate"]["ci95"][1]
