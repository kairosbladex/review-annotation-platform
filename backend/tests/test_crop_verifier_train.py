"""Tests for tools/crop_verifier/train.py data plumbing (no torch required for these).

Run: .venv-crop/bin/python -m pytest backend/tests/test_crop_verifier_train.py -q
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.crop_verifier.train import (  # noqa: E402
    RiskCheckpointSelector,
    build_samples,
    configure_reproducibility,
    confusion_metrics,
    dataset_lineage,
    reject_only_metrics,
)


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")


def _dataset(tmp_path: Path) -> Path:
    human = [
        {
            "canonical_id": "c1", "class_id": 8,
            "crop_path": "crops/train/c1.context.jpg",
            "context_crop_path": "crops/train/c1.context.jpg",
            "tight_crop_path": "crops/train/c1.tight.jpg",
            "label": "true", "split": "train",
        },
        {
            "canonical_id": "c2", "class_id": 3, "class_name": "光伏板", "bbox_px": [1, 2, 11, 14],
            "crop_path": "crops/val/c2.context.jpg",
            "context_crop_path": "crops/val/c2.context.jpg",
            "tight_crop_path": "crops/val/c2.tight.jpg",
            "label": "false", "split": "val",
        },
        {
            "canonical_id": "c3", "class_id": 9, "class_name": "违建", "bbox_px": [3, 4, 53, 64],
            "crop_path": "crops/val/c3.context.jpg",
            "context_crop_path": "crops/val/c3.context.jpg",
            "tight_crop_path": "crops/val/c3.tight.jpg",
            "label": "true", "split": "val",
        },
    ]
    teacher = [
        {
            "canonical_id": "c1", "class_id": 8,
            "crop_path": "crops/train/c1.context.jpg",
            "context_crop_path": "crops/train/c1.context.jpg",
            "tight_crop_path": "crops/train/c1.tight.jpg",
            "teacher_label": "false",
        },  # human overrides
        {
            "canonical_id": "c4", "class_id": 5,
            "crop_path": "crops/train/c4.context.jpg",
            "context_crop_path": "crops/train/c4.context.jpg",
            "tight_crop_path": "crops/train/c4.tight.jpg",
            "teacher_label": "true",
        },
        {
            "canonical_id": "c5", "class_id": 1,
            "crop_path": "crops/train/c5.context.jpg",
            "context_crop_path": "crops/train/c5.context.jpg",
            "tight_crop_path": "crops/train/c5.tight.jpg",
            "teacher_label": "uncertain",
        },  # excluded
    ]
    _write_jsonl(tmp_path / "labels_human.jsonl", human)
    _write_jsonl(tmp_path / "labels_teacher_gpt.jsonl", teacher)
    return tmp_path


def test_build_samples_defaults_to_human_only(tmp_path: Path) -> None:
    dataset = _dataset(tmp_path)
    train, val = build_samples(dataset)
    by_cid = {s["canonical_id"]: s for s in train}
    assert set(by_cid) == {"c1"}
    assert by_cid["c1"]["y"] == 1 and by_cid["c1"]["source"] == "human"
    assert [s["canonical_id"] for s in val] == ["c2", "c3"]
    assert all(s["source"] == "human" for s in val)


def test_build_samples_teacher_assisted_is_explicit_and_human_wins(tmp_path: Path) -> None:
    dataset = _dataset(tmp_path)
    train, val = build_samples(dataset, train_label_source="teacher-assisted")
    by_cid = {s["canonical_id"]: s for s in train}
    assert by_cid["c1"]["y"] == 1 and by_cid["c1"]["source"] == "human"
    assert by_cid["c4"]["y"] == 1 and by_cid["c4"]["source"] == "teacher_gpt"
    assert "c5" not in by_cid  # uncertain teacher excluded
    assert [s["canonical_id"] for s in val] == ["c2", "c3"]  # sorted, val only
    assert all(s["source"] == "human" for s in val)
    assert val[0]["class_name"] == "光伏板"
    assert val[0]["bbox_px"] == [1, 2, 11, 14]


def test_build_samples_human_only_does_not_read_teacher_file(tmp_path: Path) -> None:
    dataset = _dataset(tmp_path)
    (dataset / "labels_teacher_gpt.jsonl").write_text("not-json\n", encoding="utf-8")
    train, val = build_samples(dataset)
    assert [s["canonical_id"] for s in train] == ["c1"]
    assert [s["canonical_id"] for s in val] == ["c2", "c3"]


def test_build_samples_selects_one_explicit_crop_view(tmp_path: Path) -> None:
    dataset = _dataset(tmp_path)
    context_train, context_val = build_samples(dataset, crop_view="context")
    tight_train, tight_val = build_samples(dataset, crop_view="tight")
    assert context_train[0]["crop_path"].endswith(".context.jpg")
    assert context_val[0]["crop_path"].endswith(".context.jpg")
    assert tight_train[0]["crop_path"].endswith(".tight.jpg")
    assert tight_val[0]["crop_path"].endswith(".tight.jpg")


def test_dataset_lineage_hashes_manifest_bytes(tmp_path: Path) -> None:
    manifest_bytes = b'{"dataset_id":"cd2-fixture","schema_version":"prompt-gold-crop-dataset/2"}\n'
    (tmp_path / "dataset_manifest.json").write_bytes(manifest_bytes)
    assert dataset_lineage(tmp_path) == {
        "dataset_id": "cd2-fixture",
        "schema_version": "prompt-gold-crop-dataset/2",
        "manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
    }


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


def test_reject_only_metrics_rejects_at_threshold_and_keeps_everything_else() -> None:
    samples = [{"y": 1}, {"y": 0}, {"y": 0}]
    preds = [
        {"idx": 0, "prob_true": 0.1, "y": 1},
        {"idx": 1, "prob_true": 0.100001, "y": 0},
        {"idx": 2, "prob_true": 0.9, "y": 0},
    ]

    metrics = reject_only_metrics(preds, samples, reject_threshold=0.1)

    assert metrics["policy"] == {
        "mode": "reject_only",
        "reject_if": "prob_true <= 0.1",
        "otherwise": "KEEP",
    }
    assert metrics["rejected"] == 1
    assert metrics["kept"] == 2
    assert metrics["dangerous_miss"] == 1
    assert metrics["false_accept"] == 2
    assert metrics["auto_reject_rate"]["value"] == pytest.approx(1 / 3, abs=1e-6)
    assert metrics["safe_reject_yield"]["value"] == 0.0
    assert metrics["correct_reject_rate"]["value"] == 0.0


def test_confusion_metrics_reports_per_class_and_original_box_size() -> None:
    samples = [
        {"y": 1, "class_id": 8, "class_name": "车", "bbox_px": [0, 0, 10, 8]},
        {"y": 1, "class_id": 8, "class_name": "车", "bbox_px": [0, 0, 30, 20]},
        {"y": 0, "class_id": 3, "class_name": "光伏板", "bbox_px": [0, 0, 48, 20]},
        {"y": 0, "class_id": 3, "class_name": "光伏板", "bbox_px": [0, 0, 120, 20]},
    ]
    preds = [
        {"idx": 0, "prob_true": 0.9, "y": 1},
        {"idx": 1, "prob_true": 0.1, "y": 1},
        {"idx": 2, "prob_true": 0.1, "y": 0},
        {"idx": 3, "prob_true": 0.9, "y": 0},
    ]

    metrics = confusion_metrics(preds, samples)

    by_class = {row["class_id"]: row for row in metrics["per_class"]}
    assert by_class[8]["class_name"] == "车"
    assert by_class[8]["gold_true"] == 2
    assert by_class[8]["correct_accept"] == 1
    assert by_class[8]["dangerous_miss"] == 1
    assert by_class[3]["gold_false"] == 2
    assert by_class[3]["correct_reject"] == 1
    assert by_class[3]["false_accept"] == 1

    by_size = {row["size_bucket"]: row for row in metrics["per_size_bucket"]}
    assert list(by_size) == ["tiny_lt16", "small_16_47", "medium_48_95", "large_ge96"]
    assert by_size["tiny_lt16"]["long_edge_px"] == {"min_inclusive": 0, "max_exclusive": 16}
    assert by_size["large_ge96"]["long_edge_px"] == {"min_inclusive": 96, "max_exclusive": None}
    assert by_size["tiny_lt16"]["correct_accept"] == 1
    assert by_size["small_16_47"]["dangerous_miss"] == 1
    assert by_size["medium_48_95"]["correct_reject"] == 1
    assert by_size["large_ge96"]["false_accept"] == 1


def _risk_metrics(dangerous_miss: int, false_accept: int, correct: int) -> dict:
    return {
        "dangerous_miss_rate": {"k": dangerous_miss, "n": 10, "value": dangerous_miss / 10},
        "false_accept_rate": {"k": false_accept, "n": 10, "value": false_accept / 10},
        "accuracy": {"k": correct, "n": 20, "value": correct / 20},
    }


def test_risk_checkpoint_selector_prioritizes_misses_and_supports_early_stopping() -> None:
    selector = RiskCheckpointSelector(patience=2)

    assert selector.observe(1, _risk_metrics(dangerous_miss=2, false_accept=2, correct=16)) is True
    # One fewer dangerous miss wins even with many more false accepts.
    assert selector.observe(2, _risk_metrics(dangerous_miss=1, false_accept=9, correct=10)) is True
    # With dangerous misses tied, fewer false accepts wins.
    assert selector.observe(3, _risk_metrics(dangerous_miss=1, false_accept=4, correct=15)) is True
    assert selector.best_epoch == 3
    assert selector.stale_epochs == 0

    assert selector.observe(4, _risk_metrics(dangerous_miss=1, false_accept=4, correct=15)) is False
    assert selector.should_stop is False
    assert selector.observe(5, _risk_metrics(dangerous_miss=2, false_accept=0, correct=18)) is False
    assert selector.should_stop is True

    disabled = RiskCheckpointSelector(patience=0)
    disabled.observe(1, _risk_metrics(dangerous_miss=0, false_accept=0, correct=20))
    disabled.observe(2, _risk_metrics(dangerous_miss=1, false_accept=1, correct=18))
    assert disabled.should_stop is False


def test_configure_reproducibility_enables_and_records_cuda_controls(monkeypatch) -> None:
    calls: list[tuple[str, object]] = []
    deterministic_algorithms = {"enabled": False}
    cuda = SimpleNamespace(
        is_available=lambda: True,
        manual_seed_all=lambda seed: calls.append(("cuda_seed", seed)),
    )
    cudnn = SimpleNamespace(deterministic=False, benchmark=True)
    fake_torch = SimpleNamespace(
        cuda=cuda,
        backends=SimpleNamespace(cudnn=cudnn),
        manual_seed=lambda seed: calls.append(("torch_seed", seed)),
        use_deterministic_algorithms=lambda enabled: deterministic_algorithms.update(enabled=enabled),
        are_deterministic_algorithms_enabled=lambda: deterministic_algorithms["enabled"],
    )
    monkeypatch.delenv("CUBLAS_WORKSPACE_CONFIG", raising=False)

    record = configure_reproducibility(fake_torch, seed=123, deterministic=True)

    assert calls == [("torch_seed", 123), ("cuda_seed", 123)]
    assert cudnn.deterministic is True
    assert cudnn.benchmark is False
    assert record == {
        "seed": 123,
        "requested": True,
        "torch_deterministic_algorithms": True,
        "cudnn_deterministic": True,
        "cudnn_benchmark": False,
        "cublas_workspace_config": ":4096:8",
    }
