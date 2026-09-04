"""Crop verifier baseline trainer for content-addressed crop datasets.

Binary true/false verifier on canonical-box crops with class conditioning.
Training defaults to human gold only. ``--train-label-source teacher-assisted``
explicitly adds GPT teacher rows (true/false only; uncertain excluded); for a
canonical present in both, the HUMAN label wins. ``--crop-view`` selects one
cd2 view per controlled run instead of silently conflating tight and context.
Validation: human val rows only, reported with the same metric names as the
prompt-gold scorer (contract section 8): dangerous_miss_rate (gold true -> false)
and false_accept_rate (gold false -> true), with Wilson 95% CIs.

Reproducibility: fixed seed, sorted sample order, DataLoader workers=0; pass
``--deterministic`` to also enforce PyTorch/cuDNN deterministic controls. The
effective controls are recorded in ``metrics.json``.

Environments:
  - GPU server: container `yoloe_service:latest` (torch 2.9.1+cu128, torchvision)
  - local smoke: `.venv-crop` with `pip install torch torchvision` (CPU/MPS)

Usage:
    python tools/crop_verifier/train.py \
        --dataset data/prompt-eval/crop-datasets/<dataset-id> \
        --output data/crop-verifier-runs/run-local-001 \
        --epochs 5 --batch-size 32 --device auto
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.prompt_eval_data.scorer import wilson_interval  # noqa: E402

NUM_CLASSES = 11
LABEL_TO_INT = {"true": 1, "false": 0}
SIZE_BUCKETS = (
    ("tiny_lt16", 0, 16),
    ("small_16_47", 16, 48),
    ("medium_48_95", 48, 96),
    ("large_ge96", 96, None),
)
CHECKPOINT_RISK_PRIORITY = (
    "min_dangerous_miss_rate",
    "min_false_accept_rate",
    "max_accuracy",
    "earliest_epoch",
)
TRAIN_LABEL_SOURCES = ("human-only", "teacher-assisted")
CROP_VIEWS = ("context", "tight")


def _load_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def _crop_path(row: dict, crop_view: str) -> str:
    if crop_view not in CROP_VIEWS:
        raise ValueError(f"unsupported crop view: {crop_view}")
    view_key = f"{crop_view}_crop_path"
    if view_key in row:
        return row[view_key]
    if crop_view == "context" and "crop_path" in row:
        return row["crop_path"]
    raise ValueError(
        f"{row.get('canonical_id', '<unknown>')}: dataset does not provide {view_key}"
    )


def build_samples(
    dataset: Path,
    *,
    train_label_source: str = "human-only",
    crop_view: str = "context",
) -> tuple[list[dict], list[dict]]:
    """(train_samples, val_samples) with fields: crop_path, class_id, y, source."""
    if train_label_source not in TRAIN_LABEL_SOURCES:
        raise ValueError(f"unsupported train label source: {train_label_source}")
    human = _load_jsonl(dataset / "labels_human.jsonl")
    teacher = (
        _load_jsonl(dataset / "labels_teacher_gpt.jsonl")
        if train_label_source == "teacher-assisted"
        else []
    )

    train: dict[str, dict] = {}
    for row in teacher:  # teacher first; human overrides below
        y = LABEL_TO_INT.get(row["teacher_label"])
        if y is None:  # uncertain teacher rows are excluded from the binary objective
            continue
        train[row["canonical_id"]] = {
            "crop_path": _crop_path(row, crop_view),
            "class_id": row["class_id"], "y": y, "source": "teacher_gpt",
            **{key: row[key] for key in ("class_name", "bbox_px") if key in row},
        }
    val: list[dict] = []
    for row in human:
        sample = {
            "crop_path": _crop_path(row, crop_view),
            "class_id": row["class_id"], "y": LABEL_TO_INT[row["label"]],
            **{key: row[key] for key in ("class_name", "bbox_px") if key in row},
        }
        if row["split"] == "train":
            train[row["canonical_id"]] = {**sample, "source": "human"}
        elif row["split"] == "val":
            val.append({**sample, "source": "human", "canonical_id": row["canonical_id"]})
    train_list = [dict(v, canonical_id=cid) for cid, v in sorted(train.items())]
    val.sort(key=lambda r: r["canonical_id"])
    return train_list, val


def dataset_lineage(dataset: Path) -> dict:
    manifest_path = dataset / "dataset_manifest.json"
    manifest_bytes = manifest_path.read_bytes()
    manifest = json.loads(manifest_bytes)
    return {
        "dataset_id": manifest.get("dataset_id"),
        "schema_version": manifest.get("schema_version"),
        "manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
    }


def make_model(num_classes: int = NUM_CLASSES, pretrained: bool = True):
    from torchvision import models

    weights = models.ResNet18_Weights.DEFAULT if pretrained else None
    net = models.resnet18(weights=weights)
    in_features = net.fc.in_features
    import torch

    net.fc = torch.nn.Sequential(
        torch.nn.Linear(in_features + num_classes, 128),
        torch.nn.ReLU(),
        torch.nn.Linear(128, 2),
    )
    return net


class CropDataset:
    def __init__(self, root: Path, samples: list[dict]):
        self.root = root
        self.samples = samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        import torch
        from PIL import Image

        row = self.samples[idx]
        with Image.open(self.root / row["crop_path"]) as im:
            im = im.convert("RGB").resize((224, 224), Image.Resampling.BILINEAR)
            data = torch.ByteTensor(torch.ByteStorage.from_buffer(im.tobytes()))
            data = data.view(224, 224, 3).permute(2, 0, 1).float().div_(255.0)
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        img = (data - mean) / std
        class_onehot = torch.zeros(NUM_CLASSES)
        class_onehot[int(row["class_id"])] = 1.0
        return img, class_onehot, torch.tensor(row["y"], dtype=torch.float32), idx


def _forward(net, img, class_onehot):
    import torch

    # resnet up to the penultimate features, then the conditioned head
    x = net.conv1(img)
    x = net.bn1(x)
    x = net.relu(x)
    x = net.maxpool(x)
    x = net.layer4(net.layer3(net.layer2(net.layer1(x))))
    x = net.avgpool(x)
    feats = torch.flatten(x, 1)
    return net.fc(torch.cat([feats, class_onehot], dim=1))


def evaluate(net, loader, device) -> tuple[dict, list[dict]]:
    import torch

    net.eval()
    preds: list[dict] = []
    with torch.no_grad():
        for img, class_onehot, y, idx in loader:
            logits = _forward(net, img.to(device), class_onehot.to(device))
            prob = torch.softmax(logits, dim=1)[:, 1].cpu()
            for p, target, i in zip(prob.tolist(), y.tolist(), idx.tolist()):
                preds.append({"idx": i, "prob_true": round(p, 6), "y": int(target)})
    return {}, preds


def _metrics_for_pairs(pairs: list[tuple[dict, dict]], keep_decision) -> dict:
    buckets = {"correct_accept": 0, "dangerous_miss": 0, "correct_reject": 0, "false_accept": 0}
    for p, sample in pairs:
        pred_true = keep_decision(p["prob_true"])
        if sample["y"] == 1:
            buckets["correct_accept" if pred_true else "dangerous_miss"] += 1
        else:
            buckets["false_accept" if pred_true else "correct_reject"] += 1
    gold_true = sum(1 for _, sample in pairs if sample["y"] == 1)
    gold_false = len(pairs) - gold_true

    return {
        "samples": len(pairs),
        "gold_true": gold_true,
        "gold_false": gold_false,
        **buckets,
        "dangerous_miss_rate": _binomial_rate(buckets["dangerous_miss"], gold_true),
        "false_accept_rate": _binomial_rate(buckets["false_accept"], gold_false),
        "accuracy": _binomial_rate(buckets["correct_accept"] + buckets["correct_reject"], len(pairs)),
    }


def _binomial_rate(k: int, n: int) -> dict:
    return {
        "value": round(k / n, 6) if n else None,
        "ci95": wilson_interval(k, n),
        "k": k,
        "n": n,
    }


def _size_bucket(sample: dict) -> str | None:
    bbox = sample.get("bbox_px")
    if not isinstance(bbox, list) or len(bbox) != 4:
        return None
    long_edge = max(max(0, bbox[2] - bbox[0]), max(0, bbox[3] - bbox[1]))
    for name, lower, upper in SIZE_BUCKETS:
        if long_edge >= lower and (upper is None or long_edge < upper):
            return name
    return None


def _confusion_metrics(preds: list[dict], samples: list[dict], keep_decision) -> dict:
    pairs = [(pred, samples[pred["idx"]]) for pred in preds]
    metrics = _metrics_for_pairs(pairs, keep_decision)

    per_class: dict[int, list[tuple[dict, dict]]] = {}
    per_size: dict[str, list[tuple[dict, dict]]] = {}
    for pair in pairs:
        sample = pair[1]
        if "class_id" in sample:
            per_class.setdefault(int(sample["class_id"]), []).append(pair)
        size_bucket = _size_bucket(sample)
        if size_bucket is not None:
            per_size.setdefault(size_bucket, []).append(pair)

    metrics["per_class"] = [
        {
            "class_id": class_id,
            "class_name": class_pairs[0][1].get("class_name", str(class_id)),
            **_metrics_for_pairs(class_pairs, keep_decision),
        }
        for class_id, class_pairs in sorted(per_class.items())
    ]
    metrics["per_size_bucket"] = [
        {
            "size_bucket": name,
            "long_edge_px": {"min_inclusive": lower, "max_exclusive": upper},
            **_metrics_for_pairs(per_size[name], keep_decision),
        }
        for name, lower, upper in SIZE_BUCKETS
        if name in per_size
    ]
    return metrics


def confusion_metrics(preds: list[dict], samples: list[dict], threshold: float = 0.5) -> dict:
    return {
        "threshold": threshold,
        **_confusion_metrics(preds, samples, lambda prob_true: prob_true >= threshold),
    }


def reject_only_metrics(preds: list[dict], samples: list[dict], reject_threshold: float) -> dict:
    """Evaluate REJECT at/below the threshold and KEEP every other candidate."""
    metrics = _confusion_metrics(preds, samples, lambda prob_true: prob_true > reject_threshold)
    rejected = metrics["dangerous_miss"] + metrics["correct_reject"]
    kept = metrics["correct_accept"] + metrics["false_accept"]
    return {
        "threshold": reject_threshold,
        "policy": {
            "mode": "reject_only",
            "reject_if": f"prob_true <= {reject_threshold}",
            "otherwise": "KEEP",
        },
        "rejected": rejected,
        "kept": kept,
        "auto_reject_rate": _binomial_rate(rejected, len(samples)),
        "safe_reject_yield": _binomial_rate(metrics["correct_reject"], len(samples)),
        "correct_reject_rate": _binomial_rate(metrics["correct_reject"], metrics["gold_false"]),
        **metrics,
    }


def _rate_value(metric: dict, *, empty_value: float = 0.0) -> float:
    denominator = int(metric.get("n", 0))
    if denominator:
        return int(metric["k"]) / denominator
    value = metric.get("value")
    return empty_value if value is None else float(value)


def risk_priority_key(metrics: dict) -> tuple[float, float, float]:
    return (
        _rate_value(metrics["dangerous_miss_rate"]),
        _rate_value(metrics["false_accept_rate"]),
        -_rate_value(metrics["accuracy"]),
    )


@dataclass
class RiskCheckpointSelector:
    patience: int = 0
    best_epoch: int | None = None
    best_metrics: dict | None = None
    stale_epochs: int = 0

    def __post_init__(self) -> None:
        if self.patience < 0:
            raise ValueError("early-stopping patience must be >= 0")

    def observe(self, epoch: int, metrics: dict) -> bool:
        improved = self.best_metrics is None or risk_priority_key(metrics) < risk_priority_key(self.best_metrics)
        if improved:
            self.best_epoch = epoch
            self.best_metrics = metrics
            self.stale_epochs = 0
        else:
            self.stale_epochs += 1
        return improved

    @property
    def should_stop(self) -> bool:
        return self.patience > 0 and self.stale_epochs >= self.patience


def configure_reproducibility(torch, *, seed: int, deterministic: bool) -> dict:
    if deterministic:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    return {
        "seed": seed,
        "requested": deterministic,
        "torch_deterministic_algorithms": bool(torch.are_deterministic_algorithms_enabled()),
        "cudnn_deterministic": bool(torch.backends.cudnn.deterministic),
        "cudnn_benchmark": bool(torch.backends.cudnn.benchmark),
        "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Crop verifier baseline trainer (content-addressed datasets).")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto", help="auto|cpu|mps|cuda")
    parser.add_argument("--no-pretrained", action="store_true")
    parser.add_argument("--max-train-batches", type=int, default=0, help="smoke mode: cap batches per epoch")
    parser.add_argument(
        "--train-label-source", choices=TRAIN_LABEL_SOURCES, default="human-only",
        help="human-only by default; teacher-assisted explicitly adds GPT pseudo-labels",
    )
    parser.add_argument(
        "--crop-view", choices=CROP_VIEWS, default="context",
        help="single cd2 image view used by this run (cd1 supports context only)",
    )
    parser.add_argument(
        "--early-stopping-patience", type=int, default=0,
        help="stop after N epochs without risk-priority improvement; 0 disables",
    )
    parser.add_argument(
        "--deterministic", action="store_true",
        help="enable deterministic PyTorch/cuDNN controls (may reduce speed or reject unsupported operations)",
    )
    parser.add_argument(
        "--reject-threshold", type=float, default=None,
        help="optional policy evaluation: REJECT when p_true <= threshold, otherwise KEEP",
    )
    args = parser.parse_args(argv)

    if args.epochs < 1:
        parser.error("--epochs must be >= 1")
    if args.batch_size < 1:
        parser.error("--batch-size must be >= 1")
    if args.early_stopping_patience < 0:
        parser.error("--early-stopping-patience must be >= 0")
    if args.reject_threshold is not None and not 0.0 <= args.reject_threshold <= 1.0:
        parser.error("--reject-threshold must be in [0, 1]")

    import torch
    from torch.utils.data import DataLoader

    determinism = configure_reproducibility(torch, seed=args.seed, deterministic=args.deterministic)
    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    else:
        device = args.device

    dataset = Path(args.dataset)
    try:
        train_samples, val_samples = build_samples(
            dataset,
            train_label_source=args.train_label_source,
            crop_view=args.crop_view,
        )
        lineage = dataset_lineage(dataset)
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
        parser.error(str(exc))
    if not train_samples:
        parser.error("selected training label source produced no train samples")
    if not val_samples:
        parser.error("dataset contains no human validation samples")
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    net = make_model(pretrained=not args.no_pretrained).to(device)
    train_loader = DataLoader(
        CropDataset(dataset, train_samples), batch_size=args.batch_size, shuffle=True,
        num_workers=0, generator=torch.Generator().manual_seed(args.seed),
    )
    val_loader = DataLoader(CropDataset(dataset, val_samples), batch_size=args.batch_size, shuffle=False, num_workers=0)

    n_pos = sum(s["y"] for s in train_samples)
    n_neg = len(train_samples) - n_pos
    pos_weight = torch.tensor([n_neg / max(1, n_pos)], device=device)
    criterion = torch.nn.CrossEntropyLoss(
        weight=torch.tensor([1.0, float(pos_weight[0])], device=device)
    )
    optimizer = torch.optim.AdamW(net.parameters(), lr=args.lr)

    history = []
    selector = RiskCheckpointSelector(patience=args.early_stopping_patience)
    best_row = None
    best_preds = None
    stopped_early = False
    checkpoint_path = out_dir / "model.pt"
    checkpoint_tmp = out_dir / "model.pt.tmp"
    for epoch in range(args.epochs):
        net.train()
        total_loss, batches = 0.0, 0
        for img, class_onehot, y, _idx in train_loader:
            logits = _forward(net, img.to(device), class_onehot.to(device))
            loss = criterion(logits, y.long().to(device))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += float(loss)
            batches += 1
            if args.max_train_batches and batches >= args.max_train_batches:
                break
        _, preds = evaluate(net, val_loader, device)
        metrics = confusion_metrics(preds, val_samples)
        row = {
            "epoch": epoch + 1,
            "train_loss": round(total_loss / max(1, batches), 6),
            "val": metrics,
        }
        selection_metrics = metrics
        if args.reject_threshold is not None:
            row["reject_only_val"] = reject_only_metrics(preds, val_samples, args.reject_threshold)
            selection_metrics = row["reject_only_val"]
        improved = selector.observe(epoch + 1, selection_metrics)
        row["checkpoint_improved"] = improved
        if improved:
            best_row = row
            best_preds = list(preds)
            torch.save(net.state_dict(), checkpoint_tmp)
            checkpoint_tmp.replace(checkpoint_path)
        history.append(row)
        print(json.dumps(row, ensure_ascii=False), flush=True)
        if selector.should_stop:
            stopped_early = True
            break

    assert best_row is not None and best_preds is not None and selector.best_epoch is not None
    predictions = [
        {**p, **{k: val_samples[p["idx"]][k] for k in ("canonical_id", "class_id", "crop_path")}}
        for p in sorted(best_preds, key=lambda r: r["idx"])
    ]
    if args.reject_threshold is not None:
        for prediction in predictions:
            prediction["action"] = "REJECT" if prediction["prob_true"] <= args.reject_threshold else "KEEP"
    (out_dir / "predictions_val.jsonl").write_text(
        "".join(json.dumps(p, ensure_ascii=False, sort_keys=True) + "\n" for p in predictions), encoding="utf-8"
    )
    summary = {
        "dataset": str(dataset),
        "dataset_lineage": lineage,
        "train_label_source": args.train_label_source,
        "crop_view": args.crop_view,
        "train_samples": len(train_samples),
        "train_human": sum(1 for s in train_samples if s["source"] == "human"),
        "train_teacher": sum(1 for s in train_samples if s["source"] == "teacher_gpt"),
        "val_samples": len(val_samples),
        "device": device,
        "epochs": args.epochs,
        "epochs_ran": len(history),
        "batch_size": args.batch_size,
        "lr": args.lr,
        "seed": args.seed,
        "pretrained": not args.no_pretrained,
        "determinism": determinism,
        "runtime": {
            "torch_version": str(torch.__version__),
            "torch_cuda_version": torch.version.cuda,
            "cudnn_version": torch.backends.cudnn.version(),
        },
        "checkpoint_selection": {
            "strategy": "lexicographic_risk_priority",
            "priority": list(CHECKPOINT_RISK_PRIORITY),
            "metrics_source": "reject_only_val" if args.reject_threshold is not None else "val",
            "reject_threshold": args.reject_threshold,
        },
        "checkpoint": {
            "path": "model.pt",
            "epoch": selector.best_epoch,
            "final_val_semantics": "saved_best_checkpoint",
        },
        "best_epoch": selector.best_epoch,
        "early_stopping_patience": args.early_stopping_patience,
        "stopped_early": stopped_early,
        "history": history,
        "best_val": best_row["val"],
        "best_selection_val": selector.best_metrics,
        "last_epoch_val": history[-1]["val"],
        "final_val": best_row["val"],
    }
    if args.reject_threshold is not None:
        summary["reject_only_val"] = best_row["reject_only_val"]
    (out_dir / "metrics.json").write_text(json.dumps(summary, ensure_ascii=False, sort_keys=True, indent=1) + "\n", encoding="utf-8")
    print(json.dumps(
        {"best_epoch": selector.best_epoch, "final_val": summary["final_val"], "output": str(out_dir)},
        ensure_ascii=False,
    ))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
