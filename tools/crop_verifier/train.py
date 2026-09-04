"""Crop verifier baseline trainer for cd1 crop datasets.

Binary true/false verifier on canonical-box crops with class conditioning.
Training targets: GPT teacher rows (true/false only; uncertain excluded) merged
with human train rows — for a canonical present in both, the HUMAN label wins.
Validation: human val rows only, reported with the same metric names as the
prompt-gold scorer (contract section 8): dangerous_miss_rate (gold true -> false)
and false_accept_rate (gold false -> true), with Wilson 95% CIs.

Determinism: fixed seed, sorted sample order, DataLoader workers=0; metrics and
predictions carry no wall-clock timestamps.

Environments:
  - GPU server: container `yoloe_service:latest` (torch 2.9.1+cu128, torchvision)
  - local smoke: `.venv-crop` with `pip install torch torchvision` (CPU/MPS)

Usage:
    python tools/crop_verifier/train.py \
        --dataset data/prompt-eval/crop-datasets/cd1-693db6804f05 \
        --output data/crop-verifier-runs/run-local-001 \
        --epochs 5 --batch-size 32 --device auto
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.prompt_eval_data.scorer import wilson_interval  # noqa: E402

NUM_CLASSES = 11
LABEL_TO_INT = {"true": 1, "false": 0}


def _load_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def build_samples(dataset: Path) -> tuple[list[dict], list[dict]]:
    """(train_samples, val_samples) with fields: crop_path, class_id, y, source."""
    human = _load_jsonl(dataset / "labels_human.jsonl")
    teacher = _load_jsonl(dataset / "labels_teacher_gpt.jsonl")

    train: dict[str, dict] = {}
    for row in teacher:  # teacher first; human overrides below
        y = LABEL_TO_INT.get(row["teacher_label"])
        if y is None:  # uncertain teacher rows are excluded from the binary objective
            continue
        train[row["canonical_id"]] = {
            "crop_path": row["crop_path"], "class_id": row["class_id"], "y": y, "source": "teacher_gpt",
        }
    val: list[dict] = []
    for row in human:
        sample = {
            "crop_path": row["crop_path"], "class_id": row["class_id"], "y": LABEL_TO_INT[row["label"]],
        }
        if row["split"] == "train":
            train[row["canonical_id"]] = {**sample, "source": "human"}
        elif row["split"] == "val":
            val.append({**sample, "source": "human", "canonical_id": row["canonical_id"]})
    train_list = [dict(v, canonical_id=cid) for cid, v in sorted(train.items())]
    val.sort(key=lambda r: r["canonical_id"])
    return train_list, val


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


def confusion_metrics(preds: list[dict], samples: list[dict], threshold: float = 0.5) -> dict:
    buckets = {"correct_accept": 0, "dangerous_miss": 0, "correct_reject": 0, "false_accept": 0}
    for p in preds:
        pred_true = p["prob_true"] >= threshold
        if p["y"] == 1:
            buckets["correct_accept" if pred_true else "dangerous_miss"] += 1
        else:
            buckets["false_accept" if pred_true else "correct_reject"] += 1
    gold_true = sum(1 for s in samples if s["y"] == 1)
    gold_false = len(samples) - gold_true

    def rate(k: int, n: int) -> dict:
        return {"value": round(k / n, 6) if n else None, "ci95": wilson_interval(k, n), "k": k, "n": n}

    return {
        "threshold": threshold,
        **buckets,
        "dangerous_miss_rate": rate(buckets["dangerous_miss"], gold_true),
        "false_accept_rate": rate(buckets["false_accept"], gold_false),
        "accuracy": rate(buckets["correct_accept"] + buckets["correct_reject"], len(samples)),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Crop verifier baseline trainer (cd1 datasets).")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto", help="auto|cpu|mps|cuda")
    parser.add_argument("--no-pretrained", action="store_true")
    parser.add_argument("--max-train-batches", type=int, default=0, help="smoke mode: cap batches per epoch")
    args = parser.parse_args(argv)

    import torch
    from torch.utils.data import DataLoader

    torch.manual_seed(args.seed)
    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    else:
        device = args.device

    dataset = Path(args.dataset)
    train_samples, val_samples = build_samples(dataset)
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
        history.append(row)
        print(json.dumps(row, ensure_ascii=False), flush=True)

    _, final_preds = evaluate(net, val_loader, device)
    predictions = [
        {**p, **{k: val_samples[p["idx"]][k] for k in ("canonical_id", "class_id", "crop_path")}}
        for p in sorted(final_preds, key=lambda r: r["idx"])
    ]
    (out_dir / "predictions_val.jsonl").write_text(
        "".join(json.dumps(p, ensure_ascii=False, sort_keys=True) + "\n" for p in predictions), encoding="utf-8"
    )
    torch.save(net.state_dict(), out_dir / "model.pt")
    summary = {
        "dataset": str(dataset),
        "train_samples": len(train_samples),
        "train_human": sum(1 for s in train_samples if s["source"] == "human"),
        "train_teacher": sum(1 for s in train_samples if s["source"] == "teacher_gpt"),
        "val_samples": len(val_samples),
        "device": device,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "seed": args.seed,
        "pretrained": not args.no_pretrained,
        "history": history,
        "final_val": history[-1]["val"] if history else None,
    }
    (out_dir / "metrics.json").write_text(json.dumps(summary, ensure_ascii=False, sort_keys=True, indent=1) + "\n", encoding="utf-8")
    print(json.dumps({"final_val": summary["final_val"], "output": str(out_dir)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
