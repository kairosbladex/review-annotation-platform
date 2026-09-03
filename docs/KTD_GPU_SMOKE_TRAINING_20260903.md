# KTD GPU Smoke Training Report

> Date: 2026-09-03
> Scope: one-epoch training-chain smoke test only

## Result

The one-epoch YOLOE smoke training completed on the company GPU server `AI-lingxi`, using host GPU 4 (NVIDIA L20). The run completed 1/1 epoch and 22/22 training batches, and produced both `best.pt` and `last.pt`.

This is a training-chain verification result. It is not a model-quality, mAP, recall, or production-readiness result.

## Environment

| Item | Value |
|---|---|
| Server | `192.168.130.238` / `AI-lingxi` |
| Container | `yoloe_service:latest` |
| Python | 3.10.12 |
| PyTorch | 2.9.1+cu128 |
| Ultralytics | 8.4.89 |
| GPU | Host GPU 4, NVIDIA L20 |
| Container device | `cuda:0` (mapped from host GPU 4) |

The existing business containers were left running. The smoke container was one-shot and has exited; GPU 4 is released.

## Dataset

Source artifacts came from `data/recheck-runs/ktd-uncertain-recheck-20260903/`:

- 106 source images downloaded and checked;
- deterministic split: 85 train / 21 val images;
- 196 positive pseudo-label boxes across 10 class names;
- class IDs are dataset-local and derived from `class_name`.

The available pretrained weight is a YOLOE segmentation model (`yoloe-v8l-seg-pf.pt`), while the source labels are five-field detection boxes. For this smoke run only, each bbox was represented as a rectangular polygon in the separate directory `data/ktd-smoke-seg-dataset-20260903/`. The original detection labels were not changed.

## Training Configuration

```text
model: yoloe-v8l-seg-pf.pt
task: segment
epochs: 1
imgsz: 640
batch: 4
workers: 0
device: cuda:0 in container, host GPU 4
amp: false
validation: skipped for smoke checkpoint generation
```

The Ultralytics final-epoch loop unconditionally invokes validation. The smoke process temporarily replaced only the in-process `validate` and `final_eval` methods so that the checkpoint could be saved without downloading the optional MobileCLIP asset. No package, image, or repository source was modified.

## Observed Training Output

The single `results.csv` row reported:

| Metric | Value |
|---|---:|
| Training time | 8.85204 s |
| `train/box` | 2.12843 |
| `train/seg` | 2.89643 |
| `train/cls` | 25.6827 |
| `train/dfl` | 1.81495 |
| Peak logged GPU memory | about 6.09 GiB |

The checkpoint reload check passed locally: task `segment`, 10 classes, checkpoint metadata `train_args.epochs=1`, and Ultralytics version `8.4.89`. The checkpoint's zero-based `epoch=0` corresponds to the one completed epoch shown as row `epoch=1` in `results.csv`.

## Artifacts

Remote run directory:

```text
/data/workspace/jzheng/yoloe_service/data/ktd-smoke-runs/yoloe-v8l-seg-1epoch-20260903-gpu4-final/
```

Local copies:

- `data/ktd-smoke-runs/remote-gpu4-20260903/weights/best.pt`
- `data/ktd-smoke-runs/remote-gpu4-20260903/results.csv`
- `data/ktd-smoke-runs/remote-gpu4-20260903/args.yaml`

The local `best.pt` SHA-256 is:

```text
5b76b4c6954c3846af6ff05036e284a6a0dabf00f70c90fa55d2c2424ca55fb4
```

It matches the remote `best.pt` SHA-256.

## Limitations

- Labels are model-derived pseudo-labels; no human-confirmed gold labels were used.
- Rectangular polygons are a smoke-only representation of bbox labels, not true segmentation masks.
- Validation and mAP calculation were skipped; `fitness=0` is a smoke placeholder, not an accuracy score.
- The run does not justify formal training, deployment, or any recall/precision claim.
