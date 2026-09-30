# GPU crop verifier run

Date: 2026-09-04

This is an experiment for a candidate-box crop verifier. It is not a
deployment or a production accuracy claim.

## Runtime

- Host: `AI-lingxi`
- GPU: host GPU 4, NVIDIA L20
- Container: `yoloe_service:latest`, one-shot container
- Python: 3.10.12
- PyTorch: 2.9.1+cu128
- Torchvision: 0.24.1+cu128
- Device inside container: `cuda:0`
- Remote staging: `/data/workspace/jzheng/yoloe_service/data/crop-verifier-runs/remote-gpu4-20260904-cd1/`

The online `yoloe_service` and `yoloe_service_test` containers remained
running. No 8188/8189 service or `/app/pretrain` weight was changed.

## Data

- Dataset: `cd1-693db6804f05`
- Crops: 2,841
- Training samples: 1,925 (344 human train + 1,581 GPT teacher)
- Validation samples: 94 human gold (70 true + 24 false)
- Teacher uncertain rows: excluded from the binary objective
- Locked test: 153 images / 633 boxes excluded; no locked labels were written
- Dataset files were SHA-256 checked after transfer; see `metrics.json` and the
  dataset `dataset_manifest.json` for the source counts and split policy.

Teacher rows were used only for training. The validation metrics below use
human gold rows only. The split is image-level and has zero train/validation
image-hash overlap.

## Run

```text
python3 tools/crop_verifier/train.py \
  --dataset dataset \
  --output output-6ep \
  --epochs 6 \
  --batch-size 64 \
  --lr 1e-4 \
  --seed 42 \
  --device cuda
```

The local copy of the final weight is `model.pt` with SHA-256
`866ee7013d8cea275a711b0bcc2a9757f152baa7041b8b76570e9193f738a504`.

## Validation result

At the trainer's fixed threshold `p_true >= 0.5`:

| Metric | Result |
|---|---:|
| Accuracy | 79/94 = 84.04% |
| Dangerous miss rate | 5/70 = 7.14% |
| False accept rate | 10/24 = 41.67% |

The 95% Wilson intervals are recorded in `metrics.json`. A separate 12 epoch
run ended at 75/94 accuracy, 7/70 dangerous misses, and 12/24 false accepts;
the 6 epoch run is retained as the current experiment candidate because the
longer run did not improve the final checkpoint.

An offline safety-oriented threshold scan on the 6 epoch predictions found one
possible triage policy: `p_true <= 0.1` rejects, `p_true >= 0.9` accepts, and the
middle is sent to human review. On this same 94-row validation set it produced
1/70 dangerous misses, 8/24 false accepts, and 79/94 binary coverage. This is
not calibrated or deployed; it needs a larger independent gold set and
per-class checks first.

## Limits and next gate

This run has one seed and only 94 validation rows, with sparse or absent
classes in the validation split. It measures candidate-box classification,
not whole-image recall or mAP. GPT teacher labels may transfer teacher bias.
The weight must not be connected to 8188/8189 until threshold calibration,
class-stratified human validation, and an untouched locked-test evaluation are
complete.
