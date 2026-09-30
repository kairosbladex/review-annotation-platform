# Crop verifier multi-seed pre-calibration

This is a pre-calibration on the existing 94 human validation rows. It was
run before the human-gold expansion queue was labeled, so it is not the final
threshold decision.

## Fixed run contract

- Dataset: `cd1-693db6804f05`
- Train: 1,925 rows (GPT teacher true/false plus human train, with human
  labels taking precedence)
- Validation: 94 human gold rows (70 true, 24 false)
- Model: pretrained ResNet18 with an 11-class one-hot conditioning head
- Epochs: 6
- Batch size: 64
- Learning rate: `1e-4`
- Device: AI-lingxi host GPU 4, NVIDIA L20, one-shot container
- Locked test: excluded and not used for calibration

## Fixed threshold `p_true >= 0.5`

| Seed | Accuracy | Dangerous miss rate | False accept rate |
|---:|---:|---:|---:|
| 42 | 79/94 = 84.04% | 5/70 = 7.14% | 10/24 = 41.67% |
| 7 | 71/94 = 75.53% | 10/70 = 14.29% | 13/24 = 54.17% |
| 123 | 75/94 = 79.79% | 7/70 = 10.00% | 12/24 = 50.00% |

Across these three seeds, the simple mean is 79.79% accuracy, 10.48%
dangerous miss rate, and 48.61% false accept rate. The range is more
important than the mean given the small validation set.

## Safety-oriented triage scan

The offline policy `p_true <= 0.1` -> reject, `p_true >= 0.9` -> accept, and
the middle -> human review produced:

| Seed | Binary coverage | Dangerous miss rate | False accept rate |
|---:|---:|---:|---:|
| 42 | 79/94 = 84.04% | 1/70 = 1.43% | 8/24 = 33.33% |
| 7 | 72/94 = 76.60% | 5/70 = 7.14% | 8/24 = 33.33% |
| 123 | 76/94 = 80.85% | 2/70 = 2.86% | 10/24 = 41.67% |

This is a threshold scan over held-out predictions, not a calibrated
probability claim. It has not been integrated into the service.

## Gate for formal calibration

1. Resolve the clean-pixel, task-anchor, and upstream-time blockers in
   `data/prompt-eval/gold-expansion-20260904-v3-source-gated/`; its formal queue is
   currently empty, and the old v2 queue must not be labeled.
2. Build a new source-gated queue, complete blind human review, then export a
   new read-only prompt-gold snapshot after labels are persisted.
3. Rebuild image-isolated train/validation data; keep locked_test untouched.
4. Train multiple seeds and select thresholds using dmr, far, coverage, and
   per-class denominators together.
5. Evaluate once on the untouched locked test before any deployment decision.

No threshold or weight from this pre-calibration is approved for 8188/8189.
