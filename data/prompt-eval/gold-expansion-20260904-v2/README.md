# Human gold expansion queue

This queue is a preparation artifact, not a gold-label file. Each row still
has `label: null` and must be reviewed by a human in the annotation platform.

## Scope

- Source snapshot: `pg1-bc8c818184d8`
- Queue rows: 154 candidates from 140 new source images
- Existing gold/quarantine images and all locked-test images were excluded
- Model priors, teacher labels, and service decisions are absent from the queue
- Selection is deterministic and balances class name and normalized box size
- At most two queued boxes come from one image

The queue currently has eligible candidates for eight classes. `光伏板`,
`没戴头盔`, and `车牌` have no image-level-independent unlabeled candidates in
this snapshot, so they are not represented in this queue.

## Labeling boundary

Review the displayed image and candidate box only. Write `true` or `false` when
the evidence is sufficient. Leave the item for follow-up when the evidence is
not sufficient; an unlabeled row is not a gold label. Do not copy a model
decision or teacher label into the human label.

After labeling, create a new read-only prompt-gold snapshot. Only the clean
human labels from that new snapshot may enter multi-seed training/threshold
calibration. Do not use `locked_test_gold.jsonl` for calibration.

## Files

- `queue.jsonl`: candidate rows for manual review
- `summary.json`: deterministic counts and exclusion policy
- The queue has no model-output fields and is not suitable as model input until
  its `label` values are replaced by audited human annotations in a new export.
