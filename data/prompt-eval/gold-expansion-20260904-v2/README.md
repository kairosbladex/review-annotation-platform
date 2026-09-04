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
- `platform-map.jsonl`: live-platform lookup for the queue. Each row has one
  representative `candidate_id`, its `sample_id`, searchable `source_event_id`,
  and `target_index`; `aliases` lists duplicate candidates for audit only.
- `summary.json`: deterministic counts and exclusion policy
- The queue has no model-output fields and is not suitable as model input until
  its `label` values are replaced by audited human annotations in a new export.

## Exact platform workflow

The queue is a sampling list; the annotation platform does not load it
automatically. The API must be running on `http://127.0.0.1:8190` and the local
frontend on `http://127.0.0.1:5173`. If the API health check is already 200 but
the frontend is unavailable, start only the frontend:

```bash
cd /Users/wangxiaolin/PycharmProjects/review-annotation-platform/frontend
npm run dev
```

For each row in `platform-map.jsonl`, use the following fields in the same
row:

1. Copy `representative.source_event_id` into the workbench search box. The
   search box searches session, event, and rule values.
2. Open the matching sample. Select the candidate whose `target_index` equals
   `representative.target_index` (the UI displays it as `#N`).
3. Judge only the displayed image and red candidate box. Click `真目标` or
   `假目标`; leave it pending when the image evidence is insufficient. Do not
   use the model decision, teacher result, or any prior as the human label.
4. Move to the next row in `platform-map.jsonl`. Do not use auto-advance to
   label other boxes in the same sample.

The row's `alias_count` may be greater than one because the same canonical box
can occur in more than one model-run/sample record. Labeling one representative
candidate is sufficient for the exporter to form that canonical human-gold row;
the other aliases are retained for provenance and must not be used as separate
queue items.

When all 154 rows have a human decision, stop and report completion. Do not
edit `queue.jsonl` or `platform-map.jsonl` to record labels. The next step is a
read-only snapshot export and an audit of the clean gold, quarantine, and
split counts; `locked_test` remains excluded from training and threshold
selection.
