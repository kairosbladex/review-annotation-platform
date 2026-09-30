# Human Gold Expansion Queue: Source Gated

This is a preparation or blocker artifact, not a gold-label file.

## Result

- Source snapshot: `pg1-bc8c818184d8` (`prompt-gold/1`)
- Formal blind-label queue: **0 boxes**
- Public source-recovery blockers: **2273 boxes / 421 images**
- Planned locked candidates excluded before row output: **500 boxes / 104 images**
- Existing selection human gold: **438 boxes / 166 images**

## Audience Boundary

- `queue.jsonl` is the only blind-annotator package. Its labels remain null.
- `blocked_sources.jsonl` is for source recovery and must not be labeled.
- `summary.json` is management-only because it contains aggregate human-label coverage.
- Planned locked rows never appear in any row-level output.

A formal queue row requires a content-addressed prompt-gold/2 snapshot, alias-complete clean identity-pixel evidence, one deterministic non-event source anchor, and alias-complete timezone-aware upstream time. Until those gates pass, do not train again or claim an accuracy increase.
