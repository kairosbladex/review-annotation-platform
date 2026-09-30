# Human Gold Expansion Queue v2: Superseded Tombstone

**DO NOT LABEL OR CONSUME THIS DIRECTORY.**

The 2026-09-04 source audit invalidated this queue because its images may contain
burned-in boxes or text, task-level source grouping is missing, and upstream frame
time is unavailable. The executable `queue.jsonl` and `platform-map.jsonl` payloads
were intentionally removed so human and machine workflows fail closed.

`summary.json` retains historical counts only and declares `machine_usable=false`.
It is not a queue manifest and must not be used to reconstruct candidates.

Use `gold-expansion-20260904-v3-source-gated/`. Its formal queue remains empty until
clean pixels, deterministic source groups, and timezone-aware upstream time all pass
the provenance gate.
