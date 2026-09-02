# ACCEPTANCE_PLATFORM-003

## Delivered

- Queue filtering by status, source, model version, class, confidence, raw decision, reason code, missing assets, and model failure.
- Three-column React workbench: queue, image/red-box canvas, and candidate inspector.
- Per-candidate true/false labels, notes, undo, revision conflict protection, next/previous target navigation, and `1`/`2` keyboard shortcuts.
- Task remains incomplete until every candidate in the sample + run has a label.
- Original candidates remain immutable; annotation revisions and audit events persist separately.

## Evidence

`backend/tests/test_annotation_flow.py` verifies state transitions and durable labels. Browser verification covered login, queue loading, detail display, true/false labeling, refresh persistence, and an empty-resource state; the browser console had no errors or warnings.

## Boundary

一期不修改 bbox/class，也不支持新增漏检框、整图无目标确认或多人分配；这些属于 PLATFORM-006。
