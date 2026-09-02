# ACCEPTANCE_PLATFORM-005

## Delivered

- Asynchronous export records and job status endpoints.
- Provenance-rich JSONL with model version, normalized bbox, model decision fields, human label/note, label revision actor and time.
- YOLO ZIP with original images, one merged label file per sample, and `hard_negative.jsonl`; false candidates never enter positive labels.
- Completed-task guard prevents unfinished samples from formal export.
- Candidate precision, class/confidence/reason-code breakdowns, and explicit no-recall warning at `/api/v1/model-runs/{run_id}/evaluation`.
- `/content` and `/download` download routes.

## Evidence

`backend/tests/test_exports.py` verifies JSONL provenance, evaluation precision, no-recall declaration, ZIP image/label output, merged multi-target labels, and hard-negative file presence.

## Boundary

一期 precision is calculated only over model-produced candidates. Recall/mAP require second-phase full-image annotation and are deliberately not reported.
