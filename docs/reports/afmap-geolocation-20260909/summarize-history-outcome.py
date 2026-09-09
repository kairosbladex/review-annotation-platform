"""Produce the reconciled history outcome and explicit unresolved-input list."""
from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path

root = Path(__file__).resolve().parent
classification = json.loads((root / "history-classification.json").read_text())
replay = {r["eventId"]: r for r in json.loads((root / "history-quality-replay.json").read_text())["results"]}
api = json.loads((root / "recovered-history-api-verification.json").read_text())
assert api["result"] == "PASS"
verified = {r["eventId"]: r for r in api["events"]}
unresolved = []
for event in classification["events"]:
    event_id = event["eventId"]
    if event_id in verified or event["classification"] in {"already_time_corrected", "new_quality_managed"}: continue
    reason = event.get("reason", event["classification"])
    if event_id in replay: reason = replay[event_id]["error"]
    unresolved.append({"eventId": event_id, "taskId": event["taskId"], "eventTime": event["eventTime"],
                       "attachmentIds": event["attachmentIds"], "classification": event["classification"], "reason": reason,
                       "storedValuePreserved": True, "trustedTargetOutputDisabled": True})
assert len(unresolved) + len(verified) + 12 + 19 == classification["eventsWithTaskAndStoredLocation"]
prepare = json.loads((root / "recovered-history-correction-20260909-prepare-receipt.json").read_text())
output = {"checkedAt": datetime.now(timezone.utc).isoformat(), "snapshotAt": classification["snapshotAt"],
          "snapshotEventsWithStoredLocation": classification["eventsWithTaskAndStoredLocation"],
          "snapshotAttachmentRows": classification["snapshotRows"],
          "rowsWithoutTaskAssociationPreserved": len(classification["rowsWithoutTaskAssociation"]),
          "previousFlightEventsVerified": 12, "additionalHistoryEventsProcessed": len(verified),
          "additionalHistoryApiStatuses": dict(Counter(r["status"] for r in verified.values())),
          "additionalHistoryTargetTotals": prepare["totals"], "naturalNewEventsVerified": 19,
          "unresolvedEvents": len(unresolved), "unresolvedReasons": dict(Counter(r["reason"] for r in unresolved)),
          "unresolved": unresolved, "allHistoricalCoordinatesCorrected": False,
          "surveyedAccuracyVerified": False, "lateArrivingRowsIncluded": False}
(root / "history-remediation-outcome.json").write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n")
print(json.dumps({k: v for k, v in output.items() if k != "unresolved"}, ensure_ascii=False, indent=2))
