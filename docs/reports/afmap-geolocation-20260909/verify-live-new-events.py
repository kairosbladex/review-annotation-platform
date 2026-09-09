"""Verify naturally arriving post-release events, including partial rejection, read-only."""
from datetime import datetime, timezone
import json
from pathlib import Path
import requests

root = Path(__file__).resolve().parent
snapshot = json.loads((root / "history-input-snapshot.json").read_text())
events = {}
for row in snapshot["rows"]:
    features = json.loads(row["ar_geo_coordinates"])
    if any((f.get("properties", {}).get("geolocation") or {}).get("location_status")
           in {"screened", "unavailable"} for f in features):
        events.setdefault(row["busi_id"], []).append(row)
assert events
base = "https://lxtest.zhuoyilingxi.com:8088/gis_server_a"
def get(path):
    response = requests.get(base + path, timeout=20); response.raise_for_status()
    body = response.json(); assert body["success"] and body["code"] == 200
    return body["data"]
checks = []
for event_id, rows in events.items():
    assert len({r["ar_geo_coordinates"] for r in rows}) == 1, "attachment outputs differ"
    features = json.loads(rows[0]["ar_geo_coordinates"])
    good = [f for f in features if f.get("geometry")]
    bad = [f for f in features if not f.get("geometry")]
    main = get("/portal/task/situation/detail/event/" + event_id)
    targets = get("/portal/task/situation/event/detail?id=" + event_id)
    assert main["id"] == event_id and main["eventTime"] == rows[0]["event_time"]
    assert json.loads(main["lngLatAlt"]) == json.loads(rows[0]["lng_lat_alt"])
    assert main["geolocationStatus"] == ("partial" if good and bad else "screened" if good else "unavailable")
    assert len(targets) == len(good)
    by_index = {f["properties"]["index"]: f for f in good}
    boxes = json.loads(rows[0]["imgxy"])
    for point in targets:
        index = point["properties"]["index"]
        assert point["geometry"] == by_index[index]["geometry"]
        assert point["properties"]["imgxy"] == boxes[index]
        metadata = point["properties"]["geolocation"]
        assert metadata["calculation_version"] == "geo-quality-20260909-v1"
        assert metadata["pose_time_delta_sec"] <= 1 and metadata["accuracy_verified"] is False
    checks.append({"eventId": event_id, "taskId": rows[0]["task_id"], "eventTime": rows[0]["event_time"],
                   "status": main["geolocationStatus"], "drawnTargets": len(good), "rejectedTargets": len(bad),
                   "rejectionReasons": [f["properties"]["geolocation"]["error"] for f in bad],
                   "maximumPoseDeltaSeconds": max(f["properties"]["geolocation"]["pose_time_delta_sec"] for f in features),
                   "eventRetained": True, "boxMappingVerified": True})
output = {"checkedAt": datetime.now(timezone.utc).isoformat(), "snapshotAt": snapshot["checkedAt"],
          "result": "PASS", "source": "natural business callbacks after deployment; no synthetic events",
          "events": checks, "surveyedAccuracyVerified": False, "flightLoadP95Verified": False}
(root / "live-new-events-verification.json").write_text(json.dumps(output, indent=2) + "\n")
print(json.dumps({"result": "PASS", "events": len(checks),
                  "drawnTargets": sum(c["drawnTargets"] for c in checks),
                  "rejectedTargets": sum(c["rejectedTargets"] for c in checks),
                  "maximumPoseDeltaSeconds": max(c["maximumPoseDeltaSeconds"] for c in checks)}))
