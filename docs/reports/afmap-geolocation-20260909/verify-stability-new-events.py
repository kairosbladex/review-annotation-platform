"""Verify naturally arriving post-release warnings, including a completed failed retry."""
from datetime import datetime, timezone
import json
from pathlib import Path
import requests
from stability_metadata import expand_metadata_refs

root = Path(__file__).resolve().parent
audit = json.loads((root/"stability-history-api-verification.json").read_text())
rows = {}
for row in audit["naturalRows"]: rows.setdefault(row["id"], row)
session = requests.Session(); session.trust_env = False
base = "https://lxtest.zhuoyilingxi.com:8088/gis_server_a"
def get(path):
    response = session.get(base+path, timeout=15); response.raise_for_status()
    body = response.json(); assert body["code"] == 200 and body["success"]
    return body["data"]
checks = []
for event, row in rows.items():
    features = expand_metadata_refs(json.loads(row["ar_geo_coordinates"]))
    main = get("/portal/task/situation/detail/event/"+event)
    targets = get("/portal/task/situation/event/detail?id="+event)
    expected = {f["properties"]["index"]: f for f in features if f.get("geometry")}
    assert main["id"] == event and main["eventTime"] == row["event_time"]
    assert len(targets) == len(expected)
    for point in targets:
        index = point["properties"]["index"]; saved = expected[index]
        assert point["geometry"] == saved["geometry"]
        meta = saved["properties"]["geolocation"]
        assert meta["calculation_version"] == "geo-stability-20260909-v2"
        assert meta["quality"]["status"] == "screened" and meta["pose_time_delta_sec"] <= 1
        assert meta["quality"]["estimated_sensitivity_m"] <= 50
        assert meta["accuracy_verified"] is False
    metadata = features[0]["properties"]["geolocation"]
    req = metadata["request"]
    request_count = len(req.get("bboxes_xyxy") or req.get("bboxes_xywh"))
    replay = session.post("http://192.168.130.238:8181/v1/geo/from-bboxes", json=req, timeout=15)
    if expected:
        assert main["geolocationStatus"] == "screened" and replay.status_code == 200
        for result in replay.json()["results"]:
            assert result["ok"]
            assert expected[result["index"]]["geometry"]["coordinates"][:2] == [result["lon"], result["lat"]]
    else:
        assert main["geolocationStatus"] == "unavailable" and targets == []
        assert metadata["retry_completed"] is True and metadata["retryable"] is False
        assert "TELEMETRY_TIME_MISMATCH" in metadata["error"]
        assert replay.status_code == 400 and "TELEMETRY_TIME_MISMATCH" in replay.text
    checks.append({"eventId": event, "eventTime": row["event_time"], "detectedTargets": request_count,
        "drawableTargets": len(targets), "eventRetained": True, "status": main["geolocationStatus"],
        "poseTimeDeltaSec": metadata.get("pose_time_delta_sec"), "error": metadata.get("error"),
        "retryCompleted": metadata.get("retry_completed"), "readOnlyReplayMatchesStoredCoordinates": True})
output = {"result": "PASS", "checkedAt": datetime.now(timezone.utc).isoformat(), "events": checks,
    "naturalEvents": len(checks), "screenedTargets": sum(c["drawableTargets"] for c in checks),
    "unavailableEvents": sum(c["status"] == "unavailable" for c in checks),
    "noSyntheticEventCreated": True, "surveyedAccuracyVerified": False}
(root/"stability-natural-events-verification.json").write_text(json.dumps(output,ensure_ascii=False,indent=2)+"\n")
print(json.dumps(output,ensure_ascii=False))
