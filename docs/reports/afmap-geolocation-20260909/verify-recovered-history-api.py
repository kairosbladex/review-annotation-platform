"""Check all corrected history events via production read APIs and the frozen backup."""
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import requests

root = Path(__file__).resolve().parent
path = root / "recovered-history-correction-20260909.json"
assert hashlib.sha256(path.read_bytes()).hexdigest() == "7ad3199bd8705f72ff264d36f3c110347fe9dc369e9e0565d06f4ef742448659"
backup = json.loads(path.read_text())
rows = {}
for row in backup["rows"]: rows.setdefault(row["busi_id"], row)
base = "https://lxtest.zhuoyilingxi.com:8088/gis_server_a"
def get(path):
    response = requests.get(base + path, timeout=20); response.raise_for_status()
    body = response.json(); assert body["code"] == 200 and body["success"]
    return body["data"]
def check(item):
    event_id, row = item
    main = get("/portal/task/situation/detail/event/" + event_id)
    targets = get("/portal/task/situation/event/detail?id=" + event_id)
    assert main["id"] == event_id and main["eventTime"] == row["event_time"]
    assert json.loads(main["lngLatAlt"]) == json.loads(row["lng_lat_alt"])
    features = json.loads(row["replacement"])
    expected = {f["properties"]["index"]: f for f in features if f.get("geometry")}
    status = "unavailable" if not expected else "partial" if len(expected) < len(features) else "screened"
    assert main["geolocationStatus"] == status, (event_id, main["geolocationStatus"], status)
    assert len(targets) == len(expected), (event_id, len(targets), len(expected))
    boxes = json.loads(row["imgxy"])
    for point in targets:
        index = point["properties"]["index"]
        assert point["geometry"] == expected[index]["geometry"], event_id
        assert point["properties"]["imgxy"] == boxes[index], event_id
        assert point["properties"]["geolocation"] == expected[index]["properties"]["geolocation"]
    return {"eventId": event_id, "status": status, "drawnTargets": len(targets),
            "rejectedTargets": len(features) - len(targets), "eventRetained": True,
            "mainUnchanged": True, "boxMappingVerified": True}
with ThreadPoolExecutor(max_workers=3) as pool:
    checks = list(pool.map(check, rows.items()))
output = {"checkedAt": datetime.now(timezone.utc).isoformat(), "result": "PASS", "events": checks,
          "backupSha256": hashlib.sha256(path.read_bytes()).hexdigest(),
          "surveyedAccuracyVerified": False, "visualMapRefreshVerified": False}
(root / "recovered-history-api-verification.json").write_text(json.dumps(output, indent=2) + "\n")
print(json.dumps({"result": "PASS", "events": len(checks), "targets": sum(c["drawnTargets"] for c in checks),
                  "rejectedTargets": sum(c["rejectedTargets"] for c in checks),
                  "allTargetsUnavailableEvents": sum(c["status"] == "unavailable" for c in checks)}))
