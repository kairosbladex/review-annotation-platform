"""只读核验12条事件接口、框对应关系、主标识未改，以及旧点不再作为可靠点输出。"""
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import json
from pathlib import Path
import requests

ROOT = Path(__file__).resolve().parent
BASE = "https://lxtest.zhuoyilingxi.com:8088/gis_server_a"
backup = json.loads((ROOT / "verified-flight-correction-20260909-v2.json").read_text())
records = {}
for row in backup["rows"]: records.setdefault(row["busi_id"], row)

def get(path):
    response = requests.get(BASE + path, timeout=20)
    response.raise_for_status()
    body = response.json()
    assert body["code"] == 200 and body["success"], body.get("message")
    return body["data"]

def check(item):
    event_id, row = item
    main = get("/portal/task/situation/detail/event/" + event_id)
    targets = get("/portal/task/situation/event/detail?id=" + event_id)
    assert main["id"] == event_id and main["eventTime"] == row["event_time"]
    assert json.loads(main["lngLatAlt"]) == json.loads(row["lng_lat_alt"])
    assert main["geolocationStatus"] == "screened", main.get("geolocationStatus")
    expected = json.loads(row["replacement"])
    boxes = json.loads(row["imgxy"])
    assert len(targets) == len(expected), (event_id, len(targets), len(expected))
    for point in targets:
        index = point["properties"]["index"]
        assert point["geometry"]["coordinates"] == expected[index]["geometry"]["coordinates"]
        assert point["properties"]["imgxy"] == boxes[index]
        assert point["properties"]["geolocation"]["location_status"] == "time_corrected"
    return {"eventId": event_id, "targetsVerified": len(targets), "mainUnchanged": True,
            "boxMappingVerified": True, "geolocationStatus": main["geolocationStatus"]}

with ThreadPoolExecutor(max_workers=3) as pool:
    results = list(pool.map(check, records.items()))
# 原14:12:56样本已在第二批明确标为unavailable；用未恢复原输入的旧事件检查legacy行为。
legacy_id = "2e5ad77f4cc245baabf6cc1793208a7c"
legacy = get("/portal/task/situation/detail/event/" + legacy_id)
legacy_targets = get("/portal/task/situation/event/detail?id=" + legacy_id)
assert legacy["id"] == legacy_id and legacy["geolocationStatus"] == "legacy_unverified"
assert legacy_targets == [], "legacy target must not be displayed as verified"
output = {"checkedAt": datetime.now(timezone.utc).isoformat(), "result": "PASS", "events": results,
          "legacyEventRetained": True, "legacyTargetsHidden": True, "legacyEventId": legacy_id,
          "surveyedAccuracyVerified": False, "visualAcceptanceVerified": False}
(ROOT / "verified-flight-correction-api.json").write_text(json.dumps(output, indent=2) + "\n")
print(json.dumps({"result": "PASS", "events": len(results), "targets": sum(r["targetsVerified"] for r in results),
                  "legacyEventRetained": True, "legacyTargetsHidden": True}))
