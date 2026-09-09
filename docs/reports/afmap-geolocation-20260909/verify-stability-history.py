"""Read APIs, verify all snapshot rows, and inspect naturally arriving event versions."""
import argparse
import copy
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import zipfile
import paramiko
import pymysql
import requests
import yaml
from stability_metadata import expand_metadata_refs

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--config-jar", required=True)
parser.add_argument("--java-jar-sha256", default="be9b9ae409b1fcbda90c810758e85335fb75468a42ab805a8fb195092775fd76")
args = parser.parse_args()
root = Path(__file__).resolve().parent
snapshot = json.loads((root/"stability-input-snapshot.json").read_text())
path = root/"stability-history-correction-v2.json"
assert hashlib.sha256(path.read_bytes()).hexdigest() == json.loads((root/"stability-history-compact-receipt.json").read_text())["backupSha256"]
backup = json.loads(path.read_text())
rows = {}
for row in backup["rows"]: rows.setdefault(row["busi_id"], row)
base = "https://lxtest.zhuoyilingxi.com:8088/gis_server_a"
def get(path):
    response = requests.get(base+path, timeout=20); response.raise_for_status()
    body = response.json(); assert body["code"] == 200 and body["success"]
    return body["data"]
def check(item):
    event, row = item
    main = get("/portal/task/situation/detail/event/"+event)
    targets = get("/portal/task/situation/event/detail?id="+event)
    assert main["id"] == event and main["eventTime"] == row["event_time"]
    assert json.loads(main["lngLatAlt"]) == json.loads(row["lng_lat_alt"])
    features = expand_metadata_refs(json.loads(row["replacement"]))
    expected = {f["properties"]["index"]: f for f in features if f.get("geometry")}
    status = "unavailable" if not expected else "partial" if len(expected)<len(features) else "screened"
    assert main["geolocationStatus"] == status and len(targets) == len(expected), event
    boxes = json.loads(row["imgxy"])
    for point in targets:
        index = point["properties"]["index"]
        assert point["geometry"] == expected[index]["geometry"], event
        assert point["properties"]["imgxy"] == boxes[index], event
        expected_metadata = copy.deepcopy(expected[index]["properties"]["geolocation"])
        # The existing GIS response serializer converts null parameter values
        # to empty strings. Raw stored JSON is verified byte-for-byte below.
        parameters = expected_metadata.get("effective_overlay_parameters", {})
        for key, value in parameters.items():
            if value is None: parameters[key] = ""
        assert point["properties"]["geolocation"] == expected_metadata, event
    return {"eventId": event, "status": status, "drawnTargets": len(targets),
        "unavailableTargets": len(features)-len(targets), "eventRetained": True, "boxMappingVerified": True}
with ThreadPoolExecutor(max_workers=3) as pool:
    checks = list(pool.map(check, rows.items()))
legacy = "2e5ad77f4cc245baabf6cc1793208a7c"
assert get("/portal/task/situation/detail/event/"+legacy)["geolocationStatus"] == "legacy_unverified"
assert get("/portal/task/situation/event/detail?id="+legacy) == []
print(json.dumps({"apiEventsPassed": len(checks), "legacyStillHidden": True}), flush=True)

password = sys.stdin.readline().rstrip("\r\n")
with zipfile.ZipFile(args.config_jar) as z:
    ds = yaml.safe_load(z.read("config/application-test.yml"))["spring"]["datasource"]
assert ds["url"].startswith("jdbc:mysql://192.168.110.10:33066/ktd_lx?")
ssh = paramiko.SSHClient(); ssh.load_system_host_keys(); conn = None
try:
    ssh.connect("192.168.110.23", username="root", password=password, allow_agent=False, look_for_keys=False, timeout=10)
    _, out, err = ssh.exec_command("sha256sum /data/ktd/project/ktd-backend/biz-visualization.jar", timeout=20)
    jar_sha = out.read().decode().split()[0]
    assert out.channel.recv_exit_status() == 0 and jar_sha == args.java_jar_sha256
    ch = ssh.get_transport().open_channel("direct-tcpip", ("192.168.110.10", 33066), ("127.0.0.1", 0), timeout=10)
    conn = pymysql.connect(user=ds["username"], password=ds["password"], database="ktd_lx", charset="utf8mb4",
        cursorclass=pymysql.cursors.DictCursor, defer_connect=True, ssl_disabled=True, read_timeout=30)
    conn.connect(sock=ch)
    expected = {r["id"]: r["ar_geo_coordinates"] for r in snapshot["rows"]}
    expected.update({r["id"]: r["replacement"] for r in backup["rows"]})
    actual = {}
    with conn.cursor() as cur:
        cur.execute("START TRANSACTION READ ONLY")
        ids = list(expected)
        for start in range(0, len(ids), 500):
            batch = ids[start:start+500]
            cur.execute("SELECT id,ar_geo_coordinates FROM sys_annex_file WHERE id IN ("+",".join(["%s"]*len(batch))+")", batch)
            actual.update({r["id"]: r["ar_geo_coordinates"] for r in cur.fetchall()})
        assert actual == expected, "unexpected snapshot change"
        cur.execute("SELECT e.id,e.event_time,a.ar_geo_coordinates FROM bo_task_event e LEFT JOIN sys_annex_file a "
                    "ON a.busi_id=e.id AND a.del_flag=0 WHERE e.del_flag=0 AND e.event_time>=%s ORDER BY e.event_time",
                    ("2026-09-09 17:36:42",))
        natural = cur.fetchall()
        cur.execute("SELECT COUNT(*) AS count FROM sys_annex_file WHERE del_flag=0 AND ar_geo_coordinates LIKE %s",
                    ('%geo-quality-20260909-v1%',))
        remaining_v1 = cur.fetchone()["count"]
    result = {"result": "PASS", "checkedAt": datetime.now(timezone.utc).isoformat(), "events": checks,
        "drawnTargets": sum(c["drawnTargets"] for c in checks),
        "unavailableTargets": sum(c["unavailableTargets"] for c in checks),
        "snapshotRowsVerified": len(expected), "changedRows": len(backup["rows"]),
        "otherRowsUnchanged": len(expected)-len(backup["rows"]), "remainingV1Rows": remaining_v1,
        "naturalNewEventsSinceDeploy": len({r["id"] for r in natural}),
        "naturalRows": natural, "javaJarSha256": jar_sha,
        "legacyStillHidden": True, "visualMapVerified": False, "surveyedAccuracyVerified": False}
    (root/"stability-history-api-verification.json").write_text(json.dumps(result,ensure_ascii=False,indent=2,default=str)+"\n")
    print(json.dumps({k:v for k,v in result.items() if k not in ("events", "naturalRows")}))
finally:
    if conn: conn.rollback(); conn.close()
    ssh.close()
