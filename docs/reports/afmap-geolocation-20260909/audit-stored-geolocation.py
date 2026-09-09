"""Read-only inventory of stored event geolocation; distances are screening signals."""

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import sys
import zipfile

import paramiko
import pymysql
import yaml


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--config-jar", required=True)
parser.add_argument("--output", required=True)
args = parser.parse_args()
password = sys.stdin.readline().rstrip("\r\n")
if not password:
    raise ValueError("Provide authorized SSH password on stdin")
with zipfile.ZipFile(args.config_jar) as archive:
    ds = yaml.safe_load(archive.read("config/application-test.yml"))["spring"]["datasource"]
if not ds["url"].startswith("jdbc:mysql://192.168.110.10:33066/ktd_lx?"):
    raise ValueError("Unexpected database configuration")


def parse(value):
    try:
        return json.loads(value)
    except (ValueError, TypeError):
        return None


def valid_coord(value):
    return (isinstance(value, list) and len(value) >= 2
            and all(isinstance(v, (float, int)) and math.isfinite(v) for v in value[:2])
            and -180 <= value[0] <= 180 and -90 <= value[1] <= 90)


def distance(a, b):
    lon1, lat1, lon2, lat2 = map(math.radians, [a[0], a[1], b[0], b[1]])
    h = math.sin((lat2-lat1)/2)**2 + math.cos(lat1)*math.cos(lat2)*math.sin((lon2-lon1)/2)**2
    return 2 * 6371008.8 * math.asin(math.sqrt(min(1.0, max(0.0, h))))


ssh = paramiko.SSHClient()
ssh.load_system_host_keys()
conn = None
try:
    ssh.connect("192.168.110.23", username="root", password=password,
                allow_agent=False, look_for_keys=False, timeout=10)
    channel = ssh.get_transport().open_channel(
        "direct-tcpip", ("192.168.110.10", 33066), ("127.0.0.1", 0), timeout=10)
    conn = pymysql.connect(user=ds["username"], password=ds["password"], database="ktd_lx",
                           charset="utf8mb4", cursorclass=pymysql.cursors.DictCursor,
                           defer_connect=True, read_timeout=30, write_timeout=15, ssl_disabled=True)
    conn.connect(sock=channel)
    with conn.cursor() as cur:
        cur.execute("START TRANSACTION READ ONLY")
        cur.execute("SELECT COUNT(*) total, SUM(del_flag=0) active, "
                    "SUM(del_flag=0 AND ar_geo_coordinates IS NOT NULL "
                    "AND TRIM(ar_geo_coordinates) NOT IN ('','[]','null')) active_with_coordinates "
                    "FROM sys_annex_file")
        counts = {k: int(v or 0) for k, v in cur.fetchone().items()}
        cur.execute("SELECT a.id attachment_id,a.busi_id,a.imgxy,a.ar_geo_coordinates,a.create_time attachment_created,"
                    "e.id event_id,e.task_id,e.event_time,e.create_time event_created,e.lng_lat_alt,"
                    "e.terminal_type,e.event_from "
                    "FROM sys_annex_file a LEFT JOIN bo_task_event e ON a.busi_id=e.id AND e.del_flag=0 "
                    "WHERE a.del_flag=0 AND a.ar_geo_coordinates IS NOT NULL "
                    "AND TRIM(a.ar_geo_coordinates) NOT IN ('','[]','null') ORDER BY a.id")
        rows = cur.fetchall()
finally:
    if conn:
        conn.rollback()
        conn.close()
    ssh.close()

checks = Counter()
events = {}
for row in rows:
    if not row["event_id"]:
        checks["rows_without_active_event"] += 1
        continue
    event = events.setdefault(row["event_id"], {
        "eventId": row["event_id"], "taskId": row["task_id"], "eventTime": str(row["event_time"]),
        "eventCreated": str(row["event_created"]), "terminalType": row["terminal_type"],
        "eventFrom": row["event_from"], "attachmentCount": 0, "points": {},
    })
    event["attachmentCount"] += 1
    main = parse(row["lng_lat_alt"])
    geometry = main.get("geometry", {}) if isinstance(main, dict) else {}
    origin = geometry.get("coordinates") if geometry.get("type") == "Point" else None
    if not valid_coord(origin):
        checks["rows_without_valid_main_point"] += 1
    features = parse(row["ar_geo_coordinates"])
    boxes = parse(row["imgxy"])
    if not isinstance(features, list):
        checks["rows_invalid_coordinate_json"] += 1
        continue
    if not isinstance(boxes, list) or not boxes:
        checks["rows_without_box_array"] += 1
    else:
        checks["rows_with_box_array"] += 1
        checks["rows_with_any_box_missing_size"] += int(any(
            not isinstance(b, dict) or not b.get("width") or not b.get("height") for b in boxes))
        checks["rows_with_any_box_missing_frame_size"] += int(any(
            not isinstance(b, dict) or not b.get("frame_width") or not b.get("frame_height") for b in boxes))
    for feature in features:
        geometry = feature.get("geometry", {}) if isinstance(feature, dict) else {}
        coord = geometry.get("coordinates") if geometry.get("type") == "Point" else None
        if not valid_coord(coord):
            checks["invalid_point_entries_with_attachment_duplicates"] += 1
            continue
        index = feature.get("properties", {}).get("index")
        if not (isinstance(boxes, list) and isinstance(index, int) and 0 <= index < len(boxes)):
            checks["point_entries_without_matching_box_index"] += 1
        key = json.dumps([index, coord], separators=(",", ":"))
        event["points"][key] = distance(origin, coord) if valid_coord(origin) else None

by_task = defaultdict(list)
by_day = defaultdict(list)
by_terminal = defaultdict(list)
event_results = []
for event in events.values():
    values = [v for v in event.pop("points").values() if v is not None]
    event["measurableUniquePoints"] = len(values)
    event["maxDistanceMeters"] = round(max(values), 3) if values else None
    event["minDistanceMeters"] = round(min(values), 3) if values else None
    event_results.append(event)
    by_task[event["taskId"]].append(event)
    by_day[event["eventTime"][:10]].append(event)
    by_terminal[event["terminalType"]].append(event)


def summarize(items):
    values = [e["maxDistanceMeters"] for e in items if e["maxDistanceMeters"] is not None]
    return {"events": len(items), "measurableEvents": len(values),
            "eventsOverMeters": {str(t): sum(v > t for v in values) for t in [100, 250, 500, 1000]},
            "maximumMeters": max(values) if values else None}


output = {
    "checkedAt": datetime.now(timezone.utc).isoformat(), "database": "ktd_lx",
    "scope": "all active attachment rows with stored coordinates; read-only transaction; no updates",
    "surveyedAccuracyVerified": False, "sameCrsAssumed": True,
    "warning": "Distance is measured from event reference, not ground truth. Threshold exceedance is not a confirmed geolocation defect.",
    "databaseCounts": counts, "checks": dict(checks), "summary": summarize(event_results),
    "taskCount": len(by_task),
    "byDay": {key: summarize(value) for key, value in sorted(by_day.items())},
    "byTerminalType": {str(key): summarize(value) for key, value in by_terminal.items()},
    "byTask": {str(key): summarize(value) for key, value in by_task.items()},
    "allEvents": sorted(event_results, key=lambda e: e["eventTime"]),
    "inputSnapshotSha256": hashlib.sha256(json.dumps(rows, default=str, sort_keys=True).encode()).hexdigest(),
}
Path(args.output).write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n")
print(json.dumps({key: output[key] for key in ["checkedAt", "databaseCounts", "checks", "summary", "taskCount", "byTerminalType", "byDay"]}, ensure_ascii=False, indent=2))
