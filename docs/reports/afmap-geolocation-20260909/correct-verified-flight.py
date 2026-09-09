"""事务修正已恢复原请求的12条事件；10条更新坐标，另2条仅补追溯元数据。"""
import argparse
import copy
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import sys
import zipfile
import paramiko
import pymysql
import yaml

TASK = "5c0355c1ca2e4e56a319dfc181f58fd0"
ACCEPTED = "c5354403ae3b44edb35b2dfc01ede1a4"
ROOT = Path(__file__).resolve().parent

def require(condition, message):
    if not condition:
        raise RuntimeError(message)

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("mode", choices=["prepare", "apply", "verify", "rollback"])
parser.add_argument("--config-jar", required=True)
parser.add_argument("--replay", required=True)
parser.add_argument("--backup", required=True)
args = parser.parse_args()
originals = json.loads((ROOT / "same-task-original-requests.json").read_text())
records = {r["eventId"]: r for r in originals["records"]}
replay = json.loads(Path(args.replay).read_text())
require(replay["result"] == "PASS" and replay["normalEventsPassed"] == 12, "quality replay must pass")
fixed = {r["eventId"]: r["response"] for r in replay["replays"] if r["group"] == "normal"}
require(set(fixed) == set(records) and len(records) == 12, "event set differs")
password = sys.stdin.readline().rstrip("\r\n")
require(bool(password), "SSH password required on stdin")
with zipfile.ZipFile(args.config_jar) as archive:
    ds = yaml.safe_load(archive.read("config/application-test.yml"))["spring"]["datasource"]
match = re.fullmatch(r"jdbc:mysql://([^:/]+):(\d+)/([^?]+)\?.*", ds["url"])
require(match is not None and match.groups() == ("192.168.110.10", "33066", "ktd_lx"), "unexpected database")
ssh = paramiko.SSHClient()
ssh.load_system_host_keys()
conn = None
try:
    ssh.connect("192.168.110.23", username="root", password=password, allow_agent=False, look_for_keys=False, timeout=10)
    channel = ssh.get_transport().open_channel("direct-tcpip", (match[1], int(match[2])), ("127.0.0.1", 0), timeout=10)
    conn = pymysql.connect(user=ds["username"], password=ds["password"], database=match[3], charset="utf8mb4",
                           cursorclass=pymysql.cursors.DictCursor, defer_connect=True, ssl_disabled=True, read_timeout=20, write_timeout=20)
    conn.connect(sock=channel)
    read_only = args.mode in {"prepare", "verify"}
    with conn.cursor() as cur:
        cur.execute("SET SESSION innodb_lock_wait_timeout=5")
        cur.execute("START TRANSACTION READ ONLY" if read_only else "START TRANSACTION")
        cur.execute("SELECT TRIGGER_NAME FROM information_schema.TRIGGERS WHERE TRIGGER_SCHEMA=%s AND EVENT_OBJECT_TABLE=%s", (match[3], "sys_annex_file"))
        require(not cur.fetchall(), "inspect triggers before update")
        cur.execute("SELECT a.id,a.busi_id,a.imgxy,a.ar_geo_coordinates,e.task_id,e.lng_lat_alt,e.event_time FROM sys_annex_file a JOIN bo_task_event e ON e.id=a.busi_id WHERE e.task_id=%s AND e.del_flag=0 AND a.del_flag=0 ORDER BY a.id" + ("" if read_only else " FOR UPDATE"), (TASK,))
        rows = cur.fetchall()
        require(len(rows) == 24 and {r["busi_id"] for r in rows} == set(records), "expected 24 attachments across 12 events")
        for row in rows:
            row["event_time"] = str(row["event_time"])
        if args.mode == "prepare":
            prepared = []
            changed_events = set()
            changed_points = 0
            for row in rows:
                event_id = row["busi_id"]
                record, response = records[event_id], fixed[event_id]
                request = record["request"]
                require(row["event_time"] == record["eventTime"], "event time changed")
                features = json.loads(row["ar_geo_coordinates"])
                boxes = json.loads(row["imgxy"])
                n = len(record["originalResults"])
                require(len(features) == len(boxes) == response["ok_count"] == n, "target count mismatch")
                old_results = {p["index"]: p for p in record["originalResults"]}
                new_results = {p["index"]: p for p in response["results"]}
                require({f["properties"]["index"] for f in features} == set(range(n)), "invalid old indices")
                # 恢复请求框到已保存imgxy的对应关系，避免继续继承旧排序错误。
                available = {}
                for i, bbox in enumerate(request["bboxes_xyxy"]):
                    available.setdefault(tuple(bbox), []).append(i)
                ordered = []
                for stored_index, box in enumerate(boxes):
                    bbox = (max(box["x"]-50, 0), max(box["y"]-50, 0),
                            min(box["x"]+50, request["image_width"]), min(box["y"]+50, request["image_height"]))
                    require(bool(available.get(bbox)), "saved box cannot be matched to original request")
                    original_index = available[bbox].pop(0)
                    feature = copy.deepcopy(next(f for f in features if f["properties"]["index"] == original_index))
                    old = old_results[original_index]
                    new = new_results[original_index]
                    expected = new if event_id == ACCEPTED else old
                    current = feature["geometry"]["coordinates"]
                    require(current[:2] == [expected["lon"], expected["lat"]], "coordinates changed since original replay")
                    if current[:2] != [new["lon"], new["lat"]]:
                        changed_events.add(event_id); changed_points += 1
                    feature["geometry"]["coordinates"][:2] = [new["lon"], new["lat"]]
                    metadata = dict(response["metadata"], location_status="time_corrected", quality=new["quality"],
                                    correction_reason="stale_trajectory_cache", original_request_index=original_index,
                                    source_sha256=replay["candidateSha256"], request=request)
                    feature["properties"].update(index=stored_index, geolocation=metadata)
                    ordered.append(feature)
                prepared.append(dict(row, replacement=json.dumps(ordered, ensure_ascii=False, separators=(",", ":"))))
            require(len(changed_events) == 10 and ACCEPTED not in changed_events and changed_points == 148, "expected ten events / 74 targets duplicated in two attachments")
            backup = {"preparedAt": datetime.now(timezone.utc).isoformat(), "taskId": TASK,
                      "mode": "verified original request replay", "sourceSha256": replay["candidateSha256"],
                      "changedCoordinateEvents": sorted(changed_events), "changedCoordinateAttachmentPoints": changed_points,
                      "rows": prepared}
            with Path(args.backup).open("x") as output:
                json.dump(backup, output, ensure_ascii=False, indent=2); output.write("\n")
            conn.rollback()
            print(json.dumps({"mode": args.mode, "rows": len(prepared), "coordinateChangedEvents": len(changed_events), "uniqueCorrectedTargets": changed_points//2, "databaseChanged": False}))
        else:
            backup_bytes = Path(args.backup).read_bytes()
            backup = json.loads(backup_bytes)
            require(backup["taskId"] == TASK and backup["sourceSha256"] == replay["candidateSha256"], "backup/replay mismatch")
            saved = {r["id"]: r for r in backup["rows"]}
            require(set(saved) == {r["id"] for r in rows}, "attachment set changed")
            for row in rows:
                prior = saved[row["id"]]
                require(all(row[k] == prior[k] for k in ("busi_id", "imgxy", "event_time", "lng_lat_alt", "task_id")), "event or box changed")
                expected = prior["ar_geo_coordinates"] if args.mode == "apply" else prior["replacement"]
                replacement = prior["replacement"] if args.mode == "apply" else prior["ar_geo_coordinates"]
                require(row["ar_geo_coordinates"] == expected, "concurrent coordinate update; abort")
                if args.mode != "verify":
                    cur.execute("UPDATE sys_annex_file SET ar_geo_coordinates=%s WHERE id=%s AND busi_id=%s AND del_flag=0 AND BINARY ar_geo_coordinates=BINARY %s", (replacement, row["id"], row["busi_id"], expected))
                    require(cur.rowcount == 1, "row count mismatch")
            conn.rollback() if read_only else conn.commit()
            print(json.dumps({"mode": args.mode, "rows": len(rows), "databaseChanged": not read_only, "backupSha256": hashlib.sha256(backup_bytes).hexdigest()}))
except BaseException:
    if conn: conn.rollback()
    raise
finally:
    if conn: conn.close()
    ssh.close()
