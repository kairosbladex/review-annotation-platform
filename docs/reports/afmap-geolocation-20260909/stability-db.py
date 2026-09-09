"""Frozen v2 history batch: exact retained requests, indexed results, CAS and rollback."""
import argparse
from collections import Counter
import copy
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
from stability_metadata import compact_metadata, expand_metadata_refs

VERSION = "geo-stability-20260909-v2"
ROOT = Path(__file__).resolve().parent
SNAPSHOT = ROOT / "stability-input-snapshot.json"
BACKUP = ROOT / "stability-history-correction-v2.json"
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("mode", choices=["idle", "snapshot", "prepare", "apply", "verify", "rollback"])
parser.add_argument("--config-jar", required=True)
parser.add_argument("--backup-sha256")
parser.add_argument("--natural-state", action="store_true", help="Repair only uncorrected v2 per-target metadata")
args = parser.parse_args()
if args.natural_state:
    SNAPSHOT = ROOT / "stability-natural-state-snapshot.json"
    BACKUP = ROOT / "stability-natural-state-correction.json"
def require(value, message):
    if not value: raise RuntimeError(message)
def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()
def save(path, value):
    with path.open("x") as f: json.dump(value, f, ensure_ascii=False, indent=2, default=str); f.write("\n")
def now(): return datetime.now(timezone.utc).isoformat()
password = sys.stdin.readline().rstrip("\r\n")
with zipfile.ZipFile(args.config_jar) as z:
    cfg = yaml.safe_load(z.read("config/application-test.yml"))["spring"]
ds = cfg["datasource"]
require(ds["url"].startswith("jdbc:mysql://192.168.110.10:33066/ktd_lx?"), "wrong database")
ssh = paramiko.SSHClient(); ssh.load_system_host_keys(); conn = None
ssh.connect("192.168.110.23", username="root", password=password, allow_agent=False, look_for_keys=False, timeout=10)
try:
    if args.mode == "idle":
        redis = cfg["redis"]
        ch = ssh.get_transport().open_channel("direct-tcpip", (redis["host"], int(redis["port"])), ("127.0.0.1", 0), timeout=10)
        stream = ch.makefile("rb")
        def read():
            line = stream.readline(); kind, value = line[:1], line[1:-2]
            if kind == b"*": return [read() for _ in range(int(value))]
            if kind == b"$":
                size = int(value)
                if size < 0: return None
                result = stream.read(size); stream.read(2); return result
            if kind == b"-": raise RuntimeError("Redis preflight failed")
            return value
        def command(*values):
            parts = [str(v).encode() for v in values]
            ch.sendall(b"*"+str(len(parts)).encode()+b"\r\n"+b"".join(b"$"+str(len(v)).encode()+b"\r\n"+v+b"\r\n" for v in parts))
            return read()
        if redis.get("password"): command("AUTH", redis["password"])
        command("SELECT", redis.get("database", 0))
        cursor, count = b"0", 0
        while True:
            cursor, keys = command("SCAN", cursor.decode(), "MATCH", "uavTasks:*", "COUNT", 500)
            count += len(keys)
            if cursor == b"0": break
        ch.close()
        print(json.dumps({"checkedAt": now(), "activeFlightCacheCount": count}))
        require(count == 0, "active flight; cannot restart")
        raise SystemExit(0)

    ch = ssh.get_transport().open_channel("direct-tcpip", ("192.168.110.10", 33066), ("127.0.0.1", 0), timeout=10)
    conn = pymysql.connect(user=ds["username"], password=ds["password"], database="ktd_lx", charset="utf8mb4",
        cursorclass=pymysql.cursors.DictCursor, defer_connect=True, ssl_disabled=True, read_timeout=30)
    conn.connect(sock=ch)
    readonly = args.mode in ("snapshot", "prepare", "verify")
    with conn.cursor() as cur:
        cur.execute("SET SESSION innodb_lock_wait_timeout=5")
        cur.execute("START TRANSACTION READ ONLY" if readonly else "START TRANSACTION")
        cur.execute("SELECT TRIGGER_NAME FROM information_schema.TRIGGERS WHERE TRIGGER_SCHEMA='ktd_lx' AND EVENT_OBJECT_TABLE='sys_annex_file'")
        require(not cur.fetchall(), "unexpected triggers")
        sql = ("SELECT a.id,a.busi_id,a.imgxy,a.ar_geo_coordinates,e.task_id,e.event_time,e.lng_lat_alt,"
               "e.terminal_type,e.event_from FROM sys_annex_file a JOIN bo_task_event e ON e.id=a.busi_id "
               "WHERE a.del_flag=0 AND e.del_flag=0 AND a.ar_geo_coordinates IS NOT NULL "
               "AND TRIM(a.ar_geo_coordinates) NOT IN ('','[]','null')")
        backup = None
        if args.natural_state and args.mode in ("snapshot", "prepare"):
            sql += " AND a.ar_geo_coordinates LIKE '%geo-stability-20260909-v2%' AND a.ar_geo_coordinates NOT LIKE '%correction_reason%'"
        if args.mode in ("apply", "verify", "rollback"):
            require(args.backup_sha256 and sha(BACKUP) == args.backup_sha256, "backup hash required")
            backup = json.loads(BACKUP.read_text())
            require(backup["snapshotSha256"] == sha(SNAPSHOT), "snapshot changed")
            ids = [r["id"] for r in backup["rows"]]
            require(ids, "empty batch")
            cur.execute(sql+" AND a.id IN ("+",".join(["%s"]*len(ids))+") ORDER BY a.id"+
                        ("" if readonly else " FOR UPDATE"), ids)
        else:
            cur.execute(sql+" ORDER BY a.id")
        current = cur.fetchall()
        for row in current: row["event_time"] = str(row["event_time"]) if row["event_time"] else None
        if args.mode == "snapshot":
            save(SNAPSHOT, {"checkedAt": now(), "rows": current, "readOnly": True})
            receipt = {"mode": "snapshot", "rows": len(current), "snapshotSha256": sha(SNAPSHOT)}
        elif args.mode == "prepare":
            snapshot = json.loads(SNAPSHOT.read_text())
            original = {r["id"]: r for r in snapshot["rows"]}
            session = requests.Session(); session.trust_env = False
            replays = {}; prepared = []; target_counts = {}; shifts = []
            for row in current:
                old_features = expand_metadata_refs(json.loads(row["ar_geo_coordinates"]))
                if not isinstance(old_features, list): continue
                features = copy.deepcopy(old_features); touched = False; stats = Counter()
                for feature in features:
                    props = feature.get("properties") or {}; meta = props.get("geolocation") or {}
                    req = meta.get("request")
                    source_version = VERSION if args.natural_state else "geo-quality-20260909-v1"
                    if meta.get("calculation_version") != source_version or not isinstance(req, dict): continue
                    if "$ref" in req:
                        # Fastjson shares the identical request across features.
                        # Resolve only the observed explicit root reference.
                        require(req == {"$ref": "$[0].properties.geolocation.request"}, "unknown request reference")
                        req = old_features[0]["properties"]["geolocation"]["request"]
                        require(isinstance(req, dict) and "$ref" not in req, "invalid request reference target")
                    require(row == original.get(row["id"]), "v1 row changed since snapshot")
                    request = dict(req, debug=False)
                    boxes = request.get("bboxes_xyxy") or request.get("bboxes_xywh")
                    require(isinstance(boxes, list) and boxes, "missing retained boxes")
                    index = meta.get("original_request_index", props.get("index"))
                    require(isinstance(index, int) and 0 <= index < len(boxes), "ambiguous request index")
                    key = json.dumps(request, sort_keys=True, ensure_ascii=False)
                    if key not in replays:
                        response = session.post("http://192.168.130.238:8181/v1/geo/from-bboxes", json=request, timeout=20)
                        require(response.status_code in (200, 400), "unexpected replay HTTP status")
                        replays[key] = {"status": response.status_code, "response": response.json()}
                    result = replays[key]; body = result["response"]
                    if result["status"] == 200:
                        require(body["metadata"]["calculation_version"] == VERSION, "worker version changed")
                        require([p["index"] for p in body["results"]] == list(range(len(boxes))), "response index mismatch")
                        point = body["results"][index]
                        metadata = dict(body["metadata"], quality=point["quality"])
                    else:
                        point = {"ok": False, "error": str(body.get("detail")), "quality": {"status": "unavailable", "accuracy_verified": False}}
                        metadata = {"calculation_version": VERSION, "coordinate_system": "WGS84", "quality": point["quality"], "accuracy_verified": False}
                    metadata.update(request=request, original_request_index=index,
                        correction_reason=("per_target_metadata_isolation" if args.natural_state else
                                           "v2_continuous_pose_fixed_mirror_and_actual_perturbation_screen"))
                    old_coord = (feature.get("geometry") or {}).get("coordinates")
                    if args.natural_state and point["ok"] and old_coord is not None:
                        require(old_coord[:2] == [point["lon"], point["lat"]], "natural-state patch must not move coordinates")
                        if meta.get("quality") == point["quality"] and meta.get("location_status") == "screened" and not meta.get("error"):
                            continue
                    if point["ok"]:
                        require(point["quality"]["status"] == "screened", "unscreened point")
                        coordinates = [point["lon"], point["lat"], 0]
                        feature["geometry"] = {"type": "Point", "coordinates": coordinates}
                        metadata["location_status"] = "time_corrected"
                        stats["screened"] += 1
                        stats["coordinatesChanged"] += int(old_coord is not None and old_coord[:2] != coordinates[:2])
                    else:
                        feature["geometry"] = None
                        metadata.update(location_status="unavailable", error=point["error"])
                        stats["unavailable"] += 1
                        stats["previouslyVisibleNowRejected"] += int(old_coord is not None)
                    props["geolocation"] = metadata; feature["properties"] = props; touched = True
                if touched:
                    prepared.append(dict(row, replacement=compact_metadata(features)))
                    event = row["busi_id"]
                    require(event not in target_counts or target_counts[event] == stats, "attachment results differ")
                    target_counts[event] = dict(stats)
            require(prepared, "no retained v1 requests")
            save(BACKUP, {"preparedAt": now(), "snapshotSha256": sha(SNAPSHOT), "version": VERSION,
                          "rows": prepared, "events": target_counts, "replays": replays})
            totals = Counter()
            for stats in target_counts.values(): totals.update(stats)
            receipt = {"mode": "prepare", "rows": len(prepared), "events": len(target_counts),
                       "totals": dict(totals), "backupSha256": sha(BACKUP)}
        else:
            saved = {r["id"]: r for r in backup["rows"]}
            require(set(saved) == {r["id"] for r in current}, "attachment set changed")
            for row in current:
                prior = saved[row["id"]]
                require(all(row[k] == prior[k] for k in row if k != "ar_geo_coordinates"), "event or box changed")
                expected = prior["ar_geo_coordinates"] if args.mode == "apply" else prior["replacement"]
                replacement = prior["ar_geo_coordinates"] if args.mode == "rollback" else prior["replacement"]
                require(row["ar_geo_coordinates"] == expected, "concurrent coordinate change")
                if args.mode != "verify":
                    cur.execute("UPDATE sys_annex_file SET ar_geo_coordinates=%s WHERE id=%s AND del_flag=0 "
                                "AND BINARY ar_geo_coordinates=BINARY %s", (replacement, row["id"], expected))
                    require(cur.rowcount == 1, "unexpected updated row count")
            receipt = {"mode": args.mode, "rows": len(current), "events": len(backup["events"]), "backupSha256": sha(BACKUP)}
        conn.rollback() if readonly else conn.commit()
        receipt.update(checkedAt=now(), databaseChanged=not readonly)
        prefix = "stability-natural-state-" if args.natural_state else "stability-history-"
        save(ROOT / (prefix+args.mode+"-receipt.json"), receipt)
        print(json.dumps(receipt))
finally:
    if conn: conn.close()
    ssh.close()
