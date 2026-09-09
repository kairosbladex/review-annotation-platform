"""Prepare/apply/verify/rollback a frozen batch; never infer missing historical inputs."""
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
import yaml

ROOT = Path(__file__).resolve().parent
SOURCE = "e16f0a97beb3b2109a3e595e5b251b5488ccd57848920d15c7964bf3f9dfe22e"
def require(value, message):
    if not value: raise RuntimeError(message)
def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("mode", choices=["prepare", "apply", "verify", "rollback"])
parser.add_argument("--config-jar", required=True)
parser.add_argument("--backup", required=True)
parser.add_argument("--backup-sha256", help="Required for apply/verify/rollback")
args = parser.parse_args()
backup_path = Path(args.backup)
receipt_path = backup_path.with_name(backup_path.stem + "-" + args.mode + "-receipt.json")
require(args.mode == "verify" or not receipt_path.exists(), "existing operation receipt; inspect before retrying")
snapshot = json.loads((ROOT / "history-input-snapshot.json").read_text())
classification = json.loads((ROOT / "history-classification.json").read_text())
replay = json.loads((ROOT / "history-quality-replay.json").read_text())
require(replay["sourceSha256"] == SOURCE, "unexpected replay source")
require(replay["classificationSha256"] == sha(ROOT / "history-classification.json"), "classification changed")
require(classification["snapshotSha256"] == sha(ROOT / "history-input-snapshot.json"), "snapshot changed")
candidates = {r["eventId"]: r for r in classification["candidates"]}
results = {r["eventId"]: r for r in replay["results"] if r["status"] == 200}
require(results and set(results) <= set(candidates), "unexpected replay events")
old_rows = [r for r in snapshot["rows"] if r["busi_id"] in results]
require(set(r["busi_id"] for r in old_rows) == set(results), "missing snapshot events")
password = sys.stdin.readline().rstrip("\r\n")
with zipfile.ZipFile(args.config_jar) as z:
    ds = yaml.safe_load(z.read("config/application-test.yml"))["spring"]["datasource"]
require(ds["url"].startswith("jdbc:mysql://192.168.110.10:33066/ktd_lx?"), "unexpected database")
ssh = paramiko.SSHClient(); ssh.load_system_host_keys(); conn = None
try:
    ssh.connect("192.168.110.23", username="root", password=password, allow_agent=False, look_for_keys=False, timeout=10)
    ch = ssh.get_transport().open_channel("direct-tcpip", ("192.168.110.10", 33066), ("127.0.0.1", 0), timeout=10)
    conn = pymysql.connect(user=ds["username"], password=ds["password"], database="ktd_lx", charset="utf8mb4",
                           cursorclass=pymysql.cursors.DictCursor, defer_connect=True, ssl_disabled=True, read_timeout=30)
    conn.connect(sock=ch)
    read_only = args.mode in {"prepare", "verify"}
    with conn.cursor() as cur:
        cur.execute("SET SESSION innodb_lock_wait_timeout=5")
        cur.execute("START TRANSACTION READ ONLY" if read_only else "START TRANSACTION")
        cur.execute("SELECT TRIGGER_NAME FROM information_schema.TRIGGERS WHERE TRIGGER_SCHEMA='ktd_lx' AND EVENT_OBJECT_TABLE='sys_annex_file'")
        require(not cur.fetchall(), "unexpected DB triggers")
        ids = sorted(results)
        cur.execute("SELECT a.id,a.busi_id,a.imgxy,a.ar_geo_coordinates,e.task_id,e.event_time,e.lng_lat_alt,"
                    "e.terminal_type,e.event_from FROM sys_annex_file a JOIN bo_task_event e ON e.id=a.busi_id "
                    "WHERE a.del_flag=0 AND e.del_flag=0 AND e.id IN (" + ",".join(["%s"] * len(ids)) + ") "
                    "ORDER BY a.id" + ("" if read_only else " FOR UPDATE"), ids)
        current = cur.fetchall()
        for row in current: row["event_time"] = str(row["event_time"]) if row["event_time"] else None
        require({r["id"] for r in current} == {r["id"] for r in old_rows}, "attachment set changed")
        if args.mode == "prepare":
            original = {r["id"]: r for r in old_rows}
            prepared = []; summaries = {}
            for row in current:
                require(row == original[row["id"]], "live row changed since snapshot")
                event_id = row["busi_id"]
                record = candidates[event_id]; response = results[event_id]["response"]
                require(response["count"] == len(record["original"]["request"]["bboxes_xyxy"]), "response count mismatch")
                indexed = {p["index"]: p for p in response["results"]}
                require(len(indexed) == response["count"] and set(indexed) == set(range(response["count"])), "invalid response indices")
                old_features = {f["properties"]["index"]: f for f in json.loads(row["ar_geo_coordinates"])}
                mapped = []; changed = 0; rejected = 0; added = 0
                for stored_index, original_index in enumerate(record["storedIndexToRequestIndex"][row["id"]]):
                    point = indexed[original_index]
                    feature = copy.deepcopy(old_features.get(original_index, {"type": "Feature", "properties": {}}))
                    metadata = dict(response["metadata"], source_sha256=SOURCE,
                                    correction_reason="recovered_original_request_with_latest_trajectory_and_quality_screen",
                                    original_request_index=original_index,
                                    request=dict(record["original"]["request"], debug=False), quality=point["quality"])
                    if point["ok"]:
                        require(point["quality"]["status"] == "screened", "unscreened coordinate")
                        old_coord = (feature.get("geometry") or {}).get("coordinates")
                        added += int(old_coord is None)
                        changed += int(old_coord is not None and old_coord[:2] != [point["lon"], point["lat"]])
                        feature["geometry"] = {"type": "Point", "coordinates": [point["lon"], point["lat"], 0]}
                        metadata["location_status"] = "time_corrected"
                    else:
                        rejected += 1; feature["geometry"] = None
                        metadata.update(location_status="unavailable", error=point["error"])
                    feature["properties"].update(index=stored_index, geolocation=metadata)
                    mapped.append(feature)
                prepared.append(dict(row, replacement=json.dumps(mapped, ensure_ascii=False, separators=(",", ":"))))
                summary = {"targets": len(mapped), "screenedTargets": len(mapped) - rejected,
                           "rejectedTargets": rejected, "changedCoordinateTargets": changed, "restoredCoordinateTargets": added}
                require(event_id not in summaries or summaries[event_id] == summary, "attachment summaries differ")
                summaries[event_id] = summary
            backup = {"preparedAt": datetime.now(timezone.utc).isoformat(), "sourceSha256": SOURCE,
                      "classificationSha256": sha(ROOT / "history-classification.json"),
                      "replaySha256": sha(ROOT / "history-quality-replay.json"), "events": summaries, "rows": prepared}
            with backup_path.open("x") as f: json.dump(backup, f, ensure_ascii=False, indent=2); f.write("\n")
            conn.rollback()
            receipt = {"mode": "prepare", "events": len(summaries), "rows": len(prepared),
                       "totals": {k: sum(v[k] for v in summaries.values()) for k in next(iter(summaries.values()))},
                       "backupSha256": sha(backup_path), "databaseChanged": False}
        else:
            require(args.backup_sha256 and sha(backup_path) == args.backup_sha256, "backup hash required or mismatched")
            backup = json.loads(backup_path.read_text())
            require(backup["sourceSha256"] == SOURCE and backup["replaySha256"] == sha(ROOT / "history-quality-replay.json"), "backup lineage mismatch")
            saved = {r["id"]: r for r in backup["rows"]}
            require(set(saved) == {r["id"] for r in current}, "backup attachment set differs")
            for row in current:
                prior = saved[row["id"]]
                require(all(row[k] == prior[k] for k in row if k != "ar_geo_coordinates"), "event or box changed")
                expected = prior["ar_geo_coordinates"] if args.mode == "apply" else prior["replacement"]
                replacement = prior["replacement"] if args.mode == "apply" else prior["ar_geo_coordinates"]
                require(row["ar_geo_coordinates"] == expected, "concurrent coordinate change; batch aborted")
                if args.mode != "verify":
                    cur.execute("UPDATE sys_annex_file SET ar_geo_coordinates=%s WHERE id=%s AND busi_id=%s "
                                "AND del_flag=0 AND BINARY ar_geo_coordinates=BINARY %s", (replacement, row["id"], row["busi_id"], expected))
                    require(cur.rowcount == 1, "unexpected updated row count")
            conn.rollback() if read_only else conn.commit()
            receipt = {"mode": args.mode, "events": len(backup["events"]), "rows": len(current),
                       "backupSha256": sha(backup_path), "databaseChanged": not read_only}
        receipt["checkedAt"] = datetime.now(timezone.utc).isoformat()
        with receipt_path.open("w" if args.mode == "verify" else "x") as f:
            json.dump(receipt, f, indent=2); f.write("\n")
        print(json.dumps(receipt))
except BaseException:
    if conn: conn.rollback()
    raise
finally:
    if conn: conn.close()
    ssh.close()
