"""Prepare, verify, apply or revert the two verified attachment coordinate updates."""

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


EVENT_ID = "c5354403ae3b44edb35b2dfc01ede1a4"
ATTACHMENT_IDS = {"2c45da43f8584719854afa44ca10fd15", "38aba38a96d24e53a012faad5a4fc075"}
FIXED_SHA = "71383d766e34f15cb9074b47741af2fe74d40d18894ca6c163b8f9411ffc3242"
ROOT = Path(__file__).resolve().parent
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("mode", choices=["prepare", "verify", "apply", "rollback"])
parser.add_argument("--config-jar", required=True)
parser.add_argument("--backup", required=True)
args = parser.parse_args()
if not __debug__:
    raise RuntimeError("Run without -O; transaction preconditions must remain enabled")
backup_path = Path(args.backup)
password = sys.stdin.readline().rstrip("\r\n")
if not password:
    raise ValueError("Provide the authorized SSH password on stdin")
with zipfile.ZipFile(args.config_jar) as archive:
    ds = yaml.safe_load(archive.read("config/application-test.yml"))["spring"]["datasource"]
match = re.fullmatch(r"jdbc:mysql://([^:/]+):(\d+)/([^?]+)\?.*", ds["url"])
assert match and match.groups() == ("192.168.110.10", "33066", "ktd_lx")
ssh = paramiko.SSHClient()
ssh.load_system_host_keys()
connection = None
try:
    ssh.connect("192.168.110.23", username="root", password=password,
                allow_agent=False, look_for_keys=False, timeout=10)
    channel = ssh.get_transport().open_channel(
        "direct-tcpip", (match[1], int(match[2])), ("127.0.0.1", 0), timeout=10,
    )
    connection = pymysql.connect(
        user=ds["username"], password=ds["password"], database=match[3], charset="utf8mb4",
        cursorclass=pymysql.cursors.DictCursor, defer_connect=True,
        read_timeout=15, write_timeout=15, ssl_disabled=True,
    )
    connection.connect(sock=channel)
    with connection.cursor() as cursor:
        cursor.execute("SET SESSION innodb_lock_wait_timeout=5")
        read_only = args.mode in {"prepare", "verify"}
        cursor.execute("START TRANSACTION READ ONLY" if read_only else "START TRANSACTION")
        cursor.execute(
            "SELECT TRIGGER_NAME FROM information_schema.TRIGGERS "
            "WHERE TRIGGER_SCHEMA=%s AND EVENT_OBJECT_TABLE=%s", (match[3], "sys_annex_file"),
        )
        assert not cursor.fetchall(), "Inspect table triggers before modifying historical data"
        sql = "SELECT id,busi_id,ar_geo_coordinates,imgxy FROM sys_annex_file WHERE busi_id=%s AND del_flag=0"
        if not read_only:
            sql += " FOR UPDATE"
        cursor.execute(sql, (EVENT_ID,))
        rows = sorted(cursor.fetchall(), key=lambda row: row["id"])
        assert {row["id"] for row in rows} == ATTACHMENT_IDS and len(rows) == 2
        if args.mode == "prepare":
            record = json.loads((ROOT / "original-request-181008.json").read_text())
            replay = json.loads((ROOT / "original-event-replay.json").read_text())
            assert record["eventId"] == replay["eventId"] == EVENT_ID
            assert replay["result"] == "PASS" and replay["deployedSha256"] == FIXED_SHA
            assert replay["maxOriginalCoordinateDifferenceDegrees"] < 1e-10
            result = replay["fixed"]["response"]
            assert result["ok_count"] == result["count"] == 2
            assert result["debug_context"]["used_target"]["pose_time_delta_sec"] < 0.02
            new_results = {item["index"]: item for item in result["results"]}
            old_results = {item["index"]: item for item in record["originalResults"]}
            prepared = []
            for row in rows:
                old_features = json.loads(row["ar_geo_coordinates"])
                assert len(old_features) == 2 and {v["properties"]["index"] for v in old_features} == {0, 1}
                boxes = json.loads(row["imgxy"])
                assert [[b[key] for key in ["x", "y", "width", "height"]] for b in boxes] == record["actualDetectionBoxesXywh"]
                new_features = copy.deepcopy(old_features)
                for feature in new_features:
                    index = feature["properties"]["index"]
                    assert feature["geometry"]["coordinates"][:2] == [old_results[index]["lon"], old_results[index]["lat"]]
                    assert new_results[index]["ok"] is True
                    # Preserve the platform's existing ground-clamped altitude and metadata.
                    feature["geometry"]["coordinates"][:2] = [new_results[index]["lon"], new_results[index]["lat"]]
                prepared.append(dict(row, replacement=json.dumps(new_features, ensure_ascii=False, separators=(",", ":"))))
            backup = {
                "eventId": EVENT_ID, "preparedAt": datetime.now(timezone.utc).isoformat(),
                "database": "ktd_lx", "table": "sys_annex_file", "field": "ar_geo_coordinates",
                "deployedSha256": FIXED_SHA, "request": record["request"],
                "usedTarget": result["debug_context"]["used_target"],
                "cameraPose": result["debug_context"]["camera_pose"], "rows": prepared,
            }
            with backup_path.open("x") as output:
                json.dump(backup, output, ensure_ascii=False, indent=2)
                output.write("\n")
            connection.rollback()
            print(json.dumps({"mode": args.mode, "eventId": EVENT_ID, "preparedRows": len(rows),
                              "backup": str(backup_path), "databaseChanged": False}))
        else:
            backup_bytes = backup_path.read_bytes()
            backup = json.loads(backup_bytes)
            assert backup["eventId"] == EVENT_ID and backup["deployedSha256"] == FIXED_SHA
            saved = {row["id"]: row for row in backup["rows"]}
            assert set(saved) == ATTACHMENT_IDS
            for row in rows:
                previous = saved[row["id"]]
                expected = previous["ar_geo_coordinates"] if args.mode == "apply" else previous["replacement"]
                replacement = previous["replacement"] if args.mode == "apply" else previous["ar_geo_coordinates"]
                assert row["imgxy"] == previous["imgxy"] and row["ar_geo_coordinates"] == expected, "Concurrent record change; update aborted"
                if args.mode == "verify":
                    continue
                cursor.execute(
                    "UPDATE sys_annex_file SET ar_geo_coordinates=%s "
                    "WHERE id=%s AND busi_id=%s AND del_flag=0 AND BINARY ar_geo_coordinates=BINARY %s",
                    (replacement, row["id"], EVENT_ID, expected),
                )
                assert cursor.rowcount == 1
            if read_only:
                connection.rollback()
            else:
                connection.commit()
            print(json.dumps({"mode": args.mode, "eventId": EVENT_ID,
                              "verifiedRows" if read_only else "updatedRows": len(rows),
                              "backupSha256": hashlib.sha256(backup_bytes).hexdigest(),
                              "databaseChanged": not read_only}))
except BaseException:
    if connection:
        connection.rollback()
    raise
finally:
    if connection:
        connection.close()
    ssh.close()
