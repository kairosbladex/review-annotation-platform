"""Compare every snapshot attachment with its expected value; verify non-target preservation."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import zipfile
import paramiko
import pymysql
import yaml

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--config-jar", required=True)
args = parser.parse_args()
root = Path(__file__).resolve().parent
snapshot = json.loads((root / "history-input-snapshot.json").read_text())
backup = json.loads((root / "recovered-history-correction-20260909.json").read_text())
expected = {r["id"]: r["ar_geo_coordinates"] for r in snapshot["rows"]}
expected.update({r["id"]: r["replacement"] for r in backup["rows"]})
password = sys.stdin.readline().rstrip("\r\n")
with zipfile.ZipFile(args.config_jar) as z:
    ds = yaml.safe_load(z.read("config/application-test.yml"))["spring"]["datasource"]
assert ds["url"].startswith("jdbc:mysql://192.168.110.10:33066/ktd_lx?")
ssh = paramiko.SSHClient(); ssh.load_system_host_keys(); conn = None
try:
    ssh.connect("192.168.110.23", username="root", password=password, allow_agent=False, look_for_keys=False, timeout=10)
    ch = ssh.get_transport().open_channel("direct-tcpip", ("192.168.110.10", 33066), ("127.0.0.1", 0), timeout=10)
    conn = pymysql.connect(user=ds["username"], password=ds["password"], database="ktd_lx", charset="utf8mb4",
                           cursorclass=pymysql.cursors.DictCursor, defer_connect=True, ssl_disabled=True, read_timeout=30)
    conn.connect(sock=ch)
    actual = {}
    with conn.cursor() as cur:
        cur.execute("START TRANSACTION READ ONLY")
        ids = list(expected)
        for start in range(0, len(ids), 500):
            batch = ids[start:start + 500]
            cur.execute("SELECT id,ar_geo_coordinates FROM sys_annex_file WHERE id IN (" + ",".join(["%s"] * len(batch)) + ")", batch)
            actual.update({r["id"]: r["ar_geo_coordinates"] for r in cur.fetchall()})
    assert set(actual) == set(expected)
    mismatches = [i for i in expected if expected[i] != actual[i]]
    assert not mismatches, {"unexpectedChanges": mismatches}
    output = {"checkedAt": datetime.now(timezone.utc).isoformat(), "snapshotAt": snapshot["checkedAt"],
              "result": "PASS", "snapshotAttachmentRows": len(expected), "batchUpdatedRows": len(backup["rows"]),
              "otherSnapshotRowsUnchanged": len(expected) - len(backup["rows"]),
              "scope": "Only snapshot IDs; new rows arriving after the snapshot are outside this comparison"}
    (root / "history-snapshot-preservation.json").write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps(output))
finally:
    if conn: conn.rollback(); conn.close()
    ssh.close()
