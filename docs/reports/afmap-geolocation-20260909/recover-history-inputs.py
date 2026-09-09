"""Read-only DB snapshot and recovery of geo requests/responses from retained GIS logs."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shlex
import sys
import zipfile
import paramiko
import pymysql
import yaml

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--config-jar", required=True)
parser.add_argument("--recover-only", action="store_true", help="Reuse the immutable DB snapshot; resume retained-log scan")
args = parser.parse_args()
root = Path(__file__).resolve().parent
password = sys.stdin.readline().rstrip("\r\n")
with zipfile.ZipFile(args.config_jar) as z:
    ds = yaml.safe_load(z.read("config/application-test.yml"))["spring"]["datasource"]
assert ds["url"].startswith("jdbc:mysql://192.168.110.10:33066/ktd_lx?")
ssh = paramiko.SSHClient(); ssh.load_system_host_keys()
conn = None
try:
    ssh.connect("192.168.110.23", username="root", password=password, allow_agent=False, look_for_keys=False, timeout=10)
    channel = ssh.get_transport().open_channel("direct-tcpip", ("192.168.110.10", 33066), ("127.0.0.1", 0), timeout=10)
    conn = pymysql.connect(user=ds["username"], password=ds["password"], database="ktd_lx", charset="utf8mb4",
                           cursorclass=pymysql.cursors.DictCursor, defer_connect=True, ssl_disabled=True, read_timeout=30)
    conn.connect(sock=channel)
    with conn.cursor() as cur:
        cur.execute("START TRANSACTION READ ONLY")
        cur.execute("SELECT COUNT(*) total FROM bo_task_event WHERE del_flag=0")
        total_events = cur.fetchone()["total"]
        cur.execute("SELECT a.id,a.busi_id,a.imgxy,a.ar_geo_coordinates,e.task_id,e.event_time,"
                    "e.lng_lat_alt,e.terminal_type,e.event_from FROM sys_annex_file a "
                    "LEFT JOIN bo_task_event e ON e.id=a.busi_id AND e.del_flag=0 "
                    "WHERE a.del_flag=0 AND a.ar_geo_coordinates IS NOT NULL "
                    "AND TRIM(a.ar_geo_coordinates) NOT IN ('','[]','null') ORDER BY a.id")
        rows = cur.fetchall()
    conn.rollback(); conn.close(); conn = None
    output = {"checkedAt": datetime.now(timezone.utc).isoformat(), "database": "ktd_lx",
              "activeEventCount": total_events, "rows": rows, "readOnly": True}
    path = root / "history-input-snapshot.json"
    if not args.recover_only:
        with path.open("x") as f: json.dump(output, f, ensure_ascii=False, indent=2, default=str); f.write("\n")
    else:
        assert path.exists(), "original snapshot required"
    print(json.dumps({"snapshotRows": len(rows), "activeEvents": total_events,
                      "snapshotSha256": hashlib.sha256(path.read_bytes()).hexdigest()}), flush=True)
    remote = r'''
import json,re,os
from pathlib import Path
base=Path('/data/ktd/project/ktd-backend/logs')
def order(path):
 m=re.fullmatch(r'info\.(\d{4}-\d{2}-\d{2})-(\d+)\.log',path.name)
 return (m[1],int(m[2])) if m else ('9999',0)
paths=sorted(base.glob('info*.log'),key=order)
checkpoint=Path('/data/ktd/project/ktd-backend/geolocation-releases/20260909-fe2cba17f0/history-recovery.json')
saved=json.loads(checkpoint.read_text()) if checkpoint.exists() else {}
pending=saved.get('pending',{}); records=saved.get('records',[]); errors=saved.get('parseErrors',0); scanned=saved.get('files',[])
finished={p['name'] for p in scanned}
for path in paths:
 if path.name in finished:continue
 size=path.stat().st_size
 with path.open('rb',buffering=1024*1024) as f:
  for line_number,raw in enumerate(f,1):
   if b'ArApiUtil$HttpUtil' not in raw:continue
   line=raw.decode('utf-8',errors='replace')
   match=re.search(r'\[([^\]]+)\]\s+.*ArApiUtil\$HttpUtil',line)
   if not match:continue
   thread=match[1].strip()
   if '请求体：' in line:
    pending.pop(thread,None)
    try:request=json.loads(line.split('请求体：',1)[1])
    except ValueError:errors+=1;continue
    if isinstance(request,dict) and request.get('bboxes_xyxy') and request.get('log_url'):
     pending[thread]={'sourceLog':str(path),'requestLine':line_number,'request':request}
   elif '响应结果：' in line and thread in pending:
    record=pending.pop(thread)
    try:response=json.loads(line.split('响应结果：',1)[1])
    except ValueError:errors+=1;continue
    if isinstance(response,dict) and isinstance(response.get('results'),list):
     record.update(responseLog=str(path),responseLine=line_number,response=response)
     records.append(record)
 scanned.append({'name':path.name,'bytes':size})
 temporary=checkpoint.with_suffix('.next')
 temporary.write_text(json.dumps({'files':scanned,'parseErrors':errors,'pending':pending,'records':records},ensure_ascii=False))
 os.replace(temporary,checkpoint)
 print(json.dumps({'filesScanned':len(scanned),'originalRequests':len(records)}),flush=True)
print(json.dumps({'complete':True,'checkpoint':str(checkpoint)}),flush=True)
'''
    _, out, err = ssh.exec_command("nice -n 10 python3 -c " + shlex.quote(remote), timeout=120)
    complete = False
    for line in out:
        progress = json.loads(line)
        if progress.get("complete"): complete = True
        elif progress["filesScanned"] % 20 == 0: print(line.strip(), flush=True)
    if out.channel.recv_exit_status() or not complete: raise RuntimeError(err.read().decode()[-1000:])
    with ssh.open_sftp() as sftp:
        with sftp.open("/data/ktd/project/ktd-backend/geolocation-releases/20260909-fe2cba17f0/history-recovery.json") as f:
            recovered = json.load(f)
    recovered["pendingRequests"] = len(recovered.pop("pending"))
    recovered["checkedAt"] = datetime.now(timezone.utc).isoformat()
    with (root / "history-recovered-requests.json").open("x") as f:
        json.dump(recovered, f, ensure_ascii=False, indent=2); f.write("\n")
    print(json.dumps({"retainedLogFiles": len(recovered["files"]), "originalRequests": len(recovered["records"]),
                      "parseErrors": recovered["parseErrors"]}))
finally:
    if conn: conn.rollback(); conn.close()
    ssh.close()
