"""Replay exact recovered requests with deployed source in an isolated interpreter."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shlex
import subprocess

root = Path(__file__).resolve().parent
classification = json.loads((root / "history-classification.json").read_text())
candidates = classification["candidates"]
assert candidates
remote = r'''
import asyncio,hashlib,json,sys
from pathlib import Path
from unittest.mock import patch
from fastapi import HTTPException
import app as service
source_sha=hashlib.sha256(Path(service.__file__).read_bytes()).hexdigest()
assert source_sha=='e16f0a97beb3b2109a3e595e5b251b5488ccd57848920d15c7964bf3f9dfe22e'
records=json.load(sys.stdin)
logs={}; results=[]
for rec in records:
 req=dict(rec['original']['request'],debug=False)
 try:
  url=req['log_url']
  if url not in logs:logs[url]=service._fetch_log_poses_from_url(url)
  with patch.object(service,'_fetch_log_poses_from_url',lambda u:logs[u]):
   response=asyncio.run(service.geo_from_bboxes(service.GeoFromBBoxesRequest(**req)))
  body=json.loads(response.body)
  assert body['count']==len(req['bboxes_xyxy'])
  assert sorted(p['index'] for p in body['results'])==list(range(body['count']))
  assert body['metadata']['pose_time_delta_sec']<=1
  for p in body['results']:
   if p['ok']:
    assert p['quality']['status']=='screened'
    assert p['quality']['horizontal_range_m']<=1000 and p['quality']['estimated_sensitivity_m']<=50
  results.append({'eventId':rec['eventId'],'status':200,'response':body})
 except HTTPException as exc:
  results.append({'eventId':rec['eventId'],'status':exc.status_code,'error':str(exc.detail)})
 except Exception as exc:
  results.append({'eventId':rec['eventId'],'status':'replay_error','error':type(exc).__name__+': '+str(exc)[:300]})
print(json.dumps({'sourceSha256':source_sha,'results':results}))
'''
run = subprocess.run(["ssh", "-o", "BatchMode=yes", "ai-lingxi",
                      "docker exec -i -e PYTHONDONTWRITEBYTECODE=1 overlay-api-2 python -c " + shlex.quote(remote)],
                     input=json.dumps(candidates), text=True, capture_output=True, timeout=240)
if run.returncode:
    print(run.stderr[-2000:]); raise SystemExit(run.returncode)
output = json.loads(run.stdout)
output.update(checkedAt=datetime.now(timezone.utc).isoformat(),
              classificationSha256=hashlib.sha256((root / "history-classification.json").read_bytes()).hexdigest(),
              scope="isolated replay of original requests; no service restart or database update",
              surveyedAccuracyVerified=False)
with (root / "history-quality-replay.json").open("x") as f:
    json.dump(output, f, ensure_ascii=False, indent=2); f.write("\n")
print(json.dumps({"events": len(output["results"]),
                  "successResponses": sum(r["status"] == 200 for r in output["results"]),
                  "passedTargets": sum(r.get("response", {}).get("ok_count", 0) for r in output["results"]),
                  "rejectedEvents": [{"eventId": r["eventId"], "status": r["status"], "error": r["error"]}
                                     for r in output["results"] if r["status"] != 200]}, ensure_ascii=False))
