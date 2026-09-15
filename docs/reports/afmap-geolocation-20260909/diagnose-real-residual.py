"""只读重放真实留存请求，在部署解释器逐项量化落点变化，不把位移当精度。"""
import json
import shlex
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent
records = []
for filename in ('same-task-original-requests.json', 'recent-original-requests.json'):
    for record in json.loads((ROOT / filename).read_text())['records']:
        if record.get('request'):
            records.append(record)
reported = json.loads((ROOT / 'original-request-181008.json').read_text())
records.insert(0, reported)
unique = {}
for record in records:
    key = json.dumps(record['request'], sort_keys=True)
    unique.setdefault(key, record)
records = list(unique.values())[:24]
REMOTE = r'''
import asyncio, copy, hashlib, json, math, sys
from pathlib import Path
from unittest.mock import patch
from datetime import datetime, timezone
import app as svc
data=json.load(sys.stdin); logs={}; output=[]
def distance(a,b):
 return math.hypot(math.radians(a['lat']-b['lat'])*6371008.8, math.radians(a['lon']-b['lon'])*6371008.8*math.cos(math.radians(b['lat'])))
def solve(req):
 try:
  with patch.object(svc,'_fetch_log_poses_from_url',lambda u:logs[u]):
   return json.loads(asyncio.run(svc.geo_from_bboxes(svc.GeoFromBBoxesRequest(**req))).body)
 except Exception as exc: return {'error':str(exc)}
for record in data:
 req=copy.deepcopy(record['request']); req['debug']=True
 url=req.get('log_url')
 try:
  if url not in logs: logs[url]=svc._fetch_log_poses_from_url(url)
  baseline=solve(req)
  variants=[('principal_zero',{}, {'cx_offset_px':0,'cy_offset_px':0}),
    ('pitch_plus_1',{}, {'pitch_offset_deg':1}),('pitch_minus_1',{}, {'pitch_offset_deg':-1}),
    ('ground_plus_10',{}, {'ground_h_offset_m':10}),('ground_minus_10',{}, {'ground_h_offset_m':-10}),
    ('zoom_plus_10pct',{}, {'zoom_scale':1.1}),('sensor_16_9',{'sensor_h_mm':req['sensor_w_mm']*9/16},{}),
    ('time_plus_100ms',{'shot_time_ms':req['shot_time_ms']+100},{}),
    ('time_minus_100ms',{'shot_time_ms':req['shot_time_ms']-100},{})]
  if record.get('actualDetectionBoxesXywh'):
   boxes=[[x,y,x+w,y+h] for x,y,w,h in record['actualDetectionBoxesXywh']]
   variants += [('actual_box_center',{'bboxes_xyxy':boxes,'point_mode':'center'},{}),('actual_box_bottom',{'bboxes_xyxy':boxes,'point_mode':'bottom_center'}, {})]
  results=[]
  for name,top,params in variants:
   changed=dict(req,**top);changed['overlay_params']=dict(req.get('overlay_params') or {},**params)
   body=solve(changed); shifts=[]
   for p,b in zip(body.get('results',[]),baseline.get('results',[])):
    if p.get('ok') and b.get('ok'): shifts.append(distance(p,b))
   results.append({'name':name,'shifts_m':shifts,'errors':[p.get('error') for p in body.get('results',[]) if not p.get('ok')], 'request_error':body.get('error')})
  pose=None
  with patch.object(svc,'_fetch_log_poses_from_url',lambda u:logs[u]):
   poses,i,used=svc._resolve_nearest_pose_for_request(svc.GeoFromBBoxesRequest(**req),None)
   pose={k:poses[i].get(k) for k in ['lat','lon','alt','agl_h','yaw','pitch','roll','pan','tilt','zoomFactor']}
  output.append({'eventId':record.get('eventId'),'request':req,'baseline':baseline,'selected_pose':pose,'variants':results})
 except Exception as exc:output.append({'eventId':record.get('eventId'),'error':str(exc)})
print(json.dumps({'checked_at':datetime.now(timezone.utc).isoformat(),'source_hashes':{n:hashlib.sha256(Path(svc.__file__).with_name(n).read_bytes()).hexdigest() for n in ['app.py','args_helpers.py','project_kml_to_video_overlay.py']},'records':output,'accuracy_truth_available':False,'database_written':False}))
'''
run = subprocess.run(['ssh','-o','BatchMode=yes','ai-lingxi',
    'docker exec -i -e PYTHONDONTWRITEBYTECODE=1 overlay-api-2 python -c '+shlex.quote(REMOTE)],
    input=json.dumps(records), text=True, capture_output=True, timeout=240)
if run.returncode:
    raise SystemExit(run.stderr[-1500:])
result=json.loads(run.stdout)
out=ROOT/'real-residual-diagnosis-20260915.json'
out.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
print(json.dumps({'output':str(out),'records':len(result['records']), 'source_hashes':result['source_hashes'],
 'reported':[{k:v for k,v in r.items() if k not in ('baseline','request')} for r in result['records'][:1]]},ensure_ascii=False))
