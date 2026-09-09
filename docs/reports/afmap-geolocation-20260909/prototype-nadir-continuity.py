"""Isolated nadir continuity prototype, preserving the currently verified -90 degree convention."""
from datetime import datetime, timezone
import json
from pathlib import Path
import shlex
import subprocess

root = Path(__file__).resolve().parent
data = {"event": json.loads((root / "original-request-181008.json").read_text()),
        "flight": json.loads((root / "same-task-original-requests.json").read_text())["records"]}
remote = r'''
import asyncio,hashlib,json,math,sys
from pathlib import Path
from unittest.mock import patch
import numpy as np
import app as service
data=json.load(sys.stdin);logs={}
def rotation(yaw_deg,pitch_deg,roll_deg,yaw_sign,pitch_offset):
 psi=math.radians(yaw_deg*yaw_sign); th=math.radians(pitch_deg+pitch_offset); phi=math.radians(roll_deg)
 # 保留当前-90度分支的像面约定；用yaw连续定义right，避免cross(forward,up)在垂直处退化。
 forward=np.array([math.sin(psi)*math.cos(th),math.cos(psi)*math.cos(th),math.sin(th)])
 right=np.array([-math.cos(psi),math.sin(psi),0.0])
 down=np.cross(forward,right)
 c,s=math.cos(phi),math.sin(phi)
 return np.column_stack([c*right+s*down,-s*right+c*down,forward])
def solve(req,prototype):
 url=req['log_url']
 if url not in logs:logs[url]=service._fetch_log_poses_from_url(url)
 original_rotation=service.pkv.build_r_wc
 with patch.object(service,'_fetch_log_poses_from_url',lambda u:logs[u]),patch.object(service.pkv,'build_r_wc',rotation if prototype else original_rotation):
  return json.loads(asyncio.run(service.geo_from_bboxes(service.GeoFromBBoxesRequest(**req))).body)
def dist(a,b):
 e=math.radians(a['lon']-b['lon'])*6371008.8*math.cos(math.radians(b['lat']))
 n=math.radians(a['lat']-b['lat'])*6371008.8
 return math.hypot(e,n)
req=dict(data['event']['request'],debug=True)
base=solve(req,False)
fixed={'geo_auto_flip_y_by_ray':False,'auto_flip_y_by_ray':False,'flip_y':True}
checks=[]
for delta in [-1,-0.001,0,0.001,1]:
 q=dict(req,overlay_params=dict(fixed,pitch_offset_deg=delta))
 old=solve(q,False);new=solve(q,True)
 old_shift=[dist(p,b) for p,b in zip(old['results'],base['results'])]
 new_shift=[dist(p,b) for p,b in zip(new['results'],base['results'])]
 assert all(p['ok'] for p in new['results'])
 if abs(delta)<=0.001:assert max(new_shift)<0.01
 checks.append({'pitchOffsetDegrees':delta,'oldShiftMeters':old_shift,'prototypeShiftMeters':new_shift,
                'prototypeRoundTripPixels':[p['debug_info']['round_trip']['err_px'] for p in new['results']],
                'prototypeQualityEstimateMeters':[p['quality']['estimated_sensitivity_m'] for p in new['results']]})
baseline_checks=[]
for record in data['flight']:
 q=dict(record['request'],debug=True)
 old=solve(q,False);new=solve(dict(q,overlay_params=fixed),True)
 assert len(old['results'])==len(new['results'])
 change=max(dist(a,b) for a,b in zip(old['results'],new['results']))
 assert change<0.001,('prototype changes verified nadir coordinates',record['eventId'],change)
 baseline_checks.append({'eventId':record['eventId'],'maximumCoordinateChangeMeters':change})
max_orthogonality=0.0;min_det=1.0
for yaw in [0,45,90,180,270,359.99]:
 for pitch in [-90.001,-90,-89.999]:
  for roll in [-5,0,5]:
   r=rotation(yaw,pitch,roll,1,0)
   max_orthogonality=max(max_orthogonality,float(np.max(np.abs(r.T@r-np.eye(3)))))
   min_det=min(min_det,float(np.linalg.det(r)))
assert max_orthogonality<1e-12 and min_det>0.999999999
print(json.dumps({'result':'PASS','sourceSha256':hashlib.sha256(Path(service.__file__).read_bytes()).hexdigest(),
 'scope':'isolated prototype on 12 verified nadir frames only; no worker source or environment changes',
 'fixedImageOrientation':fixed,'angularPerturbationChecks':checks,'verifiedFlightChecks':baseline_checks,
 'rotationOrthogonalityMaxError':max_orthogonality,'rotationDeterminantMin':min_det,
 'groundTruthAccuracyVerified':False,'obliqueCompatibilityVerified':False,'deployablePatch':False}))
'''
run = subprocess.run(["ssh", "-o", "BatchMode=yes", "ai-lingxi",
                      "docker exec -i -e PYTHONDONTWRITEBYTECODE=1 overlay-api-2 python -c " + shlex.quote(remote)],
                     input=json.dumps(data), text=True, capture_output=True, timeout=90)
if run.returncode:
    print(run.stderr[-3000:]); raise SystemExit(run.returncode)
output = json.loads(run.stdout)
output["checkedAt"] = datetime.now(timezone.utc).isoformat()
(root / "nadir-continuity-prototype-result.json").write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n")
print(json.dumps(output, ensure_ascii=False, indent=2))
