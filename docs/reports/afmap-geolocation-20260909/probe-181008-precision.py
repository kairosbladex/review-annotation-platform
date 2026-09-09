"""Quantify parameter sensitivity for the reported event; no calibration or DB changes."""
from datetime import datetime, timezone
import json
from pathlib import Path
import shlex
import subprocess

root = Path(__file__).resolve().parent
original = json.loads((root / "original-request-181008.json").read_text())
remote = r'''
import asyncio,copy,hashlib,json,math,os,sys
from pathlib import Path
from unittest.mock import patch
from fastapi import HTTPException
import app as service
data=json.load(sys.stdin); original=data['request']
logs=service._fetch_log_poses_from_url(original['log_url'])
variants=[('current',{},{}),
 ('principal_point_zero',{}, {'cx_offset_px':0,'cy_offset_px':0}),
 ('principal_zero_fixed_flip_y',{}, {'cx_offset_px':0,'cy_offset_px':0,'geo_auto_flip_y_by_ray':False,'auto_flip_y_by_ray':False,'flip_y':True}),
 ('current_fixed_flip_y',{}, {'geo_auto_flip_y_by_ray':False,'auto_flip_y_by_ray':False,'flip_y':True}),
 ('scale_neutral',{}, {'fx_scale':1,'fy_scale':1}),
 ('principal_and_scale_neutral',{}, {'cx_offset_px':0,'cy_offset_px':0,'fx_scale':1,'fy_scale':1}),
 ('sensor_height_16_9_crop_hypothesis',{'sensor_h_mm':original['sensor_w_mm']*9/16},{}),
 ('yaw_plus_3deg',{}, {'yaw_offset_deg':3}),('yaw_minus_3deg',{}, {'yaw_offset_deg':-3}),
 ('pitch_plus_1deg',{}, {'pitch_offset_deg':1}),('pitch_minus_1deg',{}, {'pitch_offset_deg':-1}),
 ('pitch_plus_0_001deg',{}, {'pitch_offset_deg':0.001}),('pitch_minus_0_001deg',{}, {'pitch_offset_deg':-0.001}),
 ('pitch_plus_3deg',{}, {'pitch_offset_deg':3}),('pitch_minus_3deg',{}, {'pitch_offset_deg':-3}),
 ('ground_plus_10m',{}, {'ground_h_offset_m':10}),('ground_minus_10m',{}, {'ground_h_offset_m':-10}),
 ('zoom_plus_10pct',{}, {'zoom_scale':1.1}),('zoom_minus_10pct',{}, {'zoom_scale':0.9}),
 ('use_body_pitch',{}, {'image_style_add_uav_pitch':True}),
 ('use_body_roll',{}, {'image_style_use_uav_roll':True})]
boxes=[[x,y,x+w,y+h] for x,y,w,h in data['actualDetectionBoxesXywh']]
variants += [('bbox_top_left_to_center',{'bboxes_xyxy':boxes,'point_mode':'center'},{}),
             ('bbox_top_left_to_bottom_center',{'bboxes_xyxy':boxes,'point_mode':'bottom_center'},{}),
             ('bbox_center_neutral_principal',{'bboxes_xyxy':boxes,'point_mode':'center'}, {'cx_offset_px':0,'cy_offset_px':0}),
             ('bbox_center_16_9_crop',{'bboxes_xyxy':boxes,'point_mode':'center','sensor_h_mm':original['sensor_w_mm']*9/16},{})]
responses=[]; baseline=None
for name,top,params in variants:
 req=dict(original,**top,overlay_params=params,debug=True)
 with patch.object(service,'_fetch_log_poses_from_url',lambda u:logs):
  try:
   response=json.loads(asyncio.run(service.geo_from_bboxes(service.GeoFromBBoxesRequest(**req))).body)
  except HTTPException as exc:
   responses.append({'variant':name,'status':exc.status_code,'error':str(exc.detail)});continue
 if baseline is None:baseline=response
 points=[]
 for point,old in zip(response['results'],baseline['results']):
  if not point['ok']:
   points.append({'index':point['index'],'ok':False,'error':point.get('error')});continue
  east=math.radians(point['lon']-old['lon'])*6371008.8*math.cos(math.radians(old['lat']))
  north=math.radians(point['lat']-old['lat'])*6371008.8
  points.append({'index':point['index'],'ok':True,'coordinates':[point['lon'],point['lat']],
                 'eastChangeMeters':east,'northChangeMeters':north,'displacementFromCurrentMeters':math.hypot(east,north),
                 'anchor':point['anchor_uv'],'quality':point['quality'],'intrinsics':point['debug_info']['intrinsics'],
                 'projectionFlags':{k:point['debug_info'][k] for k in ['flip_y_effective','flip_x_effective','attitude_mode_used','yaw_used']},
                 'rayWorld':point['debug_info']['ray_world']})
 responses.append({'variant':name,'status':200,'requestOverrides':top,'overlayParams':params,'points':points})
with patch.object(service,'_fetch_log_poses_from_url',lambda u:logs):
 poses,index,used=service._resolve_nearest_pose_for_request(service.GeoFromBBoxesRequest(**original),image_bytes=None)
pose={k:poses[index].get(k) for k in ['t_sec','lat','lon','alt','agl_h','yaw','pitch','roll','pan','tilt','zoomFactor']}
print(json.dumps({'sourceSha256':hashlib.sha256(Path(service.__file__).read_bytes()).hexdigest(),
                  'supportSourceSha256':{n:hashlib.sha256(Path(service.__file__).with_name(n).read_bytes()).hexdigest() for n in ['args_helpers.py','project_kml_to_video_overlay.py']},
                  'environmentCalibration':{k:os.environ.get(k) for k in ['OVERLAY_SINGLE_IMAGE_CX_OFFSET_PX','OVERLAY_SINGLE_IMAGE_CY_OFFSET_PX','OVERLAY_SINGLE_IMAGE_FX_SCALE','OVERLAY_SINGLE_IMAGE_FY_SCALE']},
                  'selectedPose':pose,'selectedTime':used,'baselineResponse':baseline,'variants':responses}))
'''
run = subprocess.run(["ssh", "-o", "BatchMode=yes", "ai-lingxi",
                      "docker exec -i -e PYTHONDONTWRITEBYTECODE=1 overlay-api-2 python -c " + shlex.quote(remote)],
                     input=json.dumps(original), text=True, capture_output=True, timeout=90)
if run.returncode:
    print(run.stderr[-2000:]); raise SystemExit(run.returncode)
output = json.loads(run.stdout)
assert output["sourceSha256"] == "e16f0a97beb3b2109a3e595e5b251b5488ccd57848920d15c7964bf3f9dfe22e"
backup = json.loads((root / "verified-flight-correction-20260909-v2.json").read_text())
row = next(r for r in backup["rows"] if r["busi_id"] == original["eventId"])
stored = json.loads(row["replacement"])
for p in output["baselineResponse"]["results"]:
    feature = next(f for f in stored if f["properties"]["geolocation"]["original_request_index"] == p["index"])
    assert feature["geometry"]["coordinates"][:2] == [p["lon"], p["lat"]]
output.update(eventId=original["eventId"], checkedAt=datetime.now(timezone.utc).isoformat(),
              scope="read-only parameter displacement; no ground truth supplied; displacement is not accuracy improvement",
              serviceOrDataChanged=False, calibrationValidated=False)
(root / "precision-181008-parameter-probe.json").write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n")
print(json.dumps({"selectedPose":output["selectedPose"],"environmentCalibration":output["environmentCalibration"],
                  "variants":[{"name":v["variant"],"shiftsMeters":[round(p.get("displacementFromCurrentMeters",0),3) for p in v.get("points",[])]} for v in output["variants"]]},ensure_ascii=False,indent=2))
