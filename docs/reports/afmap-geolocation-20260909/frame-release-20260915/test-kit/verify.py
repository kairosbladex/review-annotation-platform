"""测试环境只读定位回放；不调用事件创建、重算写库或任务下发接口。"""
import argparse,json,math,urllib.request
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--base-url',default='http://192.168.130.238:8181');a=p.parse_args();root=Path(__file__).resolve().parent
for case in json.loads((root/'expected.json').read_text()):
 for mode,v in case['variants'].items():
  req=urllib.request.Request(a.base_url.rstrip('/')+'/v1/geo/from-bboxes',data=(root/v['request_file']).read_bytes(),headers={'Content-Type':'application/json'})
  with urllib.request.urlopen(req,timeout=30) as response:body=json.load(response)
  assert len(body['results'])==8 and all(q['ok'] for q in body['results']),body
  for q,e in zip(body['results'],v['expected_points']):
   assert math.hypot((q['lon']-e['lon'])*91000,(q['lat']-e['lat'])*111000)<.02,'冻结回放坐标不一致'
  if mode=='native':
   assert body['metadata']['calculation_version'].endswith('+frame-geometry-v1')
   assert body['metadata']['frame_geometry']['telemetry_zoom_applied'] is False
  print(case['event_id'][:5],mode,'PASS',len(body['results']),'points')
print('全部6组回放通过；这是冻结样本回归，不代表所有地点均降低79%。')
