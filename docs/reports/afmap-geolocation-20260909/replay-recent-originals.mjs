import fs from 'node:fs';
import { execFileSync } from 'node:child_process';

const root = new URL('./', import.meta.url);
const input = JSON.parse(fs.readFileSync(new URL('recent-original-requests.json', root)));
const python = String.raw`
import asyncio,json,math,sys
from unittest.mock import patch
import app
from fastapi import HTTPException

data=json.load(sys.stdin)
logs={}
results=[]
def distance(a,b):
    lon1,lat1,lon2,lat2=map(math.radians,[a[0],a[1],b[0],b[1]])
    h=math.sin((lat2-lat1)/2)**2+math.cos(lat1)*math.cos(lat2)*math.sin((lon2-lon1)/2)**2
    return 2*6371008.8*math.asin(math.sqrt(min(1,max(0,h))))
for rec in data['records']:
    req=dict(rec['request'],debug=True)
    url=req['log_url']
    if url not in logs:
        logs[url]=app._fetch_log_poses_from_url(url)
    with patch.object(app,'_fetch_log_poses_from_url',lambda u:logs[u]):
        try:
            response=json.loads(asyncio.run(app.geo_from_bboxes(app.GeoFromBBoxesRequest(**req))).body)
        except HTTPException as exc:
            results.append({'shotTimeMs':req['shot_time_ms'],'logUrl':url,'status':exc.status_code,'error':str(exc.detail)})
            continue
    camera=response['debug_context']['camera_pose']
    coords=[camera['lon'],camera['lat']]
    points=[]
    for point in response['results']:
        if not point['ok']:
            points.append({'index':point['index'],'ok':False,'error':point.get('error')})
            continue
        debug=point['debug_info']
        old=next(p for p in rec['response']['results'] if p['index']==point['index'])
        points.append({'index':point['index'],'ok':True,'coordinates':[point['lon'],point['lat'],point['alt']],
                       'originalCoordinatesDifferenceDegrees':max(abs(point[k]-old[k]) for k in ['lon','lat']) if old['ok'] else None,
                       'distanceFromCameraMeters':round(distance(coords,[point['lon'],point['lat']]),3),
                       'rayWorldZ':debug['ray_world']['z'],'poseTiltRaw':debug['pose_tilt_raw'],
                       'groundHeight':debug['ground_h'],'intrinsics':debug['intrinsics']})
    results.append({'shotTimeMs':req['shot_time_ms'],'logUrl':url,'status':200,
                    'selected':response['debug_context']['used_target'],'cameraPose':camera,
                    'count':response['count'],'okCount':response['ok_count'],'points':points})
print(json.dumps({'scope':'isolated read-only replay with current service code and archived logs; no event updates',
                  'results':results},indent=2))
`;
const quote = value => "'" + value.replace(/'/g, "'\\''") + "'";
const output = JSON.parse(execFileSync('ssh', ['-o', 'BatchMode=yes', '-o', 'ConnectTimeout=10', 'ai-lingxi',
  'docker exec -i -e PYTHONDONTWRITEBYTECODE=1 overlay-api-2 python -c ' + quote(python)],
{ input: JSON.stringify(input), encoding: 'utf8', timeout: 60000 }));
fs.writeFileSync(new URL('recent-original-replay.json', root), JSON.stringify(output, null, 2) + '\n');
console.log(JSON.stringify(output.results.map(r => ({ shotTimeMs: r.shotTimeMs, status: r.status,
  deltaSeconds: r.selected?.pose_time_delta_sec, okCount: r.okCount,
  tilt: r.points?.find(p => p.ok)?.poseTiltRaw,
  maxRangeMeters: Math.max(...(r.points ?? []).filter(p => p.ok).map(p => p.distanceFromCameraMeters)),
  maxOriginalDifferenceDeg: Math.max(...(r.points ?? []).filter(p => p.ok).map(p => p.originalCoordinatesDifferenceDegrees)),
  error: r.error })), null, 2));
