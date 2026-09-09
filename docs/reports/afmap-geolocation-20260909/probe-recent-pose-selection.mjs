import fs from 'node:fs';
import { execFileSync } from 'node:child_process';

const root = new URL('./', import.meta.url);
const record = JSON.parse(fs.readFileSync(new URL('recent-original-requests.json', root))).records
  .find(r => r.request.shot_time_ms === 1788934376000);
if (!record) throw new Error('Recent original request was not recovered');
const python = String.raw`
import asyncio,json,sys
from unittest.mock import patch
import app
from fastapi import HTTPException
record=json.load(sys.stdin)
req=dict(record['request'],debug=True)
poses,state=app._fetch_log_poses_from_url(req['log_url'])
matches=[]
for index,pose in enumerate(poses):
    time_ms=(state.t0+pose['t_sec'])*1000
    if not req['shot_time_ms']-5500 <= time_ms <= req['shot_time_ms']+1000:
        continue
    with patch.object(app,'_fetch_log_poses_from_url',lambda u:(poses[:index+1],state)):
        try:
            response=json.loads(asyncio.run(app.geo_from_bboxes(app.GeoFromBBoxesRequest(**req))).body)
        except HTTPException:
            continue
    if response['ok_count'] != record['response']['ok_count']:
        continue
    error=max(abs(a[k]-b[k]) for a,b in zip(response['results'],record['response']['results']) for k in ['lon','lat'])
    matches.append({'coordinateDifferenceDegrees':error,'prefixEndIndex':index,
                    'selected':response['debug_context']['used_target'],
                    'cameraPose':response['debug_context']['camera_pose'],
                    'points':[{k:p[k] for k in ['index','lon','lat','alt']} for p in response['results']]})
best=min(matches,key=lambda r:r['coordinateDifferenceDegrees']) if matches else None
print(json.dumps({'shotTimeMs':req['shot_time_ms'],
                  'scope':'current fixed code with reconstructed log prefixes; no updates',
                  'historicalWorkerCacheRecovered':False,'candidates':len(matches),
                  'exactOriginalMatch':best is not None and best['coordinateDifferenceDegrees']==0,
                  'bestMatch':best},indent=2))
`;
const quote = value => "'" + value.replace(/'/g, "'\\''") + "'";
const output = execFileSync('ssh', ['-o', 'BatchMode=yes', '-o', 'ConnectTimeout=10', 'ai-lingxi',
  'docker exec -i -e PYTHONDONTWRITEBYTECODE=1 overlay-api-2 python -c ' + quote(python)],
{ input: JSON.stringify(record), encoding: 'utf8', timeout: 60000 });
const result = JSON.parse(output);
fs.writeFileSync(new URL('recent-pose-selection-probe.json', root), JSON.stringify(result, null, 2) + '\n');
console.log(output);
