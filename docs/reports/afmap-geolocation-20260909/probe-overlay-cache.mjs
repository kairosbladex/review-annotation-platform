import fs from 'node:fs';
import { execFileSync } from 'node:child_process';

// Executes an isolated Python interpreter, never the running HTTP worker.
const evidence = JSON.parse(fs.readFileSync(new URL('./evidence.json', import.meta.url)));
const source = evidence.sources.find(item => item.url.includes('/task/trail/'));
const response = JSON.parse(execFileSync('curl', [
  '--fail', '--silent', '--show-error', '--max-time', '20', source.url,
], { encoding: 'utf8' }));
if (response.code !== 200 || !Array.isArray(response.data?.trails)) {
  throw new Error('Flight telemetry was not returned');
}
const python = String.raw`
import hashlib
import json
import math
import os
import sys
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import app
import numpy as np

data = json.load(sys.stdin)
def epoch(value):
    return datetime.fromisoformat(value.replace(' ', 'T') + '+08:00').timestamp()

old_time = '2026-09-07 18:09:00'
current_time = data['evidence']['eventTime']
def pose_at(value):
    row = next(row for row in data['trails'] if row['positionTime'] == value)
    pose = json.loads(row['messageInfo'])
    pose['lon'] = pose['lng']
    pose['t_sec'] = epoch(value) - epoch(old_time)
    return pose

old_pose = pose_at(old_time)
current_pose = pose_at(current_time)
state = SimpleNamespace(t0=epoch(old_time))
fetches = []
def controlled_fetch(url):
    fetches.append(url)
    return ([old_pose] if len(fetches) == 1 else [old_pose, current_pose]), state

configured_cache = os.environ.get('OVERLAY_LOG_URL_CACHE', '1')
app._get_cached_log_poses.cache_clear()
with patch.object(app, '_fetch_log_poses_from_url', controlled_fetch):
    request = app.GeoFromBBoxRequest(
        bbox_xywh=[33, 346, 28, 42], image_width=960, image_height=720,
        log_url='https://audit.invalid/growing-flight.log',
        shot_time_ms=epoch('2026-09-07 18:09:02') * 1000,
    )
    app._resolve_nearest_pose_for_request(request, None)
    request.shot_time_ms = epoch(current_time) * 1000
    poses, index, _ = app._resolve_nearest_pose_for_request(request, None)
    stale = poses[index]
    fetches_with_cache = len(fetches)
    stale_age = epoch(current_time) - (state.t0 + stale['t_sec'])
    # Environment override affects this disposable interpreter only.
    with patch.dict(os.environ, {'OVERLAY_LOG_URL_CACHE': '0'}):
        poses, index, _ = app._resolve_nearest_pose_for_request(request, None)
        refreshed = poses[index]
    fresh_age = epoch(current_time) - (state.t0 + refreshed['t_sec'])

def distance(a, b):
    lat1, lon1, lat2, lon2 = map(math.radians, [a[1], a[0], b[1], b[0]])
    h = math.sin((lat2-lat1)/2)**2 + math.cos(lat1)*math.cos(lat2)*math.sin((lon2-lon1)/2)**2
    return 2 * 6371008.8 * math.asin(math.sqrt(h))

origin = [old_pose['lon'], old_pose['lat']]
items = [target for case in data['evidence']['cases'] for target in case['targets']]
target_offsets = [
    {'index': target['index'], 'distanceToOldPoseMeters': distance(origin, target['coordinates'])}
    for target in data['evidence']['cases'][-1]['targets']
]
xy = np.array([
    [math.radians(item['coordinates'][0]-origin[0]) * 6371008.8 * math.cos(math.radians(origin[1])),
     math.radians(item['coordinates'][1]-origin[1]) * 6371008.8]
    for item in items
])
fits = []
for name, ax, ay in [('top_left', 0, 0), ('center', .5, .5), ('bottom_center', .5, 1)]:
    design = np.array([
        [item['imageBox']['x']+ax*item['imageBox']['width'],
         item['imageBox']['y']+ay*item['imageBox']['height'], 1]
        for item in items
    ])
    matrix, _, rank, _ = np.linalg.lstsq(design, xy, rcond=None)
    errors = np.linalg.norm(design @ matrix - xy, axis=1)
    center = np.array([480, 360, 1]) @ matrix
    fits.append({'anchor': name, 'pointCount': len(items), 'rank': int(rank),
                 'rmsMeters': float(np.sqrt(np.mean(errors**2))),
                 'maxResidualMeters': float(errors.max()),
                 'impliedFrameCenterEastNorthFromOldPoseMeters': center.tolist(),
                 'pixelToLocalAffine': matrix.tolist()})

result = {
    'testScope': 'isolated interpreter using deployed source and controlled growing-log fetch',
    'originalHttpRequestRecovered': False,
    'configuredCache': configured_cache,
    'warmRequestTime': '2026-09-07 18:09:02',
    'secondRequestTime': current_time,
    'underlyingFetchesForTwoCachedRequests': fetches_with_cache,
    'cachedSelectedPoseTime': old_time,
    'cachedPoseAgeSeconds': stale_age,
    'cacheDisabledInProbeOnlySelectedPoseAgeSeconds': fresh_age,
    'oldToCurrentAircraftMeters': distance(origin, [current_pose['lon'], current_pose['lat']]),
    'targetsComparedWithOldPose': target_offsets,
    'constantAffineFitsAcrossThreeEvents': fits,
    'sourceHashes': {p: hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in [
        app.__file__, app.pkv.__file__, '/app/overlay_service/args_helpers.py',
    ]},
}
print(json.dumps(result, indent=2))
assert fetches_with_cache == 1 and stale_age == 78, 'Cached stale-pose mechanism not reproduced'
assert fresh_age == 0, 'Fresh data did not restore time selection'
`;
const quote = value => "'" + value.replace(/'/g, "'\\''") + "'";
const output = execFileSync('ssh', [
  '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=10', 'ai-lingxi',
  'docker exec -i overlay-api-2 python -c ' + quote(python),
], {
  input: JSON.stringify({ evidence, trails: response.data.trails }),
  encoding: 'utf8', timeout: 60000,
});
const result = JSON.parse(output);
console.log(JSON.stringify(result, null, 2));
if (process.argv.includes('--save')) {
  fs.writeFileSync(new URL('./overlay-cache-probe.json', import.meta.url), JSON.stringify(result, null, 2) + '\n');
}
