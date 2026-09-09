import fs from 'node:fs';
import { execFileSync } from 'node:child_process';

const record = JSON.parse(fs.readFileSync(new URL('./original-request-181008.json', import.meta.url)));
const baselineRepo = process.argv.find(value => value.startsWith('--baseline-repo='))?.slice('--baseline-repo='.length);
if (!baselineRepo) throw new Error('--baseline-repo must name the overlay fix checkout containing baseline 7d34ad8');
const baselineSource = execFileSync('git', ['show', '7d34ad8:app.py'], { cwd: baselineRepo, encoding: 'utf8' });
const python = String.raw`
import asyncio
import hashlib
import json
import sys
from pathlib import Path
from types import ModuleType
from unittest.mock import patch

import app as fixed
from fastapi import HTTPException

data = json.load(sys.stdin)
source = data['baselineSource']
assert hashlib.sha256(source.encode()).hexdigest() == 'c4a652921da0dbd125f665654abec8a03f528b5a9874d8fa41d9f79d8d0f65fe'
baseline = ModuleType('original_event_baseline')
baseline.__file__ = fixed.__file__
sys.modules[baseline.__name__] = baseline
exec(compile(source, baseline.__file__, 'exec'), baseline.__dict__)
record = data['record']
request = dict(record['request'], debug=True)
poses, state = fixed._fetch_log_poses_from_url(request['log_url'])
matches = []

def run(module):
    response = asyncio.run(module.geo_from_bboxes(module.GeoFromBBoxesRequest(**request)))
    return json.loads(response.body)

def coordinate_error(result):
    assert result['ok_count'] == len(record['originalResults'])
    return max(abs(actual[key] - expected[key])
               for actual, expected in zip(result['results'], record['originalResults'])
               for key in ['lon', 'lat'])

# Recover a reproducing snapshot from the archived log, not live worker memory.
for index, pose in enumerate(poses):
    absolute_time = state.t0 + pose['t_sec']
    if not 1788775736 <= absolute_time <= 1788775742.261:
        continue
    baseline._get_cached_log_poses.cache_clear()
    with patch.object(baseline, '_fetch_log_poses_from_url', lambda url: (poses[:index + 1], state)):
        result = run(baseline)
    matches.append((coordinate_error(result), index, result))
assert matches, 'Archived log contains no pose near the initial geolocation request'
error, old_index, reproduced = min(matches, key=lambda item: item[0])
assert error < 1e-10, f'Original response was not reproduced: {error}'

def compare(module):
    downloads = []
    def fetch(url):
        downloads.append(url)
        return (poses[:old_index + 1] if len(downloads) == 1 else poses), state
    module._get_cached_log_poses.cache_clear()
    with patch.object(module, '_fetch_log_poses_from_url', fetch):
        module._get_cached_log_poses(request['log_url'])
        result = run(module)
    return {'downloads': len(downloads), 'response': result}

old = compare(baseline)
new = compare(fixed)
with patch.object(fixed, '_fetch_log_poses_from_url', lambda url: (poses[:old_index + 1], state)):
    try:
        run(fixed)
        stale_status = 200
    except HTTPException as exc:
        stale_status = exc.status_code
assert old['downloads'] == 1 and coordinate_error(old['response']) < 1e-10
assert new['downloads'] == 2 and new['response']['ok_count'] == 2
assert new['response']['debug_context']['used_target']['pose_time_delta_sec'] < 0.02
assert stale_status == 400
print(json.dumps({
    'scope': 'isolated interpreter; archived original HTTP input and flight log; no event writes',
    'eventId': record['eventId'],
    'originalRequestRecovered': True,
    'historicalWorkerCacheRecovered': False,
    'reconstructedOldPoseIndex': old_index,
    'reconstructedOldPoseTimeMs': (state.t0 + poses[old_index]['t_sec']) * 1000,
    'maxOriginalCoordinateDifferenceDegrees': error,
    'baselineSha256': hashlib.sha256(source.encode()).hexdigest(),
    'deployedSha256': hashlib.sha256(Path(fixed.__file__).read_bytes()).hexdigest(),
    'baseline': old,
    'fixed': new,
    'staleSourceStatus': stale_status,
    'result': 'PASS',
}, indent=2))
`;
const quote = value => "'" + value.replace(/'/g, "'\\''") + "'";
const output = execFileSync('ssh', [
  '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=10', 'ai-lingxi',
  'docker exec -i -e PYTHONDONTWRITEBYTECODE=1 overlay-api-2 python -c ' + quote(python),
], { input: JSON.stringify({ record, baselineSource }), encoding: 'utf8', timeout: 60000 });
const result = JSON.parse(output);
console.log(JSON.stringify(result, null, 2));
if (process.argv.includes('--save')) {
  fs.writeFileSync(new URL('./original-event-replay.json', import.meta.url), JSON.stringify(result, null, 2) + '\n');
}
