import assert from 'node:assert/strict';
import fs from 'node:fs';
import { execFileSync } from 'node:child_process';

const root = new URL('./', import.meta.url);
const records = JSON.parse(fs.readFileSync(new URL('same-task-original-requests.json', root))).records;
const baselineRepo = process.argv.find(arg => arg.startsWith('--baseline-repo='))?.slice('--baseline-repo='.length);
assert(baselineRepo);
const baselineSource = execFileSync('git', ['show', '7d34ad8:app.py'], { cwd: baselineRepo, encoding: 'utf8' });
const python = String.raw`
import asyncio, hashlib, json, math, sys
from pathlib import Path
from types import ModuleType
from unittest.mock import patch
import app as fixed

data=json.load(sys.stdin)
baseline=ModuleType('task_original_baseline')
baseline.__file__=fixed.__file__
sys.modules[baseline.__name__]=baseline
assert hashlib.sha256(data['baselineSource'].encode()).hexdigest() == 'c4a652921da0dbd125f665654abec8a03f528b5a9874d8fa41d9f79d8d0f65fe'
exec(compile(data['baselineSource'],baseline.__file__,'exec'),baseline.__dict__)
records=data['records']
assert len({r['request']['log_url'] for r in records}) == 1
poses,state=fixed._fetch_log_poses_from_url(records[0]['request']['log_url'])
old_index=218
old_ms=(state.t0+poses[old_index]['t_sec'])*1000
assert abs(old_ms-1788775738015)<0.1

def call(module,request):
    return json.loads(asyncio.run(module.geo_from_bboxes(module.GeoFromBBoxesRequest(**request))).body)

def distance(a,b):
    lon1,lat1,lon2,lat2=map(math.radians,[a[0],a[1],b[0],b[1]])
    h=math.sin((lat2-lat1)/2)**2+math.cos(lat1)*math.cos(lat2)*math.sin((lon2-lon1)/2)**2
    return 2*6371008.8*math.asin(math.sqrt(min(1,max(0,h))))

results=[]
for record in records:
    request=dict(record['request'],debug=True)
    baseline._get_cached_log_poses.cache_clear()
    with patch.object(baseline,'_fetch_log_poses_from_url',lambda url:(poses[:old_index+1],state)):
        old=call(baseline,request)
    with patch.object(fixed,'_fetch_log_poses_from_url',lambda url:(poses,state)):
        new=call(fixed,request)
    expected=record['originalResults']
    assert old['ok_count'] == len(expected)
    error=max(abs(a[k]-b[k]) for a,b in zip(old['results'],expected) for k in ['lon','lat'])
    assert error < 1e-10, (record['eventId'],error)
    assert new['ok_count'] == len(expected)
    selected=new['debug_context']['used_target']
    assert selected['time_validation_applied'] and selected['pose_time_delta_sec'] <= 0.5
    new_points=[{'index':p['index'],'coordinates':[p['lon'],p['lat'],p['alt']],
                 'distanceFromMainMeters':round(distance(record['mainCoordinates'],[p['lon'],p['lat']]),3)}
                for p in new['results']]
    results.append({'eventId':record['eventId'],'eventTime':record['eventTime'],
                    'originalCoordinateDifferenceDegrees':error,'oldPoseTimeMs':old_ms,
                    'oldPoseDeltaSeconds':round(abs(request['shot_time_ms']-old_ms)/1000,3),
                    'selected':selected,'newCameraPose':new['debug_context']['camera_pose'],
                    'oldMaxDistanceFromMainMeters':round(max(distance(record['mainCoordinates'],[p['lon'],p['lat']]) for p in old['results']),3),
                    'newMaxDistanceFromMainMeters':max(p['distanceFromMainMeters'] for p in new_points),
                    'newPoints':new_points})
print(json.dumps({'scope':'read-only isolated replay; no database updates',
                  'baselineSha256':hashlib.sha256(data['baselineSource'].encode()).hexdigest(),
                  'runtimeSha256':hashlib.sha256(Path(fixed.__file__).read_bytes()).hexdigest(),
                  'historicalWorkerCacheRecovered':False,'eventsReplayed':len(results),
                  'exactOriginalMatches':sum(r['originalCoordinateDifferenceDegrees']==0 for r in results),
                  'oldPoseOverFiveSecondsEvents':sum(r['oldPoseDeltaSeconds']>5 for r in results),
                  'results':results,'verdict':'PASS'},indent=2))
`;
const quote = value => "'" + value.replace(/'/g, "'\\''") + "'";
const result = JSON.parse(execFileSync('ssh', ['-o', 'BatchMode=yes', '-o', 'ConnectTimeout=10', 'ai-lingxi',
  'docker exec -i -e PYTHONDONTWRITEBYTECODE=1 overlay-api-2 python -c ' + quote(python)],
{ input: JSON.stringify({ baselineSource, records }), encoding: 'utf8', timeout: 60000 }));
fs.writeFileSync(new URL('same-task-replay.json', root), JSON.stringify(result, null, 2) + '\n');
console.log(JSON.stringify({ verdict: result.verdict, eventsReplayed: result.eventsReplayed,
  exactOriginalMatches: result.exactOriginalMatches, oldPoseOverFiveSecondsEvents: result.oldPoseOverFiveSecondsEvents,
  results: result.results.map(({ eventTime, oldPoseDeltaSeconds, oldMaxDistanceFromMainMeters, newMaxDistanceFromMainMeters }) =>
    ({ eventTime, oldPoseDeltaSeconds, oldMaxDistanceFromMainMeters, newMaxDistanceFromMainMeters })) }, null, 2));
