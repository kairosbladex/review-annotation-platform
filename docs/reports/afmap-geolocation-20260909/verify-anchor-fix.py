"""在测试环境v2隔离解释器回放真实日志，验证修复后的框、索引及拒绝行为。"""
import json
import shlex
import subprocess
from pathlib import Path

root = Path(__file__).resolve().parent
request = json.loads((root / 'original-request-181008.json').read_text())['request']
request.update(bboxes_xyxy=[[72, 228, 113, 324], [-1, -1, -1, -1], [844, 524, 912, 633]], debug=True)
remote = r'''
import asyncio, hashlib, json, sys
from pathlib import Path
import app
req = app.GeoFromBBoxesRequest(**json.load(sys.stdin))
response = json.loads(asyncio.run(app.geo_from_bboxes(req)).body)
points = response['results']
assert len(points) == 3 and response['count'] == 3, response
assert [p['index'] for p in points] == [0, 1, 2], points
assert points[0]['ok'] and points[2]['ok'], points
assert not points[1]['ok'] and points[1].get('error'), points[1]
assert points[1].get('lon') is None and points[1].get('lat') is None, points[1]
for p, expected in [(points[0], (92.5, 276)), (points[2], (878, 578.5))]:
    anchor = p['anchor_uv']
    assert (anchor['u'], anchor['v']) == expected, anchor
print(json.dumps({'passed':True, 'source_sha256':hashlib.sha256(Path(app.__file__).read_bytes()).hexdigest(),
    'results':points,'metadata':response.get('metadata'), 'absolute_accuracy_verified':False}, ensure_ascii=False))
'''
run = subprocess.run(['ssh', '-o', 'BatchMode=yes', 'ai-lingxi',
    'docker exec -i -e PYTHONDONTWRITEBYTECODE=1 overlay-api-2 python -c ' + shlex.quote(remote)],
    input=json.dumps(request), text=True, capture_output=True, timeout=60)
if run.returncode:
    raise SystemExit(run.stderr[-3000:])
result = json.loads(run.stdout)
(root / 'anchor-fix-live-replay-20260915.json').write_text(json.dumps(result, ensure_ascii=False, indent=2)+'\n')
print(json.dumps({'passed':result['passed'], 'source_sha256':result['source_sha256'],
    'points':[{'index':p['index'], 'ok':p['ok'], 'anchor':p.get('anchor_uv'), 'error':p.get('error')} for p in result['results']],
    'absolute_accuracy_verified':False},ensure_ascii=False))
