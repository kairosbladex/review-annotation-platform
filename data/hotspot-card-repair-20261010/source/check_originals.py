import json,pathlib,hashlib
from input_views import project_box
ROOT=pathlib.Path(__file__).resolve().parents[1];SNAP=pathlib.Path('/data/workspace/jzheng/codex-ktd-real-monitoring-20260924')
from PIL import Image
records={kind:{r['event_id']:r for r in [json.loads(l) for l in (SNAP/f'output/{kind}_matches.jsonl').read_text().splitlines()]} for kind in ('still','video')}
out=[]
for u in json.loads((ROOT/'split.json').read_text()):
 sw,sh=Image.open(u['image']).size;dw,dh=Image.open(u['clean']).size
 for key,path in [('screenshot',u['image']),('clean',u['clean'])]:assert hashlib.sha256(pathlib.Path(path).read_bytes()).hexdigest()==u[key+'_sha256']
 for b in u['boxes']:
  r=records[b['source_kind']][u['event_id']];mapped=project_box(b['screenshot_bbox'],r['homography_annotated_to_source'],(sw,sh),(dw,dh));assert mapped['complete'],b['key'];assert max(abs(a-c) for a,c in zip(mapped['bbox'],b['bbox']))<1e-10
  out.append({'key':b['key'],**mapped})
(ROOT/'original-input-audit.json').write_text(json.dumps(out,indent=2)+'\n');print('original inputs: all',len(out),'boxes complete and mapping verified')
