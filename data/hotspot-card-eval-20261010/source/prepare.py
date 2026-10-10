import collections, csv, hashlib, json
from pathlib import Path
ROOT=Path('/data/workspace/jzheng/hotspot-card-eval-20261010')
AUTO=Path('/data/workspace/jzheng/luna-prompt-autotune')
SNAP=Path('/data/workspace/jzheng/codex-ktd-real-monitoring-20260924')
CLEAN=Path('/data/workspace/jzheng/hotspot-relabel-20261010/clean')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def write(name,obj): (ROOT/name).write_text(json.dumps(obj,ensure_ascii=False,indent=2)+'\n')
def iou(a,b):
 x=max(0,min(a[2],b[2])-max(a[0],b[0]));y=max(0,min(a[3],b[3])-max(a[1],b[1]));v=x*y
 return v/((a[2]-a[0])*(a[3]-a[1])+(b[2]-b[0])*(b[3]-b[1])-v)
boxes=json.loads((ROOT/'inputs/boxes.json').read_text())['boxes']
rows=list(csv.DictReader((ROOT/'inputs/annotations.csv').open(encoding='utf-8-sig')))
labels={r['key']:r['label'] for r in rows};assert len(labels)==len(rows)==len(boxes)==93
assert set(labels)=={b['key'] for b in boxes};assert set(labels.values())<={'是光斑','不是光斑','看不清'}
items=json.loads((AUTO/'assets/items.json').read_text());units=[];training=[]
for eid in sorted({b['event_id'] for b in boxes}):
 image=SNAP/'work/annotated'/f'{eid}.jpg';clean=CLEAN/f'{eid}.jpg';h=sha(image)
 used=[u for u in items if u['sha256']==h]
 bs=[dict(b) for b in boxes if b['event_id']==eid]
 for b in bs:
  matches=[{'unit':u['unit'],'split':u['split'],'key':c['key']} for u in used for c in u['candidates'] if c['class_name']=='光斑' and iou(c['bbox'],b['screenshot_bbox'])>=.99]
  b['exposure_matches']=matches
  if matches:training.append({'key':b['key'],'event_id':eid,'label':labels[b['key']],'bbox':b['bbox'],'screenshot_bbox':b['screenshot_bbox']})
 units.append({'event_id':eid,'image':str(image),'clean':str(clean),'screenshot_sha256':h,'clean_sha256':sha(clean),'split':'development' if used else 'holdout','exposed_units':[{'unit':u['unit'],'split':u['split']} for u in used],'boxes':bs})
# Identical screenshots always stay together; reported clusters merge duplicate events.
assert not ({u['screenshot_sha256'] for u in units if u['split']=='development'}&{u['screenshot_sha256'] for u in units if u['split']=='holdout'})
write('split.json',units);write('development-labels.json',training)
for v in ('p001','p002'):
 d=json.loads((AUTO/f'state/prompts/{v}.json').read_text());write(f'source/{v}-hotspot.json',{'version':v,'card':d['cards']['光斑']})
write('provenance.json',{'annotations_sha256':sha(ROOT/'inputs/annotations.csv'),'autotune_items_sha256':sha(AUTO/'assets/items.json'),'source_commit':'b2ca913ee365c0a64ebda4e2ca28604b79d801ed','label_package_commit':'b5e6c98ead46881bd4428379ff943c5be5b6bbab','rounds':{v:json.loads((AUTO/f'state/rounds/{v}.json').read_text()) for v in ('p001','p002')},'metrics_summary':{v:{k:json.loads((AUTO/f'state/metrics/{v}.json').read_text()).get(k) for k in ('units','failed_units')} for v in ('p001','p002')},'exposure_policy':'All images in autotune assets are conservatively exposed, including unlabeled and duplicates; card revision only reads IoU>=0.99 matched boxes. No historical run directory remains.'})
print('eligible revision boxes',len(training),'labels',dict(collections.Counter(x['label'] for x in training)))
print('split counts',dict(collections.Counter(u['split'] for u in units)))
