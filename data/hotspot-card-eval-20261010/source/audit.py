"""Verify archived results independently without model calls."""
import collections,csv,hashlib,importlib.util,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def read(p):return json.loads((ROOT/p).read_text())
def sha(p):return hashlib.sha256((ROOT/p).read_bytes()).hexdigest()
for name,h in read('export-manifest.json').items():assert sha(name)==h,name
for name,h in read('freeze.json')['files'].items():assert sha(name)==h,name
spec=importlib.util.spec_from_file_location('production',ROOT/'source/production_worker.py');prod=importlib.util.module_from_spec(spec);spec.loader.exec_module(prod)
units={u['event_id']:u for u in read('split.json')};cards=read('cards.json');common=prod.load_prompt('p000')['common']
with (ROOT/'inputs/annotations.csv').open(encoding='utf-8-sig') as f:labels={r['key']:{'是光斑':'true','不是光斑':'false','看不清':'uncertain'}[r['label']] for r in csv.DictReader(f)}
ids=set();counts={};attempts=collections.Counter()
for stage in ['development','holdout']:
 files=list((ROOT/f'runs/{stage}/out').glob('*.json'));schedule=read(f'runs/{stage}/schedule.json');assert len(files)==len(schedule)
 seen=set();metrics={}
 for p in files:
  r=json.loads(p.read_text());key=(r['event_id'],r['mode'],r['card'],r['repeat']);assert key not in seen;seen.add(key);u=units[r['event_id']];assert u['split']==stage and r['success']
  cs=[{'target_index':i,'class_name':'光斑','bbox':b['screenshot_bbox' if r['mode']=='screenshot' else 'bbox']} for i,b in enumerate(u['boxes'])]
  assert r['candidates']==cs and r['keys']==[b['key'] for b in u['boxes']]
  doc={'version':'p000-pure-class-cards','common':common,'cards':{} if r['card']=='p000' else {'光斑':cards[r['card']]}}
  assert r['prompt']==prod.build_prompt(doc,cs) and r['prompt_sha256']==hashlib.sha256(r['prompt'].encode()).hexdigest()
  assert r['image_sha256']==u['screenshot_sha256' if r['mode']=='screenshot' else 'clean_sha256']
  items=r['result']['results'];assert len(items)==len(cs) and {x['target_index'] for x in items}==set(range(len(cs)))
  assert r['result']['execution_id'] not in ids;ids.add(r['result']['execution_id']);attempts.update(a['status'] for a in r['attempts'])
  m=metrics.setdefault((r['mode'],r['card'],r['repeat']),collections.Counter())
  for x in items:
   y=labels[r['keys'][x['target_index']]];v=x['decision'];assert v in ['true','false','uncertain'];m['n']+=1;m['correct']+=y==v;m['false_reject']+=y=='true' and v!='true';m['explicit_false_reject']+=y=='true' and v=='false';m['false_accept']+=y=='false' and v=='true';m['uncertain']+=v=='uncertain'
 assert seen=={(j['event_id'],j['mode'],j['card'],j['repeat']) for j in schedule}
 for m in read(stage+'-report.json')['metrics']:
  counter=metrics[m['mode'],m['card'],m['repeat']]
  for k,v in counter.items():assert m[k]==v
 counts[stage]=len(files)
assert read('selection.json')['time']<read('runs/holdout/runtime-before.json')['time']
assert not read('holdout-report.json')['gate']['passed']
result={'verified_files':len(read('export-manifest.json')),'calls':counts,'unique_worker_execution_ids':len(ids),'attempt_status_counts':dict(attempts),'prompt_and_image_hashes_match':True,'candidate_mapping_exact':True,'frozen_files_match':True,'independent_metric_recalculation_matches':True,'selection_precedes_holdout':True,'limitations':'Execution IDs are worker-generated; no claim of upstream model identity verification.'}
(ROOT/'integrity-audit.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n');print(json.dumps(result,ensure_ascii=False))
