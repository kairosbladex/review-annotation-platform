"""Independent integrity audit of completed development calls, no new inference."""
import collections,hashlib,json,pathlib
ROOT=pathlib.Path(__file__).resolve().parents[1]
def main():
 read=lambda p:json.loads((ROOT/p).read_text())
 freeze=read('freeze.json')
 for path,h in freeze['files'].items():assert hashlib.sha256((ROOT/path).read_bytes()).hexdigest()==h,path
 schedule=read('runs/development/schedule.json');records=[json.loads(p.read_text()) for p in (ROOT/'runs/development/out').glob('*.json')]
 key=lambda r:(r['event_id'],r['mode'],r['card'],r['repeat'])
 assert len(records)==len(schedule)==112 and len({key(r) for r in records})==112
 assert {key(r) for r in records}=={key(r) for r in schedule}
 assert all(r['success'] for r in records)
 assert len({r['result']['execution_id'] for r in records})==112
 units={u['event_id']:u for u in read('split.json')};gold={r['key']:{'是光斑':'true','不是光斑':'false','看不清':'uncertain'}[r['label']] for r in read('labels.json')}
 counts={};by={key(r):r for r in records};paired=0
 for r in records:
  u=units[r['event_id']];expected=[b['key'] for b in u['boxes']];assert r['keys']==expected
  assert r['image_sha256']==u['screenshot_sha256' if r['mode']=='screenshot' else 'clean_sha256']
  assert r['candidates']==[{'target_index':i,'class_name':'光斑','bbox':b['screenshot_bbox' if r['mode']=='screenshot' else 'bbox']} for i,b in enumerate(u['boxes'])]
  assert hashlib.sha256(r['prompt'].encode()).hexdigest()==r['prompt_sha256']
  c=counts.setdefault((r['mode'],r['card'],r['repeat']),collections.Counter())
  results=r['result']['results'];assert sorted(x['target_index'] for x in results)==list(range(len(expected)))
  for x in results:
   y=gold[expected[x['target_index']]];q=x['decision'];assert q in ('true','false','uncertain')
   c['n']+=1;c['correct']+=q==y;c['false_reject']+=y=='true' and q!='true';c['false_accept']+=y=='false' and q=='true';c['uncertain']+=q=='uncertain'
  counterpart={'context':'p000','repair_context':'repair'}.get(r['card'])
  if counterpart:
   b=by[r['event_id'],r['mode'],counterpart,r['repeat']]
   assert b['prompt']==r['prompt'] and b['candidates']==r['candidates'] and b['image_sha256']==r['image_sha256'];paired+=1
 for m in read('development-report.json')['metrics']:
  c=counts[m['mode'],m['card'],m['repeat']]
  assert all(m[k]==v for k,v in c.items());assert m['accuracy']==c['correct']/c['n']
 samples=[json.loads(l) for l in (ROOT/'runs/development/resources.jsonl').read_text().splitlines()]
 out={'calls':112,'development_boxes':len(gold),'decisions':sum(c['n'] for c in counts.values()),'paired_same_prompt_image_candidate_checks':paired,'frozen_files_verified':len(freeze['files']),'metrics_independently_verified':True,'final_holdout_run':False,'resource_samples':len(samples),'peak_memory_bytes':max(s['memory_current'] for s in samples),'max_swap_bytes':max(s['swap_current'] for s in samples),'minimum_host_available_fraction':min(s['available_fraction'] for s in samples),'host_swapout_delta':samples[-1]['pswpout']-samples[0]['pswpout']}
 (ROOT/'integrity-audit.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out))
if __name__=='__main__':main()
