"""Freeze a source-disjoint screenshot supplement; never promote AI labels to human gold."""
import hashlib,json,pathlib,random,collections,urllib.parse
ROOT=pathlib.Path(__file__).resolve().parents[1]
MAIN=pathlib.Path('/Users/wangxiaolin/PycharmProjects/review-annotation-platform/data')
OUT=ROOT/'data/luna-building-release-validation-20261010'
def norm(x):return urllib.parse.urlparse(x).path.removeprefix('/minio-src').lstrip('/')
def gold_source_event(row, byobj):
 # Historical source_event_id may actually be a task ID; trust attachment lineage.
 source=byobj.get(norm(row['image_object_key']))
 return source['event_id'] if source else None
def read(p):return json.loads(p.read_text())
def write(n,x):(OUT/n).write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n')
def main():
 if (OUT/'execution-units.json').exists():raise RuntimeError('Frozen evidence exists; use a separate OUT for a new selection, never overwrite this run')
 old=read(ROOT/'data/luna-clean-frame-eval-20261009/all-items.json'); included=read(ROOT/'data/luna-clean-frame-eval-20261009/included.json')
 rows=[json.loads(l) for l in (MAIN/'ktd-test-real-monitoring-full-20260924/source_manifest.jsonl').open()]
 byevent={x['event_id']:x for x in rows};byobj={x['annotated_object_path'].lstrip('/'):x for x in rows}
 ann=[json.loads(l) for l in (MAIN/'gold-export-live-20260903/annotations.jsonl').open()];bycand={x['candidate_id']:x for x in ann}
 used_events=set(); unresolved=[]
 for u in old:
  if u['unit'] in byevent:used_events.add(u['unit'])
  for c in u['candidates']:
   if c['key'].startswith('gold0903:'):
    a=bycand[c['key'].split(':')[1]];event=gold_source_event(a,byobj)
    if event:used_events.add(event)
   elif u['unit'].startswith('gold0924'):
    # The original human-gold export retains source event lineage.
    for folder in ['ktd-codex-human-gold-train-20260924','ktd-codex-human-gold-validation-20260924']:
     for line in (MAIN/folder/'human_gold_annotations.jsonl').open():
      a=json.loads(line)
      if 'gold0924:'+a['candidate_id']==c['key']:
       event=gold_source_event(a,byobj)
       if event:used_events.add(event)
 used_sources={norm(s['object_path']) for e in used_events for s in byevent[e]['raw_sources']}
 rawhashes={pathlib.Path(x['source_local_path']).stem for x in included}
 eligible=[r for r in rows if r['raw_sources'] and r['event_id'] not in used_events and not any(norm(s['object_path']) in used_sources or hashlib.sha256(s['url'].encode()).hexdigest() in rawhashes for s in r['raw_sources']) and all(c['class_name']=='违建' for c in r['candidates'])]
 random.Random(20261010).shuffle(eligible);selected=[];seen=set()
 for r in eligible:
  src={norm(s['object_path']) for s in r['raw_sources']}
  if src & seen:continue
  seen|=src;selected.append(r)
  if len(selected)==24:break
 units=[]
 for r in selected:
  cs=[]
  for c in r['candidates']:
   w,h=c['frame_size']; b=c['bbox_px'];box=[b[0]/w,b[1]/h,b[2]/w,b[3]/h]
   if not (0<=box[0]<box[2]<=1 and 0<=box[1]<box[3]<=1):raise ValueError('invalid original box')
   cs.append(dict(key=c['candidate_id'],class_name=c['class_name'],bbox=box))
  units.append(dict(unit=r['event_id'],cohort='independent_ai_review',source_objects=sorted(norm(s['object_path']) for s in r['raw_sources']),candidates=cs,label_provenance='unlabeled; AI visual review only, human acceptance pending'))
 fp=[x for x in read(ROOT/'data/luna-clean-vs-card-20261010/per-box.json') if x['class_name']=='违建' and x['label']=='false' and any(x['decisions'][k]=='true' for k in ['B-1','B-2'])]
 oldby={u['unit']:u for u in old}
 for uid in sorted({x['unit'] for x in fp}):units.append(dict(oldby[uid],cohort='known_fp_regression'))
 write('units.json',units);write('old-false-positives.json',fp)
 write('source-audit.json',dict(seed=20261010,old_images=len(old),old_events=sorted(used_events),old_sources=sorted(used_sources),old_raw_hashes=sorted(rawhashes),eligible_images=len(eligible),selected_images=len(selected),selected_candidates=sum(len(u['candidates']) for u in units if u['cohort']=='independent_ai_review'),source_overlap=0,selection='all pure-building screenshots, fixed shuffle, one image per disjoint raw-source set, first 24; no output/label selection',human_gold_available=0))
 write('protocol.json',dict(seed=20261010,repeats=2,arms=['p000','p000-building-only'],candidate_commit='42397238abe9e6fe544af826478f61757fb748ee',ports=[18401,18402],total_cgroup_memory='4G',swap=0,cpu_quota='300%',cpu_affinity=3,total_concurrency=6,gate='No full automatic release recommendation without independent human labels and business false-acceptance tolerance. AI visual labels are exploratory only. Any reproducible confident acceptance of facade/sign/fence is a quality failure to report, not relabel away.',labels_frozen_before_new_model_outputs=True,shared_services='read-only baseline and final health only'))
 print('units',len(units),'new',len(selected),'new_boxes',sum(len(u['candidates']) for u in units if u['cohort']=='independent_ai_review'))
if __name__=='__main__':main()
