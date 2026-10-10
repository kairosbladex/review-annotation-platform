"""Metadata-only inventory. Never opens candidate pixels or labels."""
import json,collections,pathlib,hashlib
ROOT=pathlib.Path('/data/workspace/jzheng');snap=ROOT/'codex-ktd-real-monitoring-20260924'
rows=[json.loads(l) for l in (snap/'input/inference_manifest.jsonl').read_text().splitlines()]
auto=json.loads((ROOT/'luna-prompt-autotune/assets/items.json').read_text())
old=json.loads((ROOT/'hotspot-relabel-20261010/boxes.json').read_text())['boxes']
seen={r['unit'] for r in auto}|{r['event_id'] for r in old};hashes={r.get('sha256') for r in auto}-{None}
by=collections.defaultdict(list)
for r in rows:by[r['event_id']].append(r)
# Block whole task/flight cohorts, not only boxes or exact files.
seen_tasks={r['task_id'] for r in rows if r['event_id'] in seen}
counts=collections.Counter();candidates=[]
for eid,rs in by.items():
 if {r['class_name'] for r in rs}!={'光斑'}:continue
 counts['pure_hotspot_events']+=1
 if eid in seen:counts['known_exposed_event']+=1;continue
 if any(r['task_id'] in seen_tasks for r in rs):counts['same_task_as_exposed']+=1;continue
 path=snap/'work/annotated'/f'{eid}.jpg'
 if not path.exists():counts['missing_screenshot']+=1;continue
 if hashlib.sha256(path.read_bytes()).hexdigest() in hashes:counts['duplicate_exposed_image']+=1;continue
 counts['candidate_events']+=1;candidates.append({'event_id':eid,'task_id':rs[0]['task_id'],'boxes':len(rs),'status':'needs_source_lineage_and_human_labels'})
print(json.dumps({'scope':str(snap),'counts':dict(counts),'candidates':candidates,'new_human_labels_available':False,'limitation':'Only this existing snapshot audited; zero does not mean no new data exists elsewhere.'},ensure_ascii=False,indent=2))
