"""Conservative connected source groups; shared source sets stay in one bootstrap cluster."""
import hashlib,json,pathlib,urllib.parse
ROOT=pathlib.Path(__file__).resolve().parents[1]
DATA=pathlib.Path('/Users/wangxiaolin/PycharmProjects/review-annotation-platform/data')
OUT=ROOT/'data/luna-building-geometry-20261010'
def norm(x):return urllib.parse.urlparse(x).path.removeprefix('/minio-src').lstrip('/')
def main():
 units=json.loads((OUT/'units.json').read_text())
 rows=[json.loads(l) for l in (DATA/'ktd-test-real-monitoring-full-20260924/source_manifest.jsonl').open()]
 byevent={x['event_id']:x for x in rows};byobj={norm(x['annotated_object_path']):x for x in rows}
 bykey={}
 for folder,file,prefix in [('gold-export-live-20260903','annotations.jsonl','gold0903:'),('ktd-codex-human-gold-train-20260924','human_gold_annotations.jsonl','gold0924:'),('ktd-codex-human-gold-validation-20260924','human_gold_annotations.jsonl','gold0924:')]:
  for line in (DATA/folder/file).open():
   a=json.loads(line);bykey[prefix+a['candidate_id']]=byobj.get(norm(a['image_object_key']))
 records=[]
 for u in units:
  events=[byevent[u['unit']]] if u['unit'] in byevent else [bykey[c['key']] for c in u['candidates'] if bykey.get(c['key'])]
  sources=sorted({norm(s['object_path']) for e in events for s in e['raw_sources']})
  records.append(dict(unit=u['unit'],events=sorted({e['event_id'] for e in events}),sources=sources,unresolved=not bool(sources)))
 parents=list(range(len(records)))
 def root(i):
  while parents[i]!=i:i=parents[i]
  return i
 owners={}
 for i,r in enumerate(records):
  # Missing lineage conservatively clusters all unresolved images together.
  for source in r['sources'] or ['unresolved-source']:
   if source in owners:parents[root(i)]=root(owners[source])
   owners[source]=i
 for i,r in enumerate(records):r['cluster']=records[root(i)]['unit']
 (OUT/'source-groups.json').write_text(json.dumps(records,ensure_ascii=False,indent=2)+'\n')
 print({'images':len(records),'groups':len({r['cluster'] for r in records}),'unresolved':sum(r['unresolved'] for r in records)})
if __name__=='__main__':main()
