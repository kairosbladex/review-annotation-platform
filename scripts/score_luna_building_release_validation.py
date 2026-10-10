"""Keep human-gold regression and independent AI-reference agreement separate."""
import json,pathlib,collections,random
R=pathlib.Path(__file__).resolve().parents[1]/'data/luna-building-release-validation-20261010'
def read(n):return json.loads((R/n).read_text())
def dump(n,x):(R/n).write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n')
def summary(rows,a,n,label_field='label'):
 x=[r for r in rows if r[label_field] in ['true','false']]; pos=[r for r in x if r[label_field]=='true'];neg=[r for r in x if r[label_field]=='false'];key=f'{a}-{n}'
 return dict(scored=len(x),positive=len(pos),negative=len(neg),agreement=sum((r['decisions'][key]=='true')==(r[label_field]=='true') for r in x)/len(x) if x else None,true_rejected=sum(r['decisions'][key]!='true' for r in pos),false_accepted=sum(r['decisions'][key]=='true' for r in neg),uncertain_scored=sum(r['decisions'][key]=='uncertain' for r in x),uncertain_all=sum(r['decisions'][key]=='uncertain' for r in rows))
def boot(rows):
 groups=collections.defaultdict(list)
 for r in rows:
  if r['label'] in ['true','false']:groups[r['unit']].append(r)
 def diff(rs):return sum(sum(((r['decisions'][f'B-{n}']=='true')==(r['label']=='true'))-((r['decisions'][f'A-{n}']=='true')==(r['label']=='true')) for n in [1,2])/2 for r in rs)/len(rs)
 keys=sorted(groups);rng=random.Random(20261010);vals=[]
 for _ in range(5000):
  sample=[r for k in rng.choices(keys,k=len(keys)) for r in groups[k]];vals.append(diff(sample))
 vals.sort();return dict(mean_difference=diff([r for g in groups.values() for r in g]),ci95=[vals[125],vals[4874]],source_groups=len(keys),interpretation='exploratory AI-reference agreement, not independent human accuracy; source separation does not prove geographic scene separation')
def main():
 units=read('units.json');refs={x['key']:x for x in read('ai-reference-frozen.json')['items']};oldlabels=json.loads((R.parent/'luna-clean-vs-card-20261010/labels-used.json').read_text());rows=[]
 for u in units:
  decisions={}
  for a in ['A','B']:
   for n in [1,2]:
    z=read(f'results/{a}-{n}/{u["unit"]}.json');assert z['status']==200
    assert z['image_sha256']==u['sha256'];decisions[f'{a}-{n}']={x['target_index']:x['decision'] for x in z['body']['results']}
  for i,c in enumerate(u['candidates']):
   ref=refs.get(c['key']);label=ref['label'] if ref else oldlabels.get(c['key'],{}).get('label')
   rows.append(dict(unit=u['unit'],key=c['key'],cohort=u['cohort'],label=label,provenance='AI visual, not human gold' if ref else 'existing human gold unchanged',image_type=ref['image_type'] if ref else 'visible',decisions={k:v[i] for k,v in decisions.items()}))
 dump('per-box.json',rows);report={}
 for cohort in ['independent_ai_review','source_overlap_regression','known_fp_regression']:
  rs=[r for r in rows if r['cohort']==cohort];report[cohort]={f'{a}-{n}':summary(rs,a,n) for a in ['A','B'] for n in [1,2]}

  if cohort=='independent_ai_review':report[cohort]['bootstrap']=boot(rs)
 oldkeys={x['key'] for x in read('old-false-positives.json')};rs=[r for r in rows if r['key'] in oldkeys];report['old_10_false_positives']={f'{a}-{n}':summary(rs,a,n) for a in ['A','B'] for n in [1,2]}
 confirmed={x['key'] for x in read('false-positive-review.json') if x['visual_status']=='confirmed_false'};report['confirmed_visual_false']={f'{a}-{n}':summary([r for r in rows if r['key'] in confirmed],a,n) for a in ['A','B'] for n in [1,2]}
 report['independent_thermal']={f'{a}-{n}':dict(collections.Counter(r['decisions'][f'{a}-{n}'] for r in rows if r['cohort']=='independent_ai_review' and r['image_type']=='thermal')) for a in ['A','B'] for n in [1,2]}
 dump('metrics.json',report);print(json.dumps(report,ensure_ascii=False,indent=2))
if __name__=='__main__':main()
