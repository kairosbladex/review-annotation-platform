"""Paired B/G human-label regression, source-cluster bootstrap; failures are not labels."""
import collections,json,pathlib,random,sys
ROOT=pathlib.Path(__file__).resolve().parents[1]/'data/luna-building-geometry-20261010'
FOCUS={'gold0903:9f3b0ba9ab134358b8f1b202bdc5454a','gold0903:445ef456f78e4547b4064cd552e1cfb9','gold0903:fe528a0df8ae465d951d326edba41363','gold0903:a33bbf274cdc410cba575f56f21e2fb7'}
def write(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n')
def rows_for(folder):
 units=json.loads((folder/'units.json').read_text());groups={r['unit']:r['cluster'] for r in json.loads((ROOT/'source-groups.json').read_text())};rows=[]
 for u in units:
  out={}
  for arm in ['B','G']:
   for n in [1,2]:
    relative=f'results/{arm}-{n}/{u["unit"]}.json'
    z=json.loads((folder/relative).read_text())
    if z['status']!=200:
     recovery=folder.parent/(folder.name+'-retry')/relative
     if recovery.exists():z=json.loads(recovery.read_text())
    assert (z['unit'],z['arm'],z['repeat'])==(u['unit'],arm,n)
    assert z['status']==200 and z['image_sha256']==u['sha256'], 'Failed calls must not be scored as false'
    result=z['body']['results'];assert {x['target_index'] for x in result}==set(range(len(u['candidates'])))
    out[f'{arm}-{n}']={x['target_index']:x for x in result}
  for i,c in enumerate(u['candidates']):rows.append(dict(unit=u['unit'],key=c['key'],label=c['label'],split=u['split'],cluster=groups[u['unit']],focus=c['key'] in FOCUS,decisions={k:v[i]['decision'] for k,v in out.items()},reasoning={k:v[i]['reasoning'] for k,v in out.items()}))
 return rows

def summary(rows,k):
 known=[r for r in rows if r['label'] in ['true','false']];pos=[r for r in known if r['label']=='true'];neg=[r for r in known if r['label']=='false']
 return dict(scored=len(known),true_targets=len(pos),false_targets=len(neg),accuracy=sum((r['decisions'][k]=='true')==(r['label']=='true') for r in known)/len(known) if known else None,true_rejected=sum(r['decisions'][k]!='true' for r in pos),false_accepted=sum(r['decisions'][k]=='true' for r in neg),uncertain=sum(r['decisions'][k]=='uncertain' for r in rows))

def paired_ci(rows,label,metric):
 rs=[r for r in rows if r['label'] in ['true','false'] and (label is None or r['label']==label)]
 grouped=collections.defaultdict(list)
 for r in rs:
  def value(k):
   d=r['decisions'][k]
   return (d=='true')==(r['label']=='true') if metric=='accuracy' else (d!='true' if metric=='true_rejected' else d=='true')
  grouped[r['cluster']].append(sum(int(value(f'G-{n}'))-int(value(f'B-{n}')) for n in [1,2])/2)
 if not rs:return None
 stats=[(sum(v),len(v)) for v in grouped.values()];rng=random.Random(20261010);vals=[]
 for _ in range(10000):
  sample=rng.choices(stats,k=len(stats));vals.append(sum(x[0] for x in sample)/sum(x[1] for x in sample))
 vals.sort();return dict(difference_G_minus_B=sum(s for s,n in stats)/sum(n for s,n in stats),ci95=[vals[250],vals[9749]],groups=len(stats),resamples=10000)

def main():
 phase=sys.argv[1] if len(sys.argv)>1 else 'regression';folder=ROOT/phase;rows=rows_for(folder);write(folder/'per-box.json',rows)
 result={}
 for subset,rs in [('all',rows),('dev',[r for r in rows if r['split']=='dev']),('holdout',[r for r in rows if r['split']=='holdout']),('focus_four',[r for r in rows if r['focus']])]:
  result[subset]={k:summary(rs,k) for k in ['B-1','B-2','G-1','G-2']}
  result[subset]['paired_ci']={m:paired_ci(rs,l,m) for l,m in [(None,'accuracy'),('true','true_rejected'),('false','false_accepted')]}
 transitions={str(n):[dict(key=r['key'],unit=r['unit'],B=r['decisions'][f'B-{n}'],G=r['decisions'][f'G-{n}']) for r in rows if r['label']=='true' and r['decisions'][f'B-{n}']=='true' and r['decisions'][f'G-{n}']!='true'] for n in [1,2]}
 result['new_true_rejections']=transitions
 result['gate_observed']=all(result['all'][f'G-{n}']['true_rejected']<=result['all'][f'B-{n}']['true_rejected'] and result['all'][f'G-{n}']['false_accepted']<result['all'][f'B-{n}']['false_accepted'] and result['focus_four'][f'G-{n}']['false_accepted']<result['focus_four'][f'B-{n}']['false_accepted'] for n in [1,2])
 result['no_new_paired_true_rejections']=not any(transitions.values())
 result['gate_warning']='Observed counts only. Any new paired true rejection requires visual review; CI and previous-label disagreement must be reported. Not a guarantee of population noninferiority.'
 write(folder/'metrics.json',result);print(json.dumps(result,ensure_ascii=False,indent=2))
if __name__=='__main__':main()
