"""Secondary region-weighted sensitivity; never alters gold or model inputs."""
import json
from pathlib import Path
import numpy as np
from audit_results import bootstrap
ROOT=Path(__file__).resolve().parents[1]
def read(n):return json.loads((ROOT/n).read_text())
def main():
    units={u['event_id']:u for u in read('split.json')};groups=read('overlap-audit.json')['groups'];ds={(d['key'],d['mode'],d['card'],d['repeat']):d for d in read('holdout-per-box.json')}
    metrics=[];comparisons=[]
    for mode in ['screenshot','original']:
        for card in ['p000','p001_p002','repair']:
            for rep in [1,2]:
                vals=[]
                for g in groups:
                    rows=[ds[k,mode,card,rep] for k in g['keys']]
                    vals.append(np.mean([[r['pred']==r['gold'],r['gold']=='true' and r['pred']!='true',r['pred']=='uncertain'] for r in rows],axis=0))
                metrics.append({'mode':mode,'card':card,'repeat':rep,'regions':len(vals),'order':['accuracy','false_reject_per_region','uncertain_per_region'],'values':np.mean(vals,axis=0).tolist()})
        for card in ['p001_p002','repair']:
            for field in ['event_id','task_id']:
                clusters={}
                for g in groups:
                    u=units[g['event_id']];arr=clusters.setdefault(u[field],np.zeros(2));arr[0]+=1;diffs=[]
                    for k in g['keys']:
                        for rep in [1,2]:
                            a=ds[k,mode,card,rep];b=ds[k,mode,'p000',rep];diffs.append(int(a['pred']==a['gold'])-int(b['pred']==b['gold']))
                    arr[1]+=np.mean(diffs)
                comparisons.append({'mode':mode,'card':card,'cluster_field':field,**bootstrap(list(clusters.values()))})
    out={'method':'IoU>=0.5 connected groups; average decisions within each group and repetitions, then paired cluster bootstrap. Geometric sensitivity only; groups are not confirmed unique objects.','metrics':metrics,'comparisons':comparisons}
    (ROOT/'overlap-sensitivity.json').write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n');print('Region-weighted sensitivity computed')
if __name__=='__main__':main()
