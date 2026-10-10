"""Exploratory paired contrasts for a frozen 2x2x2 development experiment."""
import collections,json,pathlib
import numpy as np
ROOT=pathlib.Path(__file__).resolve().parents[1]
def main():
 rows=json.loads((ROOT/'development-per-box.json').read_text());units=json.loads((ROOT/'split.json').read_text());idx={(r['key'],r['mode'],r['card'],r['repeat']):r for r in rows}
 contrasts=[]
 pairs=[('p000','repair','card_only'),('p000','context','context_only'),('context','repair_context','card_given_context'),('repair','repair_context','context_given_card')]
 for mode in ('screenshot','original'):
  for base,candidate,name in pairs:
   group={}
   for u in units:
    a=group.setdefault(u['screenshot_sha256'],np.zeros(5))
    for box in u['boxes']:
     a[0]+=1
     for rep in (1,2):
      p=idx[box['key'],mode,base,rep];q=idx[box['key'],mode,candidate,rep];y=p['gold'];p=p['pred'];q=q['pred']
      a[1:]+=.5*np.array([int(q==y)-int(p==y),int(y=='true' and q!='true')-int(y=='true' and p!='true'),int(y=='false' and q=='true')-int(y=='false' and p=='true'),int(q=='uncertain')-int(p=='uncertain')])
   arr=np.array(list(group.values()));rng=np.random.default_rng(20261010);sample=arr[rng.integers(0,len(arr),(20000,len(arr)))].sum(axis=1);dist=sample[:,1:]/sample[:,0,None]
   contrasts.append({'mode':mode,'contrast':name,'baseline':base,'candidate':candidate,'order':['accuracy','true_not_passed_per_box','false_accepted_per_box','uncertain_per_box'],'delta':(arr[:,1:].sum(axis=0)/arr[:,0].sum()).tolist(),'ci95':np.quantile(dist,[.025,.975],axis=0).T.tolist(),'clusters':len(arr),'scope':'exploratory_development_not_release_evidence'})
 out={'new_calls':False,'source':'development-per-box.json','contrasts':contrasts,'final_gate':'not_run_no_fresh_labeled_holdout'}
 (ROOT/'factorial-report.json').write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n')
 print(json.dumps(out,ensure_ascii=False))
if __name__=='__main__':main()
