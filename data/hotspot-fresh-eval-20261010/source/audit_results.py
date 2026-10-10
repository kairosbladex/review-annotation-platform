"""Independent audit and paired sensitivity summaries; never calls a model."""
import collections, hashlib, importlib.util, json
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
def read(name):return json.loads((ROOT/name).read_text())
def save(name,data):(ROOT/name).write_text(json.dumps(data,ensure_ascii=False,indent=2)+'\n')
def sha(data):return hashlib.sha256(data).hexdigest()
def bootstrap(arr):
    arr=np.asarray(arr,dtype=float);rng=np.random.default_rng(20261010)
    sums=arr[rng.integers(len(arr),size=(20000,len(arr)))].sum(axis=1)
    dist=sums[:,1:]/sums[:,0,None]
    return {'clusters':len(arr),'delta':(arr[:,1:].sum(0)/arr[:,0].sum()).tolist(),'ci95':np.quantile(dist,[.025,.975],axis=0).T.tolist()}
def main():
    frozen=read('freeze.json')
    for path,h in frozen['files'].items():assert sha((ROOT/path).read_bytes())==h,path
    spec=importlib.util.spec_from_file_location('production',ROOT/'source/production_worker.py');prod=importlib.util.module_from_spec(spec);spec.loader.exec_module(prod)
    units=read('split.json');unitby={u['event_id']:u for u in units};gold={r['key']:{'是光斑':'true','不是光斑':'false','看不清':'uncertain'}[r['label']] for r in read('labels.json')}
    protocol=read('protocol.json');cards=read('cards.json');schedule=read('runs/holdout/schedule.json');records=[json.loads(p.read_text()) for p in (ROOT/'runs/holdout/out').glob('*.json')]
    key=lambda r:(r['event_id'],r['mode'],r['card'],r['repeat'])
    assert len(records)==len(schedule)==protocol['calls']==348
    assert len({key(r) for r in records})==len(records) and {key(r) for r in records}=={key(r) for r in schedule}
    assert all(r['success'] for r in records)
    assert len({r['result']['execution_id'] for r in records})==len(records)
    counts={};decisions={};decisions_by_call={}
    for r in records:
        u=unitby[r['event_id']];ks=[b['key'] for b in u['boxes']];assert r['keys']==ks
        cs=[{'target_index':i,'class_name':'光斑','bbox':b['screenshot_bbox' if r['mode']=='screenshot' else 'bbox']} for i,b in enumerate(u['boxes'])]
        assert r['candidates']==cs
        assert r['image_sha256']==u['screenshot_sha256' if r['mode']=='screenshot' else 'clean_sha256']
        doc={'version':'p000-pure-class-cards','common':prod.load_prompt('p000')['common'],'cards':{} if r['card']=='p000' else {'光斑':cards[r['card']]}}
        assert r['prompt']==prod.build_prompt(doc,cs) and sha(r['prompt'].encode())==r['prompt_sha256']
        assert r['result']['requested_model']=='gpt-5.6-luna'
        rr=r['result']['results'];assert sorted(x['target_index'] for x in rr)==list(range(len(ks)))
        c=counts.setdefault((r['mode'],r['card'],r['repeat']),collections.Counter())
        for x in rr:
            k=ks[x['target_index']];y=gold[k];q=x['decision'];assert q in ('true','false','uncertain')
            c['n']+=1;c['correct']+=q==y;c['false_reject']+=y=='true' and q!='true';c['false_accept']+=y=='false' and q=='true';c['uncertain']+=q=='uncertain';c['explicit_false_reject']+=y=='true' and q=='false'
            decisions[k,r['mode'],r['card'],r['repeat']]=q
        decisions_by_call[key(r)]=r
    report=read('holdout-report.json')
    for m in report['metrics']:
        c=counts[m['mode'],m['card'],m['repeat']]
        assert all(m[k]==v for k,v in c.items());assert m['accuracy']==c['correct']/c['n']
    comparisons=[]
    for mode in ['screenshot','original']:
        for card in cards:comparisons.append((f'{mode}:{card}-p000',(mode,card),(mode,'p000')))
    for card in ['p000',*cards]:comparisons.append((f'{card}:original-screenshot',('original',card),('screenshot',card)))
    sensitivity=[];changes=[]
    for name,candidate,baseline in comparisons:
        for cluster_field in ['screenshot_sha256','task_id']:
            clusters={}
            for u in units:
                agg=clusters.setdefault(u[cluster_field],np.zeros(5))
                for b in u['boxes']:
                    k=b['key'];y=gold[k];agg[0]+=1
                    for rep in [1,2]:
                        q=decisions[k,*candidate,rep];p=decisions[k,*baseline,rep]
                        agg[1:]+=.5*np.array([int(q==y)-int(p==y),int(y=='false' and q=='true')-int(y=='false' and p=='true'),int(y=='true' and q!='true')-int(y=='true' and p!='true'),int(q=='uncertain')-int(p=='uncertain')])
                        if cluster_field=='screenshot_sha256' and q!=p:changes.append({'comparison':name,'event_id':u['event_id'],'key':k,'repeat':rep,'gold':y,'baseline':p,'candidate':q})
            item={'comparison':name,'cluster_field':cluster_field,'order':['accuracy','false_accept_per_box','false_reject_per_box','uncertain_per_box'],**bootstrap(list(clusters.values()))};sensitivity.append(item)
            if cluster_field=='screenshot_sha256' and candidate[0]==baseline[0]:
                original=next(c for c in report['paired'] if c['mode']==candidate[0] and c['card']==candidate[1]);assert np.allclose(original['delta'],item['delta']) and np.allclose(original['ci95'],item['ci95'])
    assert report['negative_gold_count']==0 and all(g['passed']==False and g['false_accept_not_increased_each_repeat'] is None for g in report['gates'])
    samples=[json.loads(l) for l in (ROOT/'runs/holdout/resources.jsonl').read_text().splitlines()]
    before=read('runs/holdout/runtime-before.json');after=read('runs/holdout/runtime-after.json')
    samples=[before['host'],*samples,after['host']]
    assert before['limits']=={'memory.max':'4294967296','memory.swap.max':'0','cpu.max':'300000 100000'}
    audit={'calls':len(records),'targets':len(gold),'decisions':sum(c['n'] for c in counts.values()),'frozen_files_verified':len(frozen['files']),'prompt_image_candidate_mapping_verified':True,'metrics_and_event_intervals_independently_verified':True,'negative_gold':0,'gate_not_assessable':True,'resource_samples':len(samples),'peak_memory_bytes':max(s['memory_current'] for s in samples),'maximum_swap_bytes':max(s['swap_current'] for s in samples),'minimum_host_available_fraction':min(s['available_fraction'] for s in samples),'host_swapout_delta':samples[-1]['pswpout']-samples[0]['pswpout'],'shared_health_before':before['shared'],'shared_health_after':after['shared']}
    save('integrity-audit.json',audit);save('paired-sensitivity.json',sensitivity);save('paired-changes.json',changes);print(json.dumps(audit,ensure_ascii=False,indent=2))
if __name__=='__main__':main()
