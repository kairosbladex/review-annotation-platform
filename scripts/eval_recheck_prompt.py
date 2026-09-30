#!/usr/bin/env python3
"""Paired GPT prompt evaluation; explicit budget; never selects or publishes prompts."""
from __future__ import annotations
import argparse
import collections
import hashlib
import json
import math
import os
import random
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'backend'))
from app.config import Settings
from app.adapters.recheck_relay import RecheckRelay,RelayFailure
from app.services.online_recheck import prompt_and_images,parse_advice,RECHECK_SCHEMA,SUPPORTED


def wilson(k,n):
    if not n:return {'numerator':k,'denominator':n,'rate':None,'wilson95':None}
    z=1.95996398454;p=k/n;d=1+z*z/n
    mid=(p+z*z/(2*n))/d;half=z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/d
    return {'numerator':k,'denominator':n,'rate':p,'wilson95':[max(0,mid-half),min(1,mid+half)]}

def mcnemar(pairs):
    # Exact two-sided conditional binomial; outcome = agrees with human truth.
    b=sum(a and not b for a,b in pairs);c=sum(b and not a for a,b in pairs);n=b+c
    p=min(1,2*sum(math.comb(n,k) for k in range(min(b,c)+1))/2**n) if n else 1
    return {'a_correct_b_wrong':b,'b_correct_a_wrong':c,'discordant':n,'exact_two_sided_p':p,
            'limitation':'同组框相关，McNemar独立配对假设可能不成立；不得仅凭p值选提示词'}

def read_rows(path):return [json.loads(l) for l in path.read_text().splitlines() if l.strip()]

def select_rows(rows,split,limit,seed):
    strata=collections.defaultdict(list);missing=[]
    for r in rows:
        if r['split']!=split or r['class_name'] not in SUPPORTED:continue
        if split=='locked_test_gpt' and 'gpt' not in r['valid_for']:continue
        path=next((Path(x) for x in r.get('source_images',[]) if Path(x).is_file()),None)
        if path:r={**r,'source_image':str(path)}
        if not path or not path.is_file():missing.append(r['gold_id']);continue
        if hashlib.sha256(path.read_bytes()).hexdigest()!=r['image_sha256']:raise ValueError('gold image hash mismatch')
        strata[(r['class_name'],tuple(r['population']),r['label'])].append(r)
    rng=random.Random(seed)
    for key in sorted(strata):rng.shuffle(strata[key])
    selected=[]
    while any(strata.values()) and (limit==0 or len(selected)<limit):
        for key in sorted(strata):
            if strata[key] and (limit==0 or len(selected)<limit):selected.append(strata[key].pop())
    return selected,missing

def aggregate(results,versions,max_false_reject,locked_false_count=0):
    report={};keys=sorted({(r['class_name'],p) for r in results for p in r['population']})
    for cls,pop in keys:
        allrows=[r for r in results if r['class_name']==cls and pop in r['population']]
        values={}
        for version in versions:
            rows=[r for r in allrows if r['version']==version];done=[r for r in rows if r['status']=='done']
            true=[r for r in done if r['label']=='true'];false=[r for r in done if r['label']=='false']
            values[version]={'samples':len(rows),'completed':len(done),'failed':len(rows)-len(done),'groups':len({r['group_id'] for r in rows}),
                             'false_reject':wilson(sum(r['decision']=='false' for r in true),len(true)),
                             'false_target_reject':wilson(sum(r['decision']=='false' for r in false),len(false)),
                             'uncertain':wilson(sum(r['decision']=='uncertain' for r in done),len(done)),
                             'tokens':sum(r.get('usage',{}).get('total_tokens',0) for r in rows),
                             'elapsed_seconds':sum(r['elapsed_seconds'] for r in rows)}
            fr=values[version]['false_reject'];ci=fr['wilson95']
            values[version]['vehicle_false_reject_gate']=(
                {'threshold':max_false_reject,'status':'通过' if ci and ci[1]<=max_false_reject else '不通过证据门槛',
                 'reason':'Wilson95上界需不超过阈值；零错误且样本少仍不能保证达标'} if cls=='车' else None)
            values[version]['false_target_reject_gate']='只报告数字；完整锁定集车类false不足100' if cls=='车' and locked_false_count<100 else '未设通过门槛'
        if len(versions)==2:
            a={r['gold_id']:r for r in allrows if r['version']==versions[0] and r['status']=='done'}
            b={r['gold_id']:r for r in allrows if r['version']==versions[1] and r['status']=='done'}
            values['paired_mcnemar']=mcnemar([(a[k]['decision']==a[k]['label'],b[k]['decision']==b[k]['label']) for k in a.keys()&b.keys() if a[k]['label'] in {'true','false'}])
        report[cls+'/'+pop]=values
    return report

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--gold',type=Path,required=True);p.add_argument('--split',choices=['development','locked_test_gpt'],required=True)
    p.add_argument('--out',type=Path,required=True);p.add_argument('--env-file',type=Path);p.add_argument('--limit',type=int,default=48)
    p.add_argument('--baseline',action='store_true');p.add_argument('--prompt-a',type=Path);p.add_argument('--prompt-b',type=Path)
    p.add_argument('--winner-evidence',type=Path);p.add_argument('--budget-usd',type=float,default=10);p.add_argument('--reserve-per-request-usd',type=float,default=1)
    p.add_argument('--max-false-reject',type=float,default=0.01);p.add_argument('--seed',type=int,default=20260930)
    a=p.parse_args()
    if a.budget_usd<=0 or a.reserve_per_request_usd<=0:p.error('budget must be positive')
    if a.env_file:
        from dotenv import dotenv_values
        os.environ.update({k:v for k,v in dotenv_values(a.env_file).items() if v is not None})
    settings=Settings.from_env();relay=RecheckRelay(settings)
    prompts={}
    if a.baseline:
        if a.prompt_a or a.prompt_b:p.error('baseline excludes candidate variants')
        prompts={'baseline':None}
    else:
        if not a.prompt_a or not a.prompt_b:p.error('A/B requires both prompt files')
        prompts={'A':a.prompt_a.read_text(),'B':a.prompt_b.read_text()}
        if a.split=='locked_test_gpt':
            if not a.winner_evidence:p.error('locked candidates require explicit development winner evidence')
            d=json.loads(a.winner_evidence.read_text())
            if d.get('split')!='development' or d.get('approved_prompt_hash') not in [hashlib.sha256(x.encode()).hexdigest() for x in prompts.values()]:p.error('invalid development winner evidence')
    a.out.mkdir(parents=True,exist_ok=True)
    if (a.out/'results.jsonl').exists() or (a.out/'attempts.jsonl').exists():p.error('output already used; do not repeat paid requests')
    rows=read_rows(a.gold);selected,missing=select_rows(rows,a.split,a.limit,a.seed)
    if not selected:p.error('no usable gold images')
    quota=relay.usage(force=True)
    if quota.get('unit')!='USD':p.error('budget currency could not be verified')
    estimated=len(selected)*len(prompts)*0.15
    if quota['remaining']-settings.recheck_quota_floor<min(a.budget_usd,estimated):p.error('insufficient balance for estimated baseline; no requests sent')
    started=datetime.now(timezone.utc).isoformat();base_balance=quota['remaining'];spent=0;results=[];stop=None
    plan={'split':a.split,'gold_sha256':hashlib.sha256(a.gold.read_bytes()).hexdigest(),'selected_ids':[r['gold_id'] for r in selected],
          'missing_image_ids':missing,'seed':a.seed,'sampling':'round-robin class/population/truth strata; not prevalence representative',
          'versions':{k:hashlib.sha256(v.encode()).hexdigest() if v else settings.recheck_prompt_version for k,v in prompts.items()},
          'started_at':started,'estimated_usd':estimated,'budget_usd':a.budget_usd,'reserve_per_request_usd':a.reserve_per_request_usd,
          'max_false_reject':a.max_false_reject,'baseline_exception':'reference baseline explicitly authorized; no candidate promotion' if a.baseline else None}
    (a.out/'plan.json').write_text(json.dumps(plan,ensure_ascii=False,indent=2)+'\n')
    if a.split=='locked_test_gpt':
        log=a.gold.parent/'test_usage.jsonl'
        with log.open('a') as f:f.write(json.dumps({'record_type':'test_use_started',**plan,'output':str(a.out.resolve())},ensure_ascii=False)+'\n');f.flush();os.fsync(f.fileno())
    # Same samples, alternating A/B order within the same run, not separate days.
    for index,r in enumerate(selected):
        ordered=list(prompts)
        if index%2:ordered.reverse()
        for version in ordered:
            if max(spent,len(results)*a.reserve_per_request_usd)+a.reserve_per_request_usd>a.budget_usd:stop='budget_reservation_exhausted';break
            quota=relay.usage(force=True)
            spent=max(spent,base_balance-quota['remaining'])
            if spent+a.reserve_per_request_usd>a.budget_usd or quota['remaining']<settings.recheck_quota_floor+a.reserve_per_request_usd:stop='budget_or_quota_exhausted';break
            base_prompt,images=prompt_and_images(Path(r['source_image']).read_bytes(),r['bbox'],r['class_name'])
            prompt=base_prompt if prompts[version] is None else prompts[version]+'\n候选类别：'+r['class_name']+'\n坐标仅作定位：'+json.dumps(r['bbox'])
            key=hashlib.sha256((str(a.out.resolve())+r['gold_id']+version).encode()).hexdigest()
            record={k:r[k] for k in ('gold_id','group_id','class_name','population','label')};record.update(version=version,requested_model=settings.auto_annotation_model,status='running',decision=None, image_limitation='未核验独立高清干净来源；源图可能含初审框线', prompt_sha256=hashlib.sha256(prompt.encode()).hexdigest())
            t0=time.monotonic()
            # Write intent first. Crash recovery must not repeat a possibly paid request.
            with (a.out/'attempts.jsonl').open('a') as f:f.write(json.dumps({'gold_id':r['gold_id'],'version':version,'request_key':key,'started_at':datetime.now(timezone.utc).isoformat()})+'\n');f.flush();os.fsync(f.fileno())
            try:
                answer=relay.request(model=settings.auto_annotation_model,prompt=prompt,images=images,schema=RECHECK_SCHEMA,idempotency_key=key)
                advice=parse_advice(answer['answer']);record.update(**advice,actual_model=answer.get('actual_model'),usage=answer.get('usage',{}),status='done')
                if not record['actual_model']:raise ValueError('actual model missing')
            except Exception as e:
                record.update(status='failed',decision=None,error=getattr(e,'code',type(e).__name__));stop='request_failed_no_auto_retry'
            record['elapsed_seconds']=time.monotonic()-t0;results.append(record)
            with (a.out/'results.jsonl').open('a') as f:f.write(json.dumps(record,ensure_ascii=False)+'\n');f.flush();os.fsync(f.fileno())
            try:spent=max(spent,base_balance-relay.usage(force=True)['remaining'])
            except RelayFailure:stop='usage_lost_after_request'
            print(json.dumps({'completed':len(results),'planned':len(selected)*len(prompts),'status':record['status'],'actual_model':record.get('actual_model'),'spent_usd':spent},ensure_ascii=False),flush=True)
            if spent>a.budget_usd:stop='budget_exceeded';break
            if stop:break
        if stop:break
    models=sorted({r.get('actual_model') for r in results if r.get('actual_model')})
    report={'plan':plan,'planned':len(selected)*len(prompts),'completed':len(results),'stop_reason':stop,'actual_models':models,'model_changed':len(models)>1,
            'spent_usd_account_delta':spent,'reserved_usd':len(results)*a.reserve_per_request_usd,'billing_finalized':False,'limitations':['Wilson interval treats boxes independent; same-group samples correlated','A/B paired within same period; report model identity changes','account quota delta may include concurrent usage and delayed billing; not final cost','budget reservation relies on relay charge bound; unexpected overcharge stops immediately','no prompt selected or published'],
            'metrics':aggregate(results,list(prompts),a.max_false_reject,sum(r['class_name']=='车' and r['split']=='locked_test_gpt' and r['label']=='false' for r in rows))}
    (a.out/'report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    lines=['# GPT复检基线评测' if a.baseline else '# GPT复检A/B评测','',f"切分 {a.split}，完成{len(results)}/{report['planned']}；实际模型 {', '.join(models)}；账户余额差额 ${spent:.4f}（计费延迟，非最终费用）。",'',
           '分层限量取样，不代表线上类别和真假比例。Wilson区间按框计算，同组框相关。仅展示模型建议，不选用或发布提示词。', '',
           '|类别/population|版本|框/组|真判假 分子/分母 [95%CI]|假判假 分子/分母 [95%CI]|uncertain 分子/分母 [95%CI]|', '|---|---|---|---|---|---|']
    for k,v in report['metrics'].items():
        for ver in prompts:
            m=v[ver]
            def fmt(metric):
                c=m[metric];return f"{c['numerator']}/{c['denominator']} {c['wilson95']}"
            lines.append(f"|{k}|{ver}|{m['samples']}/{m['groups']}|{fmt('false_reject')}|{fmt('false_target_reject')}|{fmt('uncertain')}|")
    lines+=['','详见report.json的指标、McNemar配对、用量与耗时。锁定车类假目标不足100，只报告假目标判假数字。真判假验收门槛默认Wilson95上界≤1%；样本不足即使零错误也不能保证通过。']
    (a.out/'README.md').write_text('\n'.join(lines)+'\n')
    if a.split=='locked_test_gpt':
        with (a.gold.parent/'test_usage.jsonl').open('a') as f:f.write(json.dumps({'record_type':'test_use_finished','started_at':started,'completed':len(results),'stop_reason':stop,'output':str(a.out.resolve())},ensure_ascii=False)+'\n')

if __name__=='__main__':main()
