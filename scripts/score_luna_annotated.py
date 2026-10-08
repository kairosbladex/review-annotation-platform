#!/usr/bin/env python3
"""Strict candidate-ID comparison of frozen human labels and three offline paths.

Manual referrals are neither right nor wrong. Report coverage and denominators;
never silently turn missing/failed results into negative predictions.
"""
from __future__ import annotations
import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import statistics

TRUE = {'真目标', '真车'}
FALSE = {'误检', '非车'}
BASE = Path('data/video-clean-source-eval-20261008')
OUT = Path('data/luna-eval-20261008')

def read(path): return json.loads(path.read_text())
def save(path, value): path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True)+'\n')
def ratio(a,b): return a/b if b else None

def load_results(directory, expected, *, luna=False):
    decisions, events, failures = {}, {}, {}
    for path in sorted(directory.glob('*.json')):
        item=read(path);eid=item['event_id']
        if eid not in expected: raise ValueError(f'unexpected event {eid}')
        if eid in events: raise ValueError(f'duplicate event {eid}')
        events[eid]=item
        ids=item['candidate_ids']; wanted=[d['candidate_id'] for d in expected[eid]['detections']]
        # Each directory must preserve the original request order, no positional guessing.
        if ids != wanted: raise ValueError(f'candidate ordering mismatch {eid}')
        resp=item['response']
        if resp.get('success') is not True:
            failures[eid]=resp.get('errorCode','execution_failed');continue
        results=resp.get('results',[])
        if len(results)!=len(ids): raise ValueError(f'result count mismatch {eid}')
        for cid,d,res in zip(ids,expected[eid]['detections'],results):
            if cid in decisions: raise ValueError(f'duplicate candidate {cid}')
            if res.get('original_class')!=d['class_name'] or res.get('original_bbox')!=d['bbox']:
                raise ValueError(f'candidate geometry/class mismatch {cid}')
            decision=res.get('decision')
            if decision not in {'confirmed','rejected','uncertain'}: raise ValueError(f'unknown decision {cid}')
            if luna:
                raw=res.get('rawDecision')
                if raw not in {'true','false','uncertain'}: raise ValueError(f'unknown raw decision {cid}')
                if decision != ('confirmed' if raw=='true' else 'rejected'): raise ValueError(f'policy mismatch {cid}')
                if resp.get('modelVersion')!='gpt-5.6-luna': raise ValueError('model mismatch')
            decisions[cid]=res
    return decisions,events,failures

def state(truth, decision):
    if decision is None: return 'missing'
    if decision=='uncertain': return 'manual'
    return 'right' if (decision=='confirmed')==truth else 'wrong'

def score(labels, predictions):
    c=Counter(); classes={True:Counter(),False:Counter()}
    for cid,truth in labels.items():
        decision=predictions.get(cid);s=state(truth,decision);c[s]+=1;classes[truth][s]+=1
    n=len(labels);decided=c['right']+c['wrong'];available=n-c['missing']
    return dict(total=n,available=available,right=c['right'],wrong=c['wrong'],manual=c['manual'],missing=c['missing'],
        true_targets=sum(classes[True].values()),non_targets=sum(classes[False].values()),
        wrongly_rejected=classes[True]['wrong'],wrongly_confirmed=classes[False]['wrong'],
        true_manual=classes[True]['manual'],false_manual=classes[False]['manual'],
        decided=decided,accuracy_decided=ratio(c['right'],decided),
        correct_fraction_available=ratio(c['right'],available),coverage_available=ratio(decided,available),
        false_rejection_rate=ratio(classes[True]['wrong'],sum(classes[True].values())-classes[True]['missing']),
        false_acceptance_rate=ratio(classes[False]['wrong'],sum(classes[False].values())-classes[False]['missing']))

def compare(labels, before, after):
    transitions=Counter();paired={}
    for cid,truth in labels.items():
        a,b=state(truth,before.get(cid)),state(truth,after.get(cid))
        transitions[f'{a}->{b}']+=1
        if a in {'right','wrong'} and b in {'right','wrong'}: paired[cid]=truth
    return dict(transitions=dict(sorted(transitions.items())),jointly_decided=len(paired),
                before_jointly_decided=score(paired,before),after_jointly_decided=score(paired,after))

def timing(events):
    times=[e['elapsed'] for e in events if e['response'].get('success') is True]
    times.sort()
    return dict(successful_events=len(times),median_seconds=statistics.median(times) if times else None,
                p90_seconds=times[math.ceil(.9*len(times))-1] if times else None,
                max_seconds=max(times) if times else None,quantile='nearest rank ceil(0.9*n); median arithmetic middle')

def main():
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,default=OUT);args=p.parse_args();out=args.out
    manifest=read(BASE/'manifest.json');selection=read(BASE/'selection.json');labels=selection['labels']
    expected={e['event_id']:e for e in manifest}
    ids=[d['candidate_id'] for e in manifest for d in e['detections']]
    assert len(expected)==113 and len(ids)==len(set(ids))==212 and set(ids)==set(labels)
    assert set(selection['events'])==set(expected)
    assert all(selection['events'][e]==[d['candidate_id'] for d in expected[e]['detections']] for e in expected)
    for v in labels.values():
        if v['group'] not in {'person','vehicle'} or v['label'] not in TRUE|FALSE|{'看不清'}:
            raise ValueError('unknown human label')
    directories={'luna':out/'out-luna','video':BASE/'out-video',
        'legacy_vehicle':BASE/'out-legacy','legacy_person':Path('data/person-labeling-20261008/eval/out-single')}
    loaded={k:load_results(v,expected,luna=k=='luna') for k,v in directories.items()}
    luna,le,lf=loaded['luna'];video,ve,vf=loaded['video']
    old={**loaded['legacy_person'][0],**loaded['legacy_vehicle'][0]}
    assert not (set(loaded['legacy_person'][0]) & set(loaded['legacy_vehicle'][0]))
    mapping={'luna_online':{c:r['decision'] for c,r in luna.items()},
             'luna_manual':{c:'uncertain' if r['rawDecision']=='uncertain' else r['decision'] for c,r in luna.items()},
             'legacy':{c:r['decision'] for c,r in old.items()},'video':{c:r['decision'] for c,r in video.items()}}
    # A counterfactual policy for an equal binary comparison, not a claim about the video service.
    mapping['video_uncertain_as_false']={c:'rejected' if d=='uncertain' else d for c,d in mapping['video'].items()}
    report={'events':113,'boxes':212,'groups':{},'execution':{},'timing':{},'input_sha256':{}}
    for name,(ds,ev,fail) in loaded.items():
        wanted=set(expected) if name in {'luna','video'} else {
            e['event_id'] for e in manifest if any(labels[d['candidate_id']]['group']==name.split('_')[1] for d in e['detections'])}
        report['execution'][name]=dict(expected_events=len(wanted),present_events=len(ev),
            failed_events=len(fail),failure_reasons=dict(Counter(fail.values())),failures=fail,
            missing_events=sorted(wanted-set(ev)))
        report['timing'][name]=timing(list(ev.values()))
    report['execution']['luna']['attempt_statuses']=dict(Counter(a['status'] for e in le.values() for a in e.get('attempts',[])))
    report['execution']['luna']['retried_events']=sum(len(e.get('attempts',[]))>1 for e in le.values())
    rows=[]
    for e in manifest:
        for d in e['detections']:
            c=d['candidate_id'];lab=labels[c];truth=lab['label'] in TRUE if lab['label']!='看不清' else None
            rows.append(dict(event_id=e['event_id'],candidate_id=c,group=lab['group'],gold=lab['label'],
                scorable=truth is not None,class_name=d['class_name'],bbox=d['bbox'],
                decisions={k:v.get(c) for k,v in mapping.items()},
                states={k:state(truth,v.get(c)) if truth is not None else 'excluded_gold_unclear' for k,v in mapping.items()},
                luna_raw=luna.get(c,{}).get('rawDecision'),luna_reasoning=luna.get(c,{}).get('reasoning'),
                luna_observed_object=luna.get(c,{}).get('observedObject')))
    for group in ['person','vehicle']:
        all_ids={c for c,l in labels.items() if l['group']==group}
        truth={c:labels[c]['label'] in TRUE for c in all_ids if labels[c]['label'] in TRUE|FALSE}
        raw=Counter(luna[c]['rawDecision'] for c in all_ids if c in luna)
        raw_scored=Counter(luna[c]['rawDecision'] for c in truth if c in luna)
        g=dict(labels=dict(Counter(labels[c]['label'] for c in all_ids)),
            raw_by_gold={label:dict(Counter(luna[c]['rawDecision'] for c in all_ids if c in luna and labels[c]['label']==label)) for label in sorted({labels[c]['label'] for c in all_ids})},
            scores={k:score(truth,v) for k,v in mapping.items()},
            uncertain=dict(all_successful_boxes=raw,scorable_successful_boxes=raw_scored,
                fraction_all=ratio(raw['uncertain'],sum(raw.values())),fraction_scorable=ratio(raw_scored['uncertain'],sum(raw_scored.values()))),
            comparisons={f'{before}_to_{after}':compare(truth,mapping[before],mapping[after])
                         for before in ['legacy','video','video_uncertain_as_false'] for after in ['luna_online','luna_manual']},
            timing={k:timing([v for e,v in ev.items() if any(c in all_ids for c in v['candidate_ids'])])
                    for k,(_,ev,_) in loaded.items() if k in {'luna','video','legacy_'+group}},
            class_breakdown={cls:score({c:t for c,t in truth.items() if next(d['class_name'] for e in manifest for d in e['detections'] if d['candidate_id']==c)==cls},mapping['luna_online'])
                             for cls in sorted({r['class_name'] for r in rows if r['group']==group})})
        report['groups'][group]=g
    for path in [BASE/'manifest.json',BASE/'selection.json',out/'allowlist.json']+[f for directory in directories.values() for f in sorted(directory.glob('*.json'))]:
        report['input_sha256'][str(path)]=hashlib.sha256(path.read_bytes()).hexdigest()
    save(out/'report.json',report);save(out/'per-box.json',rows)
    print(json.dumps({g:report['groups'][g]['scores'] for g in report['groups']},ensure_ascii=False,indent=2))

if __name__=='__main__': main()
