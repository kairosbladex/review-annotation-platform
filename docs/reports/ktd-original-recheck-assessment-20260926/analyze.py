"""Read-only analysis of frozen real-event results and explicitly non-gold visual notes.
Usage: python analyze.py --source SNAPSHOT_DIR --visual NOTES_JSONL --live LIVE_JSON --output METRICS_JSON
No inference, database writes, human-gold scoring, or network access.
"""
import argparse, collections, hashlib, json
from pathlib import Path


def count(values):
    return dict(collections.Counter(values))


def analyze(source, visual_file, live_file):
    raw = source / '真实样本复核结果.jsonl'
    rows = [json.loads(line) for line in raw.read_text().splitlines() if line]
    visual = [json.loads(line) for line in visual_file.read_text().splitlines() if line]
    live = json.loads(live_file.read_text())
    by_id = {r['sampleId']: r for r in rows}
    assert len(by_id) == len(rows), 'Duplicate candidate identifiers'
    assert len({r['sampleId'] for r in visual}) == len(visual)
    assert {r['sampleId'] for r in visual} == {r['sampleId'] for r in rows if r['cleanSource'].get('applied')}
    assert all(not r['humanGold'] for r in visual)
    for v in visual:
        r = by_id[v['sampleId']]
        assert (v['beforeDecision'], v['afterDecision']) == (r['beforeDecision'], r['decision'])
    baseline_events, baseline_candidates = collections.Counter(), collections.Counter()
    image_sources = []
    mismatches, snapshots = [], []
    for eid in sorted({r['eventId'] for r in rows}):
        visual_dir = source / 'visual' / eid
        image_sources.append({'eventId': eid, 'images': [{'file': str((visual_dir/name).relative_to(source)), 'sha256': hashlib.sha256((visual_dir/name).read_bytes()).hexdigest()} for name in ['annotated-source.jpg', 'clean-source.jpg']]})
        f = source / 'business-v6' / eid / 'before.json'
        x = json.loads(f.read_text())
        result = x.get('result') or {}
        candidates = result.get('resultList') or []
        kinds = collections.Counter('qwen' if 'Qwen3-VL' in c.get('reasoning', '') else 'resnet' if 'resnet18' in c.get('reasoning', '') else 'other' for c in candidates)
        baseline_events['mixed' if len(kinds) > 1 else next(iter(kinds), 'missing')] += 1
        baseline_candidates.update(kinds)
        dec = collections.Counter(c.get('decision') for c in candidates)
        if candidates and any(result.get(k) != dec[v] for k, v in [('trueCount', 'confirmed'), ('falseCount', 'rejected'), ('doubtfulCount', 'uncertain')]):
            mismatches.append(eid)
        for c in candidates:
            sid = eid + ':' + str(c['index'])
            assert sid in by_id and c['decision'] == by_id[sid]['beforeDecision']
        snapshots.append({'eventId': eid, 'file': str(f.relative_to(source)), 'sha256': hashlib.sha256(f.read_bytes()).hexdigest(), 'recheckTime': x.get('recheckTime')})
    pairs = [r for r in rows if r['beforeDecision'] is not None]
    transitions = count(str(r['beforeDecision']) + ' -> ' + r['decision'] for r in rows)
    cross = [dict(visual=v, before=b, after=a, count=n) for (v,b,a),n in collections.Counter((x['visualAssessment'], x['beforeDecision'], x['afterDecision']) for x in visual).items()]
    confirmed_errors = [x for x in visual if x['visualAssessment'] == 'non_target' and x['beforeDecision'] == 'confirmed']
    true_to_manual = [x for x in visual if x['visualAssessment'] == 'target' and x['beforeDecision'] == 'confirmed' and x['afterDecision'] == 'uncertain']
    matches = []
    for event in live:
        old = {r['candidateIndex']: r['decision'] for r in rows if r['eventId'] == event['eventId']}
        new = {c['index']: c['decision'] for c in event['candidates']}
        matches.append(old == new and event['status'] == '2')
    assert len(matches) == len({r['eventId'] for r in rows}) and all(matches), 'Live readback differs from snapshot'
    summary = json.loads((source/'批次汇总.json').read_text())
    assert len(rows) == summary['candidateCount'] and count(r['decision'] for r in rows) == summary['decisionDistribution']
    stats = {
        'scope': 'historical mixed baseline versus latest persisted rollout results; candidate-level',
        'accuracyMeasured': False,
        'usesHumanGold': False,
        'visualReviewType': 'assistant interpretation of original imagery; not independent blind ground truth',
        'eventCount': len(matches), 'candidateCount': len(rows), 'pairedCandidateCount': len(pairs),
        'liveCapturedAt': max(x['capturedAt'] for x in live), 'liveMatchedEvents': sum(matches),
        'allBefore': count(r['beforeDecision'] or 'missing' for r in rows), 'allAfter': count(r['decision'] for r in rows),
        'pairedBefore': count(r['beforeDecision'] for r in pairs), 'pairedAfter': count(r['decision'] for r in pairs),
        'autoCoverageBefore': sum(r['beforeDecision'] in ('confirmed','rejected') for r in pairs)/len(pairs),
        'autoCoverageAfter': sum(r['decision'] in ('confirmed','rejected') for r in pairs)/len(pairs),
        'transitions': transitions,
        'classes': {c: {'total': sum(r['candidate']['class']==c for r in rows), 'before':count(r['beforeDecision'] or 'missing' for r in rows if r['candidate']['class']==c), 'after':count(r['decision'] for r in rows if r['candidate']['class']==c)} for c in sorted({r['candidate']['class'] for r in rows})},
        'versions': {v: {'events':len({r['eventId'] for r in rows if r['releaseVersion']==v}), 'candidates':sum(r['releaseVersion']==v for r in rows)} for v in ['v6','v7','v8']},
        'baselineEvents': dict(baseline_events), 'baselineCandidates':dict(baseline_candidates),
        'baselineSummaryMismatchEvents':mismatches,
        'modelApplied':sum(bool(r['cleanSource'].get('applied')) for r in rows),
        'geometryBlocked':sum(not r['cleanSource'].get('applied') for r in rows),
        'geometryRecoveryApplied':sum(bool(r['cleanSource'].get('geometryRecovery',{}).get('applied')) for r in rows),
        'reasonCounts':count(r['reasonCode'] for r in rows),
        'visualCounts':count(v['visualAssessment'] for v in visual),
        'visualCrosstab':cross,
        'oldVisualFalseConfirmed':confirmed_errors,
        'visuallySupportedTargetsMovedToManual':true_to_manual,
        'visuallySupportedTargetsNewlyConfirmed':[x['sampleId'] for x in visual if x['visualAssessment']=='target' and x['beforeDecision']=='uncertain' and x['afterDecision']=='confirmed'],
        'sources':{
            'resultJsonl':{'file':raw.name,'sha256':hashlib.sha256(raw.read_bytes()).hexdigest()},
            'visualReview':{'file':visual_file.name,'sha256':hashlib.sha256(visual_file.read_bytes()).hexdigest()},
            'liveReadback':{'file':live_file.name,'sha256':hashlib.sha256(live_file.read_bytes()).hexdigest()},
            'beforeSnapshots':snapshots,
            'imageSources':image_sources,
            'relatedEvidence':[{'file':name,'sha256':hashlib.sha256((source/name).read_bytes()).hexdigest()} for name in ['批次汇总.json','training-overlap-audit.json','source-resolution-summary.json','all-pair-background-audit.json','source-coverage-limitations.json','training-provenance.json']]
        }
    }
    return stats

if __name__ == '__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--source',type=Path,required=True);parser.add_argument('--visual',type=Path,required=True)
    parser.add_argument('--live',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    a=parser.parse_args();s=analyze(a.source,a.visual,a.live);a.output.write_text(json.dumps(s,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({k:s[k] for k in ['candidateCount','pairedCandidateCount','liveMatchedEvents','autoCoverageBefore','autoCoverageAfter','visualCounts']},ensure_ascii=False))
