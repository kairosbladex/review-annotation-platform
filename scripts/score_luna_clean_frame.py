#!/usr/bin/env python3
"""按完全配对样本计分，所有缺失、未知标签和不适用类别单列。"""
from __future__ import annotations
import argparse
import collections
import json
from pathlib import Path
import random
import statistics

CONDITIONS = ['screenshot-1', 'screenshot-2', 'plus_clean-1', 'plus_clean-2']


def read(path):
    return json.loads(path.read_text())


def measure(rows, condition):
    true = [r for r in rows if r['label'] == 'true']
    false = [r for r in rows if r['label'] == 'false']
    correct = sum((r['decisions'][condition] == 'true') == (r['label'] == 'true') for r in rows)
    uncertain = [r for r in rows if r['decisions'][condition] == 'uncertain']
    return {'n': len(rows), 'true_n': len(true), 'false_n': len(false), 'correct': correct,
            'accuracy': correct/len(rows) if rows else None,
            'true_rejected': sum(r['decisions'][condition] != 'true' for r in true),
            'false_accepted': sum(r['decisions'][condition] == 'true' for r in false),
            'uncertain': len(uncertain), 'true_uncertain': sum(r['label'] == 'true' for r in uncertain)}


def compare(rows):
    result = {c: measure(rows, c) for c in CONDITIONS}
    for arm in ('screenshot', 'plus_clean'):
        flips = sum(r['decisions'][arm+'-1'] != r['decisions'][arm+'-2'] for r in rows)
        result[arm+'_repeat_flips'] = {'n': flips, 'rate': flips/len(rows) if rows else None}
    deltas = []
    for repeat in (1, 2):
        a, b = f'screenshot-{repeat}', f'plus_clean-{repeat}'
        good = bad = 0
        for r in rows:
            ca = (r['decisions'][a] == 'true') == (r['label'] == 'true')
            cb = (r['decisions'][b] == 'true') == (r['label'] == 'true')
            good += not ca and cb
            bad += ca and not cb
        deltas.append({'repeat': repeat, 'wrong_to_right': good, 'right_to_wrong': bad,
                       'accuracy_delta_pp': 100*(good-bad)/len(rows) if rows else None,
                       'uncertain_delta': result[b]['uncertain']-result[a]['uncertain']})
    result['paired_changes'] = deltas
    # 以原视频/原照片聚类，先平均两次重复；同视频多帧不当成独立场景。
    groups = collections.defaultdict(list)
    for r in rows:
        d = r['decisions']
        gain = sum(((d[f'plus_clean-{n}']=='true') == (r['label']=='true')) -
                   ((d[f'screenshot-{n}']=='true') == (r['label']=='true')) for n in (1, 2))/2
        uncertain_gain = sum((d[f'plus_clean-{n}']=='uncertain') - (d[f'screenshot-{n}']=='uncertain') for n in (1, 2))/2
        groups[r.get('source_local_path', r['event_id'])].append((gain, uncertain_gain))
    if groups:
        rng = random.Random(20261009)
        totals = [(sum(x[0] for x in v), sum(x[1] for x in v), len(v)) for v in groups.values()]
        boot, uncertain_boot = [], []
        for _ in range(2000):
            sampled = [rng.choice(totals) for _ in totals]
            boot.append(100*sum(x[0] for x in sampled)/sum(x[2] for x in sampled))
            uncertain_boot.append(100*sum(x[1] for x in sampled)/sum(x[2] for x in sampled))
        boot.sort()
        uncertain_boot.sort()
        result['mean_delta_pp_cluster_bootstrap95'] = [boot[49], boot[1949]]
        result['mean_uncertain_delta_pp_cluster_bootstrap95'] = [uncertain_boot[49], uncertain_boot[1949]]
        result['unique_source_files'] = len(groups)
    return result


def score(root):
    included, labels, all_items = read(root/'included.json'), read(root/'labels.json'), read(root/'all-items.json')
    outputs = {}
    for condition in CONDITIONS:
        outputs[condition] = {p.stem: read(p) for p in (root/'results'/condition).glob('*.json')}
    records = [r for group in outputs.values() for r in group.values()]
    execution_ids = [r['execution_id'] for r in records if r['success'] and r.get('execution_id')]
    assert len(execution_ids) == len(set(execution_ids)), 'reused model execution across requests'
    rows, incomplete, unlabeled = [], [], []
    for u in included:
        uid = u['unit']
        if not all(outputs[c].get(uid, {}).get('success') for c in CONDITIONS):
            incomplete.append(uid)
            continue
        results = {c: {r['key']: r for r in outputs[c][uid]['results']} for c in CONDITIONS}
        assert len({outputs[c][uid]['prompt_sha256'] for c in CONDITIONS}) == 1, 'prompt changed across arms'
        for candidate in u['candidates']:
            key = candidate['key']
            assert all(key in results[c] for c in CONDITIONS), key
            lab = labels.get(key)
            if not lab or lab['label'] not in ('true', 'false'):
                unlabeled.append({'unit': uid, 'key': key, 'split':u['split'], 'class_name':candidate['class_name'],
                                  'decisions':{c:results[c][key]['decision'] for c in CONDITIONS}})
                continue
            assert lab['split'] == u['split'] and lab['class_name'] == candidate['class_name']
            rows.append({'unit': uid, 'event_id': u['event_id'], 'key': key, 'split': u['split'], 'class_name': lab['class_name'], 'source': u['source'],
                         'source_kind': u['source_kind'], 'source_local_path': u['source_local_path'], 'label': lab['label'], 'decisions': {c: results[c][key]['decision'] for c in CONDITIONS},
                         'details': {c: results[c][key] for c in CONDITIONS}})
    full_classes = sorted({v['class_name'] for v in labels.values()} | {c['class_name'] for u in included for c in u['candidates']})
    report = {'conditions': CONDITIONS, 'included_images': len(included), 'complete_images': len(included)-len(incomplete), 'incomplete_images': incomplete,
              'unscored_candidates': unlabeled, 'scored_candidates': len(rows), 'splits': {},
              'input_counts': dict(collections.Counter(f"{u['split']}/{u['source']}" for u in all_items)),
              'included_counts': dict(collections.Counter(f"{u['split']}/{u['source']}" for u in included))}
    report['requests'] = {'records':len(records),'success':sum(r['success'] for r in records),
                          'failed':sum(not r['success'] for r in records),
                          'attempts':sum(len(r['attempts']) for r in records),
                          'unique_execution_ids':len(execution_ids),
                          'models':sorted({r.get('requested_model','unknown') for r in records if r['success']}),
                          'latency_median_seconds':{c:statistics.median(a['seconds'] for r in group.values() for a in r['attempts']) if group else None for c,group in outputs.items()}}
    all_rows = rows + unlabeled
    report['all_candidate_uncertain'] = {s:{cls:{c:sum(r['split']==s and r['class_name']==cls and r['decisions'][c]=='uncertain' for r in all_rows)
                                                      for c in CONDITIONS} for cls in full_classes} for s in ('dev','holdout')}
    for split in ('dev', 'holdout'):
        sub = [r for r in rows if r['split'] == split]
        report['splits'][split] = {'overall': compare(sub), 'classes': {cls: compare([r for r in sub if r['class_name']==cls]) for cls in full_classes},
                                  'source_kind': {kind: compare([r for r in sub if r['source_kind']==kind]) for kind in ('video', 'still')}}
    events = {s: {u['event_id'] for u in included if u['split']==s} for s in ('dev', 'holdout')}
    overlap = events['dev'] & events['holdout']
    report['cross_split_event_overlap'] = sorted(overlap)
    sources = {s: {u['source_local_path'] for u in included if u['split']==s} for s in ('dev', 'holdout')}
    report['cross_split_source_overlap'] = sorted(sources['dev'] & sources['holdout'])
    report['holdout_without_cross_split_events'] = compare([r for r in rows if r['split']=='holdout' and r['event_id'] not in overlap])
    for name, value in [('report.json', report), ('per-box.json', rows)]:
        (root/name).write_text(json.dumps(value, ensure_ascii=False, indent=1)+'\n')
    print(json.dumps({k: report[k] for k in ('included_images', 'complete_images', 'scored_candidates', 'incomplete_images', 'cross_split_event_overlap')}, ensure_ascii=False))


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--evidence', type=Path, required=True)
    score(ap.parse_args().evidence)
