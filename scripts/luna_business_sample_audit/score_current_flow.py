"""逐框对照人工金标；失败、未执行和 uncertain 分开计数。"""
from __future__ import annotations
import argparse
import collections
import csv
import json
from pathlib import Path
from current_flow import save, validate


def score(root: Path):
    units = json.loads((root/'private/units.json').read_text())
    run = root/'private/server'
    matrices = collections.defaultdict(lambda: collections.defaultdict(collections.Counter))
    all_candidates = collections.defaultdict(collections.Counter)
    event_status = collections.Counter()
    detail = []
    for ordinal, unit in enumerate(units, 1):
        uid = unit['unit']
        attempt = run/'attempts'/f'{uid}.json'
        status = json.loads(attempt.read_text()).get('status') if attempt.exists() else 'not_run'
        response = run/'responses'/f'{uid}.json'
        results = []
        if status == 'ok':
            results = validate(json.loads(response.read_text()), unit['candidates'])
        event_status[status] += 1
        for index, candidate in enumerate(unit['candidates']):
            raw = results[index]['rawDecision'] if results else status
            all_candidates[candidate['class_name']][raw] += 1
        for index, gold in unit['gold'].items():
            raw = results[int(index)]['rawDecision'] if results else status
            matrices[gold['class_name']][gold['label']][raw] += 1
            detail.append(dict(event=f'E{ordinal:03}', target_index=int(index),
                class_name=gold['class_name'], gold=gold['label'], raw_decision=raw,
                business_decision=('confirmed' if raw == 'true' else 'rejected')
                    if raw in {'true', 'false', 'uncertain'} else '',
                split=unit['split'], source=unit['source']))
    totals = collections.defaultdict(collections.Counter)
    for matrix in matrices.values():
        for gold, decisions in matrix.items():
            totals[gold].update(decisions)
    metrics = dict(events=event_status, labeled_boxes=len(detail),
                   initial_candidates=sum(len(x['candidates']) for x in units),
                   unlabeled_candidates=sum(len(x['candidates'])-len(x['gold']) for x in units),
                   gold_matrix=totals, gold_matrix_by_class=matrices,
                   raw_decisions_all_candidates=all_candidates,
                   prior_pre_model_http_failures=len(list((run/'prior-attempts').glob('*.json'))),
                   preserved_failed_model_attempts=len(list((run/'retry-history').glob('*.json'))),
                   total_model_attempts=len(list((run/'attempts').glob('*.json'))) + len(list((run/'retry-history').glob('*.json'))),
                   policy='only_true_is_true_v1', business_database_writeback=False,
                   event_period=[min(x['event_time'] for x in units), max(x['event_time'] for x in units)])
    save(root/'metrics.json', metrics)
    with (root/'per-box.csv').open('w') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(detail[0]))
        writer.writeheader();writer.writerows(detail)
    print(json.dumps(metrics, ensure_ascii=False))
    return metrics


if __name__ == '__main__':
    parser = argparse.ArgumentParser();parser.add_argument('--root', required=True, type=Path)
    score(parser.parse_args().root)
