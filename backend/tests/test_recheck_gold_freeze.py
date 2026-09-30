"""No network or models: test grouping, exposure, freezing and snapshot integrity."""
import importlib.util
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'scripts'))
SPEC = importlib.util.spec_from_file_location('freeze_split', ROOT / 'scripts/freeze_recheck_gold_split.py')
module = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(module)


def row(gid, label='true', cls='车', pop='random_traffic', exposure=(), unknown=False):
    return {'group_id': gid, 'label': label, 'class_name': cls, 'population': [pop],
            'group_exposure': list(exposure), 'gpt_history_unknown': unknown,
            'group_aliases': ['event:' + gid]}


def test_rates_are_stratified_and_input_order_independent():
    rows = [row(f'{label}-{pop}-{i}', label, pop=pop)
            for label in ('true', 'false') for pop in ('random_traffic', 'rooftop_hard_case')
            for i in range(10)]
    assignments, audit = module.stratified_assignments(rows)
    assert len(audit) == 4
    assert all(s['locked_groups'] == (7 if s['has_false'] else 5) for s in audit)
    assert module.stratified_assignments(list(reversed(rows))) == (assignments, audit)


def test_mixed_groups_are_not_split_and_exposure_stays_conservative():
    rows = [row('mixed'), row('mixed', 'false', cls='卡车', pop='rooftop_hard_case'),
            row('gpt', exposure=['gpt_prompt_exposed']), row('unknown', unknown=True),
            row('uncertain', 'uncertain')]
    assignments, audit = module.stratified_assignments(rows)
    assert audit[0]['classes'] == ['卡车']
    assert audit[0]['population'] == ['random_traffic']
    assert audit[0]['has_false']
    for gid in ('gpt', 'unknown', 'uncertain'):
        assert assignments[gid]['split'] == 'development'
        assert not assignments[gid]['valid_for']
    qwen = [row(str(i), exposure=['qwen_exposed']) for i in range(10)]
    assigned, _ = module.stratified_assignments(qwen)
    assert sum(r['split'] == 'locked_test_gpt' for r in assigned.values()) == 5
    assert all(r['valid_for'] == ['gpt'] for r in assigned.values() if r['split'] == 'locked_test_gpt')


def test_frozen_snapshot_registry_and_side_effects_are_consistent():
    out = ROOT / 'data/recheck-gold-consolidated-20260930'
    report = json.loads((out / 'summary.json').read_text())
    gold = module.read_rows(out / 'gold.jsonl')
    conflicts = module.read_rows(out / 'conflicts.jsonl')
    registry = module.read_rows(out / 'split_registry.jsonl')
    latest = {r['group_id']: r for r in registry}
    assignments, audit = module.stratified_assignments(gold)
    assert report['stratified_split'] == audit
    assert len(gold) == 1453 and len(conflicts) == 5
    assert all(r['split'] == 'quarantine' and r['valid_for'] == [] for r in conflicts)
    for r in gold:
        assert r['split'] == assignments[r['group_id']]['split'] == latest[r['group_id']]['split']
        assert set(r['group_aliases']) <= set(latest[r['group_id']]['aliases'])
        if r['split'] == 'locked_test_gpt':
            assert not r['gpt_history_unknown']
            assert 'gpt_prompt_exposed' not in r['group_exposure']
            assert r['split_status'] == latest[r['group_id']]['status'] == 'frozen'
    assert not any(report['side_effects'].values())
    assert report['classes']['车']['locked_test_samples']['false'] < 100
    assert report['vehicle_evaluation_policy']['false_target_pass_minimum'] == 100
    before = (out / 'SHA256SUMS').read_bytes()
    with pytest.raises(ValueError, match='已冻结'):
        module.freeze(out)
    assert (out / 'SHA256SUMS').read_bytes() == before


def test_small_strata_use_largest_remainder():
    rows = [row(str(i), 'false', cls=str(i)) for i in range(10)]
    assignments, audit = module.stratified_assignments(rows)
    assert sum(a['split'] == 'locked_test_gpt' for a in assignments.values()) == 7
    assert sum(s['locked_groups'] for s in audit) == 7
