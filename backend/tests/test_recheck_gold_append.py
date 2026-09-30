import importlib.util
import json
import sys
from pathlib import Path
import pytest
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'scripts'))
SPEC=importlib.util.spec_from_file_location('append_gold',ROOT/'scripts/append_recheck_gold.py');module=importlib.util.module_from_spec(SPEC);SPEC.loader.exec_module(module)

def snapshot(tmp):
    out=tmp/'gold';out.mkdir()
    (out/'gold.jsonl').write_text('');(out/'conflicts.jsonl').write_text('');(out/'split_registry.jsonl').write_text('');(out/'summary.json').write_text('{}');(out/'README.md').write_text('# gold\n')
    return out

def row(i,exposure=()):
    return {'candidate_id':str(i),'event_id':str(i),'image_sha256':format(i,'064x'),'bbox':[.1,.1,.2,.2],'class_name':'车','label':'false','label_source':'human_confirmed','labeled_by':'person','labeled_at':'2026-09-30T10:00:00+00:00','population':['random_traffic'],'exposure':list(exposure),'gpt_history_unknown':False}

def test_append_keeps_existing_group_splits_and_exposed_gold_development(tmp_path):
    out=snapshot(tmp_path);incoming=tmp_path/'incoming.jsonl';incoming.write_text(''.join(json.dumps(row(i))+'\n' for i in range(1,11)));module.append(out,incoming)
    previous={r['group_id']:r['split'] for r in module.read_rows(out/'gold.jsonl')}
    incoming.write_text(json.dumps(row(20,['gpt_prompt_exposed']))+'\n');module.append(out,incoming)
    gold=module.read_rows(out/'gold.jsonl');assert len(gold)==11
    assert all(previous[r['group_id']]==r['split'] for r in gold if r['group_id'] in previous)
    assert next(r for r in gold if 'gpt_prompt_exposed' in r['exposure'])['split']=='development'


def test_model_only_or_conflicting_labels_never_modify_gold(tmp_path):
    out=snapshot(tmp_path);incoming=tmp_path/'incoming.jsonl';r=row(1);r['label_source']='gpt';incoming.write_text(json.dumps(r)+'\n')
    with pytest.raises(ValueError,match='human'):module.append(out,incoming)
    assert not (out/'gold.jsonl').read_text()


def test_gpt_exposure_of_duplicate_locked_gold_never_silently_changes_holdout(tmp_path):
    out=snapshot(tmp_path);incoming=tmp_path/'incoming.jsonl'
    incoming.write_text(''.join(json.dumps(row(i))+'\n' for i in range(1,11)));module.append(out,incoming)
    locked=next(r for r in module.read_rows(out/'gold.jsonl') if r['split']=='locked_test_gpt')
    new=row(int(locked['image_sha256'],16),['gpt_prompt_exposed']);incoming.write_text(json.dumps(new)+'\n')
    before=(out/'gold.jsonl').read_bytes()
    with pytest.raises(ValueError,match='locked'):module.append(out,incoming)
    assert (out/'gold.jsonl').read_bytes()==before


def test_missing_history_proof_keeps_new_human_gold_out_of_locked_test(tmp_path):
    out=snapshot(tmp_path);incoming=tmp_path/'incoming.jsonl';r=row(1);r.pop('gpt_history_unknown')
    incoming.write_text(json.dumps(r)+'\n');module.append(out,incoming)
    gold=module.read_rows(out/'gold.jsonl')[0]
    assert gold['gpt_history_unknown'] and gold['split']=='development'
