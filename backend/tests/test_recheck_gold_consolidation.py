"""Gold admission and leakage tests; no API, database or model calls."""
import importlib.util
from pathlib import Path

import pytest

SPEC = importlib.util.spec_from_file_location(
    "consolidation", Path(__file__).resolve().parents[2] / "scripts/consolidate_recheck_gold.py")
module = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(module)


def record(key, label="true", **extra):
    return {"ref": key, "source_file": "data/manual/annotations.jsonl", "source_line": 1,
            "source_sha256": "a" * 64, "source_batch": "manual",
            "raw": {"candidate_id": key, "human_label": label, "labeled_by": "person",
                    "labeled_at": "2026-09-30T12:00:00+08:00", "class_name": "车",
                    "bbox": [.1, .1, .2, .2], "image_sha256": "b" * 64,
                    "source_event_id": "event", **extra}}


@pytest.mark.parametrize("change,reason", [
    ({"labeled_by": ""}, "missing_labeled_by"),
    ({"human_note": " auto annotation consensus: true"}, "auto_annotation_note"),
    ({"label_origin": "gpt"}, "non_human_label_origin"),
    ({"label_source": "pseudo"}, "non_human_label_source"),
    ({"labeled_at": None}, "missing_labeled_at"),
    ({"bbox": [0, 0, float('nan'), 1]}, "missing_or_invalid_normalized_box"),
    ({"image_sha256": "url-not-hash"}, "missing_image_hash"),
])
def test_admission_rejects_invalid_provenance(tmp_path, change, reason):
    gold, excluded, *_ = module.consolidate([record("a", **change)], tmp_path, {})
    assert not gold
    assert reason in excluded[0]["reasons"]


def test_duplicate_and_label_conflict_never_choose_a_label(tmp_path):
    gold, excluded, conflicts, dedup, _ = module.consolidate(
        [record("a"), record("b"), record("c", "看不清")], tmp_path, {})
    assert not gold and not excluded
    assert len(conflicts) == 1
    assert conflicts[0]["labels"] == ["true", "uncertain"]
    assert len(conflicts[0]["provenance"]) == 3
    assert dedup[0]["removed_duplicates"] == 2


def test_iou_boundary_and_class_separation(tmp_path):
    assert module.iou([0, 0, 1, 1], [0, 0, .5, 1]) == .5
    gold, _, _, _, _ = module.consolidate([
        record("a"), record("b"), record("c", class_name="人员")], tmp_path, {})
    assert len(gold) == 2
    assert len(next(r for r in gold if r["class_name"] == "车")["provenance"]) == 2


def test_same_task_and_image_cannot_cross_groups_and_append_keeps_assignment(tmp_path):
    gold, _, _, _, registry = module.consolidate([
        record("a", source_task_id="task1"),
        record("b", "false", bbox=[.3, .3, .4, .4], source_event_id="other", source_task_id="task2"),
        record("c", source_task_id="task1", image_sha256="c"*64)], tmp_path, {})
    assert len({r["group_id"] for r in gold}) == 1
    assert {r["split"] for r in gold} == {"development"}
    new, _, _, _, new_registry = module.consolidate([
        record("d", image_sha256="d"*64, source_task_id="task1")], tmp_path, {}, registry)
    assert new[0]["group_id"] == gold[0]["group_id"]
    assert new_registry[0]["split"] == "development"


def test_connecting_frozen_groups_requires_adjudication(tmp_path):
    registry = [{"group_id": "one", "split": "development", "aliases": ["task:t1"]},
                {"group_id": "two", "split": "development", "aliases": ["event:e1"]}]
    with pytest.raises(ValueError, match="两个已冻结分组"):
        module.consolidate([record("a", source_task_id="t1", source_event_id="e1")], tmp_path, {}, registry)


def test_development_threshold_counts_boxes():
    gold = [{"class_name": "车", "label": label, "group_id": "single-task", "split": "development"}
            for label in ("true", "false") for _ in range(200)]
    stats = module.statistics(gold)["车"]
    assert stats["tier"] == "仅可开发"
    assert stats["independent_groups"]["true"] == 1


def test_local_image_hash_mismatch_is_excluded(tmp_path):
    p = tmp_path / "image.jpg"
    p.write_bytes(b"test-image")
    gold, excluded, *_ = module.consolidate([record("a", source_image=str(p))], tmp_path, {})
    assert not gold
    assert excluded[0]["reasons"] == ["image_hash_mismatch"]


def test_unknown_label_stops_instead_of_guessing(tmp_path):
    with pytest.raises(ValueError, match="未知标签写法"):
        module.consolidate([record("a", "像是真车")], tmp_path, {})


def test_locked_threshold_requires_boxes_and_groups():
    def rows(boxes, groups):
        return [{"class_name":"车", "label":label,"group_id":str(i % groups),"split":"locked_test_gpt"}
                for label in ("true","false") for i in range(boxes)]
    assert module.statistics(rows(100,10))["车"]["tier"] == "可评测"
    assert module.statistics(rows(99,10))["车"]["tier"] == "仅可开发"
    assert module.statistics(rows(100,9))["车"]["tier"] == "仅可开发"


def test_qwen_only_is_gpt_eligible_and_proposal_is_not_frozen(tmp_path):
    records = [record(str(i), source_event_id=str(i), image_sha256=f"{i:064x}") for i in range(12)]
    evidence = {str(i):["qwen_exposed"] for i in range(12)}
    gold, _, _, _, registry = module.consolidate(records,tmp_path,evidence)
    locked = [r for r in gold if r["split"] == "locked_test_gpt"]
    assert len(locked) == 6
    assert all(r["valid_for"] == ["gpt"] for r in locked)
    assert all(r["split"] == "development" for r in registry)
    again = module.consolidate(list(reversed(records)),tmp_path,evidence)[0]
    assert {r["group_id"]:r["split"] for r in again} == {r["group_id"]:r["split"] for r in gold}


def test_gpt_exposure_or_unknown_forces_whole_group_to_development(tmp_path):
    records = [record("a", source_task_id="shared"),
               record("b",source_task_id="shared",image_sha256="c"*64)]
    for evidence in ({"a":["gpt_prompt_exposed"],"b":["qwen_exposed"]}, {"b":["qwen_exposed"]}):
        gold = module.consolidate(records,tmp_path,evidence)[0]
        assert {r["split"] for r in gold} == {"development"}


def test_web_metadata_is_inferred_only_for_nonempty_labels(tmp_path):
    path = tmp_path/'data/recheck-released-fp-20260930/labeling-package/annotations.csv'
    path.parent.mkdir(parents=True)
    path.write_text('candidate_id,label\na,真目标\nb,\n')
    _, records = module.snapshot(path,tmp_path)
    module.enrich_web(records,tmp_path,{})
    inferred = records[0]['normalized_input']
    assert inferred['labeled_by'] == 'wangxiaolin'
    assert inferred['labeled_at'] == records[0]['source_mtime']
    assert inferred['metadata_inferred'] == 'file_level'
    assert 'normalized_input' not in records[1]
    # No original source row is rewritten.
    assert 'labeled_by' not in records[0]['raw']


def test_exposure_classifies_models_separately():
    assert module.direct_exposure('data/qwen/all.jsonl',{}) == ['qwen_exposed']
    assert module.direct_exposure('data/ktd-codex-review/rows.jsonl',{}) == ['gpt_prompt_exposed']
    assert module.direct_exposure('data/frozen/rows.jsonl',{'group':'B'}) == ['gpt_prompt_exposed']
    assert module.direct_exposure('data/frozen/rows.jsonl',{'group':'A'}) == []


def test_legacy_platform_utc_timestamp_is_converted(tmp_path):
    gold = module.consolidate([record('a',labeled_at='2026-09-03 08:00:04.843239')],tmp_path,{})[0]
    assert gold[0]['provenance'][0]['labeled_at'] == '2026-09-03T16:00:04.843239+08:00'
