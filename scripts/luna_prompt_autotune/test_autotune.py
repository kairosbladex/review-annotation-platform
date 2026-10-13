"""提示词组装、合格标准与 Hermes 提交工具的定向测试：python3 -m unittest test_autotune"""
import json
import copy
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import metrics
import promptlib
import input_evidence
from test_input_evidence import SOURCE, WorkerHarness

HERE = Path(__file__).resolve().parent
OLD = ('逐一核查候选框内目标是否符合初审类别，不把框线、文字当作目标证据。编号图只用于定位；若有第四张原图，只有能唯一对应同一目标时辅助判断。'
       '看不清、遮挡、不能唯一定位则输出uncertain，明确符合类别才输出true，明显不符合输出false。'
       '车辆细分类必须看见所需结构，三轮车须有三轮证据；人员须有可见人体结构。光伏板须有规则组件结构，不推断热故障。'
       '违建没有审批或区域业务依据，一律uncertain；游商等违法属性不能仅凭外观判断。车牌不读号码。'
       '不运行任何工具，只返回schema中的JSON；target_index必须使用下面原编号，解释使用中文。候选：')


class Tests(unittest.TestCase):
    def test_p000_identical_to_production(self):
        c = [{'target_index': 0, 'class_name': '车', 'bbox': [0.1, 0.1, 0.2, 0.2]}]
        self.assertEqual(promptlib.build(json.loads((HERE / 'prompts/p000.json').read_text()), c), OLD + json.dumps(c, ensure_ascii=False))

    def test_only_present_cards_and_aliases(self):
        doc = json.loads((HERE / 'prompts/p001.json').read_text())
        text = promptlib.build(doc, [{'target_index': 0, 'class_name': '路面裂纹', 'bbox': [0, 0, 1, 1]}])
        self.assertIn('[道路裂纹]', text); self.assertNotIn('[人员]', text); self.assertTrue(text.split(promptlib.SUFFIX)[0])
        self.assertEqual(promptlib.validate(doc), [])

    def test_validate_rejects_forcing_and_unknown(self):
        errs = promptlib.validate({'common': 'x', 'cards': {'人员': '看到头顶必须输出true', '飞机': 'y'}})
        self.assertTrue(any('必须输出true' in e for e in errs)); self.assertTrue(any('飞机' in e for e in errs))

    def test_gates(self):
        labels = {f'k{i}': {'label': 'true' if i < 6 else 'false', 'class_name': '人员', 'split': 'dev'} for i in range(12)}
        base_res = {f'k{i}': ('uncertain' if i < 6 else 'false') for i in range(12)}
        cand_res = {f'k{i}': 'true' for i in range(12)}
        items = [{'split': 'unlabeled', 'candidates': [{'key': 'u1', 'class_name': '道路裂纹'}]}]
        base = {'dev': metrics.score(base_res, labels, 'dev'), 'unlabeled': metrics.unlabeled({'u1': 'uncertain'}, items)}
        cand = {'dev': metrics.score(cand_res, labels, 'dev'), 'unlabeled': metrics.unlabeled({'u1': 'true'}, items)}
        g = metrics.gates(cand, [base, base])
        self.assertFalse(g['gates']['G3_人员非人放过不比基线多3个以上']); self.assertFalse(g['passed'])
        self.assertEqual(base['dev']['by_class']['人员']['true_rejected'], 6)

    def test_submit_prompt_tool(self):
        try:
            import mcp_server
        except ModuleNotFoundError as exc:
            if exc.name == 'mcp' or exc.name.startswith('mcp.'):
                self.skipTest('mcp 未安装')
            raise
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        st = Path(temporary.name)
        patch = mock.patch.object(mcp_server, 'STATE', st)
        patch.start(); self.addCleanup(patch.stop)
        for v in ('p000', 'p001'):
            (st / 'prompts').mkdir(exist_ok=True); (st / 'rounds').mkdir(exist_ok=True)
            (st / 'prompts' / f'{v}.json').write_text((HERE / f'prompts/{v}.json').read_text())
            (st / 'rounds' / f'{v}.json').write_text(json.dumps({'status': 'done', 'submitted_at': '2026-10-09T00:00:00'}))
        r = mcp_server.submit_prompt('p001', '改人员卡', {'人员': '新的人员卡'})
        self.assertTrue(r['accepted']); self.assertEqual(r['version'], 'p002'); self.assertEqual(r['changed'], ['人员'])
        self.assertFalse(mcp_server.submit_prompt('p001', '再来', {'车': '新车卡'})['accepted'])  # 已有排队
        (st / 'STOP').write_text('test')
        self.assertTrue(mcp_server.get_state()['stop_requested'])
        mcp_server._wj(st / 'metrics/p002.json', {'dev': {}, 'unlabeled': {}, 'evaluation_status': 'incomplete',
                        'promotion_errors': ['missing evidence'], 'paired_baselines': {'p000': {'diagnostic_only': True}},
                        'holdout': {'secret': 'REDACTED_FOR_GITHUB'}})
        result = mcp_server.get_round_result('p002')
        self.assertEqual(result['evaluation_status'], 'incomplete')
        self.assertTrue(result['paired_baselines']['p000']['diagnostic_only'])
        self.assertNotIn('secret', json.dumps(result))


def fixture():
    labels = {f'{cls}-{i}': {'label': 'true' if i < 5 else 'false', 'class_name': cls, 'split': 'dev'}
              for cls in ('人员', '道路裂纹') for i in range(10)}
    items = [{'unit': key, 'split': 'dev', 'sha256': input_evidence.digest(SOURCE),
              'candidates': [{'key': key, 'class_name': lab['class_name'], 'bbox': [0, 0, 1, 1]}]}
             for key, lab in labels.items()]
    items.append({'unit': 'u', 'split': 'unlabeled', 'candidates': [{'key': 'u', 'class_name': '车', 'bbox': [0, 0, 1, 1]}]})
    candidate = {k: v['label'] for k, v in labels.items()} | {'u': 'false'}
    baseline = dict(candidate, **{'人员-0': 'uncertain', '道路裂纹-0': 'uncertain', 'u': 'uncertain'})
    return labels, items, baseline, candidate


def measurement(results, labels, items):
    units = sum(it['split'] in ('dev', 'unlabeled') for it in items)
    return {'evaluation_sha256': metrics.evaluation_identity(items, labels),
            'input_evidence': {'version': 1, 'complete': True, 'inputs_sha256': 'e' * 64,
                               'expected_units': units, 'verified_units': units, 'invalid_units': {}},
            'dev': metrics.score(results, labels, 'dev'), 'unlabeled': metrics.unlabeled(results, items)}


class MetricTests(unittest.TestCase):
    def setUp(self):
        self.labels, self.items, self.base_res, self.cand_res = fixture()
        self.base = measurement(self.base_res, self.labels, self.items)

    def gates(self, results=None, labels=None, items=None, bases=None):
        cand = measurement(results if results is not None else self.cand_res,
                           labels if labels is not None else self.labels, items if items is not None else self.items)
        return metrics.gates(cand, bases if bases is not None else [self.base, self.base])

    def test_complete_improvement_passes(self):
        self.assertTrue(self.gates()['passed'])

    def test_score_preserves_legacy_fields_and_separates_invalid(self):
        labels = {f'{gold}-{raw}': {'label': gold, 'class_name': '人员', 'split': 'dev'}
                  for gold in ('true', 'false') for raw in metrics.DECISIONS}
        results = {k: k.split('-', 1)[1] for k in labels}
        score = metrics.score(results, labels, 'dev')['overall']
        expected = {'n': 6, 'gold_true': 3, 'gold_false': 3, 'right': 3, 'accuracy': .5,
                    'true_rejected': 2, 'false_released': 1, 'uncertain': 2, 'uncertain_rate': .3333, 'failed': 0}
        self.assertEqual({k: score[k] for k in expected}, expected)
        self.assertEqual(score['confusion'], {g: {r: 1 for r in metrics.DECISIONS} for g in ('true', 'false')})
        self.assertEqual(score['true_retention_rate'], .3333)
        self.assertEqual(score['false_release_rate'], .3333)
        for invalid in ('yes', '', True, {}, []):
            with self.subTest(invalid=invalid):
                results['false-false'] = invalid
                report = metrics.score(results, labels, 'dev')
                self.assertEqual(report['overall']['invalid'], 1)
                self.assertEqual(report['overall']['failed'], 1)
                self.assertEqual(report['overall']['n'], 5)
                self.assertEqual(report['coverage']['invalid_keys'], ['false-false'])

    def test_missing_box_or_entire_class_blocks_promotion(self):
        for removed in (['人员-0'], [k for k in self.cand_res if k.startswith('道路裂纹')], ['u']):
            with self.subTest(removed=removed):
                result = {k: v for k, v in self.cand_res.items() if k not in removed}
                gates = self.gates(result)
                self.assertFalse(gates['passed'])
                self.assertTrue(any('评测不完整' in e for e in gates['comparison_errors']))

    def test_failed_baseline_also_blocks_promotion(self):
        result = dict(self.base_res); result.pop('人员-0')
        base = measurement(result, self.labels, self.items)
        self.assertFalse(self.gates(bases=[self.base, base])['passed'])

    def test_equal_counts_with_different_keys_or_labels_are_not_comparable(self):
        for changed in ('key', 'label', 'class'):
            labels = copy.deepcopy(self.labels)
            results = dict(self.cand_res)
            if changed == 'key':
                labels['replacement'] = labels.pop('人员-0'); results['replacement'] = results.pop('人员-0')
            elif changed == 'label':
                labels['人员-0']['label'] = 'false'; results['人员-0'] = 'false'
            else:
                labels['人员-0']['class_name'] = '车'
            with self.subTest(changed=changed):
                self.assertTrue(self.gates(results, labels)['comparison_errors'])

    def test_input_identity_changes_on_bbox_or_image_but_not_order_or_holdout(self):
        original = metrics.evaluation_identity(self.items, self.labels)
        self.assertEqual(original, metrics.evaluation_identity(list(reversed(self.items)), dict(reversed(list(self.labels.items())))))
        items = copy.deepcopy(self.items); items[0]['sha256'] = 'other-image'
        self.assertNotEqual(original, metrics.evaluation_identity(items, self.labels))
        items = copy.deepcopy(self.items); items[0]['candidates'][0]['bbox'] = [.1, .1, .9, .9]
        self.assertNotEqual(original, metrics.evaluation_identity(items, self.labels))
        self.assertTrue(self.gates(items=items)['comparison_errors'])
        labels = dict(self.labels, secret={'split': 'holdout', 'label': 'true', 'class_name': '人员'})
        self.assertEqual(original, metrics.evaluation_identity(self.items + [{'unit': 'secret', 'split': 'holdout'}], labels))

    def test_missing_repeated_baseline_and_legacy_summary_fail_closed(self):
        for bases in ([], [self.base]):
            self.assertFalse(self.gates(bases=bases)['passed'])
        old = copy.deepcopy(self.base)
        old.pop('evaluation_sha256')
        old['dev'].pop('coverage'); old['unlabeled'].pop('coverage')
        self.assertFalse(self.gates(bases=[old, old])['passed'])

    def test_actual_input_evidence_is_required_for_all_runs(self):
        for missing in ('candidate', 'baseline'):
            cand = measurement(self.cand_res, self.labels, self.items)
            base = copy.deepcopy(self.base)
            (cand if missing == 'candidate' else base).pop('input_evidence')
            gates = metrics.gates(cand, [self.base, base])
            self.assertFalse(gates['passed'])
            self.assertTrue(any('实际模型输入证据' in e for e in gates['comparison_errors']))

    def test_same_manifest_with_different_actual_input_cannot_pass(self):
        base = copy.deepcopy(self.base)
        base['input_evidence']['inputs_sha256'] = 'f' * 64
        gates = self.gates(bases=[self.base, base])
        self.assertFalse(gates['passed'])
        self.assertTrue(any('实际图片' in e for e in gates['comparison_errors']))

    def test_more_true_rejections_cannot_be_offset_by_fewer_false_releases(self):
        for cls in metrics.CRITICAL_CLASSES:
            base = dict(self.cand_res, **{f'{cls}-{i}': 'true' for i in range(5, 10)})
            cand = dict(self.cand_res, **{f'{cls}-0': 'false'})
            baseline = measurement(base, self.labels, self.items)
            gates = self.gates(cand, bases=[baseline, baseline])
            self.assertTrue(gates['gates']['G1_总体准确率高于两次基线中较高者'])
            self.assertEqual(gates['critical_classes'][cls]['status'], 'regressed')
            self.assertFalse(gates['passed'])

    def test_more_false_releases_cannot_be_offset_by_fewer_true_rejections(self):
        for cls in metrics.CRITICAL_CLASSES:
            base = dict(self.cand_res, **{f'{cls}-{i}': 'uncertain' for i in range(5)})
            cand = dict(self.cand_res, **{f'{cls}-5': 'true'})
            baseline = measurement(base, self.labels, self.items)
            gates = self.gates(cand, bases=[baseline, baseline])
            self.assertTrue(gates['gates']['G1_总体准确率高于两次基线中较高者'])
            self.assertEqual(gates['critical_classes'][cls]['status'], 'regressed')
            self.assertFalse(gates['passed'])

    def test_repeated_baseline_range_is_respected(self):
        first = dict(self.base_res, **{'道路裂纹-5': 'true', '人员-5': 'true'})
        second = dict(self.base_res, **{'道路裂纹-1': 'false', '人员-1': 'false'})
        cand = dict(self.cand_res, **{'道路裂纹-5': 'true'})
        bases = [measurement(r, self.labels, self.items) for r in (first, second)]
        self.assertTrue(self.gates(cand, bases=bases)['passed'])

    def test_missing_class_one_sided_or_small_samples_are_insufficient(self):
        for keep in (lambda k: not k.startswith('道路裂纹'),
                     lambda k: k not in [f'道路裂纹-{i}' for i in range(5, 10)],
                     lambda k: k != '道路裂纹-9'):
            labels = {k: v for k, v in self.labels.items() if keep(k)}
            base = measurement(self.base_res, labels, self.items)
            gates = self.gates(labels=labels, bases=[base, base])
            self.assertEqual(gates['critical_classes']['道路裂纹']['status'], 'insufficient_evidence')
            self.assertFalse(gates['passed'])

    def test_aliases_cannot_bypass_critical_gate(self):
        labels = copy.deepcopy(self.labels)
        for lab in labels.values():
            lab['class_name'] = {'人员': '人', '道路裂纹': '路面裂纹'}[lab['class_name']]
        base = measurement(self.base_res, labels, self.items)
        cand = dict(self.cand_res, **{'道路裂纹-0': 'false', '道路裂纹-1': 'false'})
        self.assertEqual(self.gates(cand, labels, bases=[base, base])['critical_classes']['道路裂纹']['status'], 'regressed')

    def test_empty_unlabeled_pool_is_not_zero_uncertainty_failure(self):
        items = [it for it in self.items if it['split'] != 'unlabeled']
        base = measurement(self.base_res, self.labels, items)
        self.assertTrue(self.gates(items=items, bases=[base, base])['passed'])
        base = measurement({}, {}, [])
        self.assertFalse(metrics.gates(base, [base, base])['passed'])

    def test_invalid_gold_and_duplicate_manifest_fail_explicitly(self):
        labels = copy.deepcopy(self.labels); labels['人员-0']['label'] = 'uncertain'
        with self.assertRaisesRegex(ValueError, '金标必须'): metrics.score(self.cand_res, labels, 'dev')
        with self.assertRaisesRegex(ValueError, '重复'): metrics.evaluation_identity(self.items + [self.items[0]], self.labels)
        with self.assertRaisesRegex(ValueError, '重复'): metrics.unlabeled(self.cand_res, self.items + [self.items[-1]])

    def test_invalid_unlabeled_output_is_a_failure_not_false(self):
        result = dict(self.cand_res, u='invalid')
        scored = metrics.unlabeled(result, self.items)
        self.assertEqual(scored['coverage']['invalid_keys'], ['u'])
        self.assertEqual(scored['overall']['failed'], 1)
        self.assertIsNone(scored['overall']['uncertain_rate'])
        self.assertFalse(self.gates(result)['passed'])

    def test_paired_comparison_is_diagnostic_only(self):
        cand = dict(self.cand_res); cand.pop('人员-0')
        pair = metrics.paired_comparison(cand, self.base_res, self.labels, 'dev')
        self.assertTrue(pair['diagnostic_only'])
        self.assertEqual((pair['expected_n'], pair['shared_n'], pair['excluded_n']), (20, 19, 1))
        self.assertEqual(pair['candidate']['overall']['n'], pair['baseline']['overall']['n'])
        self.assertFalse(self.gates(cand)['passed'])


class RunnerTests(unittest.TestCase):
    def setUp(self):
        import runner
        self.runner = runner
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.worker_harness = WorkerHarness(); self.addCleanup(self.worker_harness.close)
        self.doc = {'common': '离线测试', 'cards': {}}
        for name in ('STATE', 'ASSETS', 'RUNS', 'HUMAN'):
            path = self.root / name.lower(); path.mkdir()
            patch = mock.patch.object(runner, name, path); patch.start(); self.addCleanup(patch.stop)
        for name in ('ensure_worker', 'log'):
            patch = mock.patch.object(runner, name); patch.start(); self.addCleanup(patch.stop)
        patch = mock.patch.object(runner.samples, 'build'); patch.start(); self.addCleanup(patch.stop)
        self.labels, self.items, self.base_res, self.cand_res = fixture()
        self.labels['holdout-secret'] = {'label': 'true', 'class_name': '人员', 'split': 'holdout'}
        self.items.append({'unit': 'holdout-secret', 'split': 'holdout',
                           'candidates': [{'key': 'holdout-secret', 'class_name': '人员', 'bbox': [0, 0, 1, 1]}]})
        self.cand_res['holdout-secret'] = 'true'
        self.base_res['holdout-secret'] = 'true'
        runner.wj(runner.ASSETS / 'items.json', self.items)
        runner.wj(runner.ASSETS / 'labels.json', self.labels)
        units = [u for u in self.items if u['split'] in ('dev', 'unlabeled')]
        for version in ('p000', 'p000b'):
            base = measurement(self.base_res, self.labels, self.items) | {'version': version}
            self.write_outputs(version, self.items, self.base_res)
            outputs = [runner.rj(runner.RUNS / version / 'out' / (u['unit'] + '.json')) for u in units]
            base['input_evidence'] = runner.summarize_inputs(outputs, units, self.doc)
            runner.wj(runner.STATE / 'metrics' / f'{version}.json', base)
        runner.wj(runner.STATE / 'prompts/p001.json', dict(self.doc, parent='p000'))
        runner.wj(runner.STATE / 'rounds/p001.json', {'status': 'queued'})
        self.omit = set()
        patch = mock.patch.object(runner, 'run_units', side_effect=self.fake_run)
        self.run_units = patch.start(); self.addCleanup(patch.stop)

    def write_outputs(self, version, units, results):
        for unit in units:
            self.runner.wj(self.runner.RUNS / version / 'out' / (unit['unit'] + '.json'),
                           self.worker_harness.record(unit, results, self.doc))

    def fake_run(self, version, doc, units):
        self.write_outputs(version, [u for u in units if u['unit'] not in self.omit], self.cand_res)

    def execute(self):
        self.runner.execute('p001')
        return self.runner.rj(self.runner.STATE / 'metrics/p001.json')

    def test_complete_candidate_promotes_after_holdout_without_leaking_labels(self):
        result = self.execute()
        self.assertTrue(result['is_new_best'])
        self.assertEqual(self.runner.rj(self.runner.STATE / 'best.json')['version'], 'p001')
        self.assertNotIn('holdout-secret', json.dumps(result))
        self.assertTrue((self.runner.HUMAN / 'holdout-p001.json').exists())

    def test_missing_result_file_counts_as_failure_not_fewer_units(self):
        self.omit = {'人员-0'}
        result = self.execute()
        self.assertEqual(result['units'], 21)
        self.assertEqual(result['failed_units'], 1)
        self.assertEqual(result['evaluation_status'], 'incomplete')
        self.assertFalse(result['is_new_best'])
        self.assertEqual(self.run_units.call_count, 1)
        self.assertFalse((self.runner.STATE / 'best.json').exists())

    def test_successful_file_with_missing_or_invalid_box_never_promotes(self):
        for decision in (None, 'yes'):
            with self.subTest(decision=decision):
                if decision is None: self.cand_res.pop('人员-0', None)
                else: self.cand_res['人员-0'] = decision
                result = self.execute()
                self.assertEqual(result['failed_units'], 0)
                self.assertEqual(result['evaluation_status'], 'incomplete')
                self.assertFalse(result['is_new_best'])

    def test_no_output_does_not_crash_on_elapsed_median(self):
        self.omit = {u['unit'] for u in self.items}
        result = self.execute()
        self.assertEqual(result['failed_units'], 21)
        self.assertIsNone(result['elapsed_median'])
        self.assertFalse(result['is_new_best'])

    def test_holdout_incomplete_does_not_update_best(self):
        self.omit = {'holdout-secret'}
        result = self.execute()
        self.assertFalse(result['is_new_best'])
        self.assertTrue(result['promotion_errors'])
        self.assertFalse((self.runner.STATE / 'best.json').exists())

    def test_stale_file_cannot_supply_a_missing_expected_result(self):
        self.omit = {'人员-0'}
        def run(version, doc, units):
            self.fake_run(version, doc, units)
            self.runner.wj(self.runner.RUNS / version / 'out/stale.json',
                           {'success': True, 'results': [{'key': '人员-0', 'raw': 'true'}], 'elapsed': 1})
        self.run_units.side_effect = run
        result = self.execute()
        self.assertIn('人员-0', result['dev']['coverage']['missing_keys'])
        self.assertFalse(result['is_new_best'])

    def test_failed_or_duplicate_output_cannot_count_as_complete(self):
        for mode in ('failed', 'duplicate', 'foreign_key'):
            def run(version, doc, units):
                self.fake_run(version, doc, units)
                path = self.runner.RUNS / version / 'out/人员-0.json'
                data = self.runner.rj(path)
                if mode == 'failed': data['success'] = False
                elif mode == 'duplicate': data['results'].append(data['results'][0])
                else: data['results'][0]['key'] = '道路裂纹-0'
                self.runner.wj(path, data)
            self.run_units.side_effect = run
            with self.subTest(mode=mode):
                result = self.execute()
                self.assertFalse(result['is_new_best'])
                self.assertIn('人员-0', result['dev']['coverage']['missing_keys'])

    def test_existing_best_requires_strict_improvement(self):
        self.execute()
        previous = self.runner.rj(self.runner.STATE / 'best.json')
        self.run_units.reset_mock()
        result = self.execute()
        self.assertFalse(result['is_new_best'])
        self.assertEqual(self.runner.rj(self.runner.STATE / 'best.json'), previous)
        self.assertEqual(self.run_units.call_count, 1)

    def test_best_from_different_actual_inputs_cannot_be_overwritten(self):
        self.execute()
        previous = self.runner.rj(self.runner.STATE / 'best.json')
        previous.update(inputs_sha256='f' * 64, dev_accuracy=.1)
        self.runner.wj(self.runner.STATE / 'best.json', previous)
        result = self.execute()
        self.assertFalse(result['is_new_best'])
        self.assertTrue(result['promotion_errors'])
        self.assertEqual(self.runner.rj(self.runner.STATE / 'best.json'), previous)

    def test_legacy_best_is_not_silently_overwritten(self):
        old = {'version': 'p002', 'dev_accuracy': .1}
        self.runner.wj(self.runner.STATE / 'best.json', old)
        result = self.execute()
        self.assertFalse(result['is_new_best'])
        self.assertEqual(self.runner.rj(self.runner.STATE / 'best.json'), old)
        self.assertTrue(result['promotion_errors'])

    def test_legacy_outputs_cannot_be_bound_to_current_inputs(self):
        self.write_outputs('p001', self.items[:1], self.cand_res)
        with self.assertRaisesRegex(ValueError, '缺少冻结清单身份'): self.execute()
        self.run_units.assert_not_called()

    def test_resume_rejects_changed_input_manifest(self):
        self.runner.wj(self.runner.STATE / 'rounds/p001.json', {'status': 'running', 'evaluation_sha256': 'old-input'})
        with self.assertRaisesRegex(ValueError, '输入或标签清单已变化'): self.execute()
        self.run_units.assert_not_called()

    def test_resume_rejects_changed_holdout_without_exposing_its_labels(self):
        self.execute()
        self.labels['holdout-secret']['label'] = 'false'
        self.runner.wj(self.runner.ASSETS / 'labels.json', self.labels)
        self.run_units.reset_mock()
        with self.assertRaisesRegex(ValueError, '锁定集清单已变化'): self.execute()
        self.run_units.assert_not_called()

    def test_missing_repeat_baseline_never_promotes(self):
        (self.runner.STATE / 'metrics/p000b.json').unlink()
        result = self.execute()
        self.assertEqual(result['evaluation_status'], 'not_comparable')
        self.assertFalse(result['is_new_best'])


if __name__ == '__main__':
    unittest.main()
