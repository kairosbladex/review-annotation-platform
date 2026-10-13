"""人员视图与类别卡的离线测试。不调用模型，不把缺失结果当成判假。"""
import base64
import io
import os
import subprocess
import unittest
from pathlib import Path
from unittest import mock

from PIL import Image

import input_evidence
import metrics
import person_candidate
import person_experiment
import person_view
import promptlib
import view_protocol
import luna_experiment_worker as worker
from test_input_evidence import SOURCE, WorkerHarness


def scene():
    image = Image.new('RGB', (960, 720), (40, 120, 40))
    image.paste(Image.new('RGB', (80, 40), (0, 0, 200)), (500, 300))
    image.paste(Image.new('RGB', (11, 22), (220, 20, 20)), (400, 340))
    person = {'target_index': 0, 'class_name': '人员', 'bbox': [400 / 960, 340 / 720, 411 / 960, 362 / 720]}
    car = {'target_index': 1, 'class_name': '车', 'bbox': [100 / 960, 100 / 720, 180 / 960, 160 / 720]}
    return image, [person, car]


def cell(image, index):
    x, y = (index % 5) * 256, (index // 5) * 280
    return image.crop((x, y, x + 256, y + 280))


class ViewRenderingTests(unittest.TestCase):
    def test_no_person_matches_legacy_pixels_and_does_not_upscale(self):
        image, candidates = scene()
        cars = [dict(candidates[1], target_index=0)]
        legacy, _ = person_view.legacy_sheet(image, cars)
        rendered = person_view.render_sheet(image, cars)
        self.assertFalse(rendered.changed)
        self.assertEqual(rendered.sheet.tobytes(), legacy.tobytes())
        self.assertFalse(rendered.records[0]['upscaled'])
        self.assertLessEqual(rendered.records[0]['scale'], 1)

    def test_small_person_is_bicubic_upscaled_with_index_outside_and_context(self):
        image, candidates = scene()
        rendered = person_view.render_sheet(image, candidates)
        local = next(r for r in rendered.records if r['role'] == 'fixed_local')
        context = next(r for r in rendered.records if r['role'] == 'fixed_context')
        self.assertEqual(local['bbox'], candidates[0]['bbox'])
        self.assertEqual(local['original_pixel_size'], [11, 22])
        self.assertGreater(local['scale'], 1)
        self.assertTrue(local['upscaled'])
        self.assertEqual(local['interpolation'], 'bicubic')
        self.assertEqual(local['source_kind'], 'screenshot')
        photo = rendered.sheet.crop((local['paste_xy'][0], local['paste_xy'][1],
                                     local['paste_xy'][0] + 256, local['paste_xy'][1] + 256))
        self.assertEqual(photo.getpixel((128, 128))[:2], (220, 20))
        header = rendered.sheet.crop((0, 0, 256, 24))
        self.assertEqual(header.getpixel((128, 23)), (255, 255, 255))
        # Pillow 11 默认字体是抗锯齿灰度，最深约 (20,20,20)，不是纯黑。
        self.assertTrue(any(max(header.getpixel((x, y))) < 40 for y in range(12) for x in range(40)))
        for y in range(12, 24):
            self.assertEqual(header.getpixel((2, y)), (255, 255, 255))
        self.assertLess(context['crop_box'][0], local['crop_box'][0])
        self.assertGreater(context['crop_box'][2], local['crop_box'][2])
        context_photo = rendered.sheet.crop((context['paste_xy'][0], context['paste_xy'][1],
                                              context['paste_xy'][0] + 256, context['paste_xy'][1] + 256))
        colors = {context_photo.getpixel((x, y)) for x in range(0, 256, 8) for y in range(0, 256, 8)}
        self.assertTrue(any(px[1] > 80 and px[0] < 80 for px in colors))
        self.assertEqual(context['target_index'], 0)

    def test_pillow9_without_resampling_enum_still_upscales(self):
        image, candidates = scene()
        real = person_view.Image

        class Pillow9:
            def __getattr__(self, name):
                if name == 'Resampling':
                    raise AttributeError(name)
                return getattr(real, name)

        rendered = None
        resampled = None
        person_view.Image = Pillow9()
        try:
            resampled = person_view._resample()
            rendered = person_view.render_sheet(image, candidates)
        finally:
            person_view.Image = real
        self.assertEqual(resampled, real.BICUBIC)
        local = next(row for row in rendered.records if row['role'] == 'fixed_local')
        self.assertGreater(local['scale'], 1)
        self.assertEqual(local['interpolation'], 'bicubic')

    def test_aspect_ratio_is_preserved_with_padding(self):
        image = Image.new('RGB', (200, 80), (10, 10, 10))
        image.paste(Image.new('RGB', (10, 40), (240, 240, 240)), (20, 20))
        candidate = {'target_index': 3, 'class_name': '人', 'bbox': [20 / 200, 20 / 80, 30 / 200, 60 / 80]}
        rendered = person_view.render_sheet(image, [candidate])
        local = rendered.records[0]
        photo = rendered.sheet.crop((local['paste_xy'][0], local['paste_xy'][1],
                                     local['paste_xy'][0] + 256, local['paste_xy'][1] + 256))
        self.assertEqual(photo.getpixel((0, 128)), person_view.PAD)
        self.assertEqual(photo.getpixel((128, 128)), (240, 240, 240))
        self.assertEqual(local['target_index'], 3)

    def test_boundary_boxes_stay_inside_the_frame(self):
        image, _ = scene()
        candidates = [
            {'target_index': 0, 'class_name': '人员', 'bbox': [0, 0, 0.004, 0.006]},
            {'target_index': 1, 'class_name': '人员', 'bbox': [0.99, 0.985, 1, 1]},
        ]
        rendered = person_view.render_sheet(image, candidates)
        for record in rendered.records:
            box = record['crop_box']
            self.assertGreaterEqual(box[0], 0)
            self.assertGreaterEqual(box[1], 0)
            self.assertLessEqual(box[2], image.width)
            self.assertLessEqual(box[3], image.height)

    def test_mixed_class_keeps_non_person_cell_and_original_index(self):
        image, candidates = scene()
        legacy, legacy_records = person_view.legacy_sheet(image, candidates)
        rendered = person_view.render_sheet(image, candidates)
        self.assertEqual(cell(rendered.sheet, 1).tobytes(), cell(legacy, 1).tobytes())
        self.assertNotEqual(cell(rendered.sheet, 0).tobytes(), cell(legacy, 0).tobytes())
        self.assertEqual(rendered.records[1]['role'], 'legacy_thumbnail')
        self.assertEqual(rendered.records[1]['target_index'], 1)
        self.assertEqual(rendered.records[1]['paste_xy'], legacy_records[1]['paste_xy'])

    def test_person_limit_fails_closed(self):
        image, _ = scene()
        candidates = [{'target_index': i, 'class_name': '人员', 'bbox': [0.1, 0.1, 0.2, 0.2]} for i in range(17)]
        with self.assertRaises(person_view.PersonViewLimit):
            person_view.render_sheet(image, candidates)

    def test_renderer_has_no_generative_upscaler(self):
        text = Path(person_view.__file__).read_text(encoding='utf-8')
        for banned in ('urllib', 'esrgan', 'realesrgan', 'requests', 'subprocess'):
            self.assertNotIn(banned, text)


class CandidateCardTests(unittest.TestCase):
    def test_card_rules_and_unchanged_non_person_prompt(self):
        self.assertEqual(person_candidate.validate_card(), [])
        base = promptlib.build(json_p000(), [{'class_name': '车', 'bbox': [0, 0, 1, 1]}])
        self.assertEqual(person_candidate.build_prompt([{'class_name': '车', 'bbox': [0, 0, 1, 1]}]), base)
        mixed = person_candidate.build_prompt([
            {'class_name': '人员', 'bbox': [0, 0, 0.1, 0.1]},
            {'class_name': '车', 'bbox': [0.2, 0.2, 0.3, 0.3]}])
        self.assertIn('[人员]', mixed)
        self.assertNotIn('[车]', mixed)
        self.assertIn('正俯视', mixed)
        self.assertIn('uncertain', mixed)

    def test_references_come_from_dev_only_and_are_not_leaked(self):
        items, labels = reference_fixture()
        selected = person_candidate.select_references(items, labels, {'e': 'task-shared', 'h': 'task-shared'})
        self.assertTrue(selected['ok'])
        self.assertEqual([r['caption'] for r in selected['references']], ['P1', 'P2', 'N1', 'N2'])
        self.assertTrue({'a', 'b', 'c', 'd'} >= {r['unit'] for r in selected['references']})
        self.assertNotIn('e', {r['unit'] for r in selected['references']})
        self.assertNotIn('h', {r['unit'] for r in selected['references']})
        prompt = person_candidate.build_prompt([{'class_name': '人员', 'bbox': [0, 0, 0.2, 0.2]}])
        leaked = person_candidate.prompt_contains_audit_secrets(prompt, selected['references'], labels)
        self.assertEqual(leaked, [])
        image, candidates = scene()
        examples = [{'image': image, 'bbox': candidates[0]['bbox'], 'caption': ref['caption']}
                    for ref in selected['references']]
        sheet, visible = person_candidate.render_reference_sheet(examples)
        self.assertEqual(visible, ['P1', 'P2', 'N1', 'N2'])
        header = sheet.getpixel((80, 8))
        self.assertGreater(header[1], header[0])
        with self.assertRaises(person_view.PersonViewLimit):
            person_candidate.render_reference_sheet([dict(examples[0], key='secret-key-a')])

    def test_too_few_dev_examples_are_not_filled_from_holdout(self):
        items, labels = reference_fixture()
        items = [it for it in items if it['unit'] not in {'c', 'd'}]
        selected = person_candidate.select_references(items, labels, {'e': 'task-shared', 'h': 'task-shared'})
        self.assertFalse(selected['ok'])
        self.assertEqual(selected['references'], [])


class ProtocolTests(unittest.TestCase):
    def test_g0_still_rejects_different_attachments(self):
        labels, items, baseline, candidate = mini_labels()
        base = measurement(baseline, labels, items)
        other = measurement(baseline, labels, items)
        other['input_evidence']['inputs_sha256'] = 'f' * 64
        gates = metrics.gates(measurement(candidate, labels, items), [base, other])
        self.assertFalse(gates['gates']['G0_同一冻结清单且候选与两次基线完整计分'])
        self.assertTrue(any('实际图片' in item for item in gates['comparison_errors']))
        self.assertEqual(metrics.GATE_VERSION, 3)

    def test_view_factor_requires_same_prompt_and_same_source(self):
        base = evidence_stub('aaa', 'prompt', crops='crop-a')
        changed = evidence_stub('aaa', 'prompt', crops='crop-b', person_changed=True)
        self.assertEqual(view_protocol.factor_errors(base, changed, 'person_view'), [])
        changed['prompt_sha256'] = 'other-prompt'
        self.assertTrue(view_protocol.factor_errors(base, changed, 'person_view'))
        same = evidence_stub('aaa', 'prompt', crops='crop-a', person_changed=False)
        same['attachments'][0]['sha256'] = 'different-source'
        self.assertTrue(any('source' in item for item in view_protocol.factor_errors(base, same, 'person_view')))

    def test_both_rounds_and_missing_results_cannot_manufacture_a_gain(self):
        labels = {f'p{i}': {'label': 'true' if i < 6 else 'false', 'class_name': '人员', 'split': 'dev'} for i in range(12)}
        labels.update({f'c{i}': {'label': 'true' if i < 6 else 'false', 'class_name': '车', 'split': 'dev'} for i in range(12)})
        baseline = {key: value['label'] for key, value in labels.items()}
        baseline['p0'] = 'uncertain'
        better = dict(baseline, p0='true')
        worse = dict(better, p1='false', p2='false')
        missing = dict(better)
        missing.pop('p2')
        exposure = {'independent_validation': False, 'reason': '这批评测框已被反复查看，不能当作新的盲测'}
        result = view_protocol.evaluate(
            [{'results': better}, {'results': worse}],
            [{'results': baseline}, {'results': baseline}],
            labels, factor='person_view', exposure=exposure)
        self.assertFalse(result['measurement_passed'])
        self.assertFalse(result['release_allowed'])
        self.assertFalse(result['g0_replaced'])
        self.assertTrue(any('真人误拒' in item for item in result['errors']))
        incomplete = view_protocol.evaluate(
            [{'results': missing}, {'results': missing}],
            [{'results': baseline}, {'results': baseline}],
            labels, factor='person_view', exposure=exposure)
        self.assertFalse(incomplete['measurement_passed'])
        scored = metrics.score(missing, labels, 'dev')
        self.assertGreater(scored['by_class']['人员']['failed'], 0)
        self.assertEqual(scored['by_class']['人员']['true_rejected'], 0)
        uncertain = metrics.score({'p0': 'uncertain', **{k: v['label'] for k, v in labels.items() if k != 'p0'}}, labels, 'dev')
        self.assertGreater(uncertain['by_class']['人员']['true_rejected'], 0)
        self.assertLess(uncertain['by_class']['人员']['right'], uncertain['by_class']['人员']['n'])

    def test_improvement_on_exposed_data_is_not_a_release(self):
        labels = person_labels()
        baseline = {key: ('uncertain' if key == 'p0' else value['label']) for key, value in labels.items()}
        better = {key: value['label'] for key, value in labels.items()}
        base_ev = evidence_stub('img', 'prompt', 'crop-a')
        cand_ev = evidence_stub('img', 'prompt', 'crop-b', person_changed=True)
        result = view_protocol.evaluate(
            [seal(better, labels, cand_ev, {'a': 1.2}), seal(better, labels, cand_ev, {'a': 1.8})],
            [seal(baseline, labels, base_ev, {'a': 2}), seal(baseline, labels, base_ev, {'a': 2})],
            labels, factor='person_view',
            exposure={'independent_validation': True, 'reason': '没有独立人员正负样本'})
        self.assertTrue(result['checks_passed'])
        self.assertTrue(result['no_regression'])
        self.assertTrue(result['improvement'])
        self.assertTrue(result['measurement_passed'])
        self.assertFalse(result['release_allowed'])
        self.assertFalse(result['independence']['accepted'])
        self.assertEqual(len(result['paired']), 12)
        self.assertEqual(result['paired'][0]['baseline'], ['uncertain', 'uncertain'])
        self.assertEqual(result['person'][0]['expected_n'], 12)
        self.assertIsNotNone(result['elapsed_s']['median'])

    def test_identical_results_and_boolean_independence_do_not_pass(self):
        labels = person_labels()
        same = {key: value['label'] for key, value in labels.items()}
        evidence = evidence_stub('img', 'prompt', 'crop-a')
        bare = view_protocol.evaluate(
            [{'results': same}, {'results': same}],
            [{'results': same}, {'results': same}],
            labels, factor='person_view', exposure={'independent_validation': True})
        self.assertFalse(bare['checks_passed'])
        self.assertFalse(bare['no_regression'])
        self.assertFalse(bare['measurement_passed'])
        self.assertFalse(bare['release_allowed'])
        self.assertTrue(any('实际输入' in item for item in bare['errors']))
        sealed = view_protocol.evaluate(
            [seal(same, labels, evidence), seal(same, labels, evidence)],
            [seal(same, labels, evidence), seal(same, labels, evidence)],
            labels, factor='person_view', exposure={'independent_validation': True})
        self.assertTrue(sealed['no_regression'])
        self.assertFalse(sealed['improvement'])
        self.assertFalse(sealed['measurement_passed'])
        self.assertFalse(sealed['release_allowed'])
        self.assertTrue(any('样本来源' in item for item in sealed['errors']))

    def test_illegal_factor_and_input_drift_block_release(self):
        labels = person_labels()
        baseline = {key: ('uncertain' if key == 'p0' else value['label']) for key, value in labels.items()}
        better = {key: value['label'] for key, value in labels.items()}
        base_ev = evidence_stub('img', 'prompt', 'crop-a')
        cand_ev = evidence_stub('img', 'prompt', 'crop-b', person_changed=True)
        runs = (
            [seal(better, labels, cand_ev), seal(better, labels, cand_ev)],
            [seal(baseline, labels, base_ev), seal(baseline, labels, base_ev)])
        illegal = view_protocol.evaluate(*runs, labels, factor='zoom-binary', exposure=independent_source(labels))
        self.assertFalse(illegal['measurement_passed'])
        self.assertFalse(illegal['release_allowed'])
        self.assertTrue(any('未知实验因素' in item for item in illegal['errors']))
        drifted = evidence_stub('other-img', 'prompt', 'crop-b', person_changed=True)
        changed = view_protocol.evaluate(
            [seal(better, labels, drifted), seal(better, labels, drifted)],
            runs[1], labels, factor='person_view', exposure=independent_source(labels))
        self.assertFalse(changed['release_allowed'])
        self.assertTrue(any('source_sha256' in item for item in changed['errors']))
        low = seal(better, labels, cand_ev)
        low['model_config'] = dict(low['model_config'], reasoning_effort='low')
        binary = seal(better, labels, cand_ev)
        binary['model_config'] = dict(binary['model_config'], decision_policy='binary')
        mismatched = view_protocol.evaluate(
            [low, binary], runs[1], labels, factor='person_view', exposure=independent_source(labels))
        self.assertFalse(mismatched['release_allowed'])
        self.assertTrue(any('推理量' in item or '三态' in item for item in mismatched['errors']))
        relabeled = seal(better, labels, cand_ev)
        relabeled['labels_sha256'] = 'b' * 64
        wrong_labels = view_protocol.evaluate(
            [relabeled, seal(better, labels, cand_ev)], runs[1],
            labels, factor='person_view', exposure=independent_source(labels))
        self.assertFalse(wrong_labels['measurement_passed'])
        self.assertTrue(any('标签摘要' in item for item in wrong_labels['errors']))

    def test_one_round_gain_is_not_a_repeatable_improvement(self):
        labels = person_labels()
        baseline = {key: ('uncertain' if key == 'p0' else value['label']) for key, value in labels.items()}
        better = {key: value['label'] for key, value in labels.items()}
        evidence = evidence_stub('img', 'prompt', 'crop-a')
        result = view_protocol.evaluate(
            [seal(better, labels, evidence), seal(baseline, labels, evidence)],
            [seal(baseline, labels, evidence), seal(baseline, labels, evidence)],
            labels, factor='person_view', exposure=independent_source(labels))
        self.assertTrue(result['no_regression'])
        self.assertFalse(result['improvement'])
        self.assertFalse(result['measurement_passed'])
        self.assertFalse(result['release_allowed'])

    def test_same_frame_source_is_required_to_release(self):
        labels = person_labels()
        baseline = {key: ('uncertain' if key == 'p0' else value['label']) for key, value in labels.items()}
        better = {key: value['label'] for key, value in labels.items()}
        base_ev = evidence_stub('img', 'prompt', 'crop-a')
        cand_ev = evidence_stub('img', 'prompt', 'crop-b', person_changed=True)
        released = view_protocol.evaluate(
            [seal(better, labels, cand_ev), seal(better, labels, cand_ev)],
            [seal(baseline, labels, base_ev), seal(baseline, labels, base_ev)],
            labels, factor='person_view', exposure=independent_source(labels))
        self.assertTrue(released['measurement_passed'])
        self.assertTrue(released['release_allowed'])
        self.assertEqual(released['person'][0]['true_rejected'], 0)
        self.assertEqual(released['baseline_person'][0]['true_rejected'], 1)
        adjacent = independent_source(labels)
        adjacent['sample_source']['kind'] = 'adjacent_frame'
        adjacent['sample_source']['frame_binding'] = 'adjacent_frame'
        blocked = view_protocol.evaluate(
            [seal(better, labels, cand_ev), seal(better, labels, cand_ev)],
            [seal(baseline, labels, base_ev), seal(baseline, labels, base_ev)],
            labels, factor='person_view', exposure=adjacent)
        self.assertTrue(blocked['measurement_passed'])
        self.assertFalse(blocked['release_allowed'])
        self.assertTrue(any('同帧' in item for item in blocked['errors']))

    def test_other_class_regression_is_visible(self):
        labels = {f'p{i}': {'label': 'true' if i < 6 else 'false', 'class_name': '人员', 'split': 'dev'} for i in range(12)}
        labels.update({f'c{i}': {'label': 'true' if i < 8 else 'false', 'class_name': '车', 'split': 'dev'} for i in range(12)})
        baseline = {key: value['label'] for key, value in labels.items()}
        worse_car = dict(baseline, c0='false')
        result = view_protocol.evaluate(
            [{'results': worse_car}, {'results': worse_car}],
            [{'results': baseline}, {'results': baseline}],
            labels, factor='person_view', exposure={'independent_validation': True})
        self.assertFalse(result['measurement_passed'])
        self.assertTrue(any('其他类别 车' in item for item in result['errors']))


class WorkerModeTests(unittest.TestCase):
    def setUp(self):
        self.harness = WorkerHarness()
        self.addCleanup(self.harness.close)
        self.candidates = [{'target_index': 0, 'class_name': '人员', 'bbox': [.08, .15, .38, .52]},
                           {'target_index': 1, 'class_name': '道路裂纹', 'bbox': [.2, .2, .4, .5]}]
        self.payload = {'image': base64.b64encode(SOURCE).decode(), 'candidates': self.candidates, 'prompt': '离线中文提示词'}

    def decisions(self, payload):
        return [{'target_index': c['target_index'], 'decision': 'uncertain', 'reasoning': '离线模拟，不是阈值',
                 'observed_object': c['class_name'], 'image_limitation': '目标很小'} for c in payload['candidates']]

    def test_legacy_mode_is_unchanged_and_fixed_mode_keeps_prompt_and_screenshot(self):
        legacy = self.harness.execute(self.payload, self.decisions(self.payload))
        fixed = self.harness.execute(dict(self.payload, view_mode='fixed_view_v1'), self.decisions(self.payload))
        self.assertEqual(legacy['input_evidence']['version'], 1)
        self.assertEqual(fixed['input_evidence']['protocol'], 'person_view_experiment_v1')
        self.assertEqual(fixed['input_evidence']['factor'], 'person_view')
        self.assertEqual(fixed['input_evidence']['source_kind'], 'screenshot')
        self.assertEqual(fixed['input_evidence']['attachments'][0]['sha256'], legacy['input_evidence']['attachments'][0]['sha256'])
        self.assertEqual(fixed['input_evidence']['attachments'][1]['sha256'], legacy['input_evidence']['attachments'][1]['sha256'])
        self.assertNotEqual(fixed['input_evidence']['attachments'][2]['sha256'], legacy['input_evidence']['attachments'][2]['sha256'])
        self.assertEqual(self.harness.last_prompt, self.payload['prompt'])
        local = next(r for r in fixed['input_evidence']['person_records'] if r['role'] == 'fixed_local')
        self.assertGreater(local['scale'], 1)
        blue = Image.new('RGB', (96, 64), 'blue')
        buf = io.BytesIO(); blue.save(buf, 'PNG')
        dirty = dict(self.payload, view_mode='fixed_view_v1', clean_image=base64.b64encode(buf.getvalue()).decode())
        response = self.harness.execute(dirty, self.decisions(dirty))
        crops = Image.open(io.BytesIO(self.harness.last_attachments[2][1]))
        self.assertGreater(crops.getpixel((128, 140))[0], 150)
        self.assertEqual(response['input_evidence']['crop_source'], 'screenshot')
        self.assertEqual(response['input_evidence']['clean_binding'], 'unverified')
        for item in (legacy, fixed, response):
            config = item['model_config']
            self.assertEqual(config['requested_model'], worker.MODEL)
            self.assertEqual(config['reasoning_effort'], 'none')
            self.assertEqual(config['decision_policy'], 'three_state')
            self.assertEqual(len(config['schema_sha256']), 64)
            self.assertNotIn('model_config', item['input_evidence'])
        self.assertEqual(legacy['model_config']['schema_sha256'], fixed['model_config']['schema_sha256'])

    def test_card_mode_keeps_the_legacy_view_and_appends_references(self):
        legacy = self.harness.execute(self.payload, self.decisions(self.payload))
        image = Image.open(io.BytesIO(SOURCE)).convert('RGB')
        sheet, _ = person_candidate.render_reference_sheet([
            {'image': image, 'bbox': self.candidates[0]['bbox'], 'caption': 'P1'},
            {'image': image, 'bbox': self.candidates[1]['bbox'], 'caption': 'N1'}])
        raw = io.BytesIO(); sheet.save(raw, 'JPEG', quality=95)
        payload = dict(self.payload, view_mode='card_references_v1', prompt=person_candidate.build_prompt(self.candidates),
                       reference_sheet=base64.b64encode(raw.getvalue()).decode())
        response = self.harness.execute(payload, self.decisions(payload))
        roles = [item['role'] for item in response['input_evidence']['attachments']]
        self.assertEqual(roles, ['source', 'locator', 'crops', 'references'])
        self.assertEqual(response['input_evidence']['attachments'][2]['sha256'], legacy['input_evidence']['attachments'][2]['sha256'])
        self.assertNotIn('secret', response['input_evidence']['prompt_sha256'])
        self.assertIn('正俯视', self.harness.last_prompt)

    def test_invalid_output_and_timeout_are_not_false(self):
        payload = dict(self.payload, view_mode='fixed_view_v1')
        bad = self.decisions(payload)
        bad[0]['decision'] = 'yes'
        with self.assertRaises(RuntimeError):
            self.harness.execute(payload, bad)
        env = {'CODEX_REVIEW_BINARY': '/offline/mock-codex', 'CODEX_REVIEW_HOME': str(self.harness.root),
               'CODEX_REVIEW_TMP_DIR': str(self.harness.root / 'requests')}

        def popen(command, **kwargs):
            proc = mock.Mock()
            proc.communicate.side_effect = subprocess.TimeoutExpired('mock', 215)
            return proc

        with mock.patch.dict(os.environ, env), mock.patch.object(worker.subprocess, 'Popen', side_effect=popen), \
                mock.patch.object(worker, 'terminate'):
            with self.assertRaises(RuntimeError) as caught:
                worker.execute(payload)
        self.assertIn('timed out', str(caught.exception))
        self.assertFalse(worker.PROCESSES)

    def test_too_many_people_does_not_call_the_model(self):
        payload = dict(self.payload, view_mode='fixed_view_v1', candidates=[
            {'target_index': i, 'class_name': '人员', 'bbox': [0.1, 0.1, 0.2, 0.2]} for i in range(17)])
        with self.assertRaises(person_view.PersonViewLimit):
            self.harness.execute(payload, [])
        self.assertEqual(self.harness.calls, 0)


def json_p000():
    import json
    return json.loads(Path(promptlib.__file__).with_name('prompts').joinpath('p000.json').read_text(encoding='utf-8'))


def reference_fixture():
    units = {
        'a': ('dev', 'true'), 'b': ('dev', 'true'), 'c': ('dev', 'false'), 'd': ('dev', 'false'),
        'e': ('dev', 'true'), 'h': ('holdout', 'false'),
    }
    items, labels = [], {}
    for unit, (split, label) in units.items():
        key = f'secret-key-{unit}'
        items.append({'unit': unit, 'split': split, 'candidates': [
            {'key': key, 'class_name': '人员', 'bbox': [0.1, 0.1, 0.2, 0.3]}]})
        labels[key] = {'label': label, 'class_name': '人员', 'split': split,
                       'history_decision': 'HISTORICAL-DECISION-UNIQUE'}
    return items, labels


def mini_labels():
    labels = {f'人员-{i}': {'label': 'true' if i < 6 else 'false', 'class_name': '人员', 'split': 'dev'} for i in range(12)}
    labels.update({f'道路裂纹-{i}': {'label': 'true' if i < 6 else 'false', 'class_name': '道路裂纹', 'split': 'dev'} for i in range(12)})
    items = [{'unit': key, 'split': 'dev', 'sha256': 'abc', 'candidates': [
        {'key': key, 'class_name': value['class_name'], 'bbox': [0, 0, 1, 1]}]} for key, value in labels.items()]
    items.append({'unit': 'u', 'split': 'unlabeled', 'candidates': [{'key': 'u', 'class_name': '车', 'bbox': [0, 0, 1, 1]}]})
    baseline = {key: value['label'] for key, value in labels.items()}
    baseline['u'] = 'uncertain'
    candidate = dict(baseline)
    candidate['人员-0'] = 'true'
    return labels, items, baseline, candidate


def measurement(results, labels, items):
    units = sum(it['split'] in ('dev', 'unlabeled') for it in items)
    return {'evaluation_sha256': metrics.evaluation_identity(items, labels),
            'input_evidence': {'version': 1, 'complete': True, 'inputs_sha256': 'e' * 64,
                               'expected_units': units, 'verified_units': units, 'invalid_units': {}},
            'dev': metrics.score(results, labels, 'dev'), 'unlabeled': metrics.unlabeled(results, items)}


def evidence_stub(source, prompt, crops, person_changed=False):
    return {
        'source_sha256': source, 'source_kind': 'screenshot', 'candidates_sha256': 'boxes',
        'prompt_sha256': prompt, 'person_changed': person_changed,
        'attachments': [
            {'role': 'source', 'sha256': source},
            {'role': 'locator', 'sha256': 'locator'},
            {'role': 'crops', 'sha256': crops},
        ],
    }


def person_labels():
    return {f'p{i}': {'label': 'true' if i < 6 else 'false', 'class_name': '人员', 'split': 'dev'} for i in range(12)}


def frozen_model():
    return {
        'requested_model': 'gpt-5.6-luna',
        'reasoning_effort': 'none',
        'decision_policy': 'three_state',
        'schema_sha256': 'a' * 64,
    }


def seal(results, labels, evidence, elapsed=None, split='dev'):
    return {
        'results': results,
        'elapsed_s': elapsed or {},
        'input_evidence': evidence,
        'model_config': frozen_model(),
        'labels_sha256': view_protocol.labels_digest(labels, split),
    }


def independent_source(labels, split='dev'):
    positives = [key for key, value in labels.items() if value['split'] == split and value['label'] == 'true']
    negatives = [key for key, value in labels.items() if value['split'] == split and value['label'] == 'false']
    return {
        'sample_source': {
            'source_id': 'flight-20261013-person-holdout',
            'kind': 'same_frame_human',
            'frame_binding': 'same_frame_verified',
            'exposed': False,
            'positive_keys': positives,
            'negative_keys': negatives,
            'labels_sha256': view_protocol.labels_digest(labels, split),
        },
    }


class EvalPlanTests(unittest.TestCase):
    def test_reference_groups_are_removed_from_the_call_budget(self):
        import run_person_view_eval as eval_plan
        items, labels = reference_fixture()
        for item in items:
            item['group'] = 'shared' if item['unit'] in {'e', 'h'} else item['unit']
        plan = eval_plan.plan_calls(items, labels)
        self.assertTrue(plan['references_ok'])
        called = {item['unit'] for item in plan['scored_units']}
        self.assertNotIn('a', called)
        self.assertNotIn('b', called)
        self.assertNotIn('c', called)
        self.assertNotIn('d', called)
        self.assertIn('e', called)
        self.assertEqual(plan['call_budget'], len(plan['scored_units']) * 6)

    def test_hash_mismatch_and_nonperson_change_stop_before_calls(self):
        import run_person_view_eval as eval_plan
        directory = Path(self.id().replace('.', '_'))
        # unittest 的临时目录由 TemporaryDirectory 管，避免把图片留在仓库里。
        from tempfile import TemporaryDirectory
        with TemporaryDirectory() as raw:
            root = Path(raw)
            image = Image.new('RGB', (40, 30), (10, 20, 30))
            good = root / 'good.jpg'
            image.save(good, format='JPEG')
            person = {'unit': 'good', 'group': 'good', 'split': 'dev', 'source': 'design',
                      'image': 'snapshot:good.jpg', 'sha256': eval_plan.sha256_file(good),
                      'candidates': [{'key': 'good:0', 'class_name': '人员', 'bbox': [0.1, 0.1, 0.4, 0.5]}]}
            labels = {'good:0': {'label': 'true', 'class_name': '人员', 'split': 'dev'}}
            errors, paths, unlabeled = eval_plan.check_required_images([person], labels, root, None)
            self.assertEqual(errors, [])
            self.assertEqual(unlabeled, [])
            self.assertEqual(paths['good'], good)
            person['sha256'] = '0' * 64
            errors, _paths, _unlabeled = eval_plan.check_required_images([person], labels, root, None)
            self.assertTrue(any('哈希' in item for item in errors))
            car = {'unit': 'good', 'group': 'good', 'split': 'dev', 'image': 'snapshot:good.jpg', 'sha256': None,
                   'candidates': [{'key': 'car', 'class_name': '车', 'bbox': [0.1, 0.1, 0.4, 0.5]}]}
            pixels = eval_plan.nonperson_pixel_errors([car], root, None)
            self.assertEqual(pixels['checked'], 1)
            self.assertEqual(pixels['errors'], [])

    def test_projected_baseline_keeps_source_and_allows_crop_change(self):
        import run_person_view_eval as eval_plan
        baseline = eval_plan.project_evidence({
            'source': {'sha256': 's' * 64}, 'candidates_sha256': 'c' * 64, 'prompt_sha256': 'p' * 64,
            'clean_binding': 'not_requested',
            'attachments': [
                {'role': 'source', 'sha256': 's' * 64},
                {'role': 'locator', 'sha256': 'l' * 64},
                {'role': 'crops', 'sha256': 'old'},
            ],
        })
        candidate = evidence_stub('s' * 64, 'p' * 64, 'new', person_changed=True)
        candidate['candidates_sha256'] = 'c' * 64
        candidate['attachments'][1]['sha256'] = 'l' * 64
        self.assertEqual(view_protocol.factor_errors(baseline, candidate, 'person_view'), [])

    def test_loaded_rounds_keep_only_agreed_model_and_label_evidence(self):
        import json
        from tempfile import TemporaryDirectory
        import run_person_view_eval as eval_plan
        labels = person_labels()
        items = [{'unit': f'u{i}', 'split': 'dev', 'group': f'g{i}', 'candidates': [
            {'key': f'p{i}', 'class_name': '人员', 'bbox': [0.1, 0.1, 0.2, 0.3]}]} for i in range(12)]
        plan = {'scored_units': items}
        digest = view_protocol.labels_digest(labels, 'dev')
        config = frozen_model()
        evidence = evidence_stub('s' * 64, 'p' * 64, 'crop-a', person_changed=True)
        with TemporaryDirectory() as raw:
            out = Path(raw)
            for name, _mode, _role in eval_plan.ROUNDS:
                for item in items:
                    eval_plan._write_json(out / 'calls' / name / f'{item["unit"]}.json', {
                        'status': 'ok', 'elapsed_s': 1.0,
                        'decisions': {item['candidates'][0]['key']: 'uncertain'},
                        'evidence': evidence, 'model_config': config, 'labels_sha256': digest,
                    })
            loaded = eval_plan._load_rounds(out, plan)['baseline'][0]
            self.assertEqual(loaded['model_config'], config)
            self.assertEqual(loaded['labels_sha256'], {'dev': digest})
            drifted = json.loads((out / 'calls' / 'baseline-1' / 'u0.json').read_text(encoding='utf-8'))
            drifted['model_config'] = dict(config, reasoning_effort='low')
            drifted['labels_sha256'] = 'b' * 64
            eval_plan._write_json(out / 'calls' / 'baseline-1' / 'u0.json', drifted)
            again = eval_plan._load_rounds(out, plan)['baseline'][0]
            self.assertNotIn('model_config', again)
            self.assertNotIn('labels_sha256', again)


if __name__ == '__main__':
    unittest.main()
