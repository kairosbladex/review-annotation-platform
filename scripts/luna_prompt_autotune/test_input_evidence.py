"""真实图片转换 + mock CLI/HTTP 的离线追溯测试；不启动 worker 或调用模型。"""
import base64
import copy
import io
import json
import os
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from PIL import Image, ImageDraw

import input_evidence as evidence
import luna_experiment_worker as worker
import promptlib
import runner


def image_bytes():
    image = Image.new('RGB', (96, 64), 'white')
    draw = ImageDraw.Draw(image)
    draw.rectangle((6, 9, 40, 38), fill='red')
    draw.line((0, 63, 95, 0), fill='blue', width=3)
    buf = io.BytesIO(); image.save(buf, 'PNG')
    return buf.getvalue()


SOURCE = image_bytes()


class WorkerHarness:
    def __init__(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        (self.root / 'auth.json').write_text(json.dumps({'OPENAI_API_KEY': 'offline-test-secret'}))
        (self.root / 'requests').mkdir()
        self.before_result = None
        self.extra_events = []
        self.last_command = []
        self.last_attachments = []
        self.calls = 0

    def close(self):
        self.temporary.cleanup()

    def execute(self, payload, decisions=None):
        candidates = payload['candidates']
        decisions = decisions if decisions is not None else [
            {'target_index': c['target_index'], 'decision': 'true', 'reasoning': '离线模拟结果',
             'observed_object': c['class_name'], 'image_limitation': ''} for c in candidates]

        def popen(command, **kwargs):
            self.calls += 1
            self.last_command = command
            self.last_attachments = [(Path(command[i + 1]), Path(command[i + 1]).read_bytes())
                                     for i, value in enumerate(command) if value == '-i']
            self.encoding = kwargs['encoding']
            proc = mock.Mock(returncode=0)
            def communicate(prompt, timeout):
                self.last_prompt = prompt
                if self.before_result: self.before_result(self.last_attachments)
                events = self.extra_events + [{'type': 'item.completed', 'item': {'type': 'agent_message',
                          'text': json.dumps({'results': decisions}, ensure_ascii=False)}}]
                return '\n'.join(json.dumps(event, ensure_ascii=False) for event in events), ''
            proc.communicate.side_effect = communicate
            return proc

        env = {'CODEX_REVIEW_BINARY': '/offline/mock-codex', 'CODEX_REVIEW_HOME': str(self.root),
               'CODEX_REVIEW_TMP_DIR': str(self.root / 'requests')}
        with mock.patch.dict(os.environ, env), mock.patch.object(worker.subprocess, 'Popen', side_effect=popen):
            return worker.execute(payload)

    def record(self, unit, results, doc):
        candidates = evidence.request_candidates(unit)
        prompt = promptlib.build(doc, candidates)
        # 先构造合法 worker 输出，再由调用方按测试场景模拟磁盘上的缺失/非法结果。
        decisions = [{'target_index': i, 'decision': results.get(c['key']) if results.get(c['key']) in ('true', 'false', 'uncertain') else 'uncertain',
                      'reasoning': '离线模拟结果', 'observed_object': c['class_name'], 'image_limitation': ''}
                     for i, c in enumerate(unit['candidates'])]
        response = self.execute({'image': base64.b64encode(SOURCE).decode(), 'candidates': candidates, 'prompt': prompt}, decisions)
        rows = [{'key': c['key'], 'target_index': i, 'raw': results[c['key']], 'reasoning': decisions[i]['reasoning'],
                 'observed': decisions[i]['observed_object'], 'limitation': decisions[i]['image_limitation']}
                for i, c in enumerate(unit['candidates']) if c['key'] in results]
        return {'success': True, 'unit': unit['unit'], 'results': rows, 'elapsed': 1,
                'request_identity': evidence.request_identity(evidence.digest(SOURCE), candidates, prompt),
                'execution_id': response['execution_id'], 'requested_model': response['requested_model'],
                'input_evidence': response['input_evidence']}


class WorkerEvidenceTests(unittest.TestCase):
    def setUp(self):
        self.harness = WorkerHarness(); self.addCleanup(self.harness.close)
        self.candidates = [{'target_index': 0, 'class_name': '人员', 'bbox': [.08, .15, .38, .52]},
                           {'target_index': 1, 'class_name': '道路裂纹', 'bbox': [.78, .82, 1, 1]}]
        self.payload = {'image': base64.b64encode(SOURCE).decode(), 'candidates': self.candidates, 'prompt': '离线中文提示词'}
        self.request = evidence.request_identity(evidence.digest(SOURCE), self.candidates, self.payload['prompt'])

    def validate(self, response, **kwargs):
        evidence.validate(response['input_evidence'], request=self.request, candidates=self.candidates,
                          execution_id=response['execution_id'], model=response['requested_model'],
                          decisions=response['results'], **kwargs)

    def test_hashes_match_the_exact_cli_attachments_and_raw_source(self):
        response = self.harness.execute(self.payload)
        audit = response['input_evidence']
        self.validate(response)
        self.assertEqual(audit['source']['sha256'], evidence.digest(SOURCE))
        self.assertNotEqual(audit['source']['sha256'], audit['attachments'][0]['sha256'])
        self.assertEqual([x['sha256'] for x in audit['attachments']], [evidence.digest(data) for _, data in self.harness.last_attachments])
        self.assertEqual(self.harness.last_prompt, self.payload['prompt'])
        self.assertEqual(self.harness.encoding, 'utf-8')
        self.assertFalse(any(path.exists() for path, _ in self.harness.last_attachments))
        self.assertFalse(worker.PROCESSES)
        self.assertNotIn('offline-test-secret', json.dumps(audit))
        self.assertNotIn(str(self.harness.root), json.dumps(audit))

    def test_rendering_is_byte_identical_to_the_previous_worker(self):
        self.harness.execute(self.payload)
        source = Image.open(io.BytesIO(SOURCE)).convert('RGB')
        locator = source.copy(); draw = ImageDraw.Draw(locator)
        for c in self.candidates:
            b = [c['bbox'][0] * source.width, c['bbox'][1] * source.height, c['bbox'][2] * source.width, c['bbox'][3] * source.height]
            draw.rectangle(b, outline='yellow', width=2); draw.text((b[0], b[1]), str(c['target_index']), fill='yellow')
        sheet = Image.new('RGB', (5 * 256, ((len(self.candidates) + 4) // 5) * 280), 'white')
        sd = ImageDraw.Draw(sheet)
        for i, c in enumerate(self.candidates):
            b = c['bbox']; x1, y1, x2, y2 = b[0] * source.width, b[1] * source.height, b[2] * source.width, b[3] * source.height
            mx, my = max(20, (x2 - x1) * .25), max(20, (y2 - y1) * .25)
            crop = source.crop((max(0, x1 - mx), max(0, y1 - my), min(source.width, x2 + mx), min(source.height, y2 + my)))
            crop.thumbnail((256, 256)); sheet.paste(crop, ((i % 5) * 256, (i // 5) * 280 + 24))
            sd.text(((i % 5) * 256, (i // 5) * 280), str(i), fill='black')
        for image, (_, actual) in zip((source, locator, sheet), self.harness.last_attachments):
            out = io.BytesIO(); image.save(out, 'JPEG', quality=95)
            self.assertEqual(out.getvalue(), actual)

    def test_crop_records_bind_source_and_clipped_pixel_boxes(self):
        audit = self.harness.execute(self.payload)['input_evidence']
        for c, crop in zip(self.candidates, audit['crops']):
            self.assertEqual(crop['source_sha256'], evidence.digest(SOURCE))
            self.assertEqual(crop['pixel_box'], [round(v) for v in evidence.context_box(c['bbox'], 96, 64)])
            self.assertEqual(crop['target_index'], c['target_index'])
        self.assertEqual(audit['crops'][1]['pixel_box'][2:], [96, 64])

    def test_clean_image_is_explicitly_unverified_and_not_eligible(self):
        response = self.harness.execute(dict(self.payload, clean_image=self.payload['image'], same_frame_verified=True))
        audit = response['input_evidence']
        self.assertEqual(audit['clean_binding'], 'unverified')
        self.assertEqual(audit['attachments'][-1]['role'], 'clean_auxiliary')
        self.validate(response, allow_unverified_clean=True)
        with self.assertRaisesRegex(evidence.EvidenceError, '严格同帧'): self.validate(response)
        changed = copy.deepcopy(response)
        changed['input_evidence']['clean_binding'] = 'verified'
        self.reseal(changed)
        with self.assertRaisesRegex(evidence.EvidenceError, '同帧通过'): self.validate(changed, allow_unverified_clean=True)

    @staticmethod
    def reseal(response):
        audit = response['input_evidence']
        audit['evidence_sha256'] = evidence.json_digest({k: v for k, v in audit.items() if k != 'evidence_sha256'})

    def test_request_result_and_execution_mismatch_are_rejected(self):
        response = self.harness.execute(self.payload)
        for change in ('source', 'prompt', 'candidate', 'result', 'execution', 'model', 'crop', 'attachment'):
            modified = copy.deepcopy(response)
            audit = modified['input_evidence']
            if change == 'source': audit['source']['sha256'] = '0' * 64
            elif change == 'prompt': audit['prompt_sha256'] = '0' * 64
            elif change == 'candidate': audit['candidates_sha256'] = '0' * 64
            elif change == 'result': modified['results'][0]['decision'] = 'false'
            elif change == 'execution': modified['execution_id'] = '0' * 32
            elif change == 'model': modified['requested_model'] = 'other-model'
            elif change == 'crop': audit['crops'][0]['pixel_box'][0] += 1
            else: audit['attachments'].reverse()
            self.reseal(modified)
            with self.subTest(change=change), self.assertRaises(evidence.EvidenceError): self.validate(modified)

    def test_unknown_metadata_and_bad_versions_are_rejected(self):
        response = self.harness.execute(self.payload)
        for change in ('field', 'source_field', 'version'):
            modified = copy.deepcopy(response)
            if change == 'field': modified['input_evidence']['private_path'] = '/not-for-storage'
            elif change == 'source_field': modified['input_evidence']['source']['api_key'] = 'not-for-storage'
            else: modified['input_evidence']['version'] = True
            self.reseal(modified)
            with self.subTest(change=change), self.assertRaises(evidence.EvidenceError): self.validate(modified)

    def test_prompt_and_result_changes_do_not_change_comparable_input_identity(self):
        first = self.harness.execute(self.payload)
        second = self.harness.execute(dict(self.payload, prompt='另一版提示词'))
        self.assertNotEqual(first['input_evidence']['prompt_sha256'], second['input_evidence']['prompt_sha256'])
        self.assertNotEqual(first['execution_id'], second['execution_id'])
        self.assertEqual(evidence.comparable_identity(first['input_evidence']), evidence.comparable_identity(second['input_evidence']))

    def test_changed_loaded_implementation_blocks_health_and_cli(self):
        for module in (worker, evidence):
            with self.subTest(module=module.__name__), mock.patch.object(module, 'LOADED_SHA256', '0' * 64):
                handler = object.__new__(worker.Handler)
                handler.path = '/health'; handler.reply = mock.Mock()
                handler.do_GET()
                self.assertEqual(handler.reply.call_args.args[0], 503)
                with self.assertRaisesRegex(evidence.EvidenceError, '已加载版本'):
                    self.harness.execute(self.payload)
                self.assertEqual(self.harness.calls, 0)
                self.assertFalse(list((self.harness.root / 'requests').iterdir()))

    def test_changed_input_during_cli_execution_fails_and_cleans_up(self):
        self.harness.before_result = lambda files: files[0][0].write_bytes(b'changed')
        with self.assertRaisesRegex(RuntimeError, 'attachment changed'): self.harness.execute(self.payload)
        self.assertFalse(list((self.harness.root / 'requests').iterdir()))
        self.assertFalse(worker.PROCESSES)

    def test_cli_tool_use_timeout_and_bad_mapping_do_not_return_evidence(self):
        self.harness.extra_events = [{'item': {'type': 'command_execution'}}]
        with self.assertRaisesRegex(RuntimeError, 'unexpected tool'): self.harness.execute(self.payload)
        self.harness.extra_events = []
        with self.assertRaisesRegex(RuntimeError, 'mapping'): self.harness.execute(self.payload, [])
        def timeout(_): raise subprocess.TimeoutExpired('mock', 215)
        self.harness.before_result = timeout
        with mock.patch.object(worker, 'terminate') as terminate:
            with self.assertRaisesRegex(RuntimeError, 'timed out'): self.harness.execute(self.payload)
            terminate.assert_called_once()
        self.assertFalse(list((self.harness.root / 'requests').iterdir()))
        self.assertFalse(worker.PROCESSES)


class RunnerTransportTests(unittest.TestCase):
    def setUp(self):
        self.harness = WorkerHarness(); self.addCleanup(self.harness.close)
        root = self.harness.root
        self.unit = {'unit': 'one', 'split': 'dev', 'sha256': evidence.digest(SOURCE),
                     'candidates': [{'key': 'person-0', 'class_name': '人员', 'bbox': [.1, .1, .7, .7]}]}
        self.doc = {'common': '离线测试', 'cards': {}}
        for name in ('ASSETS', 'RUNS'):
            patch = mock.patch.object(runner, name, root / name.lower()); patch.start(); self.addCleanup(patch.stop)
        (runner.ASSETS / 'images').mkdir(parents=True)
        (runner.ASSETS / 'images/one.jpg').write_bytes(SOURCE)
        for name, value in (('token', 'offline-token'), ('health', (True, {}))):
            patch = mock.patch.object(runner, name, return_value=value); patch.start(); self.addCleanup(patch.stop)
        patch = mock.patch.object(runner, 'CONCURRENCY', 1); patch.start(); self.addCleanup(patch.stop)
        patch = mock.patch.object(runner.time, 'sleep'); self.sleep = patch.start(); self.addCleanup(patch.stop)
        self.modify_response = None
        def open_request(request, timeout):
            body = self.harness.execute(json.loads(request.data))
            if self.modify_response: self.modify_response(body)
            return io.BytesIO(json.dumps(body).encode())
        opener = mock.Mock(); opener.open.side_effect = open_request
        patch = mock.patch.object(runner.urllib.request, 'build_opener', return_value=opener)
        patch.start(); self.addCleanup(patch.stop)
        self.open = opener.open

    def run_unit(self):
        runner.run_units('p001', self.doc, [self.unit])
        return runner.rj(runner.RUNS / 'p001/out/one.json')

    def test_worker_response_is_bound_to_saved_results_and_eligible_summary(self):
        record = self.run_unit()
        self.assertTrue(record['success'])
        self.assertTrue(record['execution_id'])
        self.assertEqual(record['requested_model'], worker.MODEL)
        summary = runner.summarize_inputs([record], [self.unit], self.doc)
        self.assertTrue(summary['complete'])
        self.assertEqual(summary['verified_units'], 1)
        self.assertNotIn('offline-token', json.dumps(record))
        record['results'][0]['raw'] = 'false'
        self.assertFalse(runner.summarize_inputs([record], [self.unit], self.doc)['complete'])

    def test_legacy_worker_is_recorded_as_failure_without_retry(self):
        self.modify_response = lambda body: body.pop('input_evidence')
        record = self.run_unit()
        self.assertFalse(record['success'])
        self.assertEqual(record['attempts'][0]['status'], 'input_evidence_invalid')
        self.assertEqual(self.open.call_count, 1)
        self.sleep.assert_not_called()

    def test_protocol_probe_rejects_old_worker_before_model_calls(self):
        with mock.patch.object(runner, 'systemctl', return_value=mock.Mock(stdout='active')):
            for version in (None, 0, True, 1):
                self.open.side_effect = lambda url, timeout: io.BytesIO(json.dumps({'input_evidence_version': version}).encode())
                if type(version) is int and version == 1:
                    runner.ensure_worker()
                else:
                    with self.assertRaisesRegex(evidence.EvidenceError, '配套升级'): runner.ensure_worker()
        self.assertEqual(self.harness.calls, 0)
        self.assertTrue(all(call.args[0].endswith('/health') for call in self.open.call_args_list))

    def test_invalid_evidence_stops_remaining_requests_and_does_not_retry_on_resume(self):
        other = copy.deepcopy(self.unit); other['unit'] = 'two'; other['candidates'][0]['key'] = 'person-1'
        (runner.ASSETS / 'images/two.jpg').write_bytes(SOURCE)
        self.modify_response = lambda body: body.pop('input_evidence')
        runner.run_units('p001', self.doc, [self.unit, other])
        self.assertEqual(self.open.call_count, 1)
        self.assertFalse((runner.RUNS / 'p001/out/two.json').exists())
        self.open.reset_mock()
        runner.run_units('p001', self.doc, [self.unit, other])
        self.open.assert_not_called()

    def test_resource_guard_still_stops_before_model_calls(self):
        with mock.patch.object(runner, 'health', return_value=(False, {})):
            with self.assertRaisesRegex(RuntimeError, '资源或共享服务'): self.run_unit()
        self.open.assert_not_called()

    def test_wrong_prompt_evidence_does_not_become_a_successful_result(self):
        def change(body):
            body['input_evidence']['prompt_sha256'] = '0' * 64
            WorkerEvidenceTests.reseal(body)
        self.modify_response = change
        record = self.run_unit()
        self.assertFalse(record['success'])
        self.assertIn('提示词', record['attempts'][0]['evidence_error'])

    def test_declared_source_hash_mismatch_stops_before_http(self):
        self.unit['sha256'] = '0' * 64
        with self.assertRaisesRegex(evidence.EvidenceError, '冻结清单'): self.run_unit()
        self.open.assert_not_called()

    def test_source_or_prompt_change_cannot_reuse_existing_output(self):
        self.unit['sha256'] = None
        self.run_unit()
        self.open.reset_mock()
        self.doc['common'] = '修改后的提示词'
        with self.assertRaisesRegex(evidence.EvidenceError, '已有输出'): self.run_unit()
        self.doc['common'] = '离线测试'
        (runner.ASSETS / 'images/one.jpg').write_bytes(SOURCE + b'extra')
        with self.assertRaisesRegex(evidence.EvidenceError, '已有输出'): self.run_unit()
        self.open.assert_not_called()

    def test_resume_preflight_checks_all_existing_records_before_any_request(self):
        self.run_unit()
        other = copy.deepcopy(self.unit); other['unit'] = 'two'; other['candidates'][0]['key'] = 'person-1'
        (runner.ASSETS / 'images/two.jpg').write_bytes(SOURCE)
        self.doc['common'] = '修改后的提示词'
        self.open.reset_mock()
        with self.assertRaisesRegex(evidence.EvidenceError, '已有输出'):
            runner.run_units('p001', self.doc, [other, self.unit])
        self.open.assert_not_called()

    def test_unchanged_request_reuses_existing_output_without_model_call(self):
        previous = self.run_unit()
        self.open.reset_mock()
        self.assertEqual(self.run_unit(), previous)
        self.open.assert_not_called()


if __name__ == '__main__':
    unittest.main()
