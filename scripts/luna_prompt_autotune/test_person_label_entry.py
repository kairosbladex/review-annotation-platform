"""人员待标入口只接收未曝光、同帧可核验的初审框。不调用模型。"""
import json
import threading
import unittest
import urllib.error
import urllib.request
from pathlib import Path

from PIL import Image

import person_label_entry as entry


def box():
    return [0.1, 0.2, 0.3, 0.4]


def frame():
    return {
        'frame_binding': 'same_frame_verified',
        'frame_evidence': {name: f'value-{name}' for name in entry.FRAME_FIELDS},
    }


def candidate(**updates):
    image = Image.new('RGB', (32, 24), (20, 40, 60))
    root = updates.pop('root')
    relative = updates.pop('image_file', 'images/sample.jpg')
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path, 'JPEG')
    row = {
        'candidate_id': 'person-1', 'task_id': 'flight-new', 'event_id': 'event-new',
        'class_name': '人员', 'bbox': box(), 'box_origin': 'initial_review',
        'image_sha256': 'a' * 64, 'image_file': relative, **frame(),
    }
    row.update(updates)
    return row


class QueueTests(unittest.TestCase):
    def test_ready_box_keeps_label_empty_and_separate_from_human_added(self):
        import tempfile
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        root = Path(temporary.name)
        prepared = entry.prepare_queue([
            candidate(root=root),
            candidate(root=root, candidate_id='added-1', box_origin='human_added', image_file='images/added.jpg'),
        ], image_root=root)
        self.assertEqual([row['candidate_id'] for row in prepared['initial_review']], ['person-1'])
        self.assertEqual(prepared['initial_review'][0]['label'], '')
        self.assertEqual([row['candidate_id'] for row in prepared['human_added']], ['added-1'])
        self.assertFalse(prepared['summary']['independent_validation_ready'])

    def test_exposed_task_blocks_every_event_and_prefilled_or_model_rows_stay_out(self):
        import tempfile
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        root = Path(temporary.name)
        prepared = entry.prepare_queue([
            candidate(root=root, candidate_id='old-1', event_id='seen', image_sha256='b' * 64),
            candidate(root=root, candidate_id='old-2', event_id='other', image_file='images/other.jpg',
                      image_sha256='c' * 64),
            candidate(root=root, candidate_id='labeled', task_id='flight-clean', label='是人员',
                      image_file='images/labeled.jpg'),
            candidate(root=root, candidate_id='answered', task_id='flight-clean', raw_decision='true',
                      image_file='images/answered.jpg', image_sha256='d' * 64),
            candidate(root=root, candidate_id='car', task_id='flight-clean', class_name='车',
                      image_file='images/car.jpg', image_sha256='e' * 64),
        ], {'image_sha256': ['b' * 64]}, image_root=root)
        reasons = {row['candidate_id']: row['reason'] for row in prepared['rejected'] if row.get('candidate_id')}
        self.assertEqual(reasons['old-1'], '任务与已曝光来源重叠')
        self.assertEqual(reasons['old-2'], '任务与已曝光来源重叠')
        self.assertEqual(reasons['labeled'], '已带标签')
        self.assertEqual(reasons['answered'], '带有自动结论')
        self.assertEqual(reasons['car'], '不是人员初审框')
        self.assertEqual(prepared['initial_review'], [])

    def test_missing_frame_or_near_time_stays_unverified(self):
        import tempfile
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        root = Path(temporary.name)
        near = candidate(root=root)
        near['frame_binding'] = 'missing'
        near['frame_evidence'] = {'nearby_time': '2026-09-23 17:28'}
        prepared = entry.prepare_queue([near], image_root=root)
        self.assertEqual(prepared['initial_review'], [])
        self.assertEqual(prepared['held'][0]['frame_binding'], 'missing')
        self.assertIn('没有同帧绑定', prepared['held'][0]['frame_gap'])
        self.assertIn('同帧依据不足', prepared['held'][0]['frame_gap'])

    def test_labels_cover_validation_only_after_both_human_choices(self):
        rows = [{'label': ''}, {'label': '是人员'}, {'label': '看不清'}]
        summary = entry.summarize(rows, [], [], [])
        self.assertFalse(summary['positive_negative_covered'])
        done = [{'label': '是人员'}, {'label': '不是人员'}, {'label': '看不清'}]
        # 看不清仍保留，但正负例需要各至少一条明确人工结论。
        summary = entry.summarize(done, [], [], [])
        self.assertTrue(summary['positive_negative_covered'])
        self.assertEqual(summary['labels']['看不清'], 1)
        with self.assertRaises(ValueError):
            entry.apply_label([{'candidate_id': 'a', 'label': ''}], 'a', '真目标')


class EntryServerTests(unittest.TestCase):
    def test_empty_page_has_choices_and_does_not_show_automatic_answers(self):
        import tempfile
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        package = Path(temporary.name)
        entry.write_package(package, entry.prepare_queue([]))
        self._check_page(package, expect_rows=0)

    def test_server_saves_only_the_three_choices(self):
        import tempfile
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        root = Path(temporary.name)
        package = root / 'package'
        image_root = root / 'images-src'
        prepared = entry.prepare_queue([
            candidate(root=image_root, candidate_id='person-1'),
            candidate(root=image_root, candidate_id='person-2', image_file='images/second.jpg',
                      image_sha256='f' * 64),
        ], image_root=image_root)
        entry.write_package(package, prepared)
        (package / 'images').mkdir()
        for name in ('sample.jpg', 'second.jpg'):
            (package / 'images' / name).write_bytes((image_root / 'images' / name).read_bytes())
        (package / 'secret-answer.json').write_text('{"raw_decision":"true"}', encoding='utf-8')
        self._check_page(package, expect_rows=2)
        body = json.dumps({'candidate_id': 'person-1', 'label': '是人员'}).encode()
        request = urllib.request.Request(self.url + '/api/label', data=body)
        self.assertEqual(json.loads(urllib.request.urlopen(request).read())['label'], '是人员')
        saved = entry.read_csv(package / 'annotations.csv')
        self.assertEqual(saved[0]['label'], '是人员')
        self.assertEqual(saved[1]['label'], '')
        bad = json.dumps({'candidate_id': 'person-1', 'label': 'true'}).encode()
        with self.assertRaises(urllib.error.HTTPError):
            urllib.request.urlopen(urllib.request.Request(self.url + '/api/label', data=bad))

    def _check_page(self, package, expect_rows):
        handler = type('BoundHandler', (entry.Handler,), {'package': package.resolve()})
        server = entry.ThreadingHTTPServer(('127.0.0.1', 0), handler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        self.addCleanup(server.server_close)
        self.addCleanup(server.shutdown)
        self.url = f'http://127.0.0.1:{server.server_address[1]}'
        page = urllib.request.urlopen(self.url + '/').read().decode('utf-8')
        for text in ('是人员', '不是人员', '看不清', '人工补框', '历史回归'):
            self.assertIn(text, page)
        for text in ('raw_decision', 'luna', 'confidence', 'baseline_reason'):
            self.assertNotIn(text, page)
        rows = json.loads(urllib.request.urlopen(self.url + '/api/rows').read())['rows']
        self.assertEqual(len(rows), expect_rows)
        self.assertTrue(all(row['label'] == '' for row in rows))
        if rows:
            self.assertEqual(set(rows[0]), set(entry.API_FIELDS))
        for blocked in ('/annotations.csv', '/gaps.json', '/secret-answer.json', '/images/../gaps.json'):
            with self.assertRaises(urllib.error.HTTPError) as caught:
                urllib.request.urlopen(self.url + blocked)
            self.assertEqual(caught.exception.code, 404)


if __name__ == '__main__':
    unittest.main()
