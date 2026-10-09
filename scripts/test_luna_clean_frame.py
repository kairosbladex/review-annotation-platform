"""投影和业务计分的必要边界测试，不调用模型或业务服务。"""
import unittest
import json
import tempfile
from pathlib import Path
from luna_clean_frame_eval import project
from score_luna_clean_frame import measure, compare, score, CONDITIONS


class ProjectionTests(unittest.TestCase):
    def test_resolution_change_preserves_normalized_box(self):
        b, raw = project([.1, .2, .4, .6], [[2,0,0],[0,1.5,0],[0,0,1]], 960,720,1920,1080)
        for got, want in zip(b, [.1,.2,.4,.6]): self.assertAlmostEqual(got, want)

    def test_partial_frame_is_clipped_but_audited(self):
        b, raw = project([0,0,.2,.2], [[1,0,-10],[0,1,0],[0,0,1]],100,100,100,100)
        self.assertEqual(b, [0,0,.1,.2]); self.assertEqual(raw[0], -.1)

    def test_outside_frame_rejected(self):
        with self.assertRaisesRegex(ValueError, 'outside'):
            project([.1,.1,.2,.2], [[1,0,200],[0,1,0],[0,0,1]],100,100,100,100)


class ScoreTests(unittest.TestCase):
    def test_uncertain_rejects_true_but_does_not_accept_false(self):
        rows = [dict(label=label, decisions={c:'uncertain' for c in CONDITIONS}, event_id=str(i)) for i,label in enumerate(['true','false'])]
        m = measure(rows, 'screenshot-1')
        self.assertEqual((m['correct'],m['true_rejected'],m['false_accepted'],m['uncertain'],m['true_uncertain']), (1,1,0,2,1))
        self.assertEqual(compare(rows)['paired_changes'][0]['accuracy_delta_pp'], 0)

    def test_no_sample_is_not_zero_accuracy(self):
        self.assertIsNone(measure([], CONDITIONS[0])['accuracy'])

    def test_uncertainty_interval_uses_paired_source_clusters(self):
        rows=[dict(label='true',event_id=str(i),source_local_path='same-video',
                   decisions={c:('uncertain' if c.startswith('screenshot') else 'true') for c in CONDITIONS}) for i in range(3)]
        result=compare(rows)
        self.assertEqual(result['unique_source_files'],1)
        self.assertEqual(result['mean_delta_pp_cluster_bootstrap95'],[100,100])
        self.assertEqual(result['mean_uncertain_delta_pp_cluster_bootstrap95'],[-100,-100])

    def test_incomplete_four_way_pair_is_excluded(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp)
            def put(name, value):
                dest = p/name; dest.parent.mkdir(parents=True, exist_ok=True)
                dest.write_text(json.dumps(value))
            u = {'unit':'one', 'event_id':'event', 'split':'holdout', 'source':'test', 'source_kind':'video', 'source_local_path':'video.mp4',
                 'candidates':[{'key':'box','class_name':'车'}]}
            put('included.json', [u]); put('all-items.json', [u])
            put('labels.json', {'box': {'label':'true', 'class_name':'车', 'split':'holdout'}})
            for c in CONDITIONS[:3]:
                put(f'results/{c}/one.json', {'success':True, 'prompt_sha256':'same', 'attempts':[{'seconds':1}], 'results':[{'key':'box','decision':'true'}]})
            score(p)
            self.assertEqual(json.loads((p/'report.json').read_text())['incomplete_images'], ['one'])
            self.assertEqual(json.loads((p/'report.json').read_text())['scored_candidates'], 0)

    def test_unlabeled_uncertain_does_not_enter_accuracy(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)
            def put(name, value):
                dest=p/name;dest.parent.mkdir(parents=True,exist_ok=True);dest.write_text(json.dumps(value))
            u={'unit':'one','event_id':'event','split':'holdout','source':'test','source_kind':'video','source_local_path':'clip.mp4',
               'candidates':[{'key':'gold','class_name':'车'},{'key':'unknown','class_name':'车'}]}
            put('included.json',[u]);put('all-items.json',[u]);put('labels.json',{'gold':{'label':'true','class_name':'车','split':'holdout'}})
            for c in CONDITIONS:
                put(f'results/{c}/one.json',{'success':True,'prompt_sha256':'same','execution_id':c,'attempts':[{'seconds':1}],
                                           'results':[{'key':'gold','decision':'true'},{'key':'unknown','decision':'uncertain'}]})
            score(p);r=json.loads((p/'report.json').read_text())
            self.assertEqual(r['scored_candidates'],1)
            self.assertEqual(r['splits']['holdout']['overall'][CONDITIONS[0]]['uncertain'],0)
            self.assertEqual(r['all_candidate_uncertain']['holdout']['车'][CONDITIONS[0]],1)


if __name__ == '__main__': unittest.main()
