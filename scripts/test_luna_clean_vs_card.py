import base64, json, tempfile, unittest
from pathlib import Path
from unittest.mock import patch
import luna_clean_vs_card as experiment
from luna_clean_frame_eval import sha

class Contract(unittest.TestCase):
    def test_pairing_and_pure_building_gate(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d); (root/'clean').mkdir(); (root/'assets/images').mkdir(parents=True); (root/'baseline').mkdir()
            for file in ('p000.json','p002.json'):
                source=Path(__file__).resolve().parents[1]/'data/luna-clean-vs-card-20261010'/file
                (root/'baseline'/file).write_bytes(source.read_bytes())
            (root/'clean/x.jpg').write_bytes(b'clean');(root/'assets/images/x.jpg').write_bytes(b'screenshot')
            u={'unit':'x','sha256':sha(root/'assets/images/x.jpg'),'clean_sha256':sha(root/'clean/x.jpg'),'candidates':[{'class_name':'违建','bbox':[.1,.2,.3,.4],'clean_bbox':[.2,.3,.4,.5]}]}
            with patch.object(experiment,'AUTO',root):
                a,b,c,d=[experiment.payload(root,u,arm) for arm in 'ABCD']
                self.assertEqual(a['image'],b['image']);self.assertEqual(c['image'],d['image'])
                self.assertNotEqual(a['image'],c['image']);self.assertEqual(c['candidates'][0]['bbox'],u['candidates'][0]['clean_bbox'])
                self.assertNotIn('【类别判断要点】',a['prompt']);self.assertIn('【类别判断要点】',b['prompt'])
                self.assertEqual(c['image'],base64.b64encode(b'clean').decode());self.assertNotIn('clean_image',c)
                u['candidates'].append(dict(u['candidates'][0],class_name='人员'))
                self.assertEqual(experiment.payload(root,u,'A'),experiment.payload(root,u,'B'))
                self.assertEqual(experiment.payload(root,u,'C'),experiment.payload(root,u,'D'))
                (root/'clean/x.jpg').write_bytes(b'changed')
                with self.assertRaises(AssertionError):experiment.payload(root,u,'C')

class Scoring(unittest.TestCase):
    def test_cluster_pairing_and_uncertain_policy(self):
        from score_luna_clean_vs_card import paired
        from score_luna_clean_frame import measure
        rows=[{'label':'true','source_local_path':'same-video','decisions':{'A-1':'uncertain','A-2':'false','B-1':'true','B-2':'true'}},
              {'label':'false','source_local_path':'same-video','decisions':{'A-1':'false','A-2':'false','B-1':'true','B-2':'true'}}]
        result=paired(rows,'A','B')
        self.assertEqual(result['accuracy']['delta_pp'],0)
        self.assertEqual(result['accuracy']['source_clusters'],1)
        self.assertEqual(result['false_acceptance']['delta_pp'],100)
        self.assertEqual(result['true_rejection']['delta_pp'],-100)
        self.assertEqual(result['uncertain']['delta_pp'],-25)
        self.assertEqual(measure(rows,'A-1')['true_rejected'],1)
        self.assertIsNone(paired([],'A','B')['accuracy']['ci95_pp'])

class LabelCorrections(unittest.TestCase):
    def test_only_reviewed_changes_and_original_label_guard(self):
        from score_luna_clean_vs_card import corrected_labels
        labels={'x':{'label':'false','class_name':'车','split':'dev'},'y':{'label':'true','class_name':'人员','split':'holdout'}}
        c=[{'key':'x','class_name':'车','original_label':'false','corrected_label':'true'}]
        new=corrected_labels(labels,c)
        self.assertEqual(new['x']['label'],'true');self.assertEqual(new['y'],labels['y']);self.assertEqual(labels['x']['label'],'false')
        with self.assertRaises(AssertionError):corrected_labels(new,c)
        with self.assertRaises(AssertionError):corrected_labels(labels,c+c)

if __name__=='__main__':unittest.main()
