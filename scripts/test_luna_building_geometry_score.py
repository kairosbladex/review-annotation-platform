import unittest
from score_luna_building_geometry import summary,paired_ci
class ScoreTests(unittest.TestCase):
 def test_uncertain_rejects_true_but_does_not_accept_false(self):
  rows=[dict(label=l,decisions={'B-1':d}) for l,d in [('true','uncertain'),('false','uncertain'),(None,'true')]]
  out=summary(rows,'B-1');self.assertEqual(out['scored'],2);self.assertEqual(out['true_rejected'],1);self.assertEqual(out['false_accepted'],0);self.assertEqual(out['accuracy'],.5)
 def test_repeats_and_boxes_share_source_cluster(self):
  rows=[dict(label='false',cluster='same-raw-source',decisions={'B-1':'true','B-2':'true','G-1':'false','G-2':'false'}) for _ in range(4)]
  out=paired_ci(rows,'false','false_accepted');self.assertEqual(out['groups'],1);self.assertEqual(out['ci95'],[-1,-1]);self.assertEqual(out['difference_G_minus_B'],-1)
 def test_failed_execution_requires_recovery_and_valid_results_are_never_replaced(self):
  import json,tempfile,pathlib
  from unittest.mock import patch
  import score_luna_building_geometry as scorer
  with tempfile.TemporaryDirectory() as temp:
   root=pathlib.Path(temp);folder=root/'regression';folder.mkdir()
   (root/'source-groups.json').write_text(json.dumps([dict(unit='u',cluster='raw')]))
   (folder/'units.json').write_text(json.dumps([dict(unit='u',sha256='h',split='dev',candidates=[dict(key='k',label='true')])]))
   def result(arm,n,status=200,decision='true'):
    return dict(unit='u',arm=arm,repeat=n,status=status,image_sha256='h',body={'results':[dict(target_index=0,decision=decision,reasoning='fixture')]})
   for arm in ['B','G']:
    for n in [1,2]:
     file=folder/f'results/{arm}-{n}/u.json';file.parent.mkdir(parents=True);file.write_text(json.dumps(result(arm,n,503 if (arm,n)==('G',1) else 200)))
   with patch.object(scorer,'ROOT',root):
    with self.assertRaises(AssertionError):scorer.rows_for(folder)
    recovery=root/'regression-retry/results/G-1/u.json';recovery.parent.mkdir(parents=True);recovery.write_text(json.dumps(result('G',1,decision='false')))
    self.assertEqual(scorer.rows_for(folder)[0]['decisions']['G-1'],'false')
    original=folder/'results/G-1/u.json';self.assertEqual(json.loads(original.read_text())['status'],503)
    original.write_text(json.dumps(result('G',1)))
    self.assertEqual(scorer.rows_for(folder)[0]['decisions']['G-1'],'true')
if __name__=='__main__':unittest.main()
