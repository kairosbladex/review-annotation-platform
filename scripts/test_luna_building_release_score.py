import importlib.util,pathlib,unittest
p=pathlib.Path(__file__).with_name('score_luna_building_release_validation.py');spec=importlib.util.spec_from_file_location('score',p);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
class Metrics(unittest.TestCase):
 def row(self,u,label,a,b):return dict(unit=u,label=label,decisions={f'{arm}-{n}':d for arm,d in [('A',a),('B',b)] for n in [1,2]})
 def test_uncertain_rejects_true_but_is_not_false_accept(self):
  rows=[self.row('x','true','uncertain','true'),self.row('x','false','uncertain','uncertain')]
  a=m.summary(rows,'A',1);self.assertEqual(a['true_rejected'],1);self.assertEqual(a['false_accepted'],0);self.assertEqual(a['agreement'],.5)
 def test_unknown_reference_not_scored(self):
  rows=[self.row('x','uncertain','true','true'),self.row('y','true','true','true')]
  s=m.summary(rows,'B',1);self.assertEqual(s['scored'],1);self.assertEqual(s['agreement'],1)
 def test_bootstrap_clusters_by_source_not_boxes_or_repeats(self):
  rows=[self.row('one','true','uncertain','true') for _ in range(4)]
  s=m.boot(rows);self.assertEqual(s['source_groups'],1);self.assertEqual(s['ci95'],[1,1]);self.assertEqual(s['mean_difference'],1)
if __name__=='__main__':unittest.main()
