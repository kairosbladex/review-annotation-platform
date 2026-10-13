"""Offline accounting/mapping regression checks; no model or network calls."""
import json
from pathlib import Path
import tempfile
import unittest
from eval_luna_annotated import normalize
from score_luna_annotated import score, compare, load_results, timing

class LunaEvaluationTests(unittest.TestCase):
    def setUp(self):
        self.d=[dict(candidate_id='e:0',class_name='人员',bbox=[0,0,.1,.1]),
                dict(candidate_id='e:1',class_name='车',bbox=[.2,.2,.3,.3])]
    def body(self):
        return dict(requested_model='gpt-5.6-luna', results=[
            dict(target_index=1,decision='uncertain',reasoning='模糊'),
            dict(target_index=0,decision='true',reasoning='人体结构')])
    def test_mapping_and_policy(self):
        r=normalize(self.body(),self.d)['results']
        self.assertEqual([x['rawDecision'] for x in r],['true','uncertain'])
        self.assertEqual([x['decision'] for x in r],['confirmed','rejected'])
    def test_duplicate_or_wrong_indices_rejected(self):
        for index in [0,2,False]:
            b=self.body();b['results'][0]['target_index']=index
            with self.assertRaises(ValueError): normalize(b,self.d)
    def test_incomplete_invalid_or_wrong_model_rejected(self):
        for field,value in [('decision','maybe'),('reasoning','')]:
            b=self.body();b['results'][0][field]=value
            with self.assertRaises(ValueError): normalize(b,self.d)
        b=self.body();b['results'].pop()
        with self.assertRaises(ValueError): normalize(b,self.d)
        b=self.body();b['requested_model']='other'
        with self.assertRaises(ValueError): normalize(b,self.d)
    def test_missing_manual_and_confusion_separated(self):
        truth={'a':True,'b':False,'c':True,'d':False,'e':True}
        r=score(truth,{'a':'rejected','b':'confirmed','c':'uncertain','d':'rejected'})
        self.assertEqual([r[x] for x in ['right','wrong','manual','missing','wrongly_rejected','wrongly_confirmed']],[1,2,1,1,1,1])
        self.assertEqual(r['accuracy_decided'],1/3)
        self.assertEqual(r['coverage_available'],3/4)
    def test_paired_denominator_and_referral(self):
        r=compare({'a':True,'b':False},{'a':'uncertain','b':'confirmed'},{'a':'rejected','b':'rejected'})
        self.assertEqual(r['transitions'],{'manual->wrong':1,'wrong->right':1})
        self.assertEqual(r['jointly_decided'],1)
    def test_failure_not_false_and_order_checked(self):
        with tempfile.TemporaryDirectory() as folder:
            p=Path(folder)/'e.json';item=dict(event_id='e',candidate_ids=['e:0','e:1'],elapsed=1,
                response=dict(success=False,errorCode='http_429',results=[]))
            p.write_text(json.dumps(item));d,e,f=load_results(Path(folder),{'e':{'detections':self.d}},luna=True)
            self.assertEqual(d,{});self.assertEqual(f,{'e':'http_429'})
            item['candidate_ids'].reverse();p.write_text(json.dumps(item))
            with self.assertRaises(ValueError): load_results(Path(folder),{'e':{'detections':self.d}})
    def test_timing_nearest_rank_and_failures(self):
        r=timing([dict(elapsed=x,response={'success':True}) for x in range(1,11)]+[dict(elapsed=999,response={'success':False})])
        self.assertEqual(r['median_seconds'],5.5);self.assertEqual(r['p90_seconds'],9)

if __name__=='__main__': unittest.main()
