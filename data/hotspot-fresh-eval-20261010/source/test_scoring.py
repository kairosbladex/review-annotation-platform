"""Guard final-evaluation gates against missing negative gold and failed calls."""
import contextlib, io, json, tempfile, unittest
from pathlib import Path
import evaluate
class ScoringTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.previous=evaluate.ROOT;evaluate.ROOT=Path(self.tmp.name)
        root=evaluate.ROOT;(root/'runs/holdout/out').mkdir(parents=True)
        evaluate.write('labels.json',[{'key':'box','label':'是光斑'}])
        evaluate.write('split.json',[{'event_id':'e','split':'holdout','screenshot_sha256':'unique','boxes':[{'key':'box'}]}])
        schedule=[]
        for mode in ['screenshot','original']:
            for card in ['p000','p001_p002','repair']:
                for rep in [1,2]:
                    job={'event_id':'e','mode':mode,'card':card,'repeat':rep};schedule.append(job)
                    result={'execution_id':f'{mode}-{card}-{rep}','results':[{'target_index':0,'decision':'false' if card=='p000' else 'true'}]}
                    evaluate.write(f'runs/holdout/out/{mode}-{card}-{rep}.json',{**job,'keys':['box'],'success':True,'attempts':[{'status':'ok'}],'result':result})
        evaluate.write('runs/holdout/schedule.json',schedule)
    def tearDown(self):
        evaluate.ROOT=self.previous;self.tmp.cleanup()
    def score(self):
        with contextlib.redirect_stdout(io.StringIO()):evaluate.score('holdout')
    def test_perfect_improvement_cannot_pass_without_negative_gold(self):
        self.score();r=evaluate.read('holdout-report.json')
        self.assertEqual(r['negative_gold_count'],0)
        for g in r['gates']:
            self.assertEqual(g['accuracy_lower'],1.0);self.assertFalse(g['passed']);self.assertIsNone(g['false_accept_not_increased_each_repeat'])
    def test_failed_call_is_not_false_label(self):
        name='runs/holdout/out/screenshot-repair-1.json';r=evaluate.read(name);r.update(success=False,result=None);evaluate.write(name,r)
        with self.assertRaisesRegex(AssertionError,'Incomplete'):self.score()
        self.assertFalse((evaluate.ROOT/'holdout-report.json').exists())
    def test_reused_execution_rejected(self):
        name='runs/holdout/out/screenshot-repair-1.json';r=evaluate.read(name);r['result']['execution_id']='original-p000-1';evaluate.write(name,r)
        with self.assertRaises(AssertionError):self.score()
if __name__=='__main__':unittest.main()
