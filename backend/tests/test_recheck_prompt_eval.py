import importlib.util
import json
import sys
from pathlib import Path
import pytest
from PIL import Image
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'scripts'))
SPEC=importlib.util.spec_from_file_location('prompt_eval',ROOT/'scripts/eval_recheck_prompt.py');module=importlib.util.module_from_spec(SPEC);SPEC.loader.exec_module(module)

def test_wilson_and_exact_mcnemar():
    c=module.wilson(0,100);assert c['numerator']==0 and c['denominator']==100
    assert c['wilson95'][1]==pytest.approx(0.0369935,abs=1e-6)
    m=module.mcnemar([(True,False)]*6);assert m['exact_two_sided_p']==pytest.approx(0.03125)
    assert module.wilson(0,0)['wilson95'] is None

class FakeRelay:
    calls=[]
    def __init__(self,_):pass
    def usage(self,force=False):return {'remaining':1000,'unit':'USD'}
    def request(self,**kwargs):
        self.calls.append(kwargs)
        from app.services.online_recheck import ROLE_ORDER
        return {'actual_model':'test-model-'+str(len(self.calls)%2),'usage':{'total_tokens':10},'answer':{'scene':{'label':'ground','confidence':0.9,'reason':'test'},'decisions':[{'target_index':0,'consensus':'true','votes':[{'role':r,'label':'true','confidence':0.9,'reason':'test'} for r in ROLE_ORDER]}]}}

def test_budget_reservations_stop_even_if_usage_is_delayed(tmp_path,monkeypatch):
    import hashlib
    image=tmp_path/'input.jpg';Image.new('RGB',(100,100),'gray').save(image);digest=hashlib.sha256(image.read_bytes()).hexdigest()
    rows=[{'gold_id':str(i),'group_id':str(i),'class_name':'车','population':['random_traffic'],'label':'true','source_images':[str(image)],'image_sha256':digest,'bbox':[0.1,0.1,0.4,0.4],'split':'development','valid_for':[]} for i in range(3)]
    gold=tmp_path/'gold.jsonl';gold.write_text(''.join(json.dumps(r)+'\n' for r in rows));out=tmp_path/'eval'
    FakeRelay.calls=[];monkeypatch.setattr(module,'RecheckRelay',FakeRelay)
    monkeypatch.setattr(sys,'argv',['eval','--gold',str(gold),'--split','development','--baseline','--limit','0','--budget-usd','.5','--reserve-per-request-usd','.25','--out',str(out)])
    module.main();report=json.loads((out/'report.json').read_text())
    assert len(FakeRelay.calls)==2 and report['stop_reason']=='budget_reservation_exhausted'
    assert report['reserved_usd']==.5 and report['model_changed']
    assert all('"label":"true"' not in r['prompt'] for r in FakeRelay.calls) # no human labels in candidate payload


def test_unapproved_locked_candidate_is_rejected_before_calls(tmp_path,monkeypatch):
    a=tmp_path/'a';b=tmp_path/'b';a.write_text('A');b.write_text('B')
    monkeypatch.setattr(sys,'argv',['eval','--gold',str(tmp_path/'gold'),'--split','locked_test_gpt','--prompt-a',str(a),'--prompt-b',str(b),'--out',str(tmp_path/'out')])
    with pytest.raises(SystemExit):module.main()
