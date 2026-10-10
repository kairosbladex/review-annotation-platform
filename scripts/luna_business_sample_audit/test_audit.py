import importlib.util,json,sys
from pathlib import Path
import pytest
sys.path.insert(0,str(Path(__file__).parent))
from audit import match_boxes

def box():return {'x':10,'y':20,'width':30,'height':40,'frame_width':100,'frame_height':200,'class_name':'人员','box_format':'xywh_top_left'}
def result():return {'index':1,'originalClass':'人员','originalBbox':[.1,.1,.4,.3]}
def test_identity_requires_index_class_and_geometry():
 assert match_boxes([box()],[result()])
 for field,value in [('index',0),('originalClass','车'),('originalBbox',[.2,.1,.4,.3])]:
  r=result();r[field]=value;assert not match_boxes([box()],[r])
def test_missing_duplicate_and_reordered_results_fail():
 assert not match_boxes([box()],[])
 assert not match_boxes([box(),box()],[result(),result()])
 assert not match_boxes([box(),box()],[dict(result(),index=2),result()])
def test_readonly_snapshot_rolls_back_on_wrong_database(monkeypatch,tmp_path):
 import snapshot
 class C:
  def __init__(self):self.queries=[];self.rolled=False;self.closed=False
  def cursor(self):return self
  def __enter__(self):return self
  def __exit__(self,*args):pass
  def execute(self,sql,args=()):self.queries.append(sql)
  def fetchall(self):return [{'db':'wrong','read_only':1}]
  def rollback(self):self.rolled=True
  def close(self):self.closed=True
 c=C();monkeypatch.setattr(snapshot,'connect',lambda _:c)
 with pytest.raises(AssertionError):snapshot.run(tmp_path/'env',tmp_path/'out')
 assert c.rolled and c.closed
 assert 'START TRANSACTION WITH CONSISTENT SNAPSHOT, READ ONLY' in c.queries
 assert not (tmp_path/'out').exists()
def test_prepared_pack_is_blind_and_keeps_recent_quarantined():
 root=Path(__file__).resolve().parents[2]/'data/luna-business-sample-audit'
 if not (root/'private/diagnostic.json').exists():pytest.skip('private local evidence not shipped')
 d=json.loads((root/'private/diagnostic.json').read_text());r=json.loads((root/'private/recent-reserved.json').read_text())
 assert all(x['label']=='' for x in d+r)
 assert all(x['eligibility']=='pending' for x in r)
 assert not {x['task_id'] for x in d}&{x['task_id'] for x in r}
 html=(root/'private/index.html').read_text()
 for forbidden in ['reasoning','recheck_result','detector_confidence','falseCount','originalConfidence','candidate_key','event_id']:
  assert forbidden not in html
 assert 'fetch(' not in html and 'XMLHttpRequest' not in html
