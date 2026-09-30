"""Isolated SQLite evidence; all relay failures are injected, never billed."""
import hashlib
import io
import json
from dataclasses import replace
from datetime import timedelta
import pytest
from PIL import Image
from sqlalchemy import select
from fastapi.testclient import TestClient
from app.main import create_app
from app.models import RecheckRecord, Annotation, AuditEvent
from app.services.online_recheck import execute, recover_interrupted, pending_review, ROLE_ORDER
from app.adapters.recheck_relay import RelayFailure
from .test_auto_annotation import _settings
from .test_annotation_flow import _login, _review_event
from .test_imports import MemoryObjectStore

class Dispatch:
    def __init__(self):self.ids=[]
    def dispatch_recheck(self,record_id):self.ids.append(record_id)

class Relay:
    def __init__(self, failure=None, balance=100):self.calls=[];self.failure=failure;self.balance=balance
    def usage(self):
        if self.failure and self.failure.startswith('usage'):raise RelayFailure(self.failure)
        return {'remaining':self.balance}
    def request(self,**kwargs):
        self.calls.append(kwargs)
        if self.failure:raise RelayFailure(self.failure,1)
        return {'actual_model':'gpt-5.6-luna','usage':{'total_tokens':123},'answer':{
            'scene':{'label':'ground','confidence':0.8,'reason':'test'},
            'decisions':[{'target_index':0,'votes':[{'role':r,'label':'true','confidence':0.9,'reason':'visible'} for r in ROLE_ORDER],'consensus':'true'}]}}

@pytest.fixture
def env(tmp_path):
    settings=replace(_settings(tmp_path),recheck_enabled=True)
    store=MemoryObjectStore();b=io.BytesIO();Image.new('RGB',(1200,1100),'gray').save(b,format='JPEG');data=b.getvalue()
    store.put_bytes('images','raw/event.jpg',data,'image/jpeg')
    dispatch=Dispatch();app=create_app(settings,object_store=store,recheck_dispatcher=dispatch);client=TestClient(app)
    event=_review_event("isolated-event");event['image']={'bucket':'images','object_key':'raw/event.jpg','sha256':hashlib.sha256(data).hexdigest()}
    # Use one candidate with a Chinese supported class.
    event['targets']=event['targets'][:1];event['targets'][0]['class_name']='车';event['targets'][0]['raw_decision']='confirmed'
    return settings,store,dispatch,app,client,event

def ingest(env):
    settings,store,dispatch,app,client,event=env
    r=client.post('/api/v1/ingest/review-events',json=event,headers={'X-Service-Key':'service-secret'});assert r.status_code==201,r.text
    with app.state.session_factory() as s:rid=s.scalar(select(RecheckRecord.id))
    return rid

def run(env,rid,relay):
    settings,store,_,app,_,_=env
    return execute(rid,factory=app.state.session_factory,settings=settings,store=store,relay=relay)

def record(env,rid):
    with env[3].state.session_factory() as s:return s.get(RecheckRecord,rid)

def test_duplicate_push_and_delivery_do_not_bill_twice_or_create_gold(env):
    rid=ingest(env);client=env[4]
    assert client.post('/api/v1/ingest/review-events',json=env[5],headers={'X-Service-Key':'service-secret'}).status_code==200
    assert len(env[2].ids)==1
    relay=Relay();run(env,rid,relay);run(env,rid,relay)
    assert len(relay.calls)==1 and record(env,rid).status=='done'
    assert 'confirmed' not in relay.calls[0]['prompt']
    with env[3].state.session_factory() as s:assert s.scalar(select(Annotation)) is None

@pytest.mark.parametrize('failure,balance',[('usage_auth_failed',100),('usage_unavailable',100),(None,4)])
def test_quota_failure_is_waiting_without_request_or_decision(env,failure,balance):
    rid=ingest(env);relay=Relay(failure,balance);run(env,rid,relay)
    r=record(env,rid);assert r.status=='waiting_quota' and r.decision is None and not relay.calls

def test_429_obeys_wait_and_max_two_attempts(env):
    rid=ingest(env);relay=Relay('rate_limited');assert run(env,rid,relay)>0
    run(env,rid,relay);assert len(relay.calls)==1
    with env[3].state.session_factory() as s:
        r=s.get(RecheckRecord,rid);r.next_attempt_at=None;s.commit()
    run(env,rid,relay);assert record(env,rid).status=='failed' and len(relay.calls)==2
    assert record(env,rid).decision is None

@pytest.mark.parametrize('failure',['execution_unknown','invalid_response','unexpected_tool_use'])
def test_failed_request_never_becomes_uncertain_or_retries(env,failure):
    rid=ingest(env);relay=Relay(failure);run(env,rid,relay);run(env,rid,relay)
    r=record(env,rid);assert r.status=='failed' and r.decision is None and len(relay.calls)==1

def test_interrupted_running_record_requires_explicit_recovery_and_is_not_resent(env):
    from app.models import utc_now
    rid=ingest(env)
    with env[3].state.session_factory() as s:
        r=s.get(RecheckRecord,rid);r.status='running';r.attempts=1;r.started_at=utc_now()-timedelta(seconds=100);s.commit()
        assert recover_interrupted(s,utc_now())==1
    relay=Relay();run(env,rid,relay);assert not relay.calls and record(env,rid).status=='failed'

@pytest.mark.parametrize('handled',['unsupported','human'])
def test_skips_unsupported_and_human_handled(env,handled):
    if handled=='unsupported':env[5]['targets'][0]['class_name']='光伏板'
    else:env[5]['metadata']={'human_handled':True}
    rid=ingest(env);relay=Relay();run(env,rid,relay);assert record(env,rid).status=='skipped' and not relay.calls

def test_hash_mismatch_fails_before_external_call(env):
    env[5]['image']['sha256']='0'*64;rid=ingest(env);relay=Relay();run(env,rid,relay)
    assert record(env,rid).status=='failed' and not relay.calls

def test_clean_image_event_mismatch_rejected(env):
    env[5]['clean_image']=env[5]['image'];env[5]['clean_event_id']='wrong';env[5]['targets'][0]['clean_bbox']=[0.1,0.1,0.2,0.2]
    r=env[4].post('/api/v1/ingest/review-events',json=env[5],headers={'X-Service-Key':'service-secret'})
    assert r.status_code==422

def test_human_revisions_keep_previous_value_and_queue_excludes_confirmed(env):
    rid=ingest(env);run(env,rid,Relay());app=env[3];client=env[4]
    headers=_login(client)
    c=record(env,rid).candidate_id
    for label,rev in [('false',0),('true',1)]:
        result=client.patch('/api/v1/candidates/'+c+'/annotation',json={'label':label,'note':'human','expected_revision':rev},headers=headers)
        assert result.status_code==200,result.text
    with app.state.session_factory() as s:
        a=s.scalar(select(Annotation));assert a.label=='true' and a.labeled_by and a.revision==2
        history=s.scalars(select(AuditEvent).where(AuditEvent.action=='annotation.updated').order_by(AuditEvent.created_at)).all()
        assert json.loads(history[-1].payload_json)['previous']['label']=='false'
        assert not pending_review(s,1)

def test_disabled_default_and_old_duplicate_do_not_queue(env):
    env[3].state.settings # endpoint closure uses frozen settings; create separate app
    settings,store,_,_,_,event=env
    dispatch=Dispatch();app=create_app(replace(settings,recheck_enabled=False),object_store=store,recheck_dispatcher=dispatch);client=TestClient(app)
    client.post('/api/v1/ingest/review-events',json=event,headers={'X-Service-Key':'service-secret'})
    assert not dispatch.ids
    with app.state.session_factory() as s:assert not s.scalar(select(RecheckRecord))


def test_quota_failure_cannot_overwrite_a_concurrent_worker_or_send_twice(env):
    rid=ingest(env);second=Relay()
    class RacingQuota(Relay):
        def usage(self):
            run(env,rid,second)
            raise RelayFailure('usage_unavailable')
    run(env,rid,RacingQuota())
    assert not second.calls
    assert record(env,rid).status=='waiting_quota'


def test_manual_confirmation_before_worker_skips_without_external_call(env):
    rid=ingest(env);c=record(env,rid).candidate_id
    result=env[4].patch('/api/v1/candidates/'+c+'/annotation',json={'label':'true','expected_revision':0},headers=_login(env[4]))
    assert result.status_code==200
    relay=Relay();run(env,rid,relay)
    assert record(env,rid).status=='skipped' and not relay.calls


def test_broker_failure_keeps_committed_outbox_and_successful_intake(env):
    class BrokenBroker:
        def dispatch_recheck(self,record_id):raise ConnectionError('isolated broker unavailable')
    env[3].state.recheck_dispatcher=BrokenBroker()
    rid=ingest(env)
    assert record(env,rid).status=='queued'


def test_database_rejects_done_without_a_decision(env):
    from sqlalchemy.exc import IntegrityError
    rid=ingest(env)
    with env[3].state.session_factory() as s:
        r=s.get(RecheckRecord,rid);r.status='done'
        with pytest.raises(IntegrityError):s.commit()
        s.rollback()
