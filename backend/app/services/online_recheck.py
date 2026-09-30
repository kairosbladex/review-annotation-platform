"""Durable GPT advice outbox. Unknown execution is never automatically retried."""
from __future__ import annotations
import hashlib
import json
import time
from datetime import timedelta
from io import BytesIO
from sqlalchemy import select, update
from PIL import Image
from ..models import Annotation, Asset, Candidate, RecheckRecord, Sample, utc_now
from ..adapters.recheck_relay import RelayFailure
from ..adapters.codex_responses import CodexResponsesVisionAdapter, _normalise_decisions
from .auto_annotation_prompts import render_batch_prompt, merge_agent_results, ROLE_ORDER
from .review_views import build_visual_views

SUPPORTED={'车','公交车','卡车','三轮车','两轮车','人员'}
RULE_VERSION='conservative-v1'
CROP_VERSION='clean-three-view-v1'
PROMPT_VERSION='online-baseline-v1'
def object_schema(properties):
    return {'type':'object','additionalProperties':False,'required':list(properties),'properties':properties}

RECHECK_SCHEMA=object_schema({
    'scene':object_schema({'label':{'type':'string','enum':['rooftop','ground','unknown']},'confidence':{'type':'number'},'reason':{'type':'string'}}),
    'decisions':{'type':'array','items':object_schema({
        'target_index':{'type':'integer'},'consensus':{'type':'string','enum':['true','false','uncertain']},
        'votes':{'type':'array','items':object_schema({'role':{'type':'string','enum':list(ROLE_ORDER)},'label':{'type':'string','enum':['true','false','uncertain']},'confidence':{'type':'number'},'reason':{'type':'string'}})}
    })}
})

def configuration(settings):
    if (settings.recheck_rule_version,settings.recheck_crop_version,settings.recheck_prompt_version)!=(RULE_VERSION,CROP_VERSION,PROMPT_VERSION):
        raise ValueError('unregistered recheck version')
    spec={'provider':'gpt','model':settings.auto_annotation_model,'prompt':settings.recheck_prompt_version,
          'rule':settings.recheck_rule_version,'crop':settings.recheck_crop_version,
          'prompt_hash':hashlib.sha256(render_batch_prompt([]).encode()).hexdigest()}
    return hashlib.sha256(json.dumps(spec,sort_keys=True).encode()).hexdigest()

def create_records(session,sample,settings,candidate_ids=None):
    if not settings.recheck_enabled:return []
    config=configuration(settings);records=[]
    session.flush()
    for c in session.scalars(select(Candidate).where(Candidate.sample_id==sample.id)).all():
        if candidate_ids is not None and c.id not in candidate_ids:continue
        existing=session.scalar(select(RecheckRecord).where(RecheckRecord.candidate_id==c.id,RecheckRecord.config_version==config))
        if existing:continue
        record=RecheckRecord(candidate_id=c.id,event_id=sample.source_event_id or sample.id,class_name=c.class_name,
                             requested_model=settings.auto_annotation_model,config_version=config,
                             prompt_version=settings.recheck_prompt_version,rule_version=settings.recheck_rule_version,
                             crop_version=settings.recheck_crop_version)
        if c.class_name not in SUPPORTED:record.status='skipped';record.error='无判定标准'
        session.add(record);records.append(record)
    session.flush();return records

def prompt_and_images(data,box,class_name):
    # Whitelist: first-review decisions, labels and reasoning never enter prompts.
    candidates=[{'target_index':0,'class_name':class_name,'bbox':box}]
    views=build_visual_views(data,candidates)
    return render_batch_prompt(candidates),[data,*views.values()]

def parse_advice(answer):
    decisions=_normalise_decisions(answer)
    if len(decisions)!=1 or decisions[0]['target_index']!=0:raise ValueError('invalid_target_response')
    d=decisions[0]; votes={role:{k:v for k,v in vote.items() if k!='role'} for role,vote in zip(ROLE_ORDER,d['votes'],strict=True)}
    merged=merge_agent_results(votes)
    return {'decision':merged['label'] if merged else 'uncertain',
            'reasoning':'；'.join(str(v['reason']) for v in d['votes']),
            'scene':d.get('scene',{}).get('label','unknown')}

def execute(record_id,*,factory,settings,store,relay):
    if not settings.recheck_enabled:return None
    started=time.monotonic()
    with factory() as session:
        r=session.get(RecheckRecord,record_id)
        if not r or r.status not in {'queued','waiting_quota'}:return None
        if r.next_attempt_at and r.next_attempt_at.replace(tzinfo=utc_now().tzinfo)>utc_now():return None
        if r.config_version!=configuration(settings):
            r.status='skipped';r.error='复检配置已变化，历史任务不自动重跑';session.commit();return None
        if r.attempts>=2:
            r.status='failed';r.error='maximum_attempts_reached';session.commit();return None
        # Compare-and-set under concurrent delivery; only one caller can own work.
        changed=session.execute(update(RecheckRecord).where(RecheckRecord.id==r.id,RecheckRecord.status.in_(['queued','waiting_quota'])).values(status='running',error=None,started_at=utc_now()))
        session.commit()
        if changed.rowcount!=1:return None
        session.refresh(r)
        c=session.get(Candidate,r.candidate_id);sample=session.get(Sample,c.sample_id)
        annotation=session.scalar(select(Annotation).where(Annotation.candidate_id==c.id))
        metadata=json.loads(sample.metadata_json)
        if annotation and annotation.labeled_by and annotation.label is not None or metadata.get('human_handled'):
            r.status='skipped';r.error='事件已被人工处理';session.commit();return None
        if c.class_name not in SUPPORTED:
            r.status='skipped';r.error='无判定标准';session.commit();return None
        try:
            quota=relay.usage()
            if quota['remaining']<settings.recheck_quota_floor:raise RelayFailure('quota_below_floor')
        except Exception as e:
            r.status='waiting_quota';r.error=getattr(e,'code','usage_unavailable');session.commit();return None
        try:
            assets=list(session.scalars(select(Asset).where(Asset.sample_id==c.sample_id)).all())
            asset=next((a for a in assets if a.kind=='clean_original'),None)
            box=[c.x1,c.y1,c.x2,c.y2]
            if asset:
                if metadata.get('clean_event_id')!=r.event_id:raise ValueError('图片与事件错配')
                box=metadata.get('clean_bboxes',{}).get(str(c.target_index))
                if box is None:raise ValueError('干净原图目标坐标缺失')
                r.image_limitation=''
            else:
                asset=next((a for a in assets if a.kind=='original'),None)
                r.image_limitation='无已核验干净原图；仅接入截图，可能存在红框遮挡'
            if not asset:raise ValueError('图片缺失')
            data=store.get_bytes(asset.bucket,asset.object_key)
            digest=hashlib.sha256(data).hexdigest()
            if not asset.sha256 or digest!=asset.sha256:raise ValueError('图片哈希未核验或不匹配')
            with Image.open(BytesIO(data)) as im:
                if min(im.size)<1000:r.image_limitation+='；可用图片分辨率不足高清'
            r.image_sha256=digest
            prompt,images=prompt_and_images(data,box,c.class_name)
            r.attempts+=1;session.commit() # durably record possible paid request BEFORE send
            result=relay.request(model=r.requested_model,prompt=prompt,images=images,
                                 schema=RECHECK_SCHEMA,
                                 idempotency_key=r.id)
            r.actual_model=result.get('actual_model');r.usage_json=json.dumps(result.get('usage',{}),ensure_ascii=False);r.response_id=result.get('response_id')
            advice=parse_advice(result['answer'])
            if not result.get('actual_model'):raise ValueError('实际模型名未返回')
            r.status='done';r.decision=advice['decision'];r.reasoning=advice['reasoning'];r.scene=advice['scene']
            r.actual_model=result['actual_model'];r.usage_json=json.dumps(result.get('usage',{}),ensure_ascii=False);r.response_id=result.get('response_id')
            r.completed_at=utc_now();r.error=None
        except RelayFailure as e:
            r.decision=None;r.error=e.code
            if e.code in {'auth_failed','missing_credential'}:r.status='waiting_quota'
            elif e.code=='rate_limited' and r.attempts<2:
                r.status='queued';r.next_attempt_at=utc_now()+timedelta(seconds=e.retry_after or 60)
            else:r.status='failed'
        except Exception as e:
            r.status='failed';r.decision=None;r.error=str(e)[:200] if isinstance(e,ValueError) else type(e).__name__
        r.elapsed_seconds=time.monotonic()-started;session.commit()
        return max(1,(r.next_attempt_at.replace(tzinfo=utc_now().tzinfo)-utc_now()).total_seconds()) if r.status=='queued' and r.next_attempt_at else None

def recover_interrupted(session,before):
    # Only explicitly called after checking the old worker has stopped.
    result=session.execute(update(RecheckRecord).where(RecheckRecord.status=='running',RecheckRecord.started_at<before).values(status='failed',decision=None,error='execution_unknown: worker interrupted; do not resend automatically'))
    session.commit();return result.rowcount

def serialize_record(r):
    return {k:getattr(r,k) for k in ('id','event_id','candidate_id','image_sha256','class_name','provider','requested_model','actual_model','config_version','prompt_version','rule_version','crop_version','status','decision','reasoning','scene','image_limitation','elapsed_seconds','error','attempts','created_at','completed_at')}

def pending_review(session,rate=0.1,limit=100):
    if not 0<=rate<=1:raise ValueError('invalid sample rate')
    records=session.scalars(select(RecheckRecord).order_by(RecheckRecord.created_at.desc())).all()
    selected=[];seen=set()
    for r in records:
        if r.candidate_id in seen:continue
        seen.add(r.candidate_id)
        if r.status!='done':continue
        a=session.scalar(select(Annotation).where(Annotation.candidate_id==r.candidate_id))
        if a and a.labeled_by and a.label is not None:continue
        c=session.get(Candidate,r.candidate_id)
        first={'confirmed':'true','rejected':'false','true':'true','false':'false'}.get((c.raw_decision or '').lower())
        priority=0 if r.decision=='uncertain' or r.decision!=first else 1
        if priority==1 and int(hashlib.sha256(r.candidate_id.encode()).hexdigest()[:8],16)/2**32>=rate:continue
        selected.append({'priority':priority,'sample_id':c.sample_id,'first_review':first,**serialize_record(r)})
    return sorted(selected,key=lambda x:(x['priority'],str(x['created_at'])))[:limit]
