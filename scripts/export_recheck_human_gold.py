#!/usr/bin/env python3
"""Export human-confirmed GPT advice targets; no model-only labels."""
import argparse
import json
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'backend'))
from sqlalchemy import select
from app.config import Settings
from app.db import build_engine,build_session_factory
from app.models import Annotation,Candidate,Sample,Asset,User,AuditEvent,RecheckRecord

def export(factory):
    rows=[]
    with factory() as s:
        for a in s.scalars(select(Annotation).where(Annotation.labeled_by.is_not(None),Annotation.labeled_at.is_not(None),Annotation.label.in_(['true','false']))).all():
            c=s.get(Candidate,a.candidate_id)
            if not s.scalar(select(RecheckRecord.id).where(RecheckRecord.candidate_id==c.id)):continue
            # A durable manual annotation audit must match the current revision.
            audits=s.scalars(select(AuditEvent).where(AuditEvent.entity_id==c.id,AuditEvent.action=='annotation.updated')).all()
            if not any(e.user_id==a.labeled_by and json.loads(e.payload_json).get('revision')==a.revision for e in audits):continue
            sample=s.get(Sample,c.sample_id);metadata=json.loads(sample.metadata_json)
            asset=s.scalar(select(Asset).where(Asset.sample_id==c.sample_id,Asset.kind=='original'))
            if not asset or not asset.sha256:continue
            row={'candidate_id':c.id,'event_id':sample.source_event_id or sample.id,'source_task_id':metadata.get('source_task_id'),
                 'class_name':c.class_name,'bbox':[c.x1,c.y1,c.x2,c.y2],'image_sha256':asset.sha256,
                 'label':a.label,'labeled_by':s.get(User,a.labeled_by).username,'labeled_at':a.labeled_at.isoformat(),'label_source':'human_confirmed',
                 'revision':a.revision,'note':a.note,'exposure':['gpt_prompt_exposed'],'population':metadata.get('population',['historical_review']),
                 'image_bucket':asset.bucket,'image_object_key':asset.object_key}
            rows.append(row)
    return rows

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    factory=build_session_factory(build_engine(Settings.from_env().database_url));rows=export(factory)
    a.out.parent.mkdir(parents=True,exist_ok=True);a.out.write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows));print('human_confirmed',len(rows))
