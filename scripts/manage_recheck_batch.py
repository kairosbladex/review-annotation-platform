#!/usr/bin/env python3
"""Explicit operations only; no automatic historical reruns or unknown resend."""
import argparse
import json
import sys
from datetime import datetime,timezone
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'backend'))
from sqlalchemy import select
from app.config import Settings
from app.db import build_engine,build_session_factory
from app.models import RecheckRecord,Candidate,Sample,AuditEvent
from app.services.online_recheck import create_records,recover_interrupted
from app.tasks.dispatcher import CeleryDispatcher

def main():
    p=argparse.ArgumentParser();p.add_argument('action',choices=['enqueue-due','recover-interrupted','explicit-batch'])
    p.add_argument('--before');p.add_argument('--candidate-ids',type=Path);p.add_argument('--batch-id');p.add_argument('--confirm-old-worker-stopped',action='store_true');a=p.parse_args()
    settings=Settings.from_env()
    if not settings.recheck_enabled:p.error('feature disabled')
    factory=build_session_factory(build_engine(settings.database_url));dispatcher=CeleryDispatcher(settings.redis_url)
    with factory() as s:
        if a.action=='recover-interrupted':
            if not a.before or not a.confirm_old_worker_stopped:p.error('explicit cutoff and stopped-worker confirmation required')
            print('recovered_as_failed',recover_interrupted(s,datetime.fromisoformat(a.before)));return
        if a.action=='explicit-batch':
            if not a.candidate_ids or not a.batch_id:p.error('candidate IDs and batch ID required')
            ids=json.loads(a.candidate_ids.read_text());candidates=s.scalars(select(Candidate).where(Candidate.id.in_(ids))).all()
            if len(candidates)!=len(set(ids)):p.error('unknown candidate IDs')
            records=[]
            for sample_id in {c.sample_id for c in candidates}:
                records+=create_records(s,s.get(Sample,sample_id),settings,candidate_ids=set(ids))
            s.add(AuditEvent(action='recheck.explicit_batch',entity_type='recheck_batch',entity_id=a.batch_id,payload_json=json.dumps({'candidate_ids':ids})))
            s.commit();queued=[r.id for r in records if r.status=='queued']
        else:
            now=datetime.now(timezone.utc)
            queued=[r.id for r in s.scalars(select(RecheckRecord).where(RecheckRecord.status.in_(['queued','waiting_quota']))).all() if r.next_attempt_at is None or r.next_attempt_at.replace(tzinfo=timezone.utc)<=now]
    for rid in queued:dispatcher.dispatch_recheck(rid)
    print('queued',len(queued))

if __name__=='__main__':main()
