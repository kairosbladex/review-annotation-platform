#!/usr/bin/env python3
"""Bounded local UI preview; no Celery, shared DB, relay credentials or dispatch."""
import argparse
import os
import socket
import subprocess
import sys
import time
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source-root',type=Path,required=True)
    p.add_argument('--database',type=Path,default=ROOT/'data/recheck-online-evidence-20260930/live-chain/replay.sqlite3')
    p.add_argument('--wall-seconds',type=int,default=3600)
    a=p.parse_args();db=a.database.resolve()
    if not db.is_file():p.error('local replay database missing; do not initialize another source')
    if a.wall_seconds<=0:p.error('wall seconds must be positive')
    for port in (8197,5197):
        with socket.socket() as s:
            if s.connect_ex(('127.0.0.1',port))==0:p.error('local preview port occupied: '+str(port))
    env={**os.environ,'PYTHONPATH':str(ROOT/'backend'),'DATABASE_URL':'sqlite:///'+str(db),
         'REDIS_URL':'redis://127.0.0.1:6397/0','MINIO_ENDPOINT':'','MINIO_AUTO_CREATE_BUCKET':'false',
         'LOCAL_READONLY_IMAGE_ROOT':str(a.source_root.resolve()/'data'),'SOURCE_IMAGE_BASE_URL':'',
         'RECHECK_ENABLED':'false','AUTO_ANNOTATION_API_KEY':'','MODEL_SERVICE_BASE_URL':'','MODEL_SERVICE_API_KEY':'',
         'SERVICE_API_KEY':'local-preview-no-ingestion','ADMIN_USERNAME':'local-live-admin','ADMIN_PASSWORD':'local-live-password',
         'JWT_SECRET':'local-live-replay-jwt-secret-32-bytes','VITE_API_PROXY_TARGET':'http://127.0.0.1:8197'}
    commands=[[sys.executable,'-m','uvicorn','app.main:app','--host','127.0.0.1','--port','8197','--no-access-log'],
              ['node',str(ROOT/'frontend/node_modules/vite/bin/vite.js'),str(ROOT/'frontend'),'--host','127.0.0.1','--port','5197','--strictPort']]
    processes=[]
    try:
        for command in commands:
            processes.append(subprocess.Popen([sys.executable,str(ROOT/'scripts/limit_local_process.py'),'--memory-mb','512','--cpus','1','--wall-seconds',str(a.wall_seconds),'--',*command],cwd=ROOT,env=env))
        print('Preview http://127.0.0.1:5197 ; local-only account local-live-admin / local-live-password ; Ctrl-C stops both processes',flush=True)
        while all(x.poll() is None for x in processes):time.sleep(.2)
    except KeyboardInterrupt:pass
    finally:
        for x in processes:
            if x.poll() is None:x.terminate()
        for x in processes:
            try:x.wait(timeout=10)
            except subprocess.TimeoutExpired:x.kill();x.wait()

if __name__=='__main__':main()
