#!/usr/bin/env python3
"""Local process-group guard for macOS: RSS limit and aggregate CPU pacing.

Launch only task-owned commands. No unrelated process is signalled.
"""
import argparse
import os
import signal
import subprocess
import time
import resource

def seconds(value):
    parts=value.replace('-',':').split(':');total=0.0
    for p in parts:total=total*60+float(p)
    return total

def run(command,memory_mb=1024,cpus=1,wall_seconds=3600):
    def limits():
        resource.setrlimit(resource.RLIMIT_CORE,(0,0))
        cpu_hard=resource.getrlimit(resource.RLIMIT_CPU)[1]
        wanted=int(wall_seconds*cpus)+1
        hard=wanted if cpu_hard==resource.RLIM_INFINITY else min(wanted,cpu_hard)
        resource.setrlimit(resource.RLIMIT_CPU,(min(int(wall_seconds*cpus),hard),hard))
        resource.setrlimit(resource.RLIMIT_NOFILE,(256,256))
    def interrupted(signum,frame):
        raise KeyboardInterrupt
    signal.signal(signal.SIGTERM,interrupted)
    signal.signal(signal.SIGINT,interrupted)
    proc=subprocess.Popen(command,start_new_session=True,preexec_fn=limits)
    start=time.monotonic();reason=None
    try:
        while proc.poll() is None:
            output=subprocess.check_output(['ps','-axo','pid=,ppid=,pgid=,rss=,time='],text=True)
            members=[]
            for line in output.splitlines():
                fields=line.split()
                if len(fields)==5 and int(fields[2])==proc.pid:members.append(fields)
            rss=sum(int(x[3]) for x in members)/1024;cpu=sum(seconds(x[4]) for x in members)
            elapsed=time.monotonic()-start
            if rss>memory_mb:reason='memory_limit';break
            if elapsed>wall_seconds:reason='wall_limit';break
            if cpu>elapsed*cpus+0.1:
                os.killpg(proc.pid,signal.SIGSTOP)
                try:time.sleep(min(0.25,(cpu-elapsed*cpus)/cpus))
                finally:os.killpg(proc.pid,signal.SIGCONT)
            time.sleep(0.1)
        if reason:print('resource_guard_stopped',reason,flush=True)
        else:return proc.returncode
    finally:
        # Include descendants left running after a parent exits.
        try:os.killpg(proc.pid,signal.SIGCONT);os.killpg(proc.pid,signal.SIGTERM)
        except ProcessLookupError:pass
        try:proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            os.killpg(proc.pid,signal.SIGKILL);proc.wait()
    return 124

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--memory-mb',type=int,default=1024);p.add_argument('--cpus',type=float,default=1);p.add_argument('--wall-seconds',type=int,default=3600);p.add_argument('command',nargs=argparse.REMAINDER);a=p.parse_args()
    if a.memory_mb<=0 or a.cpus<=0 or not a.command:p.error('positive limits and command required')
    cmd=a.command[1:] if a.command[0]=='--' else a.command
    try:code=run(cmd,a.memory_mb,a.cpus,a.wall_seconds)
    except KeyboardInterrupt:code=130
    raise SystemExit(code)
