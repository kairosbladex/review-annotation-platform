#!/usr/bin/env python3
"""记录本次临时实验资源；runner结束或资源异常时仅停止自己的worker。"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import time


def props(unit):
    keys = ['ActiveState', 'MainPID', 'MemoryCurrent', 'MemorySwapCurrent', 'MemoryMax', 'MemorySwapMax', 'CPUQuotaPerSecUSec', 'TasksCurrent']
    text = subprocess.check_output(['systemctl','--user','show',unit, *[f'--property={k}' for k in keys]],text=True,timeout=10)
    return dict(line.split('=',1) for line in text.splitlines() if '=' in line)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--root',type=Path,required=True); a=ap.parse_args()
    root=a.root
    os.sched_setaffinity(0, [max(os.sched_getaffinity(0))])
    worker, runner = 'luna-clean-frame-worker-20261009', 'luna-clean-frame-runner-20261009'
    while True:
        mem={s.split(':')[0]:int(s.split()[1]) for s in Path('/proc/meminfo').read_text().splitlines()}
        r={'at':time.time(),'worker':props(worker),'runner':props(runner),'autotune':props('luna-autotune-runner.service'),
           'available_fraction':mem['MemAvailable']/mem['MemTotal'],'load1':os.getloadavg()[0], 'cpus':os.cpu_count(),
           'completed_files':sum(1 for _ in (root/'evidence/results').glob('*/*.json')),
           'shared':subprocess.check_output(['docker','inspect','--format','{{.State.Status}} {{.State.Health.Status}} {{.RestartCount}}','yoloe_service_test'],text=True,timeout=10).strip()}
        cg=Path('/sys/fs/cgroup/user.slice/user-1000.slice/user@1000.service/app.slice')/(worker+'.service')
        r['effective_worker_limits']={n:(cg/n).read_text().strip() for n in ['memory.max','memory.swap.max','memory.peak','memory.swap.current'] if (cg/n).exists()}
        pid=int(r['worker']['MainPID'])
        r['worker_allowed_cpus']=sorted(os.sched_getaffinity(pid)) if pid else []
        with (root/'evidence/resource-samples.jsonl').open('a') as f:f.write(json.dumps(r)+'\n')
        danger=r['available_fraction']<.15 or r['load1']>1.5*r['cpus'] or not r['shared'].startswith('running healthy ')
        finished=r['runner']['ActiveState'] not in ('active','activating','reloading')
        if danger or finished:
            if danger:subprocess.run(['systemctl','--user','stop',runner],check=True)
            subprocess.run(['systemctl','--user','stop',worker],check=True)
            (root/'evidence/worker-stopped.json').write_text(json.dumps({'at':time.time(),'reason':'resource_guard' if danger else 'runner_finished','runner':props(runner),'worker':props(worker)},indent=1))
            return
        time.sleep(30)


if __name__=='__main__':main()
