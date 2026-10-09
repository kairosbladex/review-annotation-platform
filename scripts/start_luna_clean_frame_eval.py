#!/usr/bin/env python3
"""仅启动本次实验的独立 worker/runner；复用凭据但不修改共享环境文件。"""
import argparse
import os
from pathlib import Path
import secrets
import socket
import subprocess
import time
import json
import urllib.request


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', type=Path, required=True)
    a = ap.parse_args()
    root = a.root.resolve()
    assert (root/'evidence/prepare-complete.json').exists()
    with socket.socket() as s:
        s.bind(('127.0.0.1', 18398))
    os.umask(0o077)
    srcenv = Path('/data/workspace/jzheng/luna-prompt-autotune/private/worker.env')
    env = dict(line.split('=', 1) for line in srcenv.read_text().splitlines() if '=' in line and not line.startswith('#'))
    token = secrets.token_urlsafe(32)
    (root/'token').write_text(token)
    private = {k: env[k] for k in ('CODEX_REVIEW_BINARY', 'CODEX_REVIEW_HOME')}
    private.update(CODEX_REVIEW_WORKER_TOKEN=token, CODEX_REVIEW_BIND='127.0.0.1', CODEX_REVIEW_PORT='18398', CODEX_REVIEW_TMP_DIR=str(root/'tmp'), CUDA_VISIBLE_DEVICES='', OPENBLAS_NUM_THREADS='1')
    (root/'worker.env').write_text(''.join(f'{k}={v}\n' for k,v in private.items()))
    wrapper = root/'src/worker_entry.py'
    wrapper.write_text('import sys,threading\nfrom pathlib import Path\nsys.path.insert(0,str(Path(__file__).parent/"luna_prompt_autotune"))\nimport luna_experiment_worker as w\nw.LIMIT=4\nw.SLOTS=threading.BoundedSemaphore(4)\nw.main()\n')
    def launch(name, memory, cpu, command, extra=()):
        # 用户 cgroup 未委派 CPU 控制器时 CPUQuota 仅有配置值；affinity 限制实际总 CPU。
        count = 3 if cpu == '300%' else 1
        cpus = ','.join(map(str, sorted(os.sched_getaffinity(0))[-count:]))
        subprocess.run(['systemd-run', '--user', '--unit='+name, '-p', 'MemoryMax='+memory, '-p', 'MemorySwapMax=0', '-p', 'CPUQuota='+cpu,
                        '-p', 'KillMode=control-group', '-p', 'Restart=no', '-p', 'Environment=OPENBLAS_NUM_THREADS=1',
                        '-p', 'StandardOutput=append:'+str(root/(name+'.log')), '-p', 'StandardError=append:'+str(root/(name+'.log')),
                        *extra, '/usr/bin/taskset', '--cpu-list', cpus, *command], check=True)
    launch('luna-clean-frame-worker-20261009', '4G', '300%', ['/usr/bin/python3', str(wrapper)], ['-p', 'EnvironmentFile='+str(root/'worker.env')])
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    for attempt in range(20):
        try:
            with opener.open('http://127.0.0.1:18398/health', timeout=2) as response:
                assert json.load(response)['max_concurrency'] == 4
            break
        except OSError:
            time.sleep(1)
    else:
        raise RuntimeError('实验 worker 未就绪，未启动 runner')
    launch('luna-clean-frame-runner-20261009', '1G', '100%', ['/usr/bin/python3', str(root/'src/luna_clean_frame_eval.py'), 'run', '--root', str(root)])


if __name__ == '__main__':
    main()
