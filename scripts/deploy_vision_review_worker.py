"""Deploy one isolated, CPU-only Luna worker; never submit review requests."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import socket
import stat
import subprocess
import sys
import time
import urllib.request

NAME = 'review-vision-test-luna-worker'
RELEASE = 'b597612d5e5703610a6f1bdc0ba5e2426947f263'
IMAGE = 'sha256:941b8e3c54001b53546aa19593b1d77c739de5ce6a04def7240cd278bccc3c00'
HOST, PORT = '192.168.130.238', 18397
RUNTIME = Path('/data/workspace/jzheng/review-vision-test/luna-worker-18397')
SOURCE_UNIT = 'codex-luna-review-worker.service'
SOURCE_SCRIPT = '/data/workspace/jzheng/codex-luna-review/releases/6755b1e-business/scripts/run_codex_luna_worker.py'
SOURCE_BINARY = '/data/workspace/jzheng/codex-luna-review/bin/codex-0.162.0-alpha.17.2/codex'
SOURCE_HOME = '/home/user/.codex'
PROMPT = 'p000-pure-class-cards'
PINS = {
    SOURCE_SCRIPT: 'd9feadd55cd42410f427e5d9b766f67698244c73b16641dac4a0f8ec20d39edf',
    str(Path(SOURCE_SCRIPT).with_name('luna_prompts') / 'p000.json'): '0f265c686c73c751c545d440e3e3e5a1de7cc8206ca86c73e6710412f70e67e9',
    str(Path(SOURCE_SCRIPT).with_name('luna_prompts') / 'p002.json'): '8da46d8cf25ce51c92291e586561e0cedfa7c268dd3d03bb824f22e7366ffa45',
    str(Path(SOURCE_SCRIPT).with_name('luna_prompts') / 'card-road-crack-20261010.json'): '4ca5e02a16b99071eaa87b8846bd2d3039518ea6c71587e241b1a0dcbf31539a',
    SOURCE_BINARY: '73b73b048546993298234a295de01ef2424e7a8b29def9d0ecb1ca7cb77500d4',
}


class DeploymentError(RuntimeError):
    pass


def require(condition, code):
    if not condition:
        raise DeploymentError(code)


def run_silent(arguments, *, timeout=20):
    try:
        result = subprocess.run(arguments, capture_output=True, text=True, timeout=timeout)
    except (OSError, subprocess.TimeoutExpired):
        raise DeploymentError('COMMAND_FAILED') from None
    require(result.returncode == 0, 'COMMAND_FAILED')
    return result.stdout.strip()


def digest(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            result.update(block)
    return result.hexdigest()


def source_state(pid):
    actual = run_silent(['systemctl', '--user', 'show', SOURCE_UNIT, '-p', 'MainPID', '--value'])
    require(actual == str(pid), 'SOURCE_CHANGED')
    proc = Path('/proc') / str(pid)
    require(SOURCE_SCRIPT.encode() in (proc / 'cmdline').read_bytes().split(b'\0'), 'SOURCE_CHANGED')
    raw = (proc / 'environ').read_bytes()
    env = dict(part.decode().split('=', 1) for part in raw.split(b'\0') if b'=' in part)
    archive = Path(env['CODEX_REVIEW_EVIDENCE_DIR'])
    return ({'pid': pid, 'start_ticks': (proc / 'stat').read_text().split(') ', 1)[1].split()[19],
             'archive_mode': stat.S_IMODE(archive.stat().st_mode),
             'environment_sha256': hashlib.sha256(raw).hexdigest()}, env)


def require_source_unchanged(before, after):
    require(before == after, 'SOURCE_CHANGED')


def worker_environment(source):
    require(source.get('CODEX_REVIEW_BINARY') == SOURCE_BINARY
            and source.get('CODEX_REVIEW_HOME') == SOURCE_HOME
            and source.get('CODEX_REVIEW_PROMPT_VERSION') == PROMPT, 'SOURCE_PROVIDER_MISMATCH')
    token = source.get('CODEX_REVIEW_WORKER_TOKEN', '')
    require(isinstance(token, str) and len(token) >= 16 and all(32 <= ord(c) < 127 for c in token), 'INVALID_ENVIRONMENT')
    return {'CODEX_REVIEW_WORKER_TOKEN': token, 'CODEX_REVIEW_HOME': '/runtime/codex-home',
            'CODEX_REVIEW_BINARY': '/opt/frozen-luna/codex', 'CODEX_REVIEW_PROMPT_VERSION': PROMPT,
            'CODEX_REVIEW_BIND': '0.0.0.0', 'CODEX_REVIEW_PORT': str(PORT),
            'CODEX_REVIEW_EVIDENCE_DIR': '/runtime/input-evidence',
            'CODEX_REVIEW_EVIDENCE_MAX_BYTES': str(512 * 1024 * 1024),
            'CODEX_REVIEW_TMP_DIR': '/runtime/tmp', 'CUDA_VISIBLE_DEVICES': '',
            'NVIDIA_VISIBLE_DEVICES': 'void', 'PYTHONDONTWRITEBYTECODE': '1', 'PYTHONUNBUFFERED': '1'}


def private_write(path, contents):
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, 'w', encoding='utf-8') as stream:
        stream.write(contents)


def prepare_runtime(root, environment, entry):
    require(not root.exists() and not root.is_symlink(), 'RUNTIME_EXISTS')
    require(all('\n' not in v and '\r' not in v and '\0' not in v for v in environment.values()), 'INVALID_ENVIRONMENT')
    root.mkdir(mode=0o700, parents=True)
    for name in ('input-evidence', 'tmp', 'logs', 'codex-home'):
        (root / name).mkdir(mode=0o700)
    encoded = ''.join(f'{key}={value}\n' for key, value in environment.items())
    private_write(root / 'runtime.env', encoded)
    private_write(root / 'worker_entry.py', entry)
    private_write(root / 'logs/worker.log', '')


def check_port_free(host, port):
    try:
        with socket.socket() as candidate:
            candidate.bind((host, port))
    except OSError:
        raise DeploymentError('PORT_IN_USE') from None


def mounts(root):
    bindings = {str(root): ('/runtime', False), SOURCE_BINARY: ('/opt/frozen-luna/codex', True),
                SOURCE_HOME + '/auth.json': ('/runtime/codex-home/auth.json', True)}
    for filename in PINS:
        if filename == SOURCE_BINARY:
            continue
        relative = Path(filename).relative_to(Path(SOURCE_SCRIPT).parent)
        bindings[filename] = ('/opt/frozen-luna/scripts/' + str(relative), True)
    return bindings


def launch_command(root, *, name=NAME):
    require(name == NAME, 'PROTECTED_TARGET')
    command = ['docker', 'run', '-d', '--pull=never', '--name', NAME,
               '--label', 'vision.managed=true', '--label', 'vision.release=' + RELEASE,
               '--label', 'vision.component=luna-worker', '--runtime', 'runc', '--user', '1000:1000',
               '--network', 'bridge', '-p', f'{HOST}:{PORT}:{PORT}', '--memory', '2g', '--cpus', '1',
               '--pids-limit', '128', '--read-only', '--cap-drop', 'ALL', '--security-opt', 'no-new-privileges',
               '--tmpfs', '/tmp:rw,nosuid,nodev,size=64m,mode=1777', '--restart', 'no', '--log-driver', 'none',
               '--env-file', str(root / 'runtime.env')]
    for source, (destination, readonly) in mounts(root).items():
        command += ['--mount', f'type=bind,src={source},dst={destination}' + (',readonly' if readonly else '')]
    return command + ['--entrypoint', 'python', IMAGE, '/runtime/worker_entry.py']


def render_entry(script, pins, *, log_path='/runtime/logs/worker.log'):
    return f'''import hashlib, importlib.util, os, pathlib, sys, threading
if sys.version_info < (3,11): raise SystemExit("PYTHON_VERSION_UNSUPPORTED")
pins={pins!r}
for filename, expected in pins.items():
    if hashlib.sha256(pathlib.Path(filename).read_bytes()).hexdigest()!=expected:
        raise SystemExit("FROZEN_INPUT_CHANGED")
log_path={log_path!r}
if log_path:
    sys.stderr=open(log_path,"a",buffering=1)
spec=importlib.util.spec_from_file_location("frozen_luna_worker",{script!r})
worker=importlib.util.module_from_spec(spec)
spec.loader.exec_module(worker)
if worker.MODEL!="gpt-5.6-luna" or worker.PROMPT_DOCUMENT["version"]!={PROMPT!r}:
    raise SystemExit("FROZEN_PROVIDER_CHANGED")
worker.LIMIT=1
worker.SLOTS=threading.BoundedSemaphore(1)
worker.main()
'''


def health(port):
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    with opener.open(f'http://{HOST}:{port}/health', timeout=3) as response:
        return json.load(response)


def inspect_owned(root, expected_environment=None):
    record = json.loads(run_silent(['docker', 'inspect', NAME]))[0]
    config, host = record['Config'], record['HostConfig']
    labels = config.get('Labels') or {}
    require(record['Name'] == '/' + NAME and record['Image'] == IMAGE, 'PROTECTED_TARGET')
    require(all(labels.get(key) == value for key, value in {'vision.managed': 'true', 'vision.release': RELEASE, 'vision.component': 'luna-worker'}.items()), 'PROTECTED_TARGET')
    require(not any(key.startswith('com.docker.compose.') for key in labels), 'UNEXPECTED_COMPOSE_OWNER')
    require(config.get('User') == '1000:1000' and host.get('Runtime') == 'runc'
            and not host.get('DeviceRequests') and not host.get('Devices') and not host.get('Privileged'), 'GPU_OR_PRIVILEGE_DRIFT')
    require(host.get('ReadonlyRootfs') and host.get('Memory') == 2 * 1024 ** 3 and host.get('NanoCpus') == 10 ** 9, 'RESOURCE_DRIFT')
    expected = {destination: (source, not readonly) for source, (destination, readonly) in mounts(root).items()}
    actual = {m['Destination']: (m['Source'], m['RW']) for m in record['Mounts'] if m['Type'] == 'bind'}
    require(actual == expected, 'MOUNT_DRIFT')
    require(host.get('PortBindings', {}).get('18397/tcp') == [{'HostIp': HOST, 'HostPort': '18397'}], 'PORT_DRIFT')
    if expected_environment is not None:
        actual_env = dict(item.split('=', 1) for item in config['Env'])
        require(all(actual_env.get(k) == v for k, v in expected_environment.items()), 'ENVIRONMENT_DRIFT')
        require({k for k in actual_env if k.startswith('CODEX_REVIEW_')} == {k for k in expected_environment if k.startswith('CODEX_REVIEW_')}, 'ENVIRONMENT_DRIFT')
    return record


def start(source_pid):
    require(os.getuid() == 1000 and sys.version_info >= (3, 11), 'RUNTIME_IDENTITY_MISMATCH')
    before, source_env = source_state(source_pid)
    for filename, expected in PINS.items():
        require(digest(filename) == expected, 'FROZEN_INPUT_CHANGED')
    require(run_silent(['docker', 'image', 'inspect', IMAGE, '--format', '{{.Id}}']) == IMAGE, 'IMAGE_MISMATCH')
    existing = run_silent(['docker', 'ps', '-a', '--filter', 'name=^/' + NAME + '$', '--format', '{{.Names}}'])
    require(not existing, 'TARGET_EXISTS')
    check_port_free(HOST, PORT)
    original_health = health(18396)
    require(original_health.get('model') == 'gpt-5.6-luna' and original_health.get('prompt_version') == PROMPT, 'SOURCE_PROVIDER_MISMATCH')
    environment = worker_environment(source_env)
    container_pins = {mounts(RUNTIME)[filename][0]: value for filename, value in PINS.items()}
    entry = render_entry('/opt/frozen-luna/scripts/run_codex_luna_worker.py', container_pins)
    os.umask(0o077)
    prepare_runtime(RUNTIME, environment, entry)
    container_id = run_silent(launch_command(RUNTIME), timeout=60)
    received = None
    for _ in range(20):
        try:
            received = health(PORT)
            break
        except (OSError, ValueError):
            time.sleep(0.25)
    require(received is not None, 'HEALTH_UNAVAILABLE')
    require(received.get('model') == original_health['model'] and received.get('prompt_version') == original_health['prompt_version']
            and received.get('policy') == original_health['policy'] and received.get('max_concurrency') == 1, 'HEALTH_CONTRACT_MISMATCH')
    inspect_owned(RUNTIME, environment)
    after, after_env = source_state(source_pid)
    require_source_unchanged(before, after)
    require(after_env['CODEX_REVIEW_WORKER_TOKEN'] == environment['CODEX_REVIEW_WORKER_TOKEN'], 'SOURCE_CHANGED')
    receipt = {'container': NAME, 'container_id': container_id, 'image': IMAGE, 'release': RELEASE,
               'endpoint': f'http://{HOST}:{PORT}', 'health': received, 'source_pins': PINS,
               'entry_sha256': digest(RUNTIME / 'worker_entry.py'), 'source_process_unchanged': True,
               'original_worker_pid': source_pid, 'original_archive_mode': oct(before['archive_mode']),
               'runtime_root': str(RUNTIME), 'runtime_mode': oct(stat.S_IMODE(RUNTIME.stat().st_mode)),
               'archive_mode': oct(stat.S_IMODE((RUNTIME / 'input-evidence').stat().st_mode)),
               'environment_mode': oct(stat.S_IMODE((RUNTIME / 'runtime.env').stat().st_mode)),
               'gpu_devices': 0, 'inference_requests': 0, 'shared_service_mutations': 0,
               'rollback': f'docker stop --time 10 {NAME}'}
    private_write(RUNTIME / 'receipt.json', json.dumps(receipt, ensure_ascii=False, indent=2) + '\n')
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--start', action='store_true')
    mode.add_argument('--stop', action='store_true')
    parser.add_argument('--source-pid', type=int, default=276692)
    args = parser.parse_args()
    try:
        if args.start:
            result = start(args.source_pid)
        else:
            before, _ = source_state(args.source_pid)
            inspect_owned(RUNTIME)
            run_silent(['docker', 'stop', '--time', '10', NAME])
            after, _ = source_state(args.source_pid)
            require_source_unchanged(before, after)
            result = {'stopped': NAME, 'shared_service_mutations': 0, 'records_preserved': True}
        print(json.dumps(result, ensure_ascii=False))
    except DeploymentError as error:
        print(str(error), file=sys.stderr)
        raise SystemExit(2) from None
    except Exception:
        print('DEPLOYMENT_FAILED', file=sys.stderr)
        raise SystemExit(2) from None


if __name__ == '__main__':
    main()
