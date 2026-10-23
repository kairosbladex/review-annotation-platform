from __future__ import annotations

import hashlib
import importlib.util
import json
import os
from pathlib import Path
import socket
import stat
import subprocess
import sys

import pytest


SCRIPT = Path(__file__).resolve().parents[2] / 'scripts/deploy_vision_review_worker.py'


def deployment():
    assert SCRIPT.is_file(), 'The isolated worker deployment contract is not implemented'
    spec = importlib.util.spec_from_file_location('vision_review_worker', SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def source_environment(d):
    return {'CODEX_REVIEW_WORKER_TOKEN': 'synthetic-private-worker-token',
            'CODEX_REVIEW_BINARY': d.SOURCE_BINARY, 'CODEX_REVIEW_HOME': '/home/user/.codex',
            'CODEX_REVIEW_PROMPT_VERSION': d.PROMPT,
            'MYSQL_PASSWORD': 'synthetic-business-secret', 'OPENAI_BASE_URL': 'https://unapproved.invalid',
            'CODEX_REVIEW_EVIDENCE_DIR': '/shared/archive'}


def test_only_allowlisted_provider_identity_crosses_into_private_runtime():
    d = deployment()
    env = d.worker_environment(source_environment(d))
    assert env['CODEX_REVIEW_WORKER_TOKEN'] == 'synthetic-private-worker-token'
    assert env['CODEX_REVIEW_HOME'] == '/runtime/codex-home'
    assert env['CODEX_REVIEW_BINARY'] == '/opt/frozen-luna/codex'
    assert env['CODEX_REVIEW_EVIDENCE_DIR'] == '/runtime/input-evidence'
    assert env['CODEX_REVIEW_PORT'] == '18397'
    assert 'MYSQL_PASSWORD' not in env and 'OPENAI_BASE_URL' not in env
    assert '/shared/archive' not in env.values()


def test_private_runtime_uses_private_modes_and_never_repairs_foreign_directory(tmp_path):
    d = deployment()
    root = tmp_path / 'new-worker'
    env = d.worker_environment(source_environment(d))
    d.prepare_runtime(root, env, 'print("synthetic-entry")\n')
    for p in [root, root / 'input-evidence', root / 'tmp', root / 'logs', root / 'codex-home']:
        assert stat.S_IMODE(p.stat().st_mode) == 0o700
    for p in [root / 'runtime.env', root / 'worker_entry.py', root / 'logs/worker.log']:
        assert stat.S_IMODE(p.stat().st_mode) == 0o600
    root.chmod(0o777)
    with pytest.raises(d.DeploymentError, match='RUNTIME_EXISTS'):
        d.prepare_runtime(root, env, 'changed')
    assert stat.S_IMODE(root.stat().st_mode) == 0o777
    assert (root / 'worker_entry.py').read_text() == 'print("synthetic-entry")\n'


def test_secret_line_injection_is_rejected_before_files_exist(tmp_path):
    d = deployment()
    env = source_environment(d)
    env['CODEX_REVIEW_WORKER_TOKEN'] = 'synthetic-secret\nINJECTED=1'
    with pytest.raises(d.DeploymentError, match='INVALID_ENVIRONMENT'):
        d.worker_environment(env)
    assert list(tmp_path.iterdir()) == []


def test_occupied_port_is_rejected_using_a_real_local_socket():
    d = deployment()
    with socket.socket() as listener:
        listener.bind(('127.0.0.1', 0))
        listener.listen()
        with pytest.raises(d.DeploymentError, match='PORT_IN_USE'):
            d.check_port_free('127.0.0.1', listener.getsockname()[1])


def test_launch_plan_is_exactly_owned_has_no_gpu_or_secret_arguments(tmp_path):
    d = deployment()
    argv = d.launch_command(tmp_path)
    assert argv[argv.index('--name') + 1] == 'review-vision-test-luna-worker'
    assert '--pull=never' in argv and '--gpus' not in argv
    assert argv[argv.index('--runtime') + 1] == 'runc'
    assert argv[argv.index('--user') + 1] == '1000:1000'
    assert d.IMAGE in argv and '--network' in argv
    assert 'vision.managed=true' in argv and 'vision.component=luna-worker' in argv
    assert 'vision.release=b597612d5e5703610a6f1bdc0ba5e2426947f263' in argv
    assert all('synthetic-private-worker-token' not in arg for arg in argv)
    with pytest.raises(d.DeploymentError, match='PROTECTED_TARGET'):
        d.launch_command(tmp_path, name='yoloe_service_test')


def test_source_identity_change_is_rejected():
    d = deployment()
    before = {'pid': 42, 'start_ticks': '100', 'archive_mode': 0o777, 'environment_sha256': 'same'}
    d.require_source_unchanged(before, dict(before))
    with pytest.raises(d.DeploymentError, match='SOURCE_CHANGED'):
        d.require_source_unchanged(before, {**before, 'pid': 43})


def test_frozen_entry_runs_one_slot_and_rejects_changed_worker_bytes(tmp_path):
    d = deployment()
    worker = tmp_path / 'worker.py'
    original = 'import json\nMODEL="gpt-5.6-luna"\nPROMPT_DOCUMENT={"version":"p000-pure-class-cards"}\nLIMIT=20\ndef main():\n print(json.dumps({"limit":LIMIT,"first":SLOTS.acquire(False),"second":SLOTS.acquire(False)}))\n'
    worker.write_text(original)
    entry = tmp_path / 'entry.py'
    entry.write_text(d.render_entry(str(worker), {str(worker): hashlib.sha256(worker.read_bytes()).hexdigest()}, log_path=None))
    completed = subprocess.run([sys.executable, str(entry)], capture_output=True, text=True, timeout=10)
    assert completed.returncode == 0, completed.stderr
    assert json.loads(completed.stdout) == {'limit': 1, 'first': True, 'second': False}
    assert worker.read_text() == original
    worker.write_text(original + '\n# changed\n')
    failed = subprocess.run([sys.executable, str(entry)], capture_output=True, text=True, timeout=10)
    assert failed.returncode != 0 and 'FROZEN_INPUT_CHANGED' in failed.stderr
    assert failed.stdout == ''


def test_child_process_failure_does_not_expose_stdout_or_stderr(capsys):
    d = deployment()
    with pytest.raises(d.DeploymentError, match='COMMAND_FAILED') as error:
        d.run_silent([sys.executable, '-c', 'import sys;print("synthetic-private-value");sys.stderr.write("synthetic-private-value");sys.exit(3)'])
    assert 'synthetic-private-value' not in str(error.value)
    captured = capsys.readouterr()
    assert 'synthetic-private-value' not in captured.out + captured.err
