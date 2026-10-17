from __future__ import annotations

import importlib.util
import copy
import json
import shutil
import subprocess
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location('deploy_independent_vision', ROOT / 'scripts/deploy_independent_vision.py')
deployment = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(deployment)


def test_only_explicit_provider_settings_cross_into_the_new_algorithm():
    actual = deployment.filter_provider_environment({
        'LLM_MODEL': 'frozen-model', 'CODEX_REVIEW_WORKER_TOKEN': 'REDACTED',
        'MYSQL_PASSWORD': 'REDACTED', 'MQTT_HOST': 'business-broker',
        'MINIO_BUCKET': 'drone-data', 'MINIO_SECRET_KEY': 'REDACTED',
        'CUDA_VISIBLE_DEVICES': '0,1,2,3,4,5', 'RESULT_DIR': '/shared/results',
        'CLEAN_SOURCE_DEVICE': 'cuda:5', 'UNRECOGNIZED_PROVIDER_KEY': 'unknown',
    })
    assert actual == {'LLM_MODEL': 'frozen-model', 'CODEX_REVIEW_WORKER_TOKEN': 'REDACTED'}


@pytest.mark.parametrize('revision', ['../prod', 'test;docker stop shared', '', 'abcd'])
def test_release_identity_rejects_paths_shell_and_ambiguous_short_ids(revision):
    with pytest.raises(ValueError):
        deployment.validate_release_id(revision)


def test_unknown_or_different_release_resources_cannot_be_overwritten():
    resource = {'name': 'review-vision-test-api-1', 'labels': {'com.docker.compose.project': 'review-vision-test'}}
    with pytest.raises(ValueError):
        deployment.validate_existing_resources([resource], 'a' * 40)


def test_protected_original_containers_cannot_be_targets():
    with pytest.raises(ValueError):
        deployment.validate_existing_resources([
            {'name': 'yoloe_service_test', 'labels': {'vision.managed': 'true', 'vision.release': 'a' * 40}},
        ], 'a' * 40)


@pytest.fixture
def resolved_compose(tmp_path):
    if not shutil.which('docker'):
        pytest.skip('Docker Compose CLI is required for offline configuration validation')
    release = tmp_path / ('a' * 40)
    deploy = release / 'source/deploy'
    deploy.mkdir(parents=True)
    shutil.copyfile(ROOT / 'deploy/vision-test.compose.yml', deploy / 'vision-test.compose.yml')
    values = deployment.private_environment(
        {'LLM_MODEL': 'test-model', 'LLM_BASE_URL': 'http://model.internal:11434', 'LLM_USE_OLLAMA': 'true'},
        {'ADMIN_USERNAME': 'admin', 'ADMIN_PASSWORD': 'REDACTED'}, 'a' * 40, '192.0.2.1',
    )
    deployment._write_new(release / '.private.env', ''.join(f"{key}='{value}'\n" for key, value in values.items()))
    raw = deployment._compose(release, 'config', '--format', 'json')
    return release, json.loads(raw)


def test_real_compose_cli_resolves_exact_project_gpu_and_mounts_without_daemon(resolved_compose):
    release, config = resolved_compose
    deployment.validate_config(config, release, 'a' * 40)
    assert config['services']['algorithm']['environment']['MYSQL_ENABLED'] == 'false'
    assert config['services']['algorithm']['environment']['MQTT_HOST'] == ''
    assert set(config['services']) == {'postgres', 'redis', 'minio', 'algorithm', 'api', 'worker', 'frontend'}


@pytest.mark.parametrize('mutation', ['fixture', 'business_database', 'foreign_volume'])
def test_composed_config_rejects_fixture_shared_database_and_foreign_volume(resolved_compose, mutation):
    release, original = resolved_compose
    config = copy.deepcopy(original)
    if mutation == 'fixture':
        config['services']['api']['environment']['MODEL_SERVICE_BASE_URL'] = 'http://model-fixture:8080'
    elif mutation == 'business_database':
        config['services']['api']['environment']['DATABASE_URL'] = 'postgresql+psycopg://REDACTED:REDACTED@lingxi-business:5432/ktd'
    else:
        config['services']['postgres']['volumes'][0]['source'] = 'drone-business-data'
    with pytest.raises(ValueError):
        deployment.validate_config(config, release, 'a' * 40)


def test_openai_compatible_base_url_cannot_be_treated_as_ollama():
    with pytest.raises(ValueError):
        deployment.private_environment(
            {'LLM_MODEL': 'test-model', 'LLM_BASE_URL': 'http://provider.internal/v1', 'LLM_USE_OLLAMA': 'false'},
            {'ADMIN_USERNAME': 'admin', 'ADMIN_PASSWORD': 'REDACTED'}, 'a' * 40, '192.0.2.1',
        )


@pytest.mark.parametrize('mutation', ['writable_weights', 'shared_outputs', 'all_gpus', 'old_port', 'extra_fixture'])
def test_runtime_boundaries_fail_closed_before_start(resolved_compose, mutation):
    release, original = resolved_compose
    config = copy.deepcopy(original)
    algorithm = config['services']['algorithm']
    if mutation == 'writable_weights':
        algorithm['volumes'][0]['read_only'] = False
    elif mutation == 'shared_outputs':
        mount = next(m for m in algorithm['volumes'] if m['target'] == '/app/results')
        mount.update(type='bind', source='/shared/results')
    elif mutation == 'all_gpus':
        devices = algorithm['deploy']['resources']['reservations']['devices']
        devices[0].pop('device_ids')
        devices[0]['count'] = -1
    elif mutation == 'old_port':
        algorithm['ports'][0]['published'] = '8189'
    else:
        config['services']['model-fixture'] = config['services']['api']
    with pytest.raises(ValueError):
        deployment.validate_config(config, release, 'a' * 40)


def test_private_file_is_exclusive_and_never_overwrites_unknown_secrets(tmp_path):
    private = tmp_path / '.private.env'
    deployment._write_new(private, 'original-synthetic-value')
    assert private.stat().st_mode & 0o777 == 0o600
    with pytest.raises(FileExistsError):
        deployment._write_new(private, 'replacement')
    assert private.read_text() == 'original-synthetic-value'


def test_docker_error_does_not_expose_raw_provider_or_environment(monkeypatch):
    monkeypatch.setattr(subprocess, 'run', lambda *args, **kwargs: subprocess.CompletedProcess(
        args[0], 1, stdout='synthetic-provider-secret', stderr='synthetic-database-secret'))
    with pytest.raises(RuntimeError) as error:
        deployment._run(['docker', 'compose', 'config'])
    assert 'synthetic' not in str(error.value)


def test_readiness_evidence_is_separate_from_business_acceptance():
    containers = [
        {'Name': '/yoloe_service_test', 'Id': 'original-id', 'Image': 'original-image',
         'State': {'StartedAt': 'original-start'}},
        {'Name': '/review-vision-test-api-1', 'Id': 'candidate-id', 'Image': 'candidate-image',
         'State': {'StartedAt': 'candidate-start'}},
    ]
    assert deployment.protected_identity(containers) == [
        {'name': 'yoloe_service_test', 'id': 'original-id', 'image': 'original-image', 'started_at': 'original-start'},
    ]


def test_occupied_candidate_ports_cannot_displace_unrelated_services():
    with pytest.raises(ValueError):
        deployment.validate_available_ports({8193, 8189}, set())
    deployment.validate_available_ports({8193, 8189}, {8193})


def test_real_compose_escapes_literal_dollars_for_config_serialization(resolved_compose):
    release, _ = resolved_compose
    private = release / '.private.env'
    private.write_text(private.read_text().replace('synthetic-admin', 'cash$HOME${USER}'))
    config = json.loads(deployment._compose(release, 'config', '--format', 'json'))
    # `config` emits a reusable Compose document, escaping literal dollars.
    # The deployment itself reads the original single-quoted dotenv file.
    assert config['services']['api']['environment']['ADMIN_PASSWORD'] == 'cash$$HOME$${USER}'


def test_prepare_refuses_unknown_private_file_before_any_docker_action(tmp_path, monkeypatch):
    release = tmp_path / ('a' * 40)
    for relative in ['backend/app/main.py', 'migrations/example.py', 'alembic.ini',
                     'frontend/dist/index.html', 'frontend/nginx.conf',
                     'deploy/vision-test.compose.yml', 'deploy/Dockerfile.api-overlay',
                     'deploy/Dockerfile.frontend-artifact', 'deploy/algorithm-privacy/sitecustomize.py']:
        path = release / 'source' / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text('synthetic artifact')
    private = release / '.private.env'
    private.write_text('unknown-existing-credential')
    def reject_docker(*args, **kwargs):
        raise AssertionError('must reject unknown credentials before external actions')
    monkeypatch.setattr(subprocess, 'run', reject_docker)
    with pytest.raises(ValueError):
        deployment.prepare(release, 'a' * 40, '192.0.2.1')
    assert private.read_text() == 'unknown-existing-credential'
