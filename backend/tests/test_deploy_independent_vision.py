from __future__ import annotations

import importlib.util
import copy
import io
import json
import runpy
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
        'LLM_MODEL': 'frozen-model', 'CODEX_REVIEW_WORKER_TOKEN': 'synthetic-provider-secret',
        'MYSQL_PASSWORD': 'synthetic-business-secret', 'MQTT_HOST': 'business-broker',
        'MINIO_BUCKET': 'drone-data', 'MINIO_SECRET_KEY': 'synthetic-object-secret',
        'CUDA_VISIBLE_DEVICES': '0,1,2,3,4,5', 'RESULT_DIR': '/shared/results',
        'CLEAN_SOURCE_DEVICE': 'cuda:5', 'UNRECOGNIZED_PROVIDER_KEY': 'unknown',
    })
    assert actual == {'LLM_MODEL': 'frozen-model', 'CODEX_REVIEW_WORKER_TOKEN': 'synthetic-provider-secret'}


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
        {'LLM_MODEL': 'qwen3-vl:8b', 'LLM_BASE_URL': 'http://model.internal:11434', 'LLM_USE_OLLAMA': 'true'},
        {'ADMIN_USERNAME': 'admin', 'ADMIN_PASSWORD': 'synthetic-admin'}, 'a' * 40, '192.0.2.1',
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


def test_real_compose_history_uses_instruct_without_changing_algorithm_model(resolved_compose):
    _, config = resolved_compose
    assert config['services']['algorithm']['environment']['LLM_MODEL'] == 'qwen3-vl:8b'
    for service in ('api', 'worker'):
        assert config['services'][service]['environment']['HISTORY_COMPARE_MODEL'] == 'qwen3-vl:8b-instruct'


@pytest.mark.parametrize('service', ['api', 'worker'])
def test_history_model_drift_cannot_reuse_thinking_algorithm_model(resolved_compose, service):
    release, original = resolved_compose
    config = copy.deepcopy(original)
    config['services'][service]['environment']['HISTORY_COMPARE_MODEL'] = 'qwen3-vl:8b'
    with pytest.raises(ValueError):
        deployment.validate_config(config, release, 'a' * 40)


@pytest.mark.parametrize('mutation', ['fixture', 'business_database', 'foreign_volume'])
def test_composed_config_rejects_fixture_shared_database_and_foreign_volume(resolved_compose, mutation):
    release, original = resolved_compose
    config = copy.deepcopy(original)
    if mutation == 'fixture':
        config['services']['api']['environment']['MODEL_SERVICE_BASE_URL'] = 'http://model-fixture:8080'
    elif mutation == 'business_database':
        config['services']['api']['environment']['DATABASE_URL'] = 'postgresql+psycopg://u:p@lingxi-business:5432/ktd'
    else:
        config['services']['postgres']['volumes'][0]['source'] = 'drone-business-data'
    with pytest.raises(ValueError):
        deployment.validate_config(config, release, 'a' * 40)


def test_openai_compatible_base_url_cannot_be_treated_as_ollama():
    with pytest.raises(ValueError):
        deployment.private_environment(
            {'LLM_MODEL': 'test-model', 'LLM_BASE_URL': 'http://provider.internal/v1', 'LLM_USE_OLLAMA': 'false'},
            {'ADMIN_USERNAME': 'admin', 'ADMIN_PASSWORD': 'synthetic-admin'}, 'a' * 40, '192.0.2.1',
        )


@pytest.mark.parametrize('crop_path', [
    '/opt/review-models/resnet18-human-20260907.pt',
    '/app/pretrain/review-crop.pt',
    '/app/pretrain/person/resnet18.pt',
])
def test_private_environment_preserves_approved_crop_model_sources(crop_path):
    values = deployment.private_environment(
        {'LLM_MODEL': 'test-model', 'LLM_BASE_URL': 'http://model.internal:11434',
         'LLM_USE_OLLAMA': 'true', 'REVIEW_CROP_MODEL_PATH': crop_path},
        {'ADMIN_USERNAME': 'admin', 'ADMIN_PASSWORD': 'synthetic-admin'}, 'a' * 40, '192.0.2.1',
    )
    assert values['REVIEW_CROP_MODEL_PATH'] == crop_path


def test_enabled_review_uses_the_independent_worker_and_preserves_authentication():
    values = deployment.private_environment(
        {'LLM_MODEL': 'test-model', 'LLM_BASE_URL': 'http://model.internal:11434',
         'LLM_USE_OLLAMA': 'true', 'CODEX_REVIEW_ENABLED': 'true',
         'CODEX_REVIEW_WORKER_URL': 'http://192.0.2.1:18396',
         'CODEX_REVIEW_WORKER_TOKEN': 'synthetic-worker-token'},
        {'ADMIN_USERNAME': 'admin', 'ADMIN_PASSWORD': 'synthetic-admin'}, 'a' * 40, '192.0.2.1',
    )
    assert values['CODEX_REVIEW_WORKER_URL'] == 'http://192.0.2.1:18397'
    assert values['CODEX_REVIEW_WORKER_TOKEN'] == 'synthetic-worker-token'


def test_composed_config_rejects_the_shared_review_worker(resolved_compose):
    release, config = resolved_compose
    config['services']['algorithm']['environment'].update(
        CODEX_REVIEW_ENABLED='true', CODEX_REVIEW_WORKER_URL='http://192.0.2.1:18396',
        CODEX_REVIEW_WORKER_TOKEN='synthetic-worker-token',
    )
    with pytest.raises(ValueError, match='independent review worker'):
        deployment.validate_config(config, release, 'a' * 40)


@pytest.mark.parametrize('crop_path', [
    '/opt/review-models/unapproved.pt',
    '/opt/other/resnet18-human-20260907.pt',
    '/app/pretrain/../unapproved.pt',
    '/app/pretrain/person/../../unapproved.pt',
    '/opt/review-models/../review-models/resnet18-human-20260907.pt',
])
def test_private_environment_rejects_unapproved_crop_paths_and_traversal(crop_path):
    with pytest.raises(ValueError):
        deployment.private_environment(
            {'LLM_MODEL': 'test-model', 'LLM_BASE_URL': 'http://model.internal:11434',
             'LLM_USE_OLLAMA': 'true', 'REVIEW_CROP_MODEL_PATH': crop_path},
            {'ADMIN_USERNAME': 'admin', 'ADMIN_PASSWORD': 'synthetic-admin'}, 'a' * 40, '192.0.2.1',
        )


@pytest.fixture
def running_release(resolved_compose, monkeypatch):
    release, config = resolved_compose
    containers = []
    for name, service in config['services'].items():
        image = service['image'] if name == 'algorithm' else 'running-' + name
        if name == 'worker':
            image = 'running-api'
        mounts = []
        for mount in service.get('volumes', []):
            mounts.append({'Type': mount['type'], 'Destination': mount['target'],
                           'Source': mount['source'], 'RW': not mount.get('read_only', False),
                           'Name': 'review-vision-test_' + mount['source']})
        containers.append({
            'Name': '/review-vision-test-' + name + '-1', 'Id': 'runtime-' + name, 'Image': image,
            'State': {'Running': True, 'StartedAt': 'candidate-start'},
            'Config': {'Labels': dict(service['labels'], **{'com.docker.compose.service': name}),
                       'Env': [key + '=' + str(value) for key, value in service.get('environment', {}).items()]},
            'Mounts': mounts,
            'HostConfig': {'DeviceRequests': [{'DeviceIDs': ['GPU-9f2ac4d0-be45-fe5b-d16a-86d1f3fd10f7']}]},
        })
    algorithm = next(item for item in containers if item['Id'] == 'runtime-algorithm')
    algorithm['Config']['Env'] = [entry for entry in algorithm['Config']['Env']
                                  if not entry.startswith('REVIEW_CROP_MODEL_PATH=')]
    algorithm['Config']['Env'].append('REVIEW_CROP_MODEL_PATH=/opt/review-models/resnet18-human-20260907.pt')
    manifest = {'revision': 'a' * 40, 'protected': [],
                'images': {'api': 'running-api', 'frontend': 'running-frontend'},
                'infrastructure_images': {name: 'running-' + name for name in ('postgres', 'redis', 'minio')}}
    calls = []
    boundary = {'crop_sha256': '933447331226ab3f05868828864e7737efc62c4c51a226cd12fc2a06a2d4e683',
                'containers': containers}

    def docker_run(arguments, **kwargs):
        calls.append(arguments)
        if arguments == ['docker', 'ps', '-aq']:
            output = '\n'.join(item['Id'] for item in containers)
        elif arguments == ['docker', 'inspect', *(item['Id'] for item in containers)]:
            output = json.dumps(containers)
        elif arguments in (['docker', 'volume', 'ls', '--format', '{{.Name}}'],
                           ['docker', 'network', 'ls', '--format', '{{.Name}}']):
            output = ''
        elif arguments[:4] == ['docker', 'exec', 'runtime-algorithm', 'python3']:
            output = ''
        elif arguments == ['docker', 'exec', 'runtime-algorithm', 'sha256sum',
                           '/opt/review-models/resnet18-human-20260907.pt']:
            output = boundary['crop_sha256'] + '  /opt/review-models/resnet18-human-20260907.pt\n'
        else:
            raise AssertionError('unexpected external command: ' + repr(arguments))
        return subprocess.CompletedProcess(arguments, 0, stdout=output, stderr='')

    class ReadyResponse(io.BytesIO):
        status = 200

    monkeypatch.setattr(subprocess, 'run', docker_run)
    monkeypatch.setattr(deployment.urllib.request, 'urlopen',
                        lambda *args, **kwargs: ReadyResponse(b'{"status":"ok"}'))
    return release, manifest, algorithm, boundary, calls


def test_verify_checks_approved_image_crop_digest_on_the_running_algorithm(running_release):
    release, manifest, _, _, calls = running_release
    result = deployment.verify(release, manifest)
    assert result['status'] == 'engineering_verified'
    assert result['business_acceptance'] == 'pending'
    assert ['docker', 'exec', 'runtime-algorithm', 'sha256sum',
            '/opt/review-models/resnet18-human-20260907.pt'] in calls


def test_verify_rejects_a_changed_image_crop_model(running_release):
    release, manifest, _, boundary, _ = running_release
    boundary['crop_sha256'] = '0' * 64
    with pytest.raises(ValueError, match='crop model'):
        deployment.verify(release, manifest)


@pytest.mark.parametrize('crop_path', ['/opt/review-models/unapproved.pt', '/app/pretrain/../unapproved.pt'])
def test_verify_rejects_runtime_crop_path_drift(running_release, crop_path):
    release, manifest, algorithm, _, _ = running_release
    algorithm['Config']['Env'] = [entry for entry in algorithm['Config']['Env']
                                  if not entry.startswith('REVIEW_CROP_MODEL_PATH=')]
    algorithm['Config']['Env'].append('REVIEW_CROP_MODEL_PATH=' + crop_path)
    with pytest.raises(ValueError, match='crop model'):
        deployment.verify(release, manifest)


def test_verify_rejects_a_running_algorithm_connected_to_the_shared_review_worker(running_release):
    release, manifest, algorithm, _, _ = running_release
    algorithm['Config']['Env'].extend([
        'CODEX_REVIEW_ENABLED=true', 'CODEX_REVIEW_WORKER_URL=http://192.0.2.1:18396',
        'CODEX_REVIEW_WORKER_TOKEN=synthetic-worker-token',
    ])
    with pytest.raises(ValueError, match='independent review worker'):
        deployment.verify(release, manifest)


@pytest.mark.parametrize('image,expected', [
    (deployment.API_BASE_IMAGE, 'engineering_verified'),
    ('unapproved-worker-image', 'reject'),
])
def test_verify_handles_only_the_exact_owned_independent_luna_worker(running_release, image, expected):
    release, manifest, algorithm, boundary, _ = running_release
    algorithm['Config']['Env'].extend([
        'CODEX_REVIEW_ENABLED=true', 'CODEX_REVIEW_WORKER_URL=http://192.0.2.1:18397',
        'CODEX_REVIEW_WORKER_TOKEN=synthetic-worker-token',
    ])
    boundary['containers'].append({
        'Name': '/review-vision-test-luna-worker', 'Id': 'isolated-luna-worker', 'Image': image,
        'State': {'Running': True, 'StartedAt': 'candidate-start'},
        'Config': {'Labels': {'vision.managed': 'true', 'vision.release': 'a' * 40,
                              'vision.component': 'luna-worker'}},
        'HostConfig': {'DeviceRequests': []},
    })
    if expected == 'reject':
        with pytest.raises(ValueError, match='review worker runtime'):
            deployment.verify(release, manifest)
    else:
        assert deployment.verify(release, manifest)['status'] == expected


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


@pytest.fixture
def preparing_release(tmp_path, monkeypatch):
    if not shutil.which('docker'):
        pytest.skip('Docker Compose CLI is required for offline configuration validation')
    release = tmp_path / ('a' * 40)
    for relative in ['backend/app/main.py', 'migrations/example.py', 'alembic.ini',
                     'frontend/dist/index.html', 'frontend/nginx.conf',
                     'deploy/vision-test.compose.yml', 'deploy/Dockerfile.api-overlay',
                     'deploy/Dockerfile.frontend-artifact', 'deploy/algorithm-privacy/sitecustomize.py']:
        path = release / 'source' / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text('synthetic artifact')
    shutil.copyfile(ROOT / 'deploy/vision-test.compose.yml', release / 'source/deploy/vision-test.compose.yml')
    containers = [
        {'Id': 'original-algorithm', 'Name': '/yoloe_service_test', 'Image': deployment.ALGORITHM_IMAGE,
         'State': {'StartedAt': 'original-start'},
         'Config': {'Env': ['LLM_MODEL=qwen3-vl:8b', 'LLM_BASE_URL=http://model.internal:11434',
                            'LLM_USE_OLLAMA=true']}},
        {'Id': 'original-api', 'Name': '/review-annotation-test-api-1', 'Image': deployment.API_BASE_IMAGE,
         'State': {'StartedAt': 'original-start'},
         'Config': {'Env': ['ADMIN_USERNAME=admin', 'ADMIN_PASSWORD=synthetic-admin']}},
    ]
    original_run = subprocess.run
    resources = {'volume': [], 'network': []}

    def docker_run(arguments, **kwargs):
        if arguments[:2] == ['docker', 'compose']:
            return original_run(arguments, **kwargs)
        if arguments == ['docker', 'ps', '-aq']:
            output = '\n'.join(item['Id'] for item in containers)
        elif arguments == ['docker', 'inspect', *(item['Id'] for item in containers)]:
            output = json.dumps(containers)
        elif arguments[1:3] in (['volume', 'ls'], ['network', 'ls']):
            output = '\n'.join(item['Name'] for item in resources[arguments[1]])
        elif arguments[1:3] in (['volume', 'inspect'], ['network', 'inspect']):
            output = json.dumps([item for item in resources[arguments[1]] if item['Name'] in arguments[3:]])
        elif arguments[:3] == ['docker', 'image', 'inspect']:
            output = json.dumps([{'Id': arguments[3]}])
        else:
            raise AssertionError('unexpected external command: ' + repr(arguments))
        return subprocess.CompletedProcess(arguments, 0, stdout=output, stderr='')

    models = ['qwen3-vl:8b']

    def tags_response(url, **kwargs):
        assert url == 'http://model.internal:11434/api/tags'
        return io.BytesIO(json.dumps({'models': [{'name': model} for model in models]}).encode())

    monkeypatch.setattr(subprocess, 'run', docker_run)
    monkeypatch.setattr(deployment.urllib.request, 'urlopen', tags_response)
    return release, models, containers, resources


def test_prepare_rejects_missing_history_instruct_before_writing_private_files(preparing_release):
    release, *_ = preparing_release
    with pytest.raises(ValueError, match='semantic model is unavailable'):
        deployment.prepare(release, 'a' * 40, '192.0.2.1')
    assert not (release / '.private.env').exists()
    assert not (release / 'deployment.json').exists()


def test_prepare_pins_available_history_instruct_and_preserves_algorithm_model(preparing_release):
    release, models, *_ = preparing_release
    models.append('qwen3-vl:8b-instruct')
    manifest = deployment.prepare(release, 'a' * 40, '192.0.2.1')
    assert manifest['status'] == 'prepared'
    config = json.loads(deployment._compose(release, 'config', '--format', 'json'))
    assert config['services']['algorithm']['environment']['LLM_MODEL'] == 'qwen3-vl:8b'
    for service in ('api', 'worker'):
        assert config['services'][service]['environment']['HISTORY_COMPARE_MODEL'] == 'qwen3-vl:8b-instruct'


@pytest.fixture
def upgrading_release(preparing_release, monkeypatch):
    previous, models, containers, resources = preparing_release
    models.append('qwen3-vl:8b-instruct')
    manifest = deployment.prepare(previous, previous.name, '192.0.2.1')
    script = previous / 'source/scripts/deploy_independent_vision.py'
    script.parent.mkdir()
    # A release-local validator accepts that version's history model contract.
    script.write_text((ROOT / 'scripts/deploy_independent_vision.py').read_text().replace(
        "HISTORY_MODEL = 'qwen3-vl:8b-instruct'", "HISTORY_MODEL = 'qwen3-vl:8b'"))
    private = previous / '.private.env'
    private.write_text(private.read_text().replace("VISION_HISTORY_MODEL='qwen3-vl:8b-instruct'",
                                                  "VISION_HISTORY_MODEL='qwen3-vl:8b'").replace(
        "VISION_ADMIN_USERNAME='admin'", "VISION_ADMIN_USERNAME='suxiaonian'").replace(
        "VISION_ADMIN_PASSWORD='synthetic-admin'", "VISION_ADMIN_PASSWORD='previous$pass${VALUE}'"))
    legacy = runpy.run_path(str(script))
    config = legacy['_config'](previous, previous.name)
    manifest['config_sha256'] = legacy['_digest'](config)
    manifest['status'] = 'started'
    (previous / 'deployment.json').write_text(json.dumps(manifest))
    for name, service in config['services'].items():
        containers.append({'Id': 'previous-' + name, 'Name': '/review-vision-test-' + name + '-1',
                           'Image': service['image'], 'State': {'StartedAt': 'previous-start'},
                           'Config': {'Labels': dict(service['labels'], **{'com.docker.compose.service': name})}})
    for kind, section in [('volume', 'volumes'), ('network', 'networks')]:
        resources[kind].extend({'Name': item['name'], 'Labels': item['labels']}
                               for item in config[section].values())
    current = previous.parent / ('b' * 40)
    shutil.copytree(previous / 'source', current / 'source')
    (current / 'source/scripts/deploy_independent_vision.py').write_text(
        (ROOT / 'scripts/deploy_independent_vision.py').read_text())
    monkeypatch.setattr(deployment, 'RELEASE_ROOT', previous.parent)
    return previous, current, containers, resources


def test_explicit_upgrade_preserves_secrets_and_previous_artifacts(upgrading_release):
    previous, current, _, _ = upgrading_release
    original_private = (previous / '.private.env').read_bytes()
    original_manifest = (previous / 'deployment.json').read_bytes()
    original_source = {str(path.relative_to(previous)): path.read_bytes()
                       for path in (previous / 'source').rglob('*') if path.is_file()}
    manifest = deployment.prepare(current, current.name, '192.0.2.1', previous_release_dir=previous)
    old_values = dict(line.split('=', 1) for line in original_private.decode().splitlines())
    new_values = dict(line.split('=', 1) for line in (current / '.private.env').read_text().splitlines())
    for key in old_values.keys() - {'VISION_REVISION', 'VISION_HISTORY_MODEL'}:
        assert new_values[key] == old_values[key]
    assert new_values['VISION_REVISION'] == "'" + current.name + "'"
    assert new_values['VISION_HISTORY_MODEL'] == "'qwen3-vl:8b-instruct'"
    assert (current / '.private.env').stat().st_mode & 0o777 == 0o600
    assert (previous / '.private.env').read_bytes() == original_private
    assert (previous / 'deployment.json').read_bytes() == original_manifest
    assert {str(path.relative_to(previous)): path.read_bytes()
            for path in (previous / 'source').rglob('*') if path.is_file()} == original_source
    assert manifest['previous_release']['revision'] == previous.name
    deployment._guard(manifest)
    assert deployment.load_manifest(current, current.name)['revision'] == current.name


def test_upgrade_retargets_review_without_changing_the_previous_worker(upgrading_release):
    previous, current, _, _ = upgrading_release
    script = previous / 'source/scripts/deploy_independent_vision.py'
    script.write_text(script.read_text().replace('REVIEW_WORKER_PORT = 18397', 'REVIEW_WORKER_PORT = 18396'))
    private = previous / '.private.env'
    private.write_text(private.read_text() + "CODEX_REVIEW_ENABLED='true'\n"
                       "CODEX_REVIEW_WORKER_URL='http://192.0.2.1:18396'\n"
                       "CODEX_REVIEW_WORKER_TOKEN='synthetic-worker-token'\n")
    legacy = runpy.run_path(str(script))
    path = previous / 'deployment.json'
    manifest = json.loads(path.read_text())
    manifest['config_sha256'] = legacy['_digest'](legacy['_config'](previous, previous.name))
    path.write_text(json.dumps(manifest))
    original = private.read_bytes()
    deployment.prepare(current, current.name, '192.0.2.1', previous_release_dir=previous)
    config = json.loads(deployment._compose(current, 'config', '--format', 'json'))
    algorithm = config['services']['algorithm']['environment']
    assert algorithm['CODEX_REVIEW_WORKER_URL'] == 'http://192.0.2.1:18397'
    assert algorithm['CODEX_REVIEW_WORKER_TOKEN'] == 'synthetic-worker-token'
    assert private.read_bytes() == original


@pytest.mark.parametrize('invalid', ['unknown', 'outside', 'symlink', 'wrong_manifest', 'source_drift',
                                   'private_permissions', 'same_release', 'protected_drift', 'invalid_validator'])
def test_upgrade_rejects_unverified_previous_before_writing_private_files(upgrading_release, invalid):
    previous, current, containers, _ = upgrading_release
    if invalid == 'unknown':
        target = previous.parent / ('c' * 40)
    elif invalid == 'outside':
        target = previous.parent.parent / ('c' * 40)
    elif invalid == 'symlink':
        target = previous.parent / ('c' * 40)
        target.symlink_to(previous, target_is_directory=True)
    elif invalid == 'wrong_manifest':
        target = previous
        path = target / 'deployment.json'
        value = json.loads(path.read_text())
        value['revision'] = 'c' * 40
        path.write_text(json.dumps(value))
    elif invalid == 'private_permissions':
        target = previous
        (target / '.private.env').chmod(0o644)
    elif invalid == 'same_release':
        target = current
    elif invalid == 'protected_drift':
        target = previous
        containers[0]['Image'] = 'changed-protected-image'
    elif invalid == 'invalid_validator':
        target = previous
        (target / 'source/scripts/deploy_independent_vision.py').write_text('def invalid(:\n')
    else:
        target = previous
        (target / 'source/frontend/dist/index.html').write_text('tampered artifact')
    with pytest.raises((ValueError, OSError)):
        deployment.prepare(current, current.name, '192.0.2.1', previous_release_dir=target)
    assert not (current / '.private.env').exists()


def test_upgrade_guard_allows_current_and_verified_old_owners_but_rejects_strangers(upgrading_release):
    previous, current, containers, resources = upgrading_release
    manifest = deployment.prepare(current, current.name, '192.0.2.1', previous_release_dir=previous)
    candidate = next(item for item in containers if item['Id'] == 'previous-api')
    candidate['Config']['Labels']['vision.release'] = current.name
    deployment._guard(manifest)
    resources['volume'][0]['Labels']['vision.release'] = 'c' * 40
    with pytest.raises(ValueError):
        deployment._guard(manifest)


def test_upgrade_guard_does_not_accept_new_resources_claiming_previous_owner(upgrading_release):
    previous, current, _, resources = upgrading_release
    manifest = deployment.prepare(current, current.name, '192.0.2.1', previous_release_dir=previous)
    resources['volume'].append({'Name': 'review-vision-test_unverified-data',
                                'Labels': {'vision.managed': 'true', 'vision.release': previous.name}})
    with pytest.raises(ValueError):
        deployment._guard(manifest)


def test_upgrade_still_requires_explicit_previous_release(upgrading_release):
    _, current, _, _ = upgrading_release
    with pytest.raises(ValueError):
        deployment.prepare(current, current.name, '192.0.2.1')


def test_loading_upgrade_rechecks_previous_validator_provenance(upgrading_release):
    previous, current, _, _ = upgrading_release
    deployment.prepare(current, current.name, '192.0.2.1', previous_release_dir=previous)
    script = previous / 'source/scripts/deploy_independent_vision.py'
    script.write_text(script.read_text() + '\n# changed after upgrade preparation\n')
    with pytest.raises(ValueError, match='previous release artifacts drifted'):
        deployment.load_manifest(current, current.name)


def test_verify_rejects_old_running_container_even_when_its_owner_was_verified(running_release):
    release, manifest, algorithm, _, _ = running_release
    algorithm['Config']['Labels']['vision.release'] = 'b' * 40
    manifest['previous_release'] = {'resources': {algorithm['Name'].lstrip('/'): 'b' * 40}}
    with pytest.raises(ValueError, match='running.*release'):
        deployment.verify(release, manifest)
