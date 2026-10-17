"""Scoped, credential-silent deployment of the independent vision test stack.

Run on the deployment host. ``prepare`` writes only release metadata and a
private environment beside source/. ``plan`` is read-only. Container changes
are confined to the explicit ``up`` and ``stop`` operations.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import secrets
import stat
import subprocess
import time
import traceback
import urllib.request
from urllib.parse import urlsplit
from pathlib import Path


PROJECT = 'review-vision-test'
RELEASE_ROOT = Path('/data/workspace/user/review-vision-test/releases')
ALGORITHM_IMAGE = 'sha256:dac104a6a8935905cfa3bbae6ed2d5114e0ed9f3bde7f90e5442361bce220ed1'
API_BASE_IMAGE = 'sha256:941b8e3c54001b53546aa19593b1d77c739de5ce6a04def7240cd278bccc3c00'
FRONTEND_BASE_IMAGE = 'sha256:8fc26b906cd0c0737a497938ef9d5b97d4a4cc5067ab54b69cce02f0613dfa3e'
WEIGHT_SHA256 = 'a9413bf3f15772c223a03bbedd71b79af5822f830e80aa1b51b1a469e65927b1'
IMAGE_CROP_MODEL_PATH = '/opt/review-models/resnet18-human-20260907.pt'
IMAGE_CROP_SHA256 = '933447331226ab3f05868828864e7737efc62c4c51a226cd12fc2a06a2d4e683'
GPU_UUID = 'GPU-9f2ac4d0-be45-fe5b-d16a-86d1f3fd10f7'
SERVICES = {'postgres', 'redis', 'minio', 'algorithm', 'api', 'worker', 'frontend'}
PORTS = {19004, 8193, 8393, 5276}
PROVIDER_KEYS = frozenset({
    'LLM_BASE_URL', 'LLM_MODEL', 'LLM_USE_OLLAMA', 'LLM_TRANSLATION_ENABLED',
    'TRANSLATE_LLM_BASE_URL', 'TRANSLATE_LLM_MODEL', 'TRANSLATE_LLM_TIMEOUT', 'TRANSLATE_LLM_KEEP_ALIVE',
    'CODEX_REVIEW_ENABLED', 'CODEX_REVIEW_WORKER_URL', 'CODEX_REVIEW_WORKER_TOKEN',
    'CLEAN_SOURCE_REVIEW_ENABLED', 'CLEAN_SOURCE_WORKER_ENABLED', 'CLEAN_SOURCE_MODEL_ID',
    'CLEAN_SOURCE_BASE_MODEL_SHA256', 'CLEAN_SOURCE_ADAPTER_SHA256', 'CLEAN_SOURCE_MAX_NEW_TOKENS',
    'CLEAN_SOURCE_MAX_PIXELS', 'CLEAN_SOURCE_MIN_PIXELS', 'CLEAN_SOURCE_MAX_SEQUENCE_LENGTH',
    'CLEAN_SOURCE_MIN_MATCH_SCORE', 'CLEAN_SOURCE_WORKER_TIMEOUT_SECONDS',
    'CLEAN_SOURCE_PERSON_AUTO_CONFIRM_ENABLED', 'CLEAN_SOURCE_RECTANGLE_RECOVERY_ENABLED',
    'CLEAN_SOURCE_VEHICLE_SURFACE_GATE_ENABLED', 'REVIEW_TWO_STATE_COMPAT', 'REVIEW_CROP_MODEL_PATH',
    'BATCH_HLS_USE_FFMPEG_SAMPLE', 'BATCH_HLS_SKIP_SEQUENTIAL_FALLBACK', 'VIDEO_FFMPEG_HWACCEL',
    'BATCH_VIDEO_FRAME_QUEUE_SIZE', 'BATCH_IO_THREAD_POOL_SIZE', 'BATCH_TICKER_INTERVAL_CAP_SECONDS',
    'OPENCV_NUM_THREADS', 'OMP_NUM_THREADS',
})
WEIGHT_BINDS = {
    '/app/pretrain': '/data/workspace/jzheng/yoloe_service_test_prod/pretrain',
    '/root/.cache/ultralytics': '/data/workspace/jzheng/yoloe_service_test_prod/pretrain',
    '/opt/clean-source/base': '/data/workspace/jzheng/yoloe_service/data/ktd-qwen-vl-8b-instruct',
    '/opt/clean-source/adapter': '/data/workspace/jzheng/ktd-clean-source-qwen-20260925-v1/selected-adapter-epoch2',
}
ALGORITHM_VOLUMES = {
    '/app/uploads': 'algorithm-uploads', '/app/results': 'algorithm-results',
    '/app/cache': 'algorithm-cache', '/app/test_result': 'algorithm-test-result',
}


def filter_provider_environment(environment):
    return {key: value for key, value in environment.items() if key in PROVIDER_KEYS}


def validate_release_id(revision):
    if not isinstance(revision, str) or not re.fullmatch('[a-f0-9]{40}', revision):
        raise ValueError('release must be a complete lowercase commit SHA')
    return revision


def validate_existing_resources(resources, revision):
    """Only exact-release owned resources can participate in an operation."""
    for resource in resources:
        name = resource['name']
        labels = resource.get('labels') or {}
        if not (name.startswith(PROJECT + '-') or name.startswith(PROJECT + '_')):
            raise ValueError('resource is outside the independent project')
        if labels.get('vision.managed') != 'true' or labels.get('vision.release') != revision:
            raise ValueError('existing resource ownership does not match this release')


def validate_crop_model_path(crop):
    if crop and crop != IMAGE_CROP_MODEL_PATH and (
        not crop.startswith('/app/pretrain/') or '..' in Path(crop).parts
    ):
        raise ValueError('crop model path is outside approved read-only sources')


def private_environment(algorithm, original_api, revision, public_host):
    validate_release_id(revision)
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9.-]{0,252}', public_host):
        raise ValueError('invalid public host')
    result = filter_provider_environment(algorithm)
    if not original_api.get('ADMIN_USERNAME') or not original_api.get('ADMIN_PASSWORD'):
        raise ValueError('existing test administrator configuration is missing')
    if not result.get('LLM_BASE_URL') or not result.get('LLM_MODEL'):
        raise ValueError('real model provider configuration is missing')
    ollama = urlsplit(result['LLM_BASE_URL'])
    if (result.get('LLM_USE_OLLAMA', '').lower() != 'true' or ollama.scheme not in {'http', 'https'}
            or not ollama.hostname or ollama.hostname in {'localhost', '127.0.0.1', '::1'}
            or ollama.username or ollama.password or ollama.query or ollama.fragment
            or ollama.path not in {'', '/'}):
        raise ValueError('a routable native Ollama endpoint is required for semantic history comparison')
    if result.get('CODEX_REVIEW_ENABLED', '').lower() == 'true' and not all(
        result.get(key) for key in ('CODEX_REVIEW_WORKER_URL', 'CODEX_REVIEW_WORKER_TOKEN')
    ):
        raise ValueError('real review worker configuration is incomplete')
    validate_crop_model_path(result.get('REVIEW_CROP_MODEL_PATH', ''))
    result.update({
        'VISION_REVISION': revision, 'VISION_PUBLIC_HOST': public_host,
        'VISION_ADMIN_USERNAME': original_api['ADMIN_USERNAME'],
        'VISION_ADMIN_PASSWORD': original_api['ADMIN_PASSWORD'],
        'VISION_MINIO_USER': 'vision_test',
        'VISION_HISTORY_OLLAMA_URL': result['LLM_BASE_URL'].rstrip('/'),
    })
    for key in ('VISION_PG_PASSWORD', 'VISION_MINIO_PASSWORD', 'VISION_JWT_SECRET',
                'VISION_SERVICE_API_KEY', 'VISION_ALGORITHM_API_KEY'):
        result[key] = secrets.token_urlsafe(40)
    # Single-quoted dotenv values preserve dollars; reject ambiguous encodings.
    if any(any(char in str(value) for char in "\r\n\0'\\") for value in result.values()):
        raise ValueError('configuration contains unsupported dotenv characters')
    return result


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def _run(arguments, *, timeout=90, environment=None):
    try:
        completed = subprocess.run(arguments, capture_output=True, text=True, timeout=timeout,
                                   env=environment)
    except (OSError, subprocess.TimeoutExpired):
        raise RuntimeError('deployment command unavailable or timed out') from None
    if completed.returncode:
        # Docker diagnostics may interpolate credentials or signed URLs.
        raise RuntimeError('deployment command failed; raw diagnostics withheld')
    return completed.stdout


def _docker_json(*arguments):
    try:
        return json.loads(_run(['docker', *arguments]))
    except json.JSONDecodeError:
        raise RuntimeError('invalid Docker response') from None


def _container_environment(container):
    return dict(entry.split('=', 1) for entry in container['Config'].get('Env') or [] if '=' in entry)


def _all_containers():
    identifiers = _run(['docker', 'ps', '-aq']).split()
    return _docker_json('inspect', *identifiers) if identifiers else []


def protected_identity(containers):
    return sorted([{'name': item['Name'].lstrip('/'), 'id': item['Id'], 'image': item['Image'],
                    'started_at': item['State'].get('StartedAt')}
                   for item in containers if not item['Name'].lstrip('/').startswith(PROJECT + '-')],
                  key=lambda item: item['name'])


def _resources(containers):
    result = [{'name': item['Name'].lstrip('/'), 'labels': item['Config'].get('Labels') or {}}
              for item in containers if item['Name'].lstrip('/').startswith(PROJECT + '-')]
    # Prefix collisions in stopped volumes/networks matter before compose up.
    for kind in ('volume', 'network'):
        names = _run(['docker', kind, 'ls', '--format', '{{.Name}}']).splitlines()
        selected = [name for name in names if name.startswith(PROJECT + '_')]
        for item in _docker_json(kind, 'inspect', *selected) if selected else []:
            result.append({'name': item['Name'], 'labels': item.get('Labels') or {}})
    return result


def validate_config(config, release_dir, revision):
    if config.get('name') != PROJECT or set(config.get('services', {})) != SERVICES:
        raise ValueError('unexpected compose project or services')
    services = config['services']
    for name, service in services.items():
        labels = service.get('labels') or {}
        if labels.get('vision.managed') != 'true' or labels.get('vision.release') != revision:
            raise ValueError('compose service ownership missing')
        if service.get('privileged') or service.get('network_mode') == 'host' or service.get('env_file'):
            raise ValueError('unbounded service permissions or environment file')
        if service.get('external_links') or service.get('devices') or service.get('volumes_from'):
            raise ValueError('unexpected host or container access')
    if services['algorithm']['image'] != ALGORITHM_IMAGE:
        raise ValueError('algorithm image is not pinned')
    for name in ('api', 'worker'):
        if services[name]['image'] != f'{PROJECT}-api:{revision}':
            raise ValueError('unexpected API image target')
        app_env = services[name].get('environment') or {}
        database = urlsplit(app_env.get('DATABASE_URL', ''))
        if database.hostname != 'postgres' or database.path != '/vision_test' or database.username != 'vision_test':
            raise ValueError('application database is not independent')
        if (app_env.get('MODEL_ADAPTER_TYPE') != 'legacy_review_agent'
                or app_env.get('MODEL_SERVICE_BASE_URL') != 'http://algorithm:8193'
                or app_env.get('RETRIEVAL_SERVICE_BASE_URL') != 'http://algorithm:8193'
                or app_env.get('MINIO_ENDPOINT') != 'minio:9000'
                or app_env.get('MINIO_BUCKET') != 'vision-test'):
            raise ValueError('application is not connected to the independent real algorithm and storage')
    if services['frontend']['image'] != f'{PROJECT}-frontend:{revision}':
        raise ValueError('unexpected frontend image target')
    algorithm = services['algorithm']
    env = algorithm.get('environment') or {}
    if env.get('MYSQL_ENABLED') != 'false' or env.get('MQTT_HOST') != '':
        raise ValueError('business writes are enabled')
    if any(key.startswith('MYSQL_') and key != 'MYSQL_ENABLED' or
           key.startswith('MQTT_') and key != 'MQTT_HOST' for key in env):
        raise ValueError('shared business configuration leaked into algorithm')
    if env.get('MINIO_ENDPOINT') != 'minio:9000' or env.get('MINIO_BUCKET') != 'vision-test':
        raise ValueError('algorithm object storage is not independent')
    if env.get('DEVICE') != 'cuda:0' or 'CUDA_VISIBLE_DEVICES' in env:
        raise ValueError('unexpected GPU index mapping')
    devices = algorithm.get('deploy', {}).get('resources', {}).get('reservations', {}).get('devices', [])
    if len(devices) != 1 or devices[0].get('device_ids') != [GPU_UUID] or 'count' in devices[0]:
        raise ValueError('GPU reservation is not a single fixed device')
    expected_binds = dict(WEIGHT_BINDS)
    expected_binds['/opt/vision-privacy/sitecustomize.py'] = str(
        Path(release_dir) / 'source/deploy/algorithm-privacy/sitecustomize.py')
    seen = set()
    for mount in algorithm.get('volumes', []):
        target = mount['target']
        if target in seen:
            raise ValueError('duplicate algorithm mount')
        seen.add(target)
        if target in expected_binds:
            if mount['type'] != 'bind' or mount.get('source') != expected_binds[target] or not mount.get('read_only'):
                raise ValueError('model or privacy mount is not the approved read-only source')
        elif target in ALGORITHM_VOLUMES:
            if mount['type'] != 'volume' or mount.get('source') != ALGORITHM_VOLUMES[target]:
                raise ValueError('algorithm output is not an independent volume')
        else:
            raise ValueError('unexpected algorithm mount')
    if seen != set(expected_binds) | set(ALGORITHM_VOLUMES):
        raise ValueError('required algorithm mount missing')
    if env.get('PYTHONPATH') != '/opt/vision-privacy:/app':
        raise ValueError('algorithm privacy hook is not enabled')
    expected_ports = {'frontend': {'5276'}, 'api': {'8393'}, 'algorithm': {'8193'}, 'minio': {'19004'}}
    for name, service in services.items():
        if {str(port.get('published')) for port in service.get('ports', [])} != expected_ports.get(name, set()):
            raise ValueError('unexpected published ports')
        if name != 'algorithm' and any(mount.get('type') != 'volume' for mount in service.get('volumes', [])):
            raise ValueError('unexpected application host mount')
        expected_mounts = {'postgres': {'/var/lib/postgresql/data': 'postgres-data'},
                           'redis': {'/data': 'redis-data'}, 'minio': {'/data': 'minio-data'}}
        if name != 'algorithm' and {mount['target']: mount['source'] for mount in service.get('volumes', [])} != expected_mounts.get(name, {}):
            raise ValueError('unexpected application data volume')
    for volume in config.get('volumes', {}).values():
        if volume.get('external') or not volume.get('name', '').startswith(PROJECT + '_'):
            raise ValueError('external or foreign data volume')
    for network in config.get('networks', {}).values():
        if network.get('external') or not network.get('name', '').startswith(PROJECT + '_'):
            raise ValueError('external or foreign network')


def _compose(release_dir, *arguments):
    release = Path(release_dir)
    environment = {key: value for key, value in os.environ.items()
                   if not key.startswith(('VISION_', 'COMPOSE_')) and key not in PROVIDER_KEYS}
    return _run(['docker', 'compose', '--project-name', PROJECT,
                 '--env-file', str(release / '.private.env'),
                 '-f', str(release / 'source/deploy/vision-test.compose.yml'), *arguments],
                timeout=600 if arguments and arguments[0] == 'up' else 90, environment=environment)


def _config(release, revision):
    config = json.loads(_compose(release, 'config', '--format', 'json'))
    validate_config(config, release, revision)
    return config


def source_fingerprint(source):
    source = Path(source)
    required = ['backend', 'migrations', 'alembic.ini', 'frontend/dist', 'frontend/nginx.conf',
                'deploy/vision-test.compose.yml', 'deploy/Dockerfile.api-overlay',
                'deploy/Dockerfile.frontend-artifact', 'deploy/algorithm-privacy/sitecustomize.py']
    files = []
    for relative in required:
        path = source / relative
        if not path.exists() or path.is_symlink():
            raise ValueError('required release artifact missing or symlinked')
        children = [] if path.is_file() else list(path.rglob('*'))
        if any(child.is_symlink() for child in children):
            raise ValueError('symlink in release artifact tree')
        files.extend([path] if path.is_file() else [p for p in children if p.is_file()])
    content = {}
    for path in sorted(set(files)):
        if path.is_symlink() or path.name.startswith('.env') or '.private' in path.name:
            raise ValueError('secret or symlink in release artifacts')
        if '__pycache__' not in path.parts and path.suffix != '.pyc':
            content[str(path.relative_to(source))] = hashlib.sha256(path.read_bytes()).hexdigest()
    if not (source / 'frontend/dist/index.html').is_file():
        raise ValueError('built frontend artifact is missing')
    return _digest(content)


def _write_new(path, text):
    with os.fdopen(os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600), 'w') as stream:
        stream.write(text)


def prepare(release, revision, public_host):
    release = Path(release)
    source_hash = source_fingerprint(release / 'source')
    if (release / 'deployment.json').exists():
        manifest = load_manifest(release, revision)
        if manifest['source_sha256'] != source_hash:
            raise ValueError('prepared source changed')
        return manifest
    if (release / '.private.env').exists() or any(p.name != 'source' for p in release.iterdir()):
        raise ValueError('unrecognized existing release files')
    containers = _all_containers()
    validate_existing_resources(_resources(containers), revision)
    by_name = {item['Name'].lstrip('/'): item for item in containers}
    original = by_name.get('yoloe_service_test')
    if not original or original['Image'] != ALGORITHM_IMAGE:
        raise ValueError('original algorithm identity drifted')
    original_api = by_name.get('review-annotation-test-api-1')
    if not original_api or original_api['Image'] != API_BASE_IMAGE:
        raise ValueError('original administrator source unavailable')
    values = private_environment(_container_environment(original), _container_environment(original_api),
                                 revision, public_host)
    try:
        with urllib.request.urlopen(values['VISION_HISTORY_OLLAMA_URL'] + '/api/tags', timeout=5) as response:
            models = json.loads(response.read(1024 * 1024))
        if not any(model.get('name') == values['LLM_MODEL'] or model.get('model') == values['LLM_MODEL']
                   for model in models.get('models', [])):
            raise ValueError('configured semantic model is not available from Ollama')
    except (OSError, ValueError, TypeError, AttributeError):
        raise ValueError('native Ollama service or selected semantic model is unavailable') from None
    for image in (ALGORITHM_IMAGE, API_BASE_IMAGE, FRONTEND_BASE_IMAGE):
        if _docker_json('image', 'inspect', image)[0]['Id'] != image:
            raise ValueError('required pinned image unavailable')
    release.chmod(0o700)
    _write_new(release / '.private.env', ''.join(f"{key}='{value}'\n" for key, value in sorted(values.items())))
    config = _config(release, revision)
    manifest = {'project': PROJECT, 'revision': revision, 'source_sha256': source_hash,
                'config_sha256': _digest(config), 'protected': protected_identity(containers),
                'infrastructure_images': {name: _docker_json('image', 'inspect', config['services'][name]['image'])[0]['Id']
                                          for name in ('postgres', 'redis', 'minio')},
                'public_host': public_host, 'status': 'prepared', 'images': {}}
    _write_new(release / 'deployment.json', json.dumps(manifest, sort_keys=True, indent=2) + '\n')
    return manifest


def load_manifest(release, revision):
    release = Path(release)
    for path in (release / 'deployment.json', release / '.private.env'):
        if path.is_symlink() or stat.S_IMODE(path.stat().st_mode) != 0o600:
            raise ValueError('private deployment file permissions or ownership changed')
        if path.stat().st_uid != os.getuid():
            raise ValueError('deployment file belongs to another account')
    manifest = json.loads((release / 'deployment.json').read_text())
    if manifest.get('project') != PROJECT or manifest.get('revision') != revision:
        raise ValueError('deployment manifest ownership mismatch')
    if source_fingerprint(release / 'source') != manifest['source_sha256']:
        raise ValueError('source artifact drifted')
    if _digest(_config(release, revision)) != manifest['config_sha256']:
        raise ValueError('deployment configuration drifted')
    return manifest


def _guard(manifest):
    containers = _all_containers()
    if protected_identity(containers) != manifest['protected']:
        raise ValueError('protected container identity changed')
    validate_existing_resources(_resources(containers), manifest['revision'])
    return containers


def _update_manifest(release, manifest):
    temporary = Path(release) / 'deployment.next.json'
    _write_new(temporary, json.dumps(manifest, sort_keys=True, indent=2) + '\n')
    temporary.replace(Path(release) / 'deployment.json')


def validate_available_ports(occupied, owned):
    if (set(occupied) & PORTS) - set(owned):
        raise ValueError('candidate port belongs to an existing process')


def _host_preflight(release, manifest, containers):
    config = _config(release, manifest['revision'])
    for name, expected in manifest['infrastructure_images'].items():
        if _docker_json('image', 'inspect', config['services'][name]['image'])[0]['Id'] != expected:
            raise ValueError('infrastructure image tag drifted after preparation')
    occupied = set()
    for filename in ('/proc/net/tcp', '/proc/net/tcp6'):
        for line in Path(filename).read_text().splitlines()[1:]:
            parts = line.split()
            if parts[3] == '0A':
                occupied.add(int(parts[1].rsplit(':', 1)[1], 16))
    owned_ports = set()
    existing_algorithm = False
    for item in containers:
        if item['Name'].lstrip('/').startswith(PROJECT + '-'):
            for bindings in item['HostConfig'].get('PortBindings', {}).values():
                owned_ports.update(int(binding['HostPort']) for binding in bindings or [])
            if item['Config'].get('Labels', {}).get('com.docker.compose.service') == 'algorithm':
                existing_algorithm = item['State']['Running']
    validate_available_ports(occupied, owned_ports)
    for source in WEIGHT_BINDS.values():
        if not Path(source).is_dir():
            raise ValueError('approved read-only model directory unavailable')
    weight = Path(WEIGHT_BINDS['/app/pretrain']) / 'yoloe-26l-seg.pt'
    digest = hashlib.sha256()
    with weight.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    if digest.hexdigest() != WEIGHT_SHA256:
        raise ValueError('frozen algorithm weight changed')
    available = _run(['nvidia-smi', '--id=' + GPU_UUID, '--query-gpu=memory.free',
                      '--format=csv,noheader,nounits']).strip()
    if not existing_algorithm and int(available) < 24000:
        raise ValueError('selected GPU has insufficient free memory for isolated startup')


def up(release, manifest):
    containers = _guard(manifest)
    _host_preflight(release, manifest, containers)
    source = Path(release) / 'source'
    # The legacy builder resolves local immutable image IDs without registry pulls.
    environment = dict(os.environ, DOCKER_BUILDKIT='0')
    for name, dockerfile in [('api', 'Dockerfile.api-overlay'), ('frontend', 'Dockerfile.frontend-artifact')]:
        base_tag = 'review-annotation-test-api:4e3e5c7' if name == 'api' else 'review-annotation-test-frontend:4e3e5c7'
        base_id = API_BASE_IMAGE if name == 'api' else FRONTEND_BASE_IMAGE
        if _docker_json('image', 'inspect', base_tag)[0]['Id'] != base_id:
            raise ValueError('named overlay base image identity drifted')
        image = f'{PROJECT}-{name}:{manifest["revision"]}'
        existing = _run(['docker', 'image', 'ls', '-q', image]).strip()
        if existing:
            config = _docker_json('image', 'inspect', image)[0]['Config']
            labels = config.get('Labels') or {}
            if labels.get('vision.release') != manifest['revision'] or labels.get('vision.source') != manifest['source_sha256']:
                raise ValueError('existing image tag is not owned by this release')
        _run(['docker', 'build', '--network', 'none', '--pull=false',
              '--label', 'vision.managed=true', '--label', 'vision.release=' + manifest['revision'],
              '--label', 'vision.source=' + manifest['source_sha256'],
              '-f', str(source / 'deploy' / dockerfile), '-t', image, str(source)],
             timeout=600, environment=environment)
        if _docker_json('image', 'inspect', base_tag)[0]['Id'] != base_id:
            raise ValueError('named overlay base changed during build')
        manifest['images'][name] = _docker_json('image', 'inspect', image)[0]['Id']
    _update_manifest(release, manifest)
    _host_preflight(release, manifest, _guard(manifest))
    _compose(release, 'up', '-d', '--no-build', '--pull', 'never')
    manifest['status'] = 'started'
    _update_manifest(release, manifest)


def verify(release, manifest):
    containers = _guard(manifest)
    candidates = [item for item in containers if item['Name'].lstrip('/').startswith(PROJECT + '-')]
    by_service = {item['Config'].get('Labels', {}).get('com.docker.compose.service'): item for item in candidates}
    if set(by_service) != SERVICES or len(candidates) != len(SERVICES):
        raise ValueError('candidate service set is incomplete')
    for service, item in by_service.items():
        if not item['State']['Running']:
            raise ValueError('candidate service is not running')
        if service == 'algorithm' and item['Image'] != ALGORITHM_IMAGE:
            raise ValueError('running algorithm image mismatch')
        if service in ('api', 'worker', 'frontend') and item['Image'] != manifest['images'].get('api' if service == 'worker' else service):
            raise ValueError('running application image mismatch')
        if service in manifest['infrastructure_images'] and item['Image'] != manifest['infrastructure_images'][service]:
            raise ValueError('running infrastructure image mismatch')
    algorithm = by_service['algorithm']
    env = _container_environment(algorithm)
    if env.get('MYSQL_ENABLED') != 'false' or env.get('MQTT_HOST') != '' or env.get('MINIO_BUCKET') != 'vision-test':
        raise ValueError('running algorithm side-effect boundary drifted')
    if any(mount['Type'] == 'bind' and mount['RW'] for mount in algorithm['Mounts']):
        raise ValueError('running algorithm has a writable host mount')
    binds = {mount['Destination']: mount['Source'] for mount in algorithm['Mounts'] if mount['Type'] == 'bind'}
    expected_binds = dict(WEIGHT_BINDS)
    expected_binds['/opt/vision-privacy/sitecustomize.py'] = str(Path(release) / 'source/deploy/algorithm-privacy/sitecustomize.py')
    if binds != expected_binds:
        raise ValueError('running algorithm read-only sources drifted')
    volumes = {mount['Destination']: mount.get('Name') for mount in algorithm['Mounts'] if mount['Type'] == 'volume'}
    if volumes != {target: PROJECT + '_' + name for target, name in ALGORITHM_VOLUMES.items()}:
        raise ValueError('running algorithm output volume ownership drifted')
    devices = algorithm['HostConfig'].get('DeviceRequests') or []
    if len(devices) != 1 or devices[0].get('DeviceIDs') != [GPU_UUID]:
        raise ValueError('running algorithm GPU identity mismatch')
    crop = env.get('REVIEW_CROP_MODEL_PATH', '')
    validate_crop_model_path(crop)
    if crop == IMAGE_CROP_MODEL_PATH:
        checksum = _run(['docker', 'exec', algorithm['Id'], 'sha256sum', IMAGE_CROP_MODEL_PATH]).split()
        if checksum != [IMAGE_CROP_SHA256, IMAGE_CROP_MODEL_PATH]:
            raise ValueError('running image crop model digest mismatch')
    _run(['docker', 'exec', algorithm['Id'], 'python3', '-c',
          "import sitecustomize; assert sitecustomize.__file__ == '/opt/vision-privacy/sitecustomize.py'"])
    for _ in range(30):
        try:
            with urllib.request.urlopen('http://127.0.0.1:8393/api/v1/health', timeout=2) as response:
                if response.status == 200 and json.loads(response.read(4096)).get('status') == 'ok':
                    break
        except (OSError, ValueError):
            time.sleep(1)
    else:
        raise ValueError('candidate API readiness timed out')
    _guard(manifest)
    return {'status': 'engineering_verified', 'business_acceptance': 'pending', 'project': PROJECT,
            'revision': manifest['revision'], 'ports': sorted(PORTS)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('operation', choices=['prepare', 'plan', 'up', 'verify', 'stop'])
    parser.add_argument('--release-dir', type=Path, required=True)
    parser.add_argument('--revision', required=True)
    parser.add_argument('--public-host', default='192.168.130.238')
    args = parser.parse_args()
    try:
        revision = validate_release_id(args.revision)
        release = args.release_dir.resolve()
        if release.parent != RELEASE_ROOT or release.name != revision:
            raise ValueError('release directory is outside the approved deployment root')
        manifest = prepare(release, revision, args.public_host) if args.operation == 'prepare' else load_manifest(release, revision)
        if args.operation == 'plan':
            _host_preflight(release, manifest, _guard(manifest))
        if args.operation == 'up':
            up(release, manifest)
        elif args.operation == 'verify':
            print(json.dumps(verify(release, manifest)))
            return 0
        elif args.operation == 'stop':
            _guard(manifest)
            _compose(release, 'stop')
            _guard(manifest)
        print(json.dumps({'status': args.operation + '_complete', 'project': PROJECT, 'revision': revision,
                          'services': sorted(SERVICES), 'ports': sorted(PORTS), 'secrets_printed': False}))
        return 0
    except (ValueError, RuntimeError, OSError, KeyError, json.JSONDecodeError) as error:
        # Exception repr/tracebacks may include provider or Docker secrets.
        frame = traceback.extract_tb(error.__traceback__)[-1]
        print(json.dumps({'status': 'failed', 'operation': args.operation,
                          'reason': type(error).__name__, 'stage': frame.name, 'line': frame.lineno,
                          'raw_diagnostics': 'withheld'}))
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
