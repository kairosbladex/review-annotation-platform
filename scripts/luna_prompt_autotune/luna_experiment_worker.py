"""Luna 提示词实验 worker：保持测试 worker（816cc81）的图像处理与判断方式，提示词随请求传入。
新增实际 CLI 附件、裁剪来源及执行结果证据；不将关联干净图自动视为严格同帧。
只监听 127.0.0.1，供提示词自动迭代 runner 使用，不接入测试环境。
Private HTTP worker running at most 20 official Codex CLI image requests.
Run under a resource-limited systemd scope. Never starts business tasks or GPU models.
"""
from __future__ import annotations
import base64
import hmac
import io
import json
import os
import signal
import subprocess
import tempfile
import threading
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from PIL import Image, ImageDraw

import input_evidence

LIMIT = 20
SLOTS = threading.BoundedSemaphore(LIMIT)
PROCESSES = set()
LOCK = threading.Lock()
MODEL = 'gpt-5.6-luna'
LOADED_SHA256 = input_evidence.file_digest(Path(__file__))
RESOURCE_LOW = threading.Event()
STOP = threading.Event()
SCHEMA = {'type': 'object', 'properties': {'results': {'type': 'array', 'items': {'type': 'object', 'properties': {
    'target_index': {'type': 'integer'}, 'decision': {'type': 'string', 'enum': ['true', 'false', 'uncertain']},
    'observed_object': {'type': 'string'}, 'reasoning': {'type': 'string'}, 'image_limitation': {'type': 'string'}},
    'required': ['target_index', 'decision', 'observed_object', 'reasoning', 'image_limitation'], 'additionalProperties': False}}},
    'required': ['results'], 'additionalProperties': False}

def _model_config(schema_text):
    """与实际写入的输出约束同一份字节。放在证据旁边，不塞进严格校验的 v1 证据。"""
    return {
        'requested_model': MODEL,
        'reasoning_effort': 'none',
        'decision_policy': 'three_state',
        'schema_sha256': input_evidence.digest(schema_text.encode('utf-8')),
    }

def terminate(proc):
    try: os.killpg(proc.pid, signal.SIGTERM)
    except ProcessLookupError: pass
    try: proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        try: os.killpg(proc.pid, signal.SIGKILL)
        except ProcessLookupError: pass
        proc.wait(timeout=5)
    for pipe in [proc.stdin, proc.stdout, proc.stderr]:
        if pipe: pipe.close()

def execute(payload):
    if isinstance(payload, dict) and payload.get('view_mode', 'legacy') != 'legacy':
        return execute_view_experiment(payload)
    candidates = payload['candidates']
    if not isinstance(candidates, list) or not 1 <= len(candidates) <= 128:
        raise ValueError('invalid candidates')
    for i, c in enumerate(candidates):
        b = c['bbox']
        if type(c['target_index']) is not int or c['target_index'] != i or not isinstance(c['class_name'], str) or not c['class_name'].strip() or len(b) != 4 or not (0 <= b[0] < b[2] <= 1 and 0 <= b[1] < b[3] <= 1):
            raise ValueError('invalid candidate')
    with tempfile.TemporaryDirectory(prefix='codex-luna-', dir=os.environ.get('CODEX_REVIEW_TMP_DIR')) as directory:
        root = Path(directory)
        source_bytes = base64.b64decode(payload['image'], validate=True)
        source = Image.open(io.BytesIO(source_bytes))
        if source.width * source.height > 40_000_000: raise ValueError('image too large')
        source = source.convert('RGB');source.save(root/'source.jpg', quality=95)
        source_sha256 = input_evidence.digest(source_bytes)
        locator = source.copy();draw = ImageDraw.Draw(locator)
        for c in candidates:
            b = [c['bbox'][0]*source.width, c['bbox'][1]*source.height, c['bbox'][2]*source.width, c['bbox'][3]*source.height]
            draw.rectangle(b, outline='yellow', width=2);draw.text((b[0], b[1]), str(c['target_index']), fill='yellow')
        locator.save(root/'located.jpg', quality=95)
        sheet = Image.new('RGB', (5*256, ((len(candidates)+4)//5)*280), 'white')
        sd = ImageDraw.Draw(sheet)
        crops = []
        for i, c in enumerate(candidates):
            box = input_evidence.context_box(c['bbox'], source.width, source.height)
            crop = source.crop(box)
            crop.thumbnail((256,256));sheet.paste(crop,((i%5)*256,(i//5)*280+24));sd.text(((i%5)*256,(i//5)*280),str(i),fill='black')
            crops.append({'target_index': c['target_index'], 'bbox': c['bbox'], 'source_sha256': source_sha256,
                          'pixel_box': [round(v) for v in box], 'thumbnail_size': list(crop.size),
                          'paste_xy': [(i % 5) * 256, (i // 5) * 280 + 24]})
        sheet.save(root/'crops.jpg', quality=95)
        schema_text = json.dumps(SCHEMA)
        schema = root/'schema.json';schema.write_text(schema_text)
        binary = os.environ['CODEX_REVIEW_BINARY'];home = Path(os.environ.get('CODEX_REVIEW_HOME', '/home/user/.codex'))
        secret = json.loads((home/'auth.json').read_text())['OPENAI_API_KEY']
        env = dict(os.environ, CODEX_HOME=str(home), CODEX_REVIEW_UPSTREAM_KEY=secret, CUDA_VISIBLE_DEVICES='')
        cmd = [binary, 'exec', '--ignore-user-config', '--ignore-rules', '--ephemeral', '--json', '--skip-git-repo-check', '-C', str(root), '-s', 'read-only', '-m', MODEL,
               '-c', 'model_provider="review"', '-c', 'model_providers.review.name="Review"',
               '-c', 'model_providers.review.base_url="https://sub.butterfly2sea.us"',
               '-c', 'model_providers.review.wire_api="responses"', '-c', 'model_providers.review.env_key="CODEX_REVIEW_UPSTREAM_KEY"',
               '-c', 'model_providers.review.requires_openai_auth=false', '-c', 'model_providers.review.request_max_retries=0', '-c', 'model_providers.review.stream_max_retries=0',
               '-c', 'model_reasoning_effort="none"', '-c', 'web_search="disabled"', '--disable', 'shell_tool', '--disable', 'multi_agent', '--disable', 'fast_mode',
               '--output-schema', str(schema)]
        attachments = [('source', root/'source.jpg'), ('locator', root/'located.jpg'), ('crops', root/'crops.jpg')]
        clean_bytes, clean_size = None, None
        if payload.get('clean_image'):
            clean_bytes = base64.b64decode(payload['clean_image'], validate=True)
            clean = Image.open(io.BytesIO(clean_bytes))
            if clean.width*clean.height > 40_000_000: raise ValueError('clean image too large')
            clean_size = clean.size
            clean.convert('RGB').save(root/'clean.jpg', quality=95);attachments.append(('clean_auxiliary', root/'clean.jpg'))
        for _, path in attachments: cmd += ['-i', str(path)]
        cmd += ['-']
        prompt = payload['prompt']  # 实验用：提示词由 runner 按版本组装后随请求传入
        if not isinstance(prompt, str) or not 0 < len(prompt) <= 20000: raise ValueError('invalid prompt')
        execution_id = uuid.uuid4().hex
        evidence = input_evidence.capture_inputs(source_bytes=source_bytes, source_size=source.size,
            attachments=attachments, candidates=candidates, prompt=prompt, crops=crops, schema_path=schema,
            worker_path=Path(__file__), loaded_worker_sha256=LOADED_SHA256, execution_id=execution_id,
            model=MODEL, clean_bytes=clean_bytes, clean_size=clean_size)
        proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, encoding='utf-8', env=env, start_new_session=True)
        with LOCK: PROCESSES.add(proc)
        try:
            try: stdout, _ = proc.communicate(prompt, timeout=215)
            except subprocess.TimeoutExpired:
                terminate(proc);raise RuntimeError('CLI timed out')
            if proc.returncode: raise RuntimeError('CLI failed')
            messages = []
            for line in stdout.splitlines():
                try: event = json.loads(line)
                except ValueError: continue
                item = event.get('item', {})
                if item.get('type') in {'command_execution', 'web_search', 'mcp_tool_call', 'file_change'}: raise RuntimeError('unexpected tool')
                if event.get('type') == 'item.completed' and item.get('type') == 'agent_message': messages.append(item['text'])
            result = json.loads(messages[-1]);items = result['results']
            if len(items) != len(candidates) or any(type(x.get('target_index')) is not int for x in items) or {x['target_index'] for x in items} != set(range(len(candidates))): raise RuntimeError('target mapping mismatch')
            if any(x.get('decision') not in {'true', 'false', 'uncertain'} or not x.get('reasoning') for x in items): raise RuntimeError('invalid decision')
            if any(input_evidence.file_digest(path) != image['sha256'] for (_, path), image in zip(attachments, evidence['attachments'])):
                raise RuntimeError('input attachment changed during execution')
            evidence = input_evidence.bind_result(evidence, items)
            input_evidence.validate(evidence, request=input_evidence.request_identity(source_sha256, candidates, prompt),
                candidates=candidates, execution_id=execution_id, model=MODEL, decisions=items, allow_unverified_clean=True)
            return dict(result, execution_id=execution_id, requested_model=MODEL, input_evidence=evidence,
                        model_config=_model_config(schema_text))
        finally:
            with LOCK: PROCESSES.discard(proc)

def execute_view_experiment(payload):
    """候选视图或参考图。失败仍是执行失败，不返回 false。裁剪只来自请求截图。"""
    import person_experiment
    candidates = payload['candidates']
    if not isinstance(candidates, list) or not 1 <= len(candidates) <= 128:
        raise ValueError('invalid candidates')
    for i, c in enumerate(candidates):
        b = c['bbox']
        if type(c['target_index']) is not int or c['target_index'] != i or not isinstance(c['class_name'], str) or not c['class_name'].strip() or len(b) != 4 or not (0 <= b[0] < b[2] <= 1 and 0 <= b[1] < b[3] <= 1):
            raise ValueError('invalid candidate')
    with tempfile.TemporaryDirectory(prefix='codex-luna-', dir=os.environ.get('CODEX_REVIEW_TMP_DIR')) as directory:
        root = Path(directory)
        attachments, evidence = person_experiment.prepare(root, payload)
        schema_text = json.dumps(SCHEMA)
        schema = root / 'schema.json'
        schema.write_text(schema_text)
        binary = os.environ['CODEX_REVIEW_BINARY']
        home = Path(os.environ.get('CODEX_REVIEW_HOME', '/home/user/.codex'))
        secret = json.loads((home / 'auth.json').read_text())['OPENAI_API_KEY']
        env = dict(os.environ, CODEX_HOME=str(home), CODEX_REVIEW_UPSTREAM_KEY=secret, CUDA_VISIBLE_DEVICES='')
        cmd = [binary, 'exec', '--ignore-user-config', '--ignore-rules', '--ephemeral', '--json', '--skip-git-repo-check', '-C', str(root), '-s', 'read-only', '-m', MODEL,
               '-c', 'model_provider="review"', '-c', 'model_providers.review.name="Review"',
               '-c', 'model_providers.review.base_url="https://sub.butterfly2sea.us"',
               '-c', 'model_providers.review.wire_api="responses"', '-c', 'model_providers.review.env_key="CODEX_REVIEW_UPSTREAM_KEY"',
               '-c', 'model_providers.review.requires_openai_auth=false', '-c', 'model_providers.review.request_max_retries=0', '-c', 'model_providers.review.stream_max_retries=0',
               '-c', 'model_reasoning_effort="none"', '-c', 'web_search="disabled"', '--disable', 'shell_tool', '--disable', 'multi_agent', '--disable', 'fast_mode',
               '--output-schema', str(schema)]
        for _, path in attachments:
            cmd += ['-i', str(path)]
        cmd += ['-']
        prompt = payload['prompt']
        execution_id = uuid.uuid4().hex
        proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, encoding='utf-8', env=env, start_new_session=True)
        with LOCK:
            PROCESSES.add(proc)
        try:
            try:
                stdout, _ = proc.communicate(prompt, timeout=215)
            except subprocess.TimeoutExpired:
                terminate(proc)
                raise RuntimeError('CLI timed out')
            if proc.returncode:
                raise RuntimeError('CLI failed')
            messages = []
            for line in stdout.splitlines():
                try:
                    event = json.loads(line)
                except ValueError:
                    continue
                item = event.get('item', {})
                if item.get('type') in {'command_execution', 'web_search', 'mcp_tool_call', 'file_change'}:
                    raise RuntimeError('unexpected tool')
                if event.get('type') == 'item.completed' and item.get('type') == 'agent_message':
                    messages.append(item['text'])
            result = json.loads(messages[-1])
            items = result['results']
            if len(items) != len(candidates) or any(type(x.get('target_index')) is not int for x in items) or {x['target_index'] for x in items} != set(range(len(candidates))):
                raise RuntimeError('target mapping mismatch')
            if any(x.get('decision') not in {'true', 'false', 'uncertain'} or not x.get('reasoning') for x in items):
                raise RuntimeError('invalid decision')
            if any(input_evidence.file_digest(path) != image['sha256'] for (_, path), image in zip(attachments, evidence['attachments'])):
                raise RuntimeError('input attachment changed during execution')
            evidence = dict(evidence, execution_id=execution_id, requested_model=MODEL)
            evidence['decisions_sha256'] = input_evidence.decisions_digest(items)
            sealed = dict(evidence, evidence_sha256=input_evidence.json_digest(evidence))
            return dict(result, execution_id=execution_id, requested_model=MODEL, input_evidence=sealed,
                        model_config=_model_config(schema_text))
        finally:
            with LOCK:
                PROCESSES.discard(proc)


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *args): pass
    def reply(self, status, body):
        data = json.dumps(body, ensure_ascii=False).encode();self.send_response(status);self.send_header('Content-Type', 'application/json');self.send_header('Content-Length', str(len(data)));self.end_headers();self.wfile.write(data)
    def do_GET(self):
        if self.path == '/health':
            try: input_evidence.implementation_identity(Path(__file__), LOADED_SHA256)
            except input_evidence.EvidenceError:
                self.reply(503, {'error': 'worker implementation changed; restart required'});return
            self.reply(200, {'model': MODEL, 'max_concurrency': LIMIT, 'policy': 'only_true_is_true_v1',
                             'input_evidence_version': input_evidence.VERSION})
        else: self.reply(404, {'error': 'not found'})
    def do_POST(self):
        expected = 'Bearer ' + os.environ['CODEX_REVIEW_WORKER_TOKEN']
        if not hmac.compare_digest(self.headers.get('Authorization', ''), expected): self.reply(401, {'error': 'unauthorized'});return
        if self.path != '/review': self.reply(404, {'error': 'not found'});return
        if RESOURCE_LOW.is_set(): self.reply(503, {'error': 'resource protection active'});return
        if not SLOTS.acquire(blocking=False): self.reply(429, {'error': 'capacity full'});return
        try:
            self.connection.settimeout(30)
            size = int(self.headers.get('Content-Length', '0'))
            if not 0 < size <= 32_000_000: self.reply(413, {'error': 'invalid body size'});return
            result = execute(json.loads(self.rfile.read(size)));self.reply(200, result)
        except (ValueError, KeyError, TypeError): self.reply(400, {'error': 'invalid review input'})
        except Exception: self.reply(503, {'error': 'review execution incomplete'})
        finally: SLOTS.release()

class Server(ThreadingHTTPServer): request_queue_size = 64

def main():
    if not os.environ.get('CODEX_REVIEW_WORKER_TOKEN'): raise SystemExit('worker token required')
    server = Server((os.environ.get('CODEX_REVIEW_BIND', '127.0.0.1'), int(os.environ.get('CODEX_REVIEW_PORT', '18396'))), Handler)
    def guard():
        while not STOP.wait(5):
            mem = {line.split(':')[0]: int(line.split()[1]) for line in Path('/proc/meminfo').read_text().splitlines()}
            if mem['MemAvailable'] / mem['MemTotal'] < 0.15:
                RESOURCE_LOW.set()
                with LOCK: own = list(PROCESSES)
                for proc in own: terminate(proc)
                return
    threading.Thread(target=guard, daemon=True).start()
    def shutdown(*_):
        with LOCK:
            for proc in list(PROCESSES): terminate(proc)
        raise SystemExit(0)
    signal.signal(signal.SIGTERM, shutdown);signal.signal(signal.SIGINT, shutdown)
    try: server.serve_forever()
    finally: STOP.set();server.server_close()
if __name__ == '__main__': main()
