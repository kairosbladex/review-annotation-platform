"""Private HTTP worker running at most 20 official Codex CLI image requests.
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

LIMIT = 20
SLOTS = threading.BoundedSemaphore(LIMIT)
PROCESSES = set()
LOCK = threading.Lock()
MODEL = 'gpt-5.6-luna'
PROMPT_SUFFIX = '不运行任何工具，只返回schema中的JSON；target_index必须使用下面原编号，解释使用中文。候选：'
PROMPT_ALIASES = {'人': '人员', '小客车': '车', '货车': '卡车', '集装箱货车': '卡车',
                  '有斗货车': '卡车', '带蓬布的卡车': '卡车', '大客车': '公交车',
                  '路面裂纹': '道路裂纹', '路面病害': '道路裂纹'}
PROMPT_CLASSES = {'车', '人员', '违建', '光斑', '道路裂纹', '光伏板', '两轮车', '三轮车',
                  '卡车', '公交车', '没戴头盔', '车牌', '其他'}


def _class_key(name):
    mapped = PROMPT_ALIASES.get(name, name)
    return mapped if mapped in PROMPT_CLASSES else '其他'


def _validate_prompt_document(version, doc):
    common, cards = doc.get('common'), doc.get('cards')
    if doc.get('version') != version or not isinstance(common, str) or not common.strip() or len(common) > 2500:
        raise ValueError('invalid prompt common/version')
    if not isinstance(cards, dict) or any(k not in PROMPT_CLASSES or not isinstance(v, str) or not v.strip() or len(v) > 700 for k, v in cards.items()):
        raise ValueError('invalid prompt cards')
    blob = common + ''.join(cards.values())
    if len(blob) > 7000 or any(w in blob for w in ('target_index', 'schema', '运行工具', '一律输出true', '必须输出true', '一律判真')):
        raise ValueError('invalid prompt content')
    return doc


def load_prompt(version):
    # 候选只能由服务配置选择，请求不可传入提示词或任意文件路径。
    if version == 'p000-building-only':
        # 旧 common 与 p002 违建卡分开取，避免带上 p002 改写过的 common。
        base = load_prompt('p000')
        source = load_prompt('p002')
        building = source['cards'].get('违建') if isinstance(source.get('cards'), dict) else None
        doc = {'version': version, 'common': base['common'], 'cards': {'违建': building}}
        return _validate_prompt_document(version, doc)
    if version == 'p000-pure-class-cards':
        # 旧 common；只在整图候选全是同一类别时插入该类别的卡：违建取 p002 卡，道路裂纹取 10-10 原帧金标验证的卡。
        base = load_prompt('p000')
        building = load_prompt('p000-building-only')['cards']['违建']
        crack = json.loads((Path(__file__).with_name('luna_prompts') / 'card-road-crack-20261010.json').read_text(encoding='utf-8'))['card']
        doc = {'version': version, 'common': base['common'], 'cards': {'违建': building, '道路裂纹': crack}}
        return _validate_prompt_document(version, doc)
    if version not in {'p000', 'p002'}:
        raise ValueError('unsupported prompt version')
    path = Path(__file__).with_name('luna_prompts') / f'{version}.json'
    return _validate_prompt_document(version, json.loads(path.read_text(encoding='utf-8')))


def build_prompt(doc, candidates):
    names = [_class_key(candidate['class_name']) for candidate in candidates]
    if doc.get('version') == 'p000-building-only':
        # 只有全部候选都是违建才插入卡。空候选或混有其他类别时不加任何卡，结果与 p000 逐字一致。
        keys = ['违建'] if names and all(name == '违建' for name in names) else []
    elif doc.get('version') == 'p000-pure-class-cards':
        # 整图候选全是同一类别且该类别有卡时才插入；混合图、其他类别与 p000 逐字一致。
        keys = [names[0]] if names and len(set(names)) == 1 and names[0] in doc['cards'] else []
    else:
        keys = []
        for name in names:
            if name in doc['cards'] and name not in keys:
                keys.append(name)
    text = doc['common']
    if keys:
        text += '\n【类别判断要点】\n' + '\n'.join(f'[{key}] {doc["cards"][key]}' for key in keys) + '\n'
    return text + PROMPT_SUFFIX + json.dumps(candidates, ensure_ascii=False)


# 默认 p000 与 816cc81 逐字一致。p000-building-only 只在纯违建图插入 p002 违建卡；
# p000-pure-class-cards 另在纯道路裂纹图插入道路裂纹卡。
# 整套 p002 未通过发布门槛，不默认启用。请求不能改版本。
PROMPT_DOCUMENT = load_prompt(os.environ.get('CODEX_REVIEW_PROMPT_VERSION', 'p000'))
RESOURCE_LOW = threading.Event()
STOP = threading.Event()
SCHEMA = {'type': 'object', 'properties': {'results': {'type': 'array', 'items': {'type': 'object', 'properties': {
    'target_index': {'type': 'integer'}, 'decision': {'type': 'string', 'enum': ['true', 'false', 'uncertain']},
    'observed_object': {'type': 'string'}, 'reasoning': {'type': 'string'}, 'image_limitation': {'type': 'string'}},
    'required': ['target_index', 'decision', 'observed_object', 'reasoning', 'image_limitation'], 'additionalProperties': False}}},
    'required': ['results'], 'additionalProperties': False}

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
    candidates = payload['candidates']
    if not isinstance(candidates, list) or not 1 <= len(candidates) <= 128:
        raise ValueError('invalid candidates')
    for i, c in enumerate(candidates):
        b = c['bbox']
        if c['target_index'] != i or not isinstance(c['class_name'], str) or not c['class_name'].strip() or len(b) != 4 or not (0 <= b[0] < b[2] <= 1 and 0 <= b[1] < b[3] <= 1):
            raise ValueError('invalid candidate')
    with tempfile.TemporaryDirectory(prefix='codex-luna-', dir=os.environ.get('CODEX_REVIEW_TMP_DIR')) as directory:
        root = Path(directory)
        source = Image.open(io.BytesIO(base64.b64decode(payload['image'], validate=True)))
        if source.width * source.height > 40_000_000: raise ValueError('image too large')
        source = source.convert('RGB');source.save(root/'source.jpg', quality=95)
        locator = source.copy();draw = ImageDraw.Draw(locator)
        for c in candidates:
            b = [c['bbox'][0]*source.width, c['bbox'][1]*source.height, c['bbox'][2]*source.width, c['bbox'][3]*source.height]
            draw.rectangle(b, outline='yellow', width=2);draw.text((b[0], b[1]), str(c['target_index']), fill='yellow')
        locator.save(root/'located.jpg', quality=95)
        sheet = Image.new('RGB', (5*256, ((len(candidates)+4)//5)*280), 'white')
        sd = ImageDraw.Draw(sheet)
        for i, c in enumerate(candidates):
            b = c['bbox'];x1,y1,x2,y2 = b[0]*source.width,b[1]*source.height,b[2]*source.width,b[3]*source.height
            mx,my = max(20,(x2-x1)*0.25),max(20,(y2-y1)*0.25)
            crop = source.crop((max(0,x1-mx),max(0,y1-my),min(source.width,x2+mx),min(source.height,y2+my)))
            crop.thumbnail((256,256));sheet.paste(crop,((i%5)*256,(i//5)*280+24));sd.text(((i%5)*256,(i//5)*280),str(i),fill='black')
        sheet.save(root/'crops.jpg', quality=95)
        schema = root/'schema.json';schema.write_text(json.dumps(SCHEMA))
        binary = os.environ['CODEX_REVIEW_BINARY'];home = Path(os.environ.get('CODEX_REVIEW_HOME', '/home/user/.codex'))
        secret = json.loads((home/'auth.json').read_text())['OPENAI_API_KEY']
        env = dict(os.environ, CODEX_HOME=str(home), CODEX_REVIEW_UPSTREAM_KEY=secret, CUDA_VISIBLE_DEVICES='')
        cmd = [binary, 'exec', '--ignore-user-config', '--ignore-rules', '--ephemeral', '--json', '--skip-git-repo-check', '-C', str(root), '-s', 'read-only', '-m', MODEL,
               '-c', 'model_provider="review"', '-c', 'model_providers.review.name="Review"',
               '-c', 'model_providers.review.base_url="https://sub.butterfly2sea.us"',
               '-c', 'model_providers.review.wire_api="responses"', '-c', 'model_providers.review.env_key="CODEX_REVIEW_UPSTREAM_KEY"',
               '-c', 'model_providers.review.requires_openai_auth=false', '-c', 'model_providers.review.request_max_retries=0', '-c', 'model_providers.review.stream_max_retries=0',
               '-c', 'model_reasoning_effort="none"', '-c', 'web_search="disabled"', '--disable', 'shell_tool', '--disable', 'multi_agent', '--disable', 'fast_mode',
               '--output-schema', str(schema), '-i', str(root/'source.jpg'), '-i', str(root/'located.jpg'), '-i', str(root/'crops.jpg')]
        if payload.get('clean_image'):
            clean = Image.open(io.BytesIO(base64.b64decode(payload['clean_image'], validate=True)))
            if clean.width*clean.height > 40_000_000: raise ValueError('clean image too large')
            clean.convert('RGB').save(root/'clean.jpg', quality=95);cmd += ['-i', str(root/'clean.jpg')]
        cmd += ['-']
        prompt = build_prompt(PROMPT_DOCUMENT, candidates)
        proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, env=env, start_new_session=True)
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
            return dict(result, execution_id=uuid.uuid4().hex, requested_model=MODEL)
        finally:
            with LOCK: PROCESSES.discard(proc)

class Handler(BaseHTTPRequestHandler):
    def log_message(self, *args): pass
    def reply(self, status, body):
        data = json.dumps(body, ensure_ascii=False).encode();self.send_response(status);self.send_header('Content-Type', 'application/json');self.send_header('Content-Length', str(len(data)));self.end_headers();self.wfile.write(data)
    def do_GET(self):
        if self.path == '/health': self.reply(200, {'model': MODEL, 'max_concurrency': LIMIT, 'policy': 'only_true_is_true_v1', 'prompt_version': PROMPT_DOCUMENT['version']})
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
