"""Luna 实验的实际输入证据：只记录摘要和几何，不保存凭据、绝对路径或图片副本。"""
from __future__ import annotations
import hashlib
import json
import re
from pathlib import Path

from PIL import Image

VERSION = 1
ROLES = ('source', 'locator', 'crops')
SHA = re.compile(r'[0-9a-f]{64}')


class EvidenceError(ValueError):
    pass


def digest(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def json_digest(value) -> str:
    return digest(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':')).encode('utf-8'))


def file_digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''): h.update(chunk)
    return h.hexdigest()


LOADED_SHA256 = file_digest(Path(__file__))


def implementation_identity(worker_path: Path, loaded_worker_sha256: str) -> str:
    if file_digest(worker_path) != loaded_worker_sha256 or file_digest(Path(__file__)) != LOADED_SHA256:
        raise EvidenceError('实现文件在加载后发生变化，需配套重启 worker，不能冒充已加载版本')
    return json_digest({'worker': loaded_worker_sha256, 'evidence': LOADED_SHA256})


def request_candidates(unit: dict) -> list:
    return [{'target_index': i, 'class_name': c['class_name'], 'bbox': c['bbox']} for i, c in enumerate(unit['candidates'])]


def request_identity(image_sha256: str, candidates: list, prompt: str) -> dict:
    return {'source_sha256': image_sha256, 'candidates_sha256': json_digest(candidates),
            'prompt_sha256': digest(prompt.encode('utf-8'))}


def context_box(bbox, width, height):
    """保留原实验 worker 的 25%/至少20像素上下文；Pillow crop 对端点取 round。"""
    x1, y1, x2, y2 = bbox[0] * width, bbox[1] * height, bbox[2] * width, bbox[3] * height
    mx, my = max(20, (x2 - x1) * .25), max(20, (y2 - y1) * .25)
    return max(0, x1 - mx), max(0, y1 - my), min(width, x2 + mx), min(height, y2 + my)


def capture_inputs(*, source_bytes, source_size, attachments, candidates, prompt, crops,
                   schema_path, worker_path, loaded_worker_sha256, execution_id, model, clean_bytes=None, clean_size=None):
    def raw_image(value, size):
        return {'sha256': digest(value), 'byte_length': len(value), 'width': size[0], 'height': size[1]}
    source = raw_image(source_bytes, source_size)
    clean = raw_image(clean_bytes, clean_size) if clean_bytes is not None else None
    images = []
    for index, (role, path) in enumerate(attachments):
        with Image.open(path) as image: width, height = image.size
        images.append({'index': index, 'role': role, 'sha256': file_digest(path), 'byte_length': path.stat().st_size,
                       'width': width, 'height': height, 'source_sha256': (clean if role == 'clean_auxiliary' else source)['sha256']})
    return {'version': VERSION, 'execution_id': execution_id, 'requested_model': model,
            'source': source, 'clean_source': clean, 'clean_binding': 'unverified' if clean else 'not_requested',
            'attachments': images, 'crops': crops, 'candidates_sha256': json_digest(candidates),
            'prompt_sha256': digest(prompt.encode('utf-8')), 'schema_sha256': file_digest(schema_path),
            'worker_sha256': implementation_identity(worker_path, loaded_worker_sha256)}


def decisions_digest(decisions: list) -> str:
    """绑定消费的五个结果字段；按原 target_index 排序，不依赖响应数组顺序。"""
    fields = ('target_index', 'decision', 'reasoning', 'observed_object', 'image_limitation')
    return json_digest(sorted(({k: row[k] for k in fields} for row in decisions), key=lambda row: row['target_index']))


def bind_result(evidence: dict, decisions: list) -> dict:
    result = dict(evidence, decisions_sha256=decisions_digest(decisions))
    return dict(result, evidence_sha256=json_digest(result))


def stored_decisions(rows: list) -> list:
    return [{'target_index': row['target_index'], 'decision': row['raw'], 'reasoning': row['reasoning'],
             'observed_object': row['observed'], 'image_limitation': row['limitation']} for row in rows]


def validate(evidence, *, request, candidates, execution_id, model, decisions, allow_unverified_clean=False):
    """校验请求/附件/结果绑定；摘要防混用，不是远端执行签名或严格同帧证明。"""
    try:
        _validate(evidence, request, candidates, execution_id, model, decisions, allow_unverified_clean)
    except EvidenceError:
        raise
    except (KeyError, TypeError, ValueError, IndexError, AttributeError) as exc:
        raise EvidenceError('输入证据结构不完整或格式非法') from exc


def _validate(e, request, candidates, execution_id, model, decisions, allow_clean):
    def require(condition, reason):
        if not condition: raise EvidenceError(reason)
    require(isinstance(e, dict) and type(e.get('version')) is int and e['version'] == VERSION, '缺少受支持的输入证据，旧 worker 响应不能晋级')
    require(set(e) == {'version', 'execution_id', 'requested_model', 'source', 'clean_source', 'clean_binding',
                       'attachments', 'crops', 'candidates_sha256', 'prompt_sha256', 'schema_sha256', 'worker_sha256',
                       'decisions_sha256', 'evidence_sha256'}, '输入证据字段不符合版本契约')
    require(e.get('evidence_sha256') == json_digest({k: v for k, v in e.items() if k != 'evidence_sha256'}), '输入证据摘要不一致')
    require(isinstance(execution_id, str) and re.fullmatch(r'[0-9a-f]{32}', execution_id) is not None
            and execution_id == e['execution_id'], '执行编号不一致')
    require(isinstance(model, str) and bool(model) and model == e['requested_model'], '请求模型不一致')
    require(e['source']['sha256'] == request['source_sha256'], '原始图片与请求不一致')
    require(e['prompt_sha256'] == request['prompt_sha256'], '实际提示词与请求不一致')
    require(e['candidates_sha256'] == request['candidates_sha256'] == json_digest(candidates), '候选框与请求不一致')
    for name in ('prompt_sha256', 'candidates_sha256', 'schema_sha256', 'worker_sha256', 'decisions_sha256'):
        require(isinstance(e[name], str) and SHA.fullmatch(e[name]) is not None, '输入证据哈希格式非法')
    require(len(decisions) == len(candidates) and all(type(x['target_index']) is int for x in decisions)
            and {x['target_index'] for x in decisions} == {c['target_index'] for c in candidates}, '结果目标编号不一致')
    require(all(x['decision'] in ('true', 'false', 'uncertain') and isinstance(x['reasoning'], str) and x['reasoning']
                and isinstance(x['observed_object'], str) and isinstance(x['image_limitation'], str) for x in decisions), '结果字段非法')
    require(e['decisions_sha256'] == decisions_digest(decisions), '实际结果与执行证据不一致')
    clean = e['clean_source']
    require(clean is None or isinstance(clean, dict) and bool(clean), '干净图来源结构非法')
    require(e['clean_binding'] == ('unverified' if clean else 'not_requested'), '不能将未核验干净图标为同帧通过')
    require(not clean or allow_clean, '干净图缺少严格同帧证据，不能纳入自动选优')
    roles = [*ROLES, 'clean_auxiliary'] if clean else list(ROLES)
    require([x['role'] for x in e['attachments']] == roles, '实际附件缺失或顺序不一致')
    for image in [e['source'], *([clean] if clean else [])]:
        require(set(image) == {'sha256', 'byte_length', 'width', 'height'}, '原始图片证据包含未知字段')
    for image in e['attachments']:
        require(set(image) == {'index', 'role', 'sha256', 'byte_length', 'width', 'height', 'source_sha256'}, '附件证据包含未知字段')
    for image in [e['source'], *([clean] if clean else []), *e['attachments']]:
        require(isinstance(image['sha256'], str) and SHA.fullmatch(image['sha256']) is not None, '图片哈希格式非法')
        require(all(type(image[k]) is int and image[k] > 0 for k in ('width', 'height', 'byte_length')), '图片尺寸或字节数非法')
        require(image['width'] * image['height'] <= 40_000_000, '图片尺寸超出上限')
    for index, image in enumerate(e['attachments']):
        require(image['index'] == index, '实际附件编号不一致')
        source = clean if image['role'] == 'clean_auxiliary' else e['source']
        require(image['source_sha256'] == source['sha256'], '附件来源不一致')
        if image['role'] != 'crops':
            require((image['width'], image['height']) == (source['width'], source['height']), '附件与来源尺寸不一致')
    sheet = e['attachments'][2]
    require((sheet['width'], sheet['height']) == (5 * 256, ((len(candidates) + 4) // 5) * 280), '裁剪拼图尺寸不一致')
    require(len(e['crops']) == len(candidates), '裁剪来源数量不一致')
    for i, (crop, c) in enumerate(zip(e['crops'], candidates)):
        require(set(crop) == {'target_index', 'bbox', 'source_sha256', 'pixel_box', 'thumbnail_size', 'paste_xy'}, '裁剪证据包含未知字段')
        box = [round(v) for v in context_box(c['bbox'], e['source']['width'], e['source']['height'])]
        require(crop['target_index'] == c['target_index'] and crop['bbox'] == c['bbox']
                and crop['source_sha256'] == e['source']['sha256'] and crop['pixel_box'] == box, '裁剪位置或原始来源不一致')
        require(crop['paste_xy'] == [(i % 5) * 256, (i // 5) * 280 + 24]
                and len(crop['thumbnail_size']) == 2 and all(type(v) is int and 0 < v <= 256 for v in crop['thumbnail_size']), '裁剪展示布局非法')


def comparable_identity(evidence: dict) -> dict:
    """提示词/执行编号/结果允许变化；实际图片、模型与输入处理实现不允许悄然变化。"""
    return {k: evidence[k] for k in ('source', 'clean_source', 'clean_binding', 'attachments', 'crops',
                                     'candidates_sha256', 'requested_model', 'schema_sha256', 'worker_sha256')}


def summarize(outputs: list, units: list, prompts: list) -> dict:
    invalid, verified = {}, {}
    for record, unit, prompt in zip(outputs, units, prompts):
        name = unit['unit']
        try:
            if not record or not record.get('success'): raise EvidenceError('执行失败或输出缺失')
            candidates = request_candidates(unit)
            request = record['request_identity']
            if unit.get('sha256') and request['source_sha256'] != unit['sha256']: raise EvidenceError('图片与冻结清单不一致')
            if request['prompt_sha256'] != digest(prompt.encode('utf-8')): raise EvidenceError('提示词版本与请求记录不一致')
            if [x['key'] for x in record['results']] != [c['key'] for c in unit['candidates']]: raise EvidenceError('结果 key 与冻结清单不一致')
            validate(record.get('input_evidence'), request=request, candidates=candidates,
                     execution_id=record.get('execution_id'), model=record.get('requested_model'), decisions=stored_decisions(record['results']))
            verified[name] = comparable_identity(record['input_evidence'])
        except (EvidenceError, KeyError, TypeError) as exc:
            invalid[name] = str(exc) if isinstance(exc, EvidenceError) else '缺少请求或执行证据'
    complete = len(outputs) == len(units) == len(prompts) == len(verified) and not invalid
    return {'version': VERSION, 'expected_units': len(units), 'verified_units': len(verified),
            'complete': complete, 'invalid_units': invalid, 'inputs_sha256': json_digest(verified) if complete else None}
