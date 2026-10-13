"""离线实验的人员视图与参考图准备。不调用模型，也不把未绑定图标成干净原帧。"""
from __future__ import annotations
import base64
import io

from PIL import Image, ImageDraw

import input_evidence
import person_view

MODES = {'fixed_view_v1': 'person_view', 'card_references_v1': 'person_card_references'}


def prepare(root, payload):
    mode = payload.get('view_mode')
    if mode not in MODES:
        raise ValueError('invalid view mode')
    candidates = payload['candidates']
    prompt = payload['prompt']
    if not isinstance(prompt, str) or not 0 < len(prompt) <= 20000:
        raise ValueError('invalid prompt')
    source_bytes = base64.b64decode(payload['image'], validate=True)
    source = Image.open(io.BytesIO(source_bytes))
    if source.width * source.height > 40_000_000:
        raise ValueError('image too large')
    source = source.convert('RGB')
    source.save(root / 'source.jpg', quality=95)
    locator = source.copy()
    draw = ImageDraw.Draw(locator)
    for candidate in candidates:
        box = [candidate['bbox'][0] * source.width, candidate['bbox'][1] * source.height,
               candidate['bbox'][2] * source.width, candidate['bbox'][3] * source.height]
        draw.rectangle(box, outline='yellow', width=2)
        draw.text((box[0], box[1]), str(candidate['target_index']), fill='yellow')
    locator.save(root / 'located.jpg', quality=95)
    rendered = person_view.render_sheet(source, candidates, source_kind='screenshot')
    if mode == 'fixed_view_v1':
        rendered.sheet.save(root / 'crops.jpg', quality=95)
        person_changed = rendered.changed
        records = rendered.records
    else:
        sheet, records = person_view.legacy_sheet(source, candidates, source_kind='screenshot')
        sheet.save(root / 'crops.jpg', quality=95)
        person_changed = False
    attachments = [('source', root / 'source.jpg'), ('locator', root / 'located.jpg'), ('crops', root / 'crops.jpg')]
    clean_bytes = None
    if payload.get('clean_image'):
        clean_bytes = base64.b64decode(payload['clean_image'], validate=True)
        clean = Image.open(io.BytesIO(clean_bytes))
        if clean.width * clean.height > 40_000_000:
            raise ValueError('clean image too large')
        clean.convert('RGB').save(root / 'clean.jpg', quality=95)
        attachments.append(('clean_auxiliary', root / 'clean.jpg'))
    if mode == 'card_references_v1':
        raw = payload.get('reference_sheet')
        if not isinstance(raw, str) or not raw:
            raise ValueError('missing reference sheet')
        reference = base64.b64decode(raw, validate=True)
        if len(reference) > 2_000_000:
            raise ValueError('reference sheet too large')
        with Image.open(io.BytesIO(reference)) as image:
            if image.width * image.height > 4_000_000 or image.width > 2048 or image.height > 2048:
                raise ValueError('reference sheet too large')
        (root / 'references.jpg').write_bytes(reference)
        attachments.append(('references', root / 'references.jpg'))
    evidence = {
        'protocol': 'person_view_experiment_v1',
        'factor': MODES[mode],
        'source_sha256': input_evidence.digest(source_bytes),
        'source_kind': 'screenshot',
        'clean_binding': 'unverified' if clean_bytes else 'not_requested',
        'candidates_sha256': input_evidence.json_digest(candidates),
        'prompt_sha256': input_evidence.digest(prompt.encode('utf-8')),
        'person_changed': person_changed,
        'person_records': records,
        'crop_source': 'screenshot',
        'attachments': [{'role': role, 'sha256': input_evidence.file_digest(path), 'byte_length': path.stat().st_size}
                        for role, path in attachments],
    }
    return attachments, evidence
