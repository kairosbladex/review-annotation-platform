"""候选 B：人员类别卡和开发集正反例。视图保持旧 thumbnail。

参考例只从开发集挑选，且不与留出集同一分组。渲染和提示词只出现 P/N 标题，
不写入样本 key、金标原文或历史模型结论。正俯视、斜俯视、骑行者是给模型的外观说明，
不要求每个视角都看见四肢，也不把理由里的把握说法当成阈值。
"""
from __future__ import annotations
import json
from pathlib import Path

from PIL import Image, ImageDraw

import person_view
import promptlib

ROOT = Path(__file__).resolve().parent
CARD = (
    '本卡只用于人员，不能用于判断其他类别。正俯视常见较小的头顶或肩背，躯干短，四肢可以看不见。'
    '斜俯视可见头肩或身体侧面，四肢可以不完整。骑行者是人与两轮车连在一起，人在车上；'
    '不要把整辆车当成人员，也不要因为人和车相连就判成非人。'
    '井盖、设备箱、阴影、杆状物、包块和车顶构件容易混淆。'
    '请写明看见的头、肩、躯干或四肢，以及遮挡、目标过小和更像哪类物体。'
    '结构足以分辨才输出true，明显是其他物体才输出false，不能分辨就输出uncertain。'
    '理由里的把握说法不能单独决定结论。'
    '若附件中有参考图，绿色标题P是人员外观正例，红色标题N是非人员反例，只供比较外观，不是当前目标的答案。'
)
HEADER_COLOR = {'P': (30, 140, 60), 'N': (180, 40, 40)}
POISON_FIELDS = ('human_label', 'gold', 'history', 'history_decision', 'codex_suggestion', 'reasoning')


def document() -> dict:
    base = json.loads((ROOT / 'prompts' / 'p000.json').read_text(encoding='utf-8'))
    return {'version': 'p000-person-card', 'common': base['common'], 'cards': {'人员': CARD}}


def validate_card() -> list[str]:
    errors = promptlib.validate(document())
    for banned in ('黑点', '白块', '屋顶', '必须判', '一律'):
        if banned in CARD:
            errors.append(f'人员卡包含禁用规则措辞：{banned}')
    for required in ('正俯视', '斜俯视', '骑行者', '四肢可以看不见', 'uncertain'):
        if required not in CARD:
            errors.append(f'人员卡缺少{required}')
    return errors


def build_prompt(candidates: list[dict]) -> str:
    errors = validate_card()
    if errors:
        raise ValueError('；'.join(errors))
    return promptlib.build(document(), candidates)


def select_references(items, labels, group_of_unit=None, *, k_pos=2, k_neg=2):
    """按分组和 key 的固定顺序取开发集正反例。不够就不补验证集或留出集。"""
    group_of_unit = group_of_unit or {}

    def group(unit):
        return group_of_unit.get(unit) or f'event:{unit}'

    rows, blocked = [], set()
    for item in items:
        name = group(item['unit'])
        for candidate in item['candidates']:
            label = labels.get(candidate['key'])
            if not label or label.get('class_name') not in {'人员', '人'} or label.get('label') not in {'true', 'false'}:
                continue
            split = item.get('split') or label.get('split')
            rows.append({'key': candidate['key'], 'unit': item['unit'], 'group': name, 'split': split,
                         'label': label['label'], 'bbox': list(candidate['bbox'])})
            if split == 'holdout':
                blocked.add(name)
    dev = [row for row in rows if row['split'] == 'dev' and row['group'] not in blocked]
    dev.sort(key=lambda row: (row['group'], row['key']))
    used, picked = set(), {'true': [], 'false': []}
    for label, count in (('true', k_pos), ('false', k_neg)):
        for row in dev:
            if row['label'] != label or row['group'] in used:
                continue
            used.add(row['group'])
            picked[label].append(row)
            if len(picked[label]) == count:
                break
    if len(picked['true']) < k_pos or len(picked['false']) < k_neg:
        return {'ok': False, 'reason': '开发集可用正反例不足，不能从留出集或验证集补齐',
                'references': [], 'excluded_groups': sorted(used), 'blocked_groups': sorted(blocked)}
    references = []
    for index, row in enumerate(picked['true'], start=1):
        references.append(_reference(row, f'P{index}', 'positive'))
    for index, row in enumerate(picked['false'], start=1):
        references.append(_reference(row, f'N{index}', 'negative'))
    return {'ok': True, 'reason': '', 'references': references,
            'excluded_groups': sorted(used), 'blocked_groups': sorted(blocked)}


def _reference(row, caption, role):
    # 审计保留 key；交给模型的只有 caption。调用方不得把本字典写进提示词。
    return {'key': row['key'], 'unit': row['unit'], 'group': row['group'], 'bbox': row['bbox'],
            'caption': caption, 'role': role}


def render_reference_sheet(examples):
    """每个例子只接收图像、框和 P/N 标题。多余字段直接拒绝，避免把金标画进图里。"""
    if not 1 <= len(examples) <= 4:
        raise person_view.PersonViewLimit('参考图每张最多 4 个，且不能为空')
    sheet = Image.new('RGB', (len(examples) * person_view.CELL, person_view.ROW), 'white')
    draw = ImageDraw.Draw(sheet)
    visible = []
    for index, example in enumerate(examples):
        unknown = set(example) - {'image', 'bbox', 'caption'}
        if unknown:
            raise person_view.PersonViewLimit('参考图渲染拒绝金标、历史结论和样本编号')
        caption = example['caption']
        if caption not in {f'P{i}' for i in range(1, 5)} | {f'N{i}' for i in range(1, 5)}:
            raise person_view.PersonViewLimit('参考图标题只能是 P/N 编号')
        image = example['image']
        box = person_view.legacy_context_box(example['bbox'], image.width, image.height)
        crop = image.crop(box)
        crop.thumbnail((person_view.CELL, person_view.CELL))
        origin = (index * person_view.CELL, 0)
        draw.rectangle([origin, (origin[0] + person_view.CELL - 1, person_view.HEADER - 1)], fill=HEADER_COLOR[caption[0]])
        sheet.paste(crop, (origin[0], person_view.HEADER))
        draw.text((origin[0] + 4, 4), caption, fill='white')
        visible.append(caption)
    return sheet, visible


def prompt_contains_audit_secrets(prompt: str, references: list[dict], labels: dict) -> list[str]:
    """检查模型能看见的提示词，不检查本地审计字典。"""
    leaked = []
    for reference in references:
        if reference['key'] and reference['key'] in prompt:
            leaked.append(reference['key'])
        label = labels.get(reference['key']) or {}
        for field in POISON_FIELDS:
            value = label.get(field)
            if isinstance(value, str) and len(value) >= 8 and value in prompt:
                leaked.append(value)
    return leaked
