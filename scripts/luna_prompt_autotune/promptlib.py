"""提示词版本的组装与校验（runner、MCP 工具、测试共用）。

提示词 = common + 本图出现类别的定义卡 + 固定结尾 + 候选 JSON。
固定结尾不可修改，保证输出格式与线上一致；p000 组装结果与线上旧提示词逐字一致。
"""
from __future__ import annotations
import json

SUFFIX = '不运行任何工具，只返回schema中的JSON；target_index必须使用下面原编号，解释使用中文。候选：'
ALIASES = {'人': '人员', '小客车': '车', '货车': '卡车', '集装箱货车': '卡车', '有斗货车': '卡车', '带蓬布的卡车': '卡车',
           '大客车': '公交车', '路面裂纹': '道路裂纹', '路面病害': '道路裂纹'}
KNOWN_CLASSES = ['车', '人员', '违建', '光斑', '道路裂纹', '光伏板', '两轮车', '三轮车', '卡车', '公交车', '没戴头盔', '车牌', '其他']
FORBIDDEN = ['target_index', 'schema', '运行工具', '一律输出true', '必须输出true', '一律判真']
MAX_COMMON, MAX_CARD, MAX_TOTAL = 2500, 700, 7000


def card_key(class_name: str) -> str:
    name = ALIASES.get(class_name, class_name)
    return name if name in KNOWN_CLASSES else '其他'


def build(doc: dict, candidates: list[dict]) -> str:
    cards = doc.get('cards') or {}
    keys = []
    for c in candidates:
        k = card_key(c['class_name'])
        if k in cards and k not in keys: keys.append(k)
    text = doc['common']
    if keys:
        text += '\n【类别判断要点】\n' + '\n'.join(f'[{k}] {cards[k]}' for k in keys) + '\n'
    return text + SUFFIX + json.dumps(candidates, ensure_ascii=False)


def validate(doc: dict) -> list[str]:
    errors = []
    common, cards = doc.get('common'), doc.get('cards') or {}
    if not isinstance(common, str) or not common.strip(): errors.append('common 不能为空')
    elif len(common) > MAX_COMMON: errors.append(f'common 超过 {MAX_COMMON} 字')
    for k, v in cards.items():
        if k not in KNOWN_CLASSES: errors.append(f'未知类别卡：{k}（可用：{"、".join(KNOWN_CLASSES)}）')
        if not isinstance(v, str) or not v.strip(): errors.append(f'{k} 定义卡为空')
        elif len(v) > MAX_CARD: errors.append(f'{k} 定义卡超过 {MAX_CARD} 字')
    total = len(common or '') + sum(len(v) for v in cards.values() if isinstance(v, str))
    if total > MAX_TOTAL: errors.append(f'总长度 {total} 超过 {MAX_TOTAL} 字')
    blob = (common or '') + ''.join(v for v in cards.values() if isinstance(v, str))
    errors += [f'不允许出现“{w}”' for w in FORBIDDEN if w in blob]
    return errors
