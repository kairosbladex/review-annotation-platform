"""组装 Luna 提示词 v3（只改车辆）新样本验证集。

新样本：gold-export-live-20260903 人工金标，按图片 SHA256 合并同图候选，排除草案设计集（113 事件）
与 v2 验证集（170 张）已用图片。人员回归：复用 v2 验证集中含人员的图片（人员提示词已恢复旧写法）。
输出 data/luna-prompt-v3-eval-20261009/{manifest.json,labels.json,build.json}；金标不外发。
"""
from __future__ import annotations
import collections
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1] / 'data'
OUT = ROOT / 'luna-prompt-v3-eval-20261009'


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def iou(a, b):
    w = max(0, min(a[2], b[2]) - max(a[0], b[0])); h = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    u = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - w * h
    return w * h / u if u else 0


def main():
    design_sha = set(json.loads((ROOT / 'luna-eval-20261008/input-audit.json').read_text())['image_sha256'].values())
    v2_manifest = json.loads((ROOT / 'luna-prompt-v2-eval-20261009/manifest.json').read_text())
    v2_labels = json.loads((ROOT / 'luna-prompt-v2-eval-20261009/labels.json').read_text())
    v2_sha = {u['sha256'] for u in v2_manifest.values()}
    excluded = collections.Counter()
    units, labels = {}, {}

    # 1. 新样本：09-03 人工金标，同一张图的多个样本合并
    src = ROOT / 'gold-export-live-20260903'
    by_sha = collections.defaultdict(list)
    for line in open(src / 'annotations.jsonl'):
        r = json.loads(line)
        if r.get('labeled_by') is None:
            excluded['gold0903:auto_label'] += 1; continue
        p = src / 'images' / (r['sample_id'] + '.jpg')
        by_sha[sha(p)].append((p, r))
    for h, rs in sorted(by_sha.items()):
        if h in design_sha or h in v2_sha:
            excluded['gold0903:image_used_before'] += len(rs); continue
        cands = []
        for p, r in rs:
            b = r['bbox']
            if not (0 <= b[0] < b[2] <= 1 and 0 <= b[1] < b[3] <= 1):
                excluded['gold0903:invalid_box'] += 1; continue
            dup = next((c for c in cands if c['class_name'] == r['class_name'] and iou(c['bbox'], b) >= 0.9), None)
            if dup:
                if labels.get(dup['key'], {}).get('label') != r['human_label']:
                    labels.pop(dup['key'], None); excluded['gold0903:conflicting_duplicate'] += 1
                else:
                    excluded['gold0903:duplicate_box'] += 1
                continue
            key = 'gold0903:' + r['candidate_id']
            cands.append({'key': key, 'class_name': r['class_name'], 'bbox': b})
            labels[key] = {'label': r['human_label'], 'class_name': r['class_name'], 'source': 'gold0903'}
        if cands:
            units['g0903_' + h[:16]] = {'source': 'gold0903', 'image': str(rs[0][0].relative_to(ROOT)), 'sha256': h, 'candidates': cands}

    # 2. 人员回归：v2 验证集中含人员金标的图片（baseline 结果复用 v2 当天共享 worker 输出）
    for uid, u in v2_manifest.items():
        if any(v2_labels.get(c['key'], {}).get('class_name') in ('人员', '人') for c in u['candidates']):
            safe = uid.replace(':', '_')
            units[safe] = dict(u, source='person_regression', reuse_baseline_from='luna-prompt-v2-eval-20261009/out-baseline/' + safe + '.json')
            for c in u['candidates']:
                if c['key'] in v2_labels:
                    labels[c['key']] = dict(v2_labels[c['key']], source='person_regression')

    OUT.mkdir(exist_ok=True)
    (OUT / 'manifest.json').write_text(json.dumps(units, ensure_ascii=False, indent=1))
    (OUT / 'labels.json').write_text(json.dumps(labels, ensure_ascii=False, indent=1))
    per = collections.Counter(f"{v['source']}/{v['class_name']}/{v['label']}" for v in labels.values())
    build = {'units': len(units), 'units_by_source': dict(collections.Counter(u['source'] for u in units.values())),
             'candidates': sum(len(u['candidates']) for u in units.values()), 'scored_labels': len(labels),
             'by_source_class_label': dict(sorted(per.items())), 'excluded': dict(excluded),
             'dedupe': 'image SHA256 vs design set (113 events) and v2 set (170 images); same-image boxes merged at IoU>=0.9'}
    (OUT / 'build.json').write_text(json.dumps(build, ensure_ascii=False, indent=1))
    print(json.dumps(build, ensure_ascii=False, indent=1))


if __name__ == '__main__':
    main()
