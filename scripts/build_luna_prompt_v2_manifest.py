"""组装 Luna 提示词 v2 新样本验证集：只用未参与提示词设计（113 事件）的人工金标。

输出 data/luna-prompt-v2-eval-20261009/{manifest.json,labels.json,build.json}。
manifest 只含图片路径/哈希与候选（类别+归一化框），labels 单独存放，执行时不外发。
"""
from __future__ import annotations
import csv
import collections
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1] / 'data'
OUT = ROOT / 'luna-prompt-v2-eval-20261009'


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def valid(b) -> bool:
    return len(b) == 4 and 0 <= b[0] < b[2] <= 1 and 0 <= b[1] < b[3] <= 1


def main():
    design = json.loads((ROOT / 'video-clean-source-eval-20261008/selection.json').read_text())
    design_events = set(design['events'])
    design_sha = set(json.loads((ROOT / 'luna-eval-20261008/input-audit.json').read_text())['image_sha256'].values())
    units, labels, excluded = {}, {}, collections.Counter()
    seen_sha = set(design_sha)

    def add(unit_id, source, image: Path, cands):
        h = sha(image)
        if unit_id in design_events or h in design_sha:
            excluded[source + ':overlap_design_set'] += 1; return
        if h in seen_sha:
            excluded[source + ':duplicate_image'] += 1; return
        seen_sha.add(h)
        out = []
        for i, (key, cls, box, label) in enumerate(cands):
            if not valid(box) or not cls.strip():
                excluded[source + ':invalid_box'] += 1; continue
            out.append({'key': key, 'class_name': cls, 'bbox': box})
            if label in ('true', 'false'):
                labels[key] = {'label': label, 'class_name': cls, 'source': source}
            elif label == 'unclear':
                excluded[source + ':unclear_label'] += 1
        if out:
            units[unit_id] = {'source': source, 'image': str(image.relative_to(ROOT)), 'sha256': h, 'candidates': out}

    # 1. 复检放过目标高清标注包（车及少量其他类别）
    pkg = ROOT / 'recheck-released-fp-20260930/labeling-package'
    ds = ROOT / 'recheck-released-fp-20260930/dataset-source'
    ann = {}
    with open(pkg / 'annotations.csv', encoding='utf-8-sig') as f:
        for r in csv.DictReader(f):
            ann[r['candidate_id']] = {'真目标': 'true', '误检': 'false', '看不清': 'unclear'}.get(r['label'])
    by_event = collections.defaultdict(list)
    for line in open(pkg / 'targets.jsonl'):
        t = json.loads(line); by_event[t['event_id']].append(t)
    for e, ts in sorted(by_event.items()):
        img = next((ds / d / f'{e}.jpg' for d in ['released_annotated_frames', 'event_annotated_frames'] if (ds / d / f'{e}.jpg').exists()), None)
        if img is None:
            excluded['released_fp:no_image'] += 1; continue
        add(e, 'released_fp', img, [('released_fp:' + t['candidate_id'], t['class_name'], t['original_bbox_norm'], ann.get(t['candidate_id'])) for t in ts])

    # 2. 楼顶难例检验集（人工：真车/屋顶设施/看不清）
    hold = ROOT / 'vehicle-surface-gate-eval-20260929/holdout'
    hl = {}
    for line in open(hold / 'sealed-copy/human_labels.server.jsonl'):
        r = json.loads(line); hl[r['candidate_id']] = {'真车': 'true', '屋顶设施': 'false', '看不清': 'unclear'}[r['label']]
    by_event = collections.defaultdict(list)
    for line in open(hold / 'selected.jsonl'):
        r = json.loads(line); by_event[r['annotated_frame'].removesuffix('.jpg')].append(r)
    for e, rs in sorted(by_event.items()):
        if e in units:
            excluded['rooftop:event_already_in_released_fp'] += 1; continue
        add(e, 'rooftop', hold / rs[0]['annotated_image'], [('rooftop:' + r['candidate_id'], r['class_name'], r['bbox'], hl.get(r['candidate_id'])) for r in rs])

    # 3. 09-24 人工金标（人员 56、车 86），按图片分组
    by_img = collections.defaultdict(list)
    for d in ['ktd-codex-human-gold-train-20260924', 'ktd-codex-human-gold-validation-20260924']:
        for line in open(ROOT / d / 'human_gold_annotations.jsonl'):
            r = json.loads(line); r['_dir'] = d; by_img[r['image_sha256']].append(r)
    for h, rs in sorted(by_img.items()):
        r0 = rs[0]
        if r0['source_event_id'] in units:
            excluded['gold0924:event_already_used'] += 1; continue
        add('gold0924:' + h[:16], 'gold0924', ROOT / r0['_dir'] / r0['source_image'],
            [('gold0924:' + r['candidate_id'], r['class_name'], r['bbox'], r['human_label']) for r in rs])

    OUT.mkdir(exist_ok=True)
    (OUT / 'manifest.json').write_text(json.dumps(units, ensure_ascii=False, indent=1))
    (OUT / 'labels.json').write_text(json.dumps(labels, ensure_ascii=False, indent=1))
    per = collections.Counter(f"{v['source']}/{v['class_name']}/{v['label']}" for v in labels.values())
    build = {'units': len(units), 'candidates': sum(len(u['candidates']) for u in units.values()), 'scored_labels': len(labels),
             'by_source_class_label': dict(sorted(per.items())), 'excluded': dict(excluded),
             'design_set': '113 events of video-clean-source-eval-20261008 excluded by event id and image SHA256'}
    (OUT / 'build.json').write_text(json.dumps(build, ensure_ascii=False, indent=1))
    print(json.dumps(build, ensure_ascii=False, indent=1))


if __name__ == '__main__':
    main()
