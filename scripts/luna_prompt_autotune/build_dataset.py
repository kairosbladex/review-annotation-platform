"""组装 Luna 分类型提示词自动迭代数据集（本地运行）。

- 有标签池：10-08 设计集（113 事件）+ 10-09 v2 验证集（170 张）+ v3 验证集 09-03 金标（143 张）。
  按事件/图片分组切成 dev（Hermes 可见，用于调提示词）与 holdout（锁定，只在出现新最好版本时评测，结果不给 Hermes）。
- 无标签池：09-24 测试环境全量快照中未用过的事件，按类别分层抽样，只统计 uncertain 比例（道路裂纹等无金标类别靠它观察）。

输出 .tmp/luna-autotune-dataset/：items.json（不含标签）、labels.json、images/（本地图片副本）、
snapshot_images.txt（需在服务器从快照目录复制的图片）。
"""
from __future__ import annotations
import collections
import hashlib
import json
import random
import shutil
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
DATA = REPO / 'data'
OUT = REPO / '.tmp/luna-autotune-dataset'
SEED = 20261009
GOLD = {'真车': 'true', '真目标': 'true', '是人': 'true', 'true': 'true',
        '非车': 'false', '误检': 'false', '不是人': 'false', 'false': 'false'}
UNLABELED_QUOTA = [('道路裂纹', 40), ('光斑', 30), ('违建', 30), ('人员', 20), ('两轮车|三轮车|卡车|公交车|没戴头盔', 20), ('车', 10)]


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    if OUT.exists(): shutil.rmtree(OUT)
    (OUT / 'images').mkdir(parents=True)
    items, labels, snapshot = [], {}, []

    def add(uid, group, image, image_sha, cands, labelmap, source):
        items.append({'unit': uid, 'group': group, 'source': source, 'image': image, 'sha256': image_sha,
                      'candidates': [{'key': k, 'class_name': c, 'bbox': b} for k, c, b in cands]})
        for k, c, _ in cands:
            if k in labelmap: labels[k] = {'label': labelmap[k], 'class_name': c, 'source': source}

    # 1. 10-08 设计集：图片在服务器快照目录
    design = json.loads((DATA / 'video-clean-source-eval-20261008/selection.json').read_text())
    shas = json.loads((DATA / 'luna-eval-20261008/input-audit.json').read_text())['image_sha256']
    for e in json.loads((DATA / 'luna-eval-20261008/allowlist.json').read_text()):
        eid = e['event_id']
        lm = {d['candidate_id']: GOLD.get(design['labels'][d['candidate_id']]['label'])
              for d in e['detections'] if GOLD.get(design['labels'].get(d['candidate_id'], {}).get('label'))}
        add(eid, eid, f'snapshot:{eid}.jpg', shas[eid], [(d['candidate_id'], d['class_name'], d['bbox']) for d in e['detections']], lm, 'design')
        snapshot.append(eid)
    used_sha = set(shas.values())

    # 2/3. v2、v3 验证集：本地图片
    for d, skip in [('luna-prompt-v2-eval-20261009', None), ('luna-prompt-v3-eval-20261009', 'person_regression')]:
        m = json.loads((DATA / d / 'manifest.json').read_text())
        lab = json.loads((DATA / d / 'labels.json').read_text())
        for uid, u in m.items():
            if u['source'] == skip or u['sha256'] in used_sha: continue
            used_sha.add(u['sha256'])
            safe = uid.replace(':', '_')
            shutil.copyfile(DATA / u['image'], OUT / 'images' / (safe + '.jpg'))
            add(safe, safe, f'local:{safe}.jpg', u['sha256'], [(c['key'], c['class_name'], c['bbox']) for c in u['candidates']],
                {c['key']: lab[c['key']]['label'] for c in u['candidates'] if c['key'] in lab}, u['source'])

    # 切分：按组、按主类别分桶，桶内随机顺序每 3 组取 1 组进 holdout
    rng = random.Random(SEED)
    bucket = collections.defaultdict(list)
    for it in items:
        cls = collections.Counter(labels[c['key']]['class_name'] for c in it['candidates'] if c['key'] in labels)
        bucket[cls.most_common(1)[0][0] if cls else '无'].append(it)
    for b, its in sorted(bucket.items()):
        rng.shuffle(its)
        for i, it in enumerate(its): it['split'] = 'holdout' if i % 3 == 2 else 'dev'
    for it in items:
        for c in it['candidates']:
            if c['key'] in labels: labels[c['key']]['split'] = it['split']

    # 4. 无标签池：09-24 快照未用事件，按类别分层抽样
    rows = collections.defaultdict(list)
    for line in open(DATA / 'ktd-test-real-monitoring-full-20260924/input/inference_manifest.jsonl'):
        r = json.loads(line); rows[r['event_id']].append(r)
    used_events = {it['unit'] for it in items} | {it['group'] for it in items}
    chosen = []
    for pattern, quota in UNLABELED_QUOTA:
        want = set(pattern.split('|'))
        pool = sorted(e for e, rs in rows.items() if e not in used_events and e not in chosen and any(r['class_name'] in want for r in rs))
        rng.shuffle(pool); chosen += pool[:quota]
    for eid in chosen:
        cands = []
        for r in rows[eid]:
            w, h = r['frame_size']; x1, y1, x2, y2 = r['bbox_px']
            b = [max(0, x1 / w), max(0, y1 / h), min(1, x2 / w), min(1, y2 / h)]
            if b[0] < b[2] and b[1] < b[3]: cands.append((r['candidate_id'], r['class_name'], b))
        items.append({'unit': eid, 'group': eid, 'source': 'unlabeled0924', 'image': f'snapshot:{eid}.jpg', 'sha256': None,
                      'split': 'unlabeled', 'candidates': [{'key': k, 'class_name': c, 'bbox': b} for k, c, b in cands]})
        snapshot.append(eid)

    (OUT / 'items.json').write_text(json.dumps(items, ensure_ascii=False))
    (OUT / 'labels.json').write_text(json.dumps(labels, ensure_ascii=False))
    (OUT / 'snapshot_images.txt').write_text('\n'.join(snapshot) + '\n')
    summary = {'items_by_split': collections.Counter(it['split'] for it in items),
               'labels_by_split_class': collections.Counter(f"{v['split']}/{v['class_name']}/{v['label']}" for v in labels.values()),
               'unlabeled_candidates_by_class': collections.Counter(c['class_name'] for it in items if it['split'] == 'unlabeled' for c in it['candidates'])}
    summary = {k: dict(sorted(v.items())) for k, v in summary.items()}
    (OUT / 'build.json').write_text(json.dumps(summary, ensure_ascii=False, indent=1))
    print(json.dumps(summary, ensure_ascii=False, indent=1))


if __name__ == '__main__':
    main()
