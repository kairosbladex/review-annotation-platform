"""按 Java 将要传的格式组装视频事件回放清单，并把快照里已下载的原视频按缓存命名预置为软链接。"""
import hashlib, json, os
from pathlib import Path
C = Path('/data/workspace/jzheng/codex-ktd-real-monitoring-20260924')
W = Path('/data/workspace/jzheng/claude-video-clean-eval-20261008')
sel = json.load(open(W / 'selection.json'))
cands = {}
for line in open(C / 'input/inference_manifest.jsonl'):
    r = json.loads(line); cands[r['candidate_id']] = r
src = {json.loads(l)['event_id']: json.loads(l) for l in open(C / 'input/source_manifest.jsonl')}
assets = {}
for line in open(C / 'input/asset_index.jsonl'):
    x = json.loads(line)
    if x['kind'] != 'annotated':
        assets[x['object_path']] = x['local_path']
man, skipped = [], {}
for eid, cids in sorted(sel['events'].items()):
    videos = {}
    for s in src.get(eid, {}).get('raw_sources', []):
        if s.get('kind') != 'video' or s['object_path'] not in assets:
            continue
        local = C / assets[s['object_path']]
        if not local.is_file():
            continue
        d = float(s.get('delta_seconds_from_event') or 1e12)
        if s['object_path'] not in videos or abs(d) < abs(videos[s['object_path']][0]):
            videos[s['object_path']] = (d, s['url'], local)
    if not videos:
        skipped[eid] = 'no_downloaded_video'; continue
    sources = []
    for d, url, local in sorted(videos.values(), key=lambda v: abs(v[0]))[:2]:
        key = hashlib.sha256(url.encode()).hexdigest()[:32]
        link = W / 'cache' / f'{key}.mp4'
        if not link.exists():
            os.symlink(local, link)
        sources.append({'url': url, 'delta_seconds': d})
    rows = []
    for cid in cids:
        c = cands[cid]; w, h = c['frame_size']; x1, y1, x2, y2 = c['bbox_px']
        rows.append({'candidate_id': cid, 'class_id': 0, 'class_name': c['class_name'],
                     'confidence': c['detector_confidence'], 'bbox': [x1 / w, y1 / h, x2 / w, y2 / h]})
    man.append({'event_id': eid, 'sources': sources, 'detections': rows})
json.dump(man, open(W / 'manifest.json', 'w'), ensure_ascii=False)
print(len(man), sum(len(m['detections']) for m in man), skipped, len(list((W / 'cache').glob('*.mp4'))))
