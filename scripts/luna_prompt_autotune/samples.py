"""错判样本拼图：每个类别、每种错误最多 12 个框，附人工标签与模型结论（JPEG ≤256KB，供 Hermes 看图）。

dev 集：true_rejected（真目标被驳回）、false_released（非目标被放过）；
无标签池：uncertain（模型判看不清，没有人工标签，只用于观察）。
"""
from __future__ import annotations
import io
import json
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

TILE, CAP, PER = 256, 40, 12
FONT_PATH = '/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc'


def _font(size):
    for p in [FONT_PATH, '/usr/share/fonts/opentype/noto/NotoSerifCJK-Regular.ttc', '/usr/share/fonts/truetype/arphic/uming.ttc']:
        try: return ImageFont.truetype(p, size)
        except OSError: continue
    return ImageFont.load_default()


def _tile(image_path: Path, bbox, caption):
    im = Image.open(image_path).convert('RGB'); w, h = im.size
    x1, y1, x2, y2 = bbox[0] * w, bbox[1] * h, bbox[2] * w, bbox[3] * h
    side = max(96, (x2 - x1) * 3, (y2 - y1) * 3); cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    box = [max(0, cx - side / 2), max(0, cy - side / 2), min(w, cx + side / 2), min(h, cy + side / 2)]
    crop = im.crop(box); d = ImageDraw.Draw(crop)
    d.rectangle([x1 - box[0], y1 - box[1], x2 - box[0], y2 - box[1]], outline='yellow', width=2)
    crop = crop.resize((TILE, TILE))
    tile = Image.new('RGB', (TILE, TILE + CAP), 'white'); tile.paste(crop, (0, 0))
    ImageDraw.Draw(tile).text((4, TILE + 4), caption, fill='black', font=_font(15))
    return tile


def _save(tiles, path: Path):
    cols = 4; rows = (len(tiles) + cols - 1) // cols
    sheet = Image.new('RGB', (cols * TILE, rows * (TILE + CAP)), 'white')
    for i, t in enumerate(tiles): sheet.paste(t, ((i % cols) * TILE, (i // cols) * (TILE + CAP)))
    for q in (85, 75, 60, 45):
        buf = io.BytesIO(); sheet.save(buf, 'JPEG', quality=q)
        if buf.tell() <= 256_000: break
    path.write_bytes(buf.getvalue())


def build(version, items, labels, detail, out: Path):
    out.mkdir(parents=True, exist_ok=True)
    assets = Path(__file__).resolve().parents[1] / 'assets/images'
    groups = {}
    for u in items:
        for c in u['candidates']:
            d = detail.get(c['key'])
            if d is None: continue
            lab = labels.get(c['key'])
            if u['split'] == 'dev' and lab:
                if lab['label'] == 'true' and d['raw'] != 'true': kind = 'true_rejected'
                elif lab['label'] == 'false' and d['raw'] == 'true': kind = 'false_released'
                else: continue
                gold = '是' if lab['label'] == 'true' else '不是'
            elif u['split'] == 'unlabeled' and d['raw'] == 'uncertain':
                kind, gold = 'uncertain', '未标注'
            else: continue
            groups.setdefault((c['class_name'], kind), []).append((u, c, d, gold))
    index = {}
    for (cls, kind), rows in sorted(groups.items()):
        rows = rows[:PER]; name = f'{cls}-{kind}'
        tiles = [_tile(assets / (u['unit'] + '.jpg'), c['bbox'], f'#{i} 人工:{g} 模型:{d["raw"]}') for i, (u, c, d, g) in enumerate(rows)]
        _save(tiles, out / f'{name}.jpg')
        (out / f'{name}.json').write_text(json.dumps([{'tile': i, 'human_label': g, 'model': d['raw'], 'reasoning': d.get('reasoning'),
                                                       'image_limitation': d.get('limitation')} for i, (u, c, d, g) in enumerate(rows)],
                                                     ensure_ascii=False, indent=1), encoding='utf-8')
        index[name] = {'class_name': cls, 'kind': kind, 'total': len(groups[(cls, kind)]), 'shown': len(rows)}
    (out / 'index.json').write_text(json.dumps(index, ensure_ascii=False, indent=1), encoding='utf-8')
