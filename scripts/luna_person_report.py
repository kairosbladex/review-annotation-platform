"""Build a local, read-only per-case evidence viewer and deidentified tables."""
import csv
import hashlib
import html
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]/'data/luna-person-diagnosis'
PRIVATE = ROOT/'private'
cases = json.loads((PRIVATE/'cases.json').read_text())
(PRIVATE/'views').mkdir(exist_ok=True)
rows, cards = [], []


def view(image, box, dest):
    """Context crop for inspection; display scaling is explicit, not new detail."""
    cx, cy = (box[0]+box[2])/2, (box[1]+box[3])/2
    side = max(80, int(max(box[2]-box[0],box[3]-box[1])*3))
    x, y = max(0,min(image.width-side,int(cx-side/2))),max(0,min(image.height-side,int(cy-side/2)))
    crop = image.crop((x,y,x+side,y+side))
    scale = 320/side
    crop = crop.resize((320,320),Image.Resampling.NEAREST)
    draw = ImageDraw.Draw(crop)
    draw.rectangle(((box[0]-x)*scale,(box[1]-y)*scale,(box[2]-x)*scale,(box[3]-y)*scale),outline='cyan',width=1)
    crop.save(dest)


for c in cases:
    case = f'P{c["case"]:02d}'
    image = Image.open(PRIVATE/'images'/f'{c["unit"]}.jpg').convert('RGB')
    source = Image.open(PRIVATE/'native'/f'{c["source_event"]}.png').convert('RGB')
    poly = np.array(c['native_quad'])
    high_box = [*poly.min(axis=0),*poly.max(axis=0)]
    view(image,c['screenshot_box_px'],PRIVATE/'views'/f'{case}-screen.png')
    view(source,high_box,PRIVATE/'views'/f'{case}-native.png')
    results, experiment_details = {}, []
    for path in sorted((PRIVATE/'ablation').glob(f'*-{c["unit"]}.json')):
        r = json.loads(path.read_text())
        if r['success']:
            detail = next(x for x in r['result']['results'] if x['target_index']==c['target_index'])
            results[r['entry']] = detail['decision']
            experiment_details.append('<p><b>'+html.escape(r['entry'])+' · '+html.escape(detail['decision'])+'</b><br>'+ '<br>'.join(html.escape(str(detail.get(k,''))) for k in ('observed_object','reasoning','image_limitation'))+'</p>')
        else:
            results[r['entry']] = '执行失败'
    row = {'case':case,'screenshot_box':f'{c["box_size"][0]}x{c["box_size"][1]}',
           'native_box':f'{high_box[2]-high_box[0]:.1f}x{high_box[3]-high_box[1]:.1f}',
           'red_fraction':round(c['red_pixel_fraction'],3),'source_box_error':round(c['bbox_source_error'],6),
           'frame_roi_correlation':round(c['alignment']['roi_ncc'],3),
           'baseline_r1':c['rounds'][0]['decision'],'baseline_r2':c['rounds'][1]['decision'],
           'finding':c['visual_findings'],'experiments':json.dumps(results,ensure_ascii=False)}
    rows.append(row)
    explain=''.join(f'<p><b>基线第 {i+1} 轮：{html.escape(r["decision"])}</b><br>{html.escape(r["observed_object"])}<br>{html.escape(r["reasoning"])}<br>{html.escape(r["image_limitation"])}</p>' for i,r in enumerate(c['rounds']))
    cards.append(f'''<article id="{case}"><h2>{case} · 截图框 {row['screenshot_box']} 像素</h2>
    <div class="pair"><figure><img src="views/{case}-screen.png"><figcaption>实际输入截图局部；仅放大显示</figcaption></figure>
    <figure><img src="views/{case}-native.png"><figcaption>匹配原生帧局部；仅放大显示</figcaption></figure></div>
    <p><a href="images/{c['unit']}.jpg" target="_blank">完整截图</a> · <a href="native/{c['source_event']}.png" target="_blank">完整原生帧</a> · 原帧框 {row['native_box']} · 局部匹配 {row['frame_roi_correlation']}</p>
    <p>{html.escape(c['visual_findings'])}</p><div class="explain">{explain}</div><p>定向对照：{html.escape(row['experiments'])}</p><details><summary>展开各条件模型解释</summary>{''.join(experiment_details)}</details></article>''')

with (ROOT/'cases.csv').open('w') as f:
    writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
labels={'A1':'旧截图基线一','A2':'旧截图基线二','P1':'只换人员卡','S1':'同尺寸无标注一','S2':'同尺寸无标注二','H1':'原始分辨率一','H2':'原始分辨率二','Z1':'启动失败，不评分','Z2':'有额外压缩，不作严格对照','Z4':'原图直接裁剪放大','C1':'原图与人员卡组合一','C2':'原图与人员卡组合二'}
metrics=json.loads((ROOT/'metrics.json').read_text())
summary='<table><tr><th>条件</th><th>真人确认 / 24</th><th>非人员误认 / 17</th><th>完成框 / 41</th></tr>'+''.join(f'<tr><td>{html.escape(k+" · "+labels.get(k,k))}</td><td>{v["tp"] if v["covered"] else "—"}</td><td>{v["fp"] if v["covered"] else "—"}</td><td>{v["covered"]}</td></tr>' for k,v in metrics.items())+'</table>'
summary+='<p>这是历史漏判诊断集，不是独立盲测。看不清仍不算确认；单轮改善不能代表修复通过。</p><nav>'+ ' · '.join(f'<a href="#P{i:02}">P{i:02}</a>' for i in range(1,25))+'</nav>'
(PRIVATE/'index.html').write_text('''<!doctype html><html lang="zh"><meta charset="utf-8"><title>人员24框逐项诊断</title>
<style>body{font:16px/1.6 system-ui;background:#f2f4f7;color:#18222d;margin:30px auto;max-width:1000px}article{background:white;padding:24px;margin:24px 0;border-radius:12px}h1,h2{line-height:1.3}.pair,.explain{display:flex;gap:24px}.explain>p{flex:1}figure{margin:0}img{max-width:100%;width:320px}figcaption{color:#586370}a{color:#1267a5}table{border-collapse:collapse;background:white;width:100%}td,th{padding:8px;text-align:left;border-bottom:1px solid #ddd}nav{line-height:2}</style>
<h1>人员 24 框逐项诊断</h1><p>24 是已有真标签的候选框，不是24名独立人员。青色线仅用于定位。所有放大都只改变显示，不生成新细节；解释来自原模型，不能替代真值。本页只读，无上传或业务操作。</p>'''+summary+''.join(cards)+'</html>')
print(json.dumps({'cases':len(rows),'viewer':str(PRIVATE/'index.html')},ensure_ascii=False))
