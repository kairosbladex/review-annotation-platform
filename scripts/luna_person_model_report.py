"""Read-only report of frozen Luna/Sol comparisons, with all 41 boxes visible."""
import csv
import html
import json
from pathlib import Path

from PIL import Image, ImageDraw
from luna_person_ablation import project_box
from luna_person_score import score


def main():
    root = Path(__file__).resolve().parents[1]/'data/luna-person-diagnosis'
    private = root/'private'; out = private/'model-compare'
    units = json.loads((private/'experiment-units.json').read_text())
    positive = {(c['unit'],c['target_index']): f'P{c["case"]:02}' for c in json.loads((private/'cases.json').read_text())}
    records = {}
    for arm in ('A','H'):
        for round_no in (1,2):
            key = f'LUNA-{arm}{round_no}'
            records[key] = {}
            for u in units:
                p = private/'raw'/f'{u["unit"]}-r{round_no}.json' if arm == 'A' else private/'ablation'/f'H{round_no}-{u["unit"]}.json'
                records[key][u['unit']] = json.loads(p.read_text())
    for key in ('SOL-A1','SOL-A2','SOL-H1','SOL-H2'):
        records[key] = {r['unit']:r for p in (out/'results').glob(key+'-*.json') for r in [json.loads(p.read_text())]}
    metrics = {k:score(units,v) for k,v in records.items()}
    (root/'model-compare-metrics.json').write_text(json.dumps(metrics,indent=2)+'\n')
    rows, cards, negative_id = [], [], 0
    (out/'views').mkdir(exist_ok=True)

    def draw_view(image, bbox, path):
        box=[bbox[0]*image.width,bbox[1]*image.height,bbox[2]*image.width,bbox[3]*image.height]
        side=max(80,int(max(box[2]-box[0],box[3]-box[1])*3))
        left=max(0,min(image.width-side,int((box[0]+box[2]-side)/2)))
        top=max(0,min(image.height-side,int((box[1]+box[3]-side)/2)))
        crop=image.crop((left,top,left+side,top+side)).resize((320,320),Image.Resampling.NEAREST)
        s=320/side
        ImageDraw.Draw(crop).rectangle(((box[0]-left)*s,(box[1]-top)*s,(box[2]-left)*s,(box[3]-top)*s),outline='cyan',width=1)
        crop.save(path)

    for u in units:
        for i,c in enumerate(u['candidates']):
            case = positive.get((u['unit'],i))
            if not case:
                negative_id+=1;case=f'N{negative_id:02}'
            row={'case':case,'original_gold':c['gold']};details=[]
            for key,group in records.items():
                r=group.get(u['unit']);decision='missing'
                if r and r.get('success'):
                    body=r.get('result',r.get('raw_body'));item=next(x for x in body['results'] if x['target_index']==i)
                    decision=item['decision']
                    details.append('<p><b>'+html.escape(key+' · '+decision)+'</b><br>'+ '<br>'.join(html.escape(str(item.get(k,''))) for k in ('observed_object','reasoning','image_limitation'))+'</p>')
                row[key]=decision
            rows.append(row)
            if case.startswith('P'):
                before=f'../views/{case}-screen.png';after=f'../views/{case}-native.png'
            else:
                before=f'views/{case}-screen.png';after=f'views/{case}-native.png'
                with Image.open(private/'images'/f'{u["unit"]}.jpg') as im:draw_view(im.convert('RGB'),c['bbox'],out/before)
                b=project_box(c['bbox'],u['H'],[960,720],u['native_size'])
                with Image.open(private/'native'/f'{u["source_event"]}.png') as im:draw_view(im.convert('RGB'),b,out/after)
            table='<table><tr><th>条件</th><th>结论</th></tr>'+''.join('<tr><td>'+k+'</td><td>'+row[k]+'</td></tr>' for k in records)+'</table>'
            warning='原框完整性/重叠关系单列复核，不改标签。' if case in ('P09','P15','P16','P17','P21') else ''
            cards.append((case,f'<article id="{case}"><h2>{case} · 原标签：'+('人员' if c['gold']=='true' else '非人员')+f'</h2><p>{warning}</p><div class="pair"><figure><img src="{before}"><figcaption>原截图局部，仅放大显示</figcaption></figure><figure><img src="{after}"><figcaption>匹配原帧局部，仅放大显示</figcaption></figure></div>'+table+'<details><summary>展开两种模型各轮解释</summary>'+''.join(details)+'</details></article>'))
    rows.sort(key=lambda r:(r['case'].startswith('N'),r['case']))
    with (root/'model-compare-cases.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator='\n');w.writeheader();w.writerows(rows)
    consistency={}
    for model in ('LUNA','SOL'):
        for arm in ('A','H'):
            complete=[r for r in rows if r[f'{model}-{arm}1']!='missing' and r[f'{model}-{arm}2']!='missing']
            consistency[f'{model}-{arm}']={'paired_boxes':len(complete),'decision_changes':sum(r[f'{model}-{arm}1']!=r[f'{model}-{arm}2'] for r in complete),
                'business_confirmation_changes':sum((r[f'{model}-{arm}1']=='true') != (r[f'{model}-{arm}2']=='true') for r in complete),
                'true_confirmed_both_rounds':sum(r['original_gold']=='true' and r[f'{model}-{arm}1']==r[f'{model}-{arm}2']=='true' for r in complete),
                'false_confirmed_either_round':sum(r['original_gold']=='false' and 'true' in (r[f'{model}-{arm}1'],r[f'{model}-{arm}2']) for r in complete)}
    (root/'model-compare-consistency.json').write_text(json.dumps(consistency,indent=2)+'\n')
    summary='<table><tr><th>条件</th><th>人员确认/24</th><th>非人员误认/17</th><th>完成框/41</th></tr>'+''.join(f'<tr><td>{k}</td><td>{m["tp"]}</td><td>{m["fp"]}</td><td>{m["covered"]}</td></tr>' for k,m in metrics.items())+'</table>'
    specials=json.loads((out/'special-cases.json').read_text())
    special_html=''.join('<article><h2>'+','.join(x['cases'])+'</h2><img class="special" src="'+x['view']+'"><p>'+html.escape(x['finding'])+'</p></article>' for x in specials)
    page='''<!doctype html><html lang="zh"><meta charset="utf-8"><title>人员同图同框模型对照</title><style>
body{font:16px/1.6 system-ui;background:#f2f4f7;color:#18222d;margin:30px auto;max-width:1000px}article{background:white;padding:24px;margin:24px 0;border-radius:12px}.pair{display:flex;gap:24px}figure{margin:0}img{width:320px;max-width:100%}.special{width:560px}table{border-collapse:collapse;background:white;width:100%}td,th{padding:6px;text-align:left;border-bottom:1px solid #ddd}a{color:#1267a5}</style>
<h1>人员同图同框模型对照</h1><p>固定历史 24 个人员框、17 个非人员框。原标签不变，含重叠框，不等于独立人数。A 为原截图，H 为原始分辨率；两种模型均使用同一提示词与三图处理。所有放大仅用于查看，不产生新细节。模型解释不是真值。仅本地查看，无上传或业务操作。</p>'''+summary+special_html+''.join(card for _,card in sorted(cards,key=lambda x:(x[0].startswith('N'),x[0])))+'</html>'
    (out/'index.html').write_text(page)
    print(json.dumps({'metrics':metrics,'consistency':consistency},ensure_ascii=False))


if __name__=='__main__':main()
