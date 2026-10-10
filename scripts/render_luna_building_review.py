"""Standalone review pack; empty human decisions and no model output by default."""
import html,json,pathlib
R=pathlib.Path(__file__).resolve().parents[1]/'data/luna-building-release-validation-20261010'
def main():
 units=json.loads((R/'units.json').read_text());fp={x['key'] for x in json.loads((R/'old-false-positives.json').read_text())};cards=[];items=[]
 for u in units:
  for c in u['candidates']:
   if u['cohort']=='known_fp_regression' and c['key'] not in fp:continue
   idx=len(items);items.append(dict(key=c['key'],bbox=c['bbox'],image='images/'+u['unit']+'.jpg'));x,y,x2,y2=c['bbox'];key=html.escape(c['key'])
   cards.append(f'<article><h3>{idx+1}. {key}</h3><p>{"新增来源" if u["cohort"]=="independent_ai_review" else "共源回归" if u["cohort"]=="source_overlap_regression" else "旧误放复核"} · <a href="images/{u["unit"]}.jpg" target="_blank">查看原截图</a></p><div class="image"><img loading="lazy" src="images/{u["unit"]}.jpg"><span style="left:{x*100}%;top:{y*100}%;width:{(x2-x)*100}%;height:{(y2-y)*100}%"></span></div><select id="label-{idx}"><option value="">待人工确认</option><option value="true">是蓝色或绿色顶棚</option><option value="false">不是蓝色或绿色顶棚</option><option value="uncertain">看不清 / 无法确认</option></select><input id="note-{idx}" placeholder="判断依据或标签争议"></article>')
 document='''<!doctype html><html lang="zh"><meta charset="utf-8"><title>B 独立来源与误放复核</title><style>body{font:16px system-ui;max-width:1100px;margin:24px auto;background:#f5f6f8;color:#172433}article{background:white;padding:20px;margin:20px 0;border-radius:8px}.image{position:relative;display:inline-block;width:100%}.image img{width:100%;display:block}.image span{position:absolute;box-sizing:border-box;border:3px solid #ffdf00;pointer-events:none}select,input,button{padding:10px;margin:10px 8px 0 0}input{width:55%}</style><h1>B 独立来源与误放复核</h1><p>业务口径是蓝色/绿色覆盖屋面，不判断建筑是否合法。黄色框是当前待复核目标；原图中的文字和其他框线不能作为判据。红外伪彩色不能证明物体真实颜色。人工栏均留空，未展示模型答案，避免引导。</p><p>9 个独立原片来源、24 个候选；另有 6 张共源回归图、24 个候选；旧误放 10 个候选。新来源不等于新地理场景。页面只在本机导出 CSV，不连接业务服务，不回写原金标。</p><button onclick="save()">导出人工复核 CSV</button>'''+''.join(cards)
 document+='<script>const items='+json.dumps(items,ensure_ascii=False)+''';function save(){const q=x=>'"'+String(x).replaceAll('"','""')+'"';let rows=[['key','image','bbox','human_label','human_note']];items.forEach((x,i)=>rows.push([x.key,x.image,JSON.stringify(x.bbox),document.getElementById('label-'+i).value,document.getElementById('note-'+i).value]));const a=document.createElement('a');a.href=URL.createObjectURL(new Blob(['\\ufeff'+rows.map(r=>r.map(q).join(',')).join('\\n')],{type:'text/csv;charset=utf-8'}));a.download='human-adjudication.csv';a.click();URL.revokeObjectURL(a.href);}</script></html>'''
 (R/'review.html').write_text(document)
if __name__=='__main__':main()
