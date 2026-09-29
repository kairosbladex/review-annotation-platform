"""Build the assessment PDF from audited metrics and frozen source images."""
from pathlib import Path
import argparse, json, html, math
from PIL import Image, ImageDraw, ImageFont, ImageOps
from reportlab.pdfgen import canvas
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.platypus import Paragraph, Table, TableStyle
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont

HERE=Path(__file__).resolve().parent
FONT='/System/Library/Fonts/STHeiti Light.ttc'
BOLD='/System/Library/Fonts/STHeiti Medium.ttc'
pdfmetrics.registerFont(TTFont('CN',FONT));pdfmetrics.registerFont(TTFont('CNB',BOLD))
pdfmetrics.registerFontFamily('CN',normal='CN',bold='CNB',italic='CN',boldItalic='CNB')
NAVY=colors.HexColor('#172d46');TEAL=colors.HexColor('#147f80');INK=colors.HexColor('#253c52')
MUTED=colors.HexColor('#61758a');PALE=colors.HexColor('#f1f5f7');AMBER=colors.HexColor('#a46918')
W,H=A4;X=40;CW=W-X*2
STYLE=ParagraphStyle('body',fontName='CN',fontSize=10.4,leading=17,textColor=INK,wordWrap='CJK')
SMALL=ParagraphStyle('small',parent=STYLE,fontSize=8.4,leading=13,textColor=MUTED)
LABEL=ParagraphStyle('label',parent=STYLE,fontName='CNB',fontSize=11.2,leading=17)


def para(text,style=STYLE): return Paragraph(text,style)

class Report:
    def __init__(self,path):
        self.c=canvas.Canvas(str(path),pagesize=A4,pageCompression=1)
        self.c.setTitle('原图复核改动效果评估报告 | 2026-09-26')
        self.c.setAuthor('Codex · 测试环境证据核查')
        self.n=0;self.y=0;self.layout=[]
    def page(self,label,title,subtitle=None):
        if self.n:self.end()
        self.n+=1;c=self.c
        c.setFillColor(TEAL);c.rect(X,H-43,25,4,fill=1,stroke=0)
        c.setFont('CNB',9);c.setFillColor(MUTED);c.drawString(X+34,H-42,'KTD  /  原图复核效果评估')
        c.setFont('CN',8);c.drawRightString(W-X,H-42,'2026.09.26  ·  测试环境')
        self.y=H-78
        self.text(label,ParagraphStyle('section',parent=SMALL,textColor=TEAL,fontName='CNB',fontSize=9),after=9)
        self.text(title,ParagraphStyle('h1',parent=STYLE,fontName='CNB',fontSize=22,leading=29,textColor=NAVY),after=10)
        if subtitle:self.text(subtitle,SMALL,after=15)
    def check(self,h):
        if self.y-h<52:raise ValueError(f'Page {self.n} overflow: y={self.y}, height={h}')
    def text(self,text,style=STYLE,after=8):
        p=para(text,style);_,h=p.wrap(CW,1000);self.check(h);p.drawOn(self.c,X,self.y-h);self.y-=h+after
    def sub(self,text):self.text(text,LABEL,after=8)
    def gap(self,h=8):self.y-=h
    def table(self,rows,widths=None,small=False):
        widths=widths or [CW/len(rows[0])]*len(rows[0]);sty=SMALL if small else ParagraphStyle('table',parent=STYLE,fontSize=9.6,leading=14)
        cells=[[para(str(v),ParagraphStyle('cellh',parent=sty,fontName='CNB',textColor=colors.white) if i==0 else sty) for v in row] for i,row in enumerate(rows)]
        t=Table(cells,colWidths=widths,hAlign='LEFT')
        cmds=[('BACKGROUND',(0,0),(-1,0),NAVY),('VALIGN',(0,0),(-1,-1),'TOP'),('LEFTPADDING',(0,0),(-1,-1),9),('RIGHTPADDING',(0,0),(-1,-1),9),('TOPPADDING',(0,0),(-1,-1),8),('BOTTOMPADDING',(0,0),(-1,-1),8),('LINEBELOW',(0,0),(-1,0),0.4,NAVY)]
        for i in range(1,len(rows)):
            cmds.extend([('BACKGROUND',(0,i),(-1,i),PALE if i%2 else colors.white),('LINEBELOW',(0,i),(-1,i),0.3,colors.HexColor('#dce4e9'))])
        t.setStyle(TableStyle(cmds));_,h=t.wrap(CW,1000);self.check(h);t.drawOn(self.c,X,self.y-h);self.y-=h+12
    def box(self,text,tone=TEAL):
        st=ParagraphStyle('box',parent=STYLE,fontName='CNB',fontSize=12,leading=20,textColor=tone)
        p=para(text,st);_,h=p.wrap(CW-30,1000);self.check(h+26);self.c.setFillColor(PALE);self.c.roundRect(X,self.y-h-22,CW,h+22,7,fill=1,stroke=0)
        self.c.setFillColor(tone);self.c.rect(X,self.y-h-22,3,h+22,fill=1,stroke=0);p.drawOn(self.c,X+15,self.y-h-11);self.y-=h+36
    def cards(self,items):
        h=77;gap=10;w=(CW-gap*(len(items)-1))/len(items);self.check(h)
        for i,(n,label) in enumerate(items):
            x=X+i*(w+gap);c=self.c;c.setFillColor(PALE);c.roundRect(x,self.y-h,w,h,6,fill=1,stroke=0)
            c.setFillColor(TEAL);c.setFont('CNB',26);c.drawString(x+13,self.y-33,str(n));c.setFont('CN',9);c.setFillColor(MUTED);c.drawString(x+13,self.y-57,label)
        self.y-=h+18
    def image(self,path):
        with Image.open(path) as im:ratio=im.height/im.width
        height=CW*ratio;self.check(height);self.c.drawImage(str(path),X,self.y-height,width=CW,height=height);self.y-=height+8
    def end(self):
        self.layout.append({'page':self.n,'contentBottom':round(self.y,2)})
        c=self.c;c.setStrokeColor(colors.HexColor('#dce4e9'));c.line(X,41,W-X,41)
        c.setFont('CN',7.5);c.setFillColor(MUTED);c.drawString(X,27,'依据真实测试样本及 Codex 图像核查；非人工金标准确率')
        c.drawRightString(W-X,27,f'{self.n:02} / 08');c.showPage()
    def save(self):self.end();self.c.save();(HERE/'pdf-layout.json').write_text(json.dumps(self.layout,indent=2))


def evidence_panel(src,row,path):
    # Cropping/zoom are display operations only; original source files are untouched.
    eid=row['eventId'];b=row['candidate']['bboxNormalized'];pan=Image.new('RGB',(1440,470),'#f5f7f8');d=ImageDraw.Draw(pan)
    f=ImageFont.truetype(BOLD,22);sm=ImageFont.truetype(FONT,17)
    for k,(name,caption,context) in enumerate([('annotated-source.jpg','初审标注图 · 对应目标',False),('clean-source.jpg','原图局部 · 保留原始像素内容',False),('clean-source.jpg','原图上下文 · 黄框仅定位',True)]):
        im=Image.open(src/'visual'/eid/name).convert('RGB');w,h=im.size
        px=[b[0]*w,b[1]*h,b[2]*w,b[3]*h];cx=(px[0]+px[2])/2;cy=(px[1]+px[3])/2
        side=max(320 if context else 70,max(px[2]-px[0],px[3]-px[1])*(7 if context else 1.8))
        bounds=[max(0,round(cx-side/2)),max(0,round(cy-side/2)),min(w,round(cx+side/2)),min(h,round(cy+side/2))]
        crop=im.crop(bounds)
        if context:ImageDraw.Draw(crop).rectangle([px[0]-bounds[0],px[1]-bounds[1],px[2]-bounds[0],px[3]-bounds[1]],outline='#ffd433',width=max(2,round(side/170)))
        crop=ImageOps.contain(crop,(446,374),Image.Resampling.LANCZOS);x=16+480*k+(446-crop.width)//2;y=68+(374-crop.height)//2;pan.paste(crop,(x,y))
        d.text((16+480*k,12),caption,font=f,fill='#172d46');d.text((16+480*k,40),f'来源尺寸 {w} × {h}；展示缩放不增加细节',font=sm,fill='#61758a')
    pan.save(path,quality=95)


def main(src,output):
    metrics=json.loads((HERE/'metrics.json').read_text());rows={r['sampleId']:r for r in map(json.loads,(src/'真实样本复核结果.jsonl').read_text().splitlines())}
    assets=HERE/'assets';assets.mkdir(exist_ok=True)
    cases=[('table','ba0131e19c1d4c368fad538e53acf09e:2'),('vehicle','03b2af909d474a96aaafddf39766d12e:1'),('pole','1093c997a8ca434fafa4c256b02277f0:1'),('roof','feb66a418f40409e949d0797105ff240:2'),('ground','04933aa6944e4294a47e76b4aa49180e:3'),('rider','d4358e429fe14aa39a709a39f8780396:1')]
    for name,sid in cases:evidence_panel(src,rows[sid],assets/(name+'.jpg'))
    r=Report(output)
    r.page('01  结论先看','原图复核改动效果评估','真实监测样本 / 本轮原图、初审框恢复及判定规则组合改动')
    r.box('局部质量改善已有图像证据；<br/>总体准确率是否提升，尚未得到证明。')
    r.cards([(180,'逐框结果重新核对'),(143,'原图可定位并目视核查'),(4,'旧误确认被拦截')])
    r.text('本次从测试环境重新读取 <b>36 个事件</b>，180 个逐框结论与冻结导出全部一致。未使用既有人工金标评分；图像判断由 Codex 逐项记录，不能替代独立真值。')
    r.table([['观察到的变化','数量','业务含义'],['旧误确认不再自动通过','4','1 个驳回、3 个转人工'],['原先存疑的清晰车辆改为确认','16','原图存在可辨认车辆依据'],['当前自动驳回均有非目标图像依据','7','核查未见真目标被误驳回'],['可辨车辆被支撑面规则转人工','5','其中 3 个此前已自动确认']], [235,48,CW-283])
    r.text('<b>代价：</b>配对 175 个候选的自动判定覆盖率从 <b>79.4%</b> 降至 <b>68.0%</b>，人工比例增加 <b>11.4 个百分点</b>。误确认拦截增强，同时自动处理能力下降。')
    r.text('<b>适用范围：</b>人员、车辆的已检出候选；当前服务为 v8，但本批持久化结果含 v6/v7/v8。该报告评价已有真实结果，不能称为完整 v8 独立准确率测试。',SMALL)

    r.page('02  前后统计','更谨慎，但自动处理比例下降','前后比例仅使用有旧结论的 175 个候选；5 个缺少旧结论的候选单列。')
    r.table([['指标','历史结果','本轮结果','变化'],['自动确认','135 / 77.1%','112 / 64.0%','-23'],['自动驳回','4 / 2.3%','7 / 4.0%','+3'],['存疑 / 人工','36 / 20.6%','56 / 32.0%','+20'],['自动判定覆盖率','139/175 = 79.4%','119/175 = 68.0%','-11.4 pp']], [139,128,128,CW-395])
    r.text('自动判定覆盖率 =（确认 + 驳回）/ 配对候选数。<b>该指标不表示准确率。</b>全量 180 个候选最终为确认 112、驳回 7、人工 61。',SMALL)
    r.sub('逐框迁移矩阵')
    r.table([['旧结论 → 新结论','确认','驳回','人工'],['确认 135','96','1','38'],['驳回 4','0','4','0'],['人工 36','16','2','18'],['缺少旧结论 5（不计配对比例）','0','0','5']], [CW-192,64,64,64])
    r.text('<b>38 个“确认 → 人工”不能全部算作纠错：</b>31 个因定位/匹配条件不足；其余 7 个的图像观察分别支持非目标 3、真车 3、身份不清 1。')
    r.sub('最终 61 个人工候选的原因')
    r.table([['定位/匹配不足','车辆支撑面限制','模型分歧','人员策略限制'],['37','10','8','6']], [CW/4]*4)
    r.text('旧结果中有 12 个事件的汇总计数与逐框 decision 不一致。本报告统一重算逐框结论，未沿用把“存疑”写成“驳回”的历史汇总。',SMALL)

    r.page('03  图像核查','改善体现在哪里','143 个可定位候选逐项查看原图；37 个定位未通过的候选不强行赋予真/假观察标签。')
    r.table([['Codex 原图观察','当前确认','当前驳回','当前人工','合计'],['支持目标','112','0','7','119'],['支持非目标','0','7','6','13'],['不足以判断','0','0','11','11'],['合计','112','7','24','143']], [179,84,84,84,CW-431])
    r.text('上表是图像核查记录，<b>不是人工金标或独立盲测</b>。若干典型案例已在此前调查中见过；当前 119 个自动结论均有图像支持，也不能据此宣称“准确率 100%”。',SMALL)
    r.sub('4 个旧误确认：1 个纠正为假目标，3 个转人工')
    r.table([['事件编号（省略 Event-）','框','原图观察','当前结果'],['20260910084419-2','2','屋顶桌台及座椅','驳回'],['20260910105113-1','2','屋顶桌台及座椅','人工'],['20260918104011-2','2','顶部开口固定设施','人工'],['20260918104011-2','3','红白边界固定设备','人工']], [185,35,180,CW-400],small=True)
    r.text('<b>其他正向信号：</b>16 个“存疑 → 确认”的候选均可从原图辨认车辆。7 个自动驳回均有非目标依据，但其中 4 个此前已经驳回，只有 3 个是本轮新增驳回。')
    r.text('<b>保留的问题：</b>13 个目视非目标中仍有 6 个需要人工；119 个目视支持目标中有 7 个人工（车辆 5、人员 2）。11 个图像仍不清楚的候选全部人工。')
    r.text('目视依据：车辆外形、车窗/车顶、覆盖车体及停车/道路上下文；非目标依据包括桌椅、固定设备、立柱、树影。低细节或身份不明时记录为“不足以判断”。',SMALL)

    r.page('04  典型案例','误确认纠正与真车确认','对照图均来自测试环境附件；黄框仅用于页面定位，展示缩放不增加原图细节。')
    r.sub('A · 屋顶桌台：确认 → 驳回')
    r.text('Event-20260910084419-2 / 框 2 / v7。原图可见桌面和周围座椅，重复排布于屋顶。旧结果将其确认为车辆，本轮改为假目标。',SMALL)
    r.image(assets/'table.jpg')
    r.text('结论：此例支持明确的局部纠错；同时发生了框恢复和模型/规则变化，不能只归因于分辨率。',SMALL,after=16)
    r.sub('B · 地面白色车辆：存疑 → 确认')
    r.text('Event-20260910084422-2 / 框 1 / v7。初审真实轮廓恢复后，原图显示车身、车窗和天窗，车辆结论有清晰图像依据。',SMALL)
    r.image(assets/'vehicle.jpg')
    r.text('结论：原图加准确定位，使部分可辨真车从存疑进入自动确认；本批共有 16 个类似状态变化。',SMALL)

    r.page('05  典型案例','高清原图与保守拦截','高清信息增益和策略防护分别陈述，避免把所有变化解释为准确率提升。')
    r.sub('C · 广场固定杆件：维持驳回，定位更可靠')
    r.text('Event-20260723105651-5 / 框 1（同事件共 2 框）/ v8。标注图 1440×1080，原图 4032×3024。可见重复排列的杆件和底部结构。',SMALL)
    r.image(assets/'pole.jpg')
    r.text('结论：v8 恢复小框后提供可靠原图依据；旧快照已是驳回，因此本例不计为新增准确率收益。',SMALL,after=16)
    r.sub('D · 屋顶固定设施：确认 → 人工')
    r.text('Event-20260918104011-2 / 框 2 / v6。原图显示顶部开口的固定设施，缺少车辆特征；支撑面规则阻止自动确认为车辆。',SMALL)
    r.image(assets/'roof.jpg')
    r.text('结论：这是拦截自动误确认，尚未自动识别为假目标。该事件框 3 也从确认转为人工。',SMALL)

    r.page('06  代价与待修正项','真车也可能被过度拦截','“更少自动通过”不能直接等价为“更准确”；需要同时关注可用性。')
    r.sub('E · 树旁停车：可辨真车，确认 → 人工')
    r.text('Event-20260910084419 / 框 3 / v7。原图可辨车辆，支撑面结果却为 building_roof，因此转人工。类似目视可辨车辆共有 5 个。',SMALL)
    r.image(assets/'ground.jpg')
    r.text('影响：其中 3 个车辆原本已自动确认，现增加人工工作量。优先修正临近建筑、树木的支撑面判断。',SMALL,after=16)
    r.sub('F · 道路骑行者：图像支持人员，仍需人工')
    r.text('Event-20260727151442-4 / 框 1 / v8。原图可见骑行者头部、上身和双臂；人员自动确认开关关闭，最终保持人工。该候选缺少旧结论。',SMALL)
    r.image(assets/'rider.jpg')
    r.text('影响：16 个人员候选最终为 0 确认、2 驳回、14 人工；这反映策略限制，不能解释为人员准确率为零。',SMALL)

    r.page('07  结论边界','为什么还不能给出总体提升百分比','本报告评估真实样本中的可观察改善，不将模型结果转成评价真值。')
    for title,body in [
      ('01 · 前后基线混合','旧基线中 129 个候选的理由为 ResNet、40 个已是 Qwen、6 个为其他理由。新结果也混合 v6/v7/v8，未形成单变量的“有原图/无原图”实验。'),
      ('02 · 批次不是独立测试集','36 个事件中，17 个与训练事件重合、3 个与验证事件重合；另外 16 个未出现在两者中，也未经独立盲测锁定。相邻帧和重复场景进一步限制统计推广。'),
      ('03 · 原图覆盖只有一部分','冻结范围 358 个事件中，36 个有可下载且尺寸兼容的关联图对，约 10.1%。其中 35 个事件至少有 1 框原图模型路径生效。其余 322 个事件不能按已验证处理。'),
      ('04 · 原图不总是更高分辨率','36 对中 33 对像素尺寸相同；3 对原图像素面积为 7.84 倍，但 1 对画面不匹配，实际只有 2 对通过画面一致性检查。去标注和正确定位也可能带来收益。'),
      ('05 · 缺少独立真值，且人工比例改变','Codex 目视观察未获用户逐条确认；37 个定位受阻候选未做有效定位的真/假判断。人工比例增加会改变自动结论的样本组成，不能据此宣称整体更准确。')]:
        r.sub(title);r.text(body,after=12)
    r.box('可确认：局部误确认风险降低。<br/>未证实：总体准确率、全场景召回率和原图的单独因果收益。',AMBER)
    r.text('评估对象仅为初审已给出的候选框，无法衡量初审未检出的目标，也不报告全场景 mAP。',SMALL)

    r.page('08  交付与验收','可复核、可追溯的交付','本次只读取结果、检查图像和运行状态；没有重新训练、发布或改变业务数据。')
    r.table([['保存结果版本','事件','候选','说明'],['v6','22','77','全部可用事件先跑基础改动'],['v7','11','76','框恢复相关事件继续复核'],['v8','3','27','小框/合并框相关事件继续复核']], [95,55,55,CW-205],small=True)
    r.text('当前测试 AI 容器为 <b>clean-source-qwen-20260926-v8</b>，本次只读检查 running / healthy。当前代码沿用已训练 adapter；本轮后续主要改动为定位恢复和策略约束，不是重新训练一个全新模型。',SMALL)
    r.sub('建议按这个顺序验收')
    r.text('<b>1.</b> 先核对 4 个旧误确认、16 个新确认、7 个自动驳回，再查看 5 个被支撑面规则拦下的可辨车辆。<br/><b>2.</b> 补齐 37 个定位受阻候选的来源与框信息，并改善支撑面判断。<br/><b>3.</b> 若需正式准确率结论，冻结统一前后版本和新事件批次，先独立看图再核对结果，分别统计误确认、误驳回和人工比例。')
    r.sub('证据文件与可复现计算')
    r.table([['交付文件','作用'],['metrics.json / analyze.py','逐框重算、175 配对分母、源文件与旧快照哈希'],['visual-observations.jsonl','143 个图像核查记录，明确 humanGold=false'],['live-results.json','2026-09-26 20:04 左右读回；36 事件全部一致'],['README.md / assets/','完整文字说明、报告使用的初审与原图证据']], [185,CW-185],small=True)
    r.text('原始证据：~/Downloads/ktd-clean-source-test-acceptance-20260926/。包括真实样本复核结果、批次汇总、训练重合检查、原图分辨率与来源缺口检查。',SMALL)
    r.text('180 条源结果 SHA-256：<br/><font size="7.1">'+metrics['sources']['resultJsonl']['sha256']+'</font>',SMALL)
    r.text('<link href="https://lxtest.zhuoyilingxi.com:8088/afmap/#/eventCenter" color="#147f80">打开测试事件中心</link>　 |　<link href="http://127.0.0.1:8194/?status=2&amp;source=linked&amp;event=ba0131e19c1d4c368fad538e53acf09e" color="#147f80">打开本机桌台误报对照</link>',SMALL)
    r.save()
    print(json.dumps({'pdf':str(output),'pages':r.n,'layout':r.layout},ensure_ascii=False))

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--source',type=Path,required=True);ap.add_argument('--output',type=Path,default=HERE/'report.pdf');args=ap.parse_args();main(args.source,args.output)
