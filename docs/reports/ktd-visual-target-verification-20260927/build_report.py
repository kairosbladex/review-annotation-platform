"""离线生成本次真假目标报告；仅使用冻结结果和既有图像，不调用模型或业务接口。"""
import argparse
import collections
import hashlib
import html
import importlib.util
import json
import shutil
from pathlib import Path

HERE = Path(__file__).resolve().parent
PRIOR = HERE.parent / 'ktd-original-recheck-assessment-20260926'


def check_evidence(data):
    rows = data['candidates']
    assert len(rows) == len({r['sampleId'] for r in rows}) == 180
    assert collections.Counter(r['decision'] for r in rows) == {
        'confirmed': 112, 'rejected': 7, 'uncertain': 61}
    assert sum(bool(r['visualId']) for r in rows) == 143
    assert all(not r['humanGold'] for r in rows)
    current = data['currentReadback']['events']
    assert len(current) == 37 and all(r['status'] == '2' for r in current)
    assert sum(r['matchesFrozenCandidateDecisions'] is True for r in current) == 36
    latest = data['latestBusinessVerification']['events']
    assert [r['total'] for r in latest] == [2, 10]
    assert all(r['sameAsPreReleaseDecisions'] for r in latest)
    assert not data['accuracyMeasured']


def build_pdf(data, out):
    spec = importlib.util.spec_from_file_location('prior_report', PRIOR / 'build_pdf.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    m = module

    class Report(m.Report):
        def page(self, label, title, subtitle=None):
            if self.n:
                self.end()
            self.n += 1
            c = self.c
            c.setFillColor(m.TEAL)
            c.rect(m.X, m.H - 43, 25, 4, fill=1, stroke=0)
            c.setFont('CNB', 9)
            c.setFillColor(m.MUTED)
            c.drawString(m.X + 34, m.H - 42, 'KTD  /  真假目标核验报告')
            c.setFont('CN', 8)
            c.drawRightString(m.W - m.X, m.H - 42, '2026.09.27  ·  测试环境')
            self.y = m.H - 78
            self.text(label, m.ParagraphStyle('section', parent=m.SMALL,
                      textColor=m.TEAL, fontName='CNB', fontSize=9), after=9)
            self.text(title, m.ParagraphStyle('h1', parent=m.STYLE, fontName='CNB',
                      fontSize=22, leading=29, textColor=m.NAVY), after=10)
            if subtitle:
                self.text(subtitle, m.SMALL, after=15)

        def end(self):
            self.layout.append({'page': self.n, 'contentBottom': round(self.y, 2)})
            self.c.setStrokeColor(m.colors.HexColor('#dce4e9'))
            self.c.line(m.X, 41, m.W - m.X, 41)
            self.c.setFont('CN', 7.5)
            self.c.setFillColor(m.MUTED)
            self.c.drawString(m.X, 27, '真实测试样本与 Codex 图像核查 / 不作为独立真值准确率')
            self.c.drawRightString(m.W - m.X, 27, f'{self.n:02} / 06')
            self.c.showPage()

        def save(self):
            self.end()
            assert self.n == 6
            self.c.save()
            (out / 'pdf-layout.json').write_text(json.dumps(self.layout, indent=2))

    r = Report(out / '真假目标核验报告-20260927.pdf')
    r.c.setTitle(data['title'])
    r.c.setAuthor('Codex · 真实测试环境图像核验')
    r.page('01  结论与范围', '原图辅助视觉复检', '真假目标核验报告 / 09-27 发布复测 + 此前原图批次观察')
    r.box('原图链路已生效，局部改善有图像依据；<br/>总体准确率提升尚未得到证明。')
    r.cards([(4, '旧误确认被拦截'), (16, '可辨真车从存疑转确认'), (5, '可辨车辆仍被转人工')])
    r.text('上述观察来自此前 <b>36 个事件、180 个候选</b>的原图批次。今天 10:38:31 只读回查，这 180 条逐框结论均未变化；没有统一重跑全部事件。')
    r.table([['证据层次', '本次可确认内容'],
             ['最新发布复测', '2 个事件、12 个候选完成；有原图 2 个均驳回，缺原图 10 个确认 4 / 驳回 6。'],
             ['原图图像核查', '143 个候选有有效定位及图像观察；37 个定位受阻，不强判真假。'],
             ['来源接入验收', '新自然知眸回调的原图落库、映射、复核全过程仍待验收。']], [119, m.CW - 119])
    r.text('<b>变化的代价：</b>175 个可配对候选的自动判定覆盖率由 <b>79.4%</b> 降至 <b>68.0%</b>，人工比例增加 11.4 个百分点。覆盖率不是准确率。')
    r.text('<b>下一步：</b>先验收新回调原图，再核查关键真假目标，修正定位与支撑面过度拦截，最后做同批新事件对照；暂不启动新的训练。')
    r.text('统计去重：37 个事件、190 个候选。最新 12 个中有 2 个已在原 180 个中，不相加为 192。本报告没有使用历史人工金标评分，图像观察也不等于人工确认。', m.SMALL)

    r.page('02  发布后真实结果', '高清原图路径已实际执行', '最新两个事件在 09-27 10:25、10:26 完成；模型权重未因回调接入而改变。')
    r.sub('A · 两个“人员”候选：原图支持固定杆件，均驳回')
    r.text('Event-20260723105651-5。1440×1080 初审标注图映射 4032×3024 原图，两个实际小框均恢复成功。', m.SMALL)
    r.image(out / 'assets/pole.jpg')
    r.table([['框', '初审图实际轮廓（像素）', '原图证据', '结果'],
             ['1', '[723,157,735,176]', '重复排列的杆件及底座', '驳回'],
             ['2', '[745,161,758,182]', '同一排列中的固定杆件', '驳回']], [28, 178, 232, m.CW - 438], small=True)
    r.text('两框均保存原图生效和四视图证据，Qwen 基础模型、适配器一致驳回。画面匹配分数 0.99254 不是分类置信度。发布前也为两框驳回，因此此例是链路验证，不计新增纠错。', m.SMALL)
    r.sub('B · 缺原图事件：继续单图复核')
    r.text('Event-20260923180729 / 10 个“人员”候选：确认 4、驳回 6；逐框序列如下。', m.SMALL)
    r.table([['框', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10'],
             ['结果', '假', '真', '真', '假', '真', '假', '假', '假', '真', '假']], [45] + [(m.CW - 45) / 10] * 10, small=True)
    r.text('表中“真/假”是旧单图模型输出，未取得本报告的独立确认；不能用于证明原图改进。该事件与发布前逐框结果相同。', m.SMALL)
    r.text('图像说明：对照图为核验展示裁切，不是模型四视图张量原样导出；未生成或补画细节，黄框仅用于定位。', m.SMALL)

    r.page('03  批量观察', '更谨慎，同时增加了人工量', '36 事件 / 180 候选；前后比例分母为 175，另 5 个缺少旧结论。')
    r.table([['指标', '历史结果', '改动后结果', '变化'],
             ['自动确认', '135 / 77.1%', '112 / 64.0%', '-23'],
             ['自动驳回', '4 / 2.3%', '7 / 4.0%', '+3'],
             ['存疑 / 人工', '36 / 20.6%', '56 / 32.0%', '+20'],
             ['自动判定覆盖率', '139/175 = 79.4%', '119/175 = 68.0%', '-11.4 pp']], [138, 132, 132, m.CW - 402])
    r.sub('143 个有效定位候选的图像观察')
    r.table([['Codex 图像观察', '确认', '驳回', '人工', '合计'],
             ['支持目标', '112', '0', '7', '119'],
             ['支持非目标', '0', '7', '6', '13'],
             ['无法可靠判断', '0', '0', '11', '11'],
             ['合计', '112', '7', '24', '143']], [195, 80, 80, 80, m.CW - 435])
    r.text('7 个驳回均有非目标图像依据，其中 4 个旧结果已驳回、3 个新增驳回。观察没有用户逐条确认，且并非独立盲测，因此不换算为“100% 准确率”。', m.SMALL)
    r.sub('为什么仍有 61 个候选需要人工')
    r.table([['定位/匹配不足', '支撑面限制', '模型分歧', '人员策略限制'],
             ['37', '10', '8', '6']], [m.CW / 4] * 4)
    r.text('38 个“确认 → 人工”中，31 个受定位/匹配限制，不能全算误报纠正。图像核查记录复用 09-26 逐项观察，本报告复看典型案例。', m.SMALL)

    r.page('04  真假目标对照', '误确认纠正与真车确认', '左右对照：初审标注图 / 原图局部 / 原图上下文。')
    r.sub('屋顶桌椅：原先确认为车，现驳回')
    r.text('Event-20260910084419-2 / 框 2 / v7。原图可见桌面和周围座椅，屋顶重复排列，支持非车辆。', m.SMALL)
    r.image(out / 'assets/table.jpg')
    r.text('另外 3 个旧误确认转为人工：Event-20260910105113-1 / 框 2，以及 Event-20260918104011-2 / 框 2、3。它们是拦截误确认，尚未自动判假。', m.SMALL, after=15)
    r.sub('地面白色车辆：原先存疑，现确认')
    r.text('Event-20260910084422-2 / 框 1 / v7。可辨车身、车窗及天窗，停车场景一致，支持车辆。', m.SMALL)
    r.image(out / 'assets/vehicle.jpg')
    r.text('共有 16 个可辨车辆从存疑转确认。前后同时涉及框恢复、模型与策略调整，不能把所有收益单独归因于原图分辨率。', m.SMALL)

    r.page('05  仍需修正', '清晰目标也可能转人工', '优先提升可用性，并保留证据不足时的人工入口。')
    r.sub('树旁真车：支撑面判断过于保守')
    r.text('Event-20260910084419 / 框 3 / v7。可辨车辆被判断为屋顶相关，确认 → 人工。共有 5 个类似可辨车辆仍人工，其中 3 个曾自动确认。', m.SMALL)
    r.image(out / 'assets/ground.jpg')
    r.text('建议定位支撑面判断为何受树木、邻近建筑干扰，再在原案例和新场景验证；不直接关闭保护策略。', m.SMALL, after=15)
    r.sub('道路骑行者：人员自动确认策略限制')
    r.text('Event-20260727151442-4 / 框 1 / v8。原图支持骑行者；当前人员自动确认关闭，最终人工，不能称为被判假。', m.SMALL)
    r.image(out / 'assets/rider.jpg')
    r.text('此前 16 个人员候选最终为 0 确认、2 驳回、14 人工。该分布受策略约束，不代表人员识别准确率为零。', m.SMALL)

    r.page('06  后续验收', '按顺序补齐证据，再决定训练', '测试入口：REDACTED_TEST_HOST.invalid:8088/afmap/#/eventCenter')
    r.table([['顺序', '建议工作', '完成条件'],
             ['P0', '新自然知眸回调贯通', '独立原图/标注图、WAR_TO_AI 关联、同帧匹配、真实框和原图生效证据均可追溯。'],
             ['P1', '人工核对关键目标', '先看 4 个旧误确认、16 个新增确认、7 个驳回、5 个真车转人工；集合有重叠，不相加。'],
             ['P1', '修正定位与支撑面', '复核 37 个定位受阻和 5 个真车人工案例，检查新场景，记录人工比例变化。'],
             ['P2', '同批新事件对照', '固定框、权重、提示词与策略，只切换原图输入；记录误确认、误驳回、存疑、原图生效及耗时。'],
             ['P3', '按真实错误决定训练', '只纳入可靠确认，按原图与任务隔离验证集；看不清的样本保留不确定。']], [35, 133, m.CW - 168], small=True)
    r.sub('仍不能发布总体准确率的原因')
    r.text('结果混合 v6/v7/v8；历史基线含已使用 Qwen 的候选；部分事件与训练/验证重合；37 个候选无法可靠定位；没有本批独立真值。只检验已检出候选，不能评价初审漏检。', m.SMALL)
    r.text('09-26 来源快照：358 个相关事件中 36 个有可下载关联图对，约 10.1%。33 对尺寸相同，3 对更高清但 1 对画面不匹配。该覆盖数字不是新回调发布后的测量值。', m.SMALL)
    r.sub('版本与证据')
    r.text('当前只读健康检查：Java UP；GPU v8 running / healthy。远端 test 为 60620f56cc；Java 功能分支头 e61ae57a3e。发布时运行包摘要 f41f3d674537…，来自发布记录，本次未重新计算远端包哈希。', m.SMALL)
    r.text('配套网页可搜索 180 条逐框结果，143 条有原图局部/上下文。evidence.json 与逐框 JSONL 保存时间、版本和理由；完整说明见 Markdown。原始业务响应与凭据不随报告分发。', m.SMALL)
    r.box('本次回调接入没有换权重；最新两例结论未变化。<br/>下一步最关键的是新回调来源验收和真实样本对照。')
    r.save()


def build_html(data, out):
    rows = data['candidates']
    decision = {'confirmed': '确认', 'rejected': '驳回', 'uncertain': '人工', None: '无旧结论'}
    visual = {'target': '图像支持目标', 'non_target': '图像支持非目标',
              'indeterminate': '看不清 / 无法确定', 'not_located': '定位未通过'}
    esc = html.escape
    trs = []
    for row in rows:
        hay = ' '.join(str(row[k]) for k in ['eventNo', 'candidateIndex', 'class', 'visualReason', 'reasonCode'])
        first = f"<button class='sample' data-image='{row['visualId']}.jpg' data-title='{esc(row['eventNo'])} / 框 {row['candidateIndex']}'>查看原图</button>" if row['visualId'] else '<span class="muted">无可靠定位</span>'
        trs.append(f"""<tr data-search="{esc(hay)}" data-decision="{row['decision']}" data-visual="{row['visualAssessment']}">
          <td>{esc(row['eventNo'])}<small>框 {row['candidateIndex']} · {row['class']} · {row['releaseVersion']}</small></td>
          <td>{decision[row['beforeDecision']]} → <b class="{row['decision']}">{decision[row['decision']]}</b></td>
          <td>{visual[row['visualAssessment']]}<small>{esc(row['visualReason'])}</small></td>
          <td>{first}<small>{esc(row['reasonCode'])}</small></td></tr>""")
    case_specs = [
        ('table', '假目标被纠正', 'Event-20260910084419-2 / 框 2', '屋顶桌台与座椅；确认 → 驳回。'),
        ('vehicle', '真车获得确认', 'Event-20260910084422-2 / 框 1', '可辨车身、车窗、天窗；存疑 → 确认。'),
        ('pole', '高清原图已生效', 'Event-20260723105651-5 / 框 1（同事件共 2 框）', '原图支持固定杆件；发布前后均驳回，不计新增纠错。'),
        ('roof', '拦下误确认，仍需人工', 'Event-20260918104011-2 / 框 2', '顶部开口固定设施；确认 → 人工。'),
        ('ground', '真车被过度转人工', 'Event-20260910084419 / 框 3', '树旁车辆清晰；支撑面判断将确认转人工。'),
        ('rider', '人员策略限制', 'Event-20260727151442-4 / 框 1', '原图支持骑行者；人员自动确认关闭，保持人工。')]
    cases = ''.join(f'<article class="case"><h3>{i+1:02} · {esc(title)}</h3><p class="muted">{eid}</p><a href="assets/{name}.jpg" target="_blank" rel="noopener"><img src="assets/{name}.jpg" alt="初审标注图、原图局部和上下文对照" loading="lazy"></a><p>{desc}</p></article>' for i, (name, title, eid, desc) in enumerate(case_specs))
    template = r'''<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>原图辅助视觉复检 · 真假目标核验报告</title>
<style>
:root{--ink:#153048;--muted:#5b7080;--teal:#087d7a;--line:#dae4e9;--bg:#f1f5f7}*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font:16px/1.7 -apple-system,BlinkMacSystemFont,"PingFang SC","Microsoft YaHei",sans-serif}main{max-width:1240px;margin:auto;padding:44px 28px 72px}header{border-top:5px solid var(--teal);padding:24px 0 18px}.eyebrow{font-size:13px;font-weight:700;letter-spacing:2px;color:var(--teal)}h1{font-size:36px;line-height:1.3;margin:12px 0}h2{font-size:25px;margin:0 0 16px}h3{font-size:20px;margin:0 0 8px}p{margin:10px 0}a{color:var(--teal)}.muted,small{color:var(--muted)}.actions{display:flex;gap:12px;flex-wrap:wrap;margin:22px 0}.button,button,select,input{font:inherit}.button,button{border:1px solid var(--line);background:white;color:var(--teal);border-radius:8px;padding:8px 14px;text-decoration:none;cursor:pointer}.primary{background:var(--teal);color:white;border-color:var(--teal)}nav{display:flex;flex-wrap:wrap;gap:18px;margin:20px 0}section{background:white;border:1px solid var(--line);border-radius:14px;padding:28px;margin:24px 0;scroll-margin-top:20px}.callout{background:#e7f3f1;border-left:4px solid var(--teal);padding:18px 22px;border-radius:4px;font-size:19px;font-weight:650}.cards{display:grid;grid-template-columns:repeat(4,1fr);gap:14px;margin:22px 0}.card{padding:17px;background:var(--bg);border-radius:10px}.num{font-size:34px;line-height:1.2;color:var(--teal);font-weight:700}.card span{display:block;font-size:13px;margin-top:8px}table{border-collapse:collapse;width:100%;font-size:14px}th{text-align:left;padding:12px 10px;color:white;background:var(--ink)}td{padding:12px 10px;border-bottom:1px solid var(--line);vertical-align:top}tr:nth-child(even){background:#f7f9fa}td small{display:block;font-size:12px;margin-top:5px;overflow-wrap:anywhere}.tablewrap{overflow-x:auto}.case{padding:24px 0;border-bottom:1px solid var(--line)}.case:last-child{border-bottom:0}.case img{display:block;width:100%;height:auto;border:1px solid var(--line);border-radius:7px}.confirmed{color:#086b65}.rejected{color:#99453c}.uncertain{color:#91621c}.filters{display:flex;gap:12px;flex-wrap:wrap;margin:20px 0}input,select{border:1px solid #b9cad3;border-radius:8px;padding:9px 11px;background:white;color:var(--ink)}input{min-width:320px;flex:1}.sample{font-size:13px;white-space:nowrap}dialog{border:0;border-radius:12px;padding:22px;max-width:900px;width:94vw;color:var(--ink)}dialog::backdrop{background:#122b45ad}dialog img{width:100%;display:block;margin:15px 0}dialog h3{padding-right:60px}.close{position:absolute;right:20px;top:17px}footer{font-size:13px;color:var(--muted);padding-top:20px}ol li{margin-bottom:12px}.note{font-size:13px;color:var(--muted)}@media(max-width:700px){main{padding:20px 12px}h1{font-size:28px}section{padding:20px 14px}.cards{grid-template-columns:1fr 1fr}input{min-width:100%}.num{font-size:29px}.tablewrap table{min-width:720px}}
</style></head><body><main><header><div class="eyebrow">KTD / TEST ENVIRONMENT / 2026.09.27</div><h1>原图辅助视觉复检<br>真假目标核验报告</h1><p class="muted">当前读回：2026-09-27 10:38:31（UTC+8） · 初审已检出的人员与车辆候选</p><div class="actions"><a class="button primary" href="真假目标核验报告-20260927.pdf">打开 6 页 PDF 报告</a><a class="button" href="逐框核验明细.jsonl" download>下载 180 条逐框明细</a><a class="button" href="https://REDACTED_TEST_HOST.invalid:8088/afmap/#/eventCenter" target="_blank" rel="noopener">测试环境事件中心</a></div><nav><a href="#summary">结论</a><a href="#latest">最新实测</a><a href="#stats">前后变化</a><a href="#cases">图像对照</a><a href="#details">逐框明细</a><a href="#next">下一步</a></nav></header>
<section id="summary"><h2>局部改善有依据，总体提升尚未量化</h2><div class="callout">原图链路已在测试环境实际生效。需要继续完成新自然回调的原图验收，并在同一批新事件上比较效果。</div><div class="cards"><div class="card"><div class="num">4</div><span>旧误确认被拦截<br>1 个驳回、3 个人工</span></div><div class="card"><div class="num">16</div><span>可辨真车从存疑转确认<br>来自此前原图批次</span></div><div class="card"><div class="num">5</div><span>可辨车辆仍转人工<br>支撑面判断待修正</span></div><div class="card"><div class="num">68.0%</div><span>当前自动判定覆盖率<br>此前 79.4%；不是准确率</span></div></div><p>今天只读回查 37 个事件，全部复核完成；其中原批次 36 个事件、180 个候选的逐框结论均未变化。最新复测的 12 个候选中有 2 个属于原批次，去重总计 190 个候选。</p><p><b>本次知眸回调接入补齐原图保存与映射，没有更换模型权重。</b>最新两例发布前后结论相同，不能把此前批量改善都归因于这次接入。</p><p class="note">图像观察为 Codex 核查，非独立人工真值；本报告没有使用历史人工金标评分，也没有将模型一致当作准确率。</p></section>
<section id="latest"><h2>最新发布后实测 · 2 事件 / 12 候选</h2><div class="tablewrap"><table><thead><tr><th>事件 / 复核时间</th><th>输入证据</th><th>最终输出</th><th>可得结论</th></tr></thead><tbody><tr><td>Event-20260723105651-5<small>09-27 10:25:09</small></td><td>1440×1080 标注图 → 4032×3024 原图<small>2 框 cleanSource.applied=true</small></td><td>2 驳回</td><td>原图支持固定杆件，非人员<small>此前也驳回；验证原图链路，非新增纠错</small></td></tr><tr><td>Event-20260923180729<small>09-27 10:26:05</small></td><td>缺少可用清晰原图<small>继续旧单图模型</small></td><td>4 确认 / 6 驳回</td><td>兼容路径正常<small>未经本报告独立确认，不能用于原图改善结论</small></td></tr></tbody></table></div><p>缺原图事件框 1 至 10：假、真、真、假、真、假、假、假、真、假。这里的“真/假”均指模型输出。</p><p class="note">高清事件模型执行记录约 92.14 秒；仅一个事件，不代表平均耗时。画面匹配分数 0.99254 不是分类置信度。</p></section>
<section id="stats"><h2>前后统计 · 175 个配对候选</h2><div class="tablewrap"><table><thead><tr><th>指标</th><th>历史结果</th><th>改动后保存结果</th><th>变化</th></tr></thead><tbody><tr><td>自动确认</td><td>135 / 77.1%</td><td>112 / 64.0%</td><td>-23</td></tr><tr><td>自动驳回</td><td>4 / 2.3%</td><td>7 / 4.0%</td><td>+3</td></tr><tr><td>存疑 / 人工</td><td>36 / 20.6%</td><td>56 / 32.0%</td><td>+20</td></tr><tr><td>自动判定覆盖率</td><td>139/175 = 79.4%</td><td>119/175 = 68.0%</td><td>-11.4 个百分点</td></tr></tbody></table></div><p>全量 180 个结果：112 确认、7 驳回、61 人工。143 个有有效定位和图像观察的候选中，119 个图像支持目标，13 个支持非目标，11 个看不清；另外 37 个定位/匹配受阻。</p><p>7 个驳回均有非目标图像依据，其中 4 个历史上已驳回。38 个“确认 → 人工”中有 31 个因定位/匹配受阻，不能全部算成纠错。</p><p class="note">保存结果混合 v6/v7/v8，历史基线也不是统一的无原图版本。36 事件与训练事件重合 17 个、与验证事件重合 3 个，不能当作独立准确率测试。</p></section>
<section id="cases"><h2>典型图像对照</h2><p class="note">本次复看的典型案例。图像为核验展示裁切，源图未补画细节；黄框仅定位。点击图片可放大，展示缩放不会增加原图信息。</p>__CASES__</section>
<section id="details"><h2>180 条逐框结果与图像观察</h2><p>143 条可展开原图局部与上下文；37 条定位未通过，保留原因。此表使用此前冻结批次，今天确认逐框决策未变化；没有伪装成最新统一 v8 重跑。</p><div class="filters"><input id="search" aria-label="搜索事件、框、观察或理由" placeholder="搜索事件编号、杆件、车辆、理由…"><select id="decision" aria-label="按模型结果筛选"><option value="">全部模型结果</option><option value="confirmed">确认</option><option value="rejected">驳回</option><option value="uncertain">人工</option></select><select id="visual" aria-label="按图像观察筛选"><option value="">全部图像观察</option><option value="target">支持目标</option><option value="non_target">支持非目标</option><option value="indeterminate">无法判断</option><option value="not_located">定位未通过</option></select></div><p id="count" role="status">显示 180 / 180 条</p><div class="tablewrap"><table id="samples"><thead><tr><th>事件 / 初审类别</th><th>历史 → 当前模型结果</th><th>图像观察（非人工确认）</th><th>图像 / 模型理由</th></tr></thead><tbody>__ROWS__</tbody></table></div></section>
<section id="next"><h2>后续按这三个阶段推进</h2><ol><li><b>先完成可验收的来源链路。</b>用新自然知眸回调验证独立原图、初审图、WAR_TO_AI 关系、同帧与真实框，确认复核证据确实使用原图；不需创建飞行任务。</li><li><b>处理已有具体问题。</b>手动核对 4 个旧误确认、16 个新确认、7 个驳回和 5 个真车人工案例（有重叠）；优先修复 37 个定位受阻与车辆支撑面过度拦截。</li><li><b>做同批新事件的原图对照，再决定训练。</b>固定框、模型、提示词和策略，只改变是否给原图；分别记录误确认、误驳回、人工比例、原图生效比例和耗时。可靠确认后才形成训练样本。</li></ol><p>09-26 的来源快照为 358 事件中 36 对可下载关联图片，约 10.1%；这是历史覆盖率，尚未测量新回调接入后的覆盖改善。36 对中 33 对尺寸相同，3 对更高清但 1 对画面不匹配。</p><p><b>只有本批新事件的独立结果核验补齐后，才能报告准确率提升百分比。</b>当前可持续提供逐例改善与疑点，不把“看不清”强制标成真假。</p></section>
<footer><p>当前健康复查：Java UP / GPU v8 running、healthy。远端 test 60620f56cc；Java 功能分支 e61ae57a3e。本报告没有重启服务、改变策略、训练模型或再次触发业务复核。</p><a href="evidence.json" download>证据与版本 JSON</a> · <a href="报告说明.md" download>完整 Markdown 说明</a><p>页面离线读取本地文件，无外部图片或分析服务；原始业务响应与认证凭据不在公开报告目录内。</p></footer></main>
<dialog id="detail"><button class="close" id="close">关闭</button><h3 id="detailtitle"></h3><p class="note">左：原图局部；右：原图上下文，黄框仅定位。观察不等于人工确认。</p><img id="detailimage" alt="原图局部与上下文"><a id="imagelink" target="_blank" rel="noopener">在新页查看图片</a></dialog>
<script>
const rows=[...document.querySelectorAll('#samples tbody tr')];const search=document.querySelector('#search'),decision=document.querySelector('#decision'),visual=document.querySelector('#visual');
function filter(){let n=0;const s=search.value.trim().toLowerCase();for(const row of rows){const show=(!s||row.dataset.search.toLowerCase().includes(s))&&(!decision.value||row.dataset.decision===decision.value)&&(!visual.value||row.dataset.visual===visual.value);row.hidden=!show;n+=show;}document.querySelector('#count').textContent=`显示 ${n} / ${rows.length} 条`;}
search.addEventListener('input',filter);decision.addEventListener('change',filter);visual.addEventListener('change',filter);
const dialog=document.querySelector('#detail');document.querySelectorAll('.sample').forEach(button=>button.addEventListener('click',()=>{document.querySelector('#detailtitle').textContent=button.dataset.title;document.querySelector('#detailimage').src='assets/contacts/'+button.dataset.image;document.querySelector('#imagelink').href='assets/contacts/'+button.dataset.image;dialog.showModal();}));document.querySelector('#close').addEventListener('click',()=>dialog.close());dialog.addEventListener('click',e=>{if(e.target===dialog)dialog.close();});
</script></body></html>'''
    (out / 'index.html').write_text(template.replace('__CASES__', cases).replace('__ROWS__', ''.join(trs)))


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--contacts', type=Path, required=True)
    args = p.parse_args()
    data = json.loads((HERE / 'evidence.json').read_text())
    check_evidence(data)
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=True)
    assets = out / 'assets'
    contacts = assets / 'contacts'
    contacts.mkdir(parents=True, exist_ok=True)
    for name in ['table', 'vehicle', 'pole', 'roof', 'ground', 'rider']:
        shutil.copy2(PRIOR / 'assets' / (name + '.jpg'), assets / (name + '.jpg'))
    for row in data['candidates']:
        if row['visualId']:
            name = row['visualId'] + '.jpg'
            shutil.copy2(args.contacts / name, contacts / name)
    # 分发目录使用本地相对图片路径，仓库说明仍引用已有资产，避免重复提交源图。
    readme = (HERE / 'README.md').read_text().replace(
        '../ktd-original-recheck-assessment-20260926/assets/', 'assets/')
    (out / '报告说明.md').write_text(readme)
    shutil.copy2(HERE / 'evidence.json', out / 'evidence.json')
    (out / '逐框核验明细.jsonl').write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in data['candidates']))
    build_pdf(data, out)
    build_html(data, out)
    manifest = [{'file': str(f.relative_to(out)), 'sha256': hashlib.sha256(f.read_bytes()).hexdigest()}
                for f in sorted(out.rglob('*')) if f.is_file() and f.name != 'manifest.json']
    (out / 'manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({'output': str(out), 'pages': 6, 'candidates': 180, 'visualImages': 143}, ensure_ascii=False))


if __name__ == '__main__':
    main()
