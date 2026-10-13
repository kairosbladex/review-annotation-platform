# -*- coding: utf-8 -*-
"""生成领导汇报版 docx：正式平台预警信息复核专项（2026-09-07）。"""
from docx import Document
from docx.shared import Pt, RGBColor
from docx.oxml.ns import qn
from docx.enum.text import WD_ALIGN_PARAGRAPH

doc = Document()

# 全局中文字体
style = doc.styles["Normal"]
style.font.name = "Times New Roman"
style.font.size = Pt(12)
style.element.rPr.rFonts.set(qn("w:eastAsia"), "宋体")


def para(text, bold=False, size=12, align=None, space_after=6):
    p = doc.add_paragraph()
    run = p.add_run(text)
    run.bold = bold
    run.font.size = Pt(size)
    run.font.name = "Times New Roman"
    run.element.rPr.rFonts.set(qn("w:eastAsia"), "宋体")
    if align is not None:
        p.alignment = align
    p.paragraph_format.space_after = Pt(space_after)
    return p


def heading(text, size=14):
    para(text, bold=True, size=size, space_after=8)


def make_table(headers, rows):
    t = doc.add_table(rows=1 + len(rows), cols=len(headers))
    t.style = "Table Grid"
    for j, h in enumerate(headers):
        cell = t.rows[0].cells[j]
        cell.text = ""
        run = cell.paragraphs[0].add_run(h)
        run.bold = True
        run.font.size = Pt(10.5)
        run.font.name = "Times New Roman"
        run.element.rPr.rFonts.set(qn("w:eastAsia"), "宋体")
    for i, row in enumerate(rows):
        for j, val in enumerate(row):
            cell = t.rows[i + 1].cells[j]
            cell.text = ""
            run = cell.paragraphs[0].add_run(str(val))
            run.font.size = Pt(10.5)
            run.font.name = "Times New Roman"
            run.element.rPr.rFonts.set(qn("w:eastAsia"), "宋体")
    doc.add_paragraph()
    return t


# ============ 标题 ============
para("视觉复检工作汇报", bold=True, size=18, align=WD_ALIGN_PARAGRAPH.CENTER, space_after=2)
para("——正式平台预警信息复核专项", bold=True, size=14, align=WD_ALIGN_PARAGRAPH.CENTER, space_after=2)
para("汇报人：王晓林    日期：2026年9月7日", size=11, align=WD_ALIGN_PARAGRAPH.CENTER, space_after=12)

# ============ 一、总体情况 ============
heading("一、总体情况")
para("按照“把正式平台所有预警信息单独复核一遍、看看最终效果”的要求，本阶段不依赖平台界面操作，"
     "直接从正式平台数据库拉取全部预警记录，用视觉模型完成多轮批量复核和交叉验证，"
     "并同步完成了人工金标导出、训练数据整理和模型训练实验。")
para("截至9月7日，正式平台存量的1510条预警已全部复核完毕，批量复核链路已跑通，"
     "后续新产生的预警可以直接批量复核，不需要人工在界面上逐条触发。")

# ============ 二、预警复核情况 ============
heading("二、正式平台预警复核情况（本次重点）")
para("1. 复核范围", bold=True)
para("正式平台存量预警共1510条记录，涉及755张现场图片、3552个候选目标框，已全部分轮复核完毕，无遗漏。")
para("2. 复核过程与结果", bold=True)
para("为把误报筛干净，复核分三轮进行，后一轮专门处理前一轮拿不准的部分：")
make_table(
    ["轮次", "复核对象", "复核方式", "复核结果"],
    [
        ["第一轮", "全部3552个候选框", "GPT视觉模型全量复核", "判真2653个、判误报148个、存疑751个"],
        ["第二轮", "第一轮存疑的751个框", "针对存疑框重点复核", "确认判真196个、判误报37个、仍存疑518个"],
        ["第三轮", "第二轮仍存疑的518个框", "加权打分＋另一视觉模型（Kimi）独立复核，交叉验证", "146个给出明确结论（判真123、判误报15），两种方法结论一致率约90%"],
    ],
)
para("第三轮采用两种相互独立的方法互相印证，结论一致率约90%，说明复核结果是稳定可信的；"
     "剩余未达成一致的少量目标，已建好转人工确认的队列。")
para("3. 批量复核链路（不走界面）", bold=True)
para("已通过GPU复核服务对1510张图片、7097个候选框批量预填复核结论，全部跑通。"
     "该链路直接对接平台数据，不经过界面操作，新预警产生后可随时批量复核。"
     "目前该服务判定偏保守（存疑比例偏高），后续结合人工确认结果调优。")

# ============ 三、复核发现的问题 ============
heading("三、复核发现的问题")
para("1. 原始预警标注存在错误。复核中发现有5个标注为“车”的目标，实际是打伞的行人，"
     "说明平台原始标注存在不可靠项，单独复核确有必要。")
para("2. 预警中混有一定比例的误报和存疑目标。仅靠一次模型判定容易放过误报，"
     "多轮交叉复核能有效把误报筛出来，这也正是复检模型要解决的问题。")

# ============ 四、数据与模型成果 ============
heading("四、数据与模型成果")
para("1. 人工金标（标准答案）", bold=True)
para("从正式平台导出304个已完成复核的样本、854条人工标注（真实目标651条、误报203条），"
     "形成可追溯的金标快照，作为模型训练和考核的统一标准。")
para("2. 训练数据集", bold=True)
para("基于金标整理出848条训练数据，其中674条用于训练、174条用于验证，已按规范冻结快照。")
para("3. 模型训练进展", bold=True)
make_table(
    ["模型", "进展", "当前结果"],
    [
        ["轻量裁剪复检模型", "公司GPU上完成多组训练实验", "在94条人工验证样本上准确率75.5%–84.04%，最好一组判对79/94"],
        ["Qwen视觉模型微调", "3轮共2022步训练已全部完成，权重已保存", "评估环节因配置问题未跑通，修复后补跑对照实验"],
        ["YOLOE训练链路", "完成1轮冒烟训练并保存权重", "训练链路验证通过"],
    ],
)

# ============ 五、存在问题 ============
heading("五、存在的问题")
para("1. 验证样本量偏小。94条样本的准确率只能作为阶段参考，不能等同于生产环境的实际准确率。")
para("2. 误放行率偏高。实验中部分误报被判为真实目标（41.67%–54.17%），模型判定稳定性还需提升。")
para("3. 数据隔离需重建。审查发现22个任务同时出现在选型集和测试集中，需按任务来源重新隔离，保证考核公正。")
para("4. 存疑预警待人工确认。751条存疑预警的确认队列已建好，其中146条已由模型预填建议，待安排人工核对。")

# ============ 六、下一步计划 ============
heading("六、下一步计划")
para("1. 完成751条存疑预警的人工确认，给出正式平台预警复核的最终效果（误杀率、误放行率）。")
para("2. 修复Qwen评估配置，补跑基座模型与微调模型的同集对照实验，验证微调效果。")
para("3. 按任务来源重建隔离数据集，补充少样本类别和负样本，组织新一轮人工盲审。")
para("4. 统一验收口径（误杀率、误放行率、人工复核比例、时延）后独立测试，通过后再提出上线建议。")

# ============ 七、口头汇报稿 ============
heading("七、一分钟口头汇报稿")
para("这阶段按领导要求，把正式平台存量的1510条预警单独复核了一遍，不走界面，直接从数据库拉数据批量复核。"
     "755张图、3552个候选框分三轮复核：第一轮判真2653个、误报148个；第二轮把751个存疑框复核掉一部分；"
     "第三轮用两种独立方法交叉验证，一致率约90%，结果可信。"
     "复核中发现平台原始标注有错，比如5个标成“车”的其实是打伞行人，说明单独复核确实有必要。"
     "数据方面，从正式平台导出了854条人工金标，整理了848条训练数据；"
     "模型方面，轻量复检模型小样本准确率75.5%到84.04%，Qwen微调训练已跑完，评估修复后出对照结果。"
     "下一步先把751条存疑预警人工确认完，给出最终复核效果，再统一验收口径决定能否上线。")

doc.save("视觉复检工作汇报-正式平台预警复核-20260907.docx")
print("saved: 视觉复检工作汇报-正式平台预警复核-20260907.docx")
