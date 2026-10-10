"""Render the final evidence report only after successful audit and cleanup."""
import json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def read(n):return json.loads((ROOT/n).read_text())
def interval(v):return f'[{v[0]*100:+.2f}，{v[1]*100:+.2f}]'
def main():
    r=read('holdout-report.json');a=read('integrity-audit.json');cleanup=read('cleanup.json');sensitivity=read('paired-sensitivity.json');overlap=read('overlap-sensitivity.json')
    assert cleanup['server_directory_removed'] and cleanup['experimental_port_closed'] and cleanup['container_removed']
    assert a['calls']==348 and not any(g['passed'] for g in r['gates'])
    metrics={(m['mode'],m['card'],m['repeat']):m for m in r['metrics']};names={'p000':'p000 无卡','p001_p002':'p000＋原 p001/p002 卡','repair':'p000＋修订卡'}
    lines=['# 光斑新独立样本复验（2026-10-10）','', '**结论：两张卡都使截图准确率明显下降，未通过，不集成光斑卡，不部署。** 修订卡主要把真光斑改判为“看不清”，按现有仅 true 放行规则仍会被拒绝。123 条人工标注全部完整、全部为“是光斑”；本轮准确率等同于真光斑保留率，另因没有负例，不能验证非光斑误放是否增加。','', '29 张照片、3 个飞行任务；每个输入/卡组合独立两轮，共 348 个计划请求、1476 个框级判定。所有请求有效，失败尝试不作负例。','']
    for card in ['repair','p001_p002']:
        q=next(x for x in r['paired'] if x['mode']=='screenshot' and x['card']==card)
        base=sum(metrics['screenshot','p000',i]['accuracy'] for i in [1,2])/2;got=sum(metrics['screenshot',card,i]['accuracy'] for i in [1,2])/2
        lines.append(f"- {names[card]}：截图平均准确率 {base:.2%} → {got:.2%}，变化 {q['delta'][0]*100:+.2f} 个百分点；事件配对 95% 区间 {interval(q['ci95'][0])} 个百分点。")
    lines+=['','本轮无卡时原图平均准确率为 97.56%，截图为 82.11%；原图减截图为 +15.45 个百分点，事件配对 95% 区间 [+4.13，+29.82]。修订卡在原图上平均 95.93%，与无卡相比变化 -1.63 个百分点，区间跨零，不能认为加卡有收益。','', '因此后续应优先验证“原图＋无卡”的实际候选流程，而不是继续加严光斑卡。现有证据只覆盖人工重画的真光斑框，仍需非光斑和真实检测框验证；本次不实现或部署该切换。']
    lines+=['','## 两轮结果','','每轮分母均为 123 条人工框记录。“真光斑未放行”包含 false 和 uncertain；“看不清”与它重叠，不能相加。无负例时误放数量记为不可检验，不把机械计数 0 当成“误放未增加”。','','| 输入 | 方案 | 准确率① / ② | 真光斑未放行① / ② | 其中明确判假① / ② | 看不清① / ② | 非光斑误放 |','| --- | --- | ---: | ---: | ---: | ---: | --- |']
    for mode in ['screenshot','original']:
        for card in names:
            x,y=[metrics[mode,card,i] for i in [1,2]]
            lines.append(f"| {'截图' if mode=='screenshot' else '原图'} | {names[card]} | {x['accuracy']:.2%} / {y['accuracy']:.2%} | {x['false_reject']} / {y['false_reject']} | {x['explicit_false_reject']} / {y['explicit_false_reject']} | {x['uncertain']} / {y['uncertain']} | 不可检验（无负例） |")
    lines+=['','## 配对区间','','两次重复先在同一人工框上平均，再整组抽取事件；固定种子 20261010、20,000 次 bootstrap，报告框加权平均差和百分位 95% 区间。29 张截图哈希不同。附按 3 个飞行任务抽样的敏感性区间；只有 3 个批次，区间离散、覆盖有限，不能视为广泛场景验证。','','| 输入 | 加卡方案 | 准确率变化（百分点） | 按事件配对 95% 区间 | 按飞行任务配对 95% 区间 |','| --- | --- | ---: | --- | --- |']
    for q in r['paired']:
        t=next(x for x in sensitivity if x['comparison']==f"{q['mode']}:{q['card']}-p000" and x['cluster_field']=='task_id')
        lines.append(f"| {'截图' if q['mode']=='screenshot' else '原图'} | {names[q['card']]} | {q['delta'][0]*100:+.2f} | {interval(q['ci95'][0])} | {interval(t['ci95'][0])} |")
    lines+=['','`holdout-report.json` 还包含真光斑未放行、看不清比例变化的区间；非光斑误放的零值没有负例支撑，不可解读为误放率。`paired-sensitivity.json` 同时保存每种卡的原图减截图配对比较。','','## 重叠框与坐标核对','','完整性检查通过：123 条键唯一、标签合法、框有效、截图反投影一致；58 张源图哈希通过；无完全相同的重复框。标注 CSV 冻结哈希为 `e6aa7906c4af1b3a5b8cb2b4613008845cee092c86c631f1d839682f487ddab3`。','', '补查发现 88 条人工框不包含旧系统的定位点；50 对同图人工框交并比不小于 0.5，按重叠连通关系形成 84 组。这里只确认几何关系，没有据此宣称 84 个独立真实目标，也没有自动删框或修改人工判断。旧系统只存了位置点，本轮框是用户重画的，结果只对应这些实际手绘范围，不能冒充线上检测器原框效果。','', '主分析保留全部 123 条。读取模型判定内容前，另记录“重叠区域等权”敏感性方案：同组先平均框和轮次，每组占一份权重；不重新请求模型。结果如下：','','| 输入 | 加卡方案 | 区域等权准确率变化（百分点） | 事件配对 95% 区间 |','| --- | --- | ---: | --- |']
    for q in overlap['comparisons']:
        if q['cluster_field']=='event_id':lines.append(f"| {'截图' if q['mode']=='screenshot' else '原图'} | {names[q['card']]} | {q['delta'][0]*100:+.2f} | {interval(q['ci95'][0])} |")
    lines+=['','## 隔离、冻结与运行','','- 新样本来源、曝光排除、任务与附件对应见 [标注包](../hotspot-fresh-labeling-20261010/README.md)。29 对照片在开发阶段未提供给模型，也没有用于写卡。本次只运行这一批预定复验；两轮是事先规定的重复，不是看结果后重跑。','- 原 p001/p002 卡逐字相同，合并为一个候选。修订卡直接使用上一轮仅根据旧开发集冻结的 `repair`；旧初版 `revised` 已被后续修订替代，本轮不临时新增候选。修订卡是事先指定的主要比较，原卡为参考比较，不在本批结果中挑选赢家。','- 提示词直接调用指定生产提交 `b2ca913ee365c0a64ebda4e2ca28604b79d801ed` 的 `build_prompt`。该脚本 SHA-256 为 `3884f658862294bce0744ca170c5e1ee4324b24652eb5d1fa4d587171522060b`；提示词函数、源图/定位图/局部拼图生成代码逐字核对一致。没有扩大局部拼图、加第四张图或改变裁图规则。','- 基线保留 p000 common；卡只作为纯光斑候选的插卡条件。源代码、图片哈希、标签与方案在模型调用前冻结并推送，提交 `722a9a66d7450ad587154bdd75f142cdce63a21a`。补充几何统计方案在查看模型判定前记录，提交 `75e480f0c964ca15cc135b2c13ab9850c751ee5a`。','- 每图轮换组合顺序，第二轮反向，线程池及 worker 均最多 6 并发。实验仅监听 `127.0.0.1:18406`；Docker 4 GiB、无 swap、3 核、无 GPU。CLI 为指定 `codex-0.162.0-alpha.17.2/codex`，SHA-256 `73b73b048546993298234a295de01ef2424e7a8b29def9d0ecb1ca7cb77500d4`，请求模型 `gpt-5.6-luna`。',f"- 共 {r['transport_retries']} 次失败尝试后的重试，最终没有缺失结果。未完成尝试没有当成 false，也未重跑成功位置。未完成尝试的状态与每次耗时保存在 `runs/holdout/out/`。",f"- 资源记录采样峰值 {a['peak_memory_bytes']/2**20:.1f} MiB（含文件缓存），实验 swap 峰值 {a['maximum_swap_bytes']} 字节，主机最低可用内存 {a['minimum_host_available_fraction']:.2%}，主机换出计数变化 {a['host_swapout_delta']}。共享服务只读健康检查前后正常，没有改配置或重启。",'','## 复算与交付','','- `source/audit_results.py` 独立核对全部计划、唯一执行号、图片/提示词/框对应、每个指标与事件区间；`integrity-audit.json` 保存结论。','- 3 项计分保护测试通过：失败请求不算负例、重复执行号拒绝计分、缺少非光斑时即使准确率完美提升也不能判通过。','- `source/overlap_sensitivity.py` 计算重叠区域等权结果。补充方案和几何清单有独立哈希；主冻结文件始终未变。','- 不创建光斑服务集成分支，不改 `p000-pure-class-cards`，不部署。没有满足原门槛，不提供“可发布”的虚假结论；后续若继续验证误放，需要新的、明确包含非光斑的人工标签，不能由模型自动补成负例。','- 交付分支 `feature/hotspot-fresh-labeling`；中文提交并推送，最终远端 SHA 以交付消息及 Git 记录为准。','','## 资源清理清单','','- 本次容器 `agent-hotspot-fresh-eval` 已退出并自动删除；worker 和 CLI 子进程已结束；18406 无监听。','- 导出结果并逐文件核对哈希后，已删除服务器专属目录 `/data/workspace/jzheng/hotspot-fresh-eval-20261010`，包括图片副本、CLI 临时数据和私有运行目录；认证源文件未动。','- 本机标注服务及监护进程已停止，8802 无监听。','- 保留本分支工作树 `/Users/wangxiaolin/PycharmProjects/review-annotation-platform-hotspot-fresh-labeling`，其中唯一一份源图约 57.3 MiB，供人工复核；标签、来源清单、源代码和结果已推送。没有新增工作树或复制整仓库。','- 共享 18396、共享容器、已有镜像、旧标注包和其他工作树均未改动。清理证据见 `cleanup.json`、导出校验见 `export-verification.json`。','']
    (ROOT/'README.md').write_text('\n'.join(lines))
if __name__=='__main__':main()
