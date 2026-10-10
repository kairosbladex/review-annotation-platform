#!/usr/bin/env python3
"""将已完成结果转成中文报告；结论按预先固定的 holdout 门槛生成。"""
import argparse,json,datetime
from pathlib import Path

def render(root):
    r=json.loads((root/'report.json').read_text());p=json.loads((root/'protocol.json').read_text())
    complete=r['complete_images']==300 and r['requests']['success']==2400
    h=r['comparisons']['holdout'];building=h['D-C']['违建']['accuracy']
    vehicle=h['D-B']['车']['uncertain'];person=h['D-B']['人员']['false_acceptance']
    building_pass=complete and building['ci95_pp'] is not None and building['ci95_pp'][0]>0 and building['source_clusters']>1
    clean_pass=complete and vehicle['ci95_pp'] is not None and person['ci95_pp'] is not None and vehicle['ci95_pp'][1]<0 and person['ci95_pp'][1]<=0 and min(vehicle['source_clusters'],person['source_clusters'])>1
    def estimate(x):
        if x['delta_pp'] is None:return '缺少样本，无法验证'
        return f"{x['delta_pp']:+.2f} 个百分点（95%区间 {x['ci95_pp'][0]:+.2f}～{x['ci95_pp'][1]:+.2f}；{x['source_clusters']} 个原片组）"
    lines=['# Luna 固定 CLI：干净帧 × 违建卡对照（2026-10-10）','',
      '**发布判断：'+('违建卡在干净帧下仍显示收益。' if building_pass else '尚不能证明违建卡在干净帧下稳定有效。')+(' 干净帧通过车辆看不清减少、人员误放不增加的统计门槛；只能进入隔离链路验收，不能直接发布。' if clean_pass else '干净帧未同时通过车辆看不清减少、人员误放不增加的统计门槛，不建议据此上线干净帧组合。')+'**','',
      '- 违建卡在干净帧下的效果（D−C，holdout）：准确率 '+estimate(building)+'。',
      '- 已开启违建卡时改用干净帧（D−B，holdout）：车辆看不清率 '+estimate(vehicle)+'；人员非目标误放率 '+estimate(person)+'。',
      '- 未开启违建卡时改用干净帧（C−A，holdout）：车辆看不清率 '+estimate(h['C-A']['车']['uncertain'])+'；人员非目标误放率 '+estimate(h['C-A']['人员']['false_acceptance'])+'。','',
      '- 具体数量：holdout违建 C 为33.3% / 33.3%，D为97.0% / 97.0%；真违建驳回22 / 22→1 / 1，非违建误放仍为0 / 0。截图卡B为84.8% / 84.8%。',
      '- 车辆看不清：B的15 / 21→D的12 / 6，但A的9 / 16→C的20 / 8，两条对照不一致。人员非目标误放由A的4 / 5、B的3 / 5降到C/D的0 / 0；不过24个真人中C仍驳回21 / 21，D仍驳回20 / 20，不能把零误放解释为人员复核已经可用。',
      '- 原划分存在37个跨dev/holdout原片源。去掉重叠来源后，holdout车辆只剩3框、1个原片，违建只剩2框、1个原片；缺少足够独立来源样本，无法验证新场景泛化。',
      '- 发布建议：本次不建议直接上线C或D。B可作为优先考虑的纯违建卡候选，但本报告不构成发布授权；应结合任务④验收并补独立来源验证。B在dev的违建准确率只有60.4% / 62.5%，误放8 / 7个非违建（共21个），需保留人工复核。',
      '- A/B在非违建图的输入与提示词逐字一致，C/D同理；这些类别的组间差别反映重复推理波动，不能归因于违建卡。仅看D−B的车辆点估计会高估证据强度。','',
      '## 对照与证据范围','',
      'A＝截图三视图＋p000；B＝截图三视图＋p000-building-only；C＝干净原帧三视图＋p000；D＝干净原帧三视图＋p000-building-only。每组两次，同图的八个条件按固定种子 20261010 打乱后交错提交。先以并发 4 测量资源，随后升至用户允许的并发 6；只重启本次 worker/runner，保留已成功结果，未完成调用重新请求。并发调整记录见 concurrency-amendment.json。完成次序由响应时间决定。没有复用历史模型输出。固定种子用于图片/条件调度和bootstrap；未设置模型内部采样种子，不能承诺逐次输出完全一致。','',
      '截图三视图为截图、编号定位图、局部裁剪图；干净帧三视图为原帧、从原帧生成的编号定位图和局部裁剪图，不发送第四张图。截图候选使用截图坐标，原帧候选使用冻结投影坐标。提示词仅含 target_index、class_name、bbox，不发送人工标签、业务编号或原片路径。', '',
      '本实验“违建”按蓝色/绿色顶棚的业务外观口径计分，不表示图像能够判断审批或合法性。B/D 的卡与任务④提交 `42397238abe9e6fe544af826478f61757fb748ee` 逐字一致：保留 p000 common，仅纯违建图加入 p002 违建卡。混合图及非违建图与 A/C 提示词逐字一致。本批有 49 张纯违建图、没有违建混合图，不能验证混合图的卡片效果。','',
      '沿用原试验 300 张、dev 200 / holdout 100；原片重新恢复并逐张通过旧 clean_sha256 校验。启动时任务③尚未完成；执行期间复查发现其已产出更正清单（提交 5b9a8dea），因此最终计分采用更正后标签。3条更正均为 dev 的 false→true，3个框均在本次300张内（2个车辆、1个两轮车）；holdout标签不变。更正清单及 labels-used.json、label-audit.json 随报告保存。标签不进入模型请求，无需重跑已完成调用。未改变划分，也未按本次模型结果改标签或筛选图片。原试验排除的126张仍排除；这是已有可信原帧的子集，不能据此推断线上找帧成功率或全部事件效果。', '',
      f"完整配对图片 {r['complete_images']}/300；可计分框 {r['scored_candidates']}，无二元标签框 {r['unscored_candidates']}。成功请求 {r['requests']['success']}/2400；已落盘尝试状态（不含并发调整时未落盘的中断调用） `{json.dumps(r['requests']['attempt_status'],ensure_ascii=False)}`。失败不当作 false 或 uncertain；若配对不完整则不能通过发布门槛。",'',
      '固定 CLI：`'+p['binary']+'`；版本 `'+p['binary_version']+'`；SHA-256 `'+p['binary_sha256']+'`。每个请求前后校验哈希。请求模型 `gpt-5.6-luna`，reasoning=none，关闭工具、搜索、多代理和 fast_mode；服务端模型权重与路由未由本实验独立校验。','',
      '实际请求时间（北京时间）：'+datetime.datetime.fromtimestamp(r['requests']['start'],datetime.timezone(datetime.timedelta(hours=8))).isoformat()+' 至 '+datetime.datetime.fromtimestamp(r['requests']['finish'],datetime.timezone(datetime.timedelta(hours=8))).isoformat()+'。','',
      '## 各类别结果','',
      '准确率按现有业务规则计算：只有 true 放过，false 和 uncertain 都驳回。因此“真目标被驳回”包括看不清；“非目标被放过”只计输出 true。“看不清”主表只计有二元人工标签的候选；全部候选数量另见 report.json。表内为第一次 / 第二次，空样本标为无法验证。','']
    for split in ('dev','holdout'):
        lines+=['### '+split,'','| 类别（总/真/非） | 组 | 准确率①/② | 真目标驳回①/② | 非目标放过①/② | 看不清①/② |','|---|---|---:|---:|---:|---:|']
        for cls,metrics in r['splits'][split].items():
            for arm in 'ABCD':
                a,b=metrics[arm+'-1'],metrics[arm+'-2']
                if not a['n']:
                    if arm=='A':lines.append(f'| {cls}（0） | — | 无法验证 | — | — | — |')
                    continue
                lines.append(f"| {cls}（{a['n']}/{a['true_n']}/{a['false_n']}） | {arm} | {a['accuracy']:.1%} / {b['accuracy']:.1%} | {a['true_rejected']} / {b['true_rejected']} | {a['false_accepted']} / {b['false_accepted']} | {a['uncertain']} / {b['uncertain']} |")
        lines+=['']
    lines+=['## 按原片分组的配对置信区间','',
      '同一原视频或原照片下的所有框作为一组，先平均两次重复，再整组有放回抽样 2,000 次，固定种子 20261010，取百分位 95% 区间。不是把框或重复调用当成独立样本。准确率/看不清率分母为同一类别全部二元标签；误放率分母为非目标；驳回率分母为真目标。差值为后组减前组。所有类别、dev/holdout、五种对照的四类指标均在 report.json；下表摘录发布相关指标。多重比较未校正，属于探索性区间。两次重复仅反映本次波动，区间不能覆盖服务端模型全部随机性。','',
      '| 范围 | 对照 | 指标 | 平均差及95%区间 |','|---|---|---|---|']
    for split in ('dev','holdout'):
        for contrast in ('B-A','D-C','C-A','D-B','D-A'):
            for cls,metric,title in [('违建','accuracy','违建准确率'),('车','uncertain','车辆看不清率'),('人员','false_acceptance','人员非目标误放率')]:
                lines.append(f'| {split} | {contrast} | {title} | '+estimate(r['comparisons'][split][contrast][cls][metric])+' |')
    lines+=['',f"dev/holdout 共用 {len(r['cross_split_sources'])} 个原片源。保留原划分，但不能把 holdout 当作完全独立的新场景；剔除跨组原片后的敏感性结果见 report.json 的 holdout_without_cross_split_sources。单原片子组无法估计场景泛化，单列 single_cluster。",'',
      '## 上线前置条件及发布、回滚交接','',
      '本次只验证离线判断，不验证线上找帧成功率、接口时延、Java持久化、用户页面或真实任务。未部署共享服务、未调用业务保存/下发或设备。以下为交接步骤，需用户具体授权后才可实施。','',
      '1. 先复核本次门槛和跨组原片敏感性；未通过的组合不进入共享发布。截图违建卡 B 的收益单看 B−A，不能据此宣称 D 已获准。',
      '2. 干净帧依赖 yoloe `feature/video-clean-source`（本次核对头 `ebfcee3b`）和 Java `feature/recheck-video-clean-source`（本次核对头 `d93df44e55`），均未据本实验发布。Java 当前分支的 test 配置 `recheckVideoEnabled=false`；必须明确视频到达/等待策略，不能把“分支存在”当作功能已启用。',
      '3. yoloe 找帧分支原有路径是 YOLO/Qwen 原图复核；文档明确 Luna 开启时会绕过该路径。需先实现并验收找帧结果接入 Luna 的三视图输入、逐框投影与编号对应。不能只合两个分支或开启配置就宣称完成。原视频可访问、缓存资源有上限、匹配质量门槛、投影出画处理、超时与无视频回退都须实际验证。',
      '4. 在独立端口/进程/目录先验证干净三视图、纯违建卡开关、混合图回退、找帧失败回退与调用失败不写假标签。Java遵循开发分支→dev-wxl→test；生产与UAT由负责人处理。验证时禁止副本调度、消费、初始化写入、任务执行和自动重试；真实保存、派发和回写由用户操作。',
      '5. 获得共享发布授权后，记录现用制品、配置、提示词版本、CLI哈希和回滚指针；仅发布获批组合。仅启用卡时设置 `CODEX_REVIEW_PROMPT_VERSION=p000-building-only`；干净帧需要完整链路制品及独立开关，不能沿用本实验 worker 作为业务服务。',
      '6. 回滚卡：恢复 `p000` 及原 worker 制品/配置。回滚干净链路：恢复截图输入、原 Java 视频策略、原 API/worker 制品与配置；保留同一候选编号契约，核对失败状态与队列，禁止批量重放真实任务。只能在授权窗口切换/重启，回滚后分别核对进程、接口、页面及用户真实流程。','',
      '## 资源与清理','',
      '只使用临时 loopback 18400，MemoryMax=4G、MemorySwapMax=0、CPUQuota=300%、并发不超过6。用户 cgroup 未委派 CPU 控制器，CPU 另以 taskset 绑定三个逻辑核（61–63）形成实际上限。准备进程2G/1核，runner 1G/1核，监护256M/1核，均禁止swap。不使用GPU。worker每5秒检查可用内存，低于15%杀掉自己的CLI；外部监护负责停自己的worker/runner。', '',
      '资源清理清单：本次prepare、worker、runner、watch进程均已停止，18400端口已关闭；服务器临时目录已删除。worker峰值约1.42 GiB，swap始终0；整机最低可用内存42.5%，未触发15%门槛。监测到的共享API容器全程running healthy、重启次数0。GPU未申请。',
      '保留内容：最终结果、纠正标签和必要运行证据，已核对2417个服务器证据文件SHA256后清理。现有CLI、认证目录和原始快照属于复用资源，未删除。详细清单与目录大小见 cleanup.json、process-cleanup.json；本次本地工作树将在提交推送核验后移除，实际状态见交付回复。','',
      '## 复算','',
      '```sh','python3 -m unittest discover -s scripts -p test_luna_clean_vs_card.py','python3 scripts/score_luna_clean_vs_card.py --evidence data/luna-clean-vs-card-20261010 --baseline data/luna-clean-frame-eval-20261009','python3 scripts/render_luna_clean_vs_card.py data/luna-clean-vs-card-20261010','```','',
      '原始八条件结果在 results/，逐框配对在 per-box.json；输入恢复清单、固定顺序、协议与资源限制均随报告保存。']
    (root/'README.md').write_text('\n'.join(lines)+'\n')
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('root',type=Path);render(p.parse_args().root)
