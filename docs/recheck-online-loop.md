# 复检最小闭环

## 授权记录

用户于 2026-09-30 在对话中同意：
- **数据外发**：复检链路和离线评测需要的事件图片（全图、目标局部、环境图，画面中可能出现人员或车牌），只能发往已配置的中转站（`AUTO_ANNOTATION_BASE_URL`），只能用于复检判断和提示词评测。不得发往其他接收方，不包括批量导出或上传原始数据集。
- 本授权不包括：发布到共享环境、修改共享测试环境的推送配置、充值或切换付费账号。

## 第0步证据

12个开发回归样本精确定位并映射，详见统一金标 regression_exposure.jsonl。
重冻前车类锁定框196真/34假/8看不清；重冻后218真/46假/12看不清。
符合条件的含false组31/44（70.45%），无false组72/143（50.35%）。
冲突5框仍隔离；未调用模型、未上传图片、未写共享数据库。

## 本轮交付结论（2026-09-30）

第0步重冻已完成，新的GPT建议链路、人工确认入口和离线评测脚本已实现，并在本机独立实例验证。功能开关保持默认关闭，未部署共享环境、未修改共享推送配置、未执行真实任务或设备操作。

有三个明确缺口：补标包仅交付120/200框；屋顶停车场缺少独立确认的验收样本；基线只评测开发集24框及锁定集64框，不是全量冻结集。当前基线出现真车判假的错误，**未通过真目标保护门槛**。代码可供人工辅助验收，不据此批准自动拦截、上线或改提示词。

## 第0步来源、分层与补标包

12框采用精确定位路径，没有使用宽泛曝光兜底。会话证据：
`/Users/wangxiaolin/.codex/sessions/2026/09/30/rollout-2026-09-30T14-13-57-01a0f0f2-c708-7932-a95e-62c05f4206e7.jsonl:362` 和 `:667`。
R045～R056通过原始候选ID映射到gold_id，所在三个完整组标记GPT曝光并进入开发集。映射和出处见 `data/recheck-gold-consolidated-20260930/regression_exposure.jsonl`。会话仅只读查询。

重冻原因原文：修正小层向下取整偏差 + 12 样本曝光核对，评测前重冻。
分层为主类别（框数最多，平局按字典序）×主population×是否含false；最大余数法分别分配含false组约70%、无false组约50%，种子20260930。GPT曝光、历史未知、uncertain-only组不参加锁定分配；冲突5框继续隔离。
对照 `bba595e`：车类锁定196真/34假/8看不清 → **218真/46假/12看不清**，含真79组、含假22组、共93组（真/假组可能重叠）。各population框数和组数详见 refreeze_comparison.json 和统一金标README。

|车类锁定来源|真框/真组|假框/假组|总组|
|---|---|---|---|
|random_traffic|76/55|6/6|60|
|released_by_recheck|93/13|9/2|14|
|rooftop_hard_case|59/28|35/19|44|
|historical_review|0/0|0/0|0|

同一框/组可具有多个population，上表不可直接相加。冻结后已登记首次基线使用，不能再无记录地重冻。

补标包的可持续路径：`/Users/wangxiaolin/PycharmProjects/review-annotation-platform/data/recheck-labeling-batch-20260930/`。
计划80楼顶/80复检已放过/40随机；实际80/0/40。只读查询283个有复检结果的事件，并盘点2748个pool候选及旧高清包。旧放过高清包剩余目标没有可用的新高清来源，未复用已标框。排除无核验高清1002框、金标或包内重复549框、原图范围外45框。
按历史误检比例，预计新增约28个车类假目标；锁定集新增约20个只是上限估算，已有组保留原切分会进一步减少。**本批不能保证一次补到100个锁定假目标**。预计45～90分钟。图片只在本机保存或以只读链接复用，未提交图片。

```sh
python3 /Users/wangxiaolin/PycharmProjects/review-annotation-platform/data/recheck-labeling-batch-20260930/label_server.py --port 8796
```

页面只监听 `127.0.0.1:8796`，有真目标/误检/看不清，无模型建议。标签保存到本地annotations.csv。标注结束后先按清单几何、图像哈希、确认人和时间转成人工来源JSONL，再用 append_recheck_gold.py 追加；已有组不重切，新组使用同一分层和最大余数法。GPT历史未核对的新组保守留开发集；只有显式核对输入历史并声明gpt_history_unknown=false，才能参与独立锁定分配。重复或冲突、连接两个既有冻结组、GPT曝光触及锁定组均需人工裁决，不默默改测试集。平台看过GPT建议后确认的新金标已曝光，只进入开发集，不能补充独立GPT锁定集。

## 记录、派发与额度

新增独立表 `recheck_records`，不使用会写人工标签的自动标注任务。候选ID+复检配置哈希唯一；配置哈希包含请求模型、提示词内容哈希及提示词/规则/裁图版本。版本不匹配的旧任务跳过，不自动重跑历史。

记录包含事件/候选/图片哈希/类别/provider、请求及实际模型、三个版本、状态、结论、依据、场景、图片局限、耗时、用量、错误、响应ID、尝试次数和时间戳。数据库约束只允许done有true/false/uncertain；queued/running/waiting_quota/failed/skipped的结论必须为空。接口错误不能变成uncertain或false。

开关开启后新事件保存候选和待处理记录，提交成功后通过HTTP后台任务派发Celery `recheck` 队列；不等待模型，不让消息服务故障阻塞接入。投递失败留下queued记录，使用显式enqueue-due恢复。worker用数据库条件更新领取，包含额度查询也不会被重复投递抢走。只处理车/公交车/卡车/三轮车/两轮车/人员；其他类别原因“无判定标准”。人工已确认目标或事件已标记人工处理时跳过。

调用前查已配置中转站 `/v1/usage`，缓存60秒；账号未有效、认证/查询失败、余额低于保底线均等待额度。默认保底**5 USD**，作为少量请求及一次重试的缓冲，不是精确报价或充值要求。只配置GPT，无Grok/Kimi备用。本账号要求官方CLI，实际调用通过只监听127.0.0.1的一次性桥接转发到唯一已配置中转站；不跟随重定向，不记录鉴权或图片请求体，不使用其他接收方。

429按Retry-After等待，最多两次模型请求。超时、断连、进程中途退出等执行不明情况存failed且不自动重发；明确确认旧worker已停止后用恢复命令登记为执行不明。这样能证明本地不重复发送，**不能保证供应商在未知请求后没有扣费**。认证失败等待额度，其他接口失败结论为空。

干净原图须有与事件相符的clean_event_id、目标在干净图上的clean_bbox和核验哈希，优先使用其原图、定位图、环境/局部和辅助遮挡图。只有截图时明确记录可能有红框遮挡；不会把路径含clean当作已验证干净来源。首次提示词只给类别和位置，不给初审结论。

## 页面与人工确认

事件页并列展示初审/GPT建议/人工结论，附模型、版本、依据和图片局限；可查看定位框，并在有已关联干净原图时切换查看。待确认入口优先分歧和uncertain，另以稳定随机哈希抽取默认10%一致目标。列表排除已确认人工标签，使用最新版本记录。

点击人工真/假确认走原有人工标注接口，保存确认人、时间、revision；审计记录追加前值和新值。GPT只写建议表，不能形成金标。 export_recheck_human_gold.py 只导出有对应当前revision人工审计的真实人工标注。验收中的人工点击均为独立SQLite模拟，未追加正式金标、未代表用户处理真实事件。

## 独立实例验收证据

|检查|结果及证据|
|---|---|
|重复HTTP推送/重复Celery投递|通过；同一候选配置只一条记录、模拟接口只调用一次；local-acceptance/acceptance.json|
|低余额/认证失败/usage不可用|通过；waiting_quota、结论空、零模型调用；同上|
|429/超时|通过；429两次后failed，超时一次failed，不变成uncertain；同上|
|worker中途SIGKILL再重启|通过；本机真实worker退出，显式恢复执行不明，不重发；模拟接口调用一次；同上|
|不支持类别/人工已处理|通过；skipped，零调用；真实人工确认前置跳过另有test_online_recheck.py测试|
|图片错配|通过明确事件ID不一致、哈希错误的拒绝；不声称识别所有语义错配|
|GPT与金标隔离、人工修订和导出|通过本地模拟；未人工确认时无Annotation，修订保存前值；导出1条人工确认|
|浏览器三栏展示、定位框/原图|通过，本地独立Chrome截图detail.png；截图留本机，不推送到Git|
|真正中转站链路|通过；live-chain/result.json：真实API→专用Celery→gpt-5.6-luna，返回false，重复推送200，attempts=1，人工标签为空|
|屋顶设备/遮挡/小目标/红框/无干净原图|有样本，详见scene-coverage.json；屋顶设备误判真、遮挡真车误判假，属于检查发现问题|
|屋顶停车场|缺少独立确认样本，暂时无法验证|

故障表使用注入的接口验证执行状态和调用次数，不制造真实账号故障、不冒充实际扣费验证。真实调用和故障模拟证据分别保留。全部只监听127.0.0.1，独立SQLite、Redis、专用worker；共享库写入0、共享配置变更0、设备动作0。

本机资源：API512MB、worker1024MB、Redis128MB（内部maxmemory64MB）、前端512MB，分别限制约1CPU；浏览器1024MB/1CPU。macOS使用任务进程组资源守护、CPU时间上限和CPU节流，0.1秒采样RSS，超过上限停止自己的进程；不是Linux内核cgroup，瞬时峰值受采样延迟限制。没有GPU、没有Docker、没有共享服务重启。

## 模型核验与基线

preflight.json：请求gpt-5.6-luna返回实际同名，请求gpt-5.6-sol返回实际同名；usage账号active/isValid、USD余额2721.65724544（当时快照，不是现在余额）。选现有配置同名的luna。
基线来自现有平台GPT提示词 `backend/app/services/auto_annotation_prompts.py::render_batch_prompt`，沿用三角色投票与保守合并，不是独立Qwen服务的提示词。本轮未更改基线内容或挑选新提示词。

开发集24框/22组，锁定集64框/41组，均完成，实际模型始终luna。分层限量抽样保证类别/来源覆盖，**不是全量冻结集，不代表线上总体比例**；缺少可访问原图的开发132框、锁定55框另列清单。锁定集没有人员样本，不能声称人员已通过锁定验收。曝光回归12框全部只在开发组。冻结测试使用前后已写test_usage.jsonl，不再重冻。

|车类唯一框统计|开发集|锁定集|
|---|---|---|
|评测框/组|9/9|50/32|
|真车判假|0/5，95%CI 0～43.45%|3/28，10.71%，95%CI 3.71～27.20%|
|假车判假|1/3，95%CI 6.15～79.23%|8/20，40%，95%CI 21.88～61.34%|
|uncertain|1/9|3/50|

每类别、每population的分子/分母、Wilson95区间、框/组数、用量及耗时见两个report.json。同一个框可属于多个population，不能把各population行直接相加。Wilson按框假设独立，同组相关，不能把窄区间当作保证。默认预登记真判假门槛Wilson95上界≤1%；当前不通过。假目标判假在完整锁定false达100前只报告数字。

开发集337.4秒/434603 tokens；锁定931.5秒/1316395 tokens。开发预算10USD、锁定20USD，调用前估算分别3.6/9.6USD；账户即时余额差额约0.0367/0.1457USD，**计费延迟，不是最终费用或费用上界**。开发首轮有限24次调用使用即时余额护栏，后续脚本已增加累计请求预留，防止延迟记账绕过预算。预留是保守护栏，并非供应商硬性单请求价格上限，超额收费无法回溯撤销，检测超额立即停。

A/B脚本支持同批交替顺序、同一时段调用、逐条实际模型名、模型变化警告、精确双侧McNemar。仅本轮单版本基线，所以没有实际A/B比较，McNemar不适用；这不是“A/B候选胜出”。开发候选进入锁定评测须提供明确开发胜出证据；本轮基线是用户明确授权的参照例外。脚本从不自动选择或发布提示词。

## 配置清单

|配置|默认/约束|
|---|---|
|RECHECK_ENABLED|false|
|RECHECK_PROMPT_VERSION|online-baseline-v1，仅已注册版本|
|RECHECK_RULE_VERSION|conservative-v1|
|RECHECK_CROP_VERSION|clean-three-view-v1|
|RECHECK_QUOTA_FLOOR|5，单位USD|
|RECHECK_AGREEMENT_SAMPLE_RATE|0.1（10%）|
|AUTO_ANNOTATION_MODEL|gpt-5.6-luna，记录实际响应模型|
|AUTO_ANNOTATION_BASE_URL/API_KEY|仅复用现有配置；密钥不进Git|
|AUTO_ANNOTATION_USE_CODEX_CLI|true，本账号官方CLI路径|
|AUTO_ANNOTATION_TIMEOUT_SECONDS/VERIFY_TLS|120/true|
|LOCAL_READONLY_IMAGE_ROOT|默认空；本机验收显式设主检出data，只允许独立SQLite且MinIO关闭|
|DATABASE_URL/REDIS_URL|验收显式覆盖为独立SQLite及127.0.0.1独立Redis|
|VITE_API_PROXY_TARGET|只在本机预览覆盖；默认沿用现有8190代理|

不得把新版本字符串改配置就当成新提示词；新增版本须注册实现、做开发验证，并通过显式批次运行历史候选。

## 操作、回滚与合并边界

使用主检出已有Python虚拟环境及frontend/node_modules只读链接，无重复安装依赖。独立故障验收（运行完自动停止）：

```sh
PY=/Users/wangxiaolin/PycharmProjects/review-annotation-platform/.venv/bin/python
cd /Users/wangxiaolin/PycharmProjects/review-annotation-platform-recheck-online-loop
$PY scripts/accept_recheck_local.py --python "$PY" --source-root /Users/wangxiaolin/PycharmProjects/review-annotation-platform --out data/recheck-online-evidence-20260930/local-acceptance
```

脚本拒绝覆盖已有验收数据库；再次验收须只清理自己创建且证据已保留的 `.tmp/local-acceptance`，不删除其他人的文件。
人工金标导出和追加：

```sh
$PY scripts/export_recheck_human_gold.py --out .tmp/human-confirmed.jsonl
$PY scripts/append_recheck_gold.py --out data/recheck-gold-consolidated-20260930 --incoming .tmp/human-confirmed.jsonl
```

以上需要显式指定独立数据库环境；不默认对共享库运行。追加要求人工来源，拒绝模型标签和冲突。显式补派使用 manage_recheck_batch.py enqueue-due；异常恢复使用 recover-interrupted --before ISO时间 --confirm-old-worker-stopped；历史新版本批次使用 explicit-batch --candidate-ids 本地ID清单 --batch-id 批次名。恢复不会重新发送failed未知请求。严禁在未确认旧worker停止时恢复running任务。

回滚：关闭RECHECK_ENABLED、停止本实例专用recheck worker，保留建议记录及人工历史；回到此前已确认提交并重新构建个人实例。新增表的迁移down_revision=0004_agent_review_fingerprint，downgrade保留付费推理证据，不删表、不改人工标签。只在独立SQLite验证迁移；未验证实际共享PostgreSQL迁移或部署。共享发布及回滚必须用户手动执行。

建议合并顺序：feature/ktd-recheck-training → feature/recheck-gold-consolidation → feature/recheck-online-loop → master，由用户决定。本次不合master，不动dev-wxl/test/uat/prod。master基线e104444未前进。

## 测试和资源清理

新增及相关回归见 tests.json。完整后端收集受到本机缺少torch限制；未安装大依赖。两个既有audit测试缺少source_object_key，已在gold分支重现同样失败，本次不修改无关审计实现或夹带修复。PostgreSQL专用测试因未配置独立PostgreSQL跳过，不能说已验收。

所有本次API、worker、Redis、前端、浏览器和CLI临时桥接已停止；8197/8198/6397/6398/5197已释放。临时数据库/中间脚本/日志清理，保留最终报告、使用日志及本机截图、一个供用户检查的独立真实回放SQLite；没有GPU占用、没有共享容器或服务变更。
补标包在主检出保留；gold工作树提交推送后移除；online工作树保留供用户审阅和本机验收，不保留后台服务。详见资源清单resources.json。


用户本机预览（仅独立真实回放SQLite，GPT开关关闭，没有worker）：

```sh
cd /Users/wangxiaolin/PycharmProjects/review-annotation-platform-recheck-online-loop
/Users/wangxiaolin/PycharmProjects/review-annotation-platform/.venv/bin/python scripts/preview_recheck_local.py --source-root /Users/wangxiaolin/PycharmProjects/review-annotation-platform
```

打开 http://127.0.0.1:5197 ，本地演示账号 local-live-admin / local-live-password（只属于本次独立库，不是共享账号）。每个服务512MB/1CPU，上限1小时；Ctrl-C停止本脚本及两个服务。无真实任务派发，无模型调用；页面操作只修改独立库。真实建议截图 live-chain/detail.png 已在浏览器检查，显示初审真/GPT假/人工待判断。
