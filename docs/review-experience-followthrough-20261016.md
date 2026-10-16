# 三功能后续核验与复检显示修复（2026-10-16）

本轮完成两处确定性的复检显示修复，并发现两个会改变下一步安排的事实：已核验的测试 Java 检索类尚未包含已知 `9381` 终态恢复修复，其他恢复实现未审计；最近两天已有 80 个人员初审框，不能再沿用“没有新人员框”的旧快照。整体目标尚未完成，当前证据不足以发布准确率候选或覆盖共享前端/Java。

恢复目标后的追加核验发现：共享 Java 和前端随后都有更新，上午的整包/静态资源身份已过时。最新边界见文末追加记录及 `docs/evidence/review-resume-current-state-20261016.json`；已有本地修复和测试结论不据此冒充已部署。

## 任务卡

- 目标：推进三功能真实体验与测试发布，核对剩余端到端等待、人员数据和发布条件。
- 验收：原始存疑不再冒充明确假目标；完整保留业务动作、原结果与几何；新来源有固定窗口和分母；部署判断绑定实际运行包。
- 非目标：重标历史真值、修改 only_true_is_true_v1、执行真实复检/检索或设备任务、用本地旧整包覆盖共享站点。
- 输入：当前本地已验证分支、实际测试容器/Java 包、只读测试库的 2026-10-14 至 2026-10-16 元数据、旧曝光分组清单。
- 输出：最小代码与 TDD、本文和脱敏证据；业务元数据仅留本机 0700 临时目录。
- 身份与模式：本机当前用户，已有 `ai-lingxi`/`gis-backend` SSH 身份；本地候选修改与只读核验。保持用户“先本地调试”的阶段，本轮不推送、不部署。
- 预算与副作用：模型调用、模型费用、业务写入、图片下载和标签写入均为 0；没有重启共享服务或执行发布脚本。
- 人工节点：发布授权已经具备；仍需恢复仓库/发布机访问并核对在役集成来源。新候选真值须由人工盲标和独立争议裁决，不能由模型生成。

## 显示修复与 TDD

本仓分支 `bugfix/review-uncertain-display` 从干净的 `b752d0ef83da874f7c5c05e9f7b95d9f4640c447` 创建，位于 `/Users/wangxiaolin/.codex/worktrees/review-uncertain-display/review-annotation-platform`。原 review 工作区及数据未改。

当前运行的 Luna 适配器保留 `rawDecision`，但旧 `decision=rejected` 与 `summary.falseCount` 同时包含 false 和 uncertain。本地历史对照台以前只读 `decision`，把“无法判断”显示为“假目标”。`backend/app/recheck_workbench.py` 仅新增 5 行：有效顶层原始判断优先；缺失/null 时读取 `codexReview.rawDecision`；没有有效原始判断时兼容旧值，不从理由自由文本猜测结论。

通过真实 `GET /workbench-api/events/{eid}` 在 MySQL 连接/游标边界提供合成记录，先红测 24 项中 10 失败/14 通过，再绿测 24/24。主 Agent 独立复跑新测试及既有适配器契约测试共 **29/29**，只有一个依赖弃用警告。原始 `rawResult`、summary、businessAction、记录顺序、类别和框坐标不变；非法事件 ID 为 400，数据库失败为 503，不生成真假结果。继承环境清空、只注入合成配置、DATA 为临时目录，测试禁止外部 DNS/connect。该证明属于展示接口契约，不是业务准确率或真实数据库回写。

```sh
/usr/bin/env -i PATH=/usr/bin:/bin PYTHONDONTWRITEBYTECODE=1 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 /Users/wangxiaolin/PycharmProjects/review-annotation-platform/.venv/bin/python -m pytest -p no:cacheprovider backend/tests/test_recheck_workbench.py backend/tests/test_adapter_contract.py -q
```

原始红绿 JUnit 留在 `/tmp/review-uncertain-tdd.LwdGaj/`。语法与 `git diff --check` 通过；没有新增依赖。

AFMAP 的真实业务组件只有 `typeStats` 汇总，无法反推出 uncertain 数量，因此只把“假目标”改为“未通过/复核未通过”，提示可能包含证据不足、请人工确认，框上的文字同步修改。所有计数、框、props/事件名、模型和业务判断保持不变。

该前端修复在 `/Users/wangxiaolin/PycharmProjects/onegismap_a-model-review-local` 的 `bugfix/model-review-verdict-label@76e2cd8b469e66c1650217492e538b0866ebec66`，基线 `09b3cc74e5354575527868a0dd17f78bf8eba0b1`；未合入 3081 主线入口。真实组件浏览器 TDD 先 1 失败/2 通过，再 3/3，主 Agent 独立复跑；`npm run build:test` 通过（1m29s）。`AiReviewResult.vue` 原有 19 个格式 lint 错误的行号/规则/严重度/消息完全相同，测试文件 lint 0/0；不宣称全量 lint 通过。画布绘制/显隐有像素测试，中文画布文字没有单独 OCR 断言。

## 新人员来源：有候选，尚无独立金标

只读一致性事务固定窗口 `[2026-10-14 00:00:00, 2026-10-16 00:00:00)`，限定当前测试项目，查询上限 2000 事件/20 秒，最后 rollback。09:37:20 快照共 **310 事件、620 附件**；其中人员为 **52 事件、80 个符合 x/y/width/height/frame_width/frame_height 契约的框、12 个任务**。

- 与旧清单 111 个任务的交集为 0/12；对子集旧人员来源、历史事件也未发现标识重叠。仅证明已核对标识不重叠，不能排除改名/不同路径的同图。
- 37/52 事件的 PRIMARYIMG 与 SHOTIMG 路径相同，15/52 不同；后续只读关系查询证实 15 个事件有 WAR_TO_AI 关联，但不同路径/关联本身尚未完成可信原帧绑定审计。
- 104 条附件中有 66 个唯一路径；一个 PRIMARYIMG 路径被两个事件复用，两事件的候选列表相同，涉及 2/80 框。未按图片内容确认，不能静默删去后改变分母。
- 新候选中的 frame/PTS 与 stream 身份字段为 0/80；没有图片 SHA-256，也没有新增人工真值。52 个事件的图片级与事件级已保存复检结果均为 0。

因此“最近没有新人员初审框”的旧材料已过时，但“80 框全部是可信独立金标”仍不成立。本轮不把它们写入 `eligible`、不填造 `source_evidence`、不将看不清当负例。下一轮应先对这 12 个新任务完成图片去重和来源绑定，再盲标真/假/看不清；与旧 443 框回归单列。

私有元数据、只读脚本和摘要位于 `/tmp/review-source-refresh.e0jSOo/`，原记录不提交 Git。主 Agent独立复核了 52/80/12、旧任务交集、37/15 路径比例和重复路径分母。

## 上午核验：Java 终态恢复尚未发布

测试算法容器仍是 `e2b673da…`、镜像 `dac104a6…`；其复检适配器 SHA-256 为 `dd24a7f215b0903c395d6f9b3bf56eed1c71d446e9a44eee3157025e049e8179`，实际 routes 为 `a56fb3bc6e9fea5ef3049960d1270052e4d2beeaf3a3903b76fa1aef4181bf00`。算法提速发布不证明 Java 已更新。

`gis-backend` 测试 Java PID 48486 运行 `/data/ktd/project/ktd-backend/biz-visualization.jar`，包 SHA-256 为 `3280e60610b5a0f92ccb2be51c7381a40cebd3ebcf4b254d618c0c994769a270`。jar mtime/ctime 早于进程启动约 1.236 秒，进程 fd 5/6 指向同一 inode；主 Agent 独立核对唯一 `.jar` 参数与文件身份，未见 classpath/javaagent 覆盖。不能简单取 `-jar` 后紧邻参数，因为该启动命令在其后仍有 JVM 选项。

实际 `AiSearchAsyncService2.class` 摘要为 `ab7dbc7a5dce69dda2ada70313251f806e5df12466894419a8ac66e42b2f7b17`，缺少 `recoverPendingTasks`、`reconcileTask`、`selectTaskForUpdate`、`publishTerminalProgress`、`queryRemoteJob`。能够证明当前包未包含 `9381d620` 的恢复修复；包内没有 Git/build 元数据，不能反推完整源码 SHA。

既有 1 条完成任务的日志显示：算法 11:48:03.461 完成 17/17，Java 在 11:48:05 接收并记录批量保存/status=1/Redis completed。按日志精度，接收相差约 1.5–2.5 秒；这不是分钟级等待的证明，也没有独立核对该记录的数据库提交或页面首次完成时刻。末帧推理、最后上传、页面完成仍缺对应时间点。Java 使用 5 秒 HTTP 轮询，相关 MQTT 订阅被注释，不能将缺口归因于 MQTT。

`9381` 跨两个 Java 模块的 7 个生产文件及 1 个测试，包含 Mapper 行锁 XML和新的恢复执行器；已有 Surefire 19/19 属历史本地证据，本轮未重复执行。扫描默认每 30 秒处理全体满足条件的运行中任务，每批 50 个；当前待恢复名单未核实。不能仅替换一个 class、直接覆盖旧整包，或声称只影响此前两条任务。

## 发布障碍与交付边界

前端 test/release 脚本声明目标为 `zhuoyi@192.168.110.17:/home/zhuoyi/ktd/web/amap/`，但一次 BatchMode 只读检查返回认证拒绝，未取得在役 manifest/index 摘要。当前主线候选与缓存 test 相差 949 个路径，包含重叠修改；脚本会整包替换且缺来源校验/统一失败中止，当前不能直接执行。

Java 仓库的 HTTP 配置先因没有非交互凭据停止；核对 URL 重写后，使用相同现有 SSH 身份、显式 `ssh://` 访问正确的 `ktd_backend` 仓库，得到 GitLab 明确的 `You are not allowed to download code from this project`。没有切换用户、修改 origin、重试前端仓库或索取其他凭据。当前无法刷新 prod/dev-wxl/test 并证明共享集成来源。

本轮代码只保留本地提交，未改变现有 8189 发布、共享前端、Java、旧任务状态或标签。停止本地候选服务/切回对应基线即可回退本轮显示修改，无业务数据需要回滚。发布下一步依赖仓库及既定发布机访问恢复、正确集成基线、待恢复任务范围和精确回滚包；准确率提升另依赖上述来源绑定及独立人工真值。原完整目标保持未完成。

## 恢复目标后的追加核验

沿用原有身份分别只读检查两个 GitLab 仓库各一次，仍被拒绝读取；前端发布机只检查一次，仍被拒绝 SSH 认证。没有切换身份、修改 origin、覆盖部署或重复已通过的工程测试。

共享环境已经由其他发布更新：测试 Java 改为 PID 91454，启动于 11:46:16 +08，整包 SHA-256 为 `ab5beba1898275cde6b0befc463f74f86ea5e1acf24a5a068c8a751ae897eb59`；主 Agent 核对了两个打开的文件描述符与当前包相同、文件更新时间早于进程启动。检索服务类仍与上午逐字节相同，未包含已知 `9381` 修复。没有据此排除其他类中存在不同恢复实现，也没有把旧整包当作当前发布/回滚基线。

公开前端入口改为 `index-DsaWFQOV.js`，HTML SHA-256 为 `6533252883b3ccd4df90e595c255b87eaab00e0831356eaad6bda68f3d88ed92`；主 Agent 独立核验了入口字节与资源身份。静态检查尚未取得本任务检索修复已上线的证据，不能绑定部署源码 SHA，也不能替代浏览器验收。

为补齐已有完成任务的重开/刷新证据，确认原本地开发进程已退出后，沿用已提交启动脚本恢复本地页面，并在认证前核对代理仍指向同一测试后台。实际只发起一次登录：HTTP 400、应用码 `10004`，现有接口定义为 `ACCOUNT_LOCK_ERROR`，随后页面要求验证码。立即停止，没有重复尝试、解锁账号或使用其他身份；尚未进入结果页面，因此“完成结果刷新后保留”仍未验证。凭据和登录态未导出，过程材料在 `/tmp/review-resume-acceptance.bp8LoJ/`。

已知人员待标文件仍为 eligible 0、人工补框 0、标签 CSV 只有表头；主 Agent 独立核对哈希与行数。这里只检查既有本地材料，没有重新查询业务库、生成真值或启动模型评测。

下一步还需恢复测试账号的正常登录，才能继续真实页面只读验收。此项与仓库/发布机访问、正确集成来源和独立人工真值分别保留，不把等待或重复审计计为功能完成。

## 新测试账号登录与页面核验（15:32 +08 更新）

用户提供新的测试平台账号后，分别在实际测试站及本地 `ac928387` 候选提交一次登录，均成功进入项目列表，账号显示与用户本次提供的一致；再进入同一测试项目。两处均取消“记住密码”。本地原有旧账号会话通过正常退出流程切换，没有提取、复制登录态或修改认证协议。此前 `10004` 只说明旧账号当时的登录失败，不能继续作为新账号的阻碍。新证据见 `docs/evidence/review-new-account-ui-20261016.json`，不覆盖上午的历史记录。

同一条已有检索记录在测试站打开及整页刷新后均显示 `车(290)` 和结果缩略图；本地候选显示相同结果，同时明确提示“检索失败”，整页刷新后两者保留。已核对前端使用详情响应的 status 显示状态，结果列表独立呈现，因此有结果不等于任务成功。本轮没有独立读取该记录的数据库状态，不能从页面补推底层完成。另一条已有常规车辆测试记录在本地显示“已取消”“检索已取消，暂无结果”。真正成功完成记录的重开/刷新仍未验证。

页面截图仅留在权限受限的 `/tmp/review-new-account-F5ClAl/`，不提交业务图片、凭据、请求头或登录态。没有新建、重跑、取消检索，没有报告生成、保存或删除业务数据；本轮没有模型调用、共享部署或重启。代码未改变，复用既有 TDD 和构建证据，本次新增的是实际浏览器的登录与读取验证。

已核对本地 origin：前端为 `git@192.168.110.2:zhuoyi/onegismap_a.git`，Java 为 `http://192.168.110.2/zhuoyi/ktd_backend.git`。最近的读取拒绝仍分别属于这两个 GitLab 项目；独立的前端发布机是 `zhuoyi@192.168.110.17:22`。本轮未重试 Git/SSH，也未将测试网站账号用于这些入口。网站登录阻碍已解除，仓库/发布机访问、正确集成来源和独立人工真值仍待补齐。
