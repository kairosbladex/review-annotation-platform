# 三功能本地候选与测试发布接续索引

这份索引集中定位已提交的本地候选、已完成的验证和剩余发布条件。测试发布授权已经具备。最近一次沿用现有身份的检查中，两个 GitLab 项目仍拒绝下载，前端发布机仍拒绝 SSH 认证；本索引不是发布成功证明。

## 已核对的源码快照

下表记录本轮核对的源码提交；review 仓库随后新增的本索引属于文档，不改变产品代码。各候选代码已保存在明确的本地提交。Java 自动恢复开关的补充验证见下节，已通过主 Agent 独立复跑。

| 用途 | 本地分支 | 完整提交 | 工作树绝对路径 |
| --- | --- | --- | --- |
| review 对照台原始存疑显示与交付证据 | `bugfix/review-uncertain-display` | `6c0ef03de72cdba3b814ca3aec7fa6ad3fa59dea`；产品修复 `5cf78bff9d91fa87e6a62283a5a610843b3fc0a9` | `/Users/wangxiaolin/.codex/worktrees/review-uncertain-display/review-annotation-platform` |
| 检索状态、有限重试、历史对比快捷键 | `bugfix/review-experience-local` | `ac92838789a8c070970d56c6ac5e9161559a5d7d` | `/Users/wangxiaolin/PycharmProjects/onegismap_a-review-experience-local` |
| 图片复检切图与未通过文案 | `bugfix/model-review-verdict-label` | `76e2cd8b469e66c1650217492e538b0866ebec66` | `/Users/wangxiaolin/PycharmProjects/onegismap_a-model-review-local` |
| Java 检索终态恢复及默认关闭的自动扫描 | `bugfix/retrieval-task-completion` | `168813558ab6448ac2a2f82cae0b50df2b051162`；原终态修复 `9381d6209c194d5ad35d4162de70c582b7369a3e` | `/Users/wangxiaolin/IdeaProjects/ktd_backend-retrieval-task-completion` |
| 算法提速与发布检查 | `perf/retrieval-response-latency` | `8adac81798c55855f686ad24b6b9c50209430662`；产品提速 `a407bf3513589ed43b604766e414f50566fd1cdf` | `/Users/wangxiaolin/PycharmProjects/yoloe-retrieval-response-latency` |

已知差异基线也已在本机解析：review 为 `b752d0ef83da874f7c5c05e9f7b95d9f4640c447`；检索/对比前端为缓存 `origin/master@35574cba46d498e0992de05504756bccf7014ade`；图片复检业务为 `3743dfdf9daa45b119c01165842429d2f21b593c`；Java 为缓存 `origin/prod@e8e31571439de72950bf865cce2cf897ce6ed3ba`；算法为 `60c23831b7261b85294ad4bbb99b267a3f179e49`。缓存引用不是已核验的最新远端。

## 变更范围与验证入口

- review 产品文件只有 `backend/app/recheck_workbench.py`，新增接口回归位于 `backend/tests/test_recheck_workbench.py`。有效 `rawDecision` 优先、必要时兼容旧值，保留原始结果和业务动作。新旧契约合跑 29/29；命令在 [交付记录](review-experience-followthrough-20261016.md) 的“显示修复与 TDD”节。
- 检索/对比前端的产品改动为 `src/views/history/detail.vue`、`src/views/compareResult/components/CompareResultContent.vue`，其余为测试、启动与 Vite 测试入口。`tests/retrieval-progress-ui.test.mjs` 9/9、`tests/history-compare-shortcuts.test.mjs` 9/9，`npm run build:test` 通过。默认 lint 的旧 7 errors/13 warnings 单列，不能声称全量 lint 通过。完整命令见对应工作树 `docs/local-review-experience-20261016.md`。
- 图片复检业务的产品文件为 `src/components/event/components/ImagePreviewDialog.vue`、`src/components/event/components/AiReviewResult.vue`；Vite 只删除过全量环境打印并加入测试入口。`tests/model-review-ui.test.mjs` 最新 3/3，构建通过；旧格式问题保留。此业务工作树未整分支合入 3081 主线入口。
- Java 候选跨 `biz-ai`、`biz-inspection` 两模块，包含 DTO、服务、控制器、恢复执行器、Mapper 接口和行锁 XML，共 7 个生产文件及 1 个测试类。本轮开关补充的有效红测为 22 项中 2 失败/20 通过，绿测及主 Agent 独立复跑均 22/22，8 模块离线构建成功。准确命令为 `mvn -o -s settings.xml -pl biz-inspection -am -Dtest=AiSearchTaskCompletionTest -Dsurefire.failIfNoSpecifiedTests=false test`，复用现有设置但未读取、修改或记录其中配置。说明和红绿证据见该工作树 `docs/retrieval-recovery-release-gate.md`。不替换单个 class 作为发布。
- 算法提速已做过测试发布，来源与回滚命令在 [发布回执](evidence/retrieval-deployment-receipt-20261016.json)；本地缓存远端分支仍指向 `8adac81798c55855f686ad24b6b9c50209430662`。固定 17 路视频减少 35.0% 耗时；完整 1655 个采样、719 个框输出一致，范围见 [同输入对比](evidence/retrieval-full-comparison-20261016.json)。它不是新任务页面总耗时或准确率提升。

未改动的前端、review 和算法测试没有在编写索引时重复运行；Java 因新增开关完成了上述红绿回归。实际页面验证已补齐：新测试账号正常登录；256 条结果、429 个目标的完成检索在本地和测试站重开/刷新保留；完成零命中在本地明确展示；历史对比 1 张基准/44 张历史在两端刷新后保留。详情见 [数据库到页面的证据](evidence/review-completed-ui-20261016.json)。图片与原始业务元数据留在受限临时目录，没有提交 Git。

## Java 自动恢复的发布边界

开关补充前，`recoverPendingTasks()` 会默认每 30 秒扫描全体满足条件的运行中任务，50 条/批，待恢复名单未核实。按用户“新能力默认 feature flag 关闭或 Shadow”的发布规范，本轮在 `168813558ab6448ac2a2f82cae0b50df2b051162` 增加默认关闭的 `ai.search.recovery-enabled` 开关；只在显式启用时扫描。未设置/显式 false 的行为测试均验证零数据库调用、零远端查询/停止、零结果写入；显式 true 仍完成原有终态恢复。单任务 progress/stop/终态收敛不受该开关控制，原 19 项回归保留。测试使用隔离 MockEnvironment 和内存替身，无业务应用、外部服务或定时器。该候选尚未部署，不用配置关闭替代恢复功能验收。

## 权限恢复后的接续顺序

1. 刷新并核对两个仓库的正式/业务基线和 `dev-wxl`，确认与当前测试环境的源码、制品和回滚包对应。Java 保持从 `prod` 同步；两条前端业务分别维护，`test` 仅作集成目标。本轮核对 `ac92838789a8c070970d56c6ac5e9161559a5d7d` 与缓存 `origin/test@e7cffeee73a22663d623e14c30900828223b3bc6` 有 943 个路径差异；缓存 master 对同一 test 为 929。旧报告的 949 属旧比较快照，不能用作当前差异分母，也不能用本地 dist 整包覆盖共享站点。
2. 仅集成本任务改动，完成受影响回归后按既定“语义分支 → dev-wxl → test”流程发布。原完整 Java 包和前端资源身份已出现过变化，发布前必须重新核对；旧包不是当前可用回滚包的证明。Java 两模块和 XML 一起构建，自动恢复默认关闭；显式启用前先核对任务范围与在途名单。
3. 发布后把运行制品绑定到实际源码，使用真实结果检查状态、刷新保留、失败/取消及旧流程兼容，再测新任务完整耗时。仅健康检查或同输入离线加速不足以完成这一步。

仓库入口为 `192.168.110.2/zhuoyi/onegismap_a` 和 `192.168.110.2/zhuoyi/ktd_backend`；前端发布入口为 `zhuoyi@192.168.110.17:22`，目标 `/home/zhuoyi/ktd/web/amap/`。测试网站账号仅用于网站认证，不用于 Git/SSH。本地 review、前端和 Java 候选未完成本轮公司推送或共享部署；算法之前的测试发布不重复执行。

图片复检仍有 80 个新来源人员候选框，但没有形成可信同帧、内容去重和独立人工真值。现有研究未给出可发布的全类别提升；历史对比的“无法可靠对齐”不能记为准确率通过。补齐这些证据前，保留当前模型与业务放行策略。
