# 独立视觉工作台验收

状态：已发布至独立测试站 [http://192.0.2.1:5276](http://192.0.2.1:5276)，部署代码 `c97e46a7d5b01fdf842b91f469a04bfc0f8fb371`。2026-10-18 第二轮真实 Chrome 验收 5/5 通过，主 Agent 随后只读复核持久化和所有者隔离。人员及全类别准确率没有新的独立提升证据；用户人工体验验收待进行。首轮失败与第二轮成功均保留在[实机证据](evidence/independent-workbench-20261018/README.md)。

本项目已独立承载三个功能，不以 `zhuoyi/onegismap_a`、`zhuoyi/ktd_backend` 的读取或修改为交付前置。测试账号使用 `REDACTED_TEST_ACCOUNT`；密码仅由安全凭据存储提供，不写入本工单。灵犀原站新账号的登录有效性需在后续实际使用时验证，本次独立站验证不能替代它。

## 任务卡与基线

目标是在本仓承载图片复检、历史图片对比、智能检索的页面、任务和结果，解除共享前端/Java 源码权限前置。验收覆盖真实上传、任务状态、取消、断网恢复、持久化重开及测试站图片展示；模型准确率单独评测。

基线：`b752d0ef83da874f7c5c05e9f7b95d9f4640c447`，分支 `feature/independent-vision-workbench`，工作树 `/REDACTED_HOME/.codex/worktrees/review-experience-evaluation/review-annotation-platform`。已核验公司 origin、正式 master 与候选；原工作区标注 CSV、进度文档和未跟踪数据均保留。

输入为既有 FastAPI/React/SQLAlchemy/Celery/MinIO、实机算法契约及冻结评测输出。开发预算 45 分钟/120 次工具调用，独立发布阶段 45 分钟。身份为本地工作区用户及 SSH `REDACTED_SSH_HOST` 用户 `REDACTED_SSH_USER`。允许隔离开发、测试和已授权的新测试空间发布及测试模型请求；不写 Lingxi 业务库、不控制设备、不更换准确率候选、不修改两个共享项目。凭据只从运行身份的安全配置读取，不进入源码、报告或命令输出。

## 已实现

- 检索：素材上传、文字/参考图、服务开关、任务/结果落库、所有者隔离、分页、提交幂等、取消保留检出。复用既有 Job，算法只接收受控素材链接。
- 恢复：保存算法任务 ID；临时错误不重发创建；未知/中断提交单列；远端任务不存在明确失败。数据库版本条件更新保护终态与较新进度。
- 私有素材：输入签名不落任务记录；结果保存无秘密的 minio 引用，只为本任务目录、本 bucket 重新签名；新 bucket 保持私有。
- 历史对比：保存两张原图、完整结果、哈希、操作者和时间，重开重新签名；无法对齐仍显示无法比较，存储失败不产生成功记录。
- 页面：三入口、结果图/框、真实阶段、连续三次读取失败后手动刷新、旧响应隔离及历史分页。上传限制与 Nginx 对齐，逐文件图片上传保留部分失败。
- 评测：固定分母保留缺失、失败及转人工并复算旧输出，见[研究报告](review-accuracy-research-20261016.md)及计分证据（原始证据未随快照导出）。该修复不是识别准确率提升。

主要变更文件为 `backend/app/services/retrieval.py`、`backend/app/adapters/retrieval.py`、`backend/app/services/history_compare_records.py`、对应 `0005/0006` 迁移、`frontend/src/components/IntelligentRetrieval.tsx`、`HistoryCompare.tsx`、`RecordPagination.tsx`、`frontend/src/secureUuid.ts`、`deploy/vision-test.compose.yml` 及两个 `scripts/deploy_*vision*.py` 发布脚本；回归位于既有 `backend/tests/` 与 `frontend/tests/`，固定分母工具为 `scripts/score_frozen_recheck.py`。未引入第二套 Agent 状态机或更换本项目依赖管理。

## TDD 与本地验证

- 检索实际 404 红测后完成权限/开关切片；再通过真实图片上传驱动创建、重启重开、幂等、断网和取消。
- 已观察并修复素材丢失误受理、算法未配置误受理、私有结果图无法展示、提交中断卡住的红测。
- 事件屏障稳定复现并修复迟到取消覆盖 completed、新旧 GET 倒序覆盖命中，以及远端 404 永远 running。明确的创建拒绝与提交结果未知分开；检索 API/适配/迁移共 28 项通过。
- 历史记录三轮红绿覆盖路由缺失、丢图仍误报可重开、已有 0004 库迁移缺表；新测试 12 项通过。
- 真实 React + Chrome 的 19 项行为测试通过，包含空时间字段、第 21 条历史重开、200+sync_error 有界重试、断网不重建任务及移动布局。新增成功/失败/部分失败终态提示红绿，避免被旧创建提示遮住。只替换 HTTP 外部边界，不代表真实算法效果。
- 算法日志先复现签名 URL/异常凭据泄露，再验证两项通过；过滤不改变模型输入、阈值、抽样或输出。

标准入口为 `.gitlab-ci.yml` 的 `python -m pytest backend/tests -q` 和前端 `npm run build`。首次收集发现训练契约测试导入 torch/transformers，随后将纯契约函数原样移到既有轻量模块并保留旧导出；新增 Python -S 验证不需要 site-packages，未把重型推理依赖加进 CI、未跳过训练契约测试。最终代码版本的主 Agent 标准入口复跑为 **286 通过、1 跳过，49.31 秒**；PostgreSQL 专用用例随后通过独立新库的临时隔离 schema 单跑，`backend/tests/test_retrieval_migration.py` 为 **2/2 通过**，schema 已清理。compileall 通过。既有 Luna autotune 为 57 项运行、1 项跳过，其余通过；不将跳过项计为通过。

最终前端 `VITE_API_BASE=/api/v1 npm run build` 通过，Chrome 行为测试共 22 项；主 Agent 首次 20 项通过，后两项因本轮 Vite 进程退出连接拒绝，重新启动该进程后定向补跑 2/2 通过。部署/升级/隐私定向 65 项及独立 worker 部署 8 项均通过。命令和边界见[前端测试说明](../frontend/tests/README.md)及[实机证据](evidence/independent-workbench-20261018/README.md)。

## 发布后的真实功能验收

2026-10-18 01:04:18–01:05:18（Asia/Taipei），真实浏览器、登录、API、存储、算法和专用 Luna worker；无 HTTP 替身、无截图、无重复提交。总耗时 59.790 秒是整轮验收时长。

| 验收项 | 已观察结果 |
|---|---|
| 登录与三入口 | 真实登录、三个 tab 可使用 |
| 图片复检 | 1/1 样本 succeeded、失败 0；保存 7 个候选，页面终态与原图可见 |
| 历史对比 | 1 处几何变化、语义 completed；刷新重开，两张已存原图均解码成功 |
| 六秒视频检索 | completed、1 个命中、1 张结果图可读取；刷新重开没有重复 POST |
| 浏览器与请求预算 | page/console/media 错误均 0；图片导入、复检 job、历史记录、检索素材、检索任务各创建 1 次 |

主 Agent 用只读 PostgreSQL 事务核对任务终态、7 个候选、历史语义结果与检索 1 条结果均已持久化；首轮失败 job 和历史记录仍保留。使用 `REDACTED_TEST_ACCOUNT` 身份的一分钟服务签名 token 对该轮其他账号的历史/检索记录读取均返回 404；匿名 API 401、匿名已保存原图 403。该 token 核对是权限验证，用户密码登录已在此前独立站检查，不混称为本次新登录。记录没有持久化签名查询串，检索图保存为私有 MinIO 引用、读取时重新签名。

复检使用既有 960×720 冻结 JPEG；视频是同图缩放后重复生成的 640×360、6 秒/72 帧 MP4；历史对比为固定种子合成两图。本轮证明功能闭环和保存重开，**7 个候选不等于 7 个正确，1 个检索命中不等于新航次准确率，6 秒素材不验证长视频性能**。

## 已交付的进步与精度限制

- 同机同批离线检索回放由 840.804 降至 542.488 秒，耗时减少 **35.5%**；1655/1655 个采样与 719 个检测框零容差等价。见完整比较（原始证据未随快照导出）。这是原流水线 A/B 结果，未计页面、上传等，也没有新站长视频 p95 证明。
- 三个页面、任务和结果已独立运行；检索有真实进度、幂等创建、取消、失败/部分失败和断网后读取恢复；历史两张原图及结果可持久重开。真实站修复了 HTTP UUID 缺失、语义模型空回答、复检私有证据目录和日志 formatter 四类故障。
- 已完成逐类研究与固定分母计分工具，保留误拒、误放、uncertain、失败和缺失。人员改模型、提示词、裁剪候选尚不满足晋级门槛。下一轮先补可信同帧原图、按航次隔离的独立盲标，再评估小目标切片与逐类路由；不凭已有业务标签宣称全类别精度提高。

## 发布、限制与回滚

按[发布计划](independent-test-deployment-plan-20261017.md)已新建 `review-vision-test`，使用独立 PostgreSQL/Redis/私有 MinIO、单 GPU 和本轮输出卷；复用精确算法镜像，启动前关闭 MySQL/MQTT 写出。七个 Compose 服务之外，独立 CPU Luna worker 使用 18397 及专用私有归档目录。旧标注站接 fixture，不作真实算法证据。119 个受保护原有容器（含停止容器）的 ID、镜像和 StartedAt 保持基线；旧 5275/8392/8189 及原 worker 18396 保留。

发布前冻结提交、产物、镜像、配置摘要及原容器身份；已验证迁移、鉴权、浏览器媒体与数据库结果。原始回执只读核验后汇总为[发布与验收摘要](evidence/independent-workbench-20261018/receipt.json)，不导出真实配置、原图或模型自由文本。运行版本是上述 c97e46a；后续文档提交不会自动改变运行镜像。

回滚只停止本轮命名空间与专用 worker，保留卷、对象和证据；未为了演示执行停止。先核对无在途任务与原服务 `MainPID`；发布后本轮已核验原 PID 为 0，如重启变化应使用新核验值：

```sh
ssh REDACTED_SSH_HOST 'python3.12 /REDACTED_RUNTIME_ROOT/review-vision-test/releases/c97e46a7d5b01fdf842b91f469a04bfc0f8fb371/source/scripts/deploy_independent_vision.py stop --release-dir /REDACTED_RUNTIME_ROOT/review-vision-test/releases/c97e46a7d5b01fdf842b91f469a04bfc0f8fb371 --revision c97e46a7d5b01fdf842b91f469a04bfc0f8fb371'
ssh REDACTED_SSH_HOST 'python3.12 /REDACTED_RUNTIME_ROOT/review-vision-test/releases/c97e46a7d5b01fdf842b91f469a04bfc0f8fb371/source/scripts/deploy_vision_review_worker.py --stop --source-pid 0'
```

不执行 `down -v`、不删除旧 release 或数据，不重启旧服务；旧 5275 入口继续存在。

已知限制：跨数据库/对象存储失败可能留下不可见孤立对象，现有 ObjectStore 无删除接口，不自动删素材。仍缺新航次可信同帧原图和独立人工标签，人员/其他类别尚无新准确率晋级证据。在线模型 token 和费用未计量。独立站为内网 HTTP 入口，本轮验证了其 UUID 环境兼容；没有将该验收等同外网 HTTPS、原灵犀平台登录或人工全场景验收。
