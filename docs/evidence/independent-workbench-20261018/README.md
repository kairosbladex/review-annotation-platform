# 独立工作台实机验收与故障闭环（2026-10-18）

本页保留首轮失败和修复依据；最终发布版本及验收状态以[验收工单](../../ACCEPTANCE_VISION-WORKBENCH.md)为准。真实测试只针对独立站 `192.168.130.238:5276`，不连接 Lingxi 业务库、不发送 MQTT、不控制设备。原共享前端、Java 和原算法/复检执行服务不改动。

## 最终发布与第二轮验收：5/5 通过

运行代码 `c97e46a7d5b01fdf842b91f469a04bfc0f8fb371`，分支 `feature/independent-vision-workbench`；公司 origin push 后已独立核对远端 SHA。汇总字段见[发布与验收摘要](receipt.json)。文档后续提交单列，不把它冒充服务器代码版本。

第二轮真实 Chrome 时间窗为 2026-10-18 01:04:18.308–01:05:18.098（Asia/Taipei），总 59.790 秒；与首轮使用相同冻结 JPEG 和派生视频。无请求拦截/替身、无截图、不重发创建。原始脱敏 JSON 保留于本机私有证据目录，SHA-256 为 `8cb3e191263c1973390255c0d1570101481db4abd8c0b24fa2830a4c10709e5a`。

| 验收项 | 第二轮结果 | 阶段耗时 |
|---|---|---:|
| 真实登录与三入口 | 通过，3 个 tab | 0.621 秒 |
| 图片复检 | succeeded，1/1，失败 0，保存 7 候选，原图与终态可见 | 21.871 秒 |
| 历史对比保存/重开 | 1 处变化，semantic completed，刷新后两图解码成功 | 27.882 秒 |
| 六秒视频检索/重开 | completed，1 命中，1 结果图可读，失败文件 0，无重复 POST | 6.448 秒 |
| 浏览器错误与预算 | page/console/媒体错误全 0；五类创建请求各 1 次 | 断言阶段未单独计时 |

各阶段合计 56.822 秒，启动/清理 2.968 秒。检索 6.448 秒包含素材上传、任务、刷新和读取结果，不是纯推理耗时，也不作为长视频加速对照。报告输入的 640×360 对应合成历史图与派生视频；复检 JPEG 实际是 960×720。视频为冻结图重复生成的六秒/72帧/12fps MP4，缺新采集视频与独立对象标签，因此识别准确率为 `not_evaluable`。

主 Agent 已独立只读复核：新库任务/候选/历史/检索终态及计数与浏览器一致；首轮失败 job 和历史结果保留。一分钟 `suxiaonian` 服务签名 token 对该轮其他所有者的 history/retrieval 返回 404，匿名 API 401、匿名已存原图 403；这次权限读取没有调用模型或创建业务任务。输入/结果 JSON 无签名凭据，检索保存 2 个私有 MinIO 图引用，读时重新签名。

专用 Luna worker 的结构化日志只有本轮 1 个请求，7 个候选、`result=ok`、`cli_exit_code=0`、16,078.149ms，原 worker PID 276692 保留。只输出白名单统计，没有复制原始日志或自由文本。

## 工程发布身份与验证

- 目录：`/data/workspace/user/review-vision-test/releases/c97e46a7d5b01fdf842b91f469a04bfc0f8fb371`；SSH 身份 `ai-lingxi` 的 `user`，Compose 项目 `review-vision-test`。
- 七个 Compose 服务：postgres、redis、minio、algorithm、api、worker、frontend；另有精确验证的专用 `review-vision-test-luna-worker`（18397）。7 服务计数不把专用容器混入，也不允许未知额外容器。
- API/前端使用冻结自建镜像，算法使用 `sha256:dac104a6a8935905cfa3bbae6ed2d5114e0ed9f3bde7f90e5442361bce220ed1`。来源、配置、制品及镜像完整哈希见 receipt；私有文件 0600、release 0700。
- 专用 CPU worker 为 UID1000、只读根文件系统、runc/bridge、1 CPU/2GiB，无 GPU；Python 3.12.14/Pillow 11.3.0。复用冻结 Luna worker/CLI 和 readonly auth，模型 `gpt-5.6-luna`、提示词 `p000-pure-class-cards`、策略 `only_true_is_true_v1`、并发 1；没有挂载旧会话或复用可写归档。
- 新算法只暴露固定单 GPU、模型只读挂载，MySQL 写出关闭、MQTT host 为空、输出使用本项目卷和 `vision-test` 私有 bucket。119 个原容器（包含停止容器）的身份保持原样，旧 release 的 16 个已核验资源和卷均保留。
- PostgreSQL 迁移 head `0006_independent_retrieval`。主 Agent 最终标准后端入口 286 通过/1 跳过；独立 PG 临时 schema 的迁移专测随后 2/2 通过并清理；compileall 和生产前端构建通过。Chrome 行为22项的基础设施失败及补跑记录保留如下；真实站5/5单列。

主 Agent 使用既有虚拟环境执行标准入口 `python -m pytest backend/tests -q --tb=short`、`python -m compileall -q backend scripts migrations` 和 `VITE_API_BASE=/api/v1 npm run build`；迁移使用同一测试文件并通过 `TEST_POSTGRES_URL` 指向本轮新库隔离 schema。测试未连接生产或设备。后续只修改交付文档，因此不重复创建实站模型任务。

发布脚本 `verify` 在真实 UI 验收后再次 exit 0、`engineering_verified`；脚本的固定 `business_acceptance=pending` 只说明脚本本身不做浏览器验收，业务功能证据来自上面的独立报告。用户人工体验尚待进行，不把自动浏览器代替用户验收。

## 首轮真实验收：未通过

运行版本 `b597612d5e5703610a6f1bdc0ba5e2426947f263`，时间窗 2026-10-18 00:24:09–00:26:57（Asia/Taipei），总耗时 167.565 秒。真实 Chrome、真实登录/API/对象存储/算法，无请求替身；原始脱敏 JSON 保存在本机私有证据目录。五项验收中一项通过、四项失败，失败不能从分母删除。

原始 JSON SHA-256：`306b34399c3f43359b101613949855690a0a2ef7e469484fa77fcf225ac6e981`。

| 验收项 | 实际结果 |
|---|---|
| 登录与三个入口 | 通过，三个 tab 均可用 |
| 图片复检 | 失败，1 个样本中完成 0、失败 1；页面显示真实失败终态 |
| 历史对比 | 几何 matched、1 处变化，保存刷新重开且两张原图解码成功；语义失败，因此整项不通过 |
| 视频检索 | 上传前出现页面异常；媒体上传和任务创建请求均为 0；不能当作视频推理超时 |
| 浏览器与写入预算 | 1 个 pageerror，媒体请求/HTTP 错误为 0；故障项不通过 |

本轮实际创建 1 次图片导入、1 个复检 job、1 条历史对比；无检索媒体或任务。复检使用既有冻结 JPEG（960×720），SHA-256 `bcc277e80027d2890e3fc28ef7085efdf15a291e5c7c19f37f6fd4e02dacbe7f`；已有人员标签框贴左边界。视频为同图派生的六秒、72 帧 MP4。历史两图为固定随机种子的合成图及新增白色矩形。没有新增独立真值，不用于准确率结论。

## 已锁定的原因与 TDD

1. **HTTP 页面 UUID 接口缺失。** 实站 `isSecureContext=false`、`crypto.randomUUID=undefined`，但 `getRandomValues` 可用。真实 Chrome 的三项环境边界测试先全部失败，再通过；改为原生 UUID 或 CSPRNG UUIDv4，无安全随机源时在上传/创建之前明确拒绝。22 项浏览器用例已由主 Agent 逐项复核：首轮 20 项通过，最后两项因本地 Vite 退出出现连接拒绝；重新启动本轮服务后只补跑两项分页用例，2/2 通过。没有把连接失败当作产品断言通过。
2. **历史语义模型无答案。** 获准的一次诊断使用相同合成两图和原 256-token 预算，HTTP 200，但 `done_reason=length`、`eval_count=256`、content 长度 0、thinking 长度 370；16.196 秒后触发 JSONDecodeError。固定 HTTP 边界的真实比较入口可在 0.337 秒复现。历史 API 独立选用已存在的 `qwen3-vl:8b-instruct`，保留检索算法原全局模型；真实 Compose/prepare 红测 5 失败后修复通过。该诊断不新建对比记录。
3. **复检执行前的证据目录权限不符。** 初检实际得到 7 个候选；原 worker 记录 `input_evidence_error`，26.481ms 返回 503，CLI 未启动。其证据目录为 0777，违反执行器私有目录门禁；剩余容量、inode、512MiB 应用配额和 CLI 文件均不是该故障原因。独立执行实例使用专用私有目录，复用冻结代码、模型与提示词；保留原 worker 现场。
4. **日志脱敏破坏 Uvicorn 结构。** 原 hook 清空 access record 的五个参数，实际 AccessFormatter 报 `expected 5, got 0`。两项真实 formatter 红测复现后，保留五参数结构并脱敏目标查询；日志回归 4/4 通过。此错误与复检上游 503 分别处理。

升级须显式指定已验证的旧 release，沿用原数据库、MinIO、JWT、服务密钥及账号，保留旧卷/网络标签；只接受逐资源核验过的旧 owner。旧 source/config/私有文件/校验器摘要持续核对，新七个 Compose 容器必须使用新 release 与固定镜像。独立 Luna 执行容器另行验证，未知额外容器不能借该例外通过。部署、升级及日志相关回归当前 65 项通过，最终全量入口见验收工单。

## 模型与准确率边界

- 复检的旧两阶段 API 初检使用 YOLOv8n，随后调用 Luna；检索的主 YOLOE26L-seg 权重属于另一条链路。首次自建 profile 的全局权重标签已修正为 legacy 镜像/Luna 提示词标记，原失败 run 元数据保留；本次未换复检模型。
- 同批离线视频/图片回放的 840.804→542.488 秒、1655/1655 采样和719框等价，是既有独立性能证据；不是本轮站点端到端 p95。
- 人员及全类别没有新的独立准确率提升证据。固定分母计分、旧标签局限、原帧与新航次金标方案见[研究报告](../../review-accuracy-research-20261016.md)。

## 工作方法

按 TDD 先复现再修复，逐项诊断并区分页面、HTTP、模型输出、输入归档和日志错误；不放宽发布或凭据边界。
