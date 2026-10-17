# 独立视觉工作台测试发布计划

> 2026-10-18 更新：本计划已执行，最终独立站、运行 SHA、真实5/5验收和回滚以[验收工单](ACCEPTANCE_VISION-WORKBENCH.md)及[实机证据](evidence/independent-workbench-20261018/README.md)为准。下方保留发布前只读审计快照，`NOT DEPLOYED`、候选端口和待执行表述均为当时状态。

状态：`PREPARED / NOT DEPLOYED`。本文件基于 2026-10-17 对 `ai-lingxi` 的只读审计；只新增本文档，服务器、容器、环境变量、数据库和对象均未修改。发布执行及真实页面验收由主任务继续。

## 任务卡与结论

- 目标：让本仓独立承载复检、历史对比和智能检索，并为真实算法联调提供独立运行空间。
- 可验收结果：新入口可登录、上传、执行、查看进度和重开结果；数据库、素材和算法产物均能证明归属；旧容器身份保持稳定。
- 输入：实际容器的镜像/Compose 标签/挂载、白名单配置和源码；本仓最终提交。凭据仅由部署主机安全配置提供。
- 允许副作用：本阶段只读检查及本文档；执行阶段仅新测试空间的容器、数据库、对象和模型请求。
- 身份：SSH 用户 `user`，具备 Docker 组权限；当前 API/worker 以 root 运行，PostgreSQL 主进程 UID 999。
- 审计预算：首轮 10 分钟/25 次工具；补充算法隔离核查 8 分钟/20 次；文档准备 10 分钟/25 次。
- 非目标：准确率提升结论、生产发布、Lingxi 业务库写回、设备动作。

建议新增独立 Compose 项目 `review-vision-test`：前端 5276、API 8393、算法 8193。使用独立 PostgreSQL/Redis/MinIO 命名卷；新 MinIO 端口 19004/19005。这样保留旧 5275/8392 入口，回退只需停止新项目。上述五个候选端口在本次实机快照中均未监听，执行前应重新核对。

资源受限时可复用既有独立标注 MinIO 的内部 endpoint，并创建新专用 bucket；应明确 bucket 的访问策略和签名 URL，不能把算法图像配置到无人机共享 bucket。复用旧标注数据库也需要单独说明表/用户/数据归属；它的影响范围大于全新命名卷方案。

## 已核验运行基线

| 项目 | 实际状态 |
|---|---|
| 旧标注项目 | `review-annotation-test`，前端 5275、API 8392、MinIO 19002/19003 |
| 旧发布目录 | `/data/workspace/user/review-annotation-test/releases/4e3e5c7` |
| 旧 Compose 文件 | `/data/workspace/user/review-annotation-test/releases/4e3e5c7/deploy/review-compose.override.yml` |
| 旧配置来源 | 同版本 `source/.env`；Compose `env_file` 和插值均可用该文件正确解析 |
| 旧源版本 | MANIFEST 记录 `4e3e5c728012d05adbe1854cad22de53ea2d06da`；发布目录和 source 均无 `.git` |
| 旧 API/worker 镜像 | `review-annotation-test-api:4e3e5c7`，`sha256:941b8e3c54001b53546aa19593b1d77c739de5ce6a04def7240cd278bccc3c00` |
| 旧前端镜像 | `review-annotation-test-frontend:4e3e5c7`，`sha256:8fc26b906cd0c0737a497938ef9d5b97d4a4cc5067ab54b69cce02f0613dfa3e` |
| 真实算法原实例 | `yoloe_service_test`，`sha256:dac104a6a8935905cfa3bbae6ed2d5114e0ed9f3bde7f90e5442361bce220ed1`，端口 8189 |
| 原算法主线权重 | `yoloe-26l-seg.pt`，78,909,291 bytes，SHA-256 `a9413bf3f15772c223a03bbedd71b79af5822f830e80aa1b51b1a469e65927b1` |

旧标注栈默认 `MODEL_SERVICE_BASE_URL` 指向 `http://model-fixture:8080`，`MODEL_ADAPTER_TYPE=legacy_review_agent`。fixture 进程是 `review_model_fixture.py`。`HISTORY_COMPARE_OLLAMA_URL` 未配置，`AUTO_ANNOTATION_API_KEY` 无值。这是接口/数据闭环基础，不能据此声称真实检测、复检或语义对比已验收；数据库 profile 可能另有覆盖，需按候选实际运行配置核验。

旧配置文件及 Compose 文件的宿主权限为 0777。审计只读取其权限元数据和必要的非秘密配置；新发布目录应为 0700，凭据文件应为 0600，排除源码包、日志、Git 和报告。

## 真实算法副作用边界

以下行号来自当前运行镜像内源码：

- `/app/api/retrieval_batch_models.py:17-38` 的 `BatchSearchBody` 无单次禁用 MySQL/MQTT 的字段，仅有可选 `mqtt_topic`。
- `/app/api/retrieval_batch_routes.py:163`、`:201` 在 topic 为空时自动生成原前缀 topic；传空 topic 不能关闭发布。
- `/app/api/retrieval_batch.py:930-947` 对检出结果调度 MySQL 入库；`/app/api/detection_mysql.py:46-52` 的全局 `MYSQL_ENABLED=false` 会拒绝这一路径。
- `/app/api/mqtt_helpers.py:91-94` 在 `MQTT_HOST` 为空时直接返回。
- 原实例实际 `MYSQL_ENABLED=true` 且 MQTT host 已配置。因此直接向原实例发真实检索任务，会触及原数据/消息链路。
- `/api/v1/review-agent/yolo-detect` 会写上传图和预览图。`/api/v1/review-agent/review` 会写结果/审计，并将存疑目标保存到本地 `ReviewAnnotationStore`；不是纯只读计算。所查两个路由没有直接调用 MySQL/MQTT，但这不足以代表整个算法服务所有路径无副作用。
- `/app/yoloe_service_main.py:207` 启动时检查 MySQL schema，`:220` 预热 GPU 模型池。隔离开关必须在启动前生效，GPU 加载应纳入资源验收。

独立算法实例以全局禁用写出的配置启动，产生的新本地结果归属本轮独立目录。真实复检若继续调用既有 Luna/clean-source worker，须记录该调用身份和预算；把模型开关关掉后跑通不等同原真实复检链路验收。

## 算法代码与权重复用

直接以完整镜像 ID `sha256:dac104a6a8935905cfa3bbae6ed2d5114e0ed9f3bde7f90e5442361bce220ed1` 启动独立容器。原容器的 `/app` 业务代码没有 bind mount；`docker diff` 对 `/app` 下 `.py/.sh/.yaml/.yml/.toml` 和 `PATCH_COMMIT` 未发现新增、修改或删除。无需复制大量历史代码覆盖层，也无需对共享容器执行 `docker commit`。

该检查验证当前业务源码未在镜像外被修改；容器 overlay 另有 308 项变化，包含运行产物/缓存等，不能称整个文件系统与镜像完全一致。若新实例必须挂载额外代码，先保存对应源码摘要与镜像版本的比较证据，并使用只读挂载；发现未知差异时停止复用推断。

关键运行源码 SHA-256：

| 镜像内路径 | SHA-256 |
|---|---|
| `/app/yoloe_service_main.py` | `bb4b17a41b6c49ac261c29977b8793ed8c3ee27fd95cd04523225b5313005a15` |
| `/app/api/retrieval_batch.py` | `42875ab385d161a0f0e07f8f131ef4b19d5119fea36d257054263d8fe82bf3da` |
| `/app/api/retrieval_batch_routes.py` | `991e74444c411a543bfabfd7ea3f6649b8512f04d6401307b760bf1c3b71ad84` |
| `/app/api/review_agent_routes.py` | `a56fb3bc6e9fea5ef3049960d1270052e4d2beeaf3a3903b76fa1aef4181bf00` |

建议挂载边界：

| 来源 | 容器目标 | 模式 |
|---|---|---|
| 原算法 `/data/workspace/jzheng/yoloe_service_test_prod/pretrain` | `/app/pretrain` | 只读，复用固定权重 |
| 同一 pretrain 或独立已冻结权重缓存 | `/root/.cache/ultralytics` | 只读；若需要新增缓存，改用本轮独立可写目录 |
| 原算法证书目录（采用 HTTPS 时） | `/app/certs` | 只读，私钥不进入文档/新源码包 |
| clean-source base/adapter（保持该链路时） | `/opt/clean-source/base`、`/opt/clean-source/adapter` | 沿用原固定版本，只读 |
| 本轮独立 runtime/uploads | `/app/uploads` | 本轮独立可写 |
| 本轮独立 runtime/results | `/app/results` | 本轮独立可写，含 retrieval_batch、review_agent、标注 SQLite 与审计 |
| 本轮独立 runtime/cache | `/app/cache` | 本轮独立可写 |
| 本轮独立 runtime/test_result | `/app/test_result` | 本轮独立可写 |

原实例的 uploads/results/cache/test_result 均是共享可写路径，新实例不能复用这些来源。原权重绑定实际可写，本轮将新实例绑定改为只读，不修改原实例的挂载。算法日志、Ultralytics 配置和临时文件也只能落本轮目录或新容器自身可丢弃层。

## GPU、模型与配置

GPU 快照：index 4，UUID `GPU-9f2ac4d0-be45-fe5b-d16a-86d1f3fd10f7`，总显存 46,068 MiB、已用 462 MiB、可用 44,998 MiB。仅给新容器该 UUID；容器内索引按 `cuda:0`，启动前再次检查可用显存。原实例当前暴露全部 GPU，不能照抄 `Count=-1`。

| 配置键 | 新实例要求 |
|---|---|
| `MYSQL_ENABLED` | `false` |
| `MQTT_HOST` | 空字符串 |
| `MYSQL_*`、`MQTT_USERNAME`、`MQTT_PASSWORD` | 不注入共享业务凭据；不能仅依赖“没有命中所以没有写入” |
| `API_PORT` | `8193` |
| `DEVICE`、`CUDA_VISIBLE_DEVICES` | 与单卡暴露方式配套；容器实际 PyTorch 可见索引为 0 |
| `YOLOE_MODEL_VERSION` / `YOLOE_MODEL_SIZE` / `YOLOE_MODEL_TYPE` | `26` / `l` / `seg`，保持固定模型 |
| `CONFIDENCE_THRESHOLD` / `IOU_THRESHOLD` / `MAX_DETECTIONS` | 原全局值 `0.15` / `0.7` / `300` |
| `BATCH_INFER_FRAME_BATCH_SIZE` | `1` |
| `BATCH_MAX_CONCURRENT_GPU_JOBS` | 初次联调设 `1`，记录其调度影响 |
| `GPU_INFER_PARALLEL_SLOTS` | 初次联调设 `1`，限制额外模型副本；吞吐比较需注明与原配置差异 |
| `MINIO_ENDPOINT` / `MINIO_BUCKET` | 新项目自身 MinIO 与专用 bucket；复用旧标注 MinIO时也必须是新专用 bucket |
| `MINIO_PUBLIC_BASE_URL` / `RESULT_PUBLIC_URL_PREFIX` | 指向新实例可访问资源；需实际验证鉴权/访问策略，不能只拼出 URL |
| `REVIEW_ANNOTATION_DB_PATH` | 新 results 目录内的专用 SQLite 路径 |
| `SSL_CERT_FILE` / `SSL_KEY_FILE` | 选择 HTTPS 时指向只读证书；HTTP 内网方案需同步配置平台 adapter |
| `API_KEY`、模型/MinIO 凭据 | 仅部署主机安全文件或秘密注入；报告只记录是否配置 |

批量请求默认 confidence=0.2 与全局初检阈值 0.15 是不同入口的配置。复算此前性能证据时固定原批量请求：每 3 秒抽样、confidence=0.2、iou=0.7、max_det=300、batch=1、切片关闭。改变模型开关、GPU 槽位或抽样参数后，应单列新结果，不沿用 35.5% 的历史收益作新实例测量结论。

`CODEX_REVIEW_ENABLED`、`CLEAN_SOURCE_WORKER_ENABLED`、`CLEAN_SOURCE_REVIEW_ENABLED` 原实例均为 true，`CLEAN_SOURCE_PERSON_AUTO_CONFIRM_ENABLED` 为 false。真实复检验收需明确保留哪些链路及 worker 路由；未接通者显示“未配置/不可用”，不得回退 fixture 后报告真实效果。

## API 与前端选择

| 方案 | 影响与建议 |
|---|---|
| 新项目、新 8393/5276、独立 PG/Redis/MinIO | 建议首发方式；旧入口和数据不动；资源、凭据及验收样本独立，回退停止候选项目即可 |
| 旧项目仅更新 api/worker/frontend | 可行，但直接改变旧入口；需保存原镜像、配置摘要、数据库备份并完成兼容性检查；`--no-deps` 限定服务，避免误动存储/fixture |
| 新端口但复用旧 PG/MinIO | 服务入口隔离、数据副作用仍共享；必须明确专用数据库/bucket及权限，不能仅凭新端口称完全隔离 |

新前端 Nginx 必须为 `/api/v1/history-comparisons` 和检索上传入口设置与后端匹配的 body 大小和超时。当前仓库只有旧 `/api/v1/history-compare` 配置 42m/120s，新路径默认限制会令页面上传失败。新接口的鉴权应同时覆盖创建、列表、详情和媒体访问。

API 镜像启动会先执行 `alembic upgrade head`。执行前冻结最终候选提交、迁移 head、前后端镜像 ID、配置摘要和网络/挂载；新表采用兼容添加，回退保留证据。旧 API/前端均无 Docker healthcheck，发布脚本需限时等待真实接口并独立核验版本。

## 执行顺序与验收

1. 冻结本仓最终提交，完成既有离线测试、TDD 红绿记录、构建及迁移回退验证。导出精确提交的最小源码包，排除 .env、业务数据、样本、日志、权重、缓存和 node_modules。
2. 在新发布目录保存非秘密 manifest；配置文件安全注入。构建新 API/前端镜像，使用精确 ID；算法使用上述固定镜像。启动前保存原项目、算法和 worker 等受保护容器的 ID/镜像 ID/StartedAt、配置摘要及挂载摘要。
3. 重新检查端口/GPU；创建候选项目自身网络、命名卷与运行目录。算法仅挂载只读模型资源，MySQL/MQTT写出在启动前关闭。禁止对原算法或旧标注栈执行重启/重建。
4. 在新数据库执行迁移；确认数据库 endpoint 属于候选项目，MinIO bucket 为候选专用，随后启动 API/worker/前端。核对全部实际生效配置与所选环境一致。
5. 用冻结、无业务含义素材做真实检测、复检、历史对比和检索；记录请求身份、任务 ID、样本哈希、错误及结果归属。验证失败/取消/超时/刷新/跨账号访问；浏览器确认图片与视频可打开、记录可重开。
6. 核对业务写出关闭证据与原容器身份保持一致；保存候选镜像/源码/配置摘要、接口结果和 UI 证据。任何健康检查、HTTP 2xx 或 fixture 结果只证明对应工程层。

## 回滚与停止条件

首选新项目方案：仅停止本轮候选项目的 frontend/api/worker/algorithm 及自身依赖，释放 5276/8393/8193 和 GPU；保留新数据卷、对象、日志及回执。旧入口继续使用原栈。正常回退不执行 `down -v`、不删除任何旧卷或原算法目录。

若最终选择更新旧栈，原制品仍在，以下是已经只读验证可解析的恢复配置；真正执行前核对其摘要和无在途任务：

```sh
docker compose \
  --env-file /data/workspace/user/review-annotation-test/releases/4e3e5c7/source/.env \
  -f /data/workspace/user/review-annotation-test/releases/4e3e5c7/deploy/review-compose.override.yml \
  --project-name review-annotation-test \
  up -d --no-deps api worker frontend
```

该命令未执行。它恢复旧 API/worker/前端镜像和旧默认 fixture 接线；不代表恢复真实算法能力。新添加的审计表应保留，不通过删表实现回退。

停止条件：发现目标镜像/原容器身份漂移、共享路径可写、MySQL/MQTT 未关闭、连接到 Lingxi 库/共享 bucket、GPU 资源不足、迁移失败、鉴权或媒体重开失败，或真实算法被 fixture 替代。停止扩大实验，保留证据并仅回退本轮候选资源。
