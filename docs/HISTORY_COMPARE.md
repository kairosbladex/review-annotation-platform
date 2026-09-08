# 历史对比

## 2026-09-08 目标范围复核（v7）

记录 924（`DJI_20260715113348_0009_V_P4A3.jpeg` 对 `DJI_20260715111248_0009_V_P4A3.jpeg`）要求“对比一下光伏板的变化”，v2 返回了 11 处道路等无关区域。根因是 v2 未将用户任务传入区域复核，描述模型也不筛选几何候选。

v7 保留 v2 的配准和轮廓定位，传入结构化任务要求，每批最多 8 个区域，逐批筛选全部候选。光伏板任务只保留组件自身明显变化；泛称“道路的区别”保留道路上的车辆、行人、堆放物变化；明确限定道路形状或施工时才排除普通车辆等。全部候选确认无关或无明显变化后，返回空区域和“没有明显变化”。模型失败、未知、缺行、无效 ID 和复核预算耗尽均不能当作无变化；只保留已经确认的区域，并提示人工复核。结果仅显示筛选数量，不拼接容易产生遮挡误述的逐行模型描述。

复核预算默认 60 秒，可配置 1 至 120 秒，各批共享剩余预算，不重试。部署配置为 `history-compare-20260908-v7`，保留 v2 镜像及 override 可回退。大面积耕地兼容未恢复，独立历史对比 API 无任务参数时维持原有行为。

候选验证：26 项测试通过，含两入口任务透传、11 个候选全部过滤、第二批真实目标保留、超时和协议错误不能误报无变化、后批失败保留已确认框。现有容器中使用同一模型回放：光伏无变化样本 0 框（34.257 秒），认可的道路样本 5 框（10.719 秒），真实光伏新增样本 4 框（8.618 秒）。线上部署与接口验收另记。

## 2026-09-08 恢复参考图效果

用户确认以参考图中车辆、行人、堆放物的 5 处局部轮廓为准，恢复 v2，而非 v4。相同道路照片和提示词“对比一下道路的区别”，v4 隔离回放为 0 处（18.070 秒），v2 为 5 处（5.848 秒）。本地通过独立 revert 提交 `ba4fea4`、`4e689f8`、`352316c` 撤回 v6、v4、v3 的相关改动，不改写历史、未 push。

按用户要求等待现有批量任务退出活动列表。17:19:36（UTC+8）确认 `active_count=0` 后，仅将 8189 第五份 Compose override 切回 `history-compare-20260908-v2/compose.override.yml` 并重建测试容器，8188 未变动，挂载、端口、GPU 配置保持一致。镜像 ID 为 `sha256:e3077572ea636a54d9576ed09aad3023887585eccc3584448a79c6e691fde851`，健康状态 healthy，三个运行文件 SHA256 与本地回退代码一致。

验证：16 项 v2 相关测试通过。部署后真实 `many-to-many` 接口使用用户提示词和 `DJI_20260715110312_0002_V_P1A3.jpeg`、`DJI_20260715104322_0002_V_P1A3.jpeg` 原图，12.207 秒返回 5 处区域；人工检查叠加图与参考图对应。当前恢复几何区域优先、语义描述失败仍保留区域的行为，模型描述等待恢复 20 秒；任务严格筛选、纯色差专项过滤、大面积耕地兼容同时撤回，已向用户说明并获确认。旧结果不回写，需重新发起对比。

登录后打开“历史对比”页签，分别选择历史照片、当前照片后开始对比。变化多边形绘制在当前照片上，可隐藏或下载 JSON。交换照片会清除旧结果，再次对比采用交换后的时间方向。

## 接口

`POST /api/v1/history-compare` 使用平台已有的 `Authorization: Bearer <token>` 登录凭证，上传 multipart 字段 `baseline`（历史照片）、`current`（当前照片）。单张文件最多 20 MiB、2500 万像素，最短边至少 32 像素。图片按 EXIF 方向解码。请求和结果不保存到样本、候选框或人工金标库。

返回值包含：

- `status`：`matched` 表示可比较，`not_found` 表示无法可靠配准，不能解释为无变化。
- `polygons`：变化区域点集，坐标归一化到 `[0, 1]`，以经 EXIF 校正后的当前照片为基准。像素坐标为 `x * current_width, y * current_height`。
- `change_count`、`diff_summary`：检测到的区域数量和几何结论。
- `alignment`：配准方式、内点数、有效重叠比例。
- `semantic`：独立的描述状态、摘要、描述覆盖的局部区域数 `region_count` 和错误信息。状态为 `completed`、`not_needed`、`unavailable` 或 `failed`。模型不可用时保留几何结果并显示提示。

错误码：未登录 `401`，文件过大 `413`，图片无效 `422`，并发已满 `503`。单进程最多同时处理两个对比请求。

## 运行配置

安装项目依赖后即可运行几何对比；语义描述需要配置可访问的 Ollama：

```dotenv
HISTORY_COMPARE_OLLAMA_URL=http://192.168.130.238:11434
HISTORY_COMPARE_MODEL=qwen3.5:9b
HISTORY_COMPARE_TIMEOUT_SECONDS=60
```

URL 默认留空，关闭语义描述。使用 Compose 时在部署环境文件中设置。前端 Nginx 为此接口单独允许 42 MiB 请求体和 120 秒读取超时，部署到其他代理时需匹配这些限制。语义接口调用不重试。

## 适用范围与验证

几何部分使用 OpenCV ORB/RANSAC 配准、全局亮度补偿和差分轮廓。适用于同一场景、相近视角；有效重叠不足 70%、配准不可信或大范围成像差异会返回无法比较。最大处理边长为 1600 像素，区域过滤阈值为处理图面积的 0.07%，较小或低对比变化可能漏检。阴影、季节、视差也可能形成差分，因此结果仍需人工复核。

Qwen 接收面积最大的最多 8 处变化区域的配对裁剪，仅负责局部语义描述，不生成或修改多边形。描述属于模型输出，不能作为全图物体数量或因果判断；超过 8 处时几何结果仍包含全部检测区域。

2026-09-08 本地验证：用户提供的 4032×3024 光伏原图识别到 4 处新增区域；交换时间顺序验证缺失方向。另覆盖相同图、平移、整体亮度变化、无法配准、无效/过大文件、认证、模型超时与 Markdown JSON 返回。桌面及手机浏览器验证上传、对比、叠加层切换、交换照片和返回人工复核。这些验证对应独立 API；AFMAP 的实际接入及验收见下文。

## 独立测试部署

2026-09-08 经用户确认，部署到本机独立 Compose 项目 `review-history-test`。入口为 <http://127.0.0.1:5275/#history-compare>，API 为 `127.0.0.1:8392`。使用本机测试账号 `admin / change-me-admin-password`，仅绑定回环地址。原有 `5274/8391` 测试栈和本机标注数据保持不变。

配置文件为 `deploy/history-compare.test.yml`，需要支持 `!override` 的 Docker Compose（已用 5.5.0 验证）。镜像 `review-history-api:683feb3` 和 `review-history-frontend:683feb3` 从干净提交 `683feb3bd36fb9fc2621c1a09a017a447a866f97` 构建。启动已构建镜像：

```bash
docker compose --env-file .env.example -p review-history-test \
  -f docker-compose.yml -f deploy/history-compare.test.yml up -d --no-build
```

该项目使用独立网络及 `review-history-test_*` 数据卷，MinIO 端口为 `127.0.0.1:9002/9003`。服务设置为 `unless-stopped`，Docker 恢复运行时自动启动。比较文件和结果仍不自动持久化，需要保留结果时在页面下载 JSON。

部署验收：实际 Nginx 入口完成原图正反向对比，均为 4 处变化，桌面和手机截图无溢出；未认证为 401，无效图片为 422，超过 20 MiB 为 413，相同图片为零变化。新 PostgreSQL 已迁移至 `0004_agent_review_fingerprint`，样本/候选框/标注仍为零。原测试栈的全部已有表记录数与部署前一致，其 API、Worker、前端启动时间仍为 2026-09-02。独立栈部署阶段未操作远端服务。

## AFMAP 兼容适配

AFMAP 的 `/ai-comparison/image/add` 经业务服务调用 8189 的 `/api/v1/vl-compare/many-to-many`。2026-09-08 页面实测记录 850 显示：`baseline_images` 实际装载当前航拍照片，`compare_images` 装载历史照片；模型连续两次 60 秒超时后返回 `not_found`。该链路与本机独立测试栈不同。

`scripts/prepare_history_overlay.py` 为已核验 SHA256 的旧路由生成可选适配补丁，复用本项目的配准服务。只在 `VL_COMPARE_ENGINE=registered` 时启用；默认仍走旧实现。`VL_COMPARE_BASELINE_IS_CURRENT=true` 专用于上述 AFMAP 入参约定，将输出坐标固定在当前航拍照片，即该调用的 baseline 图上。标准 baseline=历史图的调用方应保持此开关为 false。一对多仍按 current/history 的明确角色处理。

适配保留原 URL、认证、响应模型、配对索引、图片 ID 和记录保存流程，返回的是像素坐标。`match_score` 为 null，不将几何重叠率伪装成模型置信度。语义描述最多等待 20 秒、不重试；失败时仍返回已定位区域，并在摘要中标明描述失败。无法配准仍为 `not_found`，不能解释为无变化。

`deploy/vl-compare-overlay.Dockerfile` 在现有测试镜像上增加运行文件；`deploy/afmap-history-test.override.yml` 作为现有全部 Compose 配置之后的附加文件使用。旧路由 SHA256 不一致时准备脚本拒绝生成补丁。切换前需要确认测试服务无活动任务，保留旧镜像及原 Compose 文件列表供回退。

## AFMAP 在线验收

2026-09-08 经用户确认，已部署至 `192.168.130.238:8189`，容器 `yoloe_service_test`。镜像为 `yoloe_service_test:history-compare-20260908-v1`，镜像 ID 为 `sha256:8d304ce9c59a78445b63f754d6ec582f45a4cbb3d5353d6fb05f6db53babc574`，兼容层代码提交为 `f9cea75`。切换时 `active_count=0`；只修改 3 个历史对比环境变量，端口、挂载和 GPU 配置保持一致。8188 生产容器未重启、未修改；8189 部署后为 healthy。

通过 Computer Use 在用户指定的 AFMAP 历史任务页面，打开 `DJI_20260715113309_0005_V_P2A3.jpeg` 并发起智能对比。算法数据库 `zhuoyi_vl_compare_test` 的同一请求证据：

| 记录 ID | 结果 | 区域数 | 推理耗时 | 语义状态 |
| --- | --- | --- | --- | --- |
| 850（修复前） | not_found | 0 | 120.332 秒 | 两次超时，无结果 |
| 851（修复后） | matched | 4 | 20.545 秒 | 共享模型切换时超时，保留几何结果并明确提示 |
| 852（修复后） | matched | 4 | 11.299 秒 | 正确描述当前新增光伏板 |

平台成功结果 ID 为 `2097151269948145666`。人工查看截图确认四个区域对应新增光伏板，分屏、卷帘拖动、放大均保持区域与照片同步。报告生成触发真实下载；生成的 DOCX 结构校验通过，包含两张图片和四处变化及新增光伏板的结论。报告 URL 为 `/upload/fileRoot/task-reports/智能对比报告_2097151269948145666.docx`。

共享 Ollama 的排队和模型切换仍可能使语义描述超过 20 秒；此时几何变化检测可正常完成，不能把提示解释为“无变化”。本次验证不构成跨季节、复杂视差或各行业场景的通用准确率评估。

### AFMAP 结果称呼统一（2026-09-08）

按用户调整，保留 AFMAP 前端“基准照片/对比照片”标签，仅在多对多兼容接口返回摘要时将几何结论和模型描述中的照片称呼统一为前端标签。AFMAP 的当前图对应“基准照片”，历史图对应“对比照片”；标准 baseline=历史图的调用按相反映射处理。一对多和独立平台仍保留当前/历史称呼，图片角色、变化方向、坐标不变。

8189 已更新至 `yoloe_service_test:history-compare-20260908-v2`（镜像 ID `sha256:e3077572ea636a54d9576ed09aad3023887585eccc3584448a79c6e691fde851`）。部署目录为 `/data/workspace/jzheng/review-test-releases/history-compare-20260908-v2`，替换原第五份 Compose override，保留原四份配置。回退本次文案调整时，将第五份 override 切回 v1 目录，仅重建测试服务。切换前活动任务为零，环境变量、挂载、端口及 GPU 配置一致，8188 未变动。

16 项相关测试通过。部署后的真实 HTTP 请求使用同一光伏照片对，耗时 9.84 秒返回 4 处变化，摘要为“基准照片相较对比照片，检测到4处可定位变化。局部变化描述（4处）：基准照片相较对比照片，屋面新增了大量深蓝色光伏板阵列，覆盖了大部分原本裸露的混凝土表面。”历史持久化记录未批量改写，需要重新对比才能生成新文案。

远端部署目录：`/data/workspace/jzheng/review-test-releases/history-compare-20260908-v1`。旧镜像保留为 `yoloe_service_test:history-compare-base-20260908`（原镜像 ID `sha256:49c6f52ba9967f18f717c8fe647ac9769fc41476008bc19af83072c7a52a3106`）。原 Compose 配置依次为主项目 `docker-compose.yml`、`resnet18-20260907-v1/deploy/resnet-test/compose.override.yml`、`voice-perf-20260907-v2/compose.override.yml`、`human-20260907-v1/compose.model-test.yml`，部署时追加本次目录的 `compose.override.yml`。回退需在无活动任务时使用原四份配置、移除本次附加文件，仅重建 `yoloe_service_test`；不得删除共享网络或使用 `--remove-orphans`。
