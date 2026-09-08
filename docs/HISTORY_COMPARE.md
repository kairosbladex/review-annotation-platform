# 历史对比

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

### 任务范围和纯色差过滤（v3，2026-09-08）

蓝色屋顶样本记录 880 原先返回两处区域，模型摘要明确说第二处“无变化”，但旧兼容层既未传递用户要求，也不根据模型判断筛除区域。v3 将请求的 `task_description`、`image_description`、`compare_goal` 直接传给区域复核，不再把旧整图提示词作为新引擎输入。用户明确指定的比较对象优先于“批量识别变化”等通用目标。

几何层先排除近似恒定局部 RGB 偏移：轮廓内部腐蚀后至少 30 像素，扣除局部中位偏移的残差第 90 百分位小于 10 时视为纯外观变化。随后模型逐区域返回对象相关性和 keep/unrelated/color_only/unchanged/uncertain；只有明确相关且 keep 的区域出框。道路形状不变、仅颜色/明暗变化、无关对象和配准窄边不保留；光伏板新增/缺失等真实物体变化仍保留。

**此策略取代上述 v1/v2 在 AFMAP 中的失败降级行为**：模型失败、漏答、非法行号或超时不会展示未经确认的候选框，也不会返回“无变化”，而是 `not_found` 加人工复核提示。每次最多复核面积最大的 8 处，超额或 uncertain 区域明确提示未确认；已确认区域仍可展示。独立平台未指定任务的描述接口仍沿用原失败保留几何结果行为。

24 项相关测试通过，覆盖纯色差、真实物体变化、任务相关性、超时、漏答与非法返回。部署前完整兼容路由验证：同一蓝色屋顶照片对，屋顶任务只保留 1 处屋顶变化（9.313 秒），道路形状任务为 0 处（7.718 秒）；原光伏样本 4 处新增仍保留（最终提示词验证 13.491 秒）。这些是有限回归样例，不代表各场景准确率；其他路面样本仍需以原图验证。

测试镜像为 `yoloe_service_test:history-compare-20260908-v3`，ID `sha256:39cc95ed734aeaadea81de194c4ab54fbe611c85c4f9650c664a54824ffb4f25`。第五份 override 使用 `/data/workspace/jzheng/review-test-releases/history-compare-20260908-v3/compose.override.yml`，其余四份保持不变。回退此过滤策略时将第五份 override 切回 v2，确认无活动任务后仅重建 8189。旧记录不回写，重新发起比较才能生成筛选后的区域。

已部署且 8189 healthy。上线后真实 HTTP 复验：蓝色屋顶任务为 1 个屋顶框（8.785 秒），道路形状任务为 0 个框（7.642 秒）。8188 容器身份与启动时间核验未变动。

远端部署目录：`/data/workspace/jzheng/review-test-releases/history-compare-20260908-v1`。旧镜像保留为 `yoloe_service_test:history-compare-base-20260908`（原镜像 ID `sha256:49c6f52ba9967f18f717c8fe647ac9769fc41476008bc19af83072c7a52a3106`）。原 Compose 配置依次为主项目 `docker-compose.yml`、`resnet18-20260907-v1/deploy/resnet-test/compose.override.yml`、`voice-perf-20260907-v2/compose.override.yml`、`human-20260907-v1/compose.model-test.yml`，部署时追加本次目录的 `compose.override.yml`。回退需在无活动任务时使用原四份配置、移除本次附加文件，仅重建 `yoloe_service_test`；不得删除共享网络或使用 `--remove-orphans`。
