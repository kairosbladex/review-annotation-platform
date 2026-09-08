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

2026-09-08 本地验证：用户提供的 4032×3024 光伏原图识别到 4 处新增区域；交换时间顺序验证缺失方向。另覆盖相同图、平移、整体亮度变化、无法配准、无效/过大文件、认证、模型超时与 Markdown JSON 返回。桌面及手机浏览器验证上传、对比、叠加层切换、交换照片和返回人工复核。此次为本地实现与验证，未部署到 8188/8189。

## 独立测试部署

2026-09-08 经用户确认，部署到本机独立 Compose 项目 `review-history-test`。入口为 <http://127.0.0.1:5275/#history-compare>，API 为 `127.0.0.1:8392`。使用本机测试账号 `admin / change-me-admin-password`，仅绑定回环地址。原有 `5274/8391` 测试栈和本机标注数据保持不变。

配置文件为 `deploy/history-compare.test.yml`，需要支持 `!override` 的 Docker Compose（已用 5.5.0 验证）。镜像 `review-history-api:683feb3` 和 `review-history-frontend:683feb3` 从干净提交 `683feb3bd36fb9fc2621c1a09a017a447a866f97` 构建。启动已构建镜像：

```bash
docker compose --env-file .env.example -p review-history-test \
  -f docker-compose.yml -f deploy/history-compare.test.yml up -d --no-build
```

该项目使用独立网络及 `review-history-test_*` 数据卷，MinIO 端口为 `127.0.0.1:9002/9003`。服务设置为 `unless-stopped`，Docker 恢复运行时自动启动。比较文件和结果仍不自动持久化，需要保留结果时在页面下载 JSON。

部署验收：实际 Nginx 入口完成原图正反向对比，均为 4 处变化，桌面和手机截图无溢出；未认证为 401，无效图片为 422，超过 20 MiB 为 413，相同图片为零变化。新 PostgreSQL 已迁移至 `0004_agent_review_fingerprint`，样本/候选框/标注仍为零。原测试栈的 16 张表记录数与部署前一致，其 API、Worker、前端启动时间仍为 2026-09-02。本次未操作远端 8188/8189 服务。
