# Review Annotation Platform

独立的存疑目标人工复核与模型辅助训练数据平台。它只负责数据闭环和外部模型编排，不在平台内置 YOLOE 推理。

当前实现覆盖一期主链路：

```text
复检服务回传 / MinIO manifest / 页面上传
        ↓
Sample + Asset + ModelRun + Candidate
        ↓
队列与红框工作台
        ↓
逐框 true / false / 备注 / 撤销
        ↓
模型版本化批量运行
        ↓
YOLO ZIP / JSONL / hard-negative 数据
        ↓
候选框 precision 评估
```

## 本地启动

后端本地开发默认使用 SQLite 和内存外部存储降级，因此不安装 Docker 也可以运行 API 和测试：

```bash
cd /Users/wangxiaolin/PycharmProjects/review-annotation-platform
PYTHONPATH=backend python3 -m pytest backend/tests -q
PYTHONPATH=backend python3 -m uvicorn app.main:app --host 0.0.0.0 --port 8190
```

前端另开终端：

```bash
cd /Users/wangxiaolin/PycharmProjects/review-annotation-platform/frontend
npm install
npm run dev
```

打开 <http://localhost:5173>。默认开发账号来自环境变量：`admin / change-me-admin-password`。生产或共享环境必须修改密码、JWT 密钥、服务 API Key 和 MinIO 密钥。

## 从 ktd 测试库导入到本机标注

测试环境的 `sys_annex_file` 可以先同步到本机 MySQL，图片仍走公开前缀 `https://lxtest.zhuoyilingxi.com:8088/minio-src` + `file_path`，不把原图拷进本地 MinIO。

```bash
cp deploy/ktd-local.env.example .env.ktd.local
# 填写源库密码；本地副本默认是 docker mysql:8.0 映射的 127.0.0.1:3307

docker run -d --name review-ktd-mysql \
  -e MYSQL_ROOT_PASSWORD=local-ktd-root \
  -e MYSQL_DATABASE=ktd_lx \
  -e MYSQL_USER=ktd_local \
  -e MYSQL_PASSWORD=ktd_local \
  -p 3307:3306 mysql:8.0

PYTHONPATH=backend python3 scripts/sync_ktd_annex.py --env-file .env.ktd.local
SOURCE_IMAGE_BASE_URL=https://lxtest.zhuoyilingxi.com:8088/minio-src \
MINIO_ENDPOINT= MINIO_ACCESS_KEY= MINIO_SECRET_KEY= \
DATABASE_URL=sqlite:///./data/ktd-local.sqlite3 \
PYTHONPATH=backend python3 -m uvicorn app.main:app --host 127.0.0.1 --port 8190
PYTHONPATH=backend python3 scripts/ingest_ktd_annex.py --env-file .env.ktd.local --limit 20
```

`.env.ktd.local` 已被 git 忽略，不要提交源库密码。

## Docker Compose

### Codex 自动标注

平台的“AI 自动标注”按钮会对当前筛选队列中有候选框的样本逐图运行一次视觉复核。一次请求内包含三个独立检查视角（目标/框贴合、类别/场景、对抗误检），只有至少两票以不低于 0.6 的置信度一致时才自动写入 `Annotation`；冲突、异常和不确定结果保留待复核。提示词和门禁规则见 [`docs/AUTO_ANNOTATION.md`](docs/AUTO_ANNOTATION.md)。

公司 `zhuoyi` 网关只接受官方 Codex 客户端，因此默认通过本机 `codex exec` 调用；通用 OpenAI Responses 网关可将 `AUTO_ANNOTATION_USE_CODEX_CLI=false` 切回 HTTP Adapter。使用 CC Switch 当前配置生成本地（gitignored、0600）凭据文件：

运行 API/Worker 的环境需要把官方 `codex` CLI 放在 `PATH` 中；若部署镜像没有 CLI，任务会按失败处理并保留人工复核，不会写入不确定标签。

```bash
PYTHONPATH=backend python3 scripts/configure_auto_annotation_from_ccswitch.py
set -a; source .env.ktd.local; source .env.auto-annotation.local; set +a
PYTHONPATH=backend python3 -m uvicorn app.main:app --host 127.0.0.1 --port 8190
```

该脚本只把选定 provider 的路由、模型和 API key 写入本机忽略文件，不写数据库模型 profile、不提交仓库，也不会在日志中打印 key。默认模型为 `gpt-5.6-luna`，用于兼顾视觉能力、速度和成本；如需更高质量，可通过环境变量覆盖模型。

Compose 使用 PostgreSQL、Redis、MinIO、FastAPI、Celery Worker 和 Nginx 前端：

```bash
cp .env.example .env
# 编辑 .env，替换所有 change-me 值
docker compose up --build
```

- API：<http://localhost:8190>
- 管理后台：<http://localhost:5173>
- MinIO S3 API：<http://localhost:9000>
- MinIO Console：<http://localhost:9001>

容器启动时会先执行 `alembic upgrade head`。MinIO 初始化容器创建 `review-annotation` bucket。浏览器不直接读取公开对象地址，API 返回短期 presigned URL；容器内访问使用 `MINIO_ENDPOINT=minio:9000`，签发给浏览器的地址使用 `MINIO_PUBLIC_ENDPOINT`，生产环境需要将其改为浏览器可访问的 MinIO 域名。

确认真实 MinIO 连接和读写权限：

```bash
PYTHONPATH=backend python3 scripts/check_minio.py
```

脚本只写入一个 `_smoke/` 小对象用于回读校验，不触碰已有业务对象；验证输出包含 bucket、对象 key、回读结果和 presigned URL 是否生成。

生产环境可以直接加载部署变量执行同一个探针：

```bash
PYTHONPATH=backend python3 scripts/check_minio.py --env-file deploy/production.env
```

确认真实复检服务的检测→复核链路（需要提供一张可复现的图片）：

```bash
PYTHONPATH=backend python3 scripts/check_model_service.py \
  --env-file deploy/production.env \
  --rule-code waterline-v1 \
  /path/to/review-sample.jpg
```

探针只输出检测数量、复核数量、存疑数量和结果图是否存在，不输出 API Key 或模型响应中的图片地址。

## 主要 API

管理接口均需要 `Authorization: Bearer <token>`，登录：

```bash
curl -X POST http://localhost:8190/api/v1/auth/login \
  -H 'Content-Type: application/json' \
  -d '{"username":"admin","password":"REDACTED_FOR_GITHUB"}'
```

复检服务回传必须使用独立的服务密钥，且 `event_id` 幂等：

```bash
curl -X POST http://localhost:8190/api/v1/ingest/review-events \
  -H 'Content-Type: application/json' \
  -H 'X-Service-Key: REDACTED_FOR_GITHUB' \
  -d '{
    "event_id": "review-event-001",
    "source_service": "yoloe-review-service",
    "session_id": "session-001",
    "rule_code": "waterline-v1",
    "image": {"bucket":"review-images","object_key":"incoming/event-001.jpg"},
    "targets": [{"target_index":0,"class_name":"车","confidence":0.61,"bbox":[0.1,0.1,0.24,0.25],"raw_decision":"uncertain","reason_code":"model_conflict"}]
  }'
```

模型配置通过 `POST /api/v1/model-profiles` 保存。配置的 `config` 至少可以包含：

```json
{
  "base_url": "http://legacy-review-service:8000",
  "endpoint": "/api/v1/review-agent/yolo-detect",
  "timeout_seconds": 120
}
```

接入当前真实复检服务时，把 profile 的 `adapter_type` 设为 `legacy_review_agent`。该 Adapter 会先调用 `/api/v1/review-agent/yolo-detect` 获取候选框，再调用 `/api/v1/review-agent/review` 获取完整复核结果，并优先保存响应中的 `rawDecision`，因此服务对外的两态兼容映射不会把 `uncertain` 变成假目标：

```json
{
  "profile_key": "legacy-review-v1",
  "name": "现有 YOLOE 复检服务",
  "model_name": "yoloe-review-service",
  "model_version": "部署版本号",
  "adapter_type": "legacy_review_agent",
  "config": {
    "base_url": "http://yoloe-review-service:8188",
    "detect_endpoint": "/api/v1/review-agent/yolo-detect",
    "review_endpoint": "/api/v1/review-agent/review",
    "timeout_seconds": 120,
    "verify_tls": true
  }
}
```

生产环境使用外部 PostgreSQL、Redis、MinIO 和复检服务，不启动本地 MinIO。先复制并填写 [deploy/production.env.example](deploy/production.env.example)，实际文件 `deploy/production.env` 已被 Git 忽略；校验不会打印任何密钥：

```bash
cp deploy/production.env.example deploy/production.env
# 修改 deploy/production.env 中的地址、密码、API Key 和 bucket
python3 scripts/check_production_config.py --env-file deploy/production.env
docker compose --env-file deploy/production.env -f docker-compose.production.yml config
docker compose --env-file deploy/production.env -f docker-compose.production.yml up --build -d
```

生产 MinIO 的 SDK 地址和浏览器地址都使用 `host:port`（不带 URL scheme），由 `MINIO_SECURE=true` 控制 HTTPS。平台以 `MINIO_AUTO_CREATE_BUCKET=false` 和 `OBJECT_STORE_REQUIRED=true` 启动：bucket 不存在或 MinIO 不可用时直接失败，不会静默降级到本地空存储。浏览器访问图片使用 `MINIO_PUBLIC_ENDPOINT` 生成的短期 presigned URL。

批量运行：`POST /api/v1/model-runs`；进度：`GET /api/v1/jobs/{job_id}`；运行详情和评估分别是 `GET /api/v1/model-runs/{run_id}`、`GET /api/v1/model-runs/{run_id}/evaluation`。已完成运行才能创建正式导出：`POST /api/v1/exports`，下载支持 `/download`。

## 数据和状态约束

- `Sample` 只代表原图及来源；同一张原图可以有多个 `ModelRun`。
- `Candidate` 是不可变的模型候选框，bbox 统一为 `[x1,y1,x2,y2]` 的 0 到 1 归一化坐标。
- `Annotation` 与模型结果分离，`label` 只能是 `true`、`false` 或显式撤销后的 `null`。
- 一个标注任务只有所有候选目标都有标签后才是 `completed`；GET 序列化不会改变数据库状态。
- 新模型运行创建新的候选框和标注任务，不覆盖旧运行，也不会自动继承人工标签。
- 一期 precision 只描述“模型已经产生的候选框中有多少是真目标”，不能代表完整召回率；只有二期整图增框/漏检标注后才能计算 recall/mAP。

## 旧 SQLite 迁移

旧项目的 `review_annotation_samples` / `review_annotation_targets` 可以使用脚本迁移。脚本不会导入旧项目 Python 模块，也不会修改或删除旧数据库；本地图片会上传到新的 MinIO bucket，旧的像素坐标框会借助 Pillow 转成归一化坐标：

```bash
PYTHONPATH=backend python3 scripts/migrate_legacy_sqlite.py \
  /path/to/annotations.sqlite3 \
  --database-url 'postgresql+psycopg://review:REDACTED_FOR_GITHUB@localhost:5432/review_annotation'
```

迁移前请确认 `MINIO_ENDPOINT`、`MINIO_ACCESS_KEY`、`MINIO_SECRET_KEY` 已指向可写的目标对象存储。迁移输出 JSON 汇总 `total/imported/skipped/failed`，失败样本会打印原因，可修复后重跑；事件 ID 使重跑幂等。

## 目录

```text
backend/app/       FastAPI、领域模型、接入、标注、模型任务、导出
backend/tests/     API 和服务集成测试
frontend/src/      React/TypeScript 队列与标注工作台
migrations/        Alembic 初始 schema
scripts/           旧 SQLite 迁移工具和 MinIO smoke check
docs/              验收和交接文档
```

## 当前未实现

一期不允许移动/调整/新增/删除候选框，不自动训练或替换线上模型，不做多人任务分配；这些属于 PLATFORM-006。当前 HTTP Adapter 使用同步请求封装，Celery Worker 负责批量任务隔离、逐图失败记录和重试。
