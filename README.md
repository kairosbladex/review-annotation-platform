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

## Docker Compose

Compose 使用 PostgreSQL、Redis、MinIO、FastAPI、Celery Worker 和 Nginx 前端：

```bash
cp .env.example .env
# 编辑 .env，替换所有 change-me 值
docker compose up --build
```

- API：<http://localhost:8190>
- 管理后台：<http://localhost:5173>
- MinIO Console：<http://localhost:9001>

容器启动时会先执行 `alembic upgrade head`。MinIO 初始化容器创建 `review-annotation` bucket。浏览器不直接读取公开对象地址，API 返回短期 presigned URL。

确认真实 MinIO 连接和读写权限：

```bash
PYTHONPATH=backend python3 scripts/check_minio.py
```

脚本只写入一个 `_smoke/` 小对象用于回读校验，不触碰已有业务对象；验证输出包含 bucket、对象 key、回读结果和 presigned URL 是否生成。

## 主要 API

管理接口均需要 `Authorization: Bearer <token>`，登录：

```bash
curl -X POST http://localhost:8190/api/v1/auth/login \
  -H 'Content-Type: application/json' \
  -d '{"username":"admin","password":"change-me-admin-password"}'
```

复检服务回传必须使用独立的服务密钥，且 `event_id` 幂等：

```bash
curl -X POST http://localhost:8190/api/v1/ingest/review-events \
  -H 'Content-Type: application/json' \
  -H 'X-Service-Key: change-me-service-api-key' \
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
  --database-url 'postgresql+psycopg://review:password@localhost:5432/review_annotation'
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
