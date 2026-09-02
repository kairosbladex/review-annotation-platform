# ACCEPTANCE_REAL-STACK

## 状态

PARTIAL：等价真实服务栈验收通过；Docker Compose 本身尚未执行，因为当前主机没有 Docker runtime。

## 已执行的真实服务验收

当前主机未发现 `docker`、`docker compose`、Docker Desktop、Podman、OrbStack、Finch 或 Colima。因此不能把 Compose 容器启动报告为通过。本次使用真实的本机 PostgreSQL、Redis、MinIO、Celery Worker、FastAPI 和 HTTP 模型 fixture 完成了同等关键链路验证：

- PostgreSQL 14：`127.0.0.1:55432`，数据库 `review_annotation`；
- Redis 8：`127.0.0.1:56379`；
- MinIO：`127.0.0.1:59000`；
- HTTP 模型 fixture：`127.0.0.1:8390`；
- PostgreSQL-backed API：`127.0.0.1:8290`；
- Celery Worker：连接真实 Redis，单张任务消费成功。

真实链路结果：

1. 上传 1 张 PNG 到 MinIO；
2. 通过服务 API Key 接入 review event；
3. 创建模型配置和一次批量 ModelRun；
4. Celery Worker 从真实 Redis 领取任务并调用 HTTP Adapter；
5. 模型结果和候选框成功写入 PostgreSQL；
6. 结果图成功写回 MinIO；
7. 通过 API 返回的 presigned URL 下载原图和结果图均为 HTTP 200；
8. 任务最终状态为 `succeeded`，统计为 `total=1, completed=1, failed=0`。

已执行的 MinIO smoke 输出：

```json
{"bucket":"review-annotation","object_key":"_smoke/<uuid>.txt","exists":true,"round_trip_ok":true,"presigned_url":true}
```

API/Worker 端到端输出摘要：

```text
health.status=ok
job.status=succeeded
job.total=1
job.completed=1
job.failed=0
candidate_count=1
asset_kinds=[original,result]
presigned original: HTTP 200, 68 bytes
presigned result: HTTP 200, 13 bytes
```

## 可复现命令

目标 Docker 环境可执行：

```bash
cp .env.example .env
docker compose config
docker compose up --build
```

MinIO smoke：

```bash
MINIO_ENDPOINT=127.0.0.1:59000 \
MINIO_ACCESS_KEY=rapminio \
MINIO_SECRET_KEY=rapminio-secret \
MINIO_BUCKET=review-annotation \
PYTHONPATH=backend python3 scripts/check_minio.py
```

代码级回归验证：

```bash
PYTHONPATH=backend python3 -m pytest backend/tests -q
cd frontend && npm run build
```

## 阻塞与下一步

要完成“Docker Compose 真实容器验收”，需要在本机安装并启动 Docker Desktop 或其他兼容 Docker Compose 的 runtime。安装完成后，应重新执行：

```bash
docker compose config
docker compose up --build -d
docker compose ps
MINIO_ENDPOINT=127.0.0.1:9000 \
MINIO_ACCESS_KEY=change-me-access \
MINIO_SECRET_KEY=change-me-secret \
MINIO_BUCKET=review-annotation \
PYTHONPATH=backend python3 scripts/check_minio.py
```

本次没有修改旧项目，没有删除本机真实服务数据，也没有将等价真实服务结果冒充 Compose 验收结果。

## Git state

本文件对应本轮真实栈验收证据。提交后应以 `git status --short --branch` 确认工作区状态。
