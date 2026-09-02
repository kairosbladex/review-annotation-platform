# ACCEPTANCE_REAL-STACK

## 状态

COMPLETED：Docker Compose 已在本机 Docker Desktop 中真实启动并完成端到端验收。

## 已执行的真实服务验收

本轮先发现主机没有 Docker runtime，随后经用户明确授权安装并启动 Docker Desktop。Docker 客户端/daemon 已就绪（Docker Desktop `4.89.0`，Engine `29.7.2`），并使用独立端口避开此前仍在运行的本机开发服务：

- PostgreSQL 16 Alpine：Compose 内部 `postgres:5432`；
- Redis 7 Alpine：Compose 内部 `redis:6379`；
- MinIO：`127.0.0.1:9000`（S3 API）和 `127.0.0.1:9001`（Console）；
- HTTP 模型 fixture：`127.0.0.1:8390`，容器通过 `host.docker.internal:8390` 访问；
- PostgreSQL-backed API：`127.0.0.1:8391`；
- React/Nginx 前端：`127.0.0.1:5274`；
- Celery Worker：连接 Compose Redis，稳定运行并消费模型/导出任务。

`minio-init` 已正常退出（exit 0），其余六个运行服务均为 Up，PostgreSQL/Redis 均为 healthy。API 的 Alembic 启动迁移成功。

真实 Compose 链路结果：

1. 上传 1 张 PNG 到 MinIO；
2. 通过服务 API Key 接入 review event；
3. 创建模型配置和一次批量 ModelRun；
4. Celery Worker 从真实 Redis 领取任务并调用 HTTP Adapter；
5. 模型结果和候选框成功写入 PostgreSQL；
6. 结果图成功写回 MinIO；
7. 通过 API 返回的 presigned URL 下载原图和结果图均为 HTTP 200；
8. 任务最终状态为 `succeeded`，统计为 `total=1, completed=1, failed=0`。
9. 通过真实 API 标记候选为 `false` 后，样本状态变为 `completed`，计数为 `true=0, false=1`；
10. precision 评估为 `0.0`，并明确返回 `recall_available=false`；
11. 异步导出成功，ZIP 包含 `annotations.jsonl`、`hard_negative.jsonl`、图片和标签文件，false 候选进入 hard negative。

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

本机本轮使用了端口覆盖，命令如下：

```bash
API_PORT=8391 FRONTEND_PORT=5274 docker compose config
API_PORT=8391 FRONTEND_PORT=5274 docker compose up --build -d
API_PORT=8391 FRONTEND_PORT=5274 docker compose ps
```

默认端口仍然是 API `8190`、前端 `5173`；如果这些端口空闲，可以直接执行 `docker compose up --build -d`。

MinIO smoke：

```bash
MINIO_ENDPOINT=127.0.0.1:9000 \
MINIO_ACCESS_KEY=minioadmin \
MINIO_SECRET_KEY=change-me-minio-secret \
MINIO_BUCKET=review-annotation \
PYTHONPATH=backend python3 scripts/check_minio.py
```

代码级回归验证：

```bash
PYTHONPATH=backend python3 -m pytest backend/tests -q
cd frontend && npm run build
```

## 修复项与偏差

- 修正不存在的 `minio/mc` 镜像 tag：改为官方存在的 `RELEASE.2025-02-21T16-00-46Z`；
- 补齐 Docker runtime 依赖 `httpx`，避免 API/Worker 容器导入失败；
- 增加 `MINIO_PUBLIC_ENDPOINT`，避免容器内部 `minio:9000` 生成宿主机不可访问的 presigned URL；
- 增加 MinIO S3 API `9000` 映射；
- 增加 `API_PORT`/`FRONTEND_PORT` 覆盖，允许和已有本机开发服务并行验收；
- `docker compose` 默认开发密码仍是示例值，生产部署前必须替换 `.env` 中所有 `change-me` 密钥；
- 本轮保留了此前本机等价服务及其数据，没有删除或重置任何已有数据。

## Git state

本文件对应本轮真实 Compose 验收证据。提交后应以 `git status --short --branch` 确认工作区状态。
