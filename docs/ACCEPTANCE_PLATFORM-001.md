# ACCEPTANCE_PLATFORM-001

## 做了什么

- 建立独立 `review-annotation-platform` Git 仓库，不依赖旧项目内部路由和 Python 模块。
- 提供 FastAPI API、React/TypeScript 工作台、SQLAlchemy 领域模型、Alembic 入口和 Docker Compose。
- Compose 定义 PostgreSQL、Redis、MinIO、API、Celery Worker、Nginx 前端及 MinIO bucket 初始化。
- 提供单管理员 JWT 登录、服务间 `X-Service-Key` 校验和审计事件。
- 本地开发默认 SQLite + NullObjectStore 降级，部署环境通过 `.env` 切换 PostgreSQL/MinIO。

## 没做什么

- 没有把 YOLOE/GPU 推理代码复制进新项目；模型必须通过 Adapter 接入。
- 没有实现二期的新增/移动/缩放/删除框、修改类别、漏检标注、多人分配。
- 没有自动训练、自动替换生产模型或自动调整线上阈值。

## 如何验收

后端：

```bash
cd /REDACTED_HOME/PycharmProjects/review-annotation-platform
PYTHONPATH=backend python3 -m pytest backend/tests -q
```

前端：

```bash
cd /REDACTED_HOME/PycharmProjects/review-annotation-platform/frontend
npm install
npm run build
```

Compose：

```bash
cd /REDACTED_HOME/PycharmProjects/review-annotation-platform
cp .env.example .env
docker compose config
docker compose up --build
```

验收时重点检查：

1. 未登录无法访问样本、模型、导出接口；没有服务密钥不能提交复检事件。
2. 同一 `event_id` 重试返回同一个 sample，不重复创建候选目标。
3. 详情页红框点击、真/假标注、备注、撤销和刷新后状态均保持。
4. 第二个模型运行不会覆盖第一条运行，新的候选框进入新的标注任务。
5. 有未标注目标时导出任务失败且说明原因；YOLO ZIP 每张图片只有一个合并标签文件，false 框不会进入正样本。
6. 评估接口返回 candidate precision，并明确 `recall_available: false`。

## 可能的问题

- 尚未在当前主机实际启动 Docker Compose；需要有 Docker daemon 和网络拉取镜像。
- HTTP Adapter 对外部模型服务的响应字段做了 `predictions/detections` 兼容，但真实服务仍需补合同测试。
- 前端没有使用真实图片 fixture 做浏览器 E2E；画布在真实 MinIO 对象存在时显示图片，否则明确显示资源缺失。
- `migrations/versions/0001_initial.py` 是不可破坏的 metadata baseline，后续 schema 变更应写显式 Alembic migration。

## Git state

- 当前仓库为独立新仓库，默认分支 `master`。
- 完成验收后应创建小而完整的本地 commit；不自动 push。
