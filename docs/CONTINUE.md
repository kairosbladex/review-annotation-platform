# Continue Here

本项目已完成 PLATFORM-001 至 PLATFORM-005 的首个可运行切片，下一次继续开发时按下面顺序推进。

## 已有能力

1. FastAPI + SQLAlchemy domain schema，SQLite 可本地运行，Compose 使用 PostgreSQL。
2. `POST /api/v1/ingest/review-events` 事件幂等、资源缺失显式记录。
3. MinIO 前缀/manifest 导入和页面 multipart 上传，按对象和 SHA-256 去重。
4. React/TypeScript 标注工作台：队列筛选、图片代理、归一化红框、逐框真/假、备注、撤销、键盘快捷键。
5. `ModelProfile`、`ModelRun`、Celery Worker、HTTP Adapter/LegacyReviewAgentAdapter、逐样本失败隔离。
6. JSONL、YOLO ZIP（原图 + 每样本合并标签文件）和候选框 precision 分桶评估。
7. Alembic 初始迁移、Compose 依赖、旧 SQLite 迁移脚本、单管理员审计。

## 下一步建议

- 已补本地 HTTP Adapter Contract Test；并已在 Docker Compose 真实栈运行 `PYTHONPATH=backend python3 scripts/check_minio.py`，验证目标 MinIO 的 bucket、读写和 presigned URL。
- 为页面上传和 manifest 导入增加批次详情页与导入错误下载。
- 将 PostgreSQL/Redis/MinIO 用 Docker Compose 集成测试跑进 CI。
- 已接入真实 MinIO 结果图写回：外部 `preview_url` 会下载并持久化为 `model-runs/.../result.jpg`，并由 API 返回短期 presigned URL。
- 已完成真实复检服务接入准备：LegacyReviewAgentAdapter 会按检测→复核顺序调用旧服务，保留 `rawDecision=uncertain`；生产配置模板、严格 MinIO 模式、生产 Compose 和两个连接探针见 `docs/ACCEPTANCE_REAL-INTEGRATION.md`。
- 已完成 Docker Compose 真实验收：PostgreSQL、Redis、MinIO、API、Worker、前端均已启动；详见 `docs/ACCEPTANCE_REAL-STACK.md`。
- 增加导出历史列表、短期下载链接和精度曲线可视化。
- PLATFORM-006：整图标注、漏检/无目标确认、改框/改类，再实现 recall/mAP。

## 重要不变量

- 不把 uncertain 目标静默归为 false；平台只保存复检服务给出的原始 decision。
- 生产环境使用 `MINIO_AUTO_CREATE_BUCKET=false`、`OBJECT_STORE_REQUIRED=true`，bucket/凭证不可用必须启动失败。
- 不覆盖任何旧 ModelRun/Candidate；不同模型版本并列。
- 标注任务的 completed 必须建立在所有候选目标已 label 的基础上。
- 一期 precision 不得对外命名为完整识别准确率。

## 验证命令

```bash
PYTHONPATH=backend python3 -m pytest backend/tests -q
cd frontend && npm run build
```
