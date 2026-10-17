# ACCEPTANCE_REAL-INTEGRATION

## Status

`PARTIAL` — 真实复检服务 Adapter、生产 MinIO 严格模式和生产 Compose 配置已经实现并通过本地契约验证；当前还没有拿到生产环境的实际地址、密钥和可复现图片，因此尚未声称真实环境连通性验收。

## Ticket

`REAL-INTEGRATION-REVIEW-MINIO`

## Delivered behavior

- 新增 `LegacyReviewAgentAdapter`，兼容现有复检服务的两阶段接口：先调用 `/api/v1/review-agent/yolo-detect`，再调用 `/api/v1/review-agent/review`。
- 复检请求携带 `X-API-Key`、`detections` JSON、`yolo_conf_threshold`，并支持 `rule_code`、`rule`、超时和 TLS 校验配置。
- 统一解析旧服务的 `original_*`、`corrected_*`、`reasonCode`、`errorCode`、YOLO/LLM 判定和 reasoning 字段。
- `rawDecision` 优先于产品侧 `decision`，所以旧服务将 `uncertain` 兼容映射成 `confirmed` 时，平台仍保存原始存疑结论。
- 增加 `ModelAdapterRouter`，API 和 Celery Worker 都按 `ModelProfile.adapter_type` 选择适配器；模型 profile 的配置仍可覆盖环境默认值。
- 生产环境新增 `deploy/production.env.example`、`docker-compose.production.yml` 和 `scripts/check_production_config.py`。
- 生产 MinIO 支持 region、浏览器 presigned endpoint、关闭自动建 bucket，以及 `OBJECT_STORE_REQUIRED=true` 的 fail-fast 行为；缺少 endpoint/凭证、bucket 不存在或 SDK 初始化失败时不会静默降级到空存储。
- 新增 `scripts/check_model_service.py`，可用一张代表性图片执行检测→复核探针，输出只包含计数和结果图是否存在，不输出密钥。

## Not delivered / explicit boundary

- 未把任何旧项目 `docker-compose-prod.yml` 中的明文密码、MinIO 密钥或 API Key 复制到新仓库。
- 未填入生产地址、生产密钥或生产 bucket；这些值必须在部署主机的 `deploy/production.env` 或密钥管理系统中提供。
- 尚未执行真实生产 MinIO 的读写/presigned URL 探针，也尚未对真实复检服务执行一次检测→复核调用，因为当前会话没有生产连接信息和代表性图片。
- 不修改旧 `yoloe-algorithm-service` 的业务代码，不在新平台内置 GPU/YOLOE 推理。

## Validation evidence

已执行：

- `PYTHONPATH=backend python3 -m pytest backend/tests/test_adapter_contract.py -q` → 当前 9 passed（包含真实旧服务两阶段接口的本地 HTTP 契约 fixture）。
- `PYTHONPATH=backend python3 -m pytest backend/tests -q` → 当前 15 passed。
- `PYTHONPATH=backend python3 -m compileall -q backend scripts migrations` → passed。
- `python3 scripts/check_production_config.py --env-file deploy/production.env.example` → 按预期拒绝示例占位符、非真实密钥和示例地址。
- 使用不落盘的安全测试环境变量运行 `python3 scripts/check_production_config.py` → 按预期通过严格 MinIO、legacy adapter 和密钥长度校验。
- `docker compose --env-file deploy/production.env.example -f docker-compose.production.yml config` → 通过；生产 Compose 只包含 API、Worker、前端，不包含本地 PostgreSQL/Redis/MinIO 服务。
- `git diff --check` → passed。

## Acceptance criteria

### 已满足

- 真实复检服务的检测→复核调用顺序、字段映射和 `uncertain` 保留逻辑有契约测试。
- API 和 Worker 共享同一套 Adapter 路由和生产默认配置。
- 生产 MinIO 不自动创建 bucket，不可用时明确失败，不回退到 `NullObjectStore`。
- 生产配置模板不含真实凭证，校验脚本不会打印密钥值。
- 生产 Compose 使用外部 PostgreSQL、Redis、MinIO 和模型服务。

### 待真实环境确认

- 生产 MinIO SDK endpoint 可连通、目标 bucket 已存在、读写和 presigned URL 均正常。
- Worker 网络可访问模型服务的 detection/review endpoint，并能访问其返回的结果图 URL。
- 代表性图片能获得正确的候选框、类别、置信度、原始判定、原因码和结果图。
- 用一个真实 ModelProfile 创建批量 ModelRun 后，Candidate、结果图 Asset 和逐图错误均能落库。

## Risks and follow-ups

- `MINIO_PUBLIC_ENDPOINT` 必须是标注人员浏览器可访问的地址；它可以和容器访问的 `MINIO_ENDPOINT` 不同，但必须使用同一 HTTPS 开关和可用的签名配置。
- 旧复检服务返回的结果图是相对路径时，Worker 必须能从自身网络访问模型服务；结果图不可用不会丢弃有效 Candidate，但会记录 `result artifact unavailable`。
- 生产 ModelProfile 的 `adapter_type` 必须设置为 `legacy_review_agent`；环境变量只是没有 profile 覆盖时的适配器默认值。
- 生产 bucket 需要由运维预创建，并给平台账号授予最小的对象读写权限。

## Handoff commands

```bash
cd /REDACTED_HOME/PycharmProjects/review-annotation-platform
cp deploy/production.env.example deploy/production.env
# 仅在部署主机或密钥管理注入真实值，不要提交 deploy/production.env
python3 scripts/check_production_config.py --env-file deploy/production.env
PYTHONPATH=backend python3 scripts/check_minio.py --env-file deploy/production.env
PYTHONPATH=backend python3 scripts/check_model_service.py \
  --env-file deploy/production.env \
  --rule-code waterline-v1 \
  /path/to/review-sample.jpg
docker compose --env-file deploy/production.env -f docker-compose.production.yml up --build -d
```

## Git state at handoff

本次真实接入实现已提交到本地 `master`：`69ca041 feat(integration): connect legacy review service and production minio`。本地工作树在交接时应保持 clean；该提交尚未自动 push。真实生产连通性探针仍需在获得生产配置和样例图后执行。
