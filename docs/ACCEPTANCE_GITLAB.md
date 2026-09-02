# ACCEPTANCE_GITLAB

## 状态

PARTIAL：本地 GitLab 项目和流水线配置已准备好，等待推送后读取第一条 GitLab pipeline 结果。

## Ticket

`GITLAB-REVIEW-ANNOTATION-PLATFORM`

## 已完成

- 已在公司 GitLab 创建私有项目 `wangxiaolin/review-annotation-platform`；
- 已确认 SSH 登录身份为 `@wangxiaolin`；
- 已新增 `.gitlab-ci.yml`；
- 已新增 `docs/CICD_GITLAB.md`，说明 job、Runner tag、触发方式和凭据边界；
- 已在本地解析 `.gitlab-ci.yml` 和 `docker-compose.yml`；
- 已在正在运行的 Compose 容器中执行 Alembic、MinIO smoke 和前端页面探针。

## 流水线 job

- `backend-test`：Python 编译检查和后端单元测试；
- `compose-config`：Compose YAML 结构校验；
- `frontend-build`：TypeScript/Vite 生产构建和 dist artifact；
- `compose-integration`：手动触发的 Docker-in-Docker 真实依赖验收。

## 未做

- 尚未把本地 Git remote 添加到仓库；
- 尚未推送当前 master；
- 尚未读取 GitLab pipeline 的真实执行状态；
- 没有配置自动部署，也没有把任何生产密钥写入仓库。

## 风险与后续

- 流水线默认使用公司 Runner tag `infra-238`；如果个人命名空间无法使用该 Runner，需要在项目 CI/CD 设置中绑定可用 Runner 或修改 tag；
- `compose-integration` 要求 Runner 使用 privileged Docker executor；它设计为手动 job，不阻塞普通 MR 校验；
- 生产接入模型、MinIO 和部署主机时，应把密钥放在 GitLab CI/CD Variables，并补充部署 job 的明确目标和回滚策略。

## 本地验证

```bash
python3 -c 'import yaml; yaml.safe_load(open(".gitlab-ci.yml")); yaml.safe_load(open("docker-compose.yml"))'
PYTHONPATH=backend python3 -m pytest backend/tests -q
cd frontend && npm run build
```

## Git state

提交本文件时尚未配置 remote；推送完成后需要补充 remote、pipeline URL 和真实 job 结果。
