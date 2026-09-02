# GitLab CI/CD

## 流水线内容

`.gitlab-ci.yml` 包含四个 job：

- `backend-test`：安装 `requirements.txt`，执行 Python 编译检查和 10 个后端测试；
- `compose-config`：解析 `docker-compose.yml`，尽早发现 YAML 结构错误；
- `frontend-build`：执行 `npm ci` 和 TypeScript/Vite 生产构建，并保留 `frontend/dist` 一周；
- `compose-integration`：手动触发的 Docker-in-Docker 真实栈验收，启动 PostgreSQL、Redis、MinIO、API、Worker 和前端，执行 Alembic、API health、MinIO 读写和 Nginx 页面检查。

所有 job 默认使用公司 Runner tag `infra-238`，与现有公司 GitLab Python 项目保持一致。本项目已创建一个锁定到当前项目的 `review-annotation-platform-local` Runner，并使用同一标签执行自动校验。长期运行时建议迁移到公司在线共享 Runner，避免依赖个人电脑上的用户态 Runner 进程。

## 触发规则

- push 分支自动触发 `backend-test`、`compose-config`、`frontend-build`；
- Merge Request 自动触发同样的三个校验 job；
- `compose-integration` 在流水线中以手动 job 出现，适合需要真实 Docker runtime 的验收；它是可选 job，不会阻塞普通 MR。

## 凭据和变量

本流水线没有写入生产密钥、MinIO 密钥或模型服务 API Key。Compose 集成 job 使用隔离的开发默认值，仅写入 CI 临时容器和临时卷。

后续接入真实模型服务或生产部署时，应在 GitLab 项目/组的 CI/CD Variables 中配置：

- `MODEL_SERVICE_BASE_URL`；
- `MODEL_SERVICE_API_KEY`；
- 生产数据库、MinIO、JWT 和服务 API Key。

这些变量不应提交到仓库，也不应写入前端构建产物。

当前流水线真实结果记录在 [ACCEPTANCE_GITLAB.md](ACCEPTANCE_GITLAB.md)：最新 Pipeline `#81` 的 `backend-test`、`compose-config` 和 `frontend-build` 全部通过。`compose-integration` 保留为手动作业，因为它要求 privileged Docker executor，而当前项目 Runner 是 macOS Shell executor。

## 本地复现

```bash
PYTHONPATH=backend python3 -m pytest backend/tests -q
cd frontend && npm run build
python3 -c 'import yaml; yaml.safe_load(open("docker-compose.yml"))'
API_PORT=8391 FRONTEND_PORT=5274 docker compose up --build -d
```

真实 Compose 结果记录在 [ACCEPTANCE_REAL-STACK.md](ACCEPTANCE_REAL-STACK.md)。
