# ACCEPTANCE_GITLAB

## Status

COMPLETED：项目已创建并推送到公司 GitLab，最新默认校验流水线已真实通过。

## Ticket

`GITLAB-REVIEW-ANNOTATION-PLATFORM`

## Delivered behavior

- 在公司 GitLab 创建了私有项目 `wangxiaolin/review-annotation-platform`；
- 本地 `master` 已配置 remote：`origin` → `git@192.168.110.2:wangxiaolin/review-annotation-platform.git`；
- 当前 `master` 已推送到 `origin/master`；
- 新增 `.gitlab-ci.yml`，对 push 和 Merge Request 自动执行后端、Compose 配置和前端构建校验；
- 新增 `compose-integration` 手动作业，用于需要 Docker-in-Docker/privileged executor 的真实 Compose 验收；
- 创建并注册了项目专用 Runner `review-annotation-platform-local`，标签为 `infra-238`，锁定到当前项目；
- 未将生产数据库、MinIO、模型服务或 JWT 密钥写入仓库或前端构建产物。

## GitLab pipeline evidence

最新流水线：

- Pipeline `#82`：
  `http://192.168.110.2/wangxiaolin/review-annotation-platform/-/pipelines/82`
- Commit：`6a04d5f`（`docs: record gitlab pipeline acceptance`）；
- `backend-test`：passed；
- `compose-config`：passed；
- `frontend-build`：passed；
- `compose-integration`：skipped/manual，按设计不阻塞普通分支和 Merge Request 校验。

此前失败已被定位并修复：

- Pipeline `#77`：GitLab YAML 将带冒号的 shell 命令解析成 mapping；已通过 block scalar 修复；
- Pipeline `#78`：Runner 初始处于等待状态；已创建项目专用 Runner 并注册到本机；
- Pipeline `#79`：Shell executor 没有 `python` 命令；CI 改用 `python3`；
- Pipeline `#80`：Compose 配置 job 仍有一处 `python` 调用；已改为 `python3`。

## Repository files

- `.gitlab-ci.yml`：流水线定义；
- `docs/CICD_GITLAB.md`：job、触发策略、Runner 和凭据说明；
- `docs/ACCEPTANCE_GITLAB.md`：本次 GitLab 推送和验收记录。

## Validation evidence

本地验证：

```bash
python3 - <<'PY'
from pathlib import Path
import yaml

for path in (".gitlab-ci.yml", "docker-compose.yml"):
    yaml.safe_load(Path(path).read_text())
print("YAML OK")
PY

PYTHONPATH=backend python3 -m pytest backend/tests -q
npm --prefix frontend run build
```

结果：

- GitLab CI 结构校验通过；
- 后端测试 `10 passed`；
- 前端 TypeScript/Vite 构建通过；
- GitLab Pipeline `#82` 的三个自动 job 全部通过；
- Docker Compose PostgreSQL、Redis、MinIO、API、Worker、前端真实栈验收记录见 [ACCEPTANCE_REAL-STACK.md](ACCEPTANCE_REAL-STACK.md)。

## Not delivered

- 没有配置自动部署、生产环境发布或模型服务自动替换；
- 没有把真实模型服务 API Key、生产 MinIO 凭据或生产数据库密码写入 GitLab；
- 没有在公司基础设施上部署长期 Runner 服务；当前项目 Runner 由本机用户态进程承载，适合当前验证和内部开发。

## Risks and follow-ups

- 本机项目 Runner 必须保持运行，后续 pipeline 才能立即领取；如果本机进程退出，GitLab 作业会回到 pending。长期使用建议让公司 Runner 管理员提供在线的 `infra-238` Runner，或将项目 Runner 安装为明确受控的用户服务；
- `compose-integration` 使用 Docker-in-Docker，需要 privileged Docker executor；当前 Shell Runner 不执行该手动作业。真实 Compose 已在本机完成，因此它不会掩盖当前自动校验结果；
- CI 依赖安装已固定使用 `https://pypi.org/simple`，避免本机默认镜像缺少 `redis<6`；如公司网络要求内部 PyPI，应改为 GitLab 允许访问的内部镜像；
- SSH 推送时服务器提示未使用 post-quantum key exchange，这是公司 GitLab/SSH 服务端能力提示，不影响本次推送成功，后续可由基础设施侧升级 OpenSSH；
- Compose 默认密钥只用于开发和 CI 临时环境，生产部署前必须由部署平台 Secret 管理覆盖。

## Git state

```text
branch: master
tracking: origin/master
latest validated commit: 6a04d5f
working tree: clean after the final documentation commit
```
