# 2026-10-12 业务事件复检：代码、流程图与部署交接

## 结果与边界

**2026-10-12 14:42（UTC+8），最小业务复检版本已发布到公司测试主机 `ai-lingxi / 192.168.130.238`，测试 API 与 Luna worker 均完成版本及健康核验。** 本轮用户明确授权上传 `6755b1e`、隔离验证及可回滚更新。此前上传权限拒绝属于上一轮记录；本轮正常上传成功，没有修改宿主权限策略。

此次仅更新现有业务事件复检 API/worker。初审、现有业务规则、`p000-pure-class-cards` 提示词、固定 CLI、原 API 环境与挂载均保留。未上传或启动 `luna_pipeline/`，未主动发起真实业务任务、真实事件复检、Luna 模型请求或业务结果保存；API 沿用原有启动预热。标注平台 `5275/8392` 不在发布范围。

**当前能确认的是代码与工程链路部署成功。真实事件页面、结果保存和识别准确率尚未验收，不能据此声称准确率提升。** 没有可信业务原帧索引，因此 `CODEX_REVIEW_FRAME_INDEX` 继续不配置；关联图片仍不能当成严格同帧原图。

## 已推送源码及最小制品

| 仓库 | 分支 | 已核验远端提交 |
|---|---|---|
| review-annotation-platform | bugfix/luna-input-evidence | `dffd8d8f4a0b61075fbbf6426bbfa590f1d83211` |
| yoloe-algorithm-service | feature/luna-independent-pipeline | `6755b1e149fb1bfceb289999b4f5ab8174279bdb` |

前者是离线实验 worker，不用于替换业务服务。后者由 `git archive 6755b1e` 提取以下最小文件；包含在该提交内的独立全流程代码不属于本次发布。

- `api/review_agent_routes.py`
- `zhuoyi二次检测智能体/codex_luna_review.py`
- `scripts/run_codex_luna_worker.py`
- `scripts/luna_prompts/` 下三个当前提示词文件
- `deploy/codex-luna/Dockerfile`

另上传六个定向测试文件及两个固定夹具，共 15 个源码/测试文件，逐文件 SHA-256 与提交内容一致。API 镜像仅覆盖前两个 Python 文件，以切换前实际镜像 ID 为基底构建，未使用 Dockerfile 中较早的默认基底。worker 从新版本目录启动，三个提示词文件与旧 worker 目录逐字相同；六组纯类别、混合类别和别名输入的最终提示词也逐字相同。

原始候选包 SHA-256：`effa8fe3b8c8d09d060446e54fff4f611633f44ed16a09f4f01f5f3b334b0f58`。候选包已清理，源码与摘要清单保留在发布目录。

## 当前运行身份与配置差异

| 项目 | 本轮核验结果 |
|---|---|
| 业务 API | `https://192.168.130.238:8189`，健康路径 `/api/v1/health` |
| API 容器 | `yoloe_service_test`；2026-10-12 14:42:07（UTC+8）启动 |
| 当前镜像 | `yoloe_service_test:luna-business-6755b1e` |
| 当前镜像 ID | `sha256:63877b8ca42c9f29fbf0ddf5cb9bc8754fcf52eaecee07b7c83006c63e658ee7` |
| Worker | 用户服务 `codex-luna-review-worker.service`，`192.168.130.238:18396`；14:42:04 启动 |
| 发布目录 | `/data/workspace/jzheng/codex-luna-review/releases/6755b1e-business` |
| 提示词 | `p000-pure-class-cards`，沿用原 `47ca276…/prompt.env` |
| CLI | `codex-0.162.0-alpha.17.2/codex`；SHA-256 `73b73b048546993298234a295de01ef2424e7a8b29def9d0ecb1ca7cb77500d4` |
| API/worker 上限 | 各 8 GiB、4 CPU；API 不额外使用 swap，worker `MemorySwapMax=0` |
| 最终输入存档 | `/data/workspace/jzheng/codex-luna-review/input-evidence/6755b1e-business`，0700，最多 536870912 字节（512 MiB） |

API 保留原八层 Compose 配置、顺序、两个环境文件、所有环境变量、挂载、端口与 GPU 配置；仅追加发布目录的 `compose.image.yml` 指定候选镜像。原八个文件摘要均未变化。不要只用最后一层 Compose 启停，也不要使用 `--remove-orphans`，同项目标签下有无关服务。

Worker 保留原服务单元和 `50-prompt-release.conf`，新增 `60-business-release.conf` 改启动源码路径，并增加 `CODEX_REVIEW_EVIDENCE_DIR`、`CODEX_REVIEW_EVIDENCE_MAX_BYTES`。除 systemd 每次启动产生的进程元数据外，环境差异仅这两个存档字段；原 CLI、凭据和提示词配置不变。

存档保存实际传给 CLI 的图片、manifest 与 outcome；满额或写入失败返回执行失败，不自动删除证据，不把失败变成“判假”。本轮没有真实调用，交付时该目录为空。启用此功能不代表已经取得真实输入或严格同帧证据。

## 本轮验证证据

- 服务器候选镜像中，六个定向测试文件 **75/75 通过**（3.792 秒）：`test_luna_input_evidence.py`、`test_luna_worker_failure_log.py`、`test_codex_luna_review.py`、`test_luna_prompt_cards.py`、`test_review_agent_routes.py`、`test_clean_source_review_routes.py`。镜像无 pytest，使用标准库 unittest 执行同一批测试，没有安装或升级依赖。
- 独立 HTTP 链路 **6/6 工程检查通过**：六组提示词等价性、实际 worker HTTP、实际合成 CLI 子进程、API 三态映射、输入与 outcome 摘要、存档容量拒绝时返回 503。判定输出是测试替身，真实模型调用 0、业务写入 0，不是准确率测试。
- 两类验证容器均断网、无 GPU、无共享业务数据/凭据挂载，只使用合成图片与假密钥；上限 4 GiB、2 CPU、128 进程，`--rm`，带 owner/purpose 标签。未启动独立全流程。
- 切换前核对 worker 无子进程、8189/18396 无已建立业务连接；批量任务接口返回 0 个活动任务。API 原有 MySQL 启动检查可能检查表结构，因此先用限时只读事务核对：表和所需字段已存在，无会触发旧结构迁移的 `class_name` 字段，事务回滚；没有扩大数据库权限或执行业务查询/写回。本检查不是全数据库零写入审计。
- 发布后通过现有 mkcert CA 完整校验 TLS，API 返回 `success=true/status=ok/model_loaded=true`；worker 返回既有模型、提示词和 only-true 策略。核对实际镜像 ID、API 两个文件摘要、worker 进程源码路径与文件摘要、环境差异及资源上限。
- 本轮前后 **59/59 个无关运行容器**的 ID、镜像、启动时间和重启次数一致；没有操作生产 API、规则服务、推理 Java 或标注平台。

私有发布目录保存 `manifest.json`、`preparation.json`、`isolated-tests.log`、`isolated-http.json`、`schema-precheck.json`、`deployment-receipt.json`、`rollback-receipt.json`、`deployment-history.json` 与 `evidence-sha256.json`。完整运行环境和回滚快照仅保存在服务器 0700 私有目录，未下载或提交凭据。

部署回执 SHA-256：`c72401f3143e8bee824e15e99a92aa69f872688309fdbb9622bdd4f60a83cc72`。

## 切换中的问题与回滚验证

本轮有两次发布脚本问题，均已修正，未修改候选业务源码：第一次 worker 重启后过早检查监听，连接拒绝，自动恢复旧 worker，API 未切换；第二次候选健康正常，但 Docker 挂载列表顺序变化被误判为内容变化，触发恢复旧 API/worker。随后修正为按挂载目标排序比较，并在 14:41:54 核验旧镜像、旧 worker、提示词与健康均已恢复。最终 14:42:26 候选发布验收成功。

这验证了本轮旧版本可恢复；没有通过反复调用业务任务测试回滚。最终发布脚本保留有界就绪等待、精确配置检查与失败回滚。

## 发布复核与回滚操作

旧镜像 `yoloe_service_test:codex-luna-ef643c7` / `sha256:eb10b5b55337ae0c4be79d905c0e629b9e47892582303b0313d294dd99c62c73` 以及旧 worker `25a612f28403bed971db96430ab638508589b5f6` 目录均保留。原 `50-prompt-release.conf` 不应删除，否则会误退提示词。

只读复核已部署版本（不发送复检请求）：

```bash
ssh ai-lingxi 'python3 /data/workspace/jzheng/codex-luna-review/releases/6755b1e-business/release.py verify'
```

需要回滚时，在无在途请求后执行以下命令；脚本会核对原配置摘要、旧制品和当前候选身份，有漂移则停止，不覆盖未知改动：

```bash
ssh ai-lingxi 'python3 /data/workspace/jzheng/codex-luna-review/releases/6755b1e-business/release.py rollback'
```

回滚仅移除本次 `60-business-release.conf`，恢复旧 worker，并以保存的原八层 Compose 参数与环境恢复旧 API 镜像；随后检查 TLS 健康、源码路径、资源与配置。既有输入存档不删除、不重放，原提示词 drop-in 不动。不要直接打印私有 `rollback/` 内的环境快照。

## 流程图与人工验收

- [PNG 图片](diagrams/luna-business-review.png)
- [SVG 矢量图](diagrams/luna-business-review.svg)
- [Graphviz 源文件](diagrams/luna-business-review.dot)

流程：用户发起复检 → 业务后端提交图片/候选框 → API 校验 → worker 生成视图并存档 → 沿用当前类别卡与固定 CLI → 校验目标编号及三态输出 → `true` 保留，`false/uncertain` 剔除但保留原三态 → API 返回 → 业务页面及保存由用户验收。执行失败不等于模型判假。

建议用户手动验收纯违建、纯道路裂纹、混合类别和人员事件：核对目标框、类别、计数、原因及重新打开后的状态是否保留。记录事件编号、操作时间和执行编号，再只读核对输入 manifest/outcome；不要在公开渠道贴原图或令牌。Codex 本轮未替用户执行这些操作，也没有新增准确率结论。

重新生成图片：

```bash
dot -Tsvg docs/diagrams/luna-business-review.dot -o docs/diagrams/luna-business-review.svg
rsvg-convert -w 1800 docs/diagrams/luna-business-review.svg -o docs/diagrams/luna-business-review.png
```

## 资源与工作区

本轮临时验证容器均已删除，没有临时监听、GPU 占用或独立验证实例残留。已删除远端上传 tar 与一次性准备脚本；保留约 268 KiB 的候选源码、测试、必要回执、发布/回滚脚本和私有配置快照。输入存档目录为空，保留供用户后续手动验收。旧镜像及旧源码保留用于回滚，不删除既有资源。

交付时测试 API 约用 1.516 GiB 内存，worker 约用 13 MiB；二者是本次获授权更新的长期测试服务，保留运行。主机可用内存约 207 GiB，未出现 OOM；没有保留额外临时服务。

复用两个仓库主检出，没有新建工作树。算法仓库未改源码；本仓库仅更新本文和现有流程图。用户已有标注 CSV、`docs/CODEX_PROGRESS.md`、算法仓库未跟踪脚本及其他数据保持不动。
