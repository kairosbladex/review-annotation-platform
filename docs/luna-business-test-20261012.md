# 2026-10-12 业务事件复检：代码、流程图与部署交接

## 结果与边界

**代码已推送，流程图已生成；候选版本尚未部署。** 上传候选包到 `ai-lingxi` 的操作被执行权限策略拒绝（Data Exfiltration）。该命令没有执行，因此没有创建本轮服务器发布目录、上传文件、构建镜像、改写服务配置或重启服务。不能把旧版本健康检查说成候选部署验证。

用户选择的目标是现有测试平台的**业务事件复检**，不是标注平台 `5275/8392`，也不是尚未完成的独立全流程接管。此次范围仅为兼容的复检 API/worker；初审、现有业务规则、`p000-pure-class-cards` 提示词和固定 CLI 保持不变。真实事件复检及业务保存由用户在页面操作。

## 已推送源码

| 仓库 | 分支 | 已核验远端提交 |
|---|---|---|
| review-annotation-platform | bugfix/luna-input-evidence | `dffd8d8f4a0b61075fbbf6426bbfa590f1d83211` |
| yoloe-algorithm-service | feature/luna-independent-pipeline | `6755b1e149fb1bfceb289999b4f5ab8174279bdb` |

前者是离线实验的完整覆盖门禁及实际输入摘要，不可把它的实验 worker 替换进业务服务：实验调用方传入提示词，而业务 worker 自行组装提示词。后者虽包含独立 pipeline 代码，本次仅选择其兼容业务 API/worker 部分，不启用新 pipeline。

业务候选最小部署源码：

- `api/review_agent_routes.py`
- `zhuoyi二次检测智能体/codex_luna_review.py`
- `scripts/run_codex_luna_worker.py`
- `scripts/luna_prompts/`
- `deploy/codex-luna/Dockerfile`（构建时必须显式指定当前测试镜像作为基底，不能沿用默认的更早镜像）

## 流程图

- [PNG 图片](diagrams/luna-business-review.png)
- [SVG 矢量图](diagrams/luna-business-review.svg)
- [Graphviz 可维护源文件](diagrams/luna-business-review.dot)

图表示此次核对的现有业务链路。候选新增的实际 CLI 图片存档功能在图下注明**尚未部署／启用**，没有画成线上已完成的步骤。

流程文字版：

1. 用户在事件详情发起复检，业务后端提交图片和已有候选框。
2. API 校验请求，转交 Luna worker；此分支不先运行 YOLO/Qwen。
3. 从请求图片生成图片视图、编号定位图、目标裁剪拼图。可选关联图片只作辅助，不能仅凭关联就认定严格同帧。
4. 整图全是违建时加入违建卡；全是道路裂纹时加入裂纹卡；混合或其他类别沿用旧通用提示词。
5. 固定 CLI 请求 `gpt-5.6-luna`，一次处理本图全部候选，验证返回格式与目标编号。
6. `true`（判真）映射为 `confirmed`（保留）；`false/uncertain`（判假／无法确定）映射为 `rejected`（剔除），仍保留原始三态和原因。
7. 校验错误、超时、限流、资源不足、缺目标及非法输出属于失败，不能充当模型判假。
8. API 返回逐框结论和汇总，由业务后端处理、页面展示。真实保存与页面状态仍需人工验收。

重新生成图片：

```bash
dot -Tsvg docs/diagrams/luna-business-review.dot -o docs/diagrams/luna-business-review.svg
rsvg-convert -w 1800 docs/diagrams/luna-business-review.svg -o docs/diagrams/luna-business-review.png
```

蓝色配色已通过 dataviz 配色校验，PNG 已实际打开检查中文、节点和连线；状态由文字及形状表示，不依赖颜色。

## 验证证据

- 离线实验专项：57 项通过，无跳过；不调用模型、不写业务数据。
- 业务候选本机定向回归：75 项通过（3.68 秒），范围为以下六个测试文件：
  - `tests/test_luna_input_evidence.py`
  - `tests/test_luna_worker_failure_log.py`
  - `tests/test_codex_luna_review.py`
  - `tests/test_luna_prompt_cards.py`
  - `tests/test_review_agent_routes.py`
  - `tests/test_clean_source_review_routes.py`
- 测试出现既有 requests 依赖版本警告，没有升级无关依赖。
- 完整八层 Compose 配置已在服务器只读复现，环境差异键为空；原挂载、端口、内存与 CPU 配置一致。含凭据的完整配置没有打印、下载或提交。
- 旧测试 API 通过现有 mkcert CA 完整校验 TLS 后，健康返回 `success=true/status=ok/model_loaded=true`。这是旧服务健康，不是新版本、真实模型准确率或业务入库验收。
- **未完成：**服务器候选隔离验证、镜像构建、部署切换、切换后检查、真实事件人工验收。没有新增模型准确率结论。

## 已核对的测试现场（后续发布前须重新确认）

| 项目 | 值 |
|---|---|
| 主机 | `ai-lingxi` / `192.168.130.238` |
| 业务 API | `https://192.168.130.238:8189` |
| 健康路径 | `/api/v1/health` |
| 当前容器 | `yoloe_service_test` |
| 当前镜像 | `yoloe_service_test:codex-luna-ef643c7` |
| 当前镜像 ID | `sha256:eb10b5b55337ae0c4be79d905c0e629b9e47892582303b0313d294dd99c62c73` |
| Worker | `codex-luna-review-worker.service`（systemd 用户服务） |
| Worker 监听 | `192.168.130.238:18396`，不是 localhost |
| Worker 当前源码版本目录 | `25a612f28403bed971db96430ab638508589b5f6` |
| 当前提示词 | `p000-pure-class-cards` |
| 当前 API/worker 资源上限 | 各 8 GiB、4 CPU；worker 无 swap |

原 worker 的 `50-prompt-release.conf` 继续引用 `47ca276bf77f4901627b12e81b49b5363555757b/prompt.env`。回滚应保存并恢复发布当时的原件，而不是直接删除该 drop-in，否则会误退提示词。

Compose 必须保留容器标签 `com.docker.compose.project.config_files` 所列的全部八层配置及其顺序。只用最后一层会丢失已有功能。以运行容器的环境与挂载补齐插值参数，并使用原 `/data/workspace/jzheng/.env`；不要把 token 输出到终端或 Git。

## 权限允许后继续的最小步骤

1. 重新核对两个仓库和服务器运行状态，确认候选提交未漂移；重新检查内存、磁盘、GPU 和在途复检。
2. 在 `/data` 新版本目录放置最小候选包并逐文件校验，不覆盖运行中的版本目录。先保存原镜像身份、完整 Compose 配置身份与 worker 配置，准备精确回滚。
3. 用当前镜像构建仅覆盖两个 API 文件的 overlay 镜像。服务器隔离验证必须无共享业务写入、无真实模型调用、无共享 GPU；临时容器必须限制内存、同值 swap、CPU，带 `--rm` 和任务标签。
4. 若启用候选输入存档，使用专用 0700 私有目录和明确容量上限；存档满或写入失败必须返回执行失败，不自动清理用户证据。不启用 `CODEX_REVIEW_FRAME_INDEX`（当前没有准入的业务可信原帧索引）。
5. 隔离验证通过、无在途请求后，保持原提示词、CLI、凭据、挂载和资源限制，切换 worker/API；失败则按本轮快照回滚。
6. 核验实际运行源码摘要、镜像 ID、worker 健康及可信 TLS API 健康，然后才能通知用户已部署。
7. 更新本记录与流程图中的实际部署状态；清理本轮临时验证资源。不得替用户执行真实事件复检或业务回写。

需由正常权限审批通道允许的具体动作是：**把上述最小源码候选包上传到已确认的公司测试主机 ai-lingxi 的新版本目录，随后按隔离验证和可回滚流程发布现有业务复检 API/worker。** 不更改宿主权限策略，不换传输方式绕过本次拦截。

## 人工验收重点（候选部署确认后）

- 分别测试纯违建、纯道路裂纹、混合类别与人员事件。
- 检查逐目标框、类别、真假计数及理由是否对应；`uncertain` 应保留原始含义，但按当前口径剔除。
- 失败事件不能变成“全部为假”；不要在共享环境人为制造内存耗尽或压测来验证失败分支。
- 核对重新打开事件后结果和状态是否保留，避免仅前端临时展示正确。
- 如需排查，记录事件编号、操作时间、执行编号；不要在公开渠道贴图片或令牌。
- 只有候选存档实际启用后，才检查其执行编号对应的输入 manifest 与 outcome。单次人工测试不能证明准确率提升或严格同帧。

## 资源与工作区

本轮未启动或重启服务器进程、容器或端口；没有新建工作树，复用现有主检出。用户已有标注 CSV、`docs/CODEX_PROGRESS.md` 和其他不明来源数据保持不动。持久保留内容为本文及流程图源文件、SVG、PNG；临时发布包属于未上传的中间产物，不作为已部署制品。
