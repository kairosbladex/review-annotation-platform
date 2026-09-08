# Agent 基座

通用流水线基座:引擎稳定、节点可插拔、编排声明化。业务频繁改动只动
`nodes/`(实现)和 `pipelines/`(YAML),不动 `core/`。

```text
获取元数据 → 语义/视觉分析 → 落库 → 训练 → 评估 → 输出
 ingest       llm_analyze     persist   remote_*  gates     report
```

## 目录

```text
agent/core/       引擎:state / registry / pipeline / graph / runner / tracing / remote
agent/nodes/      节点实现(按阶段,一个文件一类)
agent/pipelines/  流水线 YAML(pipeline_id + version + nodes + params)
agent/tests/      基座契约测试
agent/cli.py      run / resume 入口
```

## 契约

- 节点:`RunState -> RunState`,注册进 `agent.core.registry`;同一 `run_id`
  下幂等;产物以 ArtifactRef(uri + sha256)进状态,字节不进状态;密钥
  只走环境变量,不进 state / checkpoint / 日志。
- 状态:`RunState`(TypedDict,JSON 可序列化)是跨节点唯一事实源,
  checkpointer 按 `run_id` 持久化。
- 流水线:YAML 声明节点顺序与参数;`pipeline_id + version` 记录进每次
  运行用于血缘;参数里 `{run_id}` 由运行器替换。
- 失败:节点异常被捕获进 `errors[]` 并 fail-fast;`resume` 的三种语义见
  `runner.resume` docstring(门禁回答 / 崩溃跳过已完成节点 / 失败重试)。

## 内置节点

| 节点 | 阶段 | 关键 params |
|---|---|---|
| `ingest_files` | ingest | `source_dir` `glob` `output_dir` |
| `llm_analyze` | analyze | `prompt_template` `manifest` `limit` `output_dir` |
| `persist_store` | persist | `input_file` `store_db` |
| `remote_preflight` | train | `gpu_index` `min_free_gpu_mib` `remote_root` |
| `remote_submit` | train | `command` `remote_root` |
| `remote_monitor` | train | `poll_seconds` `max_wait_seconds` `progress_file` |
| `remote_sync` | train | `sync_files` `local_dir` |
| `dataset_audit` | evaluate | `dataset_dir` `confirm_task` `require_gold` `on_block` |
| `approval_gate` | evaluate | `gate_name` `gate_message` |
| `render_report` | publish | `report_dir` |

新增业务 = `@registry.register("name")` 写一个节点 + 一份 YAML。

## 运行

```bash
python3 -m venv .venv && .venv/bin/pip install -r agent/requirements-agent.txt
set -a; source .env.agent.local; set +a   # LANGCHAIN_*, GPU_SSH_*, (AGENT_LLM_*)

.venv/bin/python agent/cli.py run agent/pipelines/local_ingest_report.yaml
.venv/bin/python agent/cli.py resume <pipeline.yaml> --run-id <run_id> \
  [--decision '{"approved": true, "decided_by": "...", "note": "..."}']

.venv/bin/python -m pytest agent/tests -q
```

## 环境变量

- `LANGCHAIN_API_KEY` / `LANGCHAIN_ENDPOINT` / `LANGCHAIN_PROJECT`:
  LangSmith tracing;缺失时自动降级为不 trace,流程照常(观测永远不杀任务)。
- `GPU_SSH_HOST` / `GPU_SSH_USER` / `GPU_SSH_KEY` / `GPU_SSH_PASSWORD`:
  远程作业节点用;密钥优先,密码兜底。
- `AGENT_LLM_BASE_URL` / `AGENT_LLM_API_KEY` / `AGENT_LLM_MODEL`:
  `llm_analyze` 的 OpenAI 兼容端点。

## 远程作业约定

作业目录 `<remote_root>/<run_id>/`:`run.sh`(固定模板渲染,不接受 LLM
生成命令)、`job.log`、`JOB.pid`、`JOB.exit`。提交幂等:活跃作业只
attach,已完成作业凭 `JOB.exit` 判定,绝不盲目重提。`remote_monitor`
本地阻塞轮询;本地进程死后 `resume` 会重新挂接远程标记。

AI-lingxi 红线:不动 8188/8189 在线复检服务,不动 `/app/pretrain` 线上
权重,训练完释放 GPU;回灌复检服务的模型必须保留 YOLOE 英文词表。

## 边界情况设计

- 重复提交:同一输入 + 同一 pipeline 版本 → 内容哈希去重(ingest 层)。
- LLM 失败/超时/非 JSON:进 `review_queue.jsonl` 转人工,绝不自动放行。
- 数据质量:`dataset_audit` 把正式训练 readiness 六条代码化为门禁。
- 人工审批:`approval_gate` interrupt 挂起,`resume --decision` 继续;
  拒绝即终止,审批意见进 `gates[]` 留痕。
- SSH 闪断:连接级重连一次;提交与监控分离,标记文件是唯一事实源。
- GPU 占用:`remote_preflight` 按 `min_free_gpu_mib` 门禁,device 配置化。
- checkpoint 膨胀:状态只放引用;`data/agent-runs/*.sqlite3` 已被 git 忽略。
