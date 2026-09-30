# 复检微调模型自动迭代（Hermes Agent）

由服务器上的 Hermes Agent 按提示词定时推进“训练 → 评测 → 报告 → 选下一个实验”。Hermes 只能调用 `mcp_server.py` 中的工具；
训练与评测由宿主机 runner 执行（待实现），资源限制写死在 runner 中。

| 文件 | 作用 |
|---|---|
| `prompt.md` | 每轮提示词（合格标准由 `metrics.py` 判定，Agent 不能修改） |
| `mcp_server.py` | 受控工具：状态、提交训练、申请评测、读结果、看错判样本、写报告/台账/提示词修改建议；STOP 文件与截止日期即拒绝新任务 |
| `metrics.py` | 候选与 v1 对比，四条合格标准 |
| `make_samples.py` | 错判样本拼图（附人工标签，JPEG ≤256KB 避免 Hermes 再压缩） |
| `setup_dryrun.py`、`run_brain_trial.sh` | 模拟演练：以 v2（exp-000）真实结果为输入，比较不同大脑模型 |

## 大脑模型对比（2026-10-01 模拟演练）

| 模型 | 结果 |
|---|---|
| `qwen3.5:9b` | 不合格：第一版工具桥接下多次调用不存在的 `tool_call`；改为直接工具后读完状态即结束，不执行后续步骤 |
| `qwen3.6`（35B-A3B） | 合格：连续 3 轮分别完成报告+台账、提交训练、查看进度；数字照抄正确，合格标准判定正确，每轮 2~3 分钟 |

配置要点：`tools.tool_search.enabled: off`（Hermes 默认把 MCP 工具藏在 `tool_call` 桥接后，模型常写错嵌套格式）；
样本拼图须附人工标签，否则视觉大脑会像复检模型一样把楼顶设施看成车；ollama 的 OpenAI 兼容接口不支持 keep_alive，每次调用重新加载模型。
