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

## 线上运行（2026-10-01 起，服务器 ai-lingxi）

目录 `/data/workspace/jzheng/recheck-autotrain/live/`：`state/`（任务、报告、台账、审计）、`runs/<任务>/`（训练产物与评测输出）、`logs/`（`runner.log`、`brain.log`、`brain/<时间>.log`）、`src/`（本目录代码的部署副本）、`assets/`（评测基线与标签）。

- 执行器：`autotrain-runner.timer` 每 2 分钟；训练容器 24GB/8 核、评测容器各 7GB/8 核，均断网、uid 1000、只用 L20 `GPU-c977647c`，同时只有一个任务。
- 大脑：`autotrain-brain.timer` 每小时 05、35 分唤醒 Hermes（`qwen3.6`，在共享服务 L20 上按需加载，空闲显存 < 30GB 时跳过）；容器 4GB/2 核。
- 首次真实闭环：`exp-001`（自然比例抽样）自动完成训练 → 评测 → 对比 v1：真车 179/180、63/67 与 v1 持平，误检放过 8/19（v1 7），未通过第 2 条。

### 停止入口

```bash
# 1. 立即停止：拒绝新任务，执行器下次运行（≤2 分钟）停掉本系统所有容器
touch /data/workspace/jzheng/recheck-autotrain/live/state/STOP
# 2. 停止定时（不再唤醒 Hermes、不再推进任务）
systemctl --user disable --now autotrain-brain.timer autotrain-runner.timer
# 3. 自动：2026-10-08 之后不再唤醒 Hermes，也不再接受新训练
```

恢复：删除 `STOP` 文件并 `systemctl --user enable --now autotrain-runner.timer autotrain-brain.timer`。

### 已知行为

- Hermes 常在一次唤醒中连续完成多步（例如写两份报告后再提交训练），与提示词“每轮一步”不一致；资源与参数限制在工具和执行器中强制，不受影响。
- 看图观察仅作线索，可能出现无法核实的描述；结论以 `metrics/<任务>.json` 为准。

## 本轮结束（2026-10-07）

结论：在现有训练数据（车类误检 18 个）上，只调抽样与权重找不到同时优于 v1 的配置；误检权重越高，误检放过越少、真车转人工越多。保持 v1，不替换。

| 实验 | 抽样 | 随机真车确认（v1 179） | 随机误检放过（v1 7） | 难例真车确认（v1 63） | 难例误检放过（v1 13） |
|---|---|---|---|---|---|
| exp-000（v2） | 等权 | 167 | 4 | 54 | 8 |
| exp-001 | 自然比例 | 179 | 8 | 63 | 13 |
| exp-002 | 封顶 1.0 | 178 | 7 | 61 | 13 |
| exp-003 | 封顶 1.5 | 175 | 6 | 59 | 12 |

- Hermes 于 10-02 00:07 按“连续 3 次未通过”规则写总结并停止提交；10-02 提出修改建议：未通过时提示人工评估合格标准优先级或转向数据层面（难例挖掘、数据增强）。
- 设计缺陷：停止提交后定时器仍每 30 分钟唤醒大脑，10-02 至 10-07 空转 306 次，每次在共享服务 L20 上加载约 23GB。下次应在触发停止规则或无事可做连续 N 次后由执行器自动停用定时器。
- 已执行：`STOP` 文件、`systemctl --user disable --now autotrain-brain.timer autotrain-runner.timer`。代码、状态、报告、台账与 4 个检查点保留在服务器 `live/`（runs 179MB）。
- 下一步不在本系统内：补充误检样本（楼顶设施、路面标线、三轮车等）的人工标注后再训练；本系统可复用（新数据集替换 `DATASET` 并扩展 experiment_space）。
