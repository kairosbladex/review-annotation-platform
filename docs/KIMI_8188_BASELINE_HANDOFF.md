# KIMI → Codex 交接：部署版复检服务基线与失败归因（2026-09-03 夜）

读者：Codex（qwen3-vl 评测闭环目标）与用户。本文档指向的全部产物都在
**kimi 专用 worktree** `/Users/wangxiaolin/PycharmProjects/review-annotation-platform-prompt-gold`
（分支 `kimi/prompt-gold-v1`）内，与 Codex 的 `review-annotation-platform-qwen-prompt-loop-v1`、
Grok 的 `review-annotation-platform-prompt-catalog` 完全隔离。本批改动**未提交 git**，
由用户早上检查后再决定是否提交。

## 1. 交付了什么

| 产物 | 路径（相对 kimi worktree） | 状态 |
|---|---|---|
| 部署服务适配器 | `tools/prompt_eval_data/service_attempts.py` | 7 个单元测试全过 |
| 失败模式分析器 | `tools/prompt_eval_data/service_analysis.py` | 已在真实数据运行 |
| 适配器测试 | `backend/tests/test_service_attempts.py` | 与既有 10 例共 17 例全过 |
| 基线报告（主交付） | `data/prompt-eval/baselines/BASELINE_8188_8189.md` | 数字已抽样核对 |
| 评分/分析数据 | `data/prompt-eval/baselines/*.{json,jsonl}` | 可复算 |

**给 Codex 的直接收益**：不用再自建"8188 结果按 candidate_id 对齐人工标签"的链路——
适配器 + 评分已跑通，8188 金标覆盖 535/535，报告可直接引用或复算。

## 2. 关键结论（详见基线报告）

1. 部署版 8188 基线（全量 gold 535）：危险误杀率 **23.4%**、误接受率 **57.9%**。
   误杀集中在 光伏板 52% / 公交车 57% / 违建 48%。
2. uncertain 5828 条 **100% 来自"长边<48px 小框弃权"规则**，生产中全部转真，
   是误接受率 0.579 的直接来源。
3. "YOLOE 未检出即假阳性"规则在 gold 上 69% 误杀（145 条 rejected 中 100 条为真目标）。
4. 8188 的 `llm_decision` 7104 条全 null —— **LLM 复核级根本没启用**，当前是纯 YOLOE 规则。
5. qwen3-vl 影子 326 次失败：260 次 `done_reason=length`（num_predict=256 触顶、
   thinking 吃光预算、content 为空），66 次 JSON 不可解析。**先把 num_predict 调到 ≥1024**
   即可挽回 ~80% 失败，这是 prompt-loop 实验前必须修的协议问题。
6. 同集对照：prior_gpt dmr 1.9% / far 24.3%；prior_qwen 难例子集（215 条）dmr 0.5% /
   far 0% —— 替换/增强 YOLOE 规则的空间很大。
7. 交叉校验：`8188 raw` 与 `prior_yolo` 三项指标逐位一致，链路可信。
8. 8189 覆盖不足（27/535），不下结论；补跑预填充后重跑适配器即可出基线。

## 3. 复现命令

```bash
cd /Users/wangxiaolin/PycharmProjects/review-annotation-platform-prompt-gold
PY=/Users/wangxiaolin/PycharmProjects/review-annotation-platform/.venv/bin/python3
MAIN=/Users/wangxiaolin/PycharmProjects/review-annotation-platform
SNAP=data/prompt-eval/snapshots/pg1-bc8c818184d8
OUT=data/prompt-eval/baselines

# 双视图转换（decision=上线判定 / yolo_decision=验证器原始三态）
$PY -m tools.prompt_eval_data.service_attempts \
  --attempts $MAIN/data/ktd-8188-kimi-prefill-20260903/attempts.jsonl \
  --gold $SNAP/gold_records.jsonl --source decision \
  --output $OUT/8188-final.decisions.jsonl --stats $OUT/8188-final.stats.json
$PY -m tools.prompt_eval_data.service_attempts \
  --attempts $MAIN/data/ktd-8188-kimi-prefill-20260903/attempts.jsonl \
  --gold $SNAP/gold_records.jsonl --source yolo_decision \
  --output $OUT/8188-raw.decisions.jsonl --stats $OUT/8188-raw.stats.json

# 评分（全量 gold / selection split）+ 明细
$PY -m tools.prompt_eval_data.scorer score \
  --decisions $OUT/8188-final.decisions.jsonl --gold $SNAP/gold_records.jsonl \
  --output $OUT/8188-final.score-gold.json --details-out $OUT/8188-final.details.jsonl
$PY -m tools.prompt_eval_data.scorer score \
  --decisions $OUT/8188-final.decisions.jsonl --gold $SNAP/selection_gold.jsonl \
  --output $OUT/8188-final.score-selection.json

# 失败模式分析
$PY -m tools.prompt_eval_data.service_analysis \
  --attempts $MAIN/data/ktd-8188-kimi-prefill-20260903/attempts.jsonl \
  --score $OUT/8188-final.score-gold.json --output $OUT/8188.analysis.json

# 先验同集评分（yolo/gpt/qwen 任一）
$PY -m tools.prompt_eval_data.scorer extract \
  --inference $SNAP/inference_records.jsonl --prior qwen --output $OUT/prior-qwen.decisions.jsonl
$PY -m tools.prompt_eval_data.scorer score \
  --decisions $OUT/prior-qwen.decisions.jsonl --gold $SNAP/gold_records.jsonl \
  --output $OUT/prior-qwen.score-gold.json

# 测试
PYTHONPATH=backend $PY -m pytest backend/tests/test_service_attempts.py backend/tests/test_prompt_eval_data.py -q
```

8189 把 attempts 路径换成 `$MAIN/data/ktd-8189-prefill-20260903/attempts.jsonl` 即可。

## 4. 给 Codex 下一步的建议（按预期收益排序）

1. **修 qwen 调用协议**：num_predict ≥1024（或关 thinking）→ 影子/实验失败率 43% → <10%。
2. **小框策略替换**：长边<48px 的 5828 条不再弃权转真，裁剪放大后走 qwen3-vl；
   这是降低 final 误接受率 0.579 的最大杠杆。
3. **上线 LLM 复核级**：8188 的 llm_decision 全 null，qwen3-vl 一级能力未启用；
   建议先在 selection split（438 条）上以本适配器口径做 A/B。
4. **"未检出即假阳性"规则降级为 uncertain**：在 qwen 接管前，可先把该规则从
   rejected 降级，能立刻把 dmr 从 23.4% 压到接近 0（代价是人工复核量上升）。
5. 提示词实验请只对照 `*.score-selection.json`；locked_test 保持未授权。

## 5. 边界与注意

- 本批产物全部在 kimi worktree，**未提交**；主仓库 data/ 只读未动；桌面报告目录未触碰。
- 若 Codex 要引用数字，请引用 `BASELINE_8188_8189.md` 或复算 JSON，不要复制二手数字。
- 适配器 schema：`prompt-gold-service-attempts/1`；分析 schema：`prompt-gold-service-analysis/1`。
- 契约补节见 `docs/PROMPT_GOLD_CONTRACT.md` 第 11 节。
