# Luna 当前版本 baseline 评测（tests/luna-live-baseline）

本目录是「Luna 当前版本基线评测」的执行/计分脚本。评测对象是 ai-lingxi 上**当前实际运行**的
`codex-luna-review-worker`（提交 `25a612f`，提示词版本 `p000-pure-class-cards`），
冻结数据来自 yoloe `feature/luna-class-prompt-release` 提交 `85d10ae` 的
`data/luna-prompt-autotune-final-20261009/`（最终标签 dev 685 / holdout 443 框）。

> 本轮**只跑模型推理**，不改提示词、不写业务数据、不操作 18396 共享 worker。

## 正式范围（协调方 2026-10-11 更正）

| 项 | 值 |
|---|---|
| 划分 | 仅历史 holdout |
| 图数 N | 137 |
| 有标签框 | 443 |
| 送出的全部候选 | 452（每图全部候选一次请求） |
| 轮次 | 2（同一版本重复运行） |
| 计划请求 | 274 |
| 失败重试 | 每请求最多 1 次，**跨 resume 也不得突破** |
| 总尝试上限 | 548（4N） |
| 并发 | 2 |
| 客户端资源 | systemd-run --user：MemoryMax=1G、MemorySwapMax=0、CPUQuota=100% |
| worker 资源 | 本任务独立实例：MemoryMax=4G / CPUQuota=200% / MemorySwapMax=0，绑定 127.0.0.1:18398（实测 CPUQuotaPerSecUSec=2s）；共享 worker 为 8G/400% 但本任务未使用 |

dev 289 图与无标签池**不在本轮范围**。

## 公开 / 私有边界

- **私有（self-gitignore，不入库）**：`data/luna-live-baseline/private/`
  - `frozen/`（items.json、labels.json、提示词等原始资料）
  - `grouping-private.json`（unit→已核实 task/航次哈希，协调方提供）
  - `prior_calls.jsonl`、`server_probe.txt`、`preprocess_hash.json`
  - `results/formal/`（含业务标识的逐框结果与原始台账）
- **入库（去标识/聚合）**：`data/luna-live-baseline/`
  - `manifest.json`、`provenance.json`、`REPORT.md`
  - `results/deidentified/round*.json`（稳定哈希键）、`ledger_summary.json`、
    `metrics_aggregate.json`、`report_tables.md`、`prior_summary.json`
- 原始模型自由文本只留在服务器 `raw/`，不进入本地、不入库、不进模型上下文。

## 文件

| 文件 | 作用 |
|---|---|
| `extract_frozen.py` | 按固定提交从 yoloe 提取冻结资料到 private/，核对哈希并写 `provenance.json` |
| `freeze_manifest.py` | 生成聚合冻结清单 `data/luna-live-baseline/manifest.json`（无业务标识） |
| `routing_tests.py` | 不调模型：纯类别/混合路由一致性与候选编号检查 |
| `evaluate.py` | 隔离评测客户端：每图全部候选一次请求、持久台账、按(round,unit)额度、STOP、资源检查、resume |
| `test_evaluate.py` | 单元测试：编号映射、台账、重试、STOP、预算、resume、**跨进程每单元额度** |
| `ledger_util.py` | 台账**顺序配对**：保留重复 (round,unit,attempt)，未知 attempt_uid 不回退误配，额外尝试按 (phase,unit) 计数 |
| `test_ledger.py` | 单元测试（6 项）：重启重复编号、在途保留、未知 UID 拒绝误配、超预算去标识 |
| `score_baseline.py` | 冻结官方计分：逐类、交叉表、翻转、按图 CI、已知航次子集 CI、覆盖/失败、共同覆盖子集 |
| `test_score.py` | 单元测试：交叉表/派生指标/翻转/按图与按航次 CI |
| `repro_history.py` | 复现历史 p000/p000b/p001 官方数字 |
| `preprocess_hash.py` | 服务器上按线上源码原样跑图像预处理并记录哈希（不发模型请求） |
| `deidentify_results.py` | 把私有结果转为可入库去标识产物（哈希 unit/key），并自检无泄漏 |
| `build_report.py` | 生成中文报告 `data/luna-live-baseline/REPORT.md` |
| `metrics_official.py` | 冻结官方 `metrics.py` 的可入库纯代码副本（来源见 provenance） |
| `summarize_ledger.py` | 汇总 append-only 调用台账（含在途未配对的尝试） |
| `server_probe.sh` | 只读核实线上 worker/CLI/模型/提示词/资源事实（不打印密钥） |
| `run_isolated_worker.sh` / `stop_isolated_worker.sh` | 启动/停止本任务独立 worker |
| `run_formal_client.sh` | systemd-run 限资源启动正式客户端 |
| `fetch_results.sh` | 取回脱敏结果到 private/ |

## 复现命令

```bash
# 0. 提取冻结资料 + provenance（需要本地 yoloe 仓库）
YOLOE_REPO=../yoloe-algorithm-service python3 scripts/luna_live_baseline/extract_frozen.py
python3 scripts/luna_live_baseline/freeze_manifest.py

# 1. 不调模型的路由与编号检查
python3 scripts/luna_live_baseline/routing_tests.py

# 2. 本地单元测试（不需要模型/网络/凭据）
python3 scripts/luna_live_baseline/test_evaluate.py
python3 scripts/luna_live_baseline/test_score.py

# 3. 历史口径复现（需先按 rescore.py 解出 /tmp/luna_final）
python3 scripts/luna_live_baseline/repro_history.py --autotune-final /tmp/luna_final

# 4.（服务器）只读核实线上事实
ssh ai-lingxi 'bash -s' < scripts/luna_live_baseline/server_probe.sh

# 5.（服务器）独立限资源 worker + 客户端
ssh ai-lingxi 'bash -s' < scripts/luna_live_baseline/run_isolated_worker.sh
ssh ai-lingxi 'bash -s' < scripts/luna_live_baseline/run_formal_client.sh
ssh ai-lingxi 'bash -s' < scripts/luna_live_baseline/stop_isolated_worker.sh

# 6. 取回、计分、去标识、报告
bash scripts/luna_live_baseline/fetch_results.sh
python3 scripts/luna_live_baseline/score_baseline.py --splits holdout \
  --grouping data/luna-live-baseline/private/grouping-private.json
python3 scripts/luna_live_baseline/deidentify_results.py
python3 scripts/luna_live_baseline/build_report.py \
  --metrics data/luna-live-baseline/private/results/formal/metrics.json \
  --manifest data/luna-live-baseline/manifest.json \
  --ledger data/luna-live-baseline/private/results/formal/logs/formal_ledger.jsonl \
  --prior data/luna-live-baseline/private/prior_calls.jsonl \
  --out data/luna-live-baseline/REPORT.md
```

## 运行版本 vs 交付版本（P1 修复）

- **本次正式运行使用**的服务器脚本 sha256：
  `61d41f91e2bd462d91d634c3195263a0e343c8efcf6cde03066d4ecfabc5d6b3`
  （总预算正确，但 `one(unit)` 对失败/中断单元仍从 attempt 0 循环，存在恢复后超限风险。
  最终完整台账确认本轮每单元最多 2 次，没有触发超限；2 次中断尝试已单列）。
- **交付修复版** sha256：
  `408dafdb1d345ccc2186453c391ac9a0e32bc58099d911a5612e35fe1553582a`
  新增 `Ledger.attempts_for(round, unit)`：按持久台账统计已发次数（含在途），
  `range(already, max_retries+1)`，resume 绝不突破每单元上限；新增跨进程恢复单元测试。
  运行中的客户端**未替换**。

## 口径与已知局限

- 线上规则：只有 `true` 放过；`false`/`uncertain` 都算驳回；accuracy 只在成功返回的框上计算，
  失败单列、不计对错，覆盖率单独报告，绝不静默剔除；未完成写「待完成」，不用 0 代替。
- 冻结 `items.json` 的 `group == unit`，**不能把 event/group 当航次**。按图聚类未解决同航次相关；
  已知航次子集 CI 单独给出并标注覆盖率（holdout 76/137 图可映射，61 图未知）。
- 协调方 Codex 只读核验：历史 dev 中至少 67 个目标与 holdout 同航次（含人员 64 个），
  旧 holdout **不是**按航次独立的新盲测。本任务不据此声称泛化改善。
- holdout 443 框不含道路裂纹；只评测已有候选框的真假分类，不代表全图召回率或 mAP。
- 临时进程违规与本任务全部调用（smoke、错误启动、中断、在途）均记入台账并在报告中列出，未隐藏。
