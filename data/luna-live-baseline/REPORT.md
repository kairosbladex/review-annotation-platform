# Luna 当前版本基线评测报告（tests/luna-live-baseline）

- 生成时间：2026-10-11T02:15:31+08:00
- 评测对象：ai-lingxi `codex-luna-review-worker`，提交 `25a612f28403`，提示词 `p000-pure-class-cards`，模型 `gpt-5.6-luna`
- 正式范围：holdout，137 图 / 443 有标签框 / 452 候选，2 轮

## 0. 结论摘要

- 第 1 轮 holdout 总体准确率 **81.94%**（有效 443/443 框，覆盖 100.00%）；第 2 轮 **81.26%**（有效 443 框，覆盖 100.00%）。
- 两轮同一候选三值翻转 65/443 = 14.67%；真假翻转 29 = 6.55%。
- 两轮准确率差 0.68%；按图聚类 95% 区间见下表。该对比是同版本随机波动，不是版本改进。
- 本轮只测**当前版本基线**，不比较候选提示词、不声称泛化改善。

## 1. 评测对象与只读核实

| 项 | 值 |
|---|---|
| worker_unit | `codex-luna-review-worker.service` |
| effective_execstart | `/usr/bin/python3 /data/workspace/jzheng/codex-luna-review/releases/25a612f28403bed971db96430ab638508589b5f6/scripts/run_codex_luna_worker.py` |
| effective_code_commit | `25a612f28403bed971db96430ab638508589b5f6` |
| effective_code_branch | `feature/luna-worker-failure-log` |
| prompt_env_file | `/data/workspace/jzheng/codex-luna-review/releases/47ca276bf77f4901627b12e81b49b5363555757b/prompt.env` |
| prompt_version | `p000-pure-class-cards` |
| model | `gpt-5.6-luna` |
| model_provider_base_url | `https://sub.butterfly2sea.us` |
| model_reasoning_effort | `none` |
| cli_version | `codex-cli 0.162.0-alpha.17.2` |
| cli_sha256 | `73b73b048546993298234a295de01ef2424e7a8b29def9d0ecb1ca7cb77500d4` |
| worker_source_sha256 | `b909136b16f0f357e58f9d3e221482e122195b34e7cec124bf0760aca59495e1` |
| shared_worker_limits | `MemoryMax=8G CPUQuota=400% MemorySwapMax=0（线上/共享 worker，本任务未操作）` |
| isolated_worker_limits | `MemoryMax=4G CPUQuota=200% MemorySwapMax=0 @127.0.0.1:18398（本任务独立实例，实测 CPUQuotaPerSecUSec=2s）` |
| formal_client_limits | `MemoryMax=1G CPUQuota=100% MemorySwapMax=0` |
| worker 并发上限 | 20 |
| 共享 worker | 192.168.130.238:18396 (共享，本任务不使用)；本任务未操作，MainPID 未变 |

> 10-10 的 `25a612f` / `p000-pure-class-cards` 仅为线索，已按只读核实确认为**当前实际运行版本**（与发布 `SHA256SUMS` 逐字节一致）。

## 2. 冻结数据与清单

- 冻结标签：`labels.json` sha256 `c990580379d13f4a…`，共 1128 框，划分 {'holdout': 443, 'dev': 685}。
- 正式评测集：137 图 / 443 有标签框 / 452 候选。
- 冻结 items.json sha256 `392e88ff8997dad7…`；文件哈希核对通过：True。
- 官方计分复现：p000 holdout 0.7562、p000 dev 0.7299、p000b dev 0.7124、p001 dev 0.7608、p000b-holdout 0.7675（见 `repro_history.py`）。

## 3. 方法与隔离

- 每张图**全部候选一次请求**，`target_index=0..n-1`；客户端按编号回映射，假 worker 逆序返回用例已通过。
- 本任务独立 worker（**不是**共享 worker）：MemoryMax=4G CPUQuota=200% MemorySwapMax=0 @127.0.0.1:18398（本任务独立实例，实测 CPUQuotaPerSecUSec=2s）；客户端：MemoryMax=1G CPUQuota=100% MemorySwapMax=0。共享 worker 限额 MemoryMax=8G CPUQuota=400% MemorySwapMax=0（线上/共享 worker，本任务未操作），本任务未使用。
- 并发 <= 2；每失败请求最多重试 1 次；总尝试上限 4N；append-only 台账每尝试 fsync；STOP 文件存在即停；每 60s 检查整机可用内存与共享容器健康，可用 <15% 或共享不健康即写 STOP。
- 不操作共享 worker 18396；不改提示词；不写业务数据；不部署、不重启共享服务。

## 4. 执行台账

- 正式台账 attempt=276（{'round1': 139, 'round2': 137}），attempt_result=274，按尝试计的状态 {'ok': 274, 'in_flight': 2}。
- 计划 2N=274，上限 4N=548；实际尝试 276，额外尝试（按 (round,unit) 超过 1 次计）2 次。
- 进程被杀时的在途请求 2 个（已记为调用，结果丢失；unit 以哈希列出）：
  - round1 unit_hash=e8b86f2ae3f4ad85 attempt=0 uid=None
  - round1 unit_hash=4b62cde5086328a0 attempt=0 uid=None
- 据实际台账核验：**没有任何单元超过每单元 2 次上限**。
- **临时进程违规记录（不隐藏）**：2026-10-11 01:46 前后，本任务曾用 `nohup python3` 启动**未限资源**的客户端（且当时范围仍含 dev）；01:50 被协调方叫停。随后改回 systemd-run 限资源（客户端 MemoryMax=1G / CPUQuota=100%，worker 独立限额）。该次全范围部分运行与 smoke 的全部调用已记入 prior 台账（见 `results/prior_summary.json`），未从记录中移除。
- 运行版本 vs 交付版本：本次正式运行用 evaluate.py sha256 `61d41f91e2bd462d…`；交付修复版 `408dafdb1d345ccc…`（新增按 (round,unit) 持久额度与 attempt_uid，resume 不突破每单元 2 次）。运行中客户端**未替换**；据实际台账核验，超过每单元 2 次上限的单元数 = 0，额外尝试 2 次（预期 274 唯一成功 + 2 中断多尝试 = 276）。
- 正式范围外的历史调用台账（prior）：attempt=12，按尝试计状态 {'client_keyerror_lost': 2, 'ok': 8, 'client_killed_lost': 2}；含 smoke 修复前 2 次 KeyError 丢失、修复后 2 次成功、更正前全范围部分运行 8 次（其中 2 次在途被杀）。

## 5. 两轮结果

### 第 1 轮

| 冻结框 | 有效框 | 覆盖 | 金标是 | 金标不是 | 准确率 | 真目标召回 | 非目标误放率 | 真目标驳回 | 非目标放过 | uncertain |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 443 | 443 | 100.00% | 368 | 75 | 81.94% | 84.24% | 29.33% | 58 | 22 | 71 |

| 类别 | 框数 | 金标是 | 金标不是 | 准确率 | 真目标召回 | 非目标误放率 | uncertain |
|---|---:|---:|---:|---:|---:|---:|---:|
| 车 | 306 | 292 | 14 | 89.22% | 92.47% | 78.57% | 23 |
| 人员 | 60 | 24 | 36 | 50.00% | 4.17% | 19.44% | 30 |
| 违建 | 50 | 33 | 17 | 84.00% | 78.79% | 5.88% | 8 |
| 两轮车 | 9 | 8 | 1 | 55.56% | 50.00% | 0.00% | 5 |
| 光斑 | 5 | 3 | 2 | 80.00% | 100.00% | 50.00% | 0 |
| 光伏板 | 5 | 5 | 0 | 100.00% | 100.00% | — | 0 |
| 卡车 | 4 | 1 | 3 | 25.00% | 0.00% | 66.67% | 2 |
| 三轮车 | 3 | 1 | 2 | 66.67% | 0.00% | 0.00% | 3 |
| 没戴头盔 | 1 | 1 | 0 | 100.00% | 100.00% | — | 0 |

| 金标 | 判 true | 判 false | 判 uncertain |
|---|---:|---:|---:|
| gold_true | 310 | 14 | 44 |
| gold_false | 22 | 26 | 27 |

按图（unit）聚类 95% 区间：中位 81.87%，区间 [74.75%, 87.50%]（131 图 / 443 框）。**未解决同航次相关，不能据此证明泛化提升。**
已核实航次子集聚类 95% 区间：中位 83.70%，区间 [69.27%, 91.63%]；覆盖 图 71/131、框 275/443、分组 34（未知航次的图被排除，未静默当作独立）。

### 第 2 轮

| 冻结框 | 有效框 | 覆盖 | 金标是 | 金标不是 | 准确率 | 真目标召回 | 非目标误放率 | 真目标驳回 | 非目标放过 | uncertain |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 443 | 443 | 100.00% | 368 | 75 | 81.26% | 84.24% | 33.33% | 58 | 25 | 70 |

| 类别 | 框数 | 金标是 | 金标不是 | 准确率 | 真目标召回 | 非目标误放率 | uncertain |
|---|---:|---:|---:|---:|---:|---:|---:|
| 车 | 306 | 292 | 14 | 88.89% | 92.81% | 92.86% | 21 |
| 人员 | 60 | 24 | 36 | 51.67% | 4.17% | 16.67% | 33 |
| 违建 | 50 | 33 | 17 | 82.00% | 78.79% | 11.76% | 7 |
| 两轮车 | 9 | 8 | 1 | 44.44% | 37.50% | 0.00% | 6 |
| 光斑 | 5 | 3 | 2 | 80.00% | 100.00% | 50.00% | 0 |
| 光伏板 | 5 | 5 | 0 | 100.00% | 100.00% | — | 0 |
| 卡车 | 4 | 1 | 3 | 0.00% | 0.00% | 100.00% | 1 |
| 三轮车 | 3 | 1 | 2 | 66.67% | 0.00% | 0.00% | 2 |
| 没戴头盔 | 1 | 1 | 0 | 100.00% | 100.00% | — | 0 |

| 金标 | 判 true | 判 false | 判 uncertain |
|---|---:|---:|---:|
| gold_true | 310 | 11 | 47 |
| gold_false | 25 | 27 | 23 |

按图（unit）聚类 95% 区间：中位 81.22%，区间 [73.40%, 87.24%]（131 图 / 443 框）。**未解决同航次相关，不能据此证明泛化提升。**
已核实航次子集聚类 95% 区间：中位 83.06%，区间 [68.72%, 90.98%]；覆盖 图 71/131、框 275/443、分组 34（未知航次的图被排除，未静默当作独立）。

## 6. 两轮一致性与翻转

| 同候选 | 三值翻转 | 三值翻转率 | 真假翻转 | 真假翻转率 |
|---:|---:|---:|---:|---:|
| 443 | 65 | 14.67% | 29 | 6.55% |

- 两轮共同覆盖子集口径：共同框 443/443，全覆盖=True；共同子集第 1 轮 81.94%、第 2 轮 81.26%。两轮均 100% 覆盖，全量与共同子集一致。

## 7. 失败与有效覆盖

- 第 1 轮：图 137，最终失败图 0；最终结果文件保留的尝试 137，完整台账尝试 139。中断前的尝试以完整台账为准。
- 第 2 轮：图 137，最终失败图 0；最终结果文件保留的尝试 137，完整台账尝试 137。中断前的尝试以完整台账为准。

## 8. 局限与未知

- **航次不独立**：协调方 Codex 会话只读核验 发现：历史 dev 中至少 67 个目标与 holdout 同航次（含人员 64 个）；旧 holdout 不是按航次独立的新盲测。 dev/holdout 的独立性被削弱；冻结 items.json 中 group==unit，只有图片级键，没有可直接复算的航次键，故本任务不独立重算该数字，而在报告中明确标注该局限。
- 航次可映射覆盖率（holdout）：图 76/137（未知 61 图）；有标签框 映射 275/443，已核实分组 37 个。未映射部分不纳入航次聚类。
- 冻结 items.json 的 group==unit，聚类区间只能按图片（原片/帧）；**不能把 event/group 当成航次**。已知航次子集 CI 只覆盖可映射部分并标注覆盖率，未映射的 61 图（168 框）不纳入航次聚类，不能把已知航次子集的区间推广到整个集合。
- 137 张执行图中 6 张没有二元金标；计分及图片聚类只覆盖有金标的 131 张。已知分组的 76 张执行图中 71 张有金标，因此计分子集为 71/131 图、275/443 框。
- holdout 443 框不含道路裂纹，dev 也无道路裂纹；纯类道路裂纹卡本轮不命中。
- 只评测已有候选框的真假分类，不代表全图召回率或 mAP。
- 同一提示词两次运行存在随机波动（见翻转率），两轮差异不应直接归因于版本。
- 模型账号/上游渠道为既有 Luna 渠道；未购买额度、未扩权。

## 9. 证据与复现

- 指令：见 `scripts/luna_live_baseline/README.md`。
- 冻结来源与哈希：`data/luna-live-baseline/provenance.json`、`data/luna-live-baseline/manifest.json`。
- 可入库去标识结果：`data/luna-live-baseline/results/deidentified/round*.json`（稳定哈希键）、`ledger_summary.json`、`metrics_aggregate.json`、`report_tables.md`。
- 私有（gitignore，不入库）：`data/luna-live-baseline/private/`，含线上只读核实 `server_probe.txt`、图像预处理哈希 `preprocess_hash.json`、原始 items/labels、含业务标识的逐框结果与原始台账。
- 原始模型自由文本只保留在服务器 `raw/`，不进入本地或模型上下文。
