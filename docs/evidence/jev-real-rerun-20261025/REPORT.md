# 真机 Jev-Omni 概率复核与五类金标补充审计

日期：2026-10-25（Asia/Singapore）

## 结果

本轮复核使用独立 Jev-Omni 容器 `jev-omni-t25-verify`（`127.0.0.1:18010`，GPU L20）产出的英文 prompt 真机概率文件，未使用代理概率。142/142 条概率均存在且无错误，平均单框延迟 192.5ms。

同一 142 条车/人员金标、Tier1=0.85、Tier2.5 `confirm=0.60 / reject=0.01` 的重跑结果：

| 指标 | 现有纯 Qwen3-VL 基线 | 真机 Jev-Omni 融合 | 变化 | 门禁 |
|---|---:|---:|---:|---|
| Precision | 0.9583 | 0.9681 | +0.0098 | 通过 |
| Recall | 0.9583 | 0.9479 | -0.0104 | **失败** |
| F1 | 0.9583 | 0.9579 | -0.0004 | **失败** |
| FAR | 0.0870 | 0.0652 | -0.0218 | 通过 |
| FRR | 0.0417 | 0.0521 | +0.0104 | **失败** |
| 平均单框延迟 | 4857.3ms | 2167.8ms | -2689.5ms | **失败**（目标 <1500ms） |

融合组 TP/FP/TN/FN 为 `91/3/43/5`，比基线 `92/4/42/4` 少 1 个真目标。Tier2.5 在 142 条金标上确认 34 个真阳、0 个假阳，拒识 12 个真负、1 个真阳。

因此本轮**不修改代码、不调整默认阈值、不提交算法变更**。

## 金标覆盖审计

现有可审计人工金标覆盖如下：

| 类别 | 可用人工金标 | 当前真机概率覆盖 | 状态 |
|---|---:|---:|---|
| 车辆/车 | 142（另有 307 条 reserve） | 86 | 可用于本轮同分母对比 |
| 人员 | 56 | 56 | 可用于本轮同分母对比 |
| 违建 | 9（另有 canonical crop 106 条，但未绑定当前合图契约） | 0 | 需重新绑定合图后推理 |
| 三轮车 | 1（另有 canonical crop 4 条） | 0 | 样本不足，需扩充 |
| 工程机械（挖掘机/渣土车/装载机等） | 0 | 0 | **无人工金标，无法评测** |

`other_class_candidate_pairs.jsonl` 中的 9 条违建和 1 条三轮车具备事件、框和图像映射，可作为下一批真机 bundle 输入；当前没有工程机械人工真值，不能把检测器类别或模型输出当金标。

## 独立实例边界

- Jev-Omni 独立服务 `18010`：健康运行，概率文件来源已核验。
- 新建独立后端 `18392`：尝试使用 `review-annotation-test-api:4e3e5c7` 启动，因镜像内置 `OBJECT_STORE_REQUIRED=true` 且无 MinIO 配置，在应用导入阶段失败；未接触共享 8189/8188。
- 失败容器和临时状态目录已清理；`jev-omni-t25-verify` 未修改。

## 复现

```bash
PYTHONPATH=.:backend .venv/bin/python scripts/eval_five_class_gates.py \
  --jev-omni-probs data/ktd-clean-source-full-evaluation-20260924/jev_omni_real_probs_en.jsonl \
  --jev-omni-prob-source real --jev-omni-latency-ms 212.9 \
  --output-dir /tmp/jev-real-rerun-20261025-b
```

概率文件 SHA256：`160bc08d7af2826c672ddc40c3c92a596e30d4453afbec6c998384a914dcce49`。
