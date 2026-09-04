# run-local-001：cd1 裁剪图复核器基线（2026-09-04，本机 MPS）

首个训练证据：**不依赖 qwen3-vl:8b 的裁剪图小模型路线可行**。
6 epoch（ResNet18 + 类别 one-hot 条件头，教师 GPT + 人工 train 合并，binary 强制二选一）。

## 同集对比（94 条 val 框，70T/24F，图级隔离，两模型面对的框完全相同）

| 模型 | 危险误杀率 dmr | 误接受率 far | 说明 |
|---|---|---|---|
| 部署版 8188（final 视图） | 5.7%（4/70） | **79.2%（19/24）** | uncertain 全部转真 |
| **run-local-001（6 epoch）** | 7.1%（5/70） | **45.8%（11/24）** | 无弃权档，阈值 0.5 |

结论：6 个 epoch 的本地小模型在危险误杀上已与部署版持平（4 vs 5，统计上无差异），
误接受率**几乎减半**。train_loss 仍在下降（0.032），远未训透。

## 数据与产物

- 数据集：`data/prompt-eval/crop-datasets/cd1-693db6804f05`（pg1 快照构建；
  训练 2162 教师 + 344 人工，val 94 人工；locked_test 153 图 633 框完全排除）
- 本目录：`metrics.json`（逐 epoch 指标）、`predictions_val.jsonl`、`model.pt`（git 忽略）
- 训练器：`tools/crop_verifier/train.py`（seed 42，复现命令见下）

## 复现

```bash
cd /Users/wangxiaolin/PycharmProjects/review-annotation-platform-prompt-gold
.venv-crop/bin/python tools/crop_verifier/train.py \
  --dataset data/prompt-eval/crop-datasets/cd1-693db6804f05 \
  --output data/crop-verifier-runs/run-local-001 \
  --epochs 6 --batch-size 32 --device mps
```

## 诚实的边界

- val 只有 94 框（false 仅 24），far 的 CI 宽（27.9%–64.9%）；单 seed、单阈值。
- 教师标签继承 GPT 偏置；最终指标必须只在人工金标上看（本 run 已是）。
- 模型无 uncertain 档；上线需要加弃权策略（如概率区间转人工），再与
  `8188-raw` 三态视图对比"判定的部分"才完整。
- 下一步方向：更多 epoch + 阈值扫描、按类目分层评估、扩 val（等更多人工标注）、
  在 GPU 服务器容器（torch 2.9.1+cu128 已验证可用）做正式训练。
