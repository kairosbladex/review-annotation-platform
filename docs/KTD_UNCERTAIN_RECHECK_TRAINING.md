# KTD 存疑目标二次复检与训练数据

本流程在首轮 KTD 图片复检完成后，只对首轮 `uncertain` 的候选框再请求一次视觉模型。二次运行使用独立的 `run_id` 和输出目录，首轮结果保持不变。

## 执行

```bash
cd /Users/wangxiaolin/PycharmProjects/review-annotation-platform
PYTHONPATH=backend python3 scripts/run_ktd_gpt_recheck.py \
  --env-file .env.ktd.local \
  --model-env-file .env.auto-annotation.local \
  --database-url sqlite:///./data/ktd-gpt-recheck.sqlite3 \
  --output-dir data/recheck-runs/ktd-uncertain-recheck-20260903 \
  --uncertain-only-from ktd-gpt-recheck-20260902 \
  --batch-size 10 \
  --concurrency 2 \
  --resume
```

断点续跑必须继续使用同一个 `--output-dir`、`--uncertain-only-from` 和 `--resume`。正式全量执行前可以加 `--pilot-images 10`，确认真实图片、overlay 和结构化响应正常后，再移除该参数继续执行。

## 输出文件

- `candidate_results.jsonl`：每个二次复检候选的结果、首轮证据、二轮证据和源行 ID。
- `training_data_summary.json`：训练视图统计、标签策略、类别和血缘汇总。
- `yoloe_training_manifest.jsonl`：仅包含 `training_label=true` 的正样本候选框；每行保留归一化 bbox 和 `class_name`。
- `hard_negative.jsonl`：`false` 候选框，不能写入 YOLO 正样本标签，但可用于误检分析或 hard-negative 流程。
- `qwen_vl_training.jsonl`：包含 true/false 的候选框判别对话，图片仍以源对象 key 引用。
- `uncertain_excluded.jsonl`：二次后仍 uncertain 的候选，明确排除出训练集。
- `training_manifest.jsonl`：只保留人工确认的 gold 数据；自动复检结果不会伪装成人工真值。
- `class_map.json`：按 `class_name` 排序生成的本数据集内部数字映射；业务区分以 `class_name` 为准。
- `yolo_labels/<image_id>.txt`：仅由 `true` 候选生成的 YOLO 标签文件，按图片聚合。
- `yolo_image_manifest.jsonl`：原图对象路径与对应标签文件的下载清单。

## 训练边界

`yoloe_training_manifest.jsonl` 和 `yolo_labels/` 可作为检测训练数据的候选清单；GPU 侧仍需根据 `yolo_image_manifest.jsonl` 下载 `image_bucket + image_object_key` 对应的原图。`qwen_vl_training.jsonl` 适合训练候选框真假/类别复核器，不等于完整图像检测标注。

自动二次复检数据属于 pseudo label；首轮和二轮均来自模型时不能宣称人工准确率，也不能据此计算完整 recall 或 mAP。二次后仍 uncertain、源框冲突或图片失败的记录应保留在排除/人工复核队列。
