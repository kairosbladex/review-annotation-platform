# ACCEPTANCE_KTD-GOLD-EXPORT-SNAPSHOT

## Status

`PARTIAL` — 已按「全部候选都有人工标签」的样本落了一份本地金标快照。平台 `POST /api/v1/exports` 在本机不可用。数量仍不够正式训练。

## Ticket

`KTD-GOLD-EXPORT-SNAPSHOT`

## 做了什么

查了 `data/ktd-local.sqlite3` 金标，并写出本地目录（不走 Celery / MinIO）：

- 脚本：`scripts/export_completed_gold.py`
- 产物：`data/gold-export-20260903/`（JSONL、YOLO labels、原图、`hard_negative.jsonl`、`class_map.json`）
- YOLO `class_id` 用既有 10 类 `class_name` 映射（`车=8`）。库里 annex 导入的 `class_id` 仍是空的。
- 假目标不写入正样本 `.txt`。

最新快照：

| 项 | 数量 |
|---|---:|
| 已完成样本 | 35 |
| 导出框 | 104 |
| true | 69 |
| false | 35 |

完成图内按类：车 45/15、违建 22/12、两轮车 2/2、公交车 0/2、三轮车 0/2、卡车 0/2。光伏板、没戴头盔、人员、光斑仍无完成图金标。库内未标框：没戴头盔 10、光伏板 48、公交车约 52、三轮车 78、人员 150、光斑 358。

## 没做什么

- 没有修平台导出（Celery `Connection closed by server`；HTTP 对象存储只读，不能 `put_bytes`）。
- 没有启动 GPU 训练。
- 没有把 Kimi 加权分或 8189 模型预填写成金标。
- 没有 commit / push。

## 哪里可能有问题

- `annotation_tasks.status` 有一行和实际标签不一致（stored pending / 已全部标注）。脚本已改为按标签推导完成态。
- 正样本几乎全是「车」。光伏板 / 公交车 / 没戴头盔 等仍空。
- 若干完成图的 YOLO 标签文件是空的（该图全部是假目标，这是预期）。
- 正式 `/exports` 仍坏；以后金标变多也要继续用这个脚本，或先修 dispatcher + 可写存储。

## 如何验收

```bash
python3 scripts/export_completed_gold.py --download-images
cat data/gold-export-20260903/summary.json
```

平台导出（当前应失败）：

```bash
# POST /api/v1/exports  → 503 导出任务无法入队
```

## Git state

- branch: `codex/uncertain-recheck-training-data`
- HEAD: `e104c27`
- 本切片未 commit、未 push
