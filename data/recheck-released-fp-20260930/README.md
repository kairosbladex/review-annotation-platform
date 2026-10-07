# 测试环境复检放过目标 · 误检样本收集

目标：找出测试环境复检判为“确认”（放过）但实际是误检的框，作为复检模型的训练负样本。

## 来源

- 测试库 `ktd_lx.bo_task_event` 只读查询：283 个事件带 `recheck_result`，共 1628 个目标，其中复检判为 `confirmed` 的 517 个。
- 复检版本：112 个来自当前“干净原图 + Qwen3-VL base/adapter 四视图”（prompt `2cbeee9e…`），其余 405 个来自 9 月前的旧版本（含 YOLOE 复核“存疑按产品设计计为真目标”的放行）。
- 业务处置字段 `event_result`（如“误报无需处理”）只到事件级且只有 11 个事件，不能作为逐框标签，因此逐框人工标注。

## 高清来源

- 09-24 快照事件：按归一化框 IoU ≥ 0.5 对应到清单候选，使用已核验的视频/原图匹配；434 个。
- 事件主图即 4032×3024 无人机原图（9 个事件）：使用复检记录中的 `recoveredBbox`；53 个，全部为当前版本放过。
- 排除：30 个没有高清来源（旧版本视频事件，原片匹配未成功）；7 个投影后落在匹配原片之外（`labeling-package/skipped.json`）。
- 入包 480 个（68 个事件），快照来源局部相似度中位数 0.9988、全部 ≥ 0.9；原图来源抽查框位与截图红框一致。

## 标注

- 类别：`真目标`、`误检`、`看不清`；判断黄框是否为标题所示检测类别（车、卡车、违建等）。
- 32 个框已在此前高清标注中标过，沿用（真车→真目标，屋顶设施/非车→误检）；其中 23 个属于随机流量评测集、9 个属于楼顶难例评测集，`eval_set` 列已标记，默认不进训练集，保留用于评测重训后的模型。
- 排序：当前版本放过的在前。

启动（本机 127.0.0.1）：

    python3 data/recheck-released-fp-20260930/labeling-package/label_server.py --port 8794

服务器工作目录：`/data/workspace/jzheng/claude-released-fp-20260930/`。脚本：`scripts/collect_released_recheck_targets.py`、`scripts/extract_hires_review_views.py`。

> 2026-10-07：服务器上的中间目录已清理，上面提到的服务器工作目录不再存在；本包的高清素材（frames/crops/neighbors/clips）保留在本机，未入库。需要重新生成时，从 09-24 快照 `codex-ktd-real-monitoring-20260924` 或测试库重新取原片，再运行 `scripts/extract_hires_review_views.py`。
