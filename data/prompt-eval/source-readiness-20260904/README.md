# KTD Source Readiness Audit

审计日期：2026-09-04

## 结论

`formal_training_ready=false`。

当前候选框数据引用的全帧图片存在高度 overlay 风险，且现有 SQLite、MySQL 本地副本和导入/导出链路都没有可恢复的 clean-frame 映射或本地 task 分组。因此，这批数据只能用于探索性实验，不能用于证明正式训练带来了实质准确率提升。

## 图片检查

全帧缓存共有 744 个以源 payload SHA 命名、但经 Pillow `quality=95` 重新编码的 JPEG
衍生文件；`0/744` 文件名与缓存文件自身 SHA-256 相等。以下只读颜色筛查针对这些重编码
像素，不是源对象字节完整性证明：

| 检查项 | 结果 |
|---|---:|
| 全帧缓存文件 | 744 |
| 缓存自身 SHA 与文件名一致 | 0 / 744 |
| 命中强红色像素规则 | 743 / 744 |
| 强红色像素超过 250 个 | 709 / 744 |
| 唯一未命中样本的人工复核 | 仍有紫红框和文字 overlay |

JPEG 重编码会改变源字节，并可能轻微改变像素值；颜色规则仅是 advisory 风险筛查，
不能单独证明每张图都存在 overlay，也不能证明未命中图片就是 clean。它可能把自然红色
计为命中，也可能漏掉紫红色、绿色或其他样式的框和文字。本次唯一的颜色漏检样本经
人工查看仍存在 overlay，因此该筛查没有产生可接受的 clean 候选。

## 来源映射缺口

- SQLite `assets` 有 1,510 行、755 个不同 object key；全部标为 `original`，没有 `comparison` 资产，`sha256` 均为空。
- SQLite 样本元数据中的 `source_file_path` 与资产 object key 相同；ingestion payload 没有 `comparison_image`。
- MySQL 本地副本只有 `sys_annex_file` 表。可疑替代路径 `file_lit_path` 与 `file_path` 全部相同，`file_finish_path`、`event_id`、`img_param` 为空；其余业务标识在本地没有可用于解析 clean frame 的关联表。
- KTD 导入器把 `file_path` 直接写成图片 object key；prompt-eval 导出器只读取 `assets.kind='original'` 并获取同一 object key 的字节。

这里的 `original` 只是平台内的资产角色名，不是“像素未叠框”的来源证明。

## 分组与时间缺口

- 上游只读 join 可为当前本地 755 张图恢复 80 个 task，但本地同步和平台 metadata 没有保存这层关系；`ktd-annex-<annex_id>` 只是导入幂等键。
- 仅使用无标签的 selection/locked inputs 与上游 task 映射检查，215 张 gold-input 图覆盖 30 个 task，其中 22 个 task 同时跨两个 split，影响 198 张图。
- 现有 `capture_time` 来自附件 `create_time`，不能冒充上游事件或帧时间；当前也没有 video ID 或 track ID。

因此，现有 image-SHA 零交集只排除了完全相同的图片，不能证明同任务或相邻帧隔离。

## 正式训练门禁

解除阻塞至少需要以下不可变证据：

1. 可读取且哈希匹配的 clean-frame bytes，以及 `clean_object_key`、`clean_sha256`、宽高和对象版本信息。
2. 对应 annotated-frame 的 object key、SHA-256，以及 clean/annotated 配对来源、方法和版本。
3. 可审计的 capture/frame 标识或时间戳，并能唯一回溯到上游采集帧。
4. annotated 坐标到 clean-frame 像素坐标的明确变换，且所有候选框通过边界校验。
5. `overlay_status=clean|overlay|unknown`、检查器版本和证据；`overlay` 或 `unknown` 不进入正式训练或评测。
6. 按 `clean_sha256` 分组切分数据，并证明 train、selection、locked-test 之间无图像泄漏。
7. 同步真实 task（或更粗的 session/video）上下文和上游事件时间；按来源连通组切分，并证明组 ID 零交集。

在这些门禁通过前，训练运行、loss 下降或当前评测分数变化都不能升级为正式准确率提升结论。

`artifact_manifest.json` 固定本说明与机器报告的 SHA-256/bytes，只用于发现落盘漂移；
它不保存上游数据库或对象字节，因此不能把本次只读审计变成可独立重放的数据快照。

## 本次边界

- 边界事故：一次通用 auditor 命令误对真实 v1 快照执行；当时的旧实现打开并解析了
  `gold_records.jsonl` 与 `locked_test_gold.jsonl`。标签内容未打印、未在对话中呈现，也未用于训练、
  prompt 调优、候选选择或本报告结论；但“文件内容被打开”本身已违反严格隔离边界。
- 修复后 auditor 默认完全不打开 `authoring.jsonl`、`gold_records.jsonl`、
  `locked_test_gold.jsonl` 和 `quarantine.jsonl` 这四个标签承载文件；只比较 manifest 与
  `SHA256SUMS` 中已记录的哈希，并以 stat 检查字节数，不宣称校验 protected 当前字节。
  只有显式传入 `--include-protected-gold` 才会打开、解析并验证这些 protected 内容。
- 未运行任何训练。
- 未写入生产数据，也未调用生产写接口。
- 本目录只保存来源就绪度的审计说明和机器可读报告。
