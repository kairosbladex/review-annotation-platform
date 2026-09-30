# KTD source provenance and grouping gap

## 结论

当前 KTD 数据只能证明“带候选框叠加层的单张事件图 + 人工候选框标签”链路，不能证明训练输入来自无框、无文字的 clean/raw frame，也不能证明 selection 与 locked test 已按任务、视频或轨迹隔离。

在以下门禁满足前：

- 不启动新的正式训练；
- 不把 smoke run、selection 调参或现有 image-SHA split 的结果表述为准确率提升；
- 不声称具备 video-level、track-level 或跨任务泛化证据。

本文结论未人工查看、输出或使用 `locked_test_gold.jsonl` 标签。但曾有一次旧版
auditor 误调用打开并解析 protected gold；该边界事故与修复见
`data/prompt-eval/source-readiness-20260904/README.md`。

## 已验证的现状

### 1. 本地 annex 副本没有完整分组链路

本地 MySQL 只同步了 `sys_annex_file`。当前本地筛选集为 1,510 行、755 个唯一 `file_path`：

- 每个路径恰好两行，分别为 `PRIMARYIMG` 和 `SHOTIMG`；
- 两行的 `file_path`、`file_lit_path`、`busi_id` 和 `imgxy` 相同；
- `file_lit_path` 与 `file_path` 在 1,510 行中全部相同；
- `file_finish_path`、`visual`、`geometry`、`img_param`、`callback_param` 全部为空；
- `event_id`、`serial_number`、`date_time_original`、`waypoint`、`action_number` 全部为空；
- 表中没有 `video_id` 或 `track_id` 字段。

因此：

- `annex_id` 是附件行标识；
- `busi_id` 在本数据中是一图一个的上游业务标识，不能单独当作任务、视频或轨迹分组；
- 时间、路径目录、地理坐标不能替代明确的 task/video/track ID。

### 2. 上游 task 关系存在，但没有被同步到本地

只读检查上游关系后，`sys_annex_file.busi_id` 可连接到 `bo_task_event.id`，再由 `bo_task_event.task_id` 连接任务上下文：

- 当前上游符合筛选条件的 763 张图可恢复 83 个非空 `task_id`；
- 当前本地 755 张图的固定子集全部可连接，恢复 755 个事件和 80 个 `task_id`；
- 对应 `event_time` 均非空；
- 上游 `vod_video_id` 在该批数据中为空，没有可用 video ID；
- 没有 track ID；任务记录中的 route/flight batch 也不能为该批数据提供可用分组。

这 83 个 task ID 当前只能通过上游 join 恢复。本地同步脚本只复制 `sys_annex_file`，没有复制 `bo_task_event` 或最小关系投影；导入脚本也只读取 `id/file_path/imgxy/create_time/date_time_original`，所以任务关系在本地链路中丢失。

### 3. 本地平台数据库没有保存任务分组

当前平台 SQLite 中有 1,510 个 sample：

- `source_event_id` 全部是由 annex 行 ID 构造的摄取幂等键，不是上游任务事件 ID；
- `session_id`、`camera_id`、`rule_code` 全部为空；
- `metadata_json` 只有 `annex_id` 和 `source_file_path`；后者可能是路径或签名 URL，不能直接
  进入模型侧 provenance，也不能充当来源身份；
- `capture_time` 来自附件 `create_time`，没有使用上游 `bo_task_event.event_time`。

ORM 已有 nullable 的 `session_id`、`capture_time`、`camera_id` 和开放式 `metadata_json`，因此补齐第一版 task provenance 不需要修改 `samples` 表或重写既有迁移。

### 4. 对象存储没有已确认的 clean/raw frame

本地导出的三张代表性整图均可见已经烧录到像素中的红色类别文字和紫色检测框。结合以下字段事实，当前没有已确认的 clean/raw 替代对象：

- `file_finish_path` 全空；
- `file_lit_path == file_path`；
- `PRIMARYIMG` 与 `SHOTIMG` 指向同一路径；
- 上游事件的备用图像 URL 在该批数据中为空。

当前 `assets.kind='original'` 只表示“摄取时的原始对象”，不能解释为“无 overlay 的原始传感器帧”。在取得 clean key、内容哈希、尺寸和坐标映射前，现有对象必须标记为 `annotated_overlay` 或等价的未通过状态。

### 5. 快照 aliases 不能恢复 task/video/track

现有 prompt-gold 快照有 3,474 个 canonical box、7,104 条 alias。每条 alias 只包含：

- `candidate_id`
- `sample_id`
- `model_run_id`
- `target_index`
- `confidence`

alias 没有 `annex_id`、`busi_id`、`task_id`、`event_time`、video ID、track ID 或像素 provenance。当前 split 仅对 `image_sha256` 做哈希。

仅使用无标签的 `selection_inputs.jsonl`、`locked_test_inputs.jsonl` 与上游 task 映射复核后：

- 215 张有 gold input 的图全部可映射到 30 个上游 task；
- 其中 22 个 task 同时出现在 selection 和 locked test；
- 198/215 张图属于这些跨 split 的 task。

所以“两个 split 的 image SHA 无交集”成立，但“任务级无交集”不成立。现有 locked test 不能用于证明跨任务泛化。

## 最小 companion context 契约

建议同步一个窄表或等价的不可变 JSONL，而不是复制完整业务表。每条 annex 记录一行，以 `annex_id` 为主键：

```json
{
  "schema_version": "ktd-annex-context/1",
  "annex_id": "<opaque-id>",
  "busi_id": "<upstream-event-id>",
  "task_id": "<upstream-task-id>",
  "event_time": "<ISO-8601 timestamp>",
  "terminal_id": "<nullable opaque-id>",
  "source_task_id": "<nullable opaque-id>",
  "pixels": {
    "clean": {
      "object_key": "<required for formal data>",
      "object_version": "<immutable version id>",
      "sha256": "<64 lowercase hex>",
      "width": 0,
      "height": 0
    },
    "annotated": {
      "object_key": "<existing overlay object>",
      "object_version": "<immutable version id>",
      "sha256": "<64 lowercase hex>",
      "width": 0,
      "height": 0
    },
    "geometry": {
      "bbox_coordinate_space": "annotated",
      "mapping_type": "identity|crop_resize|affine|homography",
      "annotated_to_clean": {},
      "mapping_version": "<version>",
      "verified": true
    },
    "overlay_check": {
      "method": "<human-or-versioned-detector>",
      "version": "<version>",
      "evidence_sha256": "<64 lowercase hex>"
    }
  }
}
```

约束：

- `annex_id` 必须唯一；同一 `busi_id` 的 PRIMARYIMG/SHOTIMG aliases 必须得到相同的 `task_id`、事件时间和像素引用；
- `busi_id` 只表示上游事件，不替代 `task_id`；
- `task_id` 是当前可用的最粗可靠 split group；没有真实 video/track ID 时不得构造或猜测；
- clean/annotated 两套 key、不可变对象版本、SHA-256、宽高必须来自实际对象字节；不能从文件名或数据库角色名推断；
- `geometry` 必须明确现有 `imgxy` 所在坐标系以及 annotated 到 clean 的可复现变换；仅在尺寸相同且像素几何已验证时使用 `identity`；
- companion context 必须与 annex 数据一起落到本地、内容寻址或固定版本，导出时不得临时依赖会变化的远端 join；
- 不保存连接凭据、签名 URL 或可变访问令牌。

## 最小向后兼容接入

第一阶段不需要新增 ORM 列：

1. 同步 companion context，并在导入时把权威 `task_id` 写入 `metadata_json`，把 `event_time` 写入 `capture_time`，把 `terminal_id` 写入 `camera_id`。不得把 session/video/track 冒充 task；`session_id` 仅在上游真实提供 session 时保存。
2. 在平台 `metadata_json` 中原样保留 `annex_id`、`busi_id`、`task_id`、`source_task_id` 及 pixel/geometry provenance。导出快照时仅保留契约白名单字段与 metadata 原文 SHA-256；`source_file_path` 及形似 URL/query/fragment 的标量不进入模型侧 provenance 或来源身份。保留当前 annex 行幂等键，避免改写既有 `source_event_id` 唯一语义。
3. v2 exporter 首先只接受 identity 路径：把已核验 clean 字节作为当前 `original`，人工框由 UI 动态绘制，metadata 同时声明 clean/annotated 为同一对象、同一 SHA 与 identity geometry。若 clean 与 annotated 是两个对象，先保留完整配对元数据并阻断正式导出；当前 exporter 不会假装已实现坐标变换。
4. `dbload` 读取 `session_id/capture_time/camera_id/metadata_json`；`canon` 聚合所有 aliases 的分组、pixel 与上游时间 provenance。任一 alias 缺声明、声明冲突、时间无时区或不一致都阻断正式门禁。
5. 新建 `prompt-gold/2` 导出规则：task 是唯一允许的正式 split 锚点；session/video/track/event 只能连接来源记录。旧数据允许 image-SHA fallback，但 manifest 必须分别记录 missing-task 与 task-conflict 数量，正式模式要求唯一 task 覆盖率为 100%。旧 `prompt-gold/1` 快照保持不可变。
6. exporter/auditor 同时验证来源组集合零交集和 image SHA 零交集。原始 group ID 只进入 provenance；模型输入仅保留不可逆的 `split_group_key` 哈希与状态字段。

若同一 image SHA 关联多个 task，正式导出必须失败，或先用确定性的连通分量合并相关 task；不能逐图任选一个 task 后继续切分。

## 正式数据门禁

以下条件必须全部通过：

1. **Task coverage**：所有正式样本都有非空、可追溯的 `task_id`；selection/validation/locked test 的 task 集合两两无交集。
2. **Alias consistency**：同一 `busi_id` 的 aliases 对 task、event time、对象哈希、尺寸和 `imgxy` 一致；冲突不可投票吞掉。
3. **Clean pixels**：每张训练/评测图都有可读取的 `clean_source`、真实 SHA-256 和尺寸；抽样及自动检查均确认没有烧录框或类别文字。
4. **Geometry**：每个 bbox 能通过版本化映射转换到 clean 坐标，转换后宽高为正且落在图像范围内；映射失败留在 quarantine。
5. **Temporal provenance**：使用上游 `event_time`，不以附件创建时间冒充采集时间。
6. **No invented video/track**：video/track 缺失必须显式为 null；只有上游提供稳定 ID 后才允许用作连接证据或分层指标，不能替代 task 锚点。
7. **Immutable lineage**：companion context、对象哈希、split salt、schema version 和输入文件哈希全部进入快照 manifest；同一输入重复导出字节一致。
8. **Locked isolation**：训练、阈值选择和提示词调整均不能读取 locked gold；最终配置冻结后只做一次正式评测。

## 必要测试

- companion 同步：验证 join 覆盖率、`annex_id` 唯一、同一 `busi_id` 上下文一致；缺 task 或冲突时失败。
- annex 导入：验证 `session_id/task_id`、`capture_time/event_time`、`camera_id/terminal_id` 和 metadata 完整保存；旧 payload 仍可导入。
- dbload/canon：验证旧数据库无新 metadata 时仍可读取，alias 分组冲突阻断正式门禁，错误 JSON 明确失败。
- split：两张不同 SHA、同一 task 的图必须进入同一 split；同图跨 task 必须失败或进入同一连通分量。
- strict/fallback：兼容模式对缺失 task 使用 image-SHA fallback，并将缺失与冲突互斥计数；正式模式只要 fallback 大于零就失败。
- pixels：clean/annotated 哈希与尺寸校验、identity/affine 等映射边界、越界和退化框失败路径。
- auditor：同时断言 image SHA 与 task ID 零交集，并断言 group/provenance 字段不会进入模型输入。
- determinism：同一 SQLite、companion context 和对象字节重复导出得到相同 snapshot ID 与文件哈希。

当前结论保持为：**数据 provenance 和任务级隔离未闭环，不训练，不声称准确率提升。**
