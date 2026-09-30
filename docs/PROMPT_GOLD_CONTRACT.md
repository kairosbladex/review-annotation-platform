# PROMPT_GOLD_CONTRACT — KTD 提示词优化人工金标快照与评分契约

核心快照版本：`prompt-gold/2`。`prompt-gold/1` 快照保持不可变，只能作为历史探索证据。
本契约定义 `tools/prompt_eval_data/` 的行为与产物格式。
快照格式改动必须同时更新本文件、`SCHEMA_VERSION` 与快照
`schema_version`；第 11 节之后的独立派生产物各自维护子 schema，不倒推修改核心快照版本。

## 0. 范围与边界

- 只读：exporter 对 SQLite 以 `mode=ro` 打开，对对象存储只下载，不写数据库、不改任何既有 data 产物。
- 金标只来自**人工** Annotation（`labeled_by` 非空）。`labeled_by IS NULL` 的自动标注
  （auto_annotation applied，备注形如 `auto annotation consensus: ...`）一律排除，只进
  `authoring.jsonl` 并记 `exclusion_reason="auto_annotation_applied"`。
- 本契约评估的是**候选框判定**（模型已产生候选框的对/错），不是整图 recall/mAP；
  一期不覆盖漏检目标。

## 1. 类别映射（固定，禁止重排）

| class_id | class_name |
|---|---|
| 0 | 三轮车 |
| 1 | 两轮车 |
| 2 | 人员 |
| 3 | 光伏板 |
| 4 | 光斑 |
| 5 | 公交车 |
| 6 | 卡车 |
| 7 | 没戴头盔 |
| 8 | 车 |
| 9 | 违建 |
| 10 | 车牌 |

0–9 与既有导出（`data/ktd-formal-review-20260903/classes.txt`、`scripts/export_completed_gold.py`
的 `CLASS_MAP`）逐字一致；车牌追加为 10。类别名是业务主键，class_id 仅快照内使用。

## 2. 规范化（canonicalization）

- 图片按 `object_key` 下载，计算 `image_sha256`（字节级 SHA-256）并用内嵌解析器读取
  像素尺寸（支持 JPEG/PNG，无第三方依赖）。
- 像素 bbox：`px = floor(norm * size + 0.5)`（半值向上），clamp 到 `[0, size]`；
  输出为整数 `[x1, y1, x2, y2]`。归一化框即使宽高为正，取整后也必须满足
  `x2 > x1 && y2 > y1`；退化框立即拒绝，不能进入 canonical、训练或评测产物。
- `canonical_id = sha256("{image_sha256}|{class_id}|{x1}|{y1}|{x2}|{y2}")` 的 hex。
  同一图片内容 + 类别 + 整数像素框的任意多条 DB 记录合并为一个 canonical 框。
- 合并（alias merge）：映射到同一 canonical_id 的所有 candidate 互为 aliases，
  全量列在 `aliases[]`（按 candidate_id 排序）； canonical 的 `detector_confidence`
  取 aliases 最大值，`prior_yolo` 取 primary alias（candidate_id 最小者）的 DB 列。
- v2 同时保留每个 alias 的 sample/asset/source provenance。`metadata_json` 只导出
  task/video/track/event/time 与 pixel 契约白名单字段，原字符串只保留 SHA-256，避免标签、
  用户信息、签名 URL 或 token 进入 inference；`source_file_path` 不在白名单内，且任何形似
  URI scheme、URL、query 或 fragment 的标量值都会丢弃，也不能参与来源分组。解析错误写入
  `metadata_error` 并阻断正式门禁。

## 3. 金标判定与隔离区（quarantine）

对每个 canonical 框，收集其 aliases 上的人工标注：

1. **标签冲突**：存在 ≥2 个不同人工 `label` → quarantine，`reason="label_conflict"`。
   不投票、不取多数。
2. **标签/备注疑似冲突**：单条人工标注的备注与标签疑似矛盾 → quarantine，
   `reason="note_label_suspect"`。确定性词表（不区分大小写）：
   - `label="true"` 且 note 含 `不是|并非|误报|误检|没有|无此|假目标`
   - `label="false"` 且 note 含 `确实有|确认为真|是真的`
   该规则是启发式哨兵，只用于把可疑行交给人工，不参与标签判定。
3. 否则人工标签唯一（可能多人重复同一结论）→ gold。`note/reviewer/revision/labeled_at`
   取 primary 人工标注（candidate_id 最小的人工 alias）；全部人工标注明细保留在
   `authoring.jsonl`。

quarantine 记录携带 `evidence.annotations[]`（全部人工与自动标注的完整字段），
不进入 `gold_records.jsonl`，也不进入任何 split。

## 4. 先验结论（绝不混入 gold）

每条 inference/gold 记录带三个互相独立的先验字段，缺失为 `null`：

- `prior_yolo`：DB `candidates` 列（`yolo_decision`、`raw_decision`、`reason_code`、置信度）。
- `prior_gpt`：外部 GPT 复核产物（`--prior-gpt`，如
  `data/recheck-runs/ktd-gpt-recheck-20260902/candidate_results.jsonl`），按
  `(object_key, class_name, bbox 六位小数)` 精确键连接；`passes[]` 保留每次通过的
  `run_id/run_stage/label/status`。
- `prior_qwen`：外部 Qwen 影子产物（`--prior-qwen`，如
  `data/agent-runs/ktd-qwen-shadow-20260903/shadow_results.jsonl`），同键连接；
  `attempts[]` 保留 `run_id/model/label/decision/confidence/status`。

先验只是参照结论，**永远不参与 gold 的 label/note**，scorer 把它们当作一个
"被评模型"的决策来源而非标签来源。

## 5. Split（按显式来源组隔离）

- 仅 gold 记录参与 split。
- 从 `metadata.source_task_id/task_id/video_id/track_id/event_id`、`session_id` 与真实
  `source_event_id` 收集显式上游 ID；`ktd-annex-<annex_id>` 仅是导入幂等键，不是业务事件，
  无论位于 sample 列还是 metadata 都禁止参与分组。共享图片或共享 ID 的记录构成确定性连通分量。
  连通图纳入 DB 查询得到的全部来源行，包括因未知类别或图片不可读取而不进入 canonical
  产物的行；这类行仍可能通过 video/track/event 等 ID 证明两个正式样本属于同一来源分量。
- session/video/track/event 只用于连接，不能单独充当防泄漏来源组。每个连通分量必须
  **恰有一个 task 锚点**；没有 task 记录 `grouping_unavailable`，出现多个 task 记录
  `source_group_conflict`，二者分别计数并阻断正式门禁。后补连接 ID 不改变已有 task key。
- 规则：`sha256("prompt-gold-split-v1|" + split_group_key)` 整数 `mod 5 == 0` →
  `locked_test`，否则 → `selection`。`split_group_key` 是所选锚点的哈希，不暴露原始 ID。
- 没有唯一 task 锚点时只回退到 `image_sha256`，并按实际原因记录
  `grouping_unavailable` 或 `source_group_conflict`。该回退只证明完全相同图片不跨 split，**不能**证明
  相邻帧、同任务或同航线不泄漏，也不满足正式准确率评测门禁。
- auditor 同时校验两个 split 的 `image_sha256` 与 `split_group_key` 集合零交集。

## 6. 快照文件布局

`data/prompt-eval/snapshots/<snapshot_id>/`。先把 manifest 中除 `snapshot_id/files` 外的
`schema_version/class_map/source/counts/determinism/readiness` 投影以规范 JSON 写入
`snapshot_identity.json`；再对包括该文件在内、但不包括派生的 `SHA256SUMS` 与
`snapshot_manifest.json` 的每个成员计算 SHA-256，按路径排序拼接
`"<path>:<sha256>\n"`。最终
`snapshot_id = "pg2-" + sha256(该拼接字节)[:12]`。因此来源、计数或门禁变化都会改变 ID，
同 ID 的既有目录只允许字节级幂等写入，任何不同内容都必须拒绝：

| 文件 | 内容 |
|---|---|
| `snapshot_manifest.json` | 单一 JSON 对象：schema_version、snapshot_id、class_map、source、counts（raw→canonical→clean/quarantine 全漏斗）、determinism 规则、files[]（除自身外每个文件的 sha256+bytes） |
| `SHA256SUMS` | 除自身与 manifest 外每个文件的 `<sha256>  <relpath>`，按路径排序 |
| `inference_records.jsonl` | 全部 canonical 框的模型侧事实与 `provenance`：source_records、来源连通分组、source_pixels、temporal。**不含** label/note/reviewer |
| `gold_records.jsonl` | 干净金标、完整 provenance、三个 prior 字段与 split |
| `quarantine.jsonl` | 冲突/疑似冲突记录、全部标注证据与 provenance |
| `authoring.jsonl` | 每条 DB Annotation 一行（人工与自动）：sample/asset、白名单 metadata、metadata 原文 SHA-256 + `included_in_gold` + `exclusion_reason` |
| `selection_inputs.jsonl` | 模型输入：canonical_id、类别、bbox、image 引用/尺寸、detector_confidence，以及不含原始上游 ID 的 split_group_key/grouping_status/source_pixel_status。**禁止**出现 label/note/reviewer/labeled_at/prior_* |
| `selection_gold.jsonl` | 仅 canonical_id、class_id、class_name、label |
| `locked_test_inputs.jsonl` / `locked_test_gold.jsonl` | 同上，locked_test split |
| `source_readiness.json` | 无标签数据门禁：来源分组覆盖、clean/overlay/unknown 像素覆盖、上游时间覆盖及 blockers；内容哈希纳入快照 |
| `snapshot_identity.json` | manifest 的规范身份投影；自身哈希参与 snapshot ID，防止仅 counts/source 改变却复用同一 ID |

序列化规则：UTF-8、`\n` 行尾、JSONL 每行一个对象、`sort_keys=True`、紧凑分隔符、
每行按 canonical_id 排序、文件以 `\n` 结尾；不写入任何墙钟时间戳。
同输入重复运行字节级一致（含 manifest；snapshot_id 内容寻址）。

## 7. 计数漏斗（运行时打印并落入 manifest.counts）

```
raw_candidates ──(排除 image_unavailable / unknown_class)──> canonical_boxes
canonical_boxes ──> unlabeled（无人工标注）
                ──> gold_clean（人工标签唯一）
                ──> quarantine（label_conflict + note_label_suspect）
raw_annotations = human（进 authoring，可进 gold）+ auto（进 authoring，排除）
formal_training_ready = 100% 唯一来源锚点 + 100% alias clean 证据 + 100% alias 上游时间 + 0 metadata error
```

每一种已声明的上游时间字段都必须在 aliases 内各自一致；不同字段种类（例如 event time 与
frame time）允许表示不同时间点，不要求彼此相等。优先字段必须覆盖全部 alias sample，任一
已声明字段内部冲突都记为 `conflicting_upstream_times` 并阻断正式门禁。

## 8. 统一 scorer

输入：`decisions.jsonl`（每行 `{canonical_id, execution_status, decision, ...}`）+
gold 文件（selection_gold / locked_test_gold / gold_records 均可）。

- **执行状态与最终三态分离**：`execution_status ∈ {ok, protocol_error, technical_failure}`。
  非 `ok`（技术失败、超时、非法 JSON）一律 `decision=null` 且动作语义为 **keep**
  （保留待复核，不当真/假目标处理）；`decision` 非法值等同 protocol_error。
  决策文件本身的坏行计入 `unparseable_lines`。
- **最终三态**：`decision ∈ {true, false, uncertain}`（null 仅来自执行失败）。
- 评判桶（对 gold true/false 分别计数）：
  - `correct_accept`（gold true ← true）
  - `dangerous_miss`（gold true ← false，危险误杀）
  - `correct_reject`（gold false ← false，正确拒绝）
  - `false_accept`（gold false ← true）
  - `uncertain_on_true` / `uncertain_on_false`
  - `protocol_failure` / `technical_failure`（不计入对错）
- 报告：overall 与 per-class 的上述计数、逐类分母（gold_true/gold_false）、
  `dangerous_miss_rate`（分母 gold_true）与 `false_accept_rate`（分母 gold_false）
  及 Wilson 95% 置信区间（z=1.96）；uncertain/protocol 单列比例。
- 报告头部固定声明：`scope="candidate_box"` —— 候选框级评判，不是整图 recall/mAP。
- `extract` 子命令可从 `inference_records.jsonl` 的 prior 字段生成 decisions
  （YOLO：confirmed/corrected→true、rejected→false、uncertain→uncertain；
  GPT：consensus 三态直取；Qwen：CONFIRMED/REJECTED/UNCERTAIN 映射，
  status≠succeeded → technical_failure）。

## 9. Auditor

`python3 -m tools.prompt_eval_data.auditor <snapshot_dir>` 校验：
manifest files[] 和 SHA256SUMS 的完整路径集、记录哈希与内容寻址 snapshot ID 一致，
禁止绝对路径、`..` 和 symlink 逃逸；未受保护成员实际字节与清单哈希一致；
selection/locked_test 的 image_sha256、split_group_key 与 canonical_id 集合交集为空，
且每行实际 split 与 `assign_split` 一致；`*_inputs.jsonl` 不含禁用键
（label/note/reviewer/reviewer_id/labeled_at/prior_*）；selection gold 与 selection inputs
严格对齐并拒绝重复。默认运行不打开任何 protected label 承载文件：
`authoring.jsonl`、`gold_records.jsonl`、`locked_test_gold.jsonl` 与 `quarantine.jsonl`；
只比较 manifest/SHA256SUMS 中已记录的哈希并以 stat 检查字节数；
因此默认模式不宣称验证 protected 文件的当前内容。仅在已授权的最终评估边界中显式传入
`--include-protected-gold` 时，才解析并校验 locked gold schema、locked input 对齐以及
selection/locked 互斥、split gold 与 full gold 的 ID/类别/标签逐项一致。
全部通过打印 `AUDIT PASS`，任一失败非零退出。

## 10. 验收映射

- 运行输出第 7 节漏斗计数（raw→canonical→clean/quarantine）。
- `auditor` 证明 split group 与图片 SHA 交集均为 0、输入文件无 gold/note/reviewer，
  默认不打开 protected label 文件，并验证 `source_readiness.json` 与 manifest 一致。
- `backend/tests/test_prompt_eval_data.py` 固定夹具下重复导出字节级一致；
  同时覆盖 excluded-row 来源桥、每种时间字段的 alias 冲突、URI scheme 脱敏，以及仅漏斗
  counts 改变时 snapshot ID 也必须改变；`backend/tests/test_prompt_eval_auditor.py` 覆盖
  no-open、空清单、路径逃逸、重复与标签漂移反例。
- `PYTHONPATH=backend python3 -m pytest backend/tests -q` 通过。

## 11. 部署服务适配器与失败分析（2026-09-03 增补，不改既有行为）

在既有契约之外新增两个只增不改的模块，用于把**已部署复检服务**纳入同一评分体系：

- `tools/prompt_eval_data/service_attempts.py`（schema `prompt-gold-service-attempts/1`）：
  把预填充/服务运行的 `attempts.jsonl`（如 `data/ktd-8188-kimi-prefill-20260903/`）
  转成第 8 节 scorer 的 decisions。`candidate_id → canonical_id` 经快照
  `gold_records.jsonl` 的 aliases 连接。两种决策源必须分开评：
  `--source decision`（服务最终上线判定，部署版把 uncertain 按设计转 confirmed）与
  `--source yolo_decision`（验证器原始三态）。执行状态规则：attempt 失败 →
  technical_failure；result 缺位/决策值不可映射 → protocol_error；`errorCode` 非空 →
  technical_failure。同一 candidate 多次尝试取 (started_at, attempt_no) 最新；
  输出行按 (canonical_id, candidate_id) 排序，确定性可复现。
- `tools/prompt_eval_data/service_analysis.py`（schema `prompt-gold-service-analysis/1`）：
  对 attempts 做字段分布、reasoning 模板归一化（数字/类名/标签脱敏后计数）、
  时延分位的失败模式聚合，可嵌入 scorer 报告的 per-class 表。
- 验收：`backend/tests/test_service_attempts.py` 通过；基线产物与复现命令见
  `docs/KIMI_8188_BASELINE_HANDOFF.md`。

## 12. 裁剪图训练数据集（cd2，2026-09-04 增补）

`tools/prompt_eval_data/crop_dataset.py`（schema `prompt-gold-crop-dataset/2`）从
prompt-gold 快照的 selection 分区构建裁剪图数据集。人工标签只从
`selection_gold.jsonl` 读取；构建器不打开 `gold_records.jsonl` 或
`locked_test_gold.jsonl`。GPT 教师候选仍只来自
`inference_records.jsonl.prior_gpt`。

> 该 cd2 路径是历史探索路径：其 source pixel 为 `unverified_overlay_risk`，split 也只有
> image-SHA 隔离，因此不得继续用于正式训练或准确率结论。只有后续构建器显式校验
> prompt-gold/2 的 `source_readiness.formal_training_ready=true` 后，才能恢复正式训练。

- **双视图裁剪**：tight view 使用原始 `bbox_px`；context view 默认按框尺寸
  `pad_ratio=1.0` 每边外扩并 clamp 到图内。两视图分别写入 `.tight.jpg` 与
  `.context.jpg`，长边小于 `min_long_edge=224` 时各自按比例放大（LANCZOS）；
  输出 JPEG q=95、4:4:4。兼容字段 `crop_path == context_crop_path`、
  `crop_box_px == context_box_px`。每张源图按字节 sha256 与快照 `image_sha256`
  校验，不一致则计失败。
- **可量化元数据**：标签行保留 object key、源图宽高、bbox 宽高/面积占比/长宽比、
  `bbox_touches_edge`、`context_clipped`、两视图渲染尺寸与放大倍数，供按原框
  尺寸分桶。
- **像素 provenance**：标签行写入
  `source_pixel_status="unverified_overlay_risk"`；manifest 的 `source_pixels` 明确
  `kind='original'` 等快照元数据不能证明源像素无框、无文字 overlay。
- **split（图级，零交集，第 5 节盐的扩展）**：
  `mod 5 == 0` → excluded_locked（不产裁剪不出行）；`mod 5 == 4` → val
  （只出人工标签）；其余 → train。locked 排除只由 inference 的图片哈希确定，
  过程不读取 locked gold。val 图上的教师标签一律丢弃
  （`teacher_skipped_val_image` 计数），防止训练泄入验证图。
- **标签分文件不混合**：`labels_human.jsonl`（人工 true/false，train+val）与
  `labels_teacher_gpt.jsonl`（GPT 教师 true/false/uncertain，仅 train）。同一 canonical
  可分别出现在两个文件，但行内字段永不混写。
- **确定性**：行按 canonical_id 排序、canonical JSON、无墙钟时间戳；
  `dataset_id = "cd2-" + sha256(文件哈希按路径排序拼接 + manifest 哈希)[:12]`
  内容寻址。`crops_written` 计样本数，`crop_views_written` 计实际视图文件数。
- **训练基线**：`tools/crop_verifier/train.py` 兼容 cd1/cd2。默认
  `--train-label-source human-only`；GPT 伪标只能通过
  `--train-label-source teacher-assisted` 显式加入，同 canonical 仍由人工标签覆盖。
  `--crop-view {tight,context}` 为单次实验选择一种 cd2 视图，cd1 只支持 context。
  metrics 固化 label source、crop view、dataset id/schema/manifest SHA、总体/逐类/原框尺寸分桶、
  危险误杀优先的 best checkpoint 与可选 early stopping。`--reject-threshold` 只评估
  低分 REJECT、其余 KEEP 的保守策略，并报告 auto-reject rate、safe-reject yield
  与负例 correct-reject rate。selection/val 只用于选模型与阈值，
  不得称为独立准确率证明。
- **验收**：`backend/tests/test_crop_dataset.py`（poisoned full/locked gold 不读取、
  split 不相交、标签分离、双视图/元数据、cd2 内容寻址与字节级重复构建）
  与 `backend/tests/test_crop_verifier_train.py` 通过。

## 13. paired scorer（v1.0.0 vs v1.1.0，2026-09-04 增补，不改既有行为）

`tools/prompt_eval_data/paired_scorer.py`（schema `prompt-gold-paired-score/1`、误例包
`prompt-gold-paired-error-packet/1`）对 Codex paired selection run 的两份 final decisions
做只读、确定性的成对评分。复用第 8 节的 `judge`、`normalize_decision`、三态定义与
Wilson CI，只新增成对对齐与门禁。

- **输入**：`final_decisions_v1.0.0.jsonl` 与 `final_decisions_v1.1.0.jsonl`（Codex
  paired run 产物，行含 `canonical_id/execution_status/decision/class_name/prompt_version/
  run_id/safety_action`）；两份 run `report.json`（终态证据）；固定快照
  `pg1-bc8c818184d8` 的 `selection_gold.jsonl`（438 框）。
- **显式失败（退出码 2，绝不静默丢行或按顺序对齐）**：任一行 JSON 不可解析；
  canonical_id 重复/缺失/多余（两侧集合必须各自与 selection_gold 完全相等）；
  `prompt_version` 不以对应 `.v1.0.0`/`.v1.1.0` 结尾；decisions 的 `run_id` 与 report
  不一致或一份文件内多个 run_id；manifest SHA 不等于
  `4c15c3892c69eea10f28c3cd1fbc381ed6b34c7690922a34af189d5c2cd752b2`（CLI 默认钉死，
  `--expect-manifest-sha` 仅供夹具测试覆盖）、selection_gold/
  selection_inputs SHA 与 manifest files[] 不一致；run report `status` 非终态
  （终态集 `{succeeded, partial_failed}`）或 report 中 snapshot SHA 与 manifest 不一致。
- **状态映射**：Codex 的 `execution_status="succeeded"` 归一到第 8 节 `ok`；
  `technical_failure/protocol_error` 依旧 `decision=null`、语义 keep，不算 true/false；
  合法 `uncertain` 单独统计。gold 侧 `class_name` 为业务主键，`class_id` 只按快照原值
  校验（禁止重排）；decisions 行 `class_name` 与 gold 不一致即失败。
- **统计**：overall 与 per-class 的八桶计数（分母 gold_true/gold_false）、
  dangerous_miss_rate/false_accept_rate/uncertain_rate + Wilson 95% CI，两侧各一份；
  paired 迁移矩阵（v1.0 桶 → v1.1 桶）、改善/退化 canonical_id 列表（桶严重度
  correct < uncertain < failure < false_accept < dangerous_miss）、新增危险误杀、
  恢复的正确拒绝（false_accept→correct_reject）、UNCERTAIN 增减列表。
- **口径**：调用数（两侧行数之和，成对 run 为 876）与独立框数（438）分列；
  一切准确率分母只用单侧 438 的 gold_true/gold_false。scope 固定 `candidate_box`。
- **门禁**：v1.1 须 dangerous_miss=0 且不劣于 v1.0（dangerous_miss、false_accept、
  failure 不增，correct_accept、correct_reject 不减）→ `passed`；全部指标持平 →
  `verdict="tie_keep_v1.0"`，必须建议保留 v1.0，不得写成提升；任一严格更优 →
  `improved`；任一检查失败 → `regressed`。
- **误例包**（给 Grok，去标识化）：仅含任一侧非 correct 的 canonical_id、class_name、
  两版 final/execution_status/safety_action/axes/reason_codes、两侧错误桶、图像引用
  （image_sha256/object_key）。**禁止**写入 gold label/note/reviewer；axes/reason_codes
  仅在传入对应 run 目录（attempts.jsonl + prompt_manifest.json，同版本后缀、按
  (attempt_no, attempt_id) 取最新）时填充，否则为 null。
- 验收：`backend/tests/test_paired_scorer.py` 通过；同输入重复运行字节级一致。

## 14. 分层人工 gold 扩展队列（2026-09-04）

`tools/prompt_eval_data/build_gold_expansion_queue.py`（schema
`gold-expansion-queue/2`）只读取 `selection_gold.jsonl`、`selection_inputs.jsonl`、
`locked_test_inputs.jsonl`、`inference_records.jsonl` 与 `quarantine.jsonl`；禁止打开
`gold_records.jsonl` 或 `locked_test_gold.jsonl`。

- 原框像素长边按 `<16 / 16-47 / 48-95 / >=96` 分桶，与训练指标口径一致。
- 构建前必须验证快照 schema、目录 snapshot ID、manifest 中全部依赖文件的 SHA-256/bytes，
  v2 还要求 `source_readiness.json` 与 manifest 完全一致。
- 只允许经过结构验证的 `source_pixels.status=clean`、唯一且非 event 的来源锚点，以及覆盖
  全部 alias sample 的带时区 ISO-8601 上游时间进入人工队列；其余 selection 候选写入
  `blocked_sources.jsonl`，`label` 保持 null。
- 每项在标注前计算 `planned_split`；planned locked 在生成任何逐条 queue/blocker 明细前排除，
  只输出聚合计数。v2 快照按来源组 key，
  v1 快照因无 provenance 只会进入 blocked，不得恢复旧 image-only 标注流程。
- summary 输出 selection 人工 gold 的完整 class × human label × pixel-size 覆盖矩阵。
  首批优先未覆盖的 class × pixel-size，每个来源组最多一框；待标候选真实 label 未知，
  禁止用模型预测伪装成正负配额。完成人工 blind tranche 后才能按真实标签缺口生成下一批。
- CLI 以临时目录完整写入后原子发布，并生成 `artifact_manifest.json`，记录来源快照 manifest
  与依赖成员哈希，以及 queue/blocker/summary 的哈希和字节数；既有输出目录一律拒绝覆盖。
- 2026-09-04 当前产物为 `gold-expansion-20260904-v3-source-gated`：正式队列 0，
  2,273 框 / 421 图因 clean、来源组、上游时间均未闭环而进入公开 blocker inventory；
  另有 planned locked 500 框 / 104 图在明细生成前排除。旧 v2 队列已废止。
