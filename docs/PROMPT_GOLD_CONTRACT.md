# PROMPT_GOLD_CONTRACT — KTD 提示词优化人工金标快照与评分契约

版本：`prompt-gold/1`。本契约定义 `tools/prompt_eval_data/` 的行为与产物格式。
任何改动必须同时更新本文件、`SCHEMA_VERSION` 与快照 `schema_version`。

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
  输出为整数 `[x1, y1, x2, y2]`。
- `canonical_id = sha256("{image_sha256}|{class_id}|{x1}|{y1}|{x2}|{y2}")` 的 hex。
  同一图片内容 + 类别 + 整数像素框的任意多条 DB 记录合并为一个 canonical 框。
- 合并（alias merge）：映射到同一 canonical_id 的所有 candidate 互为 aliases，
  全量列在 `aliases[]`（按 candidate_id 排序）； canonical 的 `detector_confidence`
  取 aliases 最大值，`prior_yolo` 取 primary alias（candidate_id 最小者）的 DB 列。

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

## 5. Split（按图隔离，零交集）

- 仅 gold 记录参与 split。
- 规则：`sha256("prompt-gold-split-v1|" + image_sha256)` 整数 `mod 5 == 0` → `locked_test`，
  否则 → `selection`（约 8:2）。同一 `image_sha256` 的全部 canonical 框同属一个 split。
- 因此两个 split 的 image SHA 集合交集恒为空；auditor 每次校验。

## 6. 快照文件布局

`data/prompt-eval/snapshots/<snapshot_id>/`，其中
`snapshot_id = "pg1-" + sha256(全部数据文件 sha256 按路径排序拼接)[:12]`（内容寻址，
同输入必得同 id）：

| 文件 | 内容 |
|---|---|
| `snapshot_manifest.json` | 单一 JSON 对象：schema_version、snapshot_id、class_map、source、counts（raw→canonical→clean/quarantine 全漏斗）、determinism 规则、files[]（除自身外每个文件的 sha256+bytes） |
| `SHA256SUMS` | 除自身与 manifest 外每个文件的 `<sha256>  <relpath>`，按路径排序 |
| `inference_records.jsonl` | 全部 canonical 框的模型侧事实：canonical_id、类别、norm/px bbox、image_sha256/object_key/尺寸、detector_confidence、prior_yolo/gpt/qwen、aliases[]、has_human_gold。**不含** label/note/reviewer |
| `gold_records.jsonl` | 干净金标：canonical_id、label、note、reviewer、reviewer_id、revision、labeled_at、source（sample/candidate/run/event）、aliases[]、三个 prior 字段、split |
| `quarantine.jsonl` | 冲突/疑似冲突记录与全部证据 |
| `authoring.jsonl` | 每条 DB Annotation 一行（人工与自动）：完整 provenance + `included_in_gold` + `exclusion_reason` |
| `selection_inputs.jsonl` | 模型输入：canonical_id、类别、bbox、image 引用/尺寸、detector_confidence。**禁止**出现 label/note/reviewer/labeled_at/prior_* |
| `selection_gold.jsonl` | 仅 canonical_id、class_id、class_name、label |
| `locked_test_inputs.jsonl` / `locked_test_gold.jsonl` | 同上，locked_test split |

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
```

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
manifest files[] 与 SHA256SUMS 逐项 sha256 一致；selection/locked_test 的
image_sha256 集合交集为空；`*_inputs.jsonl` 不含禁用键
（label/note/reviewer/reviewer_id/labeled_at/prior_*）；split gold ⊆ gold_records；
全部通过打印 `AUDIT PASS`，任一失败非零退出。

## 10. 验收映射

- 运行输出第 7 节漏斗计数（raw→canonical→clean/quarantine）。
- `auditor` 证明 split SHA 交集为 0、输入文件无 gold/note/reviewer。
- `backend/tests/test_prompt_eval_data.py` 固定夹具下重复导出字节级一致。
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

## 12. 裁剪图训练数据集（cd1，2026-09-04 增补）

`tools/prompt_eval_data/crop_dataset.py`（schema `prompt-gold-crop-dataset/1`）把任一
prompt-gold 快照构建成裁剪图训练数据集，供任意"候选框复核器"模型训练使用：

- **裁剪**：`bbox_px` 按框尺寸 `pad_ratio=0.25` 四向外扩并 clamp 到图内；长边小于
  `min_long_edge=224` 的裁剪按比例放大（LANCZOS）；输出 JPEG q=95、4:4:4。
  每取一张图都按字节 sha256 与快照 `image_sha256` 校验，不一致即计失败不静默。
- **split（图级，零交集，第 5 节盐的扩展）**：
  `mod 5 == 0` → excluded_locked（**完全不进入数据集**，不产裁剪不出行）；
  `mod 5 == 4` → val（**只出人工标签**）；其余 → train。
  val 图上的教师标签一律丢弃（`teacher_skipped_val_image` 计数），防止训练泄入验证图。
- **标签分文件不混合**：`labels_human.jsonl`（人工 true/false，train+val，
  无教师字段）与 `labels_teacher_gpt.jsonl`（GPT 教师 true/false/uncertain，仅 train，
  无人工字段）。同一 canonical 可分别出现在两个文件，但行内字段永不混写。
- **确定性**：行按 canonical_id 排序、canonical JSON、无墙钟时间戳；
  `dataset_id = "cd1-" + sha256(文件哈希按路径排序拼接 + manifest 哈希)[:12]` 内容寻址；
  同输入重复构建字节级一致并幂等跳过。crop 字节确定性依赖 Pillow 版本，
  版本号记录在 manifest.params。
- **依赖**：Pillow（隔离环境 `.venv-crop`，不进项目主依赖）；其余标准库。
- **训练基线**：`tools/crop_verifier/train.py` 读取 cd1 数据集，训练
  "教师(true/false) + 人工 train 合并（同 canonical 人工覆盖教师）、人工 val 验证"
  的二值复核器（ResNet18 + 类别 one-hot 条件头），验证指标与第 8 节 scorer 同名
  （dangerous_miss_rate / false_accept_rate + Wilson CI）。uncertain 教师行不进目标函数。
- 验收：`backend/tests/test_crop_dataset.py`（split 不相交、标签分离、locked 全排除、
  裁剪像素与放大、字节级重复构建）与 `backend/tests/test_crop_verifier_train.py` 通过。
