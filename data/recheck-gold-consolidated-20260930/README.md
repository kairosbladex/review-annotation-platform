# 复检统一金标：第0步待用户确认

快照时间：2026-09-30T15:38:35.681875+08:00。只读取本地文件，图片不复制、不上传。

来源行 7402（含派生副本，不等于独立样本）；合格人工来源行 1673；去重后金标 677；剔除 5729；冲突目标 4。

暂无类别可正式评测。

档位全部按独立组计数：锁定测试真、假各≥100组才可评测（因此各≥10组也满足）；开发真、假各≥30组才仅可开发。框数只作附加统计。

|类别|真/假/存疑框数|真/假/存疑独立组数|总组数|锁定测试真/假组|档位|
|---|---|---|---|---|---|
|三轮车|2/3/0|1/3/0|4|0/0|未充分验证|
|两轮车|27/3/0|15/3/0|17|0/0|未充分验证|
|人员|59/40/0|41/22/0|59|0/0|未充分验证|
|光伏板|23/1/0|6/1/0|7|0/0|未充分验证|
|光斑|15/12/0|5/8/0|10|0/0|未充分验证|
|公交车|7/6/0|7/5/0|12|0/0|未充分验证|
|卡车|3/4/0|3/4/0|7|0/0|未充分验证|
|没戴头盔|3/2/0|3/2/0|5|0/0|未充分验证|
|车|310/21/0|63/9/0|66|0/0|未充分验证|
|车牌|4/0/0|4/0/0|4|0/0|未充分验证|
|违建|71/61/0|26/48/0|70|0/0|未充分验证|

每类主要缺口：尚无未被训练/调提示词使用的锁定测试任务组；不足30组的类别还缺开发真/假组。

## 来源纳入和排除

每行来源引用包含文件、物理行号及本次文件哈希。CSV读取一次后立即固化，后续标注变化不改变本快照。

|来源|快照行数|合格人工行|剔除行|原因|
|---|---|---|---|---|
|data/gold-export-20260903/annotations.jsonl|104|0|104|缺少人工身份/时间/原始标签或必要溯源，逐行原因见剔除清单；空文件保持空|
|data/gold-export-live-20260903/annotations.jsonl|854|848|6|通过人工来源检查；仍须去重/冲突隔离|
|data/ktd-canonical-crop-20260904/human_gold_selection.jsonl|438|0|438|缺少人工身份/时间/原始标签或必要溯源，逐行原因见剔除清单；空文件保持空|
|data/ktd-canonical-crop-20260904/selection_gold.jsonl|438|0|438|缺少人工身份/时间/原始标签或必要溯源，逐行原因见剔除清单；空文件保持空|
|data/ktd-clean-source-full-evaluation-20260924/gold_labels.jsonl|142|0|142|缺少人工身份/时间/原始标签或必要溯源，逐行原因见剔除清单；空文件保持空|
|data/ktd-clean-source-qwen-20260925-v1/provenance/gold_labels.jsonl|142|0|142|缺少人工身份/时间/原始标签或必要溯源，逐行原因见剔除清单；空文件保持空|
|data/ktd-clean-source-qwen-20260925-v1.superseded-contract/provenance/gold_labels.jsonl|142|0|142|缺少人工身份/时间/原始标签或必要溯源，逐行原因见剔除清单；空文件保持空|
|data/ktd-clean-source-qwen-20260925-v1.superseded-copied-views-104633/provenance/gold_labels.jsonl|142|0|142|缺少人工身份/时间/原始标签或必要溯源，逐行原因见剔除清单；空文件保持空|
|data/ktd-clean-source-qwen-20260925-v1.superseded-maxlen2048/provenance/gold_labels.jsonl|142|0|142|缺少人工身份/时间/原始标签或必要溯源，逐行原因见剔除清单；空文件保持空|
|data/ktd-clean-source-qwen-20260925-v1.superseded-pre-audit-102341/provenance/gold_labels.jsonl|142|0|142|缺少人工身份/时间/原始标签或必要溯源，逐行原因见剔除清单；空文件保持空|
|data/ktd-clean-source-qwen-20260930-v2/provenance/gold_labels.jsonl|198|0|198|缺少人工身份/时间/原始标签或必要溯源，逐行原因见剔除清单；空文件保持空|
|data/ktd-codex-human-gold-train-20260924/codex_human_comparison.jsonl|102|0|102|缺少人工身份/时间/原始标签或必要溯源，逐行原因见剔除清单；空文件保持空|
|data/ktd-codex-human-gold-train-20260924/human_gold_annotations.jsonl|102|102|0|通过人工来源检查；仍须去重/冲突隔离|
|data/ktd-codex-human-gold-validation-20260924/codex_human_comparison.jsonl|40|0|40|缺少人工身份/时间/原始标签或必要溯源，逐行原因见剔除清单；空文件保持空|
|data/ktd-codex-human-gold-validation-20260924/human_gold_annotations.jsonl|40|40|0|通过人工来源检查；仍须去重/冲突隔离|
|data/ktd-codex-human-gold-validation-20260924/source_human_review_export.jsonl|40|0|40|缺少人工身份/时间/原始标签或必要溯源，逐行原因见剔除清单；空文件保持空|
|data/ktd-codex-human-gold-validation-trial-20260924/codex_human_comparison.jsonl|6|0|6|缺少人工身份/时间/原始标签或必要溯源，逐行原因见剔除清单；空文件保持空|
|data/ktd-codex-human-gold-validation-trial-20260924/human_gold_annotations.jsonl|6|6|0|通过人工来源检查；仍须去重/冲突隔离|
|data/ktd-codex-human-gold-validation-trial-20260924/source_human_review_export.jsonl|6|0|6|缺少人工身份/时间/原始标签或必要溯源，逐行原因见剔除清单；空文件保持空|
|data/ktd-codex-review-batch-02-20260924-audited/codex_human_comparison.jsonl|33|0|33|缺少人工身份/时间/原始标签或必要溯源，逐行原因见剔除清单；空文件保持空|
|data/ktd-codex-review-batch-02-20260924-audited/human_gold_annotations.jsonl|33|33|0|通过人工来源检查；仍须去重/冲突隔离|
|data/ktd-codex-review-batch-02-20260924-audited/source_human_review_export.jsonl|33|0|33|缺少人工身份/时间/原始标签或必要溯源，逐行原因见剔除清单；空文件保持空|
|data/ktd-codex-review-batch-03-person-20260924-audited/codex_human_comparison.jsonl|36|0|36|缺少人工身份/时间/原始标签或必要溯源，逐行原因见剔除清单；空文件保持空|
|data/ktd-codex-review-batch-03-person-20260924-audited/human_gold_annotations.jsonl|36|36|0|通过人工来源检查；仍须去重/冲突隔离|
|data/ktd-codex-review-batch-03-person-20260924-audited/source_human_review_export.jsonl|36|0|36|缺少人工身份/时间/原始标签或必要溯源，逐行原因见剔除清单；空文件保持空|
|data/ktd-codex-review-batch-04-validation-20260924-audited/codex_human_comparison.jsonl|6|0|6|缺少人工身份/时间/原始标签或必要溯源，逐行原因见剔除清单；空文件保持空|
|data/ktd-codex-review-batch-04-validation-20260924-audited/human_gold_annotations.jsonl|6|6|0|通过人工来源检查；仍须去重/冲突隔离|
|data/ktd-codex-review-batch-04-validation-20260924-audited/source_human_review_export.jsonl|6|0|6|缺少人工身份/时间/原始标签或必要溯源，逐行原因见剔除清单；空文件保持空|
|data/ktd-codex-review-batch-05-blind-validation-20260924-audited/codex_human_comparison.jsonl|34|0|34|缺少人工身份/时间/原始标签或必要溯源，逐行原因见剔除清单；空文件保持空|
|data/ktd-codex-review-batch-05-blind-validation-20260924-audited/human_gold_annotations.jsonl|34|34|0|通过人工来源检查；仍须去重/冲突隔离|
|data/ktd-codex-review-batch-05-blind-validation-20260924-audited/source_human_review_export.jsonl|34|0|34|缺少人工身份/时间/原始标签或必要溯源，逐行原因见剔除清单；空文件保持空|
|data/ktd-codex-review-pilot-20260923-audited/codex_human_comparison.jsonl|33|0|33|缺少人工身份/时间/原始标签或必要溯源，逐行原因见剔除清单；空文件保持空|
|data/ktd-codex-review-pilot-20260923-audited/human_gold_annotations.jsonl|33|33|0|通过人工来源检查；仍须去重/冲突隔离|
|data/ktd-codex-review-pilot-20260923-audited/source_human_review_export.jsonl|33|0|33|缺少人工身份/时间/原始标签或必要溯源，逐行原因见剔除清单；空文件保持空|
|data/ktd-formal-review-20260903/human_confirmation_queue.jsonl|751|0|751|缺少人工身份/时间/原始标签或必要溯源，逐行原因见剔除清单；空文件保持空|
|data/ktd-test-real-monitoring-full-20260924/gold_overlap_map.jsonl|142|0|142|缺少人工身份/时间/原始标签或必要溯源，逐行原因见剔除清单；空文件保持空|
|data/ktd-test-real-monitoring-full-20260924/input/gold_overlap_map.jsonl|142|0|142|缺少人工身份/时间/原始标签或必要溯源，逐行原因见剔除清单；空文件保持空|
|data/recheck-released-fp-20260930/dataset-source/gold_labels.jsonl|198|0|198|缺少人工身份/时间/原始标签或必要溯源，逐行原因见剔除清单；空文件保持空|
|data/recheck-released-fp-20260930/dataset-source/other_class_gold_labels.jsonl|24|0|24|缺少人工身份/时间/原始标签或必要溯源，逐行原因见剔除清单；空文件保持空|
|data/recheck-released-fp-20260930/dataset-source/reserve_gold_labels.jsonl|307|0|307|缺少人工身份/时间/原始标签或必要溯源，逐行原因见剔除清单；空文件保持空|
|data/recheck-released-fp-20260930/labeling-package/annotations.csv|480|0|480|缺少人工身份/时间/原始标签或必要溯源，逐行原因见剔除清单；空文件保持空|
|data/review-retrain-20260907/human-dataset/labels.jsonl|557|0|557|缺少人工身份/时间/原始标签或必要溯源，逐行原因见剔除清单；空文件保持空|
|data/vehicle-surface-gate-eval-20260929/frozen/rows.jsonl|215|0|215|缺少人工身份/时间/原始标签或必要溯源，逐行原因见剔除清单；空文件保持空|
|data/vehicle-surface-gate-eval-20260929/holdout/human_labels.jsonl|0|0|0|空文件；无人工标签可纳入|
|data/vehicle-surface-gate-eval-20260929/prerelease-evidence/annotation-package/annotations.csv|60|0|60|缺少人工身份/时间/原始标签或必要溯源，逐行原因见剔除清单；空文件保持空|
|data/vehicle-surface-gate-eval-20260929/prerelease-evidence/hires-review-package/annotations.csv|91|0|91|缺少人工身份/时间/原始标签或必要溯源，逐行原因见剔除清单；空文件保持空|
|data/vehicle-surface-gate-eval-20260929/prerelease-evidence/new-events-hires-package/annotations.csv|17|0|17|缺少人工身份/时间/原始标签或必要溯源，逐行原因见剔除清单；空文件保持空|
|data/vehicle-surface-gate-eval-20260929/prerelease-evidence/supplement-annotation-package/annotations.csv|31|0|31|缺少人工身份/时间/原始标签或必要溯源，逐行原因见剔除清单；空文件保持空|
|data/vehicle-surface-gate-eval-20260929/random-traffic-hires-package/annotations.csv|200|0|200|缺少人工身份/时间/原始标签或必要溯源，逐行原因见剔除清单；空文件保持空|
|/Users/wangxiaolin/PycharmProjects/review-annotation-platform-prompt-gold/data/prompt-eval/snapshots/pg1-bc8c818184d8/gold_records.jsonl|535|535|0|通过人工来源检查；仍须去重/冲突隔离|

## 子集核对

- `data/gold-export-20260903/annotations.jsonl` → `data/gold-export-live-20260903/annotations.jsonl`：标签、同类框及候选ID子集=True；104/854，不在大批内=0。
- `data/ktd-codex-human-gold-validation-trial-20260924/human_gold_annotations.jsonl` → `data/ktd-codex-human-gold-validation-20260924/human_gold_annotations.jsonl`：标签、同类框及候选ID子集=True；6/40，不在大批内=0。

## 口径与冻结

人工身份为空、auto annotation备注、明确非人工来源都剔除；未记录标注时间、图片哈希、框或分组的行也单列剔除，绝不补造元数据。reviewer_id是旧平台人工身份字段，保留原字段并归一到labeled_by。

按同图SHA-256、同类别、框IoU≥0.5建立连通组件去重。组件内任何标签冲突（含uncertain）全部隔离，保留所有人工原行，不自动选择。

图片、任务、连拍和事件别名连通后固定分组；没有任务ID时退到连拍/事件。跨任务相同图片合为一个保守组。split_registry.jsonl保持追加，已冻结分组不得改动。所有本次历史样本只进入development；已看证据与无法证明未看分别记录。

本步仅建立历史开发集。未来新组进入测试的比例/抽样规则须在用户确认第0步后、接入新金标前固定；当前脚本对历史来源没有随机分配测试的路径。

## 文件

gold.jsonl：统一金标及每次人工标注溯源；excluded.jsonl：逐行剔除原因；conflicts.jsonl：待人工裁决；deduplication.jsonl：去重关系；splits.jsonl/split_registry.jsonl：冻结清单；sources.jsonl/source_rows.jsonl：文件快照哈希与原行（original_row按ref查找原行）；exposure_sources.jsonl：仅标记训练暴露的转换文件；SHA256SUMS：产物哈希。

验证：13项定向测试通过；产物哈希、每行来源对账、冲突隔离、同输入重建切分保持、凭据字段检查通过。证据见verification.json。

## 检查范围与待确认

- 12个回归样本的唯一完整清单未在当前入口定位；所有历史来源保守限制开发集，不会漏入测试集。
- 封存测试答案 sealed-copy 未打开；本次不使用旧锁定测试集做评分。
- 有图片哈希但本机无原图的行保留原始声明，不能声称已现场校验图片；source_images为空。
- 任务/连拍/事件别名和相同图片连通分组，避免跨来源泄漏；连通组计数比任务数更保守。
- 外发授权、模型别名实测及额度接口尚未验证；本步无需调用。
- 排除且未打开：data/vehicle-surface-gate-eval-20260929/holdout/sealed-copy/human_labels.server.jsonl；封存答案，不打开；不纳入本期金标

默认素材建议：画面匹配的高清干净原图；无干净图时标明红框截图局限。依据源目录vehicle-surface-gate-eval-20260929/resolution-ab/README.md和resolution-ab-random/README.md。难例配对改善有依据，随机批收益未达到统计显著，不推广为GPT准确率结论。

线上后续记录使用标注平台DATABASE_URL主库新表；本步不连库、不做迁移、不启服务。模型别名和/v1/usage未验证。

资源：无容器、GPU、后台进程、端口监听；不复制图片与模型。只保留脚本、测试、统一清单和必要证据。

统计报告待用户确认。确认前不进入第1—4步，不调用中转站。
