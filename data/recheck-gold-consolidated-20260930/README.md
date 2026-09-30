# 复检统一金标：第0步修正重跑，待用户确认

快照：2026-09-30T16:07:31.577709+08:00；时区Asia/Shanghai。来源7402行，合格人工2517行，去重金标1453框，剔除4885行，冲突5框。

可评测：locked_test_gpt真、假各≥100框且各≥10独立组；仅可开发：全部真、假各≥30框；uncertain不计真/假。档位仅表示样本数量条件，不表示已通过模型评测。

固定种子20260930。按排序后的组列表用Python random.Random(seed).shuffle，未被GPT看过且来源可判定的组前floor(N/2)进入locked_test_gpt，其余开发。GPT曝光或历史不明的整组进入开发。仅Qwen曝光不阻止GPT锁定测试；带Qwen曝光的测试组valid_for=["gpt"]，禁止评测Qwen。

全部测试切分仅为待确认提案。split_registry.jsonl只记录development，不写入、不冻结locked_test_gpt；提案见proposed_groups.jsonl与splits.jsonl。确认前不进入第1—4步，不调用中转站。

|类别|真/假/uncertain框|真/假/uncertain组|测试真/假框|测试真/假组|档位|
|---|---|---|---|---|---|
|三轮车|3/6/0|2/6/0|1/2|1/2|未充分验证|
|两轮车|29/8/0|17/7/0|2/4|2/3|未充分验证|
|人员|59/40/0|41/22/0|0/0|0/0|仅可开发|
|光伏板|23/1/0|6/1/0|0/0|0/0|未充分验证|
|光斑|15/12/0|5/8/0|0/0|0/0|未充分验证|
|公交车|7/7/0|7/6/0|0/0|0/0|未充分验证|
|卡车|3/12/1|3/12/1|0/4|0/4|未充分验证|
|没戴头盔|3/2/0|3/2/0|0/0|0/0|未充分验证|
|车|958/95/26|241/45/14|251/29|74/15|仅可开发|
|车牌|4/0/0|4/0/0|0/0|0/0|未充分验证|
|违建|78/61/0|31/48/0|4/0|3/0|仅可开发|

## population分开清点（无准确率评测）

### historical_review

|类别|真/假/uncertain框|真/假/uncertain组|测试真/假框|测试真/假组|档位|
|---|---|---|---|---|---|
|三轮车|2/3/0|1/3/0|0/0|0/0|未充分验证|
|两轮车|27/3/0|15/3/0|0/0|0/0|未充分验证|
|人员|59/40/0|41/22/0|0/0|0/0|仅可开发|
|光伏板|23/1/0|6/1/0|0/0|0/0|未充分验证|
|光斑|15/12/0|5/8/0|0/0|0/0|未充分验证|
|公交车|7/6/0|7/5/0|0/0|0/0|未充分验证|
|卡车|3/4/0|3/4/0|0/0|0/0|未充分验证|
|没戴头盔|3/2/0|3/2/0|0/0|0/0|未充分验证|
|车|310/21/0|63/9/0|0/0|0/0|未充分验证|
|车牌|4/0/0|4/0/0|0/0|0/0|未充分验证|
|违建|71/61/0|26/48/0|0/0|0/0|仅可开发|

### random_traffic

|类别|真/假/uncertain框|真/假/uncertain组|测试真/假框|测试真/假组|档位|
|---|---|---|---|---|---|
|三轮车|0/3/0|0/3/0|0/2|0/2|未充分验证|
|公交车|0/1/0|0/1/0|0/0|0/0|未充分验证|
|卡车|0/2/0|0/2/0|0/1|0/1|未充分验证|
|车|172/11/0|113/11/0|83/5|57/5|未充分验证|

### released_by_recheck

|类别|真/假/uncertain框|真/假/uncertain组|测试真/假框|测试真/假组|档位|
|---|---|---|---|---|---|
|三轮车|1/0/0|1/0/0|1/0|1/0|未充分验证|
|两轮车|2/5/0|2/4/0|2/4|2/3|未充分验证|
|卡车|0/6/0|0/6/0|0/3|0/3|未充分验证|
|车|399/16/2|52/7/2|153/1|15/1|未充分验证|
|违建|7/0/0|5/0/0|4/0|3/0|未充分验证|

### rooftop_hard_case

|类别|真/假/uncertain框|真/假/uncertain组|测试真/假框|测试真/假组|档位|
|---|---|---|---|---|---|
|卡车|0/0/1|0/0/1|0/0|0/0|未充分验证|
|车|107/51/24|49/26/12|33/24|17/12|仅可开发|

## locked_test_gpt来源清单

同一金标若有多来源，分别贡献计数，不能把来源表直接相加。

|来源|组|框|population框分布|
|---|---|---|---|
|data/recheck-released-fp-20260930/labeling-package/annotations.csv|23|170|{"released_by_recheck":170}|
|data/vehicle-surface-gate-eval-20260929/prerelease-evidence/annotation-package/annotations.csv|15|16|{"rooftop_hard_case":16}|
|data/vehicle-surface-gate-eval-20260929/prerelease-evidence/hires-review-package/annotations.csv|21|24|{"rooftop_hard_case":24}|
|data/vehicle-surface-gate-eval-20260929/prerelease-evidence/new-events-hires-package/annotations.csv|3|10|{"rooftop_hard_case":10}|
|data/vehicle-surface-gate-eval-20260929/prerelease-evidence/supplement-annotation-package/annotations.csv|12|21|{"rooftop_hard_case":21}|
|data/vehicle-surface-gate-eval-20260929/random-traffic-hires-package/annotations.csv|64|91|{"random_traffic":91}|

## 补齐元数据

只对指定人工网页CSV的非空label补齐；labeled_by=wangxiaolin，labeled_at=该CSV的文件mtime，metadata_inferred=file_level。原行快照不篡改。

|来源|补齐行数|其中通过溯源检查|
|---|---|---|
|data/recheck-released-fp-20260930/labeling-package/annotations.csv|480|464|
|data/vehicle-surface-gate-eval-20260929/prerelease-evidence/annotation-package/annotations.csv|60|60|
|data/vehicle-surface-gate-eval-20260929/prerelease-evidence/hires-review-package/annotations.csv|91|81|
|data/vehicle-surface-gate-eval-20260929/prerelease-evidence/new-events-hires-package/annotations.csv|17|17|
|data/vehicle-surface-gate-eval-20260929/prerelease-evidence/supplement-annotation-package/annotations.csv|31|31|
|data/vehicle-surface-gate-eval-20260929/random-traffic-hires-package/annotations.csv|200|191|

## GPT曝光无法判断，保守进开发集

以下包含因同组其他来源不明而连带进入开发的来源。

|来源|组|框|
|---|---|---|
当前已纳入金标未发现额外的来源不明组；但12个回归样本独立清单未定位，跨来源重叠核验仍未完成，不能把测试候选视为完整曝光审计通过。

## 冲突隔离

|类别|冲突框|标签组合|
|---|---|---|
|两轮车|1|{"false/true":1}|
|光斑|1|{"false/true":1}|
|车|2|{"false/true":1,"false/uncertain":1}|
|违建|1|{"false/true":1}|

冲突逐框溯源见conflicts.jsonl，分类裁决索引见conflicts_by_class.jsonl。

## 来源对账

|来源|行数|合格人工行|剔除行|
|---|---|---|---|
|data/gold-export-20260903/annotations.jsonl|104|0|104|
|data/gold-export-live-20260903/annotations.jsonl|854|848|6|
|data/ktd-canonical-crop-20260904/human_gold_selection.jsonl|438|0|438|
|data/ktd-canonical-crop-20260904/selection_gold.jsonl|438|0|438|
|data/ktd-clean-source-full-evaluation-20260924/gold_labels.jsonl|142|0|142|
|data/ktd-clean-source-qwen-20260925-v1/provenance/gold_labels.jsonl|142|0|142|
|data/ktd-clean-source-qwen-20260925-v1.superseded-contract/provenance/gold_labels.jsonl|142|0|142|
|data/ktd-clean-source-qwen-20260925-v1.superseded-copied-views-104633/provenance/gold_labels.jsonl|142|0|142|
|data/ktd-clean-source-qwen-20260925-v1.superseded-maxlen2048/provenance/gold_labels.jsonl|142|0|142|
|data/ktd-clean-source-qwen-20260925-v1.superseded-pre-audit-102341/provenance/gold_labels.jsonl|142|0|142|
|data/ktd-clean-source-qwen-20260930-v2/provenance/gold_labels.jsonl|198|0|198|
|data/ktd-codex-human-gold-train-20260924/codex_human_comparison.jsonl|102|0|102|
|data/ktd-codex-human-gold-train-20260924/human_gold_annotations.jsonl|102|102|0|
|data/ktd-codex-human-gold-validation-20260924/codex_human_comparison.jsonl|40|0|40|
|data/ktd-codex-human-gold-validation-20260924/human_gold_annotations.jsonl|40|40|0|
|data/ktd-codex-human-gold-validation-20260924/source_human_review_export.jsonl|40|0|40|
|data/ktd-codex-human-gold-validation-trial-20260924/codex_human_comparison.jsonl|6|0|6|
|data/ktd-codex-human-gold-validation-trial-20260924/human_gold_annotations.jsonl|6|6|0|
|data/ktd-codex-human-gold-validation-trial-20260924/source_human_review_export.jsonl|6|0|6|
|data/ktd-codex-review-batch-02-20260924-audited/codex_human_comparison.jsonl|33|0|33|
|data/ktd-codex-review-batch-02-20260924-audited/human_gold_annotations.jsonl|33|33|0|
|data/ktd-codex-review-batch-02-20260924-audited/source_human_review_export.jsonl|33|0|33|
|data/ktd-codex-review-batch-03-person-20260924-audited/codex_human_comparison.jsonl|36|0|36|
|data/ktd-codex-review-batch-03-person-20260924-audited/human_gold_annotations.jsonl|36|36|0|
|data/ktd-codex-review-batch-03-person-20260924-audited/source_human_review_export.jsonl|36|0|36|
|data/ktd-codex-review-batch-04-validation-20260924-audited/codex_human_comparison.jsonl|6|0|6|
|data/ktd-codex-review-batch-04-validation-20260924-audited/human_gold_annotations.jsonl|6|6|0|
|data/ktd-codex-review-batch-04-validation-20260924-audited/source_human_review_export.jsonl|6|0|6|
|data/ktd-codex-review-batch-05-blind-validation-20260924-audited/codex_human_comparison.jsonl|34|0|34|
|data/ktd-codex-review-batch-05-blind-validation-20260924-audited/human_gold_annotations.jsonl|34|34|0|
|data/ktd-codex-review-batch-05-blind-validation-20260924-audited/source_human_review_export.jsonl|34|0|34|
|data/ktd-codex-review-pilot-20260923-audited/codex_human_comparison.jsonl|33|0|33|
|data/ktd-codex-review-pilot-20260923-audited/human_gold_annotations.jsonl|33|33|0|
|data/ktd-codex-review-pilot-20260923-audited/source_human_review_export.jsonl|33|0|33|
|data/ktd-formal-review-20260903/human_confirmation_queue.jsonl|751|0|751|
|data/ktd-test-real-monitoring-full-20260924/gold_overlap_map.jsonl|142|0|142|
|data/ktd-test-real-monitoring-full-20260924/input/gold_overlap_map.jsonl|142|0|142|
|data/recheck-released-fp-20260930/dataset-source/gold_labels.jsonl|198|0|198|
|data/recheck-released-fp-20260930/dataset-source/other_class_gold_labels.jsonl|24|0|24|
|data/recheck-released-fp-20260930/dataset-source/reserve_gold_labels.jsonl|307|0|307|
|data/recheck-released-fp-20260930/labeling-package/annotations.csv|480|464|16|
|data/review-retrain-20260907/human-dataset/labels.jsonl|557|0|557|
|data/vehicle-surface-gate-eval-20260929/frozen/rows.jsonl|215|0|215|
|data/vehicle-surface-gate-eval-20260929/holdout/human_labels.jsonl|0|0|0|
|data/vehicle-surface-gate-eval-20260929/prerelease-evidence/annotation-package/annotations.csv|60|60|0|
|data/vehicle-surface-gate-eval-20260929/prerelease-evidence/hires-review-package/annotations.csv|91|81|10|
|data/vehicle-surface-gate-eval-20260929/prerelease-evidence/new-events-hires-package/annotations.csv|17|17|0|
|data/vehicle-surface-gate-eval-20260929/prerelease-evidence/supplement-annotation-package/annotations.csv|31|31|0|
|data/vehicle-surface-gate-eval-20260929/random-traffic-hires-package/annotations.csv|200|191|9|
|/Users/wangxiaolin/PycharmProjects/review-annotation-platform-prompt-gold/data/prompt-eval/snapshots/pg1-bc8c818184d8/gold_records.jsonl|535|535|0|

## 检查范围

- 12个复检回归样本的独立完整清单尚未定位；已把本地recheck-runs、prompt-eval快照与formal-review曝光并入证据；其余不明历史来源保守进开发集，待逐个确认。
- 封存答案sealed-copy未打开；仅进行金标清点，不运行模型或准确率评测。
- 同一金标可能有多个来源/population，来源贡献和population表不可相加；各自去重计数。
- 本机无原图但有图片哈希的行仅保留哈希声明，不声称图片已现场核对。
- 旧平台无时区标注时间依据annotation_service.utc_now()按UTC解析后转Asia/Shanghai；原始字符串仍留在接受行快照。

source_rows.jsonl的剔除行仅保留ref、源哈希及原因，无原始内容。接受行保留原始快照；网页几何信息的补齐来源和哈希见geometry_refs及exposure_sources.jsonl。

资源清理：未启动容器、GPU、后台进程或端口；不复制图片与模型，只保留脚本、测试、最终清单和验证证据。原工作树保留供确认，本次未新增工作树。

验证：20项定向测试通过；源文件哈希、逐行对账、剔除行无原内容、敏感字段扫描、冲突隔离、整组模型曝光限制、上海时区、同快照重建一致、测试登记未冻结均已核对。12条回归的独立清单仍缺少证据，未标为验证通过。详见verification.json。
