# 复检统一金标：第0步切分已冻结

冻结时间：2026-09-30T16:19:35.268966+08:00。人工金标 1453 框；冲突 5 框继续隔离，不阻塞冻结。

固定种子20260930。分层键为类别 × population × 组内是否含false。按排序后的组ID、排序后的分层，使用同一个Python random.Random(seed)逐层shuffle；含false组70%进入locked_test_gpt，只含true的组50%；组数向下取整floor(N×比例)。跨类别或population的组以完整类别集合、population集合为独立分层，不拆组。true与uncertain混合但无false的组按50%；只有uncertain的组保留development。

GPT已曝光或历史不明的整组继续进入development，原有曝光规则不变。仅Qwen曝光不阻止GPT锁定测试，valid_for=["gpt"]，禁止用于Qwen评测。冲突框隔离，不计入分层标签和评测框数。

split_registry.jsonl保留旧记录并追加本次冻结记录；同一group_id按最后一条记录解释当前切分。locked_test_gpt现已冻结，不得重新随机切分。proposed_groups.jsonl是历史提案，当前权威为split_registry.jsonl与splits.jsonl。

12个复检回归样本ID尚未提供。收到完整ID清单后按candidate_id/gold_id及来源追溯核对；若命中locked_test_gpt，将所在整组移回development，清空valid_for，在split_registry追加原切分、新切分、触发ID、原因和时间，并同步gold/splits/统计/校验和。禁止只移单框，也不补抽其他组。不明或一对多ID须先核清。

车类当前只能正式评测“真目标被误判为假的比例”。“假目标判假率”要等locked_test_gpt中车类假目标达到100条后才给通过结论；现阶段只报告数字和置信区间。通用完整评测样本门槛仍是真/假各100框、各10独立组；uncertain不计真/假。样本档位不表示模型已通过。比例报告给出分子/分母和95%置信区间，并注明同组相关性，不能把每个框都当作独立样本。

第1～3步线上复检链路尚未开始，必须等待用户确认数据外发授权。当前model_calls、image_uploads、database_writes、event_submissions均为0。

|类别|全量 true/false/uncertain 框|locked true/false/uncertain 框|locked true/false/uncertain 组|locked 总组数|档位|
|---|---|---|---|---|---|
|三轮车|3/6/0|0/1/0|0/1/0|1|未充分验证|
|两轮车|29/8/0|0/2/0|0/1/0|1|未充分验证|
|人员|59/40/0|0/0/0|0/0/0|0|仅可开发|
|光伏板|23/1/0|0/0/0|0/0/0|0|未充分验证|
|光斑|15/12/0|0/0/0|0/0/0|0|未充分验证|
|公交车|7/7/0|0/0/0|0/0/0|0|未充分验证|
|卡车|3/12/1|0/2/0|0/2/0|2|未充分验证|
|没戴头盔|3/2/0|0/0/0|0/0/0|0|未充分验证|
|车|958/95/26|196/34/8|71/20/5|85|仅可开发|
|车牌|4/0/0|0/0/0|0/0/0|0|未充分验证|
|违建|78/61/0|3/0/0|2/0/0|2|仅可开发|

## 车类 locked_test_gpt population 分布

|population|true框|false框|uncertain框|true组|false组|uncertain组|总组数|
|---|---|---|---|---|---|---|---|
|historical_review|0|0|0|0|0|0|0|
|random_traffic|76|7|0|53|7|0|60|
|released_by_recheck|84|1|0|10|1|0|10|
|rooftop_hard_case|45|27|8|22|15|5|36|

同一框或组可能包含多个population，各行不可直接相加。

## population分开清点（无准确率评测）

### historical_review

|类别|全量 true/false/uncertain 框|locked true/false/uncertain 框|locked true/false/uncertain 组|locked 总组数|档位|
|---|---|---|---|---|---|
|三轮车|2/3/0|0/0/0|0/0/0|0|未充分验证|
|两轮车|27/3/0|0/0/0|0/0/0|0|未充分验证|
|人员|59/40/0|0/0/0|0/0/0|0|仅可开发|
|光伏板|23/1/0|0/0/0|0/0/0|0|未充分验证|
|光斑|15/12/0|0/0/0|0/0/0|0|未充分验证|
|公交车|7/6/0|0/0/0|0/0/0|0|未充分验证|
|卡车|3/4/0|0/0/0|0/0/0|0|未充分验证|
|没戴头盔|3/2/0|0/0/0|0/0/0|0|未充分验证|
|车|310/21/0|0/0/0|0/0/0|0|未充分验证|
|车牌|4/0/0|0/0/0|0/0/0|0|未充分验证|
|违建|71/61/0|0/0/0|0/0/0|0|仅可开发|

### random_traffic

|类别|全量 true/false/uncertain 框|locked true/false/uncertain 框|locked true/false/uncertain 组|locked 总组数|档位|
|---|---|---|---|---|---|
|三轮车|0/3/0|0/1/0|0/1/0|1|未充分验证|
|公交车|0/1/0|0/0/0|0/0/0|0|未充分验证|
|卡车|0/2/0|0/0/0|0/0/0|0|未充分验证|
|车|172/11/0|76/7/0|53/7/0|60|未充分验证|

### released_by_recheck

|类别|全量 true/false/uncertain 框|locked true/false/uncertain 框|locked true/false/uncertain 组|locked 总组数|档位|
|---|---|---|---|---|---|
|三轮车|1/0/0|0/0/0|0/0/0|0|未充分验证|
|两轮车|2/5/0|0/2/0|0/1/0|1|未充分验证|
|卡车|0/6/0|0/2/0|0/2/0|2|未充分验证|
|车|399/16/2|84/1/0|10/1/0|10|未充分验证|
|违建|7/0/0|3/0/0|2/0/0|2|未充分验证|

### rooftop_hard_case

|类别|全量 true/false/uncertain 框|locked true/false/uncertain 框|locked true/false/uncertain 组|locked 总组数|档位|
|---|---|---|---|---|---|
|卡车|0/0/1|0/0/0|0/0/0|0|未充分验证|
|车|107/51/24|45/27/8|22/15/5|36|仅可开发|

## locked_test_gpt来源清单

同一金标若有多来源，分别计数，不可相加。

|来源|组|框|population框分布|
|---|---|---|---|
|data/recheck-released-fp-20260930/labeling-package/annotations.csv|15|92|{"released_by_recheck":92}|
|data/vehicle-surface-gate-eval-20260929/prerelease-evidence/annotation-package/annotations.csv|25|28|{"rooftop_hard_case":28}|
|data/vehicle-surface-gate-eval-20260929/prerelease-evidence/hires-review-package/annotations.csv|32|38|{"rooftop_hard_case":38}|
|data/vehicle-surface-gate-eval-20260929/prerelease-evidence/new-events-hires-package/annotations.csv|2|2|{"rooftop_hard_case":2}|
|data/vehicle-surface-gate-eval-20260929/prerelease-evidence/supplement-annotation-package/annotations.csv|11|12|{"rooftop_hard_case":12}|
|data/vehicle-surface-gate-eval-20260929/random-traffic-hires-package/annotations.csv|61|84|{"random_traffic":84}|

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
当前已纳入金标未发现额外的来源不明组；12个回归样本ID待用户提供，跨来源重叠核验仍未完成。

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
