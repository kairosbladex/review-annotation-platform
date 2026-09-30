# 复检统一金标：评测前最后一次重冻

修正小层向下取整偏差 + 12 样本曝光核对，评测前重冻

找到12个回归样本：R045～R056。会话证据 /Users/wangxiaolin/.codex/sessions/2026/09/30/rollout-2026-09-30T14-13-57-01a0f0f2-c708-7932-a95e-62c05f4206e7.jsonl:362（样本选择）及 :667（仅开发回归）。逐条映射见 regression_exposure.jsonl，12/12唯一对应金标，3个整组标记 gpt_prompt_exposed。采用精确核对路径，未采用全输入保守兜底。

主类别按组内框数最多确定；主population按出现框数最多确定；并列按字符串排序。种子20260930。分层键：主类别 × 主population × 是否含false。含false与无false分别计算总名额，四舍五入到最近整数；每层先floor，按余数降序补齐，并列按分层键排序。uncertain-only、GPT曝光、历史未知继续只进开发集；冲突继续隔离；valid_for不变。

本次为尚未评测的最后一次重冻。后续已有组不重切；新组独立按同一规则分配并追加registry。新增来源连接两个既有组时停止并请人工裁决。

对比bba595e见 refreeze_comparison.json。比例分母为符合GPT锁定条件的组，曝光组不参与分配。

车类只对真目标误判假率给出通过/不通过；锁定集假目标达到100条前，假目标判假率只报告数字与置信区间。

|类别|全量 true/false/uncertain 框|locked true/false/uncertain 框|locked true/false/uncertain 组|locked 总组数|档位|
|---|---|---|---|---|---|
|三轮车|3/6/0|1/2/0|1/2/0|3|未充分验证|
|两轮车|29/8/0|1/3/0|1/3/0|4|未充分验证|
|人员|59/40/0|0/0/0|0/0/0|0|仅可开发|
|光伏板|23/1/0|0/0/0|0/0/0|0|未充分验证|
|光斑|15/12/0|0/0/0|0/0/0|0|未充分验证|
|公交车|7/7/0|0/1/0|0/1/0|1|未充分验证|
|卡车|3/12/1|0/6/1|0/6/1|7|未充分验证|
|没戴头盔|3/2/0|0/0/0|0/0/0|0|未充分验证|
|车|958/95/26|218/46/12|79/22/5|93|仅可开发|
|车牌|4/0/0|0/0/0|0/0/0|0|未充分验证|
|违建|78/61/0|2/0/0|2/0/0|2|仅可开发|

## 车类锁定集 population

|population|true框|false框|uncertain框|true组|false组|uncertain组|总组|
|---|---|---|---|---|---|---|---|
|historical_review|0|0|0|0|0|0|0|
|random_traffic|76|6|0|55|6|0|60|
|released_by_recheck|93|9|0|13|2|0|14|
|rooftop_hard_case|59|35|12|28|19|5|44|

population可重叠，不可相加。

## 实际分配比例

含false=True: 31/44 = 70.45%
含false=False: 72/143 = 50.35%

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
当前已纳入金标未发现额外的来源不明组；12个回归样本核对已完成，详见回归曝光映射。

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
