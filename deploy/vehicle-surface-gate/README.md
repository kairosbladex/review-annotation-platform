# 车辆支撑面关卡

## 新样本验收

本轮状态：**需要再补标 31 个，暂不作 v3 发布或放弃结论**。

- 输入基线：`docs/vehicle-surface-prerelease-evidence` 的 `c108aeb`；人工标签来自 `data/vehicle-surface-gate-eval-20260929/prerelease-evidence/annotation-package/annotations.csv`。
- 首批校验：60/60 行均已填写，且只使用 `真车`、`屋顶设施`、`看不清`。
- 首批数量：真车 38、屋顶设施 16、看不清 6。
- 门槛缺口：屋顶设施距离 30 个还缺 14 个；屋顶设施与看不清合计 22 个，距离 30 个还缺 8 个真车样本。
- 因门槛未满足，本轮没有计算 v3 的屋顶设施放过数、单侧 95% 上界、看不清误确认数或真车找回率，也没有与 holdout 合并调参。

已按规则从同一 `screen.jsonl` 的其余 `building_roof` 候选中生成补充包：

`data/vehicle-surface-gate-eval-20260929/prerelease-evidence/supplement-annotation-package/`

补充包包含 31 个候选及本地裁图，`annotations.csv` 的 31 个 `label` 全部留空。候选已排除首批候选、holdout 候选和 A 组事件；模型审计字段只保留在 `screen.jsonl` / `selected.jsonl`，不回写人工标签。原图取回路径和 SHA-256 见 `source-manifest.json`。

补充包标注完成后，重新执行标签完整性和两个数量门槛校验；只有门槛满足，才允许继续计算 v3 指标并给出发布/放弃建议。若后续观察到 v4 规则，只能记为探索性候选，必须另起未见样本验收。
