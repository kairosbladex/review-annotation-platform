# 车辆支撑面关卡

## 新样本验收

### 2026-09-29 高清原帧复核后

本轮状态：**两个数量门槛仍未满足，暂不作 v3 发布或放弃结论**。

- 标签来源：`data/vehicle-surface-gate-eval-20260929/prerelease-evidence/hires-review-package/annotations.csv`。91 个框（首批 60 + 补充 31）已投影到原始视频 1920×1080 帧或 4032×3024 原图上重新人工判断；91/91 已填写。
- 标签数量：真车 50、屋顶设施 26、看不清 15；与低清标签相比改了 9 个。
- 门槛 1（屋顶设施 ≥ 30）：26，缺 4。
- 门槛 2（首批“屋顶/看不清候选”抽样层与补充包中的真车 ≥ 30）：27，缺 3。
- 补充包抽样问题：补充包 31 个中只有 2 个经过四视图一致确认；其余 29 个在四视图阶段已被拒绝或不确定，v3 部件规则根本不会运行。只计 v3 实际作用范围时，屋顶设施 16、真车 18，距离 30 分别缺 14、12。
- 同一 `screen.jsonl` 的 348 个候选中，楼顶类候选已全部用完，剩余 257 个都是地面，无法再从中补齐；继续补标需要在新事件上重新筛选。
- 按规则，本轮未计算 v3 放过数、单侧 95% 上界、看不清误确认数或真车找回率。核对脚本：`scripts/check_vehicle_surface_gate.py`，结果见 `hires-review-package/gate-check.json`。

### 首批低清标注

本轮状态：**需要再补标 31 个，暂不作 v3 发布或放弃结论**。

- 输入基线：`docs/vehicle-surface-prerelease-evidence` 的 `c108aeb`；人工标签来自 `data/vehicle-surface-gate-eval-20260929/prerelease-evidence/annotation-package/annotations.csv`。
- 首批校验：60/60 行均已填写，且只使用 `真车`、`屋顶设施`、`看不清`。
- 首批数量：真车 38、屋顶设施 16、看不清 6。
- 门槛缺口：屋顶设施距离 30 个还缺 14 个；首批抽样层“屋顶/看不清候选”中的真车为 14 个，距离 30 个还缺 16 个。
- 因门槛未满足，本轮没有计算 v3 的屋顶设施放过数、单侧 95% 上界、看不清误确认数或真车找回率，也没有与 holdout 合并调参。

已按规则从同一 `screen.jsonl` 的其余 `building_roof` 候选中生成补充包：

`data/vehicle-surface-gate-eval-20260929/prerelease-evidence/supplement-annotation-package/`

补充包包含 31 个候选及本地裁图，`annotations.csv` 的 31 个 `label` 全部留空。候选已排除首批候选、holdout 候选和 A 组事件；模型审计字段只保留在 `screen.jsonl` / `selected.jsonl`，不回写人工标签。原图取回路径和 SHA-256 见 `source-manifest.json`。

补充包标注完成后，重新执行标签完整性和两个数量门槛校验；只有门槛满足，才允许继续计算 v3 指标并给出发布/放弃建议。若后续观察到 v4 规则，只能记为探索性候选，必须另起未见样本验收。
