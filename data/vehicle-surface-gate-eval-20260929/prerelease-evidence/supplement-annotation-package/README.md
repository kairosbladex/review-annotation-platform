# 车辆支撑面补标补充包

本包因首批 60 个人工标签未达到验收数量门槛而生成。首批结果为：屋顶设施 16 个；首批抽样层“屋顶/看不清候选”中的真车 14 个。距离两个 30 个门槛分别缺 14 个、缺 16 个。

本包从同一 `screen.jsonl` 的其余 `building_roof` 候选中选出 31 个，排除了首批候选和 holdout 候选；与 A 组事件无交集。`annotations.csv` 的 `label` 全部留空，只能由 wxl 在网页中填写 `真车`、`屋顶设施` 或 `看不清`。

原图来自服务器 `/data/workspace/jzheng/codex-ktd-real-monitoring-20260924/work/clean_frames/`，`source-manifest.json` 保存了取回时的 SHA-256。模型字段只保留在 `selected.jsonl` / `screen.jsonl` 审计文件中，不作为人工标签。
