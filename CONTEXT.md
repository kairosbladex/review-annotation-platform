# Review Annotation Platform Context

## Domain language

- **Sample（样本）**：一张原始图片及来源元数据。一个样本可以被多个模型运行复用。
- **Asset（资源）**：样本的原图、复核图、模型可视化图或导出文件在 MinIO 中的对象引用。
- **ModelRun（模型运行）**：一个模型版本对一批样本执行的一次不可覆盖的识别运行。
- **Candidate（候选目标）**：某次模型运行产生的一个归一化目标框和模型判断。
- **Annotation（人工标注）**：人工对候选目标作出的 `true`、`false` 或未标注结论；它不修改模型原始结果。
- **Annotation task（标注任务）**：一个样本在某个模型运行下的待标注工作项；所有候选目标完成后才完成。
- **Ingestion event（接入事件）**：复检服务或导入流程提交的一次幂等数据接入请求。

## Invariants

- `event_id` 幂等；重试不会创建重复样本。
- Candidate 的模型输出不可变；人工结论存放在 Annotation 中。
- bbox 在持久化和接口中统一为归一化 `[x1, y1, x2, y2]`。
- 标注任务只有在全部 Candidate 都有人工标签后才进入 `completed`。
- 图片路径不作为长期业务主键；生产数据使用 MinIO bucket + object key。
- 一期只标注已有候选框；改框、增框、删框和改类别属于二期。
