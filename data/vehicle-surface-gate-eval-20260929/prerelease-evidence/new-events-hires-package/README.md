# 车辆支撑面 · 新事件高清补标包

用于补 v3 验收的两个数量门槛。首批 + 补充包高清复核后，屋顶设施 26、相关层真车 27；只计 v3 作用范围时分别缺 14、12（见 `../hires-review-package/gate-check.json`）。

## 样本来源

- 测试库 `ktd_lx` 只读查询：`sys_annex_file` 中 `file_path LIKE '/zm-zq-test/%' AND imgxy LIKE '%width%' AND imgxy NOT LIKE '%type%' AND create_time >= '2026-09-24 22:30:36'`（即 09-24 快照生成之后）。共 68 个事件、624 个车辆框，与 09-24 快照的 1416 个事件无重叠。
- 50 个事件的主图就是无人机原始照片（4032×3024，EXIF 方向正常），检测框直接在原图坐标上；截图与主图同尺寸，只多画了红框。
- 18 个事件是 960×720 视频截图，对应 5 段原始视频（按上传时间取事件后 2~4 分钟内的那段），用 09-24 流程的 `source_match_all.py` 匹配，18/18 为 strong。

## 筛选

- 沿用上次的隔离镜像 `yoloe_service_test:vehicle-surface-v3-20260929-v1` 和判定程序（SHA-256 `057af21f…`，与 v3 评测一致）；临时容器断网、只读挂载，仅用 L20 `GPU-c977647c…`，共享服务 `yoloe_service_test` 前后未变，容器已删除。证据见 `evidence/`。
- 每个事件最多 5 个入选框；共筛 600 个框（原图 436 + 视频 164），未筛的 24 个是原图事件超出每事件上限的框。
- 入包规则：四视图两路一致确认为车、支撑面为 `building_roof`/`unknown`，也就是 v3 部件规则实际会运行的框，全部入包，不再按部件结果分层。共 17 个，来自 6 个事件（原图 16、视频 1）。

## 注意

- 17 个框集中在 6 个事件，同一楼顶/场景内的框彼此相关，统计上不等于 17 个独立样本。
- 即使全部计入，也不足以补齐缺口；需要继续积累新事件。
- `annotations.csv` 的 `label` 全部留空，只能由 wxl 在网页中填写 `真车`、`屋顶设施` 或 `看不清`。模型输出只在 `selected.jsonl` / `evidence/screen-*` 中，用于审计，不作为人工标签。

启动（本机 127.0.0.1）：

    python3 data/vehicle-surface-gate-eval-20260929/prerelease-evidence/new-events-hires-package/label_server.py --port 8792

服务器工作目录：`/data/workspace/jzheng/claude-vsg-new-events-20260929/`；高清素材可在该目录用 `src-package/extract_hires_review_views.py --jobs package/jobs.jsonl --work-root . --out package/out` 重新生成。
