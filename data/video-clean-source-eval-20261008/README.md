# 视频事件找回干净帧离线回放（2026-10-08）

结论与方法见 yoloe-algorithm-service 分支 `feature/video-clean-source` 的 `docs/verification/video-clean-source-20261008.md`。

- `selection.json`：113 个事件、212 个人工标注框（人员来自 `person-labeling-20261008`，车辆来自 random-traffic 高清标注包）。
- `manifest.json`：按 Java 将要传的格式组装的请求（截图 + 已标注框 + `clean_video_sources`）。
- `out-video/`：带视频回放的原始响应；`out-legacy/`：车辆事件不带视频的旧版对照。
- `snapshot_status.json`：09-24 快照离线找帧状态；`watchdog.log`：共享 GPU 与测试服务看护记录。
- 打分：

      python3 scripts/score_video_clean_source.py --selection data/video-clean-source-eval-20261008/selection.json \
        --video data/video-clean-source-eval-20261008/out-video --vehicle-legacy data/video-clean-source-eval-20261008/out-legacy \
        --person-single data/person-labeling-20261008/eval/out-single --person-clean data/person-labeling-20261008/eval/out-clean \
        --snapshot-status data/video-clean-source-eval-20261008/snapshot_status.json --out data/video-clean-source-eval-20261008/report.json
