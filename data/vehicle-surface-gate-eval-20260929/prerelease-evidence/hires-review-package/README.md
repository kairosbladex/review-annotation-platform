# 车辆支撑面 · 高清原帧复核包

首批 60 个和补充 31 个人工标签都是在 960×720 低清裁图上打的，很多目标只有几像素，肉眼无法可靠判断。本包把这 91 个候选框投影回服务器原始来源重新复核：

- 88 个来自原始视频（1920×1080，约 30 fps），3 个来自静态原图（4032×3024）。
- 投影使用 `ktd-test-real-monitoring-full-20260924/output/{video,still}_matches.jsonl` 中 `status=strong` 的 `homography_annotated_to_source`。
- 视频在记录时刻前后 ±0.6 秒逐帧比对，取与低清图整体最吻合的一帧（实际偏移不超过 0.17 秒）。
- 核验结果：整帧相似度（NCC，1 为完全一致）全部 ≥ 0.9965；框附近区域中位数 0.998，全部 ≥ 0.9。明细见 `results.jsonl` 的 `full_ncc` / `roi_ncc`。

每个候选的素材：高清局部（有框/无框）、完整原帧、匹配时刻前后各 2 秒的同区域短视频、±0.5/±1 秒同像素区域的相邻帧，以及原低清裁图作对照。相邻帧和短视频在无人机移动时会偏，只作辅助。

`annotations.csv` 的 `label` 全部留空，由 wxl 在网页中重新判断；`lowres_label` 仅保留原低清标签作对照，页面默认隐藏。首批和补充包自己的 `annotations.csv` 不被修改。

启动（本机 127.0.0.1）：

    python3 data/vehicle-surface-gate-eval-20260929/prerelease-evidence/hires-review-package/label_server.py --port 8791

重新生成素材：在服务器 `/data/workspace/jzheng/claude-hires-review-20260929/` 运行 `extract_hires_review_views.py --jobs jobs.jsonl --work-root /data/workspace/jzheng/codex-ktd-real-monitoring-20260924 --out out`，再把 `out/` 同步到本目录。

> 2026-10-07：服务器上的中间目录已清理，上面提到的服务器工作目录不再存在；本包的高清素材（frames/crops/neighbors/clips）保留在本机，未入库。需要重新生成时，从 09-24 快照 `codex-ktd-real-monitoring-20260924` 或测试库重新取原片，再运行 `scripts/extract_hires_review_views.py`。
