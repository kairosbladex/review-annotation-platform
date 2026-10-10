# 道路裂纹原帧复标：复验 17 事件（2026-10-10）

用途：独立复验道路裂纹卡（yoloe 分支 `feature/luna-crack-card-switch` 的 `p000-pure-class-cards`）。这 17 个事件此前没有参与写卡、调卡或任何评测。

## 来源

- 09-24 测试环境快照中 57 个道路裂纹事件，除去自动迭代无标签池已用的 40 个，剩余 17 个全部纳入，没有挑选：17 张截图都有通过质量门槛的原视频匹配，截图与已用图片无重复（SHA256），共 65 框，7 个航次。
- 局限：其中 8 个事件来自航次 `469255472307511552`，该航次在前 40 张里也有 21 张，属于“新事件、部分同航次”，不是完全独立的新场景；同为 AF5A 设备 9 月 22~23 日的 1080P 录像。
- `scripts/crack_clean_frames.py --events events.json` 在服务器取帧：匹配时刻 ±1.5 秒逐帧解码，与快照无标注参考图相关最低 0.9981，65 框全部在画面内（`frames-audit.json`、`allowlist.json`）。图片不入库，服务器目录 `/data/workspace/jzheng/crack-holdout-frame-20261010/`。

## 标注

页面与口径同 `../road-crack-relabel-hd-20261010`：左为截图局部 | 原帧局部，右为原帧大范围；不显示截图标签和模型结论；顺序种子 20261011。

```bash
python3 data/road-crack-relabel-hd-holdout-20261010/label_server.py --port 8801
```

结果写本目录 `annotations.csv`。标完后再用 Luna 跑“线上 p000”与“p000-pure-class-cards”，截图和原帧各两次。
