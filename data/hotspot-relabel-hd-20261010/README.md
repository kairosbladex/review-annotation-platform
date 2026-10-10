# 光斑原图复标（2026-10-10）

道路裂纹在 960×720 平台截图上的人工标签不可靠（原帧重标后，77 个框从“不是”改成了“是”）。光斑评测和改卡之前，先在原图上重标。本包不显示截图旧标签，也不显示模型结论。

## 口径

光斑 = 光伏板组件表面的局部异常亮斑。不在光伏板上的一律不是。整片组件均匀发亮（整体反光）不是。看不清就选“看不清”，不要猜。

## 来源

- 09-24 测试环境快照 `/data/workspace/jzheng/codex-ktd-real-monitoring-20260924/`。纯光斑事件 100 个、281 个框（`input/inference_manifest.jsonl` 里类别全是光斑）。截图在 `work/annotated/<event>.jpg`。匹配记录在 `output/still_matches.jsonl`、`output/video_matches.jsonl`。
- 合格标准沿用快照字段 `clean_usable_for_inference`（照片门 `still-alignment-candidate-aware-v2`，视频门 `video-temporal-alignment-v1`）。不另挑子集。
- 照片和视频都收。两者都合格时用照片，因为分辨率更高。这批数据里没有事件同时拥有合格照片和合格视频：视频匹配当时跳过了已经合格的照片事件。合格照片事件用的就是照片。
- 照片直接读 `source_local_path`，用 `homography_annotated_to_source` 投影。视频在 `matched_raw_time` ±1.5 秒内逐帧解码，取与快照已核验对齐帧相关最高的一帧（本批 0.9998–0.9999），再投影。
- 脚本：`scripts/crack_clean_frames.py --class-name 光斑`。服务器任务目录 `/data/workspace/jzheng/hotspot-relabel-20261010/`。运行限制 MemoryMax=3G、MemorySwapMax=0、CPUQuota=200%。没有调用模型，没有修改 `luna-prompt-autotune/assets`，没有连接测试库。

## 框数

| 范围 | 事件 | 框 |
| --- | ---: | ---: |
| 快照纯光斑 | 100 | 281 |
| 进入本标注页 | 37 | 93 |
| 有合格原片，但框投到画面外 | 4 个事件整组投出，另有 11 个框落在已收录事件上 | 9 + 11 |
| 没有合格原片 | 59 | 168 |

93 个框的原图：

- 1920×1080 原视频帧 79 框（29 个事件）
- 1280×1024 热成像照片（文件名 visual=T）12 框（6 个事件）
- 4032×3024 可见光照片（visual=W）2 框（2 个事件）

页面顺序用种子 20261010 打乱。6 个框是另一条事件的同一张截图（4 组重复截图里，有 3 组至少一框进了本页），页面写着对应事件编号，之后不要当成独立样本。

合不上的原因（事件数 / 框数）在 `excluded.json`，汇总如下：

- 快照没有原照片或原视频：40 事件 / 85 框
- 照片几何预检未过，且没有视频匹配记录：8 / 55
- 照片相关不够，视频几何预检未过：4 / 15
- 照片相关不够，视频整图相关低于 0.80：4 / 10
- 原片合格，但全部框投影后不在画面内：4 / 9
- 照片相关不够，且没有视频匹配记录：2 / 2
- 没有照片匹配记录，视频几何预检未过：1 / 1

投影后只是部分框出画的 11 个框在 `summary.json` 的 `outside_boxes`。

## 自动迭代里写过卡的框

`items.json` 里 dev 光斑框 22 个、holdout 光斑框 6 个，按截图哈希和框的交并比（≥0.99）对到快照框，记为 `autotune_dev` / `autotune_holdout`。这 28 个框全部没有合格原片，都在 `excluded.json`，不在本页。本页 93 个框的来源都是 `new`（新样本）。页面仍会显示来源，方便以后混入写过卡的样本时区分。

## 局限

- 多数合格原图不是 4032×3024 可见光照片。6 个照片事件是 1280×1024 热成像；29 个事件是 1920×1080 视频帧。抽查时，热成像截图上的红字“光斑”在原图上消失，框落在同一块组件的亮斑上。热成像上没有可见光那种“整片反光”，对不上口径时选“看不清”。
- 单应矩阵把有些截图投得比视频画面更大（面积比大约 1.25），靠上沿或下沿的框会出画，这些框没有进页面。
- 重复截图的两条事件都保留了，只做了标记，没有删掉。
- 原图不入库。换一台机器标注时，先把服务器上的 `compare/`、`context/` 同步到下面的 `images/`。

```bash
rsync -a ai-lingxi:/data/workspace/jzheng/hotspot-relabel-20261010/compare/ data/hotspot-relabel-hd-20261010/images/compare/
rsync -a ai-lingxi:/data/workspace/jzheng/hotspot-relabel-20261010/context/ data/hotspot-relabel-hd-20261010/images/context/
python3 data/hotspot-relabel-hd-20261010/label_server.py --port 8801
```

打开 http://127.0.0.1:8801/ 。结果只写本目录 `annotations.csv`。
