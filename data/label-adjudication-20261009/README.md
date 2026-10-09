# 疑似标签错误复核包（2026-10-09）

部分人工标签是在原视频高清帧上标的。移动目标如果和送进复检的画面有时间差，框会落到旁边，清楚的车或人可能被标成“不是”。Luna 提示词自动迭代遇到这种情况时，只把它们记成疑似标签问题，不据此改类别定义。

本包把方向比较稳定的不一致框交给人工再看一遍。复核结果先记在本目录，不直接改任何金标文件，也不调用模型。

## 挑选规则

数据只读自服务器 ai-lingxi 的 `/data/workspace/jzheng/luna-prompt-autotune/`：

- 人工标签：`assets/labels.json`
- 框和图片：`assets/items.json`、`assets/images/`
- 三次运行的原始结论：`runs/p000/out`、`runs/p000b/out`、`runs/p001/out`。每个框用结果里的 `raw`，取值是 `true`、`false` 或 `uncertain`。

全部纳入，不抽样。满足下面任一条的框都进入队列：

1. 人工标签为 false，且 p000、p000b、p001 三次里至少两次 `raw` 为 true。
2. 人工标签为 true，且三次 `raw` 都是 false。

`uncertain` 既不算判是，也不算判不是。某次没有结果时，不能算作“三次都是 false”；另外两次已经是 true 时，仍满足第 1 条。

有人工标签的框共 1128 个（dev 685，holdout 443）。holdout 只有 p000 的结果，没有 p000b 和 p001，做不了三次比较，所以 0 个入选。dev 里有 16 个框缺少 p001（两个图片单元那次评测失败），它们的人工标签都是 true，已有的两次也不是 false，因此也不符合第 2 条。这是规则本身的结果，不是抽样时丢掉的。无标签池没有人工标签，不在范围内。

顺序用固定种子 20261009 打乱，避免同一类框连在一起。

## 数量

共 54 个框、37 张图。

| 类别 | 人工不是，至少两次判是 | 人工是，三次都判不是 | 合计 |
|---|---:|---:|---:|
| 车 | 18 | 1 | 19 |
| 人员 | 15 | 4 | 19 |
| 光斑 | 10 | 0 | 10 |
| 两轮车 | 2 | 0 | 2 |
| 三轮车 | 1 | 0 | 1 |
| 没戴头盔 | 1 | 0 | 1 |
| 卡车 | 1 | 0 | 1 |
| 公交车 | 0 | 1 | 1 |
| 合计 | 48 | 6 | 54 |

有标签但没有框入选的类别：光伏板（18）、违建（141）。

人工不是且至少两次判是的 48 个框，三次原始结论分布是：三次都是 true 共 23 个；p000=true、p000b=uncertain、p001=true 共 11 个；p000=uncertain、p000b=true、p001=true 共 6 个；p000=true、p000b=true、p001=false 共 8 个。人工是且三次都判不是的 6 个框，三次都是 false。

## 怎么复核

```sh
python3 data/label-adjudication-20261009/label_server.py
```

浏览器打开 http://127.0.0.1:8799/ 。服务只监听本机。

页面给出局部放大、整图缩略图和原人工标签。图上原有的红字、红框是平台截图自带的，只看黄色框。选项是“维持原标签 / 改为是 / 改为不是 / 看不清”，可以填备注。快捷键 1、2、3、4 和左右方向键。

页面和接口不返回模型结论，也不提供 `selection.json`。`selection.json` 只用于事后核对某个框为什么入选，复核时不要打开。

结果只写本目录 `adjudication.csv`。图片在 `images/`、`crops/`、`thumbs/`，已写入 `.gitignore`，不进 Git。37 张原图共 11740048 字节。

## 更正清单

复核完成后，在仓库根目录运行：

```sh
python3 data/label-adjudication-20261009/build_correction_list.py
```

脚本只把“改为是”或“改为不是”、并且和原人工标签不同的框写入 `corrections.csv`。维持原标签、看不清、未复核，以及点了和原标签相同选项的框，都不列出。

`corrections.csv` 的 `original_label` 是原人工标签，`corrected_label` 是复核后应采用的标签，`true` 表示是，`false` 表示不是。脚本只写这个清单，不修改 `labels.json` 或任何金标文件。是否改金标由人工另定。

## 重新生成

`build_package.py --dataset <数据根目录>` 读取 `assets/labels.json`、`assets/items.json` 和 `runs/p000|p000b|p001/out`，写出队列、`summary.json` 和 `selection.json`。`adjudication.csv` 里已经有复核结果时，脚本拒绝覆盖。

原图不进 Git。换机器时在本目录执行下面的只读复制，再生成裁图。`prepare_views.py` 会保留已经填过的复核结果。

```sh
python3 - << 'PY'
import csv
from pathlib import Path
rows = list(csv.DictReader(open("adjudication.csv", encoding="utf-8-sig")))
Path("/tmp/label-adjudication-image-files.txt").write_text(
    "".join(f"{unit}.jpg\n" for unit in sorted({row["unit"] for row in rows}))
)
PY
mkdir -p images
rsync -a --files-from=/tmp/label-adjudication-image-files.txt \
  ai-lingxi:/data/workspace/jzheng/luna-prompt-autotune/assets/images/ images/
python3 prepare_views.py
```
