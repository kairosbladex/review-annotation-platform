# 道路裂纹人工标注包（2026-10-09）

本机标注页，用来判断无标签池里的“道路裂纹”框是不是路面裂缝。页面不显示检测置信度、复检结论或模型建议。结果只写本目录 `annotations.csv`。

## 来源

- 服务器 ai-lingxi：`/data/workspace/jzheng/luna-prompt-autotune/assets/items.json`
- 筛选：`split=unlabeled` 且 `class_name=道路裂纹`
- 197 个框，来自 40 张图。36 张图上有多个框（最少 1 个，最多 15 个）。
- 图片是 09-24 测试环境快照里的平台框截图，`images/<unit>.jpg`，全部为 960×720。清单里的 SHA-256 与服务器文件一致。
- 数据集来源字段是 `unlabeled0924`。复制时只读，没有改服务器上的 `assets`（提示词自动迭代正在用这批文件）。

图片不进 Git。换机器时按 `manifest.json` 的 `images` 列表，从上述目录把对应 jpg 拷到本目录 `images/`。

## 标注口径

道路裂纹 = 沥青或水泥路面本身的裂缝，形态不规则、可分叉。

以下都不是道路裂纹：

- 车道线、斑马线、箭头等标线
- 规则直线的伸缩缝或切缝
- 宽度均匀的灌缝修补带
- 树影、杆影、车辙、水渍油污
- 不在路面上的线条

看不清的不要猜。截图上印着的红字和红框是平台截图自带的，不作为判断依据。黄框才是当前这一框。

## 标注

在仓库根目录执行：

```bash
python3 data/road-crack-labeling-20261009/label_server.py --port 8798
```

浏览器打开 http://127.0.0.1:8798/ 。每个框同时显示两张图：左侧是原框宽和高各外扩到 2 倍的局部（黄框标在原框上），右侧是整图缩略图。快捷键 1 是裂纹 · 2 不是裂纹 · 3 看不清 · ←/→ 翻页。

顺序用固定种子 `20261009` 打乱，打乱结果写在 `manifest.json` 的 `order`。服务只监听 `127.0.0.1`。

## 导入

标完后在仓库根目录执行：

```bash
python3 data/road-crack-labeling-20261009/import_labels.py
```

脚本读本目录 `annotations.csv`，写出 `labels-road-crack.json`。结构与 `assets/labels.json` 相同：键是候选框 `key`，值包含 `label`、`class_name`、`source`、`split`。`是裂纹` 写成 `true`，`不是裂纹` 写成 `false`，看不清和未标注不写入。`source` 仍是 `unlabeled0924`，`split` 仍是 `unlabeled`。

这个文件只留在本目录。不要把它合并进正在运行的 `assets/labels.json`。若以后要放进自动迭代的有标签池，需要另行决定进 dev 还是 holdout。
