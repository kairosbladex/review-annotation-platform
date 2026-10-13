# KTD GPU 冒烟训练报告

> 日期：2026-09-03
> 范围：仅验证 1 个 epoch 的训练链路

## 结果

本次 YOLOE 1 epoch 冒烟训练已在公司 GPU 服务器 `AI-lingxi` 上完成，使用宿主机 GPU 4（NVIDIA L20）。训练完成 1/1 个 epoch、22/22 个 batch，并生成了 `best.pt` 和 `last.pt`。

这是训练链路验证结果，不代表模型质量、mAP、召回率或生产可用性。

## 运行环境

| 项目 | 值 |
|---|---|
| 服务器 | `192.168.130.238` / `AI-lingxi` |
| 容器 | `yoloe_service:latest` |
| Python | 3.10.12 |
| PyTorch | 2.9.1+cu128 |
| Ultralytics | 8.4.89 |
| GPU | 宿主机 GPU 4，NVIDIA L20 |
| 容器设备 | `cuda:0`，映射自宿主机 GPU 4 |

现有业务容器保持运行。本次冒烟训练使用的一次性容器已退出，GPU 4 已释放。

## 数据集

数据来源为 `data/recheck-runs/ktd-uncertain-recheck-20260903/`：

- 下载并检查原图 106 张；
- 固定划分为训练集 85 张、验证集 21 张；
- 10 个类别共 196 个正样本伪标签框；
- 类别编号是数据集内部编号，依据 `class_name` 确定。

当前可用预训练权重是 YOLOE 分割模型 `yoloe-v8l-seg-pf.pt`，而源标签是 5 列检测框。仅为完成本次冒烟训练，已在独立目录 `data/ktd-smoke-seg-dataset-20260903/` 中把每个 bbox 表示为矩形 polygon；原始检测标签没有修改。

## 训练配置

```text
model: yoloe-v8l-seg-pf.pt
task: segment
epochs: 1
imgsz: 640
batch: 4
workers: 0
device: 容器 cuda:0，宿主机 GPU 4
amp: false
验证: 为生成冒烟 checkpoint 而跳过
```

Ultralytics 的最后一个 epoch 会强制进入验证流程。本次冒烟进程只临时替换了进程内的 `validate` 和 `final_eval` 方法，使 checkpoint 无需下载可选的 MobileCLIP 文件即可保存；没有修改任何安装包、镜像或仓库源码。

## 训练输出

`results.csv` 中唯一一行记录如下：

| 指标 | 值 |
|---|---:|
| 训练耗时 | 8.85204 秒 |
| `train/box` | 2.12843 |
| `train/seg` | 2.89643 |
| `train/cls` | 25.6827 |
| `train/dfl` | 1.81495 |
| 日志显示的 GPU 峰值显存 | 约 6.09 GiB |

本机重新加载 checkpoint 的检查已通过：任务类型为 `segment`，类别数为 10，checkpoint 元数据 `train_args.epochs=1`，Ultralytics 版本为 `8.4.89`。checkpoint 中从 0 开始计数的 `epoch=0`，对应 `results.csv` 中已完成的第 1 个 epoch。

## 产物

远程运行目录：

```text
/data/workspace/jzheng/yoloe_service/data/ktd-smoke-runs/yoloe-v8l-seg-1epoch-20260903-gpu4-final/
```

已同步到本机的文件：

- `data/ktd-smoke-runs/remote-gpu4-20260903/weights/best.pt`
- `data/ktd-smoke-runs/remote-gpu4-20260903/results.csv`
- `data/ktd-smoke-runs/remote-gpu4-20260903/args.yaml`

本机 `best.pt` 的 SHA-256：

```text
5b76b4c6954c3846af6ff05036e284a6a0dabf00f70c90fa55d2c2424ca55fb4
```

该值与远程 `best.pt` 的 SHA-256 一致。

## 限制

- 标签来自模型复检伪标签，没有使用人工确认的金标准标签；
- 矩形 polygon 只是冒烟训练对 bbox 的临时表示，不是真实分割掩码；
- 跳过了验证和 mAP 计算，`fitness=0` 是冒烟占位值，不是准确率；
- 本次运行不能支持正式训练、部署或任何精确率/召回率结论。
