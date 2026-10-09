# Luna 复检分类型提示词自动迭代（Hermes Agent）

服务器 ai-lingxi 上按定时任务循环：执行器用 Luna 跑一整轮评测 → Hermes 看结果和错判拼图、写报告、改类别定义卡并提交下一版 → 执行器再跑。
目标是提高各类别准确率、减少“看不清”（uncertain），不改测试环境；线上替换须人工在锁定集上复核后另行决定。

## 组成

| 文件 | 作用 |
|---|---|
| `build_dataset.py` | 本地组装数据：有标签池按事件切 dev（Hermes 可见）/holdout（锁定）；09-24 快照未用事件做无标签池 |
| `prompts/p000.json` | 线上旧提示词（基线，组装结果与测试环境逐字一致）；服务器上另有 `p000b`（基线重复跑，估计随机波动） |
| `prompts/p001.json` | 人工第一版分类型定义卡：违建=蓝/绿色顶棚、光斑须在光伏板上、道路裂纹定义、其余按俯拍外观与易混物体 |
| `promptlib.py` | 提示词组装（只放入本图出现类别的卡，固定结尾保证输出格式不变）与校验（长度、禁止“必须输出true”类句子） |
| `luna_experiment_worker.py` | 实验 worker：与线上 worker（816cc81）相同，仅提示词随请求传入；只监听 127.0.0.1:18397 |
| `runner.py` | 执行器：停止条件、资源与共享服务检查、跑一轮、打分、拼图；新最好版本才跑 holdout（结果写 `human/`，Hermes 看不到） |
| `metrics.py` | 合格标准（相对 p000 与重复运行 p000b，dev 集）：总体准确率高于两次基线较高者；10 框以上类别不低于两次基线较低者 3 个百分点以上；人员误放不多于基线较多者 +3；无标签 uncertain 不高于两次基线平均 |
| `samples.py` | 错判拼图（每类每种错误 ≤12 个框，标人工标签和模型结论） |
| `mcp_server.py` | Hermes 可用的工具：读状态/结果、看拼图、读/提交提示词、写报告/台账、提建议 |
| `prompt.md` | Hermes 每次唤醒的步骤与改卡原则 |
| `run_brain_tick.sh`、`systemd/` | 定时器：执行器每 5 分钟；大脑每 10 分钟检查，只有需要时才加载模型 |

## 数据（2026-10-09 构建）

- dev：289 张图、735 个标注框；holdout：137 张、427 框（光斑只有 5 框，小类样本很少）。
- 无标签池：122 张，道路裂纹 197、车 124、违建 102、光斑 84 等。道路裂纹没有人工标签，只能观察 true/false/uncertain 分布。
- 标签来源：10-08 设计集、10-09 v2/v3 验证集的人工金标。holdout 只在出现新最好版本时评测。

## 资源与停止

- 根目录 `/data/workspace/jzheng/luna-prompt-autotune/`：`state/`（提示词、轮次、指标、拼图、报告、台账，挂给 Hermes）、`human/`（holdout 结果）、`runs/`（逐图原始输出）、`logs/`、`private/`（worker 令牌，仅本用户可读）。
- 实验 worker：systemd 用户服务 `luna-autotune-worker`，MemoryMax 4G、无 swap、CPU 300%，并发 6；无排队版本时执行器自动停掉。共享 worker 18396 与测试环境不受影响。
- 大脑：`qwen3.6` 在共享服务 L20 上按需加载（模型未加载且空闲显存 < 30GB 时跳过），容器 4GB/2 核。共享 ollama 设了 `OLLAMA_KEEP_ALIVE=0`，冷加载约 90 秒，Hermes 会等不及放弃，所以唤醒前先用原生接口加载并保持 25 分钟，结束后立即卸载。
- 限额：每天最多 8 个版本，2026-10-16 后自动停止；Hermes 连续 3 次唤醒无进展、或判定收敛且空闲 3 小时，自动写 STOP 并停用定时器。

```bash
# 立即停止（下次执行器运行时停掉 worker 并停用定时器）
echo "人工停止" > /data/workspace/jzheng/luna-prompt-autotune/state/STOP
# 直接停用定时器与 worker
systemctl --user disable --now luna-autotune-runner.timer luna-autotune-brain.timer
systemctl --user stop luna-autotune-worker
```

查看进度：`state/rounds/`、`state/reports/`、`state/ledger.jsonl`、`logs/runner.log`、`logs/brain/`。

## 已知局限

- 每轮单次调用。同一提示词跑两次（p000、p000b）有 15.4% 的框结论翻转，dev 准确率 73.0% / 71.2%、无标签 uncertain 45.2% / 39.3%，合格标准因此按两次基线的范围设定。
- Hermes 只看到 dev，反复迭代仍可能过拟合 dev；最终结论以 holdout 为准。
- 道路裂纹 p001 冒烟测试中 5 个框全部判为“直线、像标线或接缝”的 false，是否正确未知，需要人工标注一批道路裂纹。
