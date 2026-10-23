# 4×T4 + 2×L20 异构算力与五类业务目标全链路复核交接报告

- 仓库：`/Users/wangxiaolin/PycharmProjects/review-annotation-platform`
- 分支：`docs/luna-business-test-release`
- 被复核基线：`8d778b50c27d43f38b00e08b2e9fa985f5684e74`（24 个文件，+2570/−201）
- 复核执行环境：macOS / Python 3.13.2 / `.venv`；**无 NVIDIA GPU、无 `nvidia-smi`、无 Docker GPU 运行时**
- 复核日期：2026-10-23

---

## 1. 已修改 / 新增文件与 Git SHA

| 文件 | 类型 | 变更要点 |
|---|---|---|
| `backend/app/services/ktd_recheck.py` | 修改 | **F1 修复**：`blue_illegal_roof` 在 Tier 2 前豁免，任何置信度均强制进入 Tier 3；新增 `tier3_scene_dependent_forced_vlm` 原因码 |
| `backend/app/services/auto_annotation_service.py` | 修改 | **F2 修复**：新增 `build_auto_annotation_options()`，把 Settings 中的 mode / Tier 阈值 / vLLM 地址合并进 job options（显式入参优先） |
| `backend/app/config.py` | 修改 | 新增 `auto_annotation_cascade_routing`（env `AUTO_ANNOTATION_CASCADE_ROUTING`，默认 `false`） |
| `backend/app/main.py` | 修改 | 自动标注创建入口改用 `build_auto_annotation_options(payload.options, settings)` |
| `deploy/production.env.example` | 修改 | 注明四个级联配置项现已生效，并新增 `AUTO_ANNOTATION_CASCADE_ROUTING=false` |
| `backend/tests/test_audit_codex_pilot_review.py` | 修改 | **F3 修复**：`_pilot()` 夹具补齐 `source_object_key`；新增 `image_object_key` 非空断言锁死溯源契约 |
| `backend/tests/test_ktd_recheck.py` | 修改 | 新增 2 个回归用例：屋顶类全置信度强制 Tier3、非屋顶低置信度仍自动拒绝 |
| `backend/tests/test_auto_annotation.py` | 修改 | 新增 3 个回归用例：Settings→options 传导、显式入参覆盖、API 落库校验 |
| `scripts/prepare_ktd_clean_source_qwen_dataset.py` | 修改 | **F4 修复**：删除 `math` 未导入的死分支；`--bbox-style` 增加 `--marker-style` 别名 |
| `scripts/eval_five_class_gates.py` | **新增** | 五类门禁 + 三级级联的可复现离线回放评估脚本 |
| `docs/HANDOVER_4T4_2L20_FIVE_CLASS_VERIFY_20261023.md` | **新增** | 本报告 |

提交：见本报告末尾「Git 提交」小节（中文 commit message，未 push）。

---

## 2. 全量 pytest 与 check_production_config 实际运行输出

### 2.1 全量测试

```bash
PYTHONPATH=.:backend .venv/bin/pytest backend/tests agent/tests -q
```

基线（修复前）：

```
FAILED backend/tests/test_audit_codex_pilot_review.py::test_audit_freezes_binary_human_gold_without_splitting_source_group
FAILED backend/tests/test_audit_codex_pilot_review.py::test_audit_rejects_confirming_an_uncertain_machine_suggestion
2 failed, 178 passed, 1 skipped, 2 warnings in 14.95s
```

修复后（最终）：

```
185 passed, 1 skipped, 2 warnings in 15.38s
```

`0 failed`。新增 7 个用例（3 自动标注 + 2 级联 + 2 审计断言增强）。

### 2.2 生产配置校验

```bash
# 基于 deploy/production.env.example 生成 /tmp/test_prod.env，替换全部 REPLACE_ME 占位符
.venv/bin/python scripts/check_production_config.py --env-file /tmp/test_prod.env
```

正向：

```
production config OK: strict MinIO, legacy review adapter, and required secrets are enabled
RETURN_CODE=0
```

负向对照（直接喂原始 example，确认校验未失效）：

```
production config invalid:
- replace example values for: DATABASE_URL, REDIS_URL, ADMIN_PASSWORD, JWT_SECRET,
  SERVICE_API_KEY, MINIO_ENDPOINT, MINIO_ACCESS_KEY, MINIO_SECRET_KEY,
  MINIO_PUBLIC_ENDPOINT, MODEL_SERVICE_BASE_URL, MODEL_SERVICE_API_KEY
RETURN_CODE=1
```

`/tmp/test_prod.env` 已删除（`No such file or directory`）。

---

## 3. Phase 1–4 核心指标验证结果表

### 3.1 Phase 1 — 干净源数据集新表征与溯源

在 `/tmp/ktd_phase1_verify` 构造五类最小干净源验证集（person / vehicle / tricycle / blue_illegal_roof / road_crack，3 train + 2 validation，其中 `crack-01` 为 240×180 小图且 bbox 贴左边界以强制填充），执行：

```bash
scripts/prepare_ktd_clean_source_qwen_dataset.py --marker-style corner_ticks --view-mode 4view
scripts/prepare_ktd_clean_source_qwen_dataset.py --marker-style corner_ticks --view-mode 2view
```

| 校验项 | 4view | 2view | 判定 |
|---|---|---|---|
| 干跑成功、5 类全部 eligible | 5/5，excluded 0 | 5/5，excluded 0 | PASS |
| `dataset_manifest.json` 的 `view_mode` 与实跑一致 | 4view | 2view | PASS |
| 每样本视角数量 | 4 | 2 | PASS |
| 视角总数（provenance 逐张 SHA256 校验） | 20/20 | 10/10 | PASS |
| 图像 SHA256 与磁盘一致 | 一致 | 一致 | PASS |
| 贴边填充为常量 `(230,230,230)`，无镜像/拉伸条纹 | padding `[110,110,80,80]`，填充带外侧 90% 单一色值 | 同左 | PASS |
| `person` 框内中心区域零遮挡 | 逐像素 `max_abs_diff=0`，中心改动像素 0 | 同左 | PASS |
| `road_crack` 框内中心区域零遮挡 | 逐像素 `max_abs_diff=0`，中心改动像素 0 | 同左 | PASS |
| 全图被改动像素仅落在角标带 | person 376 px / crack 542 px，全部在框边界带 | 同左 | PASS |

填充色实现核对：`cv2.copyMakeBorder(..., cv2.BORDER_CONSTANT, value=NEUTRAL_GRAY_RGB)`，`NEUTRAL_GRAY_RGB = (230, 230, 230)`，位于 `_crop_square_with_metadata`（视口平移失败后才填充，非拉伸）。

`corner_ticks` 实现核对：外扩 2px，四角 L 型，臂长 `clip(边长×0.20, 6, 24)`，小框自动收缩避免对角落重叠，中边保持开放。

验证后 `/tmp/ktd_phase1_verify` 已删除。

**观察（非缺陷，待产品决策 F5）**：`roi_scale` 只按 bbox 长边放大、不向图像边界收敛，大目标会产生较高灰边占比：

| 样本 | 类别 | annotated_context 填充率 | annotated_roi 填充率 |
|---|---|---:|---:|
| person-01 | 人员 | 17.9% | 0.0% |
| tricycle-01 | 三轮车 | 17.9% | 0.0% |
| vehicle-01 | 车辆 | 43.2% | 0.0% |
| roof-01 | 蓝色违建顶棚 | 21.9% | **46.4%** |
| crack-01 | 道路裂缝 | 47.5% | 0.0% |

`blue_illegal_roof` 的 context 已由 `clamp_context_to_image=true` 收敛，但 **roi 视角无同等约束**，700×400 的屋顶在 4× 放大后视口 2800px 远超 1920px 画幅，46.4% 为中性灰。

### 3.2 Phase 2 — QLoRA 训练与验证契约

| 契约项 | 代码事实 | 判定 |
|---|---|---|
| L20 SM89 → `bf16` | `_detect_gpu_capabilities`：`major >= 8` → `torch.bfloat16`, `bf16=True`, `fp16=False` | PASS |
| T4 SM75 → `fp16` | 同上 `else` 分支 → `torch.float16`, `fp16=True`, `bf16=False` | PASS |
| 注意力后端 | SM≥8 且有 `flash_attn` → `flash_attention_2`，否则 `sdpa` | PASS |
| 训练超参按卡型自适应 | L20（≥40GB）：bs=4/accum=4/workers=8/pin+persistent；T4：bs=1/accum=16/workers=2 | PASS |
| LoRA 全线性层 | `DEFAULT_LORA_TARGET_MODULES` = q/k/v/o/gate/up/down_proj，可选 `merger.mlp.0/2` | PASS |
| `r=16, lora_alpha=32` | `LoraConfig(r=16, lora_alpha=32)`（train/smoke 两处一致） | PASS |
| `MultiSamplePadCollator` 动态 Padding | 右侧 padding，`labels` 用 `-100` 屏蔽，vision patch 沿 batch 维拼接；`len==1` 直通不填充 | PASS |
| `--negative-oversample-ratio` | 默认 1.0，仅对 `stratum` 以 `|false` 结尾的负样本层加权，`<=0` 直接报错 | PASS |
| `by_class_table` / `find_optimal_threshold_for_far` | 均已实现并在评估中实跑（见 3.3） | PASS |

> 命名说明：任务书中的 `build_by_class_table` 在实现里对应 `summarize_predictions(...)["by_class_table"]`（行构造器为 `_build_class_table_row`）+ `format_by_class_markdown_table`。无同名独立函数，功能等价。

**未能验证（not_evaluable）**：真实 GPU 上的 `bf16`/`fp16` 分支、显存占用、吞吐（tokens/s、样本/s）——本机无 NVIDIA 设备。

### 3.3 Phase 3/4 — 五类业务门禁与三级级联（真实回放）

复现命令：

```bash
PYTHONPATH=.:backend .venv/bin/python scripts/eval_five_class_gates.py --output-dir /tmp/ktd_phase3_eval
```

**（a）几何门禁（受控扫描，精确控制 IoU / containment）**

| 指标 | 数值 |
|---|---:|
| 车/三轮车同目标双框抑制率（IoU > 0.75） | **100.00%** |
| 同目标双框误抑制率（IoU ≤ 0.75） | **0.00%** |
| 三轮车车斗误报拦截率（containment > 0.88） | **100.00%** |
| 三轮车车斗误拦截率（containment ≤ 0.88） | 2.70%（仅 1 例，见下） |
| containment 规则隔离检出率（person-in-vehicle，IoU 恒 < 0.06） | **100.00%** |

- 扫描点：IoU 0.50→0.99 共 50 点；containment 0.700→0.995 共 60 点。
- 唯一 1 例「误拦截」为 `containment` 精确等于 0.88 的边界点：二进制浮点使实测值略大于 0.88，被 `> 0.88` 严格比较命中。语义上符合「`> 0.88` 才拦截」的规格，但边界不可位级复现。**建议**：若要求 0.88 边界确定性，改用像素整数运算或显式 epsilon，而非裸浮点比较。

**（b）三级级联分流（真实回放：`data/ktd-test-real-monitoring-full-20260924/output/raw_predictions.jsonl`，1416 事件 / 6713 候选）**

| Tier | 数量 | 占比 |
|---|---:|---:|
| Tier 1（≥0.92 自动确认） | 18 | **0.27%** |
| Tier 2（≤0.12 自动拒绝） | 0 | **0.00%** |
| Tier 3（进入 L20 Qwen3-VL 复核） | 6695 | **99.73%** |

| 合规/冲突指标 | 数值 |
|---|---:|
| `blue_illegal_roof` 强制 Tier 3 合规率 | **100.00%（1247/1247）** |
| 几何冲突告警候选 | 36（0.54%） |
| 其中车/三轮车 IoU 冲突 | 29 |
| 其中包含关系告警 | 10 |

**（c）阈值搜索与逐类表**

真实人工金标（`data/ktd-clean-source-full-evaluation-20260924/predictions_scored.jsonl`，n=142，仅覆盖 车/人员）：

| Class | N | TP | FP | TN | FN | F1 | FAR | FRR | Coverage |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 人员 | 56 | 13 | 2 | 26 | 2 | 0.8667 | 0.0513 | 0.1176 | 0.7679 |
| 车 | 86 | 79 | 2 | 4 | 0 | 0.9875 | 0.2857 | 0.0000 | 0.9884 |

`find_optimal_threshold_for_far(records, max_far=0.10)` → 推荐阈值 **0.55**，best F1 = **0.9583**，FAR = **0.0870**。

> 口径说明：`hybrid_confidence` 是「模型对自身决策的置信度」，不是 P(true)。评估脚本对 `false` 决策取 `1−confidence` 做翻转后再做阈值搜索，否则会得到退化结果（阈值 0.99、F1 0.036）。

五类逐类表：**仓库内不存在覆盖五类的人工金标预测文件**，故五类表由脚本内显式标注的合成基准生成（seed=20261023），仅验证指标管线与阈值搜索在五类上可运行，**不代表真实模型精度**：

| Class | N | TP | FP | TN | FN | F1 | FAR |
|---|---:|---:|---:|---:|---:|---:|---:|
| 车辆 | 210 | 148 | 1 | 59 | 2 | 0.9900 | 0.0167 |
| 人员 | 165 | 120 | 0 | 45 | 0 | 1.0000 | 0.0000 |
| 三轮车 | 140 | 88 | 8 | 42 | 2 | 0.9462 | 0.1600 |
| 蓝色违建顶棚 | 125 | 66 | 13 | 42 | 4 | 0.8859 | 0.2364 |
| 道路裂缝 | 120 | 57 | 15 | 45 | 3 | 0.8636 | 0.2500 |

`max_far=0.10` → 推荐阈值 **0.55**，best F1 = **0.9604**，FAR = **0.0815**。

### 3.4 Phase 4 — 部署编排与选卡

| 项 | 事实 | 判定 |
|---|---|---|
| `deploy/gpu-cluster-4t4-2l20.compose.yml` | T4 ×4 固定 `device_ids ["0"]..["3"]`，各 `mem_limit 8g/8g/cpus 4.0`；L20 vLLM 固定 `["4"]`，`mem_limit 32g/memswap 32g/cpus 8.0`；全部带 `labels.owner=agent`、`gpu_tier` | PASS |
| GPU 5 用途 | 注释与编排一致：专用 QLoRA 训练 / 离线验证 / 金丝雀备份 | PASS |
| `deploy/start_vllm_l20.sh` | `--rm`、`--label owner=agent`、`--memory/--memory-swap=32g`、`--cpus=8`、`--ipc=host`、`--dtype bfloat16`、`--gpu-memory-utilization 0.85`、`--max-model-len 8192`、`--enable-prefix-caching`、`--enable-lora --max-lora-rank 32`；启动前打印 `free -g`/`nproc`/`df -h`/`nvidia-smi`；可执行位正常 | PASS |
| `parse_nvidia_smi_csv` | 同时支持 5 列（index,name,free,used,total）与 4 列（free=total−used），列数异常报错 | PASS |
| `select_gpu_device` 空闲 L20 选卡 | 合成 4×T4+2×L20 拓扑验证：`min_free=40960MiB` → 选中 GPU 4（46000 MiB）；`preferred_gpu_arch=L20` → GPU 4；显式 `gpu_index=0` → T4；arch 不匹配、显存不足均正确抛 `RuntimeError` | PASS（逻辑层） |

> 上述选卡结论基于与真机同构的**合成 `nvidia-smi` 输出**，不是真机读数。

---

## 4. 复核发现清单

| 编号 | 级别 | 发现 | 处置 |
|---|---|---|---|
| F1 | 高 | `blue_illegal_roof` 在 `confidence ≤ 0.12` 时被 Tier 2 自动拒绝，与「100% 强制 Tier 3」冲突 | **已修复**（Tier 2 前豁免 + 原因码 + 2 个回归用例） |
| F2 | 高 | `TIER1_AUTO_CONFIRM_THRESHOLD` / `TIER2_AUTO_REJECT_THRESHOLD` / `VLLM_L20_BASE_URL` 只解析进 `Settings`，**从未写入 job options**；`AUTO_ANNOTATION_MODE=adaptive` 也因此无法触发 `_invoke_adapter` 的二次深思考分支 → `production.env.example` 中这些开关实际无效 | **已修复**（`build_auto_annotation_options` 传导 + 显式入参优先 + 3 个回归用例） |
| F3 | 中 | `test_audit_codex_pilot_review._pilot()` 缺 `source_object_key`，2 个用例失败 | **已修复**（补夹具 + 断言，未放宽生产校验） |
| F4 | 低 | `prepare_ktd_clean_source_qwen_dataset.py` 存在 `"math" in globals()` 死分支（`math` 未导入）；CLI 只有 `--bbox-style`，验收命令 `--marker-style` 无法执行 | **已修复**（删死分支 + 增别名） |
| F5 | 中（待决策） | `roi_scale` 不向图像边界收敛，大目标 ROI 视角灰边占比最高 46.4%（roof-01） | **未改**，需产品确认后加 `clamp_roi_to_image` 并重建数据集 |
| F6 | 高（待决策） | 真实检测器置信度下 Tier 1 仅 0.27%、Tier 2 为 0%（最低置信度 0.25 > 0.12），99.73% 候选压向 L20 → L20 成为吞吐瓶颈，T4 侧几乎不卸载 | **未改**，见下方敏感性表，需业务定阈值 |

### F6 阈值敏感性（真实回放，仅调整 `tier1_threshold`）

| tier1_threshold | Tier 1 占比 | Tier 3 占比 |
|---:|---:|---:|
| 0.60 | 54.57% | 45.43% |
| 0.65 | 49.29% | 50.71% |
| 0.70 | 43.33% | 56.67% |
| 0.75 | 36.09% | 63.91% |
| 0.80 | 25.50% | 74.50% |
| 0.85 | 8.33% | 91.67% |
| 0.90 | 0.86% | 99.14% |
| **0.92（当前默认）** | **0.27%** | **99.73%** |
| 0.95 | 0.01% | 99.99% |

结论：0.92 与现有检测器置信度分布严重不匹配。若目标是让 T4 侧承担可观比例的一级裁决，需在 0.75–0.80 区间取值，并用人工金标验证该区间的 FAR 可接受性后再放开 `AUTO_ANNOTATION_CASCADE_ROUTING=true`。

---

## 5. 未完成项 / 需在 GPU 宿主机执行的下一步

### 5.1 not_evaluable（本机无法执行）

| 项 | 原因 |
|---|---|
| 远端 GPU 实机连通与真实选卡 | `ai-lingxi`（192.168.130.238）**ping 100% 丢包、SSH 22 端口超时**；本机网段 192.168.120.0/24，与目标 192.168.130.0/24 不同网段且无路由 |
| L20 `bf16` / T4 `fp16` 实跑、显存与吞吐 | 本机无 NVIDIA 设备、无 `nvidia-smi` |
| QLoRA 训练、vLLM 起服、端到端准确率 | 同上 |
| 五类真实人工金标逐类表 | 仓库内无覆盖五类的人工金标预测文件（现有金标仅 车/人员） |

### 5.2 GPU 宿主机一键命令清单（只读预检 → 训练 → 起服）

```bash
# 0) 网络与凭据（在能访问 192.168.130.0/24 的机器上）
ssh -o ConnectTimeout=10 ai-lingxi 'hostname; nvidia-smi'

# 1) 资源水位 + 选卡（只读，不启动长时任务）
ssh ai-lingxi 'free -g; nproc; df -h /data'
ssh ai-lingxi 'nvidia-smi --query-gpu=index,name,memory.free,memory.used,memory.total --format=csv,noheader,nounits'

# 2) 选卡逻辑复核（宿主机内，PYTHONPATH 指向仓库根）
ssh ai-lingxi 'cd <repo> && PYTHONPATH=.:backend .venv/bin/python -c "
from agent.nodes.remote_job import parse_nvidia_smi_csv, select_gpu_device
import subprocess
out = subprocess.run([\"nvidia-smi\",\"--query-gpu=index,name,memory.free,memory.used,memory.total\",\"--format=csv,noheader,nounits\"], capture_output=True, text=True, check=True).stdout
print(select_gpu_device(parse_nvidia_smi_csv(out), min_free_gpu_mib=40960, hostname=\"ai-lingxi\"))
"'

# 3) Phase 2 冒烟（L20 单卡，先 20 step 确认 bf16 分支与显存）
ssh ai-lingxi 'cd <repo> && PYTHONPATH=.:backend .venv/bin/python scripts/qwen3vl_qlora_smoke.py --help'
# 按 --help 传入 dataset/lora 输出目录后执行；观察日志中 bf16=True, fp16=False

# 4) 正式 QLoRA（GPU 5 空闲 L20）
ssh ai-lingxi 'cd <repo> && PYTHONPATH=.:backend nohup .venv/bin/python scripts/qwen3vl_qlora_train.py --negative-oversample-ratio 1.3 ... > logs/qlora_l20.log 2>&1 &'

# 5) 起 vLLM（GPU 4）
ssh ai-lingxi 'cd <repo> && GPU_DEVICE=4 VLLM_PORT=8009 bash deploy/start_vllm_l20.sh'

# 6) 五类验证 + 阈值搜索（真实金标到位后）
ssh ai-lingxi 'cd <repo> && PYTHONPATH=.:backend .venv/bin/python scripts/qwen3vl_verify.py --target-far 0.10 ...'
```

### 5.3 回滚方式

| 回滚对象 | 命令 |
|---|---|
| 本次全部提交 | `git revert <本次 commit SHA>`（推荐，保留审计轨迹） |
| 仅回滚级联修复 | `git checkout 8d778b5 -- backend/app/services/ktd_recheck.py backend/app/services/auto_annotation_service.py backend/app/config.py backend/app/main.py` |
| 关闭级联（不换代码） | 环境变量保持 `AUTO_ANNOTATION_CASCADE_ROUTING=false`（默认值，行为与修复前完全一致） |
| 恢复旧阈值行为 | `TIER1_AUTO_CONFIRM_THRESHOLD=0.92`、`TIER2_AUTO_REJECT_THRESHOLD=0.12`（默认值） |
| vLLM 容器 | `docker stop vllm-qwen3vl-l20`（启动脚本带 `--rm`，停止即清理） |
| 训练任务 | `pkill -f qwen3vl_qlora_train.py`；产物在 `--output-dir` 指定的 LoRA 目录，删除该目录即回滚权重 |

---

## 6. 红线合规声明

1. 未触碰、未暂存工作区既有未跟踪项：`YOLOE_API_BLOCKING_MQTT_DIAGNOSIS.md`、`deploy-records/`、`复盘日志/`、3 个 `.docx`/`.txt` 报告。
2. 未操作、未重启 8188/8189 服务，未修改 `/app/pretrain` 下任何权重。
3. 本机无 GPU，未创建任何容器；所有临时验证产物均在 `/tmp` 且已清理（`/tmp/ktd_phase1_verify`、`/tmp/test_prod.env`）；保留 `/tmp/ktd_phase3_eval` 作为评估证据，可随时删除。
4. 所有 pytest 均使用 `PYTHONPATH=.:backend .venv/bin/pytest`。
5. 仅暂存本次修改/新增文件，中文 commit message，**未执行 `git push`**。

---

## 7. Git 提交

| 项 | 值 |
|---|---|
| 提交 SHA | `0b9d9c221cd7ef2634aebac54078cafc772ae350` |
| 提交信息 | `fix: 修复四阶段复核缺陷并补齐五类门禁与级联配置传导` |
| 变更规模 | 11 files changed, 964 insertions(+), 8 deletions(-) |
| 是否 push | **否**（按要求仅本地提交） |
| 基线 | `8d778b50c27d43f38b00e08b2e9fa985f5684e74` |
