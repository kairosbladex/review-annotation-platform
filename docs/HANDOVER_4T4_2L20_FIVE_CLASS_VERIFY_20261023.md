# 4×T4 + 2×L20 异构算力与五类业务目标全链路复核交接报告

- 仓库：`/Users/wangxiaolin/PycharmProjects/review-annotation-platform`
- 分支：`docs/luna-business-test-release`
- 被复核基线：`8d778b50c27d43f38b00e08b2e9fa985f5684e74`（24 个文件，+2570/−201）
- 复核执行环境：macOS / Python 3.13.2 / `.venv`；**无 NVIDIA GPU、无 `nvidia-smi`、无 Docker GPU 运行时**
- 复核日期：2026-10-23
- **第二轮收口日期：2026-10-23**（F5 ROI 灰边、F6 级联阈值，见第 8 章；全程未连接 GPU 服务器、未触碰 8188/8189 线上服务）

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
| `scripts/prepare_ktd_clean_source_qwen_dataset.py` | 修改（第二轮） | **F5 修复**：新增 `_clamp_roi_side_to_image` / `_tick_expansion_px`，ROI 视口收敛到图像短边（必要时长边）；crop meta 增加 `viewport_side` / `padding_ratio` / `roi_side_clamped`；`_save_generated_views` 强制 `context_side >= roi_side` |
| `scripts/prepare_ktd_canonical_crop_dataset.py` | 修改（第二轮） | **F5 同步**：`_square_context` 复用同一 ROI 收敛函数 |
| `scripts/qwen3vl_verify.py` | 修改（第二轮） | **F7 修复**：新增 `_resolve_positive_score`，`false` 决策自动折算 `P(true)=1-confidence`；阈值搜索改用正类得分 |
| `backend/app/services/ktd_recheck.py` | 修改（第二轮） | **F6 修复**：Tier1 白名单收紧到 `vehicle/person/tricycle`，`road_crack` 禁自动确认，默认阈值 0.92→0.85；新增 `车牌` 与 `road_crack` 类别归一 |
| `backend/app/config.py` | 修改（第二轮） | `tier1_auto_confirm_threshold` 默认值 0.92→0.85（`AUTO_ANNOTATION_CASCADE_ROUTING` 仍默认 `false`） |
| `deploy/production.env.example` | 修改（第二轮） | `TIER1_AUTO_CONFIRM_THRESHOLD=0.85` + 白名单注释 |
| `scripts/eval_five_class_gates.py` | 修改（第二轮） | 新增 `cascade_before_after` / `gold_cascade_replay` 两段评估与「改造前后对比」Markdown |

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

第二轮收口后（最终）：

```
197 passed, 1 skipped, 2 warnings in 14.95s
```

`0 failed`。第二轮再新增 12 个用例：5 个 ROI 收敛（`test_prepare_ktd_clean_source_qwen_dataset.py`）、3 个置信度折算（`test_qwen3vl_verify.py`）、4 个级联白名单（`test_ktd_recheck.py`），另对 2 个既有用例补了 `score_is_p_true` 语义标注。

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

**F5 缺陷与修复（第二轮已闭环，详见第 8 章）**：`roi_scale` 只按 bbox 长边放大、不向图像边界收敛，大目标会产生较高灰边占比：

| 样本 | 类别 | annotated_context 填充率 | annotated_roi 填充率（第一轮实测） |
|---|---|---:|---:|
| person-01 | 人员 | 17.9% | 0.0% |
| tricycle-01 | 三轮车 | 17.9% | 0.0% |
| vehicle-01 | 车辆 | 43.2% | 0.0% |
| roof-01 | 蓝色违建顶棚 | 21.9% | **46.4%** |
| crack-01 | 道路裂缝 | 47.5% | 0.0% |

`blue_illegal_roof` 的 context 已由 `clamp_context_to_image=true` 收敛，但 **roi 视角无同等约束**，700×400 的屋顶在 4× 放大后视口 2800px 远超 1920px 画幅。

第二轮已在 `_crop_square_with_metadata` 中对 **所有 ROI 视角**加入同一收敛规则：目标框 + 角标外扩边距仍能落在图像短边内时，视口收敛到短边（零灰边）；否则收敛到长边。修复后 `roof-01`（1920×1080 帧内 700×400）ROI 灰边 **73.41% → 0.00%**（第一轮 46.4% 口径见第 8 章脚注），真实 6678 条候选中被收敛的 238 条平均灰边 **43.60% → 0.95%**。

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

| Tier | 第一轮（≥0.92，任意非 roof 类） | 第二轮（≥0.85，刚体类白名单） |
|---|---:|---:|
| Tier 1 自动确认 | 18（**0.27%**） | 546（**8.13%**） |
| Tier 2 自动拒绝 | 0（**0.00%**） | 0（**0.00%**） |
| Tier 3 进入 L20 Qwen3-VL | 6695（**99.73%**） | 6167（**91.87%**） |

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
| F5 | 中 | `roi_scale` 不向图像边界收敛，大目标 ROI 视角灰边占比最高 46.4%（roof-01） | **已修复（第二轮）**：ROI 视口收敛到图像短边/长边 + 5 个回归用例；`roof-01` 灰边 73.41%→0.00%，见第 8 章 |
| F6 | 高 | 真实检测器置信度下 Tier 1 仅 0.27%、Tier 2 为 0%（最低置信度 0.25 > 0.12），99.73% 候选压向 L20 → L20 成为吞吐瓶颈，T4 侧几乎不卸载 | **已修复（第二轮）**：Tier1 白名单 + 阈值校准至 0.85，卸载 0.27%→8.13% 且金标零新增 FP；`AUTO_ANNOTATION_CASCADE_ROUTING` 仍默认关闭，见第 8 章 |
| F7 | 高 | `qwen3vl_verify.apply_confidence_threshold` 把 `prediction=false, confidence=0.95` 当成 `P(true)=0.95`，直接翻转决策，阈值搜索退化为 0.99 | **已修复（第二轮）**：`_resolve_positive_score` 统一折算 + 3 个回归用例 |

### F6 阈值敏感性（真实回放，仅调整 `tier1_threshold`，已含第二轮 Tier1 白名单）

| tier1_threshold | Tier 1 占比 | Tier 3 占比 |
|---:|---:|---:|
| 0.60 | 50.42% | 49.58% |
| 0.65 | 46.15% | 53.85% |
| 0.70 | 41.29% | 58.71% |
| 0.75 | 34.84% | 65.16% |
| 0.80 | 25.01% | 74.99% |
| **0.85（第二轮默认）** | **8.13%** | **91.87%** |
| 0.90 | 0.82% | 99.18% |
| 0.92（第一轮默认） | 0.27% | 99.73% |
| 0.95 | 0.01% | 99.99% |

结论：0.92 与现有检测器置信度分布严重不匹配。第二轮用 142 条真实人工金标联合校准后取 **0.85**：这是金标上「Tier1 新增 FP = 0」的最低两位小数阈值（金标最高分负样本 0.84997），对应 6713 条真实候选卸载 8.13%。0.80 虽能卸载 25.01%，但金标新增 4 个 FP（端到端 F1 0.9684→0.9485、FAR 0.087→0.174），故不采用。

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
| 恢复旧阈值行为 | `TIER1_AUTO_CONFIRM_THRESHOLD=0.92`（第二轮默认已改为 0.85）、`TIER2_AUTO_REJECT_THRESHOLD=0.12` |
| 恢复旧 Tier1 类别口径 | `git checkout 0b9d9c22 -- backend/app/services/ktd_recheck.py`（会一并回退白名单与 `车牌`/`road_crack` 归一） |
| 回退 ROI 收敛 | `TIER1_AUTO_CONFIRM_THRESHOLD` 无关；改回 `git checkout 0b9d9c22 -- scripts/prepare_ktd_clean_source_qwen_dataset.py scripts/prepare_ktd_canonical_crop_dataset.py`（会重新引入大目标灰边） |
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
| 第一轮提交 SHA | `0b9d9c221cd7ef2634aebac54078cafc772ae350` |
| 第一轮提交信息 | `fix: 修复四阶段复核缺陷并补齐五类门禁与级联配置传导` |
| 第一轮变更规模 | 11 files changed, 964 insertions(+), 8 deletions(-) |
| 第二轮提交 SHA | `960b453faedc163271a71c63c08841b08f5813f1` |
| 第二轮提交信息 | `fix: 修复大目标ROI灰边与级联阈值退化并量化卸载收益` |
| 第二轮变更规模 | 11 files changed, 851 insertions(+), 59 deletions(-) |
| 是否 push | **否**（两轮均仅本地提交） |
| 基线 | `8d778b50c27d43f38b00e08b2e9fa985f5684e74` |

---

## 8. 第二轮收口：F5 / F6 / F7 修复与可量化收益（2026-10-23）

### 8.0 执行边界（红线）

1. 全程**未连接、未重启、未修改任何 GPU 服务器**，未触碰 8188/8189 线上服务。
2. 所有验证仅基于仓库内真实离线数据：`data/ktd-clean-source-full-evaluation-20260924`（142 条人工金标）与 `data/ktd-test-real-monitoring-full-20260924`（1416 事件 / 6713 候选）。
3. 未触碰、未暂存工作区原有 6 项未跟踪文件/目录。
4. 仅本地中文 commit，**未执行 `git push`**。

### 8.1 改造前后真实收益对比表

| 维度 | 改造前 | 改造后 | 判定 |
|---|---|---|---|
| 大框 `roof-01`（1920×1080 帧内 700×400）`annotated_roi` 灰边占比 | **73.41%**（视口 2800px） | **0.00%**（视口 1080px） | 达标（要求 ≤15%） |
| 真实候选 ROI 视口灰边（238 条被收敛样本，均值） | **43.60%** | **0.95%**（229 条归零） | 显著改善 |
| `qwen3vl_verify` 原始记录直传阈值搜索 F1 | **0.0360**（阈值退化 0.99） | **0.9583**（阈值 0.55） | 修复 |
| 6713 条真实候选 VLM 卸载比例（Tier1+Tier2） | **0.27%**（18 条） | **8.13%**（546 条） | **30×** |
| 142 条金标 Tier1 新增 FP | 0 | **0** | 零精度损失 |
| 142 条金标端到端 F1 / FAR | 0.9684 / 0.0870 | **0.9684 / 0.0870** | 完全不变 |

> F1 口径：`find_optimal_threshold_for_far(max_far=0.10)` 在 142 条金标（`hybrid_decision`/`hybrid_confidence` 原始记录）上的 best F1。改造前把 `confidence` 直接当 `P(true)`，高置信 `false` 全被翻成 `true`，任何阈值下 FAR 都 ≥0.28，无法满足 `max_far=0.10`，于是退化到兜底阈值 0.99（F1=0.0360）。改造后无需任何外部脚本手工翻转即可得到 0.55 / F1=0.9583 / FAR=0.0870。

### 8.2 F5：大尺寸目标 ROI 无效灰边

**根因**：`_crop_square_with_metadata` 中 context 视角有 `clamp_context_to_image` 约束，ROI 视角没有。`roi_scale=4.0` 直接把 700×400 的屋顶放大到 2800px 视口，超出 1920×1080 画幅后由 `cv2.copyMakeBorder` 补中性灰。

**修复**（`scripts/prepare_ktd_clean_source_qwen_dataset.py`，`scripts/prepare_ktd_canonical_crop_dataset.py` 同步）：

- 新增 `_tick_expansion_px()`：角标 L 型臂是**向内**绘制的，只有 2px 外扩 + 描边会落在框外，因此外扩边距恒为 8px（不再按臂长 6–24px 预留，否则大目标会被迫退到长边）。
- 新增 `_clamp_roi_side_to_image()`：目标框 + 8px 边距仍能落在图像短边内 → 视口收敛到**短边**（零灰边）；否则收敛到**长边**（仍完整包含目标）。
- 小目标不受影响：`desired_side <= short_side` 时原样返回，`person` 100×100 仍保持 5× = 500px 视口。
- 退化图像（短边 < `min_side`）保留原有补边路径，不破坏既有行为。
- crop meta 新增 `viewport_side` / `padding_ratio` / `desired_side` / `roi_side_clamped`，`_save_generated_views` 强制 `context_side >= roi_side`（不满足直接 `ValueError`）。

**实测（`roof-01`：700×400 in 1920×1080，`blue_illegal_roof`，`annotated_roi`）**：

| 项 | 改造前 | 改造后 |
|---|---|---|
| 视口边长 | 2800px | 1080px |
| viewport | `[-440, -860, 2360, 1940]` | `[420, 0, 1500, 1080]` |
| padding | `[860, 860, 440, 440]` | `[0, 0, 0, 0]` |
| 灰边占比（逐像素 `==(230,230,230)`） | **73.41%** | **0.00%** |
| 目标框在 512px 输出中的位置 | 被灰边包围 | `[90.07, 161.19, 421.93, 350.81]`，完整且四角留白 ≥90px |
| `context_side >= roi_side` | 1920 ≥ 2800 ❌ | 1920 ≥ 1080 ✅ |

> 口径脚注：第一轮报告的 roof-01 灰边 46.4% 来自 `/tmp/ktd_phase1_verify`（已删除）的另一画幅夹具；本轮用可复现代码在 1920×1080 上重建同一缺陷，实测 73.41%。两者是同一缺陷的不同画幅表现，修复后均为 0%。真实 6678 条候选快照中被收敛的 238 条（3.56%）平均灰边 43.60% → 0.95%，其中 229 条完全归零，9 条因目标本身宽于短边而保留 ≤25% 灰边（正方形视口的几何下限）。

### 8.3 F6：三级级联阈值校准

**安全边界强化**（`backend/app/services/ktd_recheck.py`）：

| 类别 | Tier 1 自动确认 | Tier 2 自动拒识 | Tier 3 VLM |
|---|---|---|---|
| `vehicle` / `person` / `tricycle`（车/人员/三轮车） | ✅ 允许（≥阈值且无几何冲突） | ✅ ≤0.12 | 其余 |
| `blue_illegal_roof`（蓝色违建顶棚） | ❌ 禁止 | ❌ 禁止 | **100% 强制** |
| `road_crack`（道路裂缝/裂纹） | ❌ 禁止 | ✅ ≤0.12 | 其余（含所有高分） |
| 其它（光斑/光伏板/车牌/没戴头盔…） | ❌ 禁止 | ✅ ≤0.12 | 其余 |

- 新增 `TIER1_ELIGIBLE_CLASSES = {vehicle, person, tricycle}` 白名单与 `_ROAD_CRACK_ALIASES`、`_PLATE_ALIASES`；修正 `车牌`（含「车」字）此前会被归一成 `vehicle` 的缺陷。
- 新增原因码：`tier3_road_crack_never_auto_confirmed` / `tier3_class_outside_tier1_allowlist` / `tier3_geometric_conflict`。

**阈值校准**（142 条金标 × 真实检测器分数，通过 `gold_overlap_map.jsonl` 关联监控快照的 `detector_confidence`）：

| Tier1 阈值 | 金标自动确认数 | 新增 FP | 端到端 F1 | 端到端 FAR |
|---:|---:|---:|---:|---:|
| 0.75 | 68 | 5 | 0.9436 | 0.1957 |
| 0.80 | 64 | 4 | 0.9485 | 0.1739 |
| 0.82 | 55 | 3 | 0.9534 | 0.1522 |
| **0.85** | **36** | **0** | **0.9684** | **0.0870** |
| 0.92 | 0 | 0 | 0.9684 | 0.0870 |

金标中最高分的负样本为 **0.84997**，因此 0.85 是「零新增 FP」的最低两位小数阈值；再往下每降一档都开始出现金标 FP 且 F1 单调下降。**采用 0.85**，并保持 `AUTO_ANNOTATION_CASCADE_ROUTING=false` 默认关闭（开启即生效）。

**白名单护栏价值**：在 0.85 阈值下，若无白名单会有 **573 条**非刚体类高分候选被自动确认（违建 560 条、光伏板 13 条）。

### 8.4 F7：`qwen3vl_verify` 正类置信度折算

- 新增 `_resolve_positive_score(row)`：优先 `positive_score`/`p_true`；其次 `score`/`confidence`，当原始决策为 `false`/`rejected`/`no`/`0` 且未标 `score_is_p_true=True` 时折算为 `1-value`。
- `apply_confidence_threshold` 与 `find_optimal_threshold_for_far` 统一走该函数，并把折算结果写回 `positive_score` 便于审计。
- `qwen3vl_verify.main()` 中 `score` 是检测器 P(true)，显式标记 `score_is_p_true=True`，避免被二次翻转。

### 8.5 复现命令

```bash
# 全量测试（0 failed）
PYTHONPATH=.:backend .venv/bin/pytest backend/tests agent/tests -q

# 级联 / 金标 / 灰边的可量化对比（输出 JSON + Markdown）
PYTHONPATH=.:backend .venv/bin/python scripts/eval_five_class_gates.py \
    --output-dir /tmp/ktd_phase4_eval

# ROI 灰边回归（含 roof-01 与真实 960x720 帧两个用例）
PYTHONPATH=.:backend .venv/bin/pytest backend/tests/test_prepare_ktd_clean_source_qwen_dataset.py -q
```

### 8.6 残余风险与后续

| 项 | 说明 | 建议 |
|---|---|---|
| 0.85 的统计置信度 | 金标仅在 36 个样本上验证零 FP，按「rule of three」其 95% 上界约 8% | 线上开启 cascade 前，用 2–3 周真实告警抽样复核该区间 |
| Tier 2 仍为 0% | 真实检测器最低置信度 0.25 > 0.12，当前 `tier2_auto_reject` 完全不触发 | 若要释放 Tier 2 卸载，需重新校准 0.12 或用低分候选单独验证 |
| 9 条宽目标仍有 ≤25% 灰边 | 目标本身宽于图像短边，正方形视口无法避免 | 若需彻底消除，二期可改为非正方形视口或按目标长宽比自适应 |
| 几何门禁不直接裁决 | 当前只把冲突候选升级到 Tier3，直接拦截占比 0% | 若需进一步卸载，可对「同目标双框」的重复框做自动抑制（需人工金标验证） |
| 数据集需重建 | 灰边修复只影响新建数据集，已产出的 `ktd-clean-source-qwen-20260930-v2` 仍是旧视口 | 下次训练前用同一命令重建并比对 `padding_ratio` |
