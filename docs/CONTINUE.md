# Continue Here

## 当前定位 Frontier：测试 Java 恢复阻塞（2026-09-09 18:51）

用户已授权测试环境整体修复和复测。目前 8181 稳定性版本已发布，测试 Java 启动退出后 192.168.110.23 不可访问；用户选择“我恢复110.23访问后通知你”。等待该通知，任务未完成，不重复询问实施授权，不运行结束交接。

权威断点是 `docs/reports/afmap-geolocation-20260909/STABILITY_CURRENT_STATE.md`，验收入口是同目录 `ACCEPTANCE_GEO-STABILITY.md`。下方“方案待审阅”与单事件阶段内容仅作历史，不能据此重复执行旧 apply。当前 Java 候选含逐目标 Map 隔离和 TEXT 容量保护，16 项测试及 11 模块 package 通过，尚待在线恢复验证。

等待期间已增强 `verify-stability-new-events.py`：检查 API 和回放的目标 index 完整性，逐目标比对 API、存储与 HTTP 回放 quality。冻结的 4 事件/18 目标离线通过，旧共享 quality、API 质量错配、重复 index 均被拒绝；证据为 `stability-natural-verifier-offline-validation.json`。这不代表新 Java 已上线或自然回调已验收。

恢复访问后先检查实际 Java 包 SHA、进程、18081、活动任务，按 Java 修复工作树 `docs/CONTINUE.md` 恢复/发布；再 apply 尚未写入的 4 事件/8 附件自然质量修正，刷新 `verify-stability-history.py` 快照并运行 `verify-stability-new-events.py`。186 事件历史已写入，不得重复 apply。不得影响 8188/8189/Triton，不制造正式预警或飞行。Git 分支 `codex/uncertain-recheck-training-data`，仅本次文件本地 commit，未 push，保留无关未跟踪内容。

## 历史定位阶段：整体审计与方案待审阅（2026-09-09）

用户已确认 18:10:08 的两个点回到附近，随后要求“整体排查一下，然后给我一个解决方案”。本轮只读完成 3,155 事件/306 任务盘点、同航次 12/12 原坐标精确复现；另 10 条旧事件未更新。发布后新事件可接受 3.488 秒旧遥测（回放差约 87 米），运行 Java 轨迹写入器每 5 秒/100 条刷新，另有近水平交点、框顺序和失败处理风险。

先读 `docs/reports/无人机预警定位整体排查与解决方案-20260909.md` 与 `docs/reports/afmap-geolocation-20260909/ACCEPTANCE_GEO-AUDIT.md`。执行清单在 Java 独立检出的 `docs/GEOLOCATION_IMPLEMENTATION_PLAN.md`，状态 Draft；未实施新增业务代码或批量数据写入。审阅方案是下一步，不重复请求已完成单事件修正的授权。

验证入口：`node docs/reports/afmap-geolocation-20260909/replay-task-originals.mjs --baseline-repo=/Users/wangxiaolin/PycharmProjects/drone-overlay-geolocation-fix/overlay_service`。所有原请求、只读回放及盘点 JSON 均在同一证据目录。提交仅本地，未 push。

## GEO-181008 定位修正（2026-09-09）

本节独立于下方原标注任务。枫香街8号桩 18:10:08 事件的真实请求已恢复，旧版精确复现，修复后已更新两条附件的历史目标坐标；数据库与平台 API/实时回放一致。用户地图视觉已确认，测绘精度未验收。

验收入口：`docs/reports/afmap-geolocation-20260909/ACCEPTANCE_GEO-181008.md`。当前结果：两点距主标识 36.642/33.948 米，原为 346.168/290.722 米。

继续先运行 `node docs/reports/afmap-geolocation-20260909/verify-corrected-event.mjs`，再重新加载平台该事件的目标图层。其他历史事件须分别查原请求，不能重复对本事件执行 `apply`。Java 的 `x/y` 语义需核对实际发送端；Java 独立检出 `/Users/wangxiaolin/IdeaProjects/ktd_backend-geolocation-fix` 当前只有诊断文档，没有业务改动，基线构建依赖未齐。

Git：本仓库分支 `codex/uncertain-recheck-training-data`；仅本次证据与文档本地提交，未 push。定位修复已在独立仓库 `ec5e944` 发布；保留下方其他任务上下文。

## 原标注任务上下文

当前 Frontier：**用类别筛选标空类**。5173 队列已有「没戴头盔」等 class 下拉。完成图 35 张 / 104 框。没戴头盔还剩 10 张单框图。

验收文档：

- `docs/ACCEPTANCE_KTD-CLASS-FILTER.md`
- `docs/ACCEPTANCE_KTD-GOLD-EXPORT-SNAPSHOT.md`（本地金标快照，PARTIAL）
- `docs/ACCEPTANCE_KTD-ADVANCE-NEXT-BOX.md`
- `docs/ACCEPTANCE_KTD-ANNEX-LOCAL-IMPORT.md`（导入，仍 PARTIAL）

## 现场进程（不要杀）

| 谁 | 在干什么 | 不要碰 |
|---|---|---|
| Codex | 曾用 `tmux ktd8189` 跑 8189 预填。当前 tmux 已不在、进程已退出；`attempts.jsonl` 停在 **23/1510**。HEAD 把脚本改名为 `scripts/prefill_ktd_review_service.py`。可用 `--resume` 续跑，不要另开一份同时写同一 SQLite | 预填脚本、`data/ktd-8189-prefill-20260903/`、候选模型字段 |
| Kimi | 下午会话给 518 个存疑框加权打分 | `agent/`、`data/ktd-uncertain-weighted-20260903/` |
| 本机标注 | API `127.0.0.1:8190` + 前端 `127.0.0.1:5173` | Docker `8391/5274` 是另一套数据 |

## 已有能力

1. 本机 Docker MySQL `review-ktd-mysql`：`127.0.0.1:3307` / `ktd_lx` / `ktd_local` / `ktd_local`，**1510** 行 `sys_annex_file`。
2. 图片仍用 `https://lxtest.zhuoyilingxi.com:8088/minio-src` + `file_path`，不拷原图。
3. 本机标注 API：`http://127.0.0.1:8190`（SQLite `data/ktd-local.sqlite3`）。
4. 本机前端：`http://127.0.0.1:5173`，账号 `admin / change-me-admin-password`。
5. SQLite 已 ingest **1510** 样本 / **7104** 候选。人工金标仍很少（约 19 条，以库为准）。
6. 队列默认过滤 `needs_review`（pending + in_progress）。

两条数据不要混：

- 昨夜复检小集：106 图 / 751 框 / 518 存疑 → Kimi 打分 + Codex 冒烟训练
- 全量 annex：1510 图 / 7104 框 → 8189 预填 + 5173 人工标

## 下一步具体动作

1. **人**继续在 `http://127.0.0.1:5173` 看图。快捷键 1/2。优先补空类（光伏板、公交车、没戴头盔）和假目标，不要只标「车」。
2. 再出快照：

```bash
python3 scripts/export_completed_gold.py --download-images --output-dir data/gold-export-20260903
cat data/gold-export-20260903/summary.json
```

3. 不要在本机训练。不要把 Kimi lean_true 或 8189 预填当金标。
4. 可选后续：修本机导出（Celery 入队失败 + HTTP 对象存储只读）。
5. 产品决策仍待人定：7 类 vs 10 类、检测 vs 分割、YOLOE vs Qwen-VL。

禁止：

- 把 `.env.ktd.local` 或源库密码提交进 git
- 把 uncertain 静默改成 true/false
- 为了本机训练去拷 19801 张原图
- 覆盖 Docker 8391 那套正在跑的栈数据
- 在这份混合 working tree 上随便 commit 别人的未提交改动

## 验收命令

```bash
PYTHONPATH=backend python3 -m pytest backend/tests/test_annotation_flow.py -q
cd frontend && node --experimental-strip-types --test src/selection.test.ts
curl -sS http://127.0.0.1:8190/api/v1/health
```

重启本机 API（必须清空 MinIO 环境变量）：

```bash
SOURCE_IMAGE_BASE_URL=https://lxtest.zhuoyilingxi.com:8088/minio-src \
SOURCE_IMAGE_BUCKET=minio-src \
MINIO_ENDPOINT= MINIO_ACCESS_KEY= MINIO_SECRET_KEY= \
DATABASE_URL=sqlite:///./data/ktd-local.sqlite3 \
SERVICE_API_KEY=change-me-service-api-key \
ADMIN_PASSWORD=change-me-admin-password \
PYTHONPATH=backend python3 -m uvicorn app.main:app --host 127.0.0.1 --port 8190
```

## Git state

- branch: `codex/uncertain-recheck-training-data`
- HEAD: `e104c27` `fix: target deployed Kimi review service by default`
- KTD-ADVANCE-NEXT-BOX 未单独 commit、未 push
- 工作区还有 annex 导入、`agent/`、object_store、前端角标等其他人的未提交改动
