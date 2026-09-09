# Continue Here

## 当前定位 Frontier：测试环境稳定性修复已验收（2026-09-09 19:16）

当前 Frontier：测试环境稳定性修复、发布及现有事件复测完成，待用户交接。2026-09-09 19:16。之前的“等待 110.23 恢复”阻塞已解除，不要重新发布或重复 apply。

权威断点：/Users/wangxiaolin/PycharmProjects/review-annotation-platform/docs/reports/afmap-geolocation-20260909/STABILITY_CURRENT_STATE.md。本仓库验收入口：docs/ACCEPTANCE_GEO-STABILITY.md。

8181 已发布 geo-stability-20260909-v2，108 测试、182 次 HTTP 验证通过；Java 已发布逐目标 Map 隔离与 TEXT 容量保护，16 项测试和 11 模块 package 通过。Java 活动 SHA256 dfd679d2aa48bbbedf3d68c1ccd48eec7c88eb3e62c1882651c0d70ff151fdcf，最后 PID 4884 / test / 18081。本机直连和平台 API 均已核验；两级原包保留。发布 PID 4466 曾关闭，另一会话随后启动 4884；没有继续重启覆盖该进程，关闭触发方未核实。最终证据在 Java docs/stability-capacity-final-runtime.json。

历史 186 事件/372 附件已修正、数据库和 API 验证通过，931 目标可绘制、77 不可用；6408 快照中其他 6036 行不变。自然质量小批次 4 事件/8 附件已 apply/verify，坐标不变。两个批次均不得重复 prepare/apply；不要用初次未压缩的失败备份重试。

报告仓库 verify-stability-history.py 已刷新到最终 Java SHA；verify-stability-new-events.py 比对 index、API/存储/回放逐目标 quality 和坐标，通过已有 5 条自然事件（18 点可绘制、另 1 事件时间差 1.017 秒保留但不撒点）。这些自然事件均早于最终 Java 发布，最新 Java 自然回调还没有业务样本；不制造正式预警或飞行来验证。

后续真实新事件验收：只读核对活动包，运行报告仓库 verify-stability-history.py --config-jar <私有配置jar> --java-jar-sha256 <真实活动SHA>（凭据经私有 stdin），刷新快照后运行 verify-stability-new-events.py。新失败类型或部分定位需按真实请求分析，不能修改脚本掩盖失败。约 20 米真实残差与斜拍精度仍无真值验收；3030 条缺原始输入的历史在报告待办 #8。

Git：实现和证据仅本地 commit，不 push；原 /Users/wangxiaolin/IdeaProjects/ktd_backend 为干净 prod。保留无关 dirty/untracked。禁止更改 8188、8189、Triton；19:15 容器、启动时间和源码 SHA 复核未变。不要打印或提交配置 jar/密码。不要使用子代理或修改 memory。当前任务内没有新的实施授权问题，交接用 AskHuman；尚未取得结束批准。

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
