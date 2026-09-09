# Continue Here

## GEO-181008 定位修正（2026-09-09）

本节独立于下方原标注任务。枫香街8号桩 18:10:08 事件的真实请求已恢复，旧版精确复现，修复后已更新两条附件的历史目标坐标；数据库与平台 API/实时回放一致。地图视觉和测绘精度未验收。

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
