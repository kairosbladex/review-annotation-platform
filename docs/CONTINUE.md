# Continue Here

> 2026-10-17：三功能独立工作台在 `feature/independent-vision-workbench` 继续，任务与门禁见 [独立工作台验收](ACCEPTANCE_VISION-WORKBENCH.md)，部署边界见 [独立测试发布计划](independent-test-deployment-plan-20261017.md)。本条只索引本轮任务；下方历史上下文保留。

> 2026-09-29：无人机预警空间定位（AFMAP 撒点偏移、AR 投影、光伏组件定位）已拆到独立仓库 `/Users/wangxiaolin/PycharmProjects/skypin`，相关文档、脚本、证据数据和 10 个分支都在那里继续；本仓库只做视觉复检、标注与训练。定位的原交接内容见 skypin 的 `docs/CONTINUE.md`。

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
