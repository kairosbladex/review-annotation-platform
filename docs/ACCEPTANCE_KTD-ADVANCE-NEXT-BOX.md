# ACCEPTANCE_KTD-ADVANCE-NEXT-BOX

## Status

`COMPLETED` — 5173 标注台：标完一个框后选中同一张图的下一个未标框；本图标完后打开下一张仍有 pending 的图。浏览器已点过真目标验证。

## Ticket

`KTD-ADVANCE-NEXT-BOX`

## 做了什么

根因：默认队列过滤是 `pending`。标完第一框后样本变成 `in_progress`，被踢出队列，`refreshQueue()` 把选中样本打回当前页第一张，看起来像「不跳下一框 / 直接翻图」。

Delivered behavior:

- 新过滤 `needs_review` = `pending` + `in_progress`，工作台默认用它。
- `advanceAfterLabel`：同图还有未标框 → 留在该图并选下一框；本图无未标框 → 打开下一张 `counts.pending > 0` 的样本。
- `keepQueueSelection`：保存标注时即使当前图暂时不在过滤结果里也不抢选中项。

Changed files:

- `backend/app/main.py`
- `backend/tests/test_annotation_flow.py`
- `frontend/src/selection.ts`
- `frontend/src/selection.test.ts`
- `frontend/src/App.tsx`
- `frontend/src/components/QueuePanel.tsx`
- `frontend/tsconfig.app.json`
- `scripts/verify_advance_next_box.py`
- `docs/verify-ktd-advance-next-box/*.png`
- `docs/CONTINUE.md`

## 没做什么

- 没有动 Kimi 的 `agent/` 或 `data/ktd-uncertain-weighted-20260903/`。
- 没有主动停 Codex 预填。本切片结束时 `tmux ktd8189` 已不在（约 15:33 停在 23/1510，与 Codex 15:34 重命名预填脚本同一时段）；Grok 只重启过本机 `8190`。
- 没有改 8188/8189 线上服务或权重。
- 没有把模型打分写成人工金标。
- 没有 commit / push（工作区里还有其他 Agent 的未提交改动）。

## 哪里可能有问题

- 默认过滤从「未开始」改成「待复核」。只想看从未标过的图时，要手动选「未开始」。
- 本图标完后，下一张取当前过滤结果第 0 页里下一条 pending，不是跨全部 1510 张的绝对顺序。
- 本机 SQLite 同时被 8189 预填写入；短暂 `database is locked` 可能导致单次保存失败，再点一次即可。
- 验证时用真目标点了 3 个框，随后已 PATCH `label=null` 撤回；若撤回失败会多 3 条金标，当前库 counts 已回到 pending。
- 未做移动端视口验证（标注台是桌面工作台）。

## 如何验收

```bash
cd /Users/wangxiaolin/PycharmProjects/review-annotation-platform
PYTHONPATH=backend python3 -m pytest backend/tests/test_annotation_flow.py -q
# 期望含 test_needs_review_queue_keeps_in_progress_sample，全部 passed

cd frontend && node --experimental-strip-types --test src/selection.test.ts
# 期望 10 passed

# 浏览器：http://127.0.0.1:5173  admin / change-me-admin-password
# 选一张 ≥2 个框的图，点「真目标」或按 1：应停在同一张图，右侧变成 2/N
# 把该图标完：应打开下一张待复核，提示「本图已标完，已打开下一张待复核」
```

本次已执行证据：

- `pytest backend/tests/test_annotation_flow.py` → **7 passed**
- `pytest backend/tests`（排除 GPU dataset prep）→ **66 passed**
- `node --experimental-strip-types --test frontend/src/selection.test.ts` → **10 passed**
- `npx tsc -b`（frontend）→ 通过
- Playwright Chrome 无头：`1/7` → `2/7` → `3/7` 停在 `da1d1fde…`；再标一张 1 框图 `50d68df7…` 后跳回下一张待复核
- 截图：`docs/verify-ktd-advance-next-box/02-after-first-true.png`、`03-after-second-true.png`、`04-after-last-box.png`
- 验证后已撤销这 3 条标注，两张图 pending 恢复

## Standards review

- 人工结论仍只写 Annotation，不改 Candidate 模型字段。
- 未把 uncertain 默认改成 true/false。
- 只改本工单需要的文件；未覆盖 Kimi/Codex 正在写的目录。
- 未提交密钥。

## Spec review

- 满足「标完跳同图下一未标框；本图标完再翻下一张 pending」。
- 不满足人工金标规模或正式训练；那些仍要人看图。

## Git state

- branch: `codex/uncertain-recheck-training-data`
- HEAD: `e104c27`（Codex 在本切片期间提交了预填脚本重命名，与本工单无文件冲突）
- 本切片未 commit、未 push
