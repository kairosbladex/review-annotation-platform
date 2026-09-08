# ACCEPTANCE_KTD-CLASS-FILTER

## Status

`COMPLETED` — 5173 队列可按 `class_name` 筛选。选「没戴头盔」后显示 10 条待复核。

## Ticket

`KTD-CLASS-FILTER`

## 做了什么

后端 `/api/v1/samples?class_name=` 本来就有。前端接上第二个下拉，默认「全部类别」。标完一张后找下一张时沿用当前类别过滤。

Changed files: `frontend/src/api.ts`、`App.tsx`、`components/QueuePanel.tsx`、`backend/tests/test_annotation_flow.py`

## 没做什么

- 没有按「只显示该类未标框」裁剪画布上的其它类框（图上其它类仍可见）。
- 没有 commit。

## 哪里可能有问题

- 一张图若同时有「车」和「没戴头盔」，筛没戴头盔仍会整图进来，需要把其它框也标完才算出完成图。
- 类别列表写死 10 类。

## 如何验收

5173 登录 → 类别选「没戴头盔」→ 页脚 `1-10 / 10`。Playwright 已验证。测试：`test_sample_queue_can_filter_by_class_name` passed。

## Git state

- branch: `codex/uncertain-recheck-training-data`
- HEAD: `e104c27`
- 未 commit、未 push
