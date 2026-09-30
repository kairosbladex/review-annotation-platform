# 本机GPT复检闭环证据

完整结论、授权原文、配置、回滚、缺口和操作见 ../../docs/recheck-online-loop.md。

- preflight.json：两个请求别名和实际模型、usage账号/余额当时快照。
- local-acceptance/acceptance.json：真实独立API/Celery/Redis的故障注入验收；不证明供应商扣费。
- live-chain/result.json：真实中转站的一次调用；GPT建议不写人工标签。
- baseline-development、baseline-locked：分别24、64框的分层限量基线，非全量冻结集。
- baseline-summary.json：按唯一车框汇总，避免重复population相加；真判假3/28，不通过保护门槛。
- scene-coverage.json：实际看过的场景及缺少样本的边界。
- tests.json：195项范围回归及6项传输检查；旧失败和未验范围另列。
- resources.json：本次资源归还和保留清单。

图片与截图只留本机，不推送Git；live-chain/replay.sqlite3为本机独立真实回放库，不推送Git。本轮冻结使用日志在 ../recheck-gold-consolidated-20260930/test_usage.jsonl。原金标不改，未重冻、未修改共享环境。
