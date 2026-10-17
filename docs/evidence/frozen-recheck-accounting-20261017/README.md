# 固定分母复检计分证据（2026-10-17）

目标：修复旧评测对缺失/失败先跳过、导致覆盖率分母缩小的问题。仅读取已有输出；无模型调用、业务读写、标签变更或外部写入。

基线为 `feature/independent-vision-workbench@b752d0ef83da874f7c5c05e9f7b95d9f4640c447`。复用 `scripts/score_luna_annotated.py` 的固定标签计数，不改其原有 API。新增工具由聚合报告中的 `implementation_sha256` 定位；原有其他工作区改动未触碰。

## 输入与边界

输入来自主仓 `/Users/wangxiaolin/PycharmProjects/review-annotation-platform`，不复制到本证据目录：

| 角色 | 来源相对主仓路径 | SHA-256 |
|---|---|---|
| source | `data/luna-replace-eval-20261016/tasks.json` | `11f33bbc092e530a5e027c08ea3da20d57843f98183bc1261308e7f087dd33a7` |
| labels | `data/luna-eval-20261008/per-box.json` | `2ebb63045be936a896ab2fea78497ca30b654d1a08d130bc8ccfb2cceedcb9ca` |
| B-8B predictions | `data/luna-replace-eval-20261016/out/B-qwen3-vl_8b-instruct.jsonl` | `5124527767fac1e6f1d8c4794b7844a36ff2e5c2860fba44696d5d87e1130ac4` |
| B-36B predictions | `data/luna-replace-eval-20261016/out/B-qwen3.6_latest.jsonl` | `1eedd01081503e2b9b88bc05a6d40d9c5bf5a22765bf9473e1f1fb8cc14e7377` |
| A-8B predictions | `data/luna-replace-eval-20261016/out/A-qwen3-vl_8b-instruct.jsonl` | `9343d258c786d68cd8f87fee4f5bcb2f4b6edb8a2957610af0c2a4b069b20a72` |

113 事件、212 候选，210 框可计分；人员 110（58 真/52 假），车辆组 100（82 真/18 假），2 个人工看不清标签单列。该旧集不是新独立真值，文件绑定校验不是可信原帧来源认证。输出仅含逐组/逐类聚合和文件摘要，不含事件 ID、候选 ID、原图路径、模型自由文本或凭据。

## TDD 与离线验证

同一独立合成集合共 9 个可计分框，预期正确 3、错误 3、缺失 2、失败 1。直接调用旧 `score_eval.score_file` 得到 `total=6`、`coverage=1`，断言 `total==9` 失败；这是实际旧行为的红证据。

新 CLI 的 13 项测试先红（入口尚未实现，见 `red.xml`），最小适配实现后 13/13 绿（`green.xml`）。覆盖固定总分母、乱序索引、缺失事件/目标、执行失败、uncertain 转人工、误拒/误放、空输出、布尔/重复/越界索引、非法决定、来源/标签几何和候选顺序不符，以及聚合脱敏。无内部函数 mock；均通过真实 CLI 读写合成临时文件。既有 `test_luna_annotated_eval.py` 7/7 回归通过。

在隔离工作树执行：

```sh
PYTHONDONTWRITEBYTECODE=1 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest -p no:cacheprovider backend/tests/test_score_frozen_recheck.py -q
PYTHONDONTWRITEBYTECODE=1 python3 -m unittest discover -s scripts -p test_luna_annotated_eval.py -v
python3 scripts/score_frozen_recheck.py --source /absolute/path/to/tasks.json --labels /absolute/path/to/per-box.json --predictions /absolute/path/to/existing-output.jsonl --out /absolute/path/to/aggregate.json
```

实际使用主仓已有 `.venv/bin/python`，清空继承环境后执行，未加载业务配置。三份 `*-fixed.json` 是上述五份输入的本地确定性复算，未重新请求模型。输入绑定异常报 `INPUT_CONTRACT_ERROR`，不输出业务标识；非法模型响应按失败单列，不能生成 false。

## 结果与限制

| 输出 | 有效决定 / 固定分母 | 缺失 | 人员自动覆盖 | 车辆组自动覆盖 |
|---|---:|---:|---:|---:|
| B-8B | 208/210 | 2 | 108/110=98.18% | 100/100=100% |
| B-36B | 210/210 | 0 | 110/110=100% | 100/100=100% |
| A-8B | 178/210 | 32 | 85/110=77.27% | 93/100=93% |

三份输出均 113/113 事件、0 技术失败。完整真假、uncertain、人工转交、缺失及失败计数和条件准确率见对应 JSON。正确计分不构成模型提升或发布证据；原标签、新航次独立性、来源绑定和探针校准的缺口仍保留。

副作用仅为本地新增计分工具、测试及这些聚合产物；未 commit/push。主代理统一审阅后处理提交。回滚只需撤回本次新增文件和研究报告补充，不涉及共享服务。
