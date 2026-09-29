# 车辆支撑面关卡 v3 发布前补证据

## 结论

- 当前不建议发布 v3：屋顶设施只有 10 个，零放过的单侧 95% 上界仍约为 26%；真实流量收益也只是非随机历史前缀上的估算。
- 已验证：holdout 68 个真车中 v3 找回 8 个，60 个未找回；10 个屋顶设施（holdout 6 + A 组 4）仍为 0 个放过，看不清 26 个仍为 0 个确认。
- 已验证：支撑面、四视图和部件三个阶段的解析失败与运行异常均为 0。
- 样本不足无法判断：B/C 没有任何已标注真车进入 building_roof/unknown 分支，因此不能证明新规则能多找回真车。
- 探索性结论，需要新样本验收：只凭车身轮廓放行不是安全候选；C 组 7 个已知假框里有 2 个部件结果会命中。
- v4 只有在 wxl 完成新补标包后才值得继续评估；当前没有达到发布门槛的候选规则。

## 三项部件分布

```json
{
  "真车": {
    "count": 68,
    "combination_counts": {
      "not_run": 8,
      "not_visible/not_visible/not_visible": 15,
      "not_visible/not_visible/visible": 37,
      "not_visible/visible/visible": 8
    },
    "parts_not_run": 8,
    "parts_parse_failures": 0,
    "parse_failures_any_stage": 0,
    "runtime_exceptions_any_stage": 0,
    "unexpected_errors": 0,
    "confirmed": 8
  },
  "屋顶设施": {
    "count": 10,
    "combination_counts": {
      "not_run": 3,
      "not_visible/not_visible/not_visible": 3,
      "not_visible/not_visible/visible": 4
    },
    "parts_not_run": 3,
    "parts_parse_failures": 0,
    "parse_failures_any_stage": 0,
    "runtime_exceptions_any_stage": 0,
    "unexpected_errors": 0,
    "confirmed": 0
  },
  "看不清": {
    "count": 26,
    "combination_counts": {
      "not_run": 15,
      "not_visible/not_visible/not_visible": 6,
      "not_visible/not_visible/visible": 5
    },
    "parts_not_run": 15,
    "parts_parse_failures": 0,
    "parse_failures_any_stage": 0,
    "runtime_exceptions_any_stage": 0,
    "unexpected_errors": 0,
    "confirmed": 0
  }
}
```

60 个未找回真车：四视图未一致确认 8 个；三项都不可见 15 个；只有车身轮廓 37 个。逐框清单见 `missed-real-vehicles.jsonl`。

## B/C 规则探索

现有 B/C 共 205 个框，四视图一致后进入屋顶/未知分支 0 个，且没有可用于衡量找回率的已标注真车。任何候选变体都只能标为“探索性结论，需要新样本验收”。

## 真实流量估算

按 2026-09-24 真实事件快照与 513 框筛选前缀外推，每 1000 个车辆框约 95.7 个进入 building_roof/unknown 分支，v3 约少 10.3 次人工复核，增加约 249 秒总推理时间。该数字受非随机前缀和历史流量结构限制，不是线上验收值。
