# 定位稳定性修复验收

测试环境修复、发布和现有事件复测完成；最后核验于 2026-09-09 19:16。权威继续入口为 STABILITY_CURRENT_STATE.md，旧文档中的未恢复状态为历史快照。

## 修复及发布

- 8181 使用连续垂直姿态矩阵、显式镜像、真实角度扰动和正反向闭合检查。原来 0.001 度微扰导致约 69 米跳变，修复后约 0.002 米；这表示数值稳定性，不代表毫米级地理精度。同航次 12 事件、77 点基准坐标保持不变。
- Java 修复 JSONObject(Map) 共享状态导致的逐目标 quality/status 串扰；TEXT 超过 65535 字节时保存紧凑不可用状态，避免定位元数据溢出中断预警保存。16 项定向测试、11 模块 package 通过；Python 108 测试、182 次实际 HTTP 验证通过。
- 最终 Java 包 SHA256：dfd679d2aa48bbbedf3d68c1ccd48eec7c88eb3e62c1882651c0d70ff151fdcf，仅变更 GeoFromBboxesUtil.class，其余 ZIP entry 内容不变。19:16 PID 4884 / test / 18081 正常，本机直连 Java 与平台接口通过。
- 19:15 再次核对 8181 活动源码、容器 ID 和启动时间；8188、8189、Triton 均未改变。证据在 Python 修复仓库 docs/stability-final-runtime-verification.json。

## 数据和接口复测

- 186 历史事件/372 附件已 apply/verify，931 目标可绘制、77 不可用；6408 快照中的其余 6036 条逐字不变。19:15 全部 186 事件的地图接口复测通过。
- 4 条已有自然事件/8 附件的逐目标质量修正已 apply/verify，坐标未移动；冻结备份及快照已上传服务器并校验 SHA。
- 19:16 实际 HTTP 回放核对 5 条已有自然事件：18 个可绘制目标的坐标、各自 quality、API/存储映射和 index 完整性均通过；另一事件姿态时差 1.017 秒，重试后仍保留事件、不提供目标坐标。
- 离线反例证明验收器能拒绝旧共享 quality、API quality 错配和重复 index，修正的 4 事件/18 目标通过。

具体证据：stability-history-api-verification.json、stability-natural-state-apply-receipt.json、stability-natural-state-verify-receipt.json、stability-natural-events-verification.json、stability-natural-verifier-offline-validation.json。

## 运行经过与限制

110.23 曾失联，19:05 已恢复。容量发布进程 PID 4466 在完成启动后曾于 19:11:19 返回正常 API，19:11:31 正常关闭；另一登录会话随后启动 PID 4884，仍为同一补丁。验收继续使用该进程，没有重复重启覆盖。触发关闭的人或机制未核实，不能据此认定补丁崩溃；最终证据在 Java 修复仓库 docs/stability-capacity-final-runtime.json，原瞬时健康记录仍作为历史保留。

最新 Java 发布后尚无新的自然预警写入；当前 5 条自然业务样本发生在发布之前，修正和回放不能替代新回调验收。没有为了测试而制造正式预警或飞行。

真实约 20 米残差仍缺地面真值，未证明消除；斜拍旧模型因云台角度参考系未核验而拒绝可靠落点。screened 仅代表通过工程检查，accuracy_verified 保持 false。本轮没有地图浏览器视觉验收，3030 条缺原输入历史仍按用户决定延期。

## 如何复核和继续

真实新事件到来后，先核对活动运行包，使用私有配置 jar 和 stdin 凭据刷新 verify-stability-history.py，再执行：

```bash
python3 docs/reports/afmap-geolocation-20260909/verify-stability-new-events.py
```

检查逐目标原请求、index、quality、时间差和失败事件保留。出现新失败类型或部分定位时应按真实请求排查，不能放宽断言掩盖错误。已有两批 apply 均已完成，禁止重复执行；原始备份、快照、回退包全部保留。部署前仍应核对实际包与活动任务缓存，不能按旧 PID 或早期状态操作。

## Git state

报告分支 codex/uncertain-recheck-training-data，验收器修复提交 c0a3dd5；Java 实现位于独立分支 codex/geolocation-event-fix，容量保护提交 54ec6099dc；Python 分支 codex/geolocation-telemetry-cache，实现提交 f5f4e3a。所有实现和证据仅本地提交，未 push。原 ktd_backend 保持干净 prod，无关 IDE、数据、模型和 pyc 保留。凭据、私有配置 jar 未提交。
