# 定位稳定性验收与恢复断点

状态：未完成。2026-09-09 18:51，等待用户恢复 192.168.110.23 访问后继续。当前状态以 STABILITY_CURRENT_STATE.md 为准，早期 PRECISION/AUDIT 文档中的线上状态是历史快照。

## 已实施和验证

8181 已发布 geo-stability-20260909-v2。108 项 Python 测试、182 次真实 HTTP 验证通过。垂直姿态 0.001 度微扰造成的约 69 米跳变已被修复；同航次 12 事件坐标保持不变。186 事件/372 附件历史修正已提交并核验，其他 6036 条快照记录逐字不变，禁止重复 apply。

Java 已修复逐目标 JSONObject 包装共享 Map 的状态串扰，并补充 65535 字节 TEXT 容量保护。16 项定向测试和 11 模块 package 通过。最新候选 SHA256 为 dfd679d2aa48bbbedf3d68c1ccd48eec7c88eb3e62c1882651c0d70ff151fdcf，尚未发布。服务器最后确认包为 15fde7d9606bf271a191946e7a12e58621ac5965b69eada6df551159c0b6eb25；启动进程退出，后续 SSH 及 18081 不可达，原因尚未确认。

本报告仓库增强了 verify-stability-new-events.py：API 目标编号必须完整，API quality 必须与数据库相同，数据库逐目标 quality 必须与同编号的 HTTP 回放结果相同；保留坐标、版本、质量门限与事件保留检查。历史修正合法状态 time_corrected 与正常新写入 screened 均可接受。

离线验证使用已冻结的 stability-natural-state-correction.json，4 事件/18 目标修正候选通过；旧共享 quality、API 质量错误和重复回放 index 三种反例均被拒绝。输出为 stability-natural-verifier-offline-validation.json。没有覆盖已有在线验收证据。

## 未完成和限制

- 测试 Java 服务恢复、最新候选发布、真实 API 健康和新 Java 自然回调验证仍未完成。
- 4 事件/8 附件质量修正已经 prepare，尚未 apply，坐标不变。冻结备份 SHA256 为 1e36ac8728035760dfc8811b5fa38c5336d886b3da0340b920d3d1082d72b1d7。
- 离线验证使用模拟 HTTP，只证明验收器可识别已知错误，不能证明线上修复完成。
- 用户尚未提供地面真值，不能声称约 20 米残差消除。斜拍旧模型仍被拒绝可靠落点输出，地图视觉尚未验收。
- 当前自然事件验证针对已有的正常事件和姿态时间不匹配事件；若恢复后出现其他失败类型或部分定位事件，应按其真实请求核验，不能忽略报错或直接改成通过。

## 恢复后验收

先读 Java 工作树 `/Users/wangxiaolin/IdeaProjects/ktd_backend-geolocation-fix/docs/CONTINUE.md`，核对用户可能自行恢复过的实际包、进程和飞行缓存，确保真实 API 健康后，再处理尚未写入的小批次。186 事件批次不得重复执行。

刷新 verify-stability-history.py 输出时传入真实活动 Java 包 SHA 和私有配置 jar，并从私有 stdin 传凭据。随后执行：

```bash
python3 docs/reports/afmap-geolocation-20260909/verify-stability-new-events.py
```

离线验证临时复跑入口为 `/tmp/geo-natural-verifier-check.py`，结果已落盘；临时文件不属于长期构建依赖。

## Git state

分支 codex/uncertain-recheck-training-data。当前切片仅修改验收脚本、验收/继续文档和离线验证证据，本地提交，不 push。未修改无关 IDE、数据、模型、既有冻结修正文件或线上服务。Java、Python 实现分别位于独立工作树。
