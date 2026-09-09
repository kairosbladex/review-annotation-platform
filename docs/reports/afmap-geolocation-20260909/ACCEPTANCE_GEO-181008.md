# GEO-181008 单事件修正验收

日期：2026-09-09。状态：原请求复现、单事件更新及接口验证通过；登录后地图和测绘精度待验收。

## 做了什么

- 从 GIS 日志恢复枫香街8号桩 18:10:08 的真实定位请求及响应，以旧日志前缀精确复现两点旧经纬度，最大差值 0；定位遥测落后 69.985 秒。
- 使用已修复的 8181 服务复算，选中 18:10:08.010 遥测。备份后在事务中只修改该事件两条附件的 `ar_geo_coordinates`，保存旧值、原框、请求及计算版本。
- 增加只读数据库核验模式、API/实时原请求一致性验证脚本。数据库 2 行及两个接口结果一致，主坐标不变；距主标识从 346.168/290.722 米变成 36.642/33.948 米。
- 更新当前报告，记录 Java 入口、落库复用机制和协议不一致；未将未确认的锚点语义直接做成代码补丁。

## 没做什么

没有修改其他事件、Java 业务源码、相机标定参数或原始检测框，没有登录后视觉验收或地面真值验证。本次数据更新不是将目标设置为飞机坐标，也不是承诺米级精度。

## 哪里可能有问题

- 历史 worker 缓存内存未取回，使用归档日志前缀重建；原始 HTTP 报文已找到，重建结果精确吻合。
- `x/y` 中心点和左上角定义在 Java 两处不一致，发送端协议未确认。有效变焦、图像变换、高度和 CRS 仍需核对。
- Java 全量基线构建因 Heisha SDK 和 GeoTools 依赖解析失败。没有 Java 可部署产物。
- 另外两条样本及更大历史范围未更新。不要覆盖 `live-c5354403ae3b44edb35b2dfc01ede1a4.json` 的更新前快照。

## 如何验收

在本仓库执行：

```bash
node docs/reports/afmap-geolocation-20260909/verify-corrected-event.mjs
node docs/reports/afmap-geolocation-20260909/replay-original-event.mjs --baseline-repo=/Users/wangxiaolin/PycharmProjects/drone-overlay-geolocation-fix/overlay_service
```

已通过：`post-correction-181008.json`、`database-postcheck-181008.json`、`original-event-replay.json`。备份文件 `event-correction-181008.json` SHA-256：`3e0b2a48300ddd1181bf0b3e435a9b930f72be0db2a555c2a379949e3aaccb50`。

数据脚本 `correct-event-181008.py` 支持 `verify`（只读）、`rollback`；通过标准输入提供已有授权的 SSH 密码，`--config-jar` 指向受控的运行配置 jar，`--backup` 指向本目录备份。密码和配置 jar 不进入 Git。`apply` 已执行，不能重复执行。回退前会检查原框和当前坐标；检查不符将回滚事务。

重新选择概览中的该事件并重新加载目标图层，可核验地图呈现。单靠本脚本通过不能代替该步骤或测绘精度验收。

## Git state

分支 `codex/uncertain-recheck-training-data`，本轮基线 `1a4eb02`，仅提交本次报告、脚本和脱敏证据，未 push。定位服务修复在独立仓库 `ec5e944`，Java 文档在独立 `origin/test` 检出。其他未跟踪数据、模型及 IDE 文件均保留。

本次报告提交号使用 `git log -1 --oneline -- docs/reports/afmap-geolocation-20260909/ACCEPTANCE_GEO-181008.md` 查询；继续入口是 `docs/CONTINUE.md` 的 GEO-181008 小节。
