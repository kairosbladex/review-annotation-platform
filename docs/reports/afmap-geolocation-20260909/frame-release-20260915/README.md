# 测试环境部署验收

2026-09-15，用户授权检查条件并部署到测试环境供测试人员验收。结论：满足受控接口联调和平台回归条件，Python/Java均已部署，线上回放通过；不满足“所有事件统一降低79%”或屋顶测量级精度验收。

## 运行版本

- Python：192.168.130.238:8181，容器overlay-api-2，提交a4ff81a。app.py SHA256 `841b17ee38f78c642ed23876c08755283f69a329c465ca5ff3678d1921561409`；frame_geometry.py SHA256 `9c7fd9db16491e3ccbc4bb26bedce3a3bea7ac142f340dcaab1038081f4f954f`。
- 显式帧参数结果版本：`geo-stability-20260909-v2+frame-geometry-v1`。旧请求保留原版本后缀行为。
- Java：192.168.110.23:18081，test profile，PID882266。以现场运行包为基底，装入841c684107的帧协议及必需的既有真实框修复依赖；27个类哈希验证一致。
- 活动JAR SHA256 `b7a6b642c5ddfc1542fbba53f1e3d9baba8efedd0f1fbcbcc289e877565b8fa2`。保留现场`--spring.config.additional-location=file:/etc/lingxi-agent-v1/agent.yml`及其他启动参数和环境。
- Java只替换manifest列出的biz-ai、biz-inspection类；其他ZIP entry内容逐项保持一致。

## 发布中发现并处理的问题

1. 现场Java包较本机缺少此前真实框和重算质量状态修复。首个候选包的运行依赖测试4项失败，未发布该候选；补齐DetectionBoxGeometry、TaskEventServiceImpl相关既有修复及类族后，最终运行包10项测试全部通过。字节码差异核对涉及真实框/质量状态变更及编译器生成代码差异，没有覆盖无关模块。
2. 测试8181默认内参补偿与本机通用默认值不同。保留测试配置；显式历史参数/新参数分别验证，旧代码兼容比较改在同一容器环境执行，避免误把环境差异当回归。
3. Java健康探针初稿遗漏现场`/api/`上下文，停止了该轮询脚本，改为正确业务URL检查。候选Java进程保持运行，没有因此回滚或再次启动；最终业务检查通过。

## 验证

- 发布前两次即时读取活动飞行缓存：0。原Java进程优雅退出，无强制结束。
- Python健康检查及OpenAPI新字段通过。除overlay-api-2外46个容器的实例与启动时间不变。
- 打包后的候选Java及现场依赖：10项定向测试通过。
- 候选Java的强类型ArApiUtil实际调用已发布8181：8点成功，版本后缀和冻结请求追溯通过。
- 在线3事件×历史/新参数共6组、48点，坐标与离线基准最大差约1.3e-9m（数值计算差异）。沿用已核验正射参考的降幅49.2%/79.5%/83.8%，不将该数值差异解释为测量精度。
- 同一运行容器比较旧app.py与新app.py，12条原请求、77目标的坐标及质量状态一致。请求schema扩展可能改变指纹，不宣称与旧版所有元数据逐字节相同。
- 显式参数冲突与时钟不确定性过大均返回HTTP400。
- 平台 `/gis_server_a/portal/task/situation/detail/event/c5354403ae3b44edb35b2dfc01ede1a4` 返回HTTP200、code200、success=true。
- 未制造新预警或飞行任务，未更新历史事件；新航次自然回调及生产者元数据尚未验收。

## 测试入口和边界

[test-kit/README.md](test-kit/README.md) 给出平台/Swagger入口、六组可直接使用的JSON、Python回归脚本和UI检查要点。该脚本已实际请求部署后的测试接口并全部通过。

未携带frame_geometry的普通回调仍走旧内参路径；本次上线不把探索性1344像素内参设置为所有设备默认值。真实框修复已进入新请求和手动重算计算链，但本次未执行历史回填。屋顶高程、曝光同步和其他地点的绝对精度仍需独立数据验证。

## 回滚资产

- Python：`/data/workspace/qzhang/code/infer_api/overlay-geolocation-releases/20260915-frame-a4ff81a/`，含app.py.before、候选和manifest/receipt。回滚需恢复app.py并移除本次新增frame_geometry.py，再仅重启overlay-api-2。
- Java：`/data/ktd/project/ktd-backend/geolocation-releases/20260915-frame-841c684107/before.jar`。备份SHA256 `75a1ea9767346737eefdf815364b8197112af4e9f494202f86ece098a898a41b`。回滚同样须活动任务检查、优雅退出并保留当前附加配置。

回执、类级manifest和线上验证JSON保存在本目录。无服务器凭据或运行配置文件入库。
