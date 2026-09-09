测试环境稳定性修复、发布及现有事件复测已完成，等待用户交接。最后核验：2026-09-09 19:16。不能将本批工程稳定性验收视作真实地理精度验收。

8181 正在运行 geo-stability-20260909-v2；app.py SHA256 为 fdfac3f15644da4382583c1fea46eced6cd74e6e39c4b0afd7018eebd4de899e。108 项 Python 测试、182 次真实 HTTP 验证通过。19:15 再核对容器 ID、启动时间、活动源码 SHA，8188/8189/Triton 均保持不变；证据在 Python 仓库 docs/stability-final-runtime-verification.json。

Java 容量保护候选已发布，活动包 SHA256 为 dfd679d2aa48bbbedf3d68c1ccd48eec7c88eb3e62c1882651c0d70ff151fdcf。当前 PID 4884，19:11:33 启动，test/18081，日志 /data/ktd/project/ktd-backend/logs/app.log。19:16 的本机直连 Java API、19:15 的平台 186 事件接口均通过。16 项定向测试、11 模块 package 通过；运行包只变更 GeoFromBboxesUtil.class，其余 ZIP entry 内容不变。

恢复和重启经过：19:05 只读检查发现主机和 PID 2247 已恢复，运行包为 15fde7d9。空闲检查后发布容量包并启动 PID 4466。启动后首次 API 为 502，随后 19:11:19 验证正常；19:11:31 出现正常关闭流程，19:11:33 另一个登录会话启动 PID 4884，仍使用同一 dfd679d2 包。未再覆盖或重启该进程。当前服务已恢复；触发关闭的人或机制未核实，不能断言由定位补丁或主机故障导致。证据为 Java docs/stability-capacity-startup-investigation.json、stability-capacity-final-runtime.json。早期 live-health 只证明当时瞬时正常。

回退包保留：/data/ktd/project/ktd-backend/geolocation-releases/20260909-stability-capacity/biz-visualization.before.jar 为 15fde7d9606bf271a191946e7a12e58621ac5965b69eada6df551159c0b6eb25；更早 20260909-stability-java-v2/biz-visualization.before.jar 为 be9b9ae409b1fcbda90c810758e85335fb75468a42ab805a8fb195092775fd76。恢复后的实际包与两级备份已重新核验。

已完成历史批次：186 事件/372 附件的 v2 历史更新已 apply、数据库 verify、API 验证；931 目标可绘制、77 不可用，6408 条快照中其他 6036 条逐字不变。有效备份 stability-history-correction-v2.json SHA256 为 f937863041c92c6dd4f3e747b7f24aa358eb758f41d4f19010cbe74c72287bfb。不得重复 apply；初次未压缩候选 stability-history-correction.json 曾事务回滚，不得用于重试。

已完成自然状态小批次：4 事件/8 附件的逐目标 quality 已 apply/verify，全部坐标不变。冻结备份 stability-natural-state-correction.json SHA256 为 1e36ac8728035760dfc8811b5fa38c5336d886b3da0340b920d3d1082d72b1d7。快照和备份先上传到服务器 20260909-stability-natural-state 并核对 SHA。此批位于旧 6408 行快照之外；不得重复 prepare/apply。

19:15 刷新 verify-stability-history.py，186 事件 API、所有 6408 快照、活动包 SHA 一致。19:16 verify-stability-new-events.py 通过：4 条已有自然事件共 18 目标的 API/存储/实际 HTTP 回放坐标和各自 quality 一致，index 无缺漏或重复；1 条姿态时间差 1.017 秒的事件保留、重试完成而不撒点。没有人为创建飞行或正式预警。5 条自然事件都发生在最新 Java 发布之前；最新 Java 自然写入尚未观察到，不能把历史修正当作新回调验收。

离线验收器反例检查通过：修正候选的 4 事件/18 目标可通过，旧共享 quality、API 质量错配、重复 index 被拒绝。证据 stability-natural-verifier-offline-validation.json；在线逐目标结果另见 stability-natural-events-verification.json。

真实约 20 米残差仍缺地面真值，未声称消除。斜拍旧 body+gimbal 模型仍被质量检查拒绝，需确认云台参考系和标定；screened 不等于测绘精度通过。地图视觉未由本轮浏览器验收。3030 条缺原输入历史已延期到报告工作树待办 #8。

后续如有真实新预警，按保留原请求回放核对 index、逐目标 quality、时间差和事件保留；不要制造正式事件。只有用户要求新精度任务后才安排真值/镜头/姿态标定。所有仓库仅本地提交，不 push；原 ktd_backend 保持干净 prod，无关 IDE、数据、模型和 pyc 不动。不使用子代理，不写 memory。交接前清理仓库外临时认证文件；配置 jar 和凭据不得入库。
