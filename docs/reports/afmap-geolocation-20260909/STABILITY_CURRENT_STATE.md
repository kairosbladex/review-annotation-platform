当前任务未完成：测试Java接口尚未恢复，正在等待192.168.110.23的主机/SSH访问恢复。

已完成：8181已发布geo-stability-20260909-v2；108项Python测试及182次真实HTTP验证通过。186事件/372附件已按新检查更新，931目标可绘制、77目标不可用；6408快照记录中其余6036逐字保持不变。5条自然新预警已验证：4条正常定位共18目标，1条姿态时差1.017秒，重试后仍保留事件、不输出目标坐标。

Java补丁：修复Fastjson JSONObject(Map)共享逐目标状态，15项定向测试和11模块package通过。活动包只改GeoFromBboxesUtil.class，其余entry内容不变。原检出ktd_backend仍为干净prod；实现工作树提交dde37aeb8a。

发布后启动PID1175181提前消失，boot-apply.log为0字节，18081未监听。随后SSH也不可达：本机110.23:22连接被拒绝；从238侧到110.23:22和18081均超时。尚无证据证明主机为何失联，不将其归因于某个系统组件。

活动Java包SHA256：15fde7d9606bf271a191946e7a12e58621ac5965b69eada6df551159c0b6eb25。

原包SHA256：be9b9ae409b1fcbda90c810758e85335fb75468a42ab805a8fb195092775fd76；备份路径：/data/ktd/project/ktd-backend/geolocation-releases/20260909-stability-java-v2/biz-visualization.before.jar。

已准备修正启动脚本：显式nohup、stdin=/dev/null并核对启动进程存活。尚未成功执行，因为SSH连接被拒绝。恢复访问后先确认没有重复进程及飞行任务，再启动test/18081并检查真实API；如补丁启动失败，使用已核验原包回退。

另有4条自然新事件、8附件的逐目标质量记录修正已准备，坐标不变，尚未apply。备份stability-natural-state-correction.json，SHA256为1e36ac8728035760dfc8811b5fa38c5336d886b3da0340b920d3d1082d72b1d7。

不能报告本任务完成，也不能报告约20米真实位置残差已消除；用户暂未提供目标真值。斜拍旧模型仍被质量检查拒绝，需确认云台角度参考系后再核验。
