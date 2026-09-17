# 光伏光斑组件定位实施记录

2026-09-17。当前交付：**P0 数据冻结完成，P1 人工配准接口／简化标注页／验收工具已实现；真实 P1 精度验收仍待独立人工角点。** 用户要求实施完整阶段路线，但未通过门槛不进入后续自动确认。没有 GPU 推理、服务器部署、Java 事件改写或历史回填。

## 代码与数据

- 本仓库：`scripts/pv_component_workbench.py`（freeze / verify / replay / assess / serve）、同名 HTML、`backend/tests/test_pv_component_workbench.py`。
- AR 实现：`/Users/wangxiaolin/PycharmProjects/drone-overlay-ar-right-20260917`，本地提交 `f830fc2`。`pv_component_match.py` 和挂入实际 FastAPI 的 `/api/v1/pv/component-match`；接口说明 `docs/PV_COMPONENT_MATCH.md`。
- Java：本轮未更改；此前 `4c073821ef` 原图 EXIF 镜头补丁仍未部署。
- 冻结数据：`data/pv-component-20260917/manifest.json`，7 张原图、27 个原始索引检测框、逐图真实组件几何、原始请求与旧结果来源、代码和文件哈希。图片／原始数据不进 Git。
- 真值单独放在 `gold/labels.json`，不出现在推理输入或工作台数据接口。只有原样例第 1 号框（从 0 起算）由用户确认组件 `4028ee97a0a8029801a0a81dad230278`，即 `qx1fz3H3-qx1fz3l10`；其他 26 框是 unknown，全部为 development，不能当独立盲测数据。

## 已实现行为

图片与图层哈希绑定；局部米制 XYZ 按平面配准；至少 4 个拟合点和 2 个不重复的独立检查点；检查误差相对组件短边 ≤10%；平面残差 ≤0.05m；不共面／角点失效／不支持的图片 EXIF 方向明确失败。检测框可用显式仿射变换还原缩放、裁剪和旋转坐标。

匹配只在控制点覆盖范围内确认，并检查整个检测框到边界的距离。重复候选、跨边界、近边界及外推不输出已确认组件。逐框保留 index；输出真实 ID、code、行列及投影轮廓；`absolute_geolocation_verified=false`、`business_write_enabled=false`。

工作台根据用户“交互简单一些”的反馈改成逐步操作：

1. 选择能确认编号的组件，按左图依次在照片点四角。
2. 在同一排另一端再标一块，让关注区域夹在两块之间。
3. 选中间的一块，独立点四角作检查。
4. 点击确认验证，显示组件轮廓、逐光斑结果或失败原因。

拟合／检查角色由程序分配，用户不填写平面参数；支持放大、90°显示旋转（点击还原到原图）、撤销、换组件、浏览器本地草稿、导出及追加其他行。共享物理顶点只计一次，不能重复充当独立检查点。原始失败／成功验证记录按内容哈希保留，重复提交不制造重复记录。此处幂等仅验证本地记录，尚未接入业务数据库幂等。

## 验证证据与真实状态

- 95 项测试通过：新增匹配模块 27 项、原照片／frame geometry／地理接口回归 62 项、工作台跨仓库测试 6 项；包含实际 AR 应用挂载新路由。JUnit 记录 `data/pv-component-20260917/runs/tests.xml`。
- 真实 7 图／27 框已通过本地实际 HTTP 接口回放，索引和数量保持一致；当前无人工锚点，27 项全部 `unavailable`。见 `runs/live-http-p0.json`。这不是精度通过。
- 在单独端口 8197、独立合成图片上，实际浏览器完成三块组件共 12 次点角、确认和 HTTP 保存，获得正确 H1/l4；该测试只证明交互／接口闭环，不计真实组件精度。临时测试服务完成后关闭。
- 原图工作台已在浏览器检查加载和布局。真实锚点目录当前为空，原 H3/l10 仍未匹配；邻近行列独立检查仍缺失。
- `assess` 对所有冻结槽位检查完整性，未知真值不计正确，独立盲测始终未通过。相邻横向／纵向组件的独立检查不足时，P1 几何门槛不放行。业务描述保存／浏览器业务页面尚未验收。

## 运行和继续

```sh
# cwd: review-annotation-platform
.venv/bin/python scripts/pv_component_workbench.py verify \
  --snapshot data/pv-component-20260917

.venv/bin/python scripts/pv_component_workbench.py assess \
  --snapshot data/pv-component-20260917 \
  --matcher-root /Users/wangxiaolin/PycharmProjects/drone-overlay-ar-right-20260917

.venv/bin/python scripts/pv_component_workbench.py serve \
  --snapshot data/pv-component-20260917 \
  --matcher-root /Users/wangxiaolin/PycharmProjects/drone-overlay-ar-right-20260917 \
  --port 8196

PV_MATCHER_ROOT=/Users/wangxiaolin/PycharmProjects/drone-overlay-ar-right-20260917 \
  .venv/bin/python -m pytest -q backend/tests/test_pv_component_workbench.py
```

本地入口：`http://127.0.0.1:8196/?case=4e95d41793e143969a61cbc66d285343`。仅监听 loopback，不访问业务数据库；新端口，不干扰原 8190/5173 等服务。重启先确认本工单 PID，不能关闭未知进程。

当前阶段门槛：P0 完成；P1 待真实人工锚点与相邻行列核验；P2 自动网格／GPU 对比尚未执行；P3 缺独立 50 图／100 个真值及歧义样本；P4 Java 接入／业务闭环尚未执行；P5 未获发布执行授权。

用户随后提出 Hermes Agent 本地模型路线并要求调研：已只读检查官方能力、GPU 和 Ollama 模型元数据，未安装 Hermes。若后续采用，让代理生成机器候选并由人工确认，不能伪装成现有 `human_confirmed` 标注，也不能读取 gold 后反向拟合。现有几何校验与本地工作台可复用。

所有实现仅本地小提交、不 push，保留原工作区四个历史验证文件改动及原有数据。没有修改共享标定、模型服务或内存记忆。
