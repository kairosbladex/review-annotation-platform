# 实际测试镜像的复检与历史对比回归

2026-10-16。本任务先只读核对 8189 的运行镜像和测试副作用，再在断网、无 GPU 的独立容器验证。发现并修复了坏图片触发历史对比 HTTP 500 的确定性问题；未改比较算法、复检模型或业务判断。本文不是准确率报告，也不是共享页面验收。

## 基线与红绿结果

实际镜像为 `sha256:a7fc3867056b89fe5aee410ad499e8243d05cb537d81c9cfd24ebbff0455698e`。其两份历史比较核心文件与 `feature/history-compare-concise-summary@f96def36` 逐字节相同，保留现有 v9c 分类归并摘要；不能用主仓旧版测试冒充实际运行版本验收。

| 验证 | 结果 |
|---|---|
| 本地解码边界红测 | 3 项中 2 通过、1 失败，确认缺失可选 HEIF 解码器时没有正确分类 |
| 本地最小修复与历史三测试文件 | 30/30 通过 |
| 原测试镜像完整红测 | 31 项中 27 通过、3 断言失败、1 错误、0 跳过；8.059 秒 |
| 同一镜像 + 两个函数增量 | 33/33 通过、0 跳过；7.188 秒 |

候选比完整红测多出的 2 项为正常本地图片不调用 `Path.read_bytes`、本地文件不存在不冒充图片格式错误；其余原 31 项全部保留。首次目录挂载和首次候选文件权限失败均发生在用例执行前，单独记录在 `results.json`，没有当作识别失败或从测试分母偷偷删除。

红测具体发现：导入 ultralytics 后，它会包装 `PIL.Image.open`；Pillow 无法识别输入时尝试导入可选 `pi_heif`，镜像中该包不存在，产生 `ModuleNotFoundError`。`decode_image` 原来没有处理这个可识别的降级边界；实际 `/api/v1/vl-compare/*` 入口还直接解码图片，根本不经过 `decode_image`。因此必须分别修复实际加载入口和共用解码函数，不能只修本地 helper 就宣称 HTTP 已修好。

## 最小修复契约

| 运行文件 | 仅允许替换的函数 | 候选源码 |
|---|---|---|
| `api/vl_compare.py` | `_load_image` | 同目录 `candidate_vl_compare_image_loader.py` |
| `history_compare_runtime/services/history_compare.py` | `decode_image` | 本仓 `backend/app/services/history_compare.py` 中同名函数 |

已独立验证两份候选文件在目标函数之外逐字节不变；完整文件与函数 SHA-256 见 `results.json`。实际发布构建器按原始完整文件摘要和函数白名单替换，不能用本仓旧的整份 history 文件覆盖测试镜像的新摘要实现。

无效或不支持的 base64、HTTP 下载、本地图片由实际 HTTP 入口返回 422。`pi_heif` 缺失只在 `ModuleNotFoundError.name == "pi_heif"` 且异常上下文为 `PIL.UnidentifiedImageError` 时转换，其他依赖故障继续作为执行错误。API 对具有 OS errno 的 I/O 异常也保持原错误语义。本地文件以 `Path.open('rb')` 文件流进入解码，在解码边界外完成打开；使用 `with Image.open` 及时关闭，未新增整文件预读。

文件不存在、其他模块缺失在本轮仍返回 500，但不会被写成成功或“没有明显变化”。本轮没有扩展这些错误的业务分类，也没有给服务器内部故障套用“图片格式错误”。

## 回归覆盖与限制

直接复用镜像原有 `test_review_agent_routes.py` 1 项与 `test_mask_for_draw.py` 10 项。新增 `unittest` 用例覆盖实际 Luna HTTP 适配路径的三态、乱序编号、重复编号拒绝、超时、非法框和原帧绑定；历史比较覆盖真实几何配准、两处变化保持、亮度/平移不报变化、跨批保留、注册引擎实际 HTTP、最新分组摘要、many-to-many 方向、鉴权、模型超时和图片错误。

替身位于模型 HTTP、下载 HTTP、持久化及通知边界。正常历史几何、结果整理和 HTTP 路由都使用镜像真实代码。没有载入 `main` 的共享业务启动流程；未连接模型、MySQL、MinIO 或 MQTT。禁止真实网络连接和权重加载的防线在回归运行期间启用。

隔离参数：`--network none --runtime runc --memory 4g --memory-swap 4g --cpus 2 --pids-limit 256 --cap-drop ALL --security-opt no-new-privileges`，清空继承环境后只设置合成测试配置。仅挂载本轮测试脚本、候选摘要和两个候选源文件，均只读；未挂载共享业务缓存、上传目录、模型权重或凭据。临时图像、日志和可选本地测试状态仅在容器私有层或 tmpfs。

这 33 项不包含智能检索性能或全套瓦片初检验收；该部分由主任务的检索测试单独证明。mask 测试的单次 50ms 断言通过，也不能用作视频整任务延迟收益。真实模型识别准确率、页面交互、数据库写回、真实设备效果均未在本任务验证。

## 复现

本地 TDD 命令与结果见 `results.json`。容器设置 `PYTHONPATH=/app` 后执行 `python3 /regression.py`，其中 `/regression.py` 为同目录 `run_live_regression.py` 的只读挂载，合成测试文件与挂载目录须对容器用户可读。原镜像运行时不设置 `REGRESSION_CANDIDATE_HASHES`；候选验证把两个精确生成的运行文件挂到对应 `/app/...` 路径，并设置 `REGRESSION_CANDIDATE_HASHES=/candidate_hashes.json`。启动前会核对全部选定源文件摘要，摘要不同直接停止。

主任务尚需把这里已经验证的两个函数并入统一发布包，并独立复核发布包摘要与最终共享版本。本子任务没有提交、推送、部署、重启共享服务或写业务数据。

结束前已核对临时容器不存在，移除本轮服务器上的四个测试/候选副本及其唯一临时目录；本地源文件、候选构建输出和复现证据保留。共享容器仍为原 `b654ec3957ed...`、镜像仍为 `a7fc3867...`。
