# 初审图片来源、MinIO 原图与数据库映射调查

调查日期：2026-09-26。数据库快照：20:38:11（UTC+8），后续读取 MinIO 和录像元数据。

## 结论

**MinIO 里确实存在清晰、无初审红框的原图。此前只沿初审附件的直接映射查找，覆盖不完整。** 本次补出 8 条事件的原图来源，并从一段任务录像中找到了另一条事件的无框候选画面。

实时告警链路还存在一个关键缺口：后端把回调中的同一张标注图同时保存为 `PRIMARYIMG` 和 `SHOTIMG`；因此“主图”这个名称不能证明文件是无框原图。知眸回调 DTO 已声明 `frame.image` 原图字段，但所审查实时事件保存代码没有使用它。历史回调是否实际传入该字段，当前数据库没有保存足够证据。

本调查完成取图与映射追踪，没有改变模型、复核结论或测试环境服务，也没有产生新的识别准确率。清晰文件存在、画面背景匹配、同一帧确认和复核准确率提升需要分别验证。

## 1. 范围及读操作

沿用上一份评估的 358 条冻结测试事件，涉及 78 个任务；不是测试库全部事件。本次读取 1,581 条相关附件、453 条源目标关系，并核查相同文件路径的其他附件记录。

- 数据库：只读事务，结束后回滚；读取附件、事件、任务、源目标关系、附件事件关联、任务文件和上游事件存档。
- MinIO：已使用测试环境配置完成认证 `stat/get/list`；未上传、删除或改写对象。读取 188 个任务媒体对象的状态，另核查 30 条事件附件记录对应的 21 个对象键。
- 视频：读取 4 个实际文件的媒体元数据；选取 1 段 235.97 秒录像做检索实验。
- 代码：Java 工作树 `feature/clean-source-recheck`，提交 `860c7e35c331d1b546d11fb8b09c0f3ed094685f`。本次未核验正在运行的 Java 包与该源码完全一致。
- 数据快照、下载图片及录像保存在本机调查目录；仓库仅交付报告、汇总、必要映射证据和离线汇总脚本，不提交凭据、完整数据库快照或录像。

## 2. 后端从哪里取得初审图

### 实时知眸／灵眸回调

```text
初审平台回调 rules[].image（开放平台 apps[].image 经适配转换）
  → 下载回调图片
  → MinIO: zm-zq-test/ktdTest/aiEventImg/YYYYMMDD/UUID.jpg
  → sys_annex_file: PRIMARYIMG 和 SHOTIMG 共用同一路径
  → 复核读取 SHOTIMG.file_path
```

代码位置：

- `biz-inspection/.../ai/ZhimouAiEventController.java:368-373`：只保存 `rule.getImage()`，同一个 `FileInfo` 写入主图和初审图。
- `biz-inspection/.../ai/LingmouAiEventController.java:384-389`：同类行为。
- `biz-ai/.../gdd/util/ZhimouUtil.java:1811`：开放平台 `apps[].image` 转成内部规则图片。
- `biz-inspection/.../ai/util/AiEventImageStorageUtil.java:104-116`：下载、UUID 重命名、保存；该段未做图片缩放。
- `sys-core/.../oss/MinIORepository.java:373-402`：保存传入文件字节。
- `biz-inspection/.../recheck/service/impl/TaskEventRecheckServiceImpl.java:1003-1056`：优先初审图，读取 `file_path`；未把 `file_lit_path` 当原图。

因此，所查链路的模糊不能简单归因为“复核误取缩略图”。它首先拿到的是初审平台已经输出的标注图片，部分样本只有 960×720。上游取流、缩放、压缩及成像质量仍需从初审服务进一步确定。

### 任务结束后的图片识别

```text
任务 CKIMG 附件 file_path（无人机照片）
  → 可选 AR 叠加图
  → 上传初审平台，device 携带输入附件 ID
  → 接收初审标注图
  → sys_annex_source_target 保存关联
```

代码：`TaskAfterAiIdentifyExecutor.java:151-167`，`TaskAfterAiZhimouCallbackService.java:193-260`。这条链路具备明确追溯条件。部分其他供应商也分别接收原图、标注图，例如 `HansuiEventController` 的 `raw_frame_path` / `processed_frame_path`；不同入口需要分开处理。

## 3. 数据库确实有映射关系

| 表／字段 | 含义 |
|---|---|
| `bo_task_event.task_id` | 事件所属任务 |
| `sys_annex_file.busi_id` | 事件 ID 或任务 ID，须同时看附件类型 |
| `busi_type_code=SHOTIMG` | 初审标注图 |
| `busi_type_code=PRIMARYIMG` | 主图，可能仍是同一张标注图或 AR 图 |
| `busi_type_code=CKIMG` | 任务媒体；实库同时包含照片、录像，不能只按类型当照片 |
| `sys_annex_file.file_path` | 存储对象路径 |
| `sys_annex_source_target.source_id / target_id` | 输入与输出附件 ID |
| `busi_type=WAR_TO_AI` | 原图与 AI 图关系；历史数据需归一化 |
| `busi_type=WAR_TO_AR` | 原图 → AR 图，作为可靠的方向依据 |
| `busi_type=AR_TO_AI` | AR 图与 AI 图关系；历史数据需归一化 |

本次 358 条事件的数据库分组如下，四组互斥：

| 分组 | 事件数 | 含义 |
|---|---:|---|
| 初审图有直接源目标关系 | 40 | 之前 36 条可下载执行，另 4 条旧初审图访问失败；有关系不等于已通过同帧检查 |
| 主图与初审图路径不同，但初审图无直接关系 | 8 | 本次通过主图路径补找原图 |
| 主图与初审图是同一个文件，且无直接关系 | 301 | 主图记录不能提供额外图像信息 |
| 无 SHOTIMG | 9 | 不能直接形成原图／初审图配对 |
| **合计** | **358** | |

其他关联检查：所查事件在 `sys_annex_file_relate_event` 中无记录；所查任务在 `bo_task_file_path`、`bo_ai_task_event` 中无记录。**这不代表没有录像**，录像实际还保存在 `sys_annex_file` 的 CKIMG 中。全库相同路径额外找到 30 条附件别名，但未发现新增直接源目标关系。`callback_param` 均为空，`img_param` 保存的是飞行和相机几何参数，未发现原始回调图片地址。

`bo_task.source_task_id` 的代码定义是“复飞来源历史任务 ID”，不能把它当本次飞行的原图关联。

## 4. 新补出的 8 条原图来源

实际追溯链路：

```text
事件 PRIMARYIMG.file_path
  → 同一路径的 AR 附件记录
  → WAR_TO_AR.target_id
  → WAR_TO_AR.source_id
  → 原图附件 file_path
  → MinIO 原始照片
```

8 条事件最终指向两张不同原始照片，均已下载，实际解码尺寸 **4032×3024**，视觉查看未见初审红框或 AR 标注。主图对应的 AR 图片为 1440×1080。多条附件 ID 指向同一个存储对象，按对象路径归一化后来源唯一。

| 原图拍摄时间（文件名） | 涉及事件 | 原图 SHA-256 前缀 |
|---|---|---|
| 2026-07-23 09:43:23 | `1ae610…`、`20bdc7…`、`4a02d0…`、`ad343c…` | `4d41a6d940b2` |
| 2026-07-23 10:28:01 | `0e2e65…`、`92d84f…`、`9b7c54…`、`cf1811…` | `a9f4fb625d78` |

完整事件 ID、附件 ID、关系和对象路径见 `recovered-primary-lineage.json`。

这 8 条旧 SHOTIMG 指向 `/upload/lingmouAiImg/...`。在当前配置的 MinIO 桶按对应对象键查询均返回 `NoSuchKey`；相关两个目录前缀也为空。另 4 条此前公共 URL 访问失败的旧初审图属于同类路径。**这只证明当前桶里对应键不存在，不能据此断言旧服务器本地文件或备份已删除。** 下一步还需追查旧 `/upload` 文件存储。8 条新增原图来源暂不计为新增完整复核配对。

## 5. 从任务录像补找无框画面：已做一个真实样本

在没有初审图直接映射的 309 条事件所关联任务中，已确认存在：

- 126 个带分辨率元数据的任务照片对象：80 张 4032×3024、8 张 3840×2160、1 张 14400×7200、23 张 1280×1024、14 张 640×512。照片对象存在已核验，未逐张确认都是无框可见光图；其中包括红外媒体。
- 36 个 MP4 录像对象，关联 31 个任务；161 条事件有同任务录像线索。
- 54 条事件有同任务照片线索；与录像线索合并去重为 173 条。**173 是可继续检索的事件数，不是找回同帧原图的数量。**

对 4 个真实视频文件读取元数据，分别确认 1280×1024、3840×2160、3840×2160、1920×1080 主视频流。高像素尺寸本身不证明成像细节提升。

### 录像检索实例

- 事件：`062d97ceb7464760b4bdac8780a73867`，`Event-20260910141741`。
- 任务：`5535fb8c26e849afb8b74a925b660ff6`，`TC-Task-202609100011`。
- 初审图：960×720，红外屋顶画面，顶部两个红框标注“人员”。
- 任务录像：1920×1080，235.97 秒，附件 `0bebe79c031a40cca9ab96d74d09ce24`。
- 检索：先每 4 秒取帧，背景特征检索后局部按 0.25 秒细查；去除红框特征影响、统一分析尺寸，使用 CLAHE-SIFT 与 RANSAC 配准。
- 最佳候选：**约 140.5 秒**；284 个特征匹配中有 **247 个配准内点**，分析尺寸下中位重投影误差约 **1.01 像素**。
- 视觉查看：无初审红框，同一屋顶布局和三组圆形设备清楚可见，红框位置对应的物体外观更像固定屋顶设备。此项为分析观察，未写回真／假目标标签。

该结果支持“从录像找回对应场景的无框画面”这条路线。原始流的时间戳对齐、精确同帧及目标一致性尚未完成；当前只保存为候选证据。初审图与录像纵横比不同，接入时还需经过坐标配准，不能直接按单一比例缩放红框。现有自动复核服务尚未使用这条录像候选。

## 6. 用户最初指定的事件

`b3ac0a6c3cd34dc2be9821bf41614558` / `Event-20260923180729`：

- 主图附件 `2b0d2ba8866d45659bc1f5aa162dbc32` 和初审附件 `672f78cc09a84bd392ae3cbec2742f77` 指向同一 MinIO 对象。
- 对象：`ktdTest/aiEventImg/20260923/4680f5f1738e447cb9adcced2bb204f5.jpg`，220,162 字节，实际尺寸 960×720。
- 下载 SHA-256 完全相同：`5a88db4dada7971682a728f5aaf92c67dabbd8aba45a173db3190cd45813cd21`。视觉查看有“人员”红框。
- 无直接源目标关系、无任务附件、无附件事件关系、无任务文件表记录、无对应任务上游存档。当前桶顶层也未出现该任务 ID 的目录。

**在本次已查数据库关系和存储路径内，这条仍未找回无框原图。** 没有遍历所有历史备份，也没有证明上游初审平台未保存原始帧。

## 7. 对后续实现的具体建议

1. **补齐新增实时事件的原图保存。** 知眸 DTO `ZhimouTaskEventDataDTO.EventDataFrame.image`（第 48-49 行）已注明“消息原图地址”。核实真实回调非空后分别保存 `frame.image` 和规则标注图，并写入明确的 `WAR_TO_AI` 关系；原图缺失应如实记录。
2. **扩展历史原图追溯。** 在已有 `SHOTIMG → 源目标关系` 之外，支持 `PRIMARYIMG 相同存储对象 → AR 附件 → WAR_TO_AR → 原图`。保留唯一来源、对象可读、画面和目标一致性检查；不能仅凭主图名称选取。
3. **增加任务录像候选检索。** 用 `event.task_id → CKIMG 录像 → 时间窗口／图像检索 → 配准 → 红框目标一致性` 补齐实时事件。无时间戳或存在移动目标时先供人工查看；不直接改变自动结论。
4. **恢复旧标注图访问。** 追查 `/upload/lingmouAiImg` 的原服务器目录或备份，恢复后再验证这 8 条新增原图来源及此前 4 条失败事件。
5. **接入复核效果页后再评估。** 展示初审图、无框来源、映射类型、分辨率和配准状态；将“直接同帧来源”与“同任务录像候选”区分。完成原图接入和实际复核后，才能另行评估准确率变化。

## 8. 附件及复算

- `metrics.json`：按冻结数据复算的汇总和视频检索证据。
- `recovered-primary-lineage.json`：8 条事件新增原图链路，已移除本机文件路径。
- `summarize.py`：离线汇总脚本；输入本机只读快照目录，不含连接凭据。
- `match_recording.py`：使用已下载录像与初审图复现候选检索，不访问外部服务。OpenCV 版本差异可能影响具体匹配数量。
- `原图来源与取回证据.html`、`build_evidence_page.py`：页面模板和组装脚本。已生成本机调查目录下的 `view/index.html`，可查看图片对比；图片留在本机调查目录。

本机完整调查目录：`/REDACTED_HOME/Downloads/ktd-original-image-lineage-20260926`。

```sh
python3 summarize.py /path/to/private-snapshot-directory metrics.json
python3 build_evidence_page.py /path/to/private-snapshot-directory
# 已安装 opencv-python、numpy 的 Python 环境中：
python3 match_recording.py /path/to/annotated.jpg /path/to/recording.mp4 /path/to/candidate-output
```

原始数据库快照仅留本机，文件权限为 0600；报告未包含 Bearer Token、数据库密码、MinIO 访问密钥或临时签名 URL。
