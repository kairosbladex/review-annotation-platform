# ACCEPTANCE_KTD-ANNEX-LOCAL-IMPORT

## Status

`PARTIAL` — 本机已能从 ktd 测试库同步候选框、按拼接 URL 看图，并在标注平台小批量导入；人工标注 1510 张全量、class_id 官方映射和 GPU 复检写回都还没做。

## Ticket

`KTD-ANNEX-LOCAL-IMPORT`

## 做了什么

- 把 `imgxy` 像素 `xywh` + `frame_width/height` 转成平台归一化 `[x1,y1,x2,y2]`，候选框 `raw_decision=uncertain`。
- 新增只读 `HttpPrefixObjectStore`：图片不落本地 MinIO，浏览器直接打开 `SOURCE_IMAGE_BASE_URL + file_path`。
- 脚本 `scripts/sync_ktd_annex.py` 把符合条件的 `sys_annex_file` 行同步到本机 MySQL 副本。
- 脚本 `scripts/ingest_ktd_annex.py` 从本机 MySQL 调 `POST /api/v1/ingest/review-events`。
- 源库密码只放 gitignore 的 `.env.ktd.local`；仓库模板是 `deploy/ktd-local.env.example`。

Changed files:

- `backend/app/services/annex_import.py`
- `backend/app/adapters/object_store.py`
- `backend/app/config.py`
- `backend/tests/test_annex_import.py`
- `backend/tests/test_object_store.py`
- `backend/tests/test_config.py`
- `scripts/sync_ktd_annex.py`
- `scripts/ingest_ktd_annex.py`
- `deploy/ktd-local.env.example`
- `.env.example`
- `docker-compose.yml`
- `requirements.txt`
- `README.md`

## 没做什么

- 没有在本机训练 YOLO / YOLOE。
- 没有把 1510 张图拷进本地 MinIO。
- 没有把测试库整库同步，只同步了 `/zm-zq-test/` 且 `imgxy` 含 width、不含 type 的 1510 行。
- 没有官方 `class_id` 映射，导入后 `class_id` 为空、只保留 `class_name`。
- 没有调用 GPU `/review` 把真/假/存疑预填进标注结果。
- 没有改 Docker 已运行栈（8391/5274）；本机标注用 8190/5173 + SQLite。
- 未提交、未 push。

## 哪里可能有问题

- 本机 Homebrew MySQL 没有在跑；当前副本是 Docker `review-ktd-mysql`（`127.0.0.1:3307`）。容器删了要重新 sync。
- 启动本机 API 必须把 `MINIO_ENDPOINT` 置空，否则会去连默认 `minio:9000`，远程图会被判缺失或变慢。
- 前端 `<img>` 直连测试 MinIO；证书是 DigiCert，本机校验通过，但没有用浏览器点过标注台。
- 超小框（长边 < 48px）复检服务会标存疑；当前导入只是把旧框交给人标，没有先跑 GPU。
- 源库密码出现在本机 `.env.ktd.local`，不要提交。

## 如何验收

```bash
cd /Users/wangxiaolin/PycharmProjects/review-annotation-platform
PYTHONPATH=backend python3 -m pytest backend/tests -q
# 期望 21 passed

# 本机 MySQL 副本
mysql -h 127.0.0.1 -P 3307 -uktd_local -pktd_local ktd_lx -e "SELECT COUNT(*) FROM sys_annex_file"
# 期望 1510

# API 已在 8190、前端已在 5173 时：
curl -sS -X POST http://127.0.0.1:8190/api/v1/auth/login \
  -H 'Content-Type: application/json' \
  -d '{"username":"admin","password":"REDACTED_FOR_GITHUB"}'
# 打开 http://127.0.0.1:5173 登录后应看到 20 条待标样本，红框叠在远程原图上
```

本次已执行证据：

- `PYTHONPATH=backend python3 -m pytest backend/tests -q` → **21 passed**
- `python3 scripts/sync_ktd_annex.py --env-file .env.ktd.local` → `copied: 1510`
- `python3 scripts/ingest_ktd_annex.py --env-file .env.ktd.local --limit 20` → `created: 20, failed: 0`
- `GET /api/v1/samples` → `total=20`，`asset_status=ready`，`raw_decision=uncertain`
- 资产 URL 返回 `https://lxtest.zhuoyilingxi.com:8088/minio-src/zm-zq-test/...jpg`，HTTP 200 JPEG
- 前端 `http://127.0.0.1:5173/` HTTP 200
- 未在浏览器里实际点 true/false（当前环境没有浏览器自动化工具）
- 角标过大挡住小目标：已改为框外小号序号、去掉框内半透明填充、缩放上限 400%。
- 自动跳下一框：代码已写（`saveLabel` + `nextPendingCandidateId`），用户在 http://127.0.0.1:5173 多次硬刷新后反馈仍停在当前框；两次点真目标会翻图。留给新会话继续修。

## Standards review

- 未把源库密码写入可提交文件。
- 平台 bbox 仍是归一化 `[x1,y1,x2,y2]`；uncertain 没有被改成 false。
- 不覆盖已有 ModelRun；这次导入是新的 review-event 样本。
- 只改本工单需要的文件。

## Spec review

- 满足「本机同步元数据 + 远程拼 URL 看图 + 小批量导入标注」。
- 不满足「本机训练」；那不是本平台一期范围。

## Git state

- branch: `master`
- HEAD: `9fc5ded docs(integration): record local handoff state`
- 本次改动未 commit、未 push
- `.env.ktd.local` 被 `.gitignore` 的 `.env.*` 忽略
