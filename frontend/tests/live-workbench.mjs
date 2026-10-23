// Explicitly opt in after the independent test release has been deployed.
// No API interception, recorded browser state, credential output, or model retries.
import assert from "node:assert/strict";
import { execFileSync } from "node:child_process";
import { createHash } from "node:crypto";
import { chmod, mkdir, mkdtemp, readFile, realpath, rm, stat, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import path from "node:path";
import { pathToFileURL } from "node:url";

const TARGET = "http://192.168.130.238:5276";
const RELEASE_PATTERN = /^\/data\/workspace\/user\/review-vision-test\/releases\/[a-f0-9]{40}$/;
const TERMINAL_JOB = new Set(["succeeded", "completed", "success", "failed", "error", "partial_failed"]);
const TERMINAL_RETRIEVAL = new Set(["completed", "partial", "failed", "cancelled", "submission_unknown"]);
const PYTHON = process.env.VISION_PYTHON ?? "/Users/wangxiaolin/PycharmProjects/review-annotation-platform/.venv/bin/python";
const HASH = (bytes) => createHash("sha256").update(bytes).digest("hex");
class CheckFailure extends Error { constructor(code) { super(code); this.code = code; } }
const check = (condition, code) => { if (!condition) throw new CheckFailure(code); };
const enumValue = (value) => typeof value === "string" && /^[a-z0-9_-]{1,80}$/i.test(value) ? value : "unknown";
const identifier = (value) => typeof value === "string" && /^[a-z0-9-]{8,80}$/i.test(value) ? value : null;
const quote = (value) => "'" + value.replaceAll("'", "'\"'\"'") + "'";
const delay = (ms) => new Promise((resolve) => setTimeout(resolve, ms));

const FIXTURE_PROGRAM = String.raw`
import sys
from pathlib import Path
import cv2
import numpy as np
from PIL import Image, ImageDraw

root = Path(sys.argv[1])
rng = np.random.default_rng(20261018)
image = Image.new('RGB', (640, 360), (45, 59, 71))
draw = ImageDraw.Draw(image)
for x, y, radius in rng.integers([8, 8, 2], [632, 352, 9], size=(260, 3)):
    color = tuple(int(v) for v in rng.integers(60, 230, size=3))
    draw.ellipse((int(x-radius), int(y-radius), int(x+radius), int(y+radius)), fill=color)
draw.line((20, 30, 600, 320), fill=(240, 190, 70), width=3)
image.save(root / 'baseline.png')
current = image.copy()
ImageDraw.Draw(current).rectangle((270, 130, 370, 230), fill=(245, 245, 245))
current.save(root / 'current.png')
frame = cv2.imread(sys.argv[2]) if sys.argv[2] else cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR)
if frame is None:
    raise ValueError('fixture image cannot be decoded')
frame = cv2.resize(frame, (640, 360), interpolation=cv2.INTER_AREA)
writer = cv2.VideoWriter(str(root / 'clip.mp4'), cv2.VideoWriter_fourcc(*'mp4v'), 12, (640, 360))
if not writer.isOpened():
    raise RuntimeError('MP4 fixture writer unavailable')
for _ in range(72):
    writer.write(frame)
writer.release()
capture = cv2.VideoCapture(str(root / 'clip.mp4'))
assert capture.isOpened() and int(capture.get(cv2.CAP_PROP_FRAME_COUNT)) == 72
assert abs(capture.get(cv2.CAP_PROP_FPS) - 12) < 0.1
ok, decoded = capture.read()
capture.release()
assert ok and decoded.shape[:2] == (360, 640)
`;

const LOGIN_PROGRAM = String.raw`
import json, re, stat, subprocess, sys, urllib.request
from pathlib import Path
try:
    release = Path(sys.argv[1])
    if not re.fullmatch(r'/data/workspace/user/review-vision-test/releases/[a-f0-9]{40}', str(release)):
        raise ValueError()
    if release.resolve() != release or release.is_symlink():
        raise ValueError()
    identities = subprocess.check_output(['docker', 'inspect', '--format',
        '{{.State.Running}} {{index .Config.Labels "vision.release"}}',
        'review-vision-test-api-1', 'review-vision-test-worker-1',
        'review-vision-test-frontend-1', 'review-vision-test-algorithm-1'],
        text=True, timeout=10, stderr=subprocess.DEVNULL).splitlines()
    if len(identities) != 4 or any(value != 'true ' + release.name for value in identities):
        raise ValueError()
    source = release / '.private.env'
    info = source.stat()
    if source.is_symlink() or not stat.S_ISREG(info.st_mode) or info.st_size > 65536 or info.st_mode & 0o077:
        raise ValueError()
    values = {}
    for line in source.read_text().splitlines():
        key, separator, value = line.partition('=')
        if key in {'VISION_ADMIN_USERNAME', 'VISION_ADMIN_PASSWORD'}:
            if not separator or not value.startswith("'") or not value.endswith("'"):
                raise ValueError()
            values[key] = value[1:-1]
    request = urllib.request.Request('http://127.0.0.1:5276/api/v1/auth/login',
        data=json.dumps({'username': values['VISION_ADMIN_USERNAME'], 'password': values['VISION_ADMIN_PASSWORD']}).encode(),
        headers={'Content-Type': 'application/json'}, method='POST')
    with urllib.request.urlopen(request, timeout=15) as response:
        body = json.loads(response.read(65536))
    token = body['access_token']
    if not isinstance(token, str) or not 16 <= len(token) <= 16384:
        raise ValueError()
    sys.stdout.write(json.dumps({'access_token': token}))
except Exception:
    sys.stderr.write('Live authentication failed.\n')
    sys.exit(1)
`;

async function fixtures() {
  check(path.isAbsolute(PYTHON), "PYTHON_PATH_MUST_BE_ABSOLUTE");
  let liveImage = process.env.VISION_LIVE_IMAGE || null;
  if (liveImage) {
    check(path.isAbsolute(liveImage), "LIVE_IMAGE_PATH_MUST_BE_ABSOLUTE");
    liveImage = await realpath(liveImage);
    check((await stat(liveImage)).isFile(), "LIVE_IMAGE_MUST_BE_FILE");
  }
  const directory = await mkdtemp(path.join(tmpdir(), "vision-live-workbench-"));
  await chmod(directory, 0o700);
  try {
    execFileSync(PYTHON, ["-c", FIXTURE_PROGRAM, directory, liveImage ?? ""], { stdio: ["ignore", "pipe", "pipe"], timeout: 30000 });
    const baseline = path.join(directory, "baseline.png"), current = path.join(directory, "current.png"), video = path.join(directory, "clip.mp4");
    const metadata = { baseline_sha256: HASH(await readFile(baseline)), current_sha256: HASH(await readFile(current)), video_sha256: HASH(await readFile(video)), width: 640, height: 360, duration_seconds: 6, frames: 72, frames_per_second: 12, video_source: liveImage ? "user_supplied_frozen_image" : "synthetic_fixture", live_image_sha256: liveImage ? HASH(await readFile(liveImage)) : null, independent_object_labels: false };
    return { directory, baseline, current, video, reviewImage: liveImage || baseline, metadata };
  } catch {
    await rm(directory, { recursive: true, force: true });
    throw new CheckFailure("FIXTURE_GENERATION_FAILED");
  }
}

async function main() {
  if (!process.argv.includes("--execute") && !process.argv.includes("--prepare-only")) {
    process.stdout.write("Prepared runner only. Use --prepare-only for offline fixtures, or --execute after deployment authorization.\n");
    return;
  }
  if (process.argv.includes("--prepare-only")) {
    const input = await fixtures();
    try { process.stdout.write(JSON.stringify({ status: "fixtures_verified_offline", ...input.metadata }) + "\n"); }
    finally { await rm(input.directory, { recursive: true, force: true }); }
    return;
  }
  const release = process.env.VISION_RELEASE_DIR;
  check(typeof release === "string" && RELEASE_PATTERN.test(release), "INVALID_RELEASE_DIRECTORY");
  const evidenceDir = process.env.VISION_EVIDENCE_DIR;
  check(typeof evidenceDir === "string" && path.isAbsolute(evidenceDir), "EVIDENCE_DIRECTORY_MUST_BE_ABSOLUTE");
  check(process.env.PLAYWRIGHT_MODULE_PATH && path.isAbsolute(process.env.PLAYWRIGHT_MODULE_PATH), "EXISTING_PLAYWRIGHT_PATH_REQUIRED");
  // Prevent inherited Playwright diagnostics from logging protocol/session data.
  process.env.DEBUG = ""; process.env.PWDEBUG = "";
  const start = Date.now(), deadline = start + 15 * 60 * 1000;
  const report = { schema_version: 1, target: TARGET, release_sha: release.split("/").at(-1), started_at: new Date(start).toISOString(), auth: "real_remote_private_env_login_in_memory", interception: false, checks: [], created_resources: {}, writes: { review_uploads: 0, model_tasks: 0, history_comparisons: 0, retrieval_media: 0, retrieval_tasks: 0 }, browser: { page_errors: 0, console_errors: 0, failed_media_requests: 0, failed_media_http_responses: 0 }, accuracy: "not_evaluable_without_independent_object_labels" };
  let browser, context, token, input, deadlineTimer;
  const stage = async (name, action) => {
    const before = Date.now();
    try { const outcome = await action(); report.checks.push({ name, outcome: "passed", elapsed_ms: Date.now() - before, ...outcome }); }
    catch (error) { report.checks.push({ name, outcome: "failed", code: error instanceof CheckFailure ? error.code : "RUNNER_ASSERTION_FAILED", elapsed_ms: Date.now() - before }); }
  };
  const waitUntil = async (read, predicate, timeoutMs = 180000) => {
    const end = Math.min(deadline, Date.now() + timeoutMs);
    let value;
    while (Date.now() < end) { value = await read(); if (predicate(value)) return value; await delay(1500); }
    throw new CheckFailure("BOUNDED_WAIT_EXHAUSTED");
  };
  try {
    input = await fixtures(); report.inputs = input.metadata;
    try {
      const command = "python3 -c " + quote(LOGIN_PROGRAM) + " " + quote(release);
      token = JSON.parse(execFileSync("ssh", ["-o", "BatchMode=yes", "-o", "ConnectTimeout=10", "ai-lingxi", command], { stdio: ["ignore", "pipe", "pipe"], timeout: 30000, maxBuffer: 65536 }).toString()).access_token;
      check(typeof token === "string" && token.length >= 16, "AUTHENTICATION_FAILED");
    } catch { throw new CheckFailure("AUTHENTICATION_FAILED"); }
    const { chromium } = await import(pathToFileURL(process.env.PLAYWRIGHT_MODULE_PATH).href);
    browser = await chromium.launch({ channel: "chrome", headless: true });
    deadlineTimer = setTimeout(() => void browser.close().catch(() => undefined), Math.max(1, deadline - Date.now()));
    context = await browser.newContext({ viewport: { width: 1440, height: 1000 } });
    await context.addInitScript((realToken) => { sessionStorage.setItem("review-annotation-token", realToken); }, token);
    const page = await context.newPage(); page.setDefaultTimeout(15000);
    page.on("pageerror", () => report.browser.page_errors++);
    page.on("console", (entry) => { if (entry.type() === "error") report.browser.console_errors++; });
    page.on("requestfailed", (request) => { if (["image", "media"].includes(request.resourceType())) report.browser.failed_media_requests++; });
    page.on("response", (response) => { if (response.status() >= 400 && ["image", "media"].includes(response.request().resourceType())) report.browser.failed_media_http_responses++; });
    page.on("request", (request) => {
      if (request.method() !== "POST") return;
      const pathname = new URL(request.url()).pathname;
      const category = { "/api/v1/imports/uploads": "review_uploads", "/api/v1/model-runs": "model_tasks", "/api/v1/history-comparisons": "history_comparisons", "/api/v1/retrieval/media": "retrieval_media", "/api/v1/retrieval/tasks": "retrieval_tasks" }[pathname];
      if (category) report.writes[category]++;
    });
    const api = async (pathname) => {
      const response = await context.request.get(TARGET + "/api/v1" + pathname, { headers: { Authorization: `Bearer ${token}` }, timeout: 20000 });
      check(response.ok(), `READ_HTTP_${response.status()}`); return response.json();
    };
    const created = (pathname) => page.waitForResponse((response) => response.request().method() === "POST" && new URL(response.url()).pathname === "/api/v1" + pathname, { timeout: 130000 });
    const loadedImages = async (locator, expected) => {
      await waitUntil(() => locator.count(), (count) => count === expected, 20000);
      for (const element of await locator.all()) {
        await element.scrollIntoViewIfNeeded();
        await waitUntil(() => element.evaluate((image) => image.complete && image.naturalWidth > 0 && image.naturalHeight > 0), Boolean, 20000);
      }
    };
    const screenshot = async (name) => {
      if (process.env.VISION_LIVE_SCREENSHOTS === "1" && !process.env.VISION_LIVE_IMAGE && await page.locator(".notice.error:visible,.form-error:visible,.history-warning:visible").count() === 0) {
        await mkdir(evidenceDir, { recursive: true, mode: 0o700 });
        await page.screenshot({ path: path.join(evidenceDir, `${name}-${start}.png`), fullPage: true });
      }
    };
    await stage("real_login_and_three_tabs", async () => {
      await page.goto(TARGET, { waitUntil: "domcontentloaded", timeout: 30000 });
      for (const name of ["人工复核", "历史对比", "智能检索"]) check(await page.getByRole("tab", { name, exact: true }).count() === 1, "REQUIRED_TAB_MISSING");
      check(await page.getByRole("heading", { name: "进入标注工作台" }).count() === 0, "REAL_LOGIN_NOT_ACCEPTED");
      return { tab_count: 3 };
    });
    await stage("image_review_real_model_job", async () => {
      await page.getByRole("tab", { name: "人工复核" }).click();
      const profiles = await api("/model-profiles");
      const profile = profiles.profiles.find((item) => item.adapter_type === "legacy_review_agent");
      check(profile && identifier(profile.model_profile_id), "REAL_LEGACY_MODEL_PROFILE_REQUIRED");
      const uploaded = created("/imports/uploads");
      await page.getByLabel("上传复核图片").setInputFiles(input.reviewImage);
      const response = await uploaded; check(response.ok(), `REVIEW_UPLOAD_HTTP_${response.status()}`);
      const imported = await response.json(); check(imported.failed === 0 && imported.imported + imported.skipped === 1, "REVIEW_UPLOAD_NOT_ACCEPTED");
      let scope = await api("/samples?status_filter=imported&limit=100&offset=0");
      let filter = "imported";
      if (scope.samples.length === 0) { scope = await api("/samples?status_filter=all&limit=100&offset=0"); filter = "all"; }
      const sourceSha = HASH(await readFile(input.reviewImage));
      check(scope.pagination.total >= 1 && scope.pagination.total <= 3 && scope.samples.length === scope.pagination.total, "REVIEW_BATCH_SCOPE_EXCEEDS_BUDGET");
      check(scope.samples.every((sample) => sample.assets.some((asset) => asset.kind === "original" && asset.sha256 === sourceSha)), "REVIEW_SCOPE_NOT_ISOLATED");
      await page.getByLabel("按状态筛选").selectOption(filter);
      await page.getByLabel("模型配置").selectOption(profile.model_profile_id);
      await waitUntil(() => page.getByRole("button", { name: "批量识别", exact: true }).isEnabled(), Boolean, 20000);
      await page.locator(".queue-item").first().click();
      await loadedImages(page.getByRole("img", { name: "待复核原图", exact: true }), 1);
      const submitted = created("/model-runs"); await page.getByRole("button", { name: "批量识别", exact: true }).click();
      const submission = await submitted; check(submission.ok(), `MODEL_SUBMIT_HTTP_${submission.status()}`);
      const payload = submission.request().postDataJSON(); assert.deepEqual([...payload.sample_ids].sort(), scope.samples.map((item) => item.sample_id).sort());
      const run = await submission.json(); check(identifier(run.job_id) && identifier(run.run_id), "MODEL_JOB_ID_MISSING");
      report.created_resources.review_job_id = run.job_id; report.created_resources.review_run_id = run.run_id;
      const job = await waitUntil(() => api(`/jobs/${run.job_id}`), (value) => TERMINAL_JOB.has(value.status));
      const detail = await api(`/model-runs/${run.run_id}`);
      const notice = page.getByRole("status", { name: "批量任务", exact: true });
      await waitUntil(() => notice.textContent(), (text) => typeof text === "string" && text.includes(`${job.completed}/${job.total}`) && (job.failed ? /失败/.test(text) : /完成/.test(text)), 10000);
      await screenshot("review-terminal");
      return { outcome: ["failed", "error", "partial_failed"].includes(job.status) ? "failed" : "passed", code: enumValue(job.status), job_id: run.job_id, run_id: run.run_id, sample_count: scope.samples.length, processed: job.completed, failed: job.failed, api_candidate_count: detail.candidate_count, visible_terminal_status: true, profile_adapter: "legacy_review_agent", original_image_loaded: true };
    });
    await stage("history_save_reload_images", async () => {
      await page.getByRole("tab", { name: "历史对比" }).click();
      await page.getByLabel("选择历史照片").setInputFiles(input.baseline); await page.getByLabel("选择当前照片").setInputFiles(input.current);
      const pending = created("/history-comparisons"); await page.getByRole("button", { name: "开始对比", exact: true }).click();
      const response = await pending; check(response.ok(), `HISTORY_CREATE_HTTP_${response.status()}`);
      const comparison = await response.json(); check(identifier(comparison.comparison_id), "HISTORY_ID_MISSING");
      report.created_resources.history_comparison_id = comparison.comparison_id;
      check(comparison.result.status === "matched" && comparison.result.change_count >= 1, "SYNTHETIC_CHANGE_NOT_FOUND");
      await page.reload({ waitUntil: "domcontentloaded" });
      const reopened = page.waitForResponse((item) => item.request().method() === "GET" && new URL(item.url()).pathname === `/api/v1/history-comparisons/${comparison.comparison_id}`);
      await page.locator(".history-saved-list > button").first().click();
      check((await reopened).ok(), "HISTORY_REOPEN_FAILED");
      await loadedImages(page.getByRole("img", { name: "历史照片", exact: true }), 1); await loadedImages(page.getByRole("img", { name: "当前照片", exact: true }), 1);
      check(report.writes.history_comparisons === 1, "HISTORY_REFRESH_REPOSTED"); await screenshot("history-reopened");
      return { outcome: comparison.result.semantic?.error ? "failed" : "passed", code: comparison.result.semantic?.error ? "HISTORY_SEMANTIC_FAILED" : "saved_reopened", comparison_id: comparison.comparison_id, changes: comparison.result.change_count, semantic_status: enumValue(comparison.result.semantic?.status), persisted_images_loaded: 2, create_requests: report.writes.history_comparisons };
    });
    await stage("video_retrieval_real_task_reload", async () => {
      await page.getByRole("tab", { name: "智能检索" }).click();
      check((await api("/capabilities")).retrieval_enabled === true, "RETRIEVAL_NOT_ENABLED");
      await page.getByLabel("检索目标").fill("person");
      const upload = created("/retrieval/media"); await page.getByLabel("上传图片或视频").setInputFiles(input.video);
      check((await upload).ok(), "VIDEO_UPLOAD_FAILED");
      const pending = created("/retrieval/tasks"); await page.getByRole("button", { name: "开始检索", exact: true }).click();
      const response = await pending; check(response.ok(), `RETRIEVAL_SUBMIT_HTTP_${response.status()}`);
      const creation = await response.json(); check(identifier(creation.task_id), "RETRIEVAL_TASK_ID_MISSING");
      report.created_resources.retrieval_task_id = creation.task_id;
      let syncFailures = 0;
      const task = await waitUntil(async () => { const current = await api(`/retrieval/tasks/${creation.task_id}`); syncFailures = current.sync_error ? syncFailures + 1 : 0; check(syncFailures < 3, "RETRIEVAL_UPSTREAM_UNAVAILABLE"); return current; }, (value) => TERMINAL_RETRIEVAL.has(value.status));
      await page.getByRole("button", { name: "刷新任务状态" }).click();
      const resultCount = task.results?.length ?? 0;
      if (resultCount) await loadedImages(page.locator('img[alt^="命中画面 "]'), resultCount);
      if (task.status === "completed" && task.total_hits === 0) await page.getByText("检索完成，未发现符合条件的目标。", { exact: true }).waitFor();
      if (task.status === "failed") await page.getByText(/^检索失败：/).waitFor();
      await page.reload({ waitUntil: "domcontentloaded" });
      const reopened = page.waitForResponse((item) => item.request().method() === "GET" && new URL(item.url()).pathname === `/api/v1/retrieval/tasks/${creation.task_id}`);
      await page.locator(".retrieval-task-row").first().click(); check((await reopened).ok(), "RETRIEVAL_REOPEN_FAILED");
      if (resultCount) await loadedImages(page.locator('img[alt^="命中画面 "]'), resultCount);
      check(report.writes.retrieval_tasks === 1, "RETRIEVAL_REFRESH_REPOSTED"); await screenshot("retrieval-reopened");
      return { outcome: task.status === "completed" ? "passed" : "failed", code: enumValue(task.status), task_id: creation.task_id, hits: task.total_hits, result_images_loaded: resultCount, failed_files: task.file_errors?.length ?? 0, create_requests: report.writes.retrieval_tasks, source: input.metadata.video_source, zero_hits_is_accuracy_claim: false };
    });
    await stage("browser_errors_and_write_budget", async () => {
      check(Object.values(report.browser).every((count) => count === 0), "BROWSER_OR_MEDIA_ERROR");
      check(report.writes.model_tasks <= 1 && report.writes.retrieval_tasks <= 1 && report.writes.model_tasks + report.writes.retrieval_tasks <= 2, "MODEL_TASK_BUDGET_EXCEEDED");
      check(report.writes.history_comparisons <= 1 && report.writes.retrieval_media <= 1, "AUXILIARY_WRITE_BUDGET_EXCEEDED");
      return { task_count: report.writes.model_tasks + report.writes.retrieval_tasks };
    });
  } catch (error) {
    report.checks.push({ name: "runner_setup", outcome: "failed", code: error instanceof CheckFailure ? error.code : "SETUP_FAILED" });
  } finally {
    clearTimeout(deadlineTimer); await context?.close().catch(() => undefined); await browser?.close().catch(() => undefined);
    if (input) await rm(input.directory, { recursive: true, force: true });
    report.finished_at = new Date().toISOString(); report.elapsed_ms = Date.now() - start;
    report.status = report.checks.length === 5 && report.checks.every((item) => item.outcome === "passed") ? "passed" : "incomplete_or_failed";
    const serialized = JSON.stringify(report, null, 2) + "\n";
    check(!token || !serialized.includes(token), "SENSITIVE_EVIDENCE_BLOCKED");
    token = undefined;
    await mkdir(evidenceDir, { recursive: true, mode: 0o700 });
    const reportPath = path.join(evidenceDir, `live-workbench-${start}.json`);
    await writeFile(reportPath, serialized, { mode: 0o600, flag: "wx" });
    process.stdout.write(JSON.stringify({ status: report.status, checks: report.checks.length, report_path: reportPath }) + "\n");
    if (report.status !== "passed") process.exitCode = 1;
  }
}

main().catch((error) => {
  process.stderr.write(JSON.stringify({ status: "runner_failed", code: error instanceof CheckFailure ? error.code : "UNEXPECTED_RUNNER_FAILURE" }) + "\n");
  process.exitCode = 1;
});
