import assert from "node:assert/strict";
import { after, before, test } from "node:test";
import { pathToFileURL } from "node:url";

const playwrightPath = process.env.PLAYWRIGHT_MODULE_PATH;
if (!playwrightPath) throw new Error("Set PLAYWRIGHT_MODULE_PATH to the existing playwright/index.mjs");
const { chromium } = await import(pathToFileURL(playwrightPath).href);
const baseURL = process.env.FRONTEND_TEST_URL ?? "http://127.0.0.1:5178";
let browser;
before(async () => { browser = await chromium.launch({ channel: "chrome", headless: true }); });
after(async () => { await browser?.close(); });

const task = (extra = {}) => ({ task_id: "task-1", status: "running", query: "人员", message: "分析中", media_count: 1, total_hits: 0, processed_count: 0, total_count: 1, video_progress: { progress: 0.4, current_seconds: 4, duration_seconds: 10 }, error_code: null, sync_error: null, created_at: "2026-10-17T00:00:00Z", finished_at: null, results: [], file_errors: [], ...extra });
const comparison = { comparison_id: "compare-1", created_at: "2026-10-17T00:00:00Z", baseline_url: "/synthetic-baseline.png", current_url: "/synthetic-current.png", result: { status: "matched", polygons: [[[0.1, 0.1], [0.4, 0.1], [0.4, 0.4]]], change_count: 1, current_width: 10, current_height: 10, baseline_width: 10, baseline_height: 10, coordinate_space: "normalized_current", alignment: { method: "synthetic", inliers: 10, overlap: 0.8 }, diff_summary: "新增一处变化", semantic: { status: "skipped", summary: null, error: null, region_count: 1 } } };
async function setup(handler) {
  const context = await browser.newContext();
  await context.addInitScript(() => sessionStorage.setItem("review-annotation-token", "synthetic-test-session"));
  const page = await context.newPage();
  await page.route("**/synthetic-*.png", (route) => route.fulfill({ contentType: "image/svg+xml", body: '<svg xmlns="http://www.w3.org/2000/svg" width="640" height="360"><rect width="640" height="360" fill="#243746"/><rect x="230" y="70" width="70" height="190" rx="20" fill="#9db9bc"/><text x="20" y="330" fill="#d3e0e0" font-size="18">Synthetic browser fixture</text></svg>' }));
  await page.route("**/api/v1/**", async (route) => {
    const url = new URL(route.request().url());
    const custom = await handler?.(route, url);
    if (custom) return;
    let body = {};
    if (url.pathname.endsWith("/samples")) body = { samples: [], pagination: { total: 0, offset: 0, limit: 100 } };
    else if (url.pathname.endsWith("/model-profiles")) body = { profiles: [] };
    else if (url.pathname.endsWith("/capabilities")) body = { retrieval_enabled: true };
    else if (url.pathname.endsWith("/retrieval/tasks")) body = { tasks: [], total: 0 };
    else if (url.pathname.endsWith("/history-comparisons")) body = { comparisons: [], total: 0 };
    await route.fulfill({ json: body });
  });
  await page.goto(baseURL);
  return { page, context };
}
const visible = (page, text) => page.getByText(text, { exact: true }).first().waitFor({ timeout: 4000 });

test("retrieval uploads media, creates once, shows terminal results and survives refresh", async () => {
  let current = task(); let creates = 0; let reads = 0; let key;
  const { page, context } = await setup(async (route, url) => {
    if (url.pathname.endsWith("/retrieval/media")) { await route.fulfill({ json: { media_id: "media-1", filename: "clip.mp4", media_type: "video", size_bytes: 3 } }); return true; }
    if (url.pathname.endsWith("/retrieval/tasks") && route.request().method() === "POST") {
      creates++; key = route.request().headers()["idempotency-key"];
      assert.deepEqual(route.request().postDataJSON().media_ids, ["media-1"]);
      await route.fulfill({ json: current }); return true;
    }
    if (url.pathname.endsWith("/retrieval/tasks")) { await route.fulfill({ json: { tasks: creates ? [current] : [], total: creates } }); return true; }
    if (url.pathname.endsWith("/retrieval/tasks/task-1")) { reads++; await route.fulfill({ json: current }); return true; }
  });
  try {
    await page.getByRole("tab", { name: "智能检索" }).click({ timeout: 4000 });
    await page.getByLabel("检索目标").fill("人员");
    await page.getByLabel("上传图片或视频").setInputFiles({ name: "clip.mp4", mimeType: "video/mp4", buffer: Buffer.from("abc") });
    await visible(page, "clip.mp4");
    await page.getByRole("button", { name: "开始检索", exact: true }).click();
    await visible(page, "视频进度 40%");
    assert.equal(creates, 1); assert.match(key, /^[0-9a-f-]{36}$/);
    current = task({ status: "completed", total_hits: 1, processed_count: 1, video_progress: 1, results: [{ file_index: 0, frame_index: 30, timestamp_seconds: 3, image_url: "/synthetic-hit.png", thumbnail_url: "/synthetic-hit.png", detections: [{ class_name: "person", confidence: 0.92, bbox: [1, 2, 8, 9] }] }] });
    await page.getByRole("button", { name: "刷新任务状态" }).click();
    await visible(page, "已完成");
    await visible(page, "person · 92.0% · 框 [1, 2, 8, 9]");
    assert.equal(await page.getByRole("img", { name: "命中画面 1", exact: true }).count(), 1);
    const terminalReads = reads; await page.waitForTimeout(2400); assert.equal(reads, terminalReads, "terminal tasks stop polling");
    await page.reload(); await page.getByRole("tab", { name: "智能检索" }).click();
    await page.getByRole("button", { name: /人员.*已完成/ }).click();
    await visible(page, "person · 92.0% · 框 [1, 2, 8, 9]");
    assert.equal(creates, 1);
    if (process.env.FRONTEND_SCREENSHOT_PATH) await page.screenshot({ path: process.env.FRONTEND_SCREENSHOT_PATH, fullPage: true });
  } finally { await context.close(); }
});

test("retrieval disabled capability prevents upload and submit", async () => {
  const { page, context } = await setup(async (route, url) => {
    if (url.pathname.endsWith("/capabilities")) { await route.fulfill({ json: { retrieval_enabled: false } }); return true; }
  });
  try {
    await page.getByRole("tab", { name: "智能检索" }).click({ timeout: 4000 });
    await visible(page, "智能检索尚未启用，请联系管理员完成服务配置。");
    assert.equal(await page.getByRole("button", { name: "开始检索", exact: true }).isDisabled(), true);
    assert.equal(await page.getByLabel("上传图片或视频").isDisabled(), true);
  } finally { await context.close(); }
});

test("history can reopen saved images and comparison without re-upload", async () => {
  const { page, context } = await setup(async (route, url) => {
    if (url.pathname.endsWith("/history-comparisons")) { await route.fulfill({ json: { comparisons: [{ comparison_id: "compare-1", created_at: comparison.created_at, status: "matched", change_count: 1, diff_summary: "新增一处变化" }], total: 1 } }); return true; }
    if (url.pathname.endsWith("/history-comparisons/compare-1")) { await route.fulfill({ json: comparison }); return true; }
  });
  try {
    await page.getByRole("tab", { name: "历史对比" }).click();
    await page.getByRole("button", { name: /新增一处变化/ }).click({ timeout: 4000 });
    await visible(page, "1 处变化");
    assert.equal(await page.getByRole("img", { name: "历史照片", exact: true }).getAttribute("src"), "/synthetic-baseline.png");
    assert.equal(await page.getByRole("img", { name: "当前照片", exact: true }).getAttribute("src"), "/synthetic-current.png");
  } finally { await context.close(); }
});

test("image reference allows visual retrieval without a text query", async () => {
  let media = 0; let payload;
  const { page, context } = await setup(async (route, url) => {
    if (url.pathname.endsWith("/retrieval/media")) { media++; await route.fulfill({ json: { media_id: `media-${media}`, filename: "image.png", media_type: "image", size_bytes: 3 } }); return true; }
    if (url.pathname.endsWith("/retrieval/tasks") && route.request().method() === "POST") { payload = route.request().postDataJSON(); await route.fulfill({ json: task({ status: "completed" }) }); return true; }
    if (url.pathname.endsWith("/retrieval/tasks/task-1")) { await route.fulfill({ json: task({ status: "completed" }) }); return true; }
  });
  try {
    await page.getByRole("tab", { name: "智能检索" }).click();
    await page.getByLabel("上传目标参考图").setInputFiles({ name: "reference.png", mimeType: "image/png", buffer: Buffer.from("abc") });
    await visible(page, "reference.png");
    await page.getByLabel("上传图片或视频").setInputFiles({ name: "scene.png", mimeType: "image/png", buffer: Buffer.from("abc") });
    await page.getByRole("button", { name: "开始检索", exact: true }).click({ timeout: 4000 });
    await visible(page, "检索完成，未发现符合条件的目标。");
    assert.deepEqual(payload.prompt_media_ids, ["media-1"]);
    assert.deepEqual(payload.media_ids, ["media-2"]);
    assert.equal(payload.query, "");
  } finally { await context.close(); }
});

test("cancel requires confirmation and preserves cancelled as distinct from no hits", async () => {
  let cancelled = false; let cancels = 0;
  const { page, context } = await setup(async (route, url) => {
    if (url.pathname.endsWith("/retrieval/tasks")) { await route.fulfill({ json: { tasks: [task()], total: 1 } }); return true; }
    if (url.pathname.endsWith("/cancel")) { cancels++; cancelled = true; await route.fulfill({ json: task({ status: "cancelled" }) }); return true; }
    if (url.pathname.endsWith("/retrieval/tasks/task-1")) { await route.fulfill({ json: task({ status: cancelled ? "cancelled" : "running" }) }); return true; }
  });
  try {
    await page.getByRole("tab", { name: "智能检索" }).click();
    await page.getByRole("button", { name: /人员.*分析中/ }).click();
    await page.getByRole("button", { name: "取消任务", exact: true }).click();
    await page.getByRole("dialog", { name: "取消当前检索任务？" }).waitFor(); assert.equal(cancels, 0);
    await page.getByRole("button", { name: "确认取消任务" }).click();
    await visible(page, "任务已取消，已产生的结果保留供查看。");
    assert.equal(cancels, 1); assert.equal(await page.getByText("检索完成，未发现符合条件的目标。", { exact: true }).count(), 0);
  } finally { await context.close(); }
});

test("temporary disconnect stops after three failed reads and manual reread never creates a task", async () => {
  let reads = 0; let posts = 0; let recover = false;
  const { page, context } = await setup(async (route, url) => {
    if (route.request().method() === "POST") posts++;
    if (url.pathname.endsWith("/retrieval/tasks")) { await route.fulfill({ json: { tasks: [task()], total: 1 } }); return true; }
    if (url.pathname.endsWith("/retrieval/tasks/task-1")) { reads++; if (recover) await route.fulfill({ json: task({ status: "completed" }) }); else await route.abort("failed"); return true; }
  });
  try {
    await page.getByRole("tab", { name: "智能检索" }).click();
    await page.getByRole("button", { name: /人员.*分析中/ }).click();
    await page.getByText("连续 3 次无法连接，已暂停自动刷新。可手动重新读取任务。", { exact: true }).waitFor({ timeout: 8000 });
    assert.equal(reads, 3); await page.waitForTimeout(2200); assert.equal(reads, 3);
    recover = true; await page.getByRole("button", { name: "刷新任务状态" }).click();
    await visible(page, "检索完成，未发现符合条件的目标。"); assert.equal(posts, 0);
  } finally { await context.close(); }
});

test("switching tasks discards a delayed response from the previously selected task", async () => {
  let release; const delayed = new Promise((resolve) => { release = resolve; });
  const { page, context } = await setup(async (route, url) => {
    if (url.pathname.endsWith("/retrieval/tasks")) { await route.fulfill({ json: { tasks: [task(), task({ task_id: "task-2", query: "车辆", status: "failed", error_code: "MODEL_UNAVAILABLE" })], total: 2 } }); return true; }
    if (url.pathname.endsWith("/retrieval/tasks/task-1")) { await delayed; await route.fulfill({ json: task({ query: "迟到旧任务", status: "completed", message: "旧任务迟到响应" }) }).catch(() => undefined); return true; }
    if (url.pathname.endsWith("/retrieval/tasks/task-2")) { await route.fulfill({ json: task({ task_id: "task-2", query: "车辆", status: "failed", error_code: "MODEL_UNAVAILABLE" }) }); return true; }
  });
  try {
    await page.getByRole("tab", { name: "智能检索" }).click();
    await page.getByRole("button", { name: /人员.*分析中/ }).click();
    await page.getByRole("button", { name: /车辆.*检索失败/ }).click();
    await visible(page, "检索失败：MODEL_UNAVAILABLE"); release(); await page.waitForTimeout(300);
    assert.equal(await page.getByText("旧任务迟到响应", { exact: true }).count(), 0);
    await page.getByRole("tab", { name: "人工复核" }).click();
    assert.equal(await page.getByRole("heading", { name: "存疑目标复核" }).count(), 1);
  } finally { release(); await context.close(); }
});

test("a lost submission response is not retried by refresh or manual status reads", async () => {
  let creates = 0;
  const { page, context } = await setup(async (route, url) => {
    if (url.pathname.endsWith("/retrieval/media")) { await route.fulfill({ json: { media_id: "media-1", filename: "clip.mp4", media_type: "video", size_bytes: 3 } }); return true; }
    if (url.pathname.endsWith("/retrieval/tasks") && route.request().method() === "POST") { creates++; await route.abort("failed"); return true; }
    if (url.pathname.endsWith("/retrieval/tasks")) { await route.fulfill({ json: { tasks: creates ? [task({ status: "completed" })] : [], total: creates } }); return true; }
    if (url.pathname.endsWith("/retrieval/tasks/task-1")) { await route.fulfill({ json: task({ status: "completed" }) }); return true; }
  });
  try {
    await page.getByRole("tab", { name: "智能检索" }).click();
    await page.getByLabel("检索目标").fill("人员");
    await page.getByLabel("上传图片或视频").setInputFiles({ name: "clip.mp4", mimeType: "video/mp4", buffer: Buffer.from("abc") });
    await page.getByRole("button", { name: "开始检索", exact: true }).click();
    await visible(page, "提交结果暂时未知，请刷新任务列表确认。页面不会自动重新提交。");
    assert.equal(await page.getByRole("button", { name: "开始检索", exact: true }).isDisabled(), true);
    await page.getByRole("button", { name: "刷新任务列表" }).click();
    await page.getByRole("button", { name: /人员.*已完成/ }).click();
    await visible(page, "检索完成，未发现符合条件的目标。");
    await page.getByRole("button", { name: "刷新任务状态" }).click();
    assert.equal(creates, 1);
  } finally { await context.close(); }
});

test("history uploads save a comparison and reload reopens it without another create", async () => {
  let creates = 0;
  const { page, context } = await setup(async (route, url) => {
    if (url.pathname.endsWith("/history-comparisons") && route.request().method() === "POST") { creates++; assert.match(route.request().headers()["content-type"], /multipart\/form-data/); await route.fulfill({ json: comparison }); return true; }
    if (url.pathname.endsWith("/history-comparisons")) { await route.fulfill({ json: { comparisons: creates ? [{ comparison_id: "compare-1", created_at: comparison.created_at, status: "matched", change_count: 1, diff_summary: "新增一处变化" }] : [], total: creates } }); return true; }
    if (url.pathname.endsWith("/history-comparisons/compare-1")) { await route.fulfill({ json: comparison }); return true; }
  });
  try {
    await page.getByRole("tab", { name: "历史对比" }).click();
    const png = Buffer.from("iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+aV2cAAAAASUVORK5CYII=", "base64");
    await page.getByLabel("选择历史照片").setInputFiles({ name: "baseline.png", mimeType: "image/png", buffer: png });
    await page.getByLabel("选择当前照片").setInputFiles({ name: "current.png", mimeType: "image/png", buffer: png });
    await page.getByRole("button", { name: "开始对比", exact: true }).click(); await visible(page, "1 处变化");
    await page.reload();
    await page.getByRole("button", { name: /新增一处变化/ }).click(); await visible(page, "1 处变化");
    assert.equal(creates, 1);
  } finally { await context.close(); }
});

test("mobile retrieval layout keeps upload, submit and task controls inside the viewport", async () => {
  const { page, context } = await setup();
  try {
    await page.setViewportSize({ width: 390, height: 844 });
    await page.getByRole("tab", { name: "智能检索" }).click();
    for (const locator of [page.getByLabel("检索目标"), page.getByLabel("上传目标参考图"), page.getByLabel("上传图片或视频"), page.getByRole("button", { name: "开始检索", exact: true }), page.getByRole("button", { name: "刷新任务列表" })]) {
      const box = await locator.boundingBox(); assert.ok(box && box.x >= 0 && box.x + box.width <= 390, "essential control stays inside mobile viewport");
    }
    assert.equal(await page.evaluate(() => document.documentElement.scrollWidth <= window.innerWidth), true);
  } finally { await context.close(); }
});

test("review cannot submit a recognition job before a model and samples are available", async () => {
  const { page, context } = await setup();
  try {
    await page.getByRole("heading", { name: "存疑目标复核" }).waitFor();
    assert.equal(await page.getByRole("button", { name: "批量识别", exact: true }).isDisabled(), true);
  } finally { await context.close(); }
});

for (const outcome of [
  { status: "succeeded", completed: 2, failed: 0, label: "已完成" },
  { status: "failed", completed: 0, failed: 2, label: "失败" },
  { status: "partial_failed", completed: 1, failed: 1, label: "部分失败" },
]) {
  test(`review batch notice replaces creation with ${outcome.status} and actual counts`, async () => {
    const sample = (id) => ({ sample_id: id, source_type: "upload", session_id: id, status: "imported", created_at: "2026-10-17T00:00:00Z", candidates: [], assets: [], runs: [], dataset_tags: [], counts: { total: 0, pending: 0, labeled: 0, true: 0, false: 0 } });
    let current = { job_id: "batch-job", status: "running", total: 2, completed: 0, failed: 0, error_summary: null };
    const { page, context } = await setup(async (route, url) => {
      if (url.pathname.endsWith("/samples")) { await route.fulfill({ json: { samples: [sample("sample-a"), sample("sample-b")], pagination: { total: 2, offset: 0, limit: 100 } } }); return true; }
      if (/\/samples\/sample-[ab]$/.test(url.pathname)) { await route.fulfill({ json: sample(url.pathname.split("/").at(-1)) }); return true; }
      if (url.pathname.endsWith("/model-profiles")) { await route.fulfill({ json: { profiles: [{ model_profile_id: "profile-1", name: "Synthetic model", model_version: "fixture" }] } }); return true; }
      if (url.pathname.endsWith("/model-runs")) { assert.deepEqual(route.request().postDataJSON().sample_ids, ["sample-a", "sample-b"]); await route.fulfill({ json: { run_id: "run-fixture", job_id: "batch-job", status: "pending" } }); return true; }
      if (url.pathname.endsWith("/jobs/batch-job")) { await route.fulfill({ json: current }); return true; }
    });
    try {
      await page.getByRole("button", { name: "批量识别", exact: true }).click({ timeout: 4000 });
      await page.getByText(/已创建批量识别任务/).waitFor({ timeout: 4000 });
      current = { ...current, ...outcome };
      const notice = page.getByRole("status", { name: "批量任务", exact: true }).filter({ hasText: `任务${outcome.label} · ${outcome.completed}/2 · 失败 ${outcome.failed}` });
      await notice.waitFor({ timeout: 5000 });
      assert.equal(await page.getByText(/已创建批量识别任务/).count(), 0);
    } finally { await context.close(); }
  });
}

test("review sends each uploaded image in a separate request and reports partial failures", async () => {
  const uploads = [];
  const { page, context } = await setup(async (route, url) => {
    if (url.pathname.endsWith("/imports/uploads")) {
      uploads.push(route.request().postDataBuffer().toString());
      if (uploads.length === 1) await route.fulfill({ json: { imported: 1, skipped: 0, failed: 0, errors: [] } });
      else await route.fulfill({ status: 413, json: { detail: "file too large" } });
      return true;
    }
  });
  try {
    const input = page.locator('input[type="file"]');
    await input.setInputFiles([{ name: "first.png", mimeType: "image/png", buffer: Buffer.from("abc") }, { name: "second.png", mimeType: "image/png", buffer: Buffer.from("def") }]);
    await page.getByText(/已导入 1 张，.*1 张失败/).waitFor({ timeout: 4000 });
    assert.equal(uploads.length, 2);
    assert.equal((uploads[0].match(/filename=/g) ?? []).length, 1);
    assert.equal((uploads[1].match(/filename=/g) ?? []).length, 1);
    assert.match(uploads[0], /first\.png/); assert.match(uploads[1], /second\.png/);
    await page.getByText(/HTTP 413/).first().waitFor();
  } finally { await context.close(); }
});

test("image hits with unknown file or null video fields stay readable and progress does not invent a percentage", async () => {
  const snapshot = task({ status: "partial", total_hits: 1, video_progress: { current_seconds: 9, sampled_frames: 4, phase: "inferencing" }, results: [{ file_index: null, frame_index: null, timestamp_seconds: null, image_url: "/synthetic-hit.png", detections: [{ class_name: "person", confidence: 0.92, bbox: [1, 2, 8, 9] }] }], file_errors: [{ file_index: null, error: "格式不支持" }] });
  const { page, context } = await setup(async (route, url) => {
    if (url.pathname.endsWith("/retrieval/tasks")) { await route.fulfill({ json: { tasks: [snapshot], total: 1 } }); return true; }
    if (url.pathname.endsWith("/retrieval/tasks/task-1")) { await route.fulfill({ json: snapshot }); return true; }
  });
  try {
    await page.getByRole("tab", { name: "智能检索" }).click();
    await page.getByRole("button", { name: /人员.*部分完成/ }).click();
    await visible(page, "来源文件未知");
    await visible(page, "来源文件未知：格式不支持");
    await visible(page, "已分析 9.0 秒"); await visible(page, "已采样 4 帧");
    assert.equal(await page.getByText(/视频进度.*%/).count(), 0);
    assert.equal(await page.getByLabel("检索目标").getAttribute("maxlength"), "256");
  } finally { await context.close(); }
});

test("upstream sync errors in successful HTTP snapshots also exhaust the bounded read budget", async () => {
  let reads = 0;
  const { page, context } = await setup(async (route, url) => {
    if (url.pathname.endsWith("/retrieval/tasks")) { await route.fulfill({ json: { tasks: [task()], total: 1 } }); return true; }
    if (url.pathname.endsWith("/retrieval/tasks/task-1")) { reads++; await route.fulfill({ json: task({ sync_error: "algorithm_unavailable" }) }); return true; }
  });
  try {
    await page.getByRole("tab", { name: "智能检索" }).click();
    await page.getByRole("button", { name: /人员.*分析中/ }).click();
    await page.getByText("连续 3 次无法连接，已暂停自动刷新。可手动重新读取任务。", { exact: true }).waitFor({ timeout: 8000 });
    assert.equal(reads, 3); await page.waitForTimeout(2200); assert.equal(reads, 3);
    await visible(page, "上次同步暂时失败：algorithm_unavailable");
  } finally { await context.close(); }
});

test("retrieval pagination requests older tasks and reopens the twenty-first record", async () => {
  const offsets = [];
  const older = task({ task_id: "older-task", query: "第21条旧任务", status: "completed", message: "已重开旧检索" });
  const { page, context } = await setup(async (route, url) => {
    if (url.pathname.endsWith("/retrieval/tasks")) { const offset = Number(url.searchParams.get("offset")); offsets.push(offset); await route.fulfill({ json: { tasks: offset === 20 ? [older] : [task()], total: 21 } }); return true; }
    if (url.pathname.endsWith("/retrieval/tasks/older-task")) { await route.fulfill({ json: older }); return true; }
  });
  try {
    await page.getByRole("tab", { name: "智能检索" }).click();
    await page.getByRole("button", { name: "检索任务下一页" }).click({ timeout: 4000 });
    await page.getByRole("button", { name: /第21条旧任务.*已完成/ }).click();
    await visible(page, "已重开旧检索"); assert.ok(offsets.includes(20));
    await visible(page, "第 2 / 2 页 · 共 21 条");
    await page.getByRole("button", { name: "检索任务上一页" }).click(); await page.getByRole("button", { name: /人员.*分析中/ }).waitFor();
  } finally { await context.close(); }
});

test("history pagination requests page two and reopens the twenty-first comparison", async () => {
  const pages = [];
  const { page, context } = await setup(async (route, url) => {
    if (url.pathname.endsWith("/history-comparisons")) { const index = Number(url.searchParams.get("page") ?? 1); pages.push(index); await route.fulfill({ json: { comparisons: index === 2 ? [{ comparison_id: "compare-1", created_at: comparison.created_at, status: "matched", change_count: 1, diff_summary: "第21条旧对比" }] : [], total: 21 } }); return true; }
    if (url.pathname.endsWith("/history-comparisons/compare-1")) { await route.fulfill({ json: comparison }); return true; }
  });
  try {
    await page.getByRole("tab", { name: "历史对比" }).click();
    await page.getByRole("button", { name: "对比记录下一页" }).click({ timeout: 4000 });
    await page.getByRole("button", { name: /第21条旧对比/ }).click(); await visible(page, "1 处变化");
    assert.ok(pages.includes(2)); await visible(page, "第 2 / 2 页 · 共 21 条");
    await page.getByRole("button", { name: "对比记录上一页" }).click(); await visible(page, "第 1 / 2 页 · 共 21 条");
  } finally { await context.close(); }
});
