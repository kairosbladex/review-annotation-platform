import { LoaderCircle, RefreshCw, Search, UploadCloud, X } from "lucide-react";
import { useCallback, useEffect, useRef, useState } from "react";
import { cancelRetrievalTask, createRetrievalTask, getCapabilities, getRetrievalTask, listRetrievalTasks, uploadRetrievalMedia } from "../api";
import type { RetrievalMedia, RetrievalResult, RetrievalTask } from "../api";
import { RecordPagination } from "./RecordPagination";

const TERMINAL = new Set(["completed", "partial", "failed", "cancelled", "submission_unknown"]);
const STATUS: Record<string, string> = { pending: "排队中", submitting: "提交中", running: "分析中", stopping: "停止中", completed: "已完成", partial: "部分完成", failed: "检索失败", cancelled: "已取消", submission_unknown: "提交结果待确认" };
type Upload = { id: string; file: File; reference: boolean; state: "uploading" | "ready" | "failed"; media?: RetrievalMedia; error?: string };
const message = (reason: unknown, fallback: string) => reason instanceof Error ? reason.message : fallback;
const finite = (value: unknown): value is number => typeof value === "number" && Number.isFinite(value);
const fileLabel = (index: unknown) => finite(index) && Number.isInteger(index) && index >= 0 ? `文件 ${index + 1}` : "来源文件未知";
function progressValue(progress: RetrievalTask["video_progress"]): number | null {
  if (finite(progress)) return Math.min(1, Math.max(0, progress));
  if (typeof progress !== "object") return null;
  if (!progress) return null;
  if (finite(progress.progress)) return Math.min(1, Math.max(0, progress.progress > 1 ? progress.progress / 100 : progress.progress));
  if (finite(progress.duration_seconds) && progress.duration_seconds > 0 && finite(progress.current_seconds)) return Math.min(1, Math.max(0, progress.current_seconds / progress.duration_seconds));
  const total = progress.sample_count || progress.estimated_total_samples || progress.total_frames;
  const done = progress.processed_samples ?? progress.processed_frames;
  return finite(total) && total > 0 && finite(done) ? Math.min(1, Math.max(0, done / total)) : null;
}

function HitCard({ result, index }: { result: RetrievalResult; index: number }) {
  const [size, setSize] = useState({ width: 0, height: 0 });
  return <article className="retrieval-hit"><div className="retrieval-image-wrap">
    {result.image_url || result.thumbnail_url ? <img src={result.image_url || result.thumbnail_url || ""} alt={`命中画面 ${index + 1}`} loading="lazy" onLoad={(event) => setSize({ width: event.currentTarget.naturalWidth, height: event.currentTarget.naturalHeight })} /> : <p>该命中暂无可用画面</p>}
    {size.width > 0 && <svg viewBox={`0 0 ${size.width} ${size.height}`} aria-label={`命中画面 ${index + 1} 检测框`} className="retrieval-boxes">{result.detections.map((detection, n) => <rect key={n} x={detection.bbox[0]} y={detection.bbox[1]} width={Math.max(0, detection.bbox[2] - detection.bbox[0])} height={Math.max(0, detection.bbox[3] - detection.bbox[1])} vectorEffect="non-scaling-stroke"><title>{detection.class_name}</title></rect>)}</svg>}
  </div><strong>{fileLabel(result.file_index)}{finite(result.timestamp_seconds) ? ` · ${result.timestamp_seconds.toFixed(1)} 秒` : ""}{finite(result.frame_index) ? ` · 帧 ${result.frame_index}` : ""}</strong>
    <ul>{result.detections.map((detection, n) => <li key={n}>{detection.class_name} · {(detection.confidence * 100).toFixed(1)}% · 框 [{detection.bbox.join(", ")}]</li>)}</ul>
  </article>;
}

export function IntelligentRetrieval({ token }: { token: string }) {
  const [enabled, setEnabled] = useState<boolean | null>(null);
  const [capabilityError, setCapabilityError] = useState("");
  const [capabilityEpoch, setCapabilityEpoch] = useState(0);
  const [query, setQuery] = useState("");
  const [uploads, setUploads] = useState<Upload[]>([]);
  const [tasks, setTasks] = useState<RetrievalTask[]>([]);
  const [total, setTotal] = useState(0);
  const [taskPage, setTaskPage] = useState(1);
  const [listError, setListError] = useState("");
  const [listBusy, setListBusy] = useState(false);
  const [selectedId, setSelectedId] = useState<string | null>(null);
  const [task, setTask] = useState<RetrievalTask | null>(null);
  const [detailBusy, setDetailBusy] = useState(false);
  const [connectionError, setConnectionError] = useState("");
  const [readEpoch, setReadEpoch] = useState(0);
  const [submitting, setSubmitting] = useState(false);
  const [submitError, setSubmitError] = useState("");
  const [submissionUnknown, setSubmissionUnknown] = useState(false);
  const [cancelConfirm, setCancelConfirm] = useState(false);
  const [cancelling, setCancelling] = useState(false);
  const alive = useRef(false);
  const generation = useRef(0);
  const listRequest = useRef(0);
  const submissionLock = useRef(false);
  const controllers = useRef(new Set<AbortController>());
  const selectedRef = useRef(selectedId); selectedRef.current = selectedId;
  useEffect(() => {
    alive.current = true;
    return () => { alive.current = false; generation.current++; listRequest.current++; controllers.current.forEach((controller) => controller.abort()); controllers.current.clear(); };
  }, []);

  useEffect(() => {
    const controller = new AbortController(); let active = true;
    setEnabled(null); setCapabilityError("");
    getCapabilities(token, controller.signal).then((data) => { if (active) setEnabled(data.retrieval_enabled === true); }).catch(() => { if (active) setCapabilityError("暂时无法读取服务状态，请重试。"); });
    return () => { active = false; controller.abort(); };
  }, [token, capabilityEpoch]);

  const refreshList = useCallback(async () => {
    const version = ++listRequest.current;
    const controller = new AbortController(); controllers.current.add(controller);
    const timeout = window.setTimeout(() => controller.abort(), 15000);
    setListBusy(true);
    try {
      const data = await listRetrievalTasks(token, controller.signal, (taskPage - 1) * 20);
      if (alive.current && version === listRequest.current) { setTasks(data.tasks); setTotal(data.total); setListError(""); const lastPage = Math.max(1, Math.ceil(data.total / 20)); if (taskPage > lastPage) setTaskPage(lastPage); }
    } catch { if (alive.current && version === listRequest.current) setListError("暂时无法读取任务列表，可手动刷新。"); }
    finally { window.clearTimeout(timeout); controllers.current.delete(controller); if (alive.current && version === listRequest.current) setListBusy(false); }
  }, [token, taskPage]);
  useEffect(() => { void refreshList(); }, [refreshList]);

  useEffect(() => {
    if (!selectedId) return;
    let active = true; let failures = 0; let timer: number | undefined; let current: AbortController | null = null;
    setConnectionError(""); setDetailBusy(true);
    function failedRead() {
      if (!active) return;
      failures++; setDetailBusy(false);
      setConnectionError(failures >= 3 ? "连续 3 次无法连接，已暂停自动刷新。可手动重新读取任务。" : `暂时连接错误，正在重新读取（${failures}/3）。`);
      if (failures < 3) timer = window.setTimeout(() => void poll(), failures * 1000);
    }
    async function poll() {
      const controller = new AbortController(); current = controller;
      const timeout = window.setTimeout(() => controller.abort(), 15000);
      try {
        const data = await getRetrievalTask(token, selectedId!, controller.signal);
        if (!active) return;
        setTask(data); setDetailBusy(false);
        setTasks((items) => items.map((item) => item.task_id === data.task_id ? { ...item, ...data } : item));
        if (data.sync_error && !TERMINAL.has(data.status)) { failedRead(); return; }
        failures = 0; setConnectionError("");
        if (!TERMINAL.has(data.status)) timer = window.setTimeout(() => void poll(), 2000);
      } catch {
        failedRead();
      } finally { window.clearTimeout(timeout); }
    }
    void poll();
    return () => { active = false; current?.abort(); window.clearTimeout(timer); };
  }, [selectedId, readEpoch, token]);

  async function uploadOne(item: Upload) {
    const version = generation.current; const controller = new AbortController(); controllers.current.add(controller);
    const timeout = window.setTimeout(() => controller.abort(), 120000);
    setUploads((items) => items.map((entry) => entry.id === item.id ? { ...entry, state: "uploading", error: undefined } : entry));
    try {
      const media = await uploadRetrievalMedia(token, item.file, controller.signal);
      if (alive.current && version === generation.current) setUploads((items) => items.map((entry) => entry.id === item.id ? { ...entry, media, state: "ready" } : entry));
    } catch (reason) {
      if (alive.current && version === generation.current) setUploads((items) => items.map((entry) => entry.id === item.id ? { ...entry, state: "failed", error: message(reason, "上传失败") } : entry));
    } finally { window.clearTimeout(timeout); controllers.current.delete(controller); }
  }
  async function addFiles(files: File[], reference = false) {
    if (!enabled || submitting || submissionUnknown) return;
    const maximum = reference ? 4 : 20;
    if (uploads.filter((item) => item.reference === reference).length + files.length > maximum) { setSubmitError(reference ? "最多选择 4 张目标参考图。" : "一次最多选择 20 个文件。"); return; }
    const supported = reference ? /^image\/(jpeg|png|webp)$/ : /^(image\/(jpeg|png|webp)|video\/(mp4|quicktime|webm))$/;
    if (files.some((file) => file.size > 256 * 1024 * 1024 || !supported.test(file.type))) { setSubmitError(reference ? "参考图仅支持 JPG、PNG、WebP，单个文件不超过 256MB。" : "请选择 JPG、PNG、WebP 图片或 MP4、MOV、WebM 视频，单个文件不超过 256MB。"); return; }
    setSubmitError("");
    const entries: Upload[] = files.map((file) => ({ id: crypto.randomUUID(), file, reference, state: "uploading" }));
    setUploads((items) => [...items, ...entries]);
    for (const item of entries) { if (!alive.current) break; await uploadOne(item); }
  }
  function selectTask(id: string) { setTask(null); setSelectedId(id); setReadEpoch((value) => value + 1); setCancelConfirm(false); }
  async function submit() {
    if (submissionLock.current || !enabled || submissionUnknown || (!query.trim() && !uploads.some((item) => item.reference)) || !uploads.some((item) => !item.reference) || uploads.some((item) => item.state !== "ready")) return;
    submissionLock.current = true; setSubmitting(true); setSubmitError("");
    const version = generation.current; const controller = new AbortController(); controllers.current.add(controller);
    const timeout = window.setTimeout(() => controller.abort(), 30000);
    const key = crypto.randomUUID();
    try {
      const data = await createRetrievalTask(token, { query: query.trim(), media_ids: uploads.filter((item) => !item.reference).map((item) => item.media!.media_id), prompt_media_ids: uploads.filter((item) => item.reference).map((item) => item.media!.media_id), confidence: 0.2, sample_interval_seconds: 3 }, key, controller.signal);
      if (!alive.current || version !== generation.current) return;
      setTask(data); setSelectedId(data.task_id); setReadEpoch((value) => value + 1); if (taskPage === 1) void refreshList(); else setTaskPage(1);
    } catch {
      if (alive.current && version === generation.current) { setSubmissionUnknown(true); setSubmitError("提交结果暂时未知，请刷新任务列表确认。页面不会自动重新提交。"); }
    } finally { window.clearTimeout(timeout); controllers.current.delete(controller); submissionLock.current = false; if (alive.current && version === generation.current) setSubmitting(false); }
  }
  async function cancel() {
    if (!task || cancelling) return;
    const id = task.task_id; const controller = new AbortController(); controllers.current.add(controller);
    const timeout = window.setTimeout(() => controller.abort(), 15000);
    setCancelling(true); setCancelConfirm(false);
    try {
      const data = await cancelRetrievalTask(token, id, controller.signal);
      if (alive.current && selectedRef.current === id) { setTask(data); setReadEpoch((value) => value + 1); }
    } catch { if (alive.current && selectedRef.current === id) setConnectionError("取消请求暂时无法确认，请刷新任务状态后再操作。"); }
    finally { window.clearTimeout(timeout); controllers.current.delete(controller); if (alive.current) setCancelling(false); }
  }
  const ready = enabled === true && !submitting && !submissionUnknown;
  const canSubmit = ready && (query.trim() || uploads.some((item) => item.reference)) && uploads.some((item) => !item.reference) && uploads.every((item) => item.state === "ready");
  const progress = progressValue(task?.video_progress);
  const progressDetail = task?.video_progress && typeof task.video_progress === "object" ? task.video_progress : null;
  return <main className="workspace retrieval-workspace"><section className="workspace-heading"><div><span className="eyebrow">INTELLIGENT RETRIEVAL</span><h1>智能检索</h1><p>上传图片或视频，描述目标；任务保存后可随时回来查看。</p></div></section>
    {enabled === false && <div className="notice" role="status">智能检索尚未启用，请联系管理员完成服务配置。</div>}
    {enabled === null && !capabilityError && <div role="status">正在读取服务状态…</div>}
    {capabilityError && <div className="notice error" role="alert">{capabilityError}<button className="secondary-button" onClick={() => setCapabilityEpoch((value) => value + 1)}>重试服务状态</button></div>}
    <div className="retrieval-layout"><section className="panel retrieval-form"><h2>新建检索</h2><label>检索目标<input aria-label="检索目标" value={query} disabled={!ready} onChange={(event) => setQuery(event.target.value)} placeholder="例如：穿反光背心的人员、白色卡车" maxLength={256} /></label>
      <label className="secondary-button retrieval-upload">上传目标参考图（可选）<input aria-label="上传目标参考图" type="file" multiple accept="image/jpeg,image/png,image/webp" disabled={!ready} onChange={(event) => { void addFiles(Array.from(event.target.files ?? []), true); event.target.value = ""; }} /></label><p className="retrieval-help">支持以图找目标，最多 4 张参考图。有参考图时可不填写文字。</p>
      <label className="secondary-button retrieval-upload"><UploadCloud size={17} />上传图片或视频<input aria-label="上传图片或视频" type="file" multiple accept="image/jpeg,image/png,image/webp,video/mp4,video/quicktime,video/webm" disabled={!ready} onChange={(event) => { void addFiles(Array.from(event.target.files ?? [])); event.target.value = ""; }} /></label>
      <p className="retrieval-help">待检索媒体最多 20 个，单个不超过 256MB。视频每 3 秒取样，结果不代表视频中所有目标。</p>
      <ul className="retrieval-uploads">{uploads.map((item) => <li key={item.id}><div><strong>{item.file.name}</strong><span>{item.reference ? "目标参考图 · " : "待检索媒体 · "}{item.state === "ready" ? "上传完成" : item.state === "uploading" ? "上传中…" : item.error || "上传失败"}</span></div>{item.state === "failed" && <button className="ghost-button" disabled={!ready} onClick={() => void uploadOne(item)}>重试上传</button>}<button className="icon-button" aria-label={`移除 ${item.file.name}`} disabled={!ready || item.state === "uploading"} onClick={() => setUploads((items) => items.filter((entry) => entry.id !== item.id))}><X size={15} /></button></li>)}</ul>
      {submitError && <p className="form-error" role="alert">{submitError}</p>}
      <button className="primary-button wide" disabled={!canSubmit} onClick={() => void submit()}>{submitting ? <LoaderCircle size={16} className="spin" /> : <Search size={16} />}{submitting ? "正在提交…" : "开始检索"}</button>
    </section><section className="panel retrieval-tasks"><header><h2>检索任务 <small>{total}</small></h2><button className="secondary-button" disabled={listBusy} onClick={() => void refreshList()}><RefreshCw size={14} />刷新任务列表</button></header>{listError && <p className="form-error" role="alert">{listError}</p>}
      <div className="retrieval-task-list">{tasks.map((item) => <button className={`retrieval-task-row ${selectedId === item.task_id ? "selected" : ""}`} key={item.task_id} onClick={() => selectTask(item.task_id)}><strong>{item.query || "图片目标检索"}</strong><span>{STATUS[item.status] || item.status}</span><small>{new Date(item.created_at).toLocaleString("zh-CN")} · 命中 {item.total_hits}</small></button>)}{!listBusy && tasks.length === 0 && <p className="retrieval-help">还没有检索任务。上传媒体并输入目标后开始检索。</p>}</div>
      <RecordPagination label="检索任务" page={taskPage} total={total} busy={listBusy} onChange={setTaskPage} />
    </section></div>
    {(selectedId || task) && <section className="panel retrieval-detail" aria-live="polite"><header><h2>任务详情{task ? ` · ${STATUS[task.status] || task.status}` : ""}</h2><div className="history-actions"><button className="secondary-button" onClick={() => setReadEpoch((value) => value + 1)}>刷新任务状态</button>{task && !TERMINAL.has(task.status) && <button className="secondary-button" disabled={cancelling || task.status === "stopping"} onClick={() => setCancelConfirm(true)}>{cancelling ? "正在取消…" : "取消任务"}</button>}</div></header>
      {detailBusy && !task && <p>正在读取任务…</p>}{connectionError && <p className="form-error" role="alert">{connectionError}</p>}
      {task && <><div className="retrieval-task-summary"><strong>{STATUS[task.status] || task.status}</strong><span>{task.query || "图片目标检索"}</span><span>文件进度 {task.processed_count ?? 0}/{task.total_count ?? task.media_count}</span>{progress !== null && <span>视频进度 {Math.round(progress * 100)}%</span>}{finite(progressDetail?.current_seconds) && <span>已分析 {progressDetail.current_seconds.toFixed(1)} 秒</span>}{finite(progressDetail?.sampled_frames) && <span>已采样 {progressDetail.sampled_frames} 帧</span>}<span>命中 {task.total_hits}</span></div>
        {task.message && <p>{task.message}</p>}{task.status === "completed" && task.total_hits === 0 && <p className="retrieval-empty">检索完成，未发现符合条件的目标。</p>}{task.status === "cancelled" && <p>任务已取消，已产生的结果保留供查看。</p>}{task.status === "failed" && <p className="form-error">检索失败：{task.error_code || "请查看任务信息"}</p>}{task.status === "partial" && <p className="history-warning">部分文件分析失败，以下为已完成部分的结果。</p>}{task.status === "submission_unknown" && <p className="history-warning">提交状态尚未确认，请稍后手动读取；不要重复创建任务。</p>}
        {task.sync_error && <p className="history-warning">上次同步暂时失败：{task.sync_error}</p>}{task.file_errors?.map((error, index) => <p className="history-warning" key={index}>{fileLabel(error.file_index)}：{error.message || error.error || "分析失败"}</p>)}
        <div className="retrieval-results">{task.results?.map((result, index) => <HitCard key={`${result.file_index}-${result.frame_index ?? index}`} result={result} index={index} />)}</div></>}
    </section>}
    {cancelConfirm && <div className="retrieval-modal-backdrop"><section className="retrieval-modal" role="dialog" aria-modal="true" aria-labelledby="cancel-title"><h2 id="cancel-title">取消当前检索任务？</h2><p>停止后保留已分析的结果，剩余文件将不再继续处理。</p><div className="history-actions"><button className="secondary-button" onClick={() => setCancelConfirm(false)}>继续检索</button><button className="primary-button" onClick={() => void cancel()}>确认取消任务</button></div></section></div>}
  </main>;
}
