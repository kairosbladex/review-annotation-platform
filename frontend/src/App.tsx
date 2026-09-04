import { Activity, Archive, ChevronDown, Database, Download, LoaderCircle, Play, RefreshCw, Settings2, ShieldCheck, Sparkles, UploadCloud } from "lucide-react";
import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { createAutoAnnotations, createExport, createModelRun, createProfile, getAssetUrl, getJob, getSample, labelCandidate, listProfiles, listSamples, login, uploadImages } from "./api";
import { Inspector } from "./components/Inspector";
import { QueuePanel } from "./components/QueuePanel";
import { ReviewCanvas } from "./components/ReviewCanvas";
import { advanceAfterLabel, keepQueueSelection, nextPendingCandidateId } from "./selection";
import type { AutoAnnotationJob, Candidate, Job, ModelProfile, Sample } from "./types";
import "./styles.css";

const TOKEN_KEY = "review-annotation-token";

const ACTIVE_JOB_STATUSES = new Set(["pending", "queued", "running", "processing", "in_progress"]);
const COMPLETED_JOB_STATUSES = new Set(["completed", "succeeded", "success"]);
const FAILED_JOB_STATUSES = new Set(["failed", "error", "partial_failed"]);
const QUEUE_PAGE_SIZE = 100;
const CLASS_FILTERS = ["", "三轮车", "两轮车", "人员", "光伏板", "光斑", "公交车", "卡车", "没戴头盔", "车", "违建", "车牌"];

function isDatasetCandidate(candidate: Candidate, datasetFilter: string) {
  return datasetFilter !== "v2" || candidate.dataset_tag === "v2";
}

function isActiveJobStatus(status: string) {
  return ACTIVE_JOB_STATUSES.has(status);
}

function isCompletedJobStatus(status: string) {
  return COMPLETED_JOB_STATUSES.has(status);
}

function isFailedJobStatus(status: string) {
  return FAILED_JOB_STATUSES.has(status);
}

function isTerminalJobStatus(status: string) {
  return isCompletedJobStatus(status) || isFailedJobStatus(status);
}

function autoAnnotationStatusLabel(status: string) {
  if (isActiveJobStatus(status)) return status === "pending" || status === "queued" ? "排队中" : "运行中";
  if (isCompletedJobStatus(status)) return "已完成";
  if (status === "partial_failed") return "部分失败";
  if (isFailedJobStatus(status)) return "失败";
  return "处理中";
}

function LoginScreen({ onLogin }: { onLogin: (username: string, password: string) => Promise<void> }) {
  const [username, setUsername] = useState("admin");
  const [password, setPassword] = useState("");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  return <main className="login-shell"><div className="login-card"><div className="brand-mark"><Database size={22} /></div><span className="eyebrow">REVIEW ANNOTATION PLATFORM</span><h1>进入标注工作台</h1><p>把存疑目标从模型输出中筛出，留给人工判断和下一轮训练。</p><form onSubmit={async (event) => { event.preventDefault(); setBusy(true); setError(""); try { await onLogin(username, password); } catch (reason) { setError(reason instanceof Error ? reason.message : "登录失败"); } finally { setBusy(false); } }}><label>管理员账号<input value={username} onChange={(event) => setUsername(event.target.value)} autoComplete="username" /></label><label>登录密码<input type="password" value={password} onChange={(event) => setPassword(event.target.value)} autoComplete="current-password" /></label>{error && <div className="form-error">{error}</div>}<button className="primary-button wide" disabled={busy}>{busy ? "正在验证…" : "登录工作台"}</button></form><div className="login-foot"><ShieldCheck size={15} />单管理员模式 · 操作自动写入审计日志</div></div></main>;
}

function App() {
  const [token, setToken] = useState(() => sessionStorage.getItem(TOKEN_KEY));
  const [samples, setSamples] = useState<Sample[]>([]);
  const [selectedId, setSelectedId] = useState<string | null>(null);
  const [sample, setSample] = useState<Sample | null>(null);
  const [statusFilter, setStatusFilter] = useState("needs_review");
  const [queuePage, setQueuePage] = useState(0);
  const [queueTotal, setQueueTotal] = useState(0);
  const [query, setQuery] = useState("");
  const [classFilter, setClassFilter] = useState("");
  const [datasetFilter, setDatasetFilter] = useState("all");
  const [imageUrl, setImageUrl] = useState<string | null>(null);
  const [profiles, setProfiles] = useState<ModelProfile[]>([]);
  const [profileId, setProfileId] = useState("");
  const [profileFormOpen, setProfileFormOpen] = useState(false);
  const [profileBaseUrl, setProfileBaseUrl] = useState("");
  const [selectedCandidateId, setSelectedCandidateId] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const [saving, setSaving] = useState(false);
  const [uploading, setUploading] = useState(false);
  const [actionMessage, setActionMessage] = useState("");
  const [error, setError] = useState("");
  const [job, setJob] = useState<Job | null>(null);
  const [autoAnnotationJob, setAutoAnnotationJob] = useState<AutoAnnotationJob | null>(null);
  const [autoAnnotationSubmitting, setAutoAnnotationSubmitting] = useState(false);
  const queueRequestRef = useRef(0);
  const detailRequestRef = useRef(0);

  const refreshQueue = useCallback(async (options?: { stealIfMissing?: boolean; page?: number }) => {
    if (!token) return;
    const requestId = ++queueRequestRef.current;
    const page = options?.page ?? queuePage;
    let result: Awaited<ReturnType<typeof listSamples>>;
    try {
      result = await listSamples(token, statusFilter, page * QUEUE_PAGE_SIZE, QUEUE_PAGE_SIZE, classFilter, datasetFilter);
    } catch (reason) {
      if (requestId === queueRequestRef.current) throw reason;
      return;
    }
    if (requestId !== queueRequestRef.current) return;
    setSamples(result.samples);
    setQueueTotal(result.pagination.total);
    if (options?.page !== undefined && options.page !== queuePage) setQueuePage(options.page);
    const lastPage = Math.max(0, Math.ceil(result.pagination.total / QUEUE_PAGE_SIZE) - 1);
    if (page > lastPage) {
      setQueuePage(lastPage);
      return;
    }
    setSelectedId((current) => keepQueueSelection(current, result.samples, { stealIfMissing: options?.stealIfMissing === true }));
  }, [classFilter, datasetFilter, queuePage, statusFilter, token]);

  const refreshDetail = useCallback(async (sampleId: string) => {
    if (!token) return;
    const requestId = ++detailRequestRef.current;
    setSample(null);
    setImageUrl(null);
    setSelectedCandidateId(null);
    try {
      const result = await getSample(token, sampleId);
      if (requestId !== detailRequestRef.current) return;
      setSample(result);
      setSelectedCandidateId((current) => {
        const candidates = result.candidates.filter((item) => isDatasetCandidate(item, datasetFilter));
        const currentItem = candidates.find((item) => item.candidate_id === current);
        if (currentItem && currentItem.annotation.label === null) return current;
        return nextPendingCandidateId(candidates, current) ?? candidates[0]?.candidate_id ?? null;
      });
      const original = result.assets.find((asset) => asset.kind === "original");
      if (original?.status === "ready") {
        try {
          const asset = await getAssetUrl(token, original.asset_id);
          if (requestId === detailRequestRef.current) setImageUrl(asset.url);
        } catch {
          if (requestId === detailRequestRef.current) setImageUrl(null);
        }
      }
    } catch (reason) {
      if (requestId === detailRequestRef.current) throw reason;
    }
  }, [datasetFilter, token]);

  useEffect(() => {
    if (!token || !selectedId) {
      detailRequestRef.current += 1;
      setSample(null);
      setImageUrl(null);
      setSelectedCandidateId(null);
      return;
    }
    refreshDetail(selectedId).catch((reason) => setError(reason instanceof Error ? reason.message : "读取样本失败"));
  }, [refreshDetail, selectedId, token]);

  useEffect(() => {
    if (!token) return;
    setLoading(true); setError("");
    Promise.all([refreshQueue({ stealIfMissing: true }), listProfiles(token).then((result) => { setProfiles(result.profiles); setProfileId((current) => current || result.profiles[0]?.model_profile_id || ""); })]).catch((reason) => setError(reason instanceof Error ? reason.message : "读取数据失败")).finally(() => setLoading(false));
  }, [token, statusFilter, refreshQueue]);

  const visibleCandidates = sample?.candidates.filter((item) => isDatasetCandidate(item, datasetFilter)) ?? [];
  const candidate = visibleCandidates.find((item) => item.candidate_id === selectedCandidateId) ?? null;
  const stats = useMemo(() => samples.reduce((acc, item) => {
    const candidates = datasetFilter === "v2" ? item.candidates.filter((candidateItem) => isDatasetCandidate(candidateItem, datasetFilter)) : item.candidates;
    return {
      pending: acc.pending + candidates.filter((candidateItem) => candidateItem.annotation.label === null).length,
      labeled: acc.labeled + candidates.filter((candidateItem) => candidateItem.annotation.label !== null).length,
      true: acc.true + candidates.filter((candidateItem) => candidateItem.annotation.label === "true").length,
      false: acc.false + candidates.filter((candidateItem) => candidateItem.annotation.label === "false").length,
    };
  }, { pending: 0, labeled: 0, true: 0, false: 0 }), [datasetFilter, samples]);
  const activeRun = sample?.runs.find((run) => run.annotation_task_id === sample.annotation_task_id) ?? sample?.runs[0];
  const autoAnnotationSampleIds = useMemo(() => {
    const normalizedQuery = query.trim().toLowerCase();
    return samples.filter((item) => {
      if (item.status !== "pending" && item.status !== "in_progress") return false;
      if (item.candidates.length === 0) return false;
      if (!normalizedQuery) return true;
      const text = `${item.session_id ?? ""} ${item.source_event_id ?? ""} ${item.rule_code ?? ""}`.toLowerCase();
      return text.includes(normalizedQuery);
    }).map((item) => item.sample_id);
  }, [query, samples]);
  const autoAnnotationRunning = autoAnnotationJob !== null && !isTerminalJobStatus(autoAnnotationJob.status);
  const autoAnnotationFailed = autoAnnotationJob !== null && isFailedJobStatus(autoAnnotationJob.status);
  const autoAnnotationTerminal = autoAnnotationJob !== null && isTerminalJobStatus(autoAnnotationJob.status);

  async function handleLogin(username: string, password: string) { const result = await login(username, password); sessionStorage.setItem(TOKEN_KEY, result.access_token); setToken(result.access_token); }
  function logout() { sessionStorage.removeItem(TOKEN_KEY); setToken(null); setSample(null); }
  const saveLabel = useCallback(async (label: "true" | "false" | null, note: string, options?: { advance?: boolean }) => {
    if (!token || !candidate || !sample) return;
    setSaving(true); setError("");
    try {
      const labeledId = candidate.candidate_id;
      const currentSampleId = sample.sample_id;
      const requestId = detailRequestRef.current;
      const result = await labelCandidate(token, labeledId, { label, note, expected_revision: candidate.annotation.revision });
      if (requestId !== detailRequestRef.current) return;
      const shouldAdvance = options?.advance !== false && label !== null;
      setSample(result.sample);
      if (!shouldAdvance) {
        setSelectedCandidateId(labeledId);
        await refreshQueue({ stealIfMissing: false });
        setActionMessage(label === null ? "已撤销标注" : "已保存备注");
        return;
      }
      let queueSamples = samples;
      const scopedCandidates = result.sample.candidates.filter((item) => isDatasetCandidate(item, datasetFilter));
      if (!nextPendingCandidateId(scopedCandidates, labeledId)) {
        const listed = await listSamples(token, statusFilter, 0, QUEUE_PAGE_SIZE, classFilter, datasetFilter);
        queueSamples = listed.samples;
      }
      if (datasetFilter === "v2") {
        queueSamples = queueSamples.map((item) => ({
          ...item,
          counts: { ...item.counts, pending: item.v2_pending_count },
        }));
      }
      const advance = advanceAfterLabel({
        candidates: scopedCandidates,
        labeledId,
        currentSampleId,
        queueSamples,
      });
      if (advance.kind === "next_box") {
        setSelectedCandidateId(advance.candidateId);
        await refreshQueue({ stealIfMissing: false });
        setActionMessage(`已标记，当前目标 ${result.sample.candidates.findIndex((item) => item.candidate_id === advance.candidateId) + 1}/${result.sample.candidates.length}`);
        return;
      }
      if (advance.kind === "next_sample") {
        setActionMessage("本图已标完，已打开下一张待复核");
        setSelectedId(advance.sampleId);
        await refreshQueue({ stealIfMissing: false, page: 0 });
        return;
      }
      await refreshQueue({ stealIfMissing: false });
      setActionMessage("本图所有目标已标完，队列中没有下一张待复核");
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : "保存标注失败");
    } finally {
      setSaving(false);
    }
  }, [candidate, classFilter, datasetFilter, refreshQueue, sample, samples, statusFilter, token]);

  useEffect(() => {
    const handler = (event: KeyboardEvent) => {
      if (!selectedCandidateId || event.target instanceof HTMLInputElement || event.target instanceof HTMLTextAreaElement) return;
      if (event.key !== "1" && event.key !== "2") return;
      const current = sample?.candidates.find((item) => item.candidate_id === selectedCandidateId);
      if (!current || saving) return;
      event.preventDefault();
      void saveLabel(event.key === "1" ? "true" : "false", current.annotation.note ?? "", { advance: true });
    };
    window.addEventListener("keydown", handler); return () => window.removeEventListener("keydown", handler);
  }, [sample, saveLabel, saving, selectedCandidateId]);
  async function handleUpload(files: File[]) { if (!token) return; setUploading(true); try { const result = await uploadImages(token, files); setActionMessage(`上传完成：${result.imported} 张图片进入队列`); setQueuePage(0); setStatusFilter("all"); await refreshQueue(); } catch (reason) { setError(reason instanceof Error ? reason.message : "上传失败"); } finally { setUploading(false); } }
  async function handleRun() { if (!token || !profileId || samples.length === 0) { setError("请先选择模型配置并确保队列中有样本"); return; } try { const result = await createModelRun(token, samples.map((item) => item.sample_id), profileId); setJob({ job_id: result.job_id, status: result.status, total: samples.length, completed: 0, failed: 0, error_summary: null }); setActionMessage(`已创建批量识别任务 ${result.run_id.slice(0, 8)}`); } catch (reason) { setError(reason instanceof Error ? reason.message : "创建模型任务失败"); } }
  const refreshAutoAnnotationData = useCallback(async (sampleId: string | null) => {
    await refreshQueue();
    if (sampleId) await refreshDetail(sampleId);
  }, [refreshDetail, refreshQueue]);
  async function handleAutoAnnotate() {
    if (!token) return;
    if (autoAnnotationRunning || autoAnnotationSubmitting) return;
    if (autoAnnotationSampleIds.length === 0) {
      setError("当前筛选队列没有可自动标注的待复核样本");
      return;
    }
    setAutoAnnotationSubmitting(true);
    setError("");
    try {
      const result = await createAutoAnnotations(token, autoAnnotationSampleIds);
      setAutoAnnotationJob(result);
      if (!isTerminalJobStatus(result.status)) {
        setActionMessage(`AI 自动标注已提交：${result.total} 个样本等待处理`);
      } else {
        await refreshAutoAnnotationData(selectedId);
        setActionMessage(isFailedJobStatus(result.status) ? "AI 自动标注任务失败，请检查任务详情" : `AI 自动标注已完成：${result.completed}/${result.total} 个样本`);
      }
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : "创建 AI 自动标注任务失败");
    } finally {
      setAutoAnnotationSubmitting(false);
    }
  }
  async function handleCreateProfile() { if (!token || !profileBaseUrl.trim()) { setError("请填写外部模型服务地址"); return; } try { const result = await createProfile(token, { profile_key: `http-${Date.now()}`, name: "外部模型服务", model_name: "review-detector", model_version: "v1", adapter_type: "http", config: { base_url: profileBaseUrl.trim(), endpoint: "/api/v1/review-agent/yolo-detect" } }); const refreshed = await listProfiles(token); setProfiles(refreshed.profiles); setProfileId(result.model_profile_id); setProfileFormOpen(false); setProfileBaseUrl(""); setActionMessage("模型配置已保存"); } catch (reason) { setError(reason instanceof Error ? reason.message : "保存模型配置失败"); } }
  async function handleExport() { if (!token || !activeRun) { setError("当前样本没有可导出的模型运行"); return; } try { const result = await createExport(token, activeRun.model_run_id); setJob({ job_id: result.job_id, status: result.status, total: 1, completed: 0, failed: 0, error_summary: null }); setActionMessage("导出任务已创建，完成后可在导出记录中下载"); } catch (reason) { setError(reason instanceof Error ? reason.message : "创建导出失败"); } }

  useEffect(() => { if (!token || !job || !["pending", "running"].includes(job.status)) return; const timer = window.setInterval(() => getJob(token, job.job_id).then(setJob).catch(() => undefined), 2000); return () => window.clearInterval(timer); }, [job, token]);

  useEffect(() => {
    if (!token || !autoAnnotationJob || isTerminalJobStatus(autoAnnotationJob.status)) return;
    let cancelled = false;
    const poll = async () => {
      try {
        const latest = await getJob(token, autoAnnotationJob.job_id);
        if (cancelled) return;
        setAutoAnnotationJob(latest);
        if (isTerminalJobStatus(latest.status)) {
          await refreshAutoAnnotationData(selectedId);
          if (cancelled) return;
          setActionMessage(isFailedJobStatus(latest.status) ? "AI 自动标注任务失败，请检查任务详情" : `AI 自动标注已完成：${latest.completed}/${latest.total} 个样本`);
        }
      } catch (reason) {
        if (!cancelled) setError(reason instanceof Error ? reason.message : "读取 AI 自动标注任务状态失败");
      }
    };
    void poll();
    const timer = window.setInterval(() => void poll(), 2000);
    return () => { cancelled = true; window.clearInterval(timer); };
  }, [autoAnnotationJob?.job_id, autoAnnotationJob?.status, refreshAutoAnnotationData, selectedId, token]);

  if (!token) return <LoginScreen onLogin={handleLogin} />;
  return <div className="app-shell">
    <header className="topbar"><div className="topbar-brand"><div className="brand-mark small"><Database size={18} /></div><div><strong>复核标注工作台</strong><span>REVIEW ANNOTATION PLATFORM</span></div></div><div className="topbar-center"><span className="system-status"><i />系统运行正常</span><span className="separator" /><span className="workspace-name">算法数据闭环 / 默认工作区</span></div><div className="topbar-actions"><button className="icon-button" onClick={() => refreshQueue()} title="刷新队列"><RefreshCw size={17} /></button><button className="icon-button" title="系统设置"><Settings2 size={17} /></button><button className="user-menu" onClick={logout}><span className="avatar">A</span><span>admin</span><ChevronDown size={14} /></button></div></header>
    <main className="workspace"><section className="workspace-heading"><div><span className="eyebrow">HUMAN-IN-THE-LOOP</span><h1>存疑目标复核</h1><p>确认模型候选框的真伪，为下一轮阈值校准和训练积累金标。</p></div><div className="heading-actions">{profiles.length > 0 ? <label className="select-control"><span>模型配置</span><select value={profileId} onChange={(event) => setProfileId(event.target.value)}><option value="">未配置</option>{profiles.map((profile) => <option key={profile.model_profile_id} value={profile.model_profile_id}>{profile.name} · {profile.model_version}</option>)}</select></label> : <button className="secondary-button" onClick={() => setProfileFormOpen(true)}><Settings2 size={15} />添加模型配置</button>}<button className="secondary-button" onClick={handleRun}><Play size={15} />批量识别</button><div className="auto-annotation-entry"><button className="secondary-button" onClick={handleAutoAnnotate} disabled={autoAnnotationRunning || autoAnnotationSubmitting || autoAnnotationSampleIds.length === 0} title={autoAnnotationSampleIds.length === 0 ? "当前筛选队列没有可自动标注的样本" : `将对 ${autoAnnotationSampleIds.length} 个待复核样本运行 AI 自动标注`}><Sparkles size={15} />AI 自动标注{autoAnnotationRunning && <LoaderCircle className="spin" size={13} />}</button><span>仅自动写入高置信多数结论；冲突/不确定保留待复核（{autoAnnotationSampleIds.length} 个样本）</span></div><button className="secondary-button" onClick={handleExport}><Download size={15} />导出金标</button></div></section>
      {profileFormOpen && <div className="profile-form"><div><strong>连接外部模型服务</strong><span>保存地址后，Worker 会通过 HTTP Adapter 批量调用。</span></div><input value={profileBaseUrl} onChange={(event) => setProfileBaseUrl(event.target.value)} placeholder="例如 http://localhost:8000" /><button className="primary-button" onClick={handleCreateProfile}>保存配置</button><button className="ghost-button" onClick={() => setProfileFormOpen(false)}>取消</button></div>}
      <section className="metric-strip"><div><span>待复核目标</span><strong>{stats.pending}</strong><small>当前页</small></div><div><span>已完成人工判断</span><strong>{stats.labeled}</strong><small>当前页</small></div><div><span>真目标</span><strong className="positive">{stats.true}</strong><small>当前页人工标注</small></div><div><span>假目标</span><strong className="negative">{stats.false}</strong><small>当前页人工标注</small></div><div className="metric-note"><Activity size={17} /><div><strong>一期评估口径</strong><span>候选框 precision · 不代表完整召回率</span></div></div></section>
      {autoAnnotationJob && <div className={`notice auto-annotation-notice ${autoAnnotationFailed ? "error" : ""}`}><Sparkles size={16} /><div className="auto-annotation-copy"><strong>AI 自动标注 · {autoAnnotationStatusLabel(autoAnnotationJob.status)}</strong><span>{autoAnnotationJob.completed}/{autoAnnotationJob.total} 个样本已处理{autoAnnotationJob.failed > 0 ? ` · 失败 ${autoAnnotationJob.failed}` : ""}</span><small>模型仅自动写入高置信多数结论；冲突/不确定保留待复核。</small>{autoAnnotationJob.error_summary && <small>任务信息：{autoAnnotationJob.error_summary}</small>}</div>{autoAnnotationTerminal && <button onClick={() => setAutoAnnotationJob(null)} aria-label="关闭 AI 自动标注状态">×</button>}</div>}
      {(error || actionMessage || job) && <div className={`notice ${error ? "error" : ""}`}>{error || actionMessage || `${job?.status === "running" ? "任务执行中" : "任务已创建"} · ${job?.completed ?? 0}/{job?.total ?? 0}`}<button onClick={() => { setError(""); setActionMessage(""); }}>×</button></div>}
      <div className="review-layout"><QueuePanel samples={samples} selectedId={selectedId} statusFilter={statusFilter} classFilter={classFilter} datasetFilter={datasetFilter} classOptions={CLASS_FILTERS} query={query} pageOffset={queuePage * QUEUE_PAGE_SIZE} pageSize={QUEUE_PAGE_SIZE} total={queueTotal} onSelect={setSelectedId} onFilter={(value) => { setQueuePage(0); setSelectedId(null); setStatusFilter(value); }} onClassFilter={(value) => { setQueuePage(0); setSelectedId(null); setClassFilter(value); }} onDatasetFilter={(value) => { setQueuePage(0); setSelectedId(null); setDatasetFilter(value); }} onQuery={(value) => { setQueuePage(0); setQuery(value); }} onPageChange={(page) => { setQueuePage(page); setSelectedId(null); }} onUpload={handleUpload} uploading={uploading} /><div className="review-main">{loading && <div className="loading-overlay">加载队列…</div>}{sample ? <><div className="sample-context"><div><span className="eyebrow">SAMPLE DETAIL</span><h2>{sample.session_id ?? sample.sample_id}</h2><span>{sample.source_service ?? sample.source_type} · {sample.rule_code ?? "未指定规则"} · {sample.capture_time ? new Date(sample.capture_time).toLocaleString("zh-CN") : "时间未知"}</span></div><div className="context-actions">{sample.dataset_tags.includes("v2") && <span className="dataset-badge">V2 {sample.v2_pending_count}/{sample.v2_target_count}</span>}<span className="run-pill"><Archive size={13} />{activeRun?.model_run.model_version ?? "无模型版本"}</span><span className={`status-text ${sample.status}`}>{sample.status === "completed" ? "已完成" : sample.status === "in_progress" ? "进行中" : "待标注"}</span></div></div><div className="detail-layout"><ReviewCanvas imageUrl={imageUrl} candidates={visibleCandidates} selectedId={selectedCandidateId} onSelect={setSelectedCandidateId} /><Inspector key={selectedCandidateId ?? "none"} candidate={candidate} candidates={visibleCandidates} onSelect={setSelectedCandidateId} onLabel={saveLabel} saving={saving} /></div></> : <div className="no-sample"><UploadCloud size={34} /><h2>选择一个样本开始复核</h2><p>左侧队列会展示来自复检服务、MinIO 导入或页面上传的图片。</p></div>}</div></div>
    </main><footer className="app-footer"><span>RAP v0.1 · 数据由 PostgreSQL / MinIO 管理</span><span><ShieldCheck size={14} />所有标注操作均已审计</span></footer>
  </div>;
}

export default App;
