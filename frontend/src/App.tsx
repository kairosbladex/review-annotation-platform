import { Activity, Archive, ChevronDown, Database, Download, LogOut, Play, RefreshCw, Settings2, ShieldCheck, UploadCloud } from "lucide-react";
import { useCallback, useEffect, useMemo, useState } from "react";
import { createExport, createModelRun, createProfile, getAssetUrl, getJob, getSample, labelCandidate, listProfiles, listSamples, login, uploadImages } from "./api";
import { Inspector } from "./components/Inspector";
import { QueuePanel } from "./components/QueuePanel";
import { ReviewCanvas } from "./components/ReviewCanvas";
import type { Candidate, Job, ModelProfile, Sample } from "./types";
import "./styles.css";

const TOKEN_KEY = "review-annotation-token";

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
  const [statusFilter, setStatusFilter] = useState("pending");
  const [query, setQuery] = useState("");
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

  const refreshQueue = useCallback(async () => {
    if (!token) return;
    const result = await listSamples(token, statusFilter);
    setSamples(result.samples);
    setSelectedId((current) => current && result.samples.some((item) => item.sample_id === current) ? current : result.samples[0]?.sample_id ?? null);
  }, [statusFilter, token]);

  const refreshDetail = useCallback(async (sampleId: string) => {
    if (!token) return;
    const result = await getSample(token, sampleId);
    setSample(result);
    const original = result.assets.find((asset) => asset.kind === "original");
    if (original?.status === "ready") {
      try { setImageUrl((await getAssetUrl(token, original.asset_id)).url); } catch { setImageUrl(null); }
    } else setImageUrl(null);
    const firstPending = result.candidates.find((candidate) => candidate.annotation.label === null);
    setSelectedCandidateId((current) => current && result.candidates.some((candidate) => candidate.candidate_id === current) ? current : firstPending?.candidate_id ?? result.candidates[0]?.candidate_id ?? null);
  }, [token]);

  useEffect(() => {
    if (!token) return;
    setLoading(true); setError("");
    Promise.all([refreshQueue(), listProfiles(token).then((result) => { setProfiles(result.profiles); setProfileId((current) => current || result.profiles[0]?.model_profile_id || ""); })]).catch((reason) => setError(reason instanceof Error ? reason.message : "读取数据失败")).finally(() => setLoading(false));
  }, [token, statusFilter, refreshQueue]);

  useEffect(() => { if (token && selectedId) refreshDetail(selectedId).catch((reason) => setError(reason instanceof Error ? reason.message : "读取样本失败")); }, [refreshDetail, selectedId, token]);

  useEffect(() => {
    const handler = (event: KeyboardEvent) => {
      if (!selectedCandidateId || event.target instanceof HTMLInputElement || event.target instanceof HTMLTextAreaElement) return;
      if (event.key !== "1" && event.key !== "2") return;
      const label = event.key === "1" ? "true" : "false";
      const candidate = sample?.candidates.find((item) => item.candidate_id === selectedCandidateId);
      if (!candidate || saving) return;
      setSaving(true); labelCandidate(token ?? "", candidate.candidate_id, { label, note: candidate.annotation.note ?? "", expected_revision: candidate.annotation.revision }).then((result) => { setSample(result.sample); setActionMessage(label === "true" ? "已标记为真目标" : "已标记为假目标"); refreshQueue(); }).catch((reason) => setError(reason instanceof Error ? reason.message : "标注失败")).finally(() => setSaving(false));
    };
    window.addEventListener("keydown", handler); return () => window.removeEventListener("keydown", handler);
  }, [refreshQueue, sample, saving, selectedCandidateId, token]);

  const candidate = sample?.candidates.find((item) => item.candidate_id === selectedCandidateId) ?? null;
  const stats = useMemo(() => samples.reduce((acc, item) => ({ pending: acc.pending + item.counts.pending, labeled: acc.labeled + item.counts.labeled, true: acc.true + item.counts.true, false: acc.false + item.counts.false }), { pending: 0, labeled: 0, true: 0, false: 0 }), [samples]);
  const activeRun = sample?.runs.find((run) => run.annotation_task_id === sample.annotation_task_id) ?? sample?.runs[0];

  async function handleLogin(username: string, password: string) { const result = await login(username, password); sessionStorage.setItem(TOKEN_KEY, result.access_token); setToken(result.access_token); }
  function logout() { sessionStorage.removeItem(TOKEN_KEY); setToken(null); setSample(null); }
  async function saveLabel(label: "true" | "false" | null, note: string) {
    if (!token || !candidate) return;
    setSaving(true); setError("");
    try { const result = await labelCandidate(token, candidate.candidate_id, { label, note, expected_revision: candidate.annotation.revision }); setSample(result.sample); await refreshQueue(); setActionMessage(label === null ? "已撤销标注" : label === "true" ? "已标记为真目标" : "已标记为假目标"); } catch (reason) { setError(reason instanceof Error ? reason.message : "保存标注失败"); } finally { setSaving(false); }
  }
  async function handleUpload(files: File[]) { if (!token) return; setUploading(true); try { const result = await uploadImages(token, files); setActionMessage(`上传完成：${result.imported} 张图片进入队列`); setStatusFilter("all"); await refreshQueue(); } catch (reason) { setError(reason instanceof Error ? reason.message : "上传失败"); } finally { setUploading(false); } }
  async function handleRun() { if (!token || !profileId || samples.length === 0) { setError("请先选择模型配置并确保队列中有样本"); return; } try { const result = await createModelRun(token, samples.map((item) => item.sample_id), profileId); setJob({ job_id: result.job_id, status: result.status, total: samples.length, completed: 0, failed: 0, error_summary: null }); setActionMessage(`已创建批量识别任务 ${result.run_id.slice(0, 8)}`); } catch (reason) { setError(reason instanceof Error ? reason.message : "创建模型任务失败"); } }
  async function handleCreateProfile() { if (!token || !profileBaseUrl.trim()) { setError("请填写外部模型服务地址"); return; } try { const result = await createProfile(token, { profile_key: `http-${Date.now()}`, name: "外部模型服务", model_name: "review-detector", model_version: "v1", adapter_type: "http", config: { base_url: profileBaseUrl.trim(), endpoint: "/api/v1/review-agent/yolo-detect" } }); const refreshed = await listProfiles(token); setProfiles(refreshed.profiles); setProfileId(result.model_profile_id); setProfileFormOpen(false); setProfileBaseUrl(""); setActionMessage("模型配置已保存"); } catch (reason) { setError(reason instanceof Error ? reason.message : "保存模型配置失败"); } }
  async function handleExport() { if (!token || !activeRun) { setError("当前样本没有可导出的模型运行"); return; } try { const result = await createExport(token, activeRun.model_run_id); setJob({ job_id: result.job_id, status: result.status, total: 1, completed: 0, failed: 0, error_summary: null }); setActionMessage("导出任务已创建，完成后可在导出记录中下载"); } catch (reason) { setError(reason instanceof Error ? reason.message : "创建导出失败"); } }

  useEffect(() => { if (!token || !job || !["pending", "running"].includes(job.status)) return; const timer = window.setInterval(() => getJob(token, job.job_id).then(setJob).catch(() => undefined), 2000); return () => window.clearInterval(timer); }, [job, token]);

  if (!token) return <LoginScreen onLogin={handleLogin} />;
  return <div className="app-shell">
    <header className="topbar"><div className="topbar-brand"><div className="brand-mark small"><Database size={18} /></div><div><strong>复核标注工作台</strong><span>REVIEW ANNOTATION PLATFORM</span></div></div><div className="topbar-center"><span className="system-status"><i />系统运行正常</span><span className="separator" /><span className="workspace-name">算法数据闭环 / 默认工作区</span></div><div className="topbar-actions"><button className="icon-button" onClick={() => refreshQueue()} title="刷新队列"><RefreshCw size={17} /></button><button className="icon-button" title="系统设置"><Settings2 size={17} /></button><button className="user-menu" onClick={logout}><span className="avatar">A</span><span>admin</span><ChevronDown size={14} /></button></div></header>
    <main className="workspace"><section className="workspace-heading"><div><span className="eyebrow">HUMAN-IN-THE-LOOP</span><h1>存疑目标复核</h1><p>确认模型候选框的真伪，为下一轮阈值校准和训练积累金标。</p></div><div className="heading-actions">{profiles.length > 0 ? <label className="select-control"><span>模型配置</span><select value={profileId} onChange={(event) => setProfileId(event.target.value)}><option value="">未配置</option>{profiles.map((profile) => <option key={profile.model_profile_id} value={profile.model_profile_id}>{profile.name} · {profile.model_version}</option>)}</select></label> : <button className="secondary-button" onClick={() => setProfileFormOpen(true)}><Settings2 size={15} />添加模型配置</button>}<button className="secondary-button" onClick={handleRun}><Play size={15} />批量识别</button><button className="secondary-button" onClick={handleExport}><Download size={15} />导出金标</button></div></section>
      {profileFormOpen && <div className="profile-form"><div><strong>连接外部模型服务</strong><span>保存地址后，Worker 会通过 HTTP Adapter 批量调用。</span></div><input value={profileBaseUrl} onChange={(event) => setProfileBaseUrl(event.target.value)} placeholder="例如 http://localhost:8000" /><button className="primary-button" onClick={handleCreateProfile}>保存配置</button><button className="ghost-button" onClick={() => setProfileFormOpen(false)}>取消</button></div>}
      <section className="metric-strip"><div><span>待复核目标</span><strong>{stats.pending}</strong><small>当前队列</small></div><div><span>已完成人工判断</span><strong>{stats.labeled}</strong><small>跨当前筛选样本</small></div><div><span>真目标</span><strong className="positive">{stats.true}</strong><small>人工标注</small></div><div><span>假目标</span><strong className="negative">{stats.false}</strong><small>可作为 hard negative</small></div><div className="metric-note"><Activity size={17} /><div><strong>一期评估口径</strong><span>候选框 precision · 不代表完整召回率</span></div></div></section>
      {(error || actionMessage || job) && <div className={`notice ${error ? "error" : ""}`}>{error || actionMessage || `${job?.status === "running" ? "任务执行中" : "任务已创建"} · ${job?.completed ?? 0}/{job?.total ?? 0}`}<button onClick={() => { setError(""); setActionMessage(""); }}>×</button></div>}
      <div className="review-layout"><QueuePanel samples={samples} selectedId={selectedId} statusFilter={statusFilter} query={query} onSelect={setSelectedId} onFilter={setStatusFilter} onQuery={setQuery} onUpload={handleUpload} uploading={uploading} /><div className="review-main">{loading && <div className="loading-overlay">加载队列…</div>}{sample ? <><div className="sample-context"><div><span className="eyebrow">SAMPLE DETAIL</span><h2>{sample.session_id ?? sample.sample_id}</h2><span>{sample.source_service ?? sample.source_type} · {sample.rule_code ?? "未指定规则"} · {sample.capture_time ? new Date(sample.capture_time).toLocaleString("zh-CN") : "时间未知"}</span></div><div className="context-actions"><span className="run-pill"><Archive size={13} />{activeRun?.model_run.model_version ?? "无模型版本"}</span><span className={`status-text ${sample.status}`}>{sample.status === "completed" ? "已完成" : sample.status === "in_progress" ? "进行中" : "待标注"}</span></div></div><div className="detail-layout"><ReviewCanvas imageUrl={imageUrl} candidates={sample.candidates} selectedId={selectedCandidateId} onSelect={setSelectedCandidateId} /><Inspector candidate={candidate} candidates={sample.candidates} onSelect={setSelectedCandidateId} onLabel={saveLabel} saving={saving} /></div></> : <div className="no-sample"><UploadCloud size={34} /><h2>选择一个样本开始复核</h2><p>左侧队列会展示来自复检服务、MinIO 导入或页面上传的图片。</p></div>}</div></div>
    </main><footer className="app-footer"><span>RAP v0.1 · 数据由 PostgreSQL / MinIO 管理</span><span><ShieldCheck size={14} />所有标注操作均已审计</span></footer>
  </div>;
}

export default App;
