import { useCallback, useEffect, useRef, useState } from "react";
import { createRoot } from "react-dom/client";
import { ArrowLeft, ArrowRight, Check, ChevronDown, ChevronLeft, ChevronRight, Download, Expand, ExternalLink, Focus, ImageOff, Layers3, LoaderCircle, RefreshCw, ScanLine, Search, X, ZoomIn, ZoomOut } from "lucide-react";
import "./workbench.css";

type Box = [number, number, number, number];
type Candidate = { index: number; className: string; initialConfidence: number | null; bbox: Box | null; geometry: string; decision: string; reason: string; reasonCode: string; model: string; cleanApplied: boolean; cleanRequested: boolean; pairPassed: boolean | null; baseDecision: string; adapterDecision: string; personManual: boolean; audit: Record<string, unknown> };
type ReviewImage = { id: string; role: string; annotatedUrl: string; originalUrl: string | null; sourceNote: string; recordedResolution: string; candidates: Candidate[]; recheckTime: string; hasResult: boolean; rawResult: unknown };
type EventItem = { id: string; eventNo: string; typeCode: string; typeName: string; eventTime: string; recheckTime: string; status: string; storedStatus: string; supported: boolean; hasOriginalRelation: boolean; thumbnail: string };
type Detail = EventItem & { images: ReviewImage[]; capturedAt: string; submitEnabled: boolean; platformUrl: string; resultIsHistorical: boolean; submission: { state: string; message: string; submittedAt: string } | null };
type Queue = { items: EventItem[]; total: number; page: number; limit: number; counts: Record<string, number>; allCount: number; capturedAt: string; projectName: string; types: { code: string; name: string }[]; submitEnabled: boolean };
type View = { zoom: number; cx: number; cy: number; target: boolean };
const initialView: View = { zoom: 1, cx: .5, cy: .5, target: false };
const statuses: Record<string, string> = { all: "全部事件", "0": "待复核", "1": "复核中", "2": "已复核", "3": "复核失败", "4": "未启用复核" };
const labels: Record<string, string> = { confirmed: "真目标", rejected: "假目标", uncertain: "存疑", pending: "待复核" };
const color: Record<string, string> = { confirmed: "#1a9b78", rejected: "#e05164", uncertain: "#dc982e", pending: "#7292c7" };
const pct = (v: number | null) => v == null ? "未记录" : `${(v * 100).toFixed(1)}%`;
const clockTime = (s?: string) => s ? new Date(s).toLocaleTimeString("zh-CN", { hour12: false }) : "—";
async function api<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(`/workbench-api${path}`, init);
  const body = await response.json().catch(() => ({}));
  if (!response.ok) throw new Error(body.detail || `读取失败（${response.status}）`);
  return body;
}
function explanation(c: Candidate) {
  if (c.decision === "pending") return "尚未生成复核结果，可提交当前事件后查看。";
  if (c.personManual) return "模型判断可能为人员，当前复核规则保留人工确认。";
  if (c.reasonCode === "clean_source_consensus_rejected") return "基础模型与训练后模型对照原图后，均判定为假目标。";
  if (c.cleanApplied && c.decision === "confirmed") return "原图复核通过，保留该目标。";
  if (c.cleanRequested && c.pairPassed === false) return "原图配对或目标位置的校验未通过，需要结合全图人工判断。";
  return c.reason;
}
function Badge({ value }: { value: string }) { return <span className={`badge ${value}`}>{labels[value] || statuses[value] || value}</span>; }
function EmptyImage({ message }: { message: string }) { return <div className="image-empty"><ImageOff size={28} /><strong>{message}</strong><span>可继续查看另一侧图片与已有结果</span></div>; }

function ImageCanvas({ src, title, subtitle, candidates, selected, onSelect, original, view, setView }: {
  src: string | null; title: string; subtitle: string; candidates: Candidate[]; selected?: Candidate; onSelect: (i: number) => void;
  original?: boolean; view: View; setView: (v: View) => void;
}) {
  const canvas = useRef<HTMLCanvasElement>(null), holder = useRef<HTMLDivElement>(null);
  const [image, setImage] = useState<HTMLImageElement | null>(null), [error, setError] = useState(false), [attempt, setAttempt] = useState(0);
  const [dimensions, setDimensions] = useState({ w: 400, h: 330 });
  const transform = useRef({ left: 0, top: 0, scale: 1, sx: 0, sy: 0 });
  const pointer = useRef<{ x: number; y: number; cx: number; cy: number; moved: boolean } | null>(null);
  useEffect(() => {
    setImage(null); setError(false); if (!src) return;
    const im = new Image(); let active = true;
    im.onload = () => { if (active) setImage(im); }; im.onerror = () => { if (active) setError(true); };
    im.src = src + (attempt ? `?refresh=${attempt}` : "");
    return () => { active = false; };
  }, [src, attempt]);
  useEffect(() => {
    const observer = new ResizeObserver(([entry]) => setDimensions({ w: entry.contentRect.width, h: entry.contentRect.height }));
    if (holder.current) observer.observe(holder.current);
    return () => observer.disconnect();
  }, []);
  useEffect(() => {
    if (!image || !canvas.current) return;
    const c = canvas.current, ctx = c.getContext("2d")!;
    const { w, h } = dimensions, dpr = window.devicePixelRatio || 1;
    c.width = Math.round(w * dpr); c.height = Math.round(h * dpr); ctx.scale(dpr, dpr);
    ctx.fillStyle = "#101927"; ctx.fillRect(0, 0, w, h);
    const base = Math.min(w / image.width, h / image.height), scale = base * view.zoom;
    const left = w / 2 - view.cx * image.width * scale, top = h / 2 - view.cy * image.height * scale;
    transform.current = { left, top, scale, sx: image.width, sy: image.height };
    ctx.drawImage(image, left, top, image.width * scale, image.height * scale);
    for (const candidate of candidates) {
      if (!candidate.bbox) continue;
      const b = candidate.bbox, focused = selected?.index === candidate.index;
      const x = left + b[0] * image.width * scale, y = top + b[1] * image.height * scale;
      const bw = (b[2] - b[0]) * image.width * scale, bh = (b[3] - b[1]) * image.height * scale;
      ctx.strokeStyle = focused ? "#ffdd75" : (original ? color[candidate.decision] : "#ff6072");
      ctx.lineWidth = focused ? 2.5 : 1.5; ctx.strokeRect(x, y, bw, bh);
      ctx.fillStyle = focused ? "#ffdd75" : (original ? color[candidate.decision] : "#ff6072");
      ctx.fillRect(x, y - 21, 26, 21); ctx.font = "bold 12px sans-serif"; ctx.fillStyle = focused ? "#1b2532" : "#fff";
      ctx.fillText(String(candidate.index).padStart(2, "0"), x + 5, y - 6);
    }
  }, [image, dimensions, candidates, selected, original, view]);
  return <section className="image-panel">
    <header><span className={`image-label ${original ? "original" : "annotated"}`}>{original ? "02" : "01"}</span><div><h3>{title}</h3><p>{subtitle}</p></div><span className="pixel-size">{image ? `${image.width} × ${image.height}` : ""}</span></header>
    <div className="canvas-holder" ref={holder}>
      {!src ? <EmptyImage message="未找到关联原图" /> : error ? <div className="image-empty"><ImageOff size={28} /><strong>测试环境图片暂不可用</strong><button onClick={() => setAttempt(v => v + 1)}>重新读取</button></div> : !image ? <div className="image-empty"><LoaderCircle className="spin" size={24} /><span>正在读取测试环境图片…</span></div> : null}
      <canvas ref={canvas} aria-label={`${title}，可拖动查看，点击框选择目标`} style={{ visibility: image ? "visible" : "hidden" }}
        onPointerDown={e => { e.currentTarget.setPointerCapture(e.pointerId); pointer.current = { x: e.clientX, y: e.clientY, cx: view.cx, cy: view.cy, moved: false }; }}
        onPointerMove={e => {
          const p = pointer.current, t = transform.current; if (!p || !image) return;
          if (Math.abs(e.clientX - p.x) + Math.abs(e.clientY - p.y) > 4) p.moved = true;
          if (p.moved) setView({ ...view, cx: Math.max(0, Math.min(1, p.cx - (e.clientX - p.x) / (t.sx * t.scale))), cy: Math.max(0, Math.min(1, p.cy - (e.clientY - p.y) / (t.sy * t.scale))) });
        }}
        onPointerUp={e => {
          const p = pointer.current; pointer.current = null; if (!p || p.moved || !image) return;
          const t = transform.current, rect = e.currentTarget.getBoundingClientRect();
          const x = (e.clientX - rect.left - t.left) / (t.sx * t.scale), y = (e.clientY - rect.top - t.top) / (t.sy * t.scale);
          const hit = candidates.filter(c => c.bbox && x >= c.bbox[0] && x <= c.bbox[2] && y >= c.bbox[1] && y <= c.bbox[3]).sort((a, b) => (a.bbox![2]-a.bbox![0])*(a.bbox![3]-a.bbox![1]) - (b.bbox![2]-b.bbox![0])*(b.bbox![3]-b.bbox![1]))[0];
          if (hit) onSelect(hit.index - 1);
        }} onPointerCancel={() => { pointer.current = null; }} />
    </div>
    <footer><span className={`legend-dot ${original ? "teal" : "red"}`} />{original ? "框线为页面定位层，原图文件未改动" : "初审图保留原有标注 · 黄色为当前目标"}</footer>
  </section>;
}

function Workbench() {
  const params = new URLSearchParams(location.search);
  const [status, setStatus] = useState(params.get("status") || "0"), [type, setType] = useState(params.get("type") || ""), [source, setSource] = useState(params.get("source") || "all");
  const [search, setSearch] = useState(params.get("search") || ""), [term, setTerm] = useState(params.get("search") || ""), [page, setPage] = useState(Math.max(1, Number(params.get("page")) || 1));
  const [queue, setQueue] = useState<Queue | null>(null), [queueBusy, setQueueBusy] = useState(true), [queueError, setQueueError] = useState("");
  const [selectedId, setSelectedId] = useState(params.get("event") || ""), [detail, setDetail] = useState<Detail | null>(null), [detailBusy, setDetailBusy] = useState(false), [detailError, setDetailError] = useState("");
  const [imageIndex, setImageIndex] = useState(0), [candidateIndex, setCandidateIndex] = useState(0), [view, setView] = useState<View>(initialView);
  const [fullscreen, setFullscreen] = useState(false), [submitting, setSubmitting] = useState(false), [message, setMessage] = useState("");
  const queueGeneration = useRef(0), detailGeneration = useRef(0), deepLink = useRef(!!params.get("event"));
  const currentSelection = useRef(selectedId);
  currentSelection.current = selectedId;
  const activeImage = detail?.images[imageIndex], candidates = activeImage?.candidates || [], selected = candidates[candidateIndex];
  useEffect(() => { if (search === term) return; const t = setTimeout(() => { setTerm(search); setPage(1); }, 300); return () => clearTimeout(t); }, [search, term]);
  const refreshQueue = useCallback(async (keep = false) => {
    const gen = ++queueGeneration.current; setQueueBusy(true); setQueueError("");
    try {
      const q = await api<Queue>(`/events?${new URLSearchParams({ status, type_code: type, source, search: term, page: String(page) })}`);
      if (gen !== queueGeneration.current) return;
      setQueue(q);
      const keepDeepLink = deepLink.current;
      setSelectedId(current => (keep || keepDeepLink || q.items.some(x => x.id === current)) && current ? current : q.items[0]?.id || "");
      deepLink.current = false;
    } catch (e) { if (gen === queueGeneration.current) setQueueError(String((e as Error).message)); }
    finally { if (gen === queueGeneration.current) setQueueBusy(false); }
  }, [status, type, source, term, page]);
  useEffect(() => { void refreshQueue(); }, [refreshQueue]);
  const refreshDetail = useCallback(async (eid: string, preserve = false) => {
    const gen = ++detailGeneration.current; setDetailError("");
    if (!preserve) { setDetail(null); setDetailBusy(true); setImageIndex(0); setCandidateIndex(0); setView(initialView); setMessage(""); }
    try { const value = await api<Detail>(`/events/${eid}`); if (gen === detailGeneration.current && currentSelection.current === eid) setDetail(value); }
    catch (e) { if (gen === detailGeneration.current) setDetailError((e as Error).message); }
    finally { if (gen === detailGeneration.current) setDetailBusy(false); }
  }, []);
  useEffect(() => {
    if (selectedId) { void refreshDetail(selectedId); }
    else { ++detailGeneration.current; setDetail(null); }
  }, [selectedId, refreshDetail]);
  useEffect(() => {
    history.replaceState(null, "", `?${new URLSearchParams({ status, event: selectedId, source, type, search: term, page: String(page) })}`);
  }, [status, selectedId, source, type, term, page]);
  // Auto-refresh reads only. It never submits or retries model work.
  useEffect(() => {
    const pending = detail?.status === "1" || ["awaiting", "running"].includes(detail?.submission?.state || "");
    const t = setInterval(() => { if (!document.hidden) { void refreshQueue(true); if (selectedId) void refreshDetail(selectedId, true); } }, pending ? 10000 : 60000);
    return () => clearInterval(t);
  }, [detail?.status, detail?.submission?.state, selectedId, refreshQueue, refreshDetail]);
  const selectCandidate = useCallback((index: number) => {
    setCandidateIndex(index); const b = candidates[index]?.bbox;
    if (b) setView({ target: true, cx: (b[0] + b[2]) / 2, cy: (b[1] + b[3]) / 2, zoom: Math.max(1, Math.min(12, .6 / Math.max(b[2] - b[0], b[3] - b[1], .05))) });
    else setView(initialView);
  }, [candidates]);
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") { setFullscreen(false); return; }
      if ((e.target as HTMLElement).closest("input,select,textarea,button")) return;
      if (e.key === "ArrowRight" && candidateIndex < candidates.length - 1) { e.preventDefault(); selectCandidate(candidateIndex + 1); }
      if (e.key === "ArrowLeft" && candidateIndex > 0) { e.preventDefault(); selectCandidate(candidateIndex - 1); }
    };
    window.addEventListener("keydown", onKey); return () => window.removeEventListener("keydown", onKey);
  }, [candidateIndex, candidates.length, selectCandidate]);
  function filterStatus(value: string) { deepLink.current = false; setStatus(value); setPage(1); }
  async function submit() {
    if (!detail || submitting) return;
    const eid = detail.id; setSubmitting(true); setMessage("");
    try { const r = await api<{ message: string }>(`/events/${eid}/recheck`, { method: "POST", headers: { "X-Workbench-Action": "recheck" } }); if (currentSelection.current === eid) { setMessage(r.message); await refreshDetail(eid, true); } }
    catch (e) { if (currentSelection.current === eid) setMessage((e as Error).message); }
    finally { setSubmitting(false); }
  }
  const isRunning = detail?.status === "1" || ["awaiting", "running"].includes(detail?.submission?.state || "");
  const canSubmit = !!detail?.submitEnabled && detail.supported && ["0", "3"].includes(detail.status) && !isRunning && !!detail.images.length;
  const decisionCounts = candidates.reduce<Record<string, number>>((m, c) => { m[c.decision] = (m[c.decision] || 0) + 1; return m; }, {});
  const queueIndex = queue?.items.findIndex(x => x.id === selectedId) ?? -1;
  const controls = <div className="image-controls">
    <div className="segmented"><button className={!view.target ? "active" : ""} onClick={() => setView(initialView)}><Layers3 size={14} />查看全图</button><button disabled={!selected?.bbox} className={view.target ? "active" : ""} onClick={() => selectCandidate(candidateIndex)}><Focus size={14} />目标细节</button></div>
    <div className="zoom-control"><button aria-label="缩小" disabled={view.zoom <= 1} onClick={() => setView({ ...view, zoom: Math.max(1, view.zoom / 1.4) })}><ZoomOut size={15} /></button><span>{view.zoom.toFixed(1)}×</span><button aria-label="放大" disabled={view.zoom >= 24} onClick={() => setView({ ...view, zoom: Math.min(24, view.zoom * 1.4) })}><ZoomIn size={15} /></button><button aria-label={fullscreen ? "退出大图" : "展开对照大图"} onClick={() => setFullscreen(!fullscreen)}>{fullscreen ? <X size={16} /> : <Expand size={16} />}</button></div>
  </div>;
  const pair = activeImage && <div className="image-pair">
    <ImageCanvas key={`${activeImage.id}-initial`} src={activeImage.annotatedUrl} title="初审结果" subtitle={activeImage.role === "SHOTIMG" ? "初审标注图" : "事件主图 · 未提供初审截图"} candidates={candidates} selected={selected} onSelect={selectCandidate} view={view} setView={setView} />
    <ImageCanvas key={`${activeImage.id}-original`} src={activeImage.originalUrl} title="原图对照" subtitle="使用关联原始附件查看目标细节" original candidates={candidates} selected={selected} onSelect={selectCandidate} view={view} setView={setView} />
  </div>;
  return <div className="workbench">
    <header className="app-header"><div className="brand"><span className="brand-icon"><ScanLine size={23} /></span><div><strong>初审 · 复核对照台</strong><span>VISUAL REVIEW WORKSPACE</span></div></div><div className="environment"><i />测试环境<span className="environment-project">空天地一体化测试平台</span></div><div className="live-info"><span>更新于 {clockTime(queue?.capturedAt)}</span><button className="button secondary" disabled={queueBusy || detailBusy} onClick={() => { void refreshQueue(true); if (selectedId) void refreshDetail(selectedId, true); }}><RefreshCw size={14} className={queueBusy ? "spin" : ""} />刷新数据</button></div></header>
    <section className="workspace-heading"><div><h1>让每一次复核，都看得清楚。</h1><p>从待复核事件开始，对照初审框与原图，逐个查看目标结论。</p></div><div className="stat-row">{["0", "1", "2"].map(s => <button key={s} className={`stat ${status === s ? "selected" : ""}`} onClick={() => filterStatus(s)}><span>{statuses[s]}</span><strong>{queue ? (queue.counts[s] || 0).toLocaleString() : "—"}</strong></button>)}</div></section>
    <main className="workspace-grid">
      <aside className="queue-panel">
        <div className="queue-title"><h2>事件队列</h2><span>{queue?.total ?? "—"} 条</span></div>
        <div className="filters"><label className="search"><Search size={16} /><input value={search} onChange={e => setSearch(e.target.value)} placeholder="搜索事件编号、类型" aria-label="搜索事件" />{search && <button aria-label="清空搜索" onClick={() => setSearch("")}><X size={13} /></button>}</label>
          <div className="filter-row"><select aria-label="复核状态" value={status} onChange={e => filterStatus(e.target.value)}>{["0", "1", "2", "3", "all", "4"].map(s => <option key={s} value={s}>{statuses[s]}</option>)}</select><select aria-label="原图筛选" value={source} onChange={e => { setSource(e.target.value); setPage(1); }}><option value="all">全部图片</option><option value="linked">有关联原图</option><option value="missing">缺少原图关联</option></select></div>
          <select className="type-select" aria-label="识别类型" value={type} onChange={e => { setType(e.target.value); setPage(1); }}><option value="">全部识别类型</option>{queue?.types.map(t => <option key={t.code + t.name} value={t.code}>{t.name}</option>)}</select>
        </div>
        {queueError && <div className="error-box" role="alert">{queueError}<button onClick={() => void refreshQueue()}>重试</button></div>}
        <div className={`event-list ${queueBusy ? "loading-list" : ""}`} aria-busy={queueBusy}>
          {!queue && queueBusy && <div className="empty-state"><LoaderCircle className="spin" /><p>读取测试环境队列…</p></div>}
          {queue?.items.map(item => <button key={item.id} className={`event-card ${selectedId === item.id ? "active" : ""}`} onClick={() => setSelectedId(item.id)}><div className="event-card-top"><strong>{item.typeName}</strong><Badge value={item.status} /></div><div className="event-card-body"><img src={item.thumbnail} alt="初审缩略图" loading="lazy" onError={e => { e.currentTarget.style.visibility = "hidden"; }} /><div><span className="event-number">{item.eventNo.replace("Event-", "")}</span><time>{item.eventTime}</time><span className={`source-tag ${item.hasOriginalRelation ? "available" : ""}`}>{item.hasOriginalRelation ? "有关联原图" : "缺少原图关联"}</span></div></div></button>)}
          {queue && !queue.items.length && !queueBusy && <div className="empty-state"><Search /><h3>没有匹配的事件</h3><p>试试切换复核状态或清空筛选。</p><button className="button secondary" onClick={() => { setSearch(""); setType(""); setSource("all"); filterStatus("0"); }}>返回待复核</button></div>}
        </div>
        <div className="pagination"><button aria-label="上一页" disabled={page <= 1 || queueBusy} onClick={() => setPage(p => p - 1)}><ChevronLeft size={16} /></button><span>第 {page} / {Math.max(1, Math.ceil((queue?.total || 0) / 24))} 页</span><button aria-label="下一页" disabled={!queue || page * 24 >= queue.total || queueBusy} onClick={() => setPage(p => p + 1)}><ChevronRight size={16} /></button></div>
      </aside>
      <section className="detail-panel" aria-busy={detailBusy}>
        {detailError && <div className="error-box" role="alert">{detailError}<button onClick={() => selectedId && void refreshDetail(selectedId)}>重试读取</button></div>}
        {detailBusy ? <div className="empty-state detail-loading"><LoaderCircle className="spin" /><h3>正在读取事件与图片关联…</h3></div> : !detail ? <div className="empty-state detail-loading"><ScanLine size={40} /><h3>选择一条事件，开始对照</h3><p>左侧可按状态、识别类型和原图关联筛选。</p></div> : <>
          <div className="detail-heading"><div><div className="detail-kicker">当前事件 <Badge value={detail.status} /></div><h2>{detail.typeName}<span>{detail.eventNo}</span></h2><p>{detail.eventTime} <span>·</span> {detail.images.length} 张图片 <span>·</span> {activeImage?.hasResult ? `复核于 ${activeImage.recheckTime}` : "尚无复核结果"}</p></div><div className="event-navigation"><button aria-label="上一条事件" disabled={queueIndex <= 0} onClick={() => queue && setSelectedId(queue.items[queueIndex - 1].id)}><ArrowLeft size={16} /></button><button aria-label="下一条事件" disabled={!queue || queueIndex < 0 || queueIndex >= queue.items.length - 1} onClick={() => queue && setSelectedId(queue.items[queueIndex + 1].id)}><ArrowRight size={16} /></button></div></div>
          <div className="event-actions"><div className="hint"><span className="live-dot" />实时读取测试数据 · 复核结论为模型输出</div><div><a className="button secondary" href={`/workbench-api/events/${detail.id}/export`} download><Download size={14} />导出当前结果</a><a className="button secondary" href={detail.platformUrl} target="_blank" rel="noreferrer"><ExternalLink size={14} />测试平台</a>{detail.submitEnabled && <button className="button primary" disabled={!canSubmit || submitting} onClick={() => void submit()}>{submitting || isRunning ? <LoaderCircle size={14} className="spin" /> : <ScanLine size={14} />}{isRunning ? "正在复核，自动更新" : detail.status === "2" ? "复核已完成" : "提交当前事件复核"}</button>}</div></div>
          {(message || (isRunning && detail.submission?.message)) && <div className="notice" role="status">{message || detail.submission?.message}</div>}
          {detail.resultIsHistorical && <div className="notice">下面保留的是上次复核结果；当前状态为“{statuses[detail.status]}”。</div>}
          {detail.images.length > 1 && <div className="image-tabs">{detail.images.map((im, i) => <button className={imageIndex === i ? "active" : ""} key={im.id} onClick={() => { setImageIndex(i); setCandidateIndex(0); setView(initialView); }}>图片 {i + 1} <span>{im.candidates.length} 个目标</span></button>)}</div>}
          {!activeImage ? <EmptyImage message="事件没有可用的初审图片附件" /> : <>
            <div className="comparison-heading"><div><h3>图像对照</h3><span>两侧同步缩放 · 拖动平移 · 点击框选目标</span></div>{controls}</div>
            {!fullscreen && pair}
            <div className={`source-note ${!activeImage.originalUrl || selected?.pairPassed === false ? "attention" : ""}`}><span>{selected?.pairPassed === false ? "此目标的原图配对或位置校验未通过，请结合全图人工判断。" : activeImage.sourceNote}</span><span>显示放大不增加原始细节</span></div>
            <section className="candidate-section"><header><div><h3>目标复核 <span>{candidates.length} 个</span></h3><p>点选目标，两侧同步定位。← → 切换目标</p></div><div className="result-counts">{["confirmed", "rejected", "uncertain", "pending"].filter(k => decisionCounts[k]).map(k => <span className={k} key={k}><i />{labels[k]} <strong>{decisionCounts[k]}</strong></span>)}</div></header>
              <div className="candidate-strip">{candidates.map((c, i) => <button className={`candidate-chip ${candidateIndex === i ? "selected" : ""}`} key={c.index} onClick={() => selectCandidate(i)}><span className={`candidate-dot ${c.decision}`} />{String(c.index).padStart(2, "0")}<span>{c.className}</span><small>{labels[c.decision]}</small></button>)}</div>
              {!candidates.length ? <div className="empty-candidates">未记录可解析的目标框，请直接查看初审全图。</div> : selected && <div className="candidate-detail"><div className="initial-finding"><span className="section-eyebrow">初审识别</span><h4><span className="target-number">{String(selected.index).padStart(2, "0")}</span>{selected.className}</h4><dl><div><dt>初审置信度</dt><dd>{pct(selected.initialConfidence)}</dd></div><div><dt>目标位置</dt><dd>{selected.geometry}</dd></div></dl></div><div className="review-finding"><div className="review-finding-heading"><span className="section-eyebrow">复核结论</span><Badge value={selected.decision} /></div><h4>{selected.decision === "pending" ? "等待复核，尚无结论" : selected.decision === "uncertain" ? "证据不足，保留人工判断" : selected.decision === "confirmed" ? "复核保留该目标" : "复核判定为假目标"}</h4><p>{explanation(selected)}</p><div className="method-tags"><span>{selected.cleanApplied ? "已用原图分析" : selected.cleanRequested ? "原图分析受限" : selected.decision === "pending" ? "未执行复核" : "历史复核结果"}</span>{selected.personManual && <span>人员判真保留人工确认</span>}{!selected.bbox && <span>缺少完整框，暂不能自动定位</span>}</div></div></div>}
            </section>
            <details className="evidence"><summary>查看复核依据与记录 <ChevronDown size={14} /></summary><div><p>事件 ID：{detail.id}</p><p>图片 ID：{activeImage.id}</p><p>结果时间：{activeImage.recheckTime || "未复核"} · 数据读取：{clockTime(detail.capturedAt)}</p>{selected && <><p>模型：{selected.model || "尚未执行"}</p><p>基础模型：{selected.baseDecision || "未记录"} · 训练后模型：{selected.adapterDecision || "未记录"}</p><p>原因代码：{selected.reasonCode || "无"}</p><pre>{JSON.stringify(selected.audit, null, 2)}</pre></>}</div></details>
          </>}
          <footer className="detail-footnote"><Check size={13} />浏览与筛选仅查询数据；提交按钮只复核当前事件。页面不将模型输出计算为准确率。</footer>
        </>}
      </section>
    </main>
    {fullscreen && <div className="fullscreen" role="dialog" aria-modal="true" aria-label="图像放大对照"><div className="fullscreen-heading"><div><strong>{detail?.eventNo}</strong><span>目标 {candidateIndex + 1} · {selected?.className} · {labels[selected?.decision || "pending"]}</span></div>{controls}</div>{pair}<div className="fullscreen-footer"><button className="button secondary" disabled={candidateIndex <= 0} onClick={() => selectCandidate(candidateIndex - 1)}><ArrowLeft size={14} />上一目标</button><span>拖动任一侧同步查看 · Esc 退出</span><button className="button secondary" disabled={candidateIndex >= candidates.length - 1} onClick={() => selectCandidate(candidateIndex + 1)}>下一目标<ArrowRight size={14} /></button></div></div>}
  </div>;
}

createRoot(document.getElementById("root")!).render(<Workbench />);
