import { ArrowLeftRight, Download, ImagePlus, LoaderCircle, Play, X } from "lucide-react";
import { useCallback, useEffect, useRef, useState } from "react";
import { createHistoryComparison, getHistoryComparison, listHistoryComparisons } from "../api";
import type { HistoryCompareResult, HistoryComparison, HistoryComparisonSummary } from "../api";
import { RecordPagination } from "./RecordPagination";

function useImageUrl(file: File | null) {
  const [url, setUrl] = useState("");
  useEffect(() => {
    if (!file) { setUrl(""); return; }
    const next = URL.createObjectURL(file); setUrl(next);
    return () => URL.revokeObjectURL(next);
  }, [file]);
  return url;
}

export function HistoryCompare({ token }: { token: string }) {
  const [baseline, setBaseline] = useState<File | null>(null);
  const [current, setCurrent] = useState<File | null>(null);
  const [result, setResult] = useState<HistoryCompareResult | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  const [showRegions, setShowRegions] = useState(true);
  const [saved, setSaved] = useState<HistoryComparison | null>(null);
  const [comparisons, setComparisons] = useState<HistoryComparisonSummary[]>([]);
  const [historyError, setHistoryError] = useState("");
  const [historyBusy, setHistoryBusy] = useState(false);
  const [historyEpoch, setHistoryEpoch] = useState(0);
  const [historyPage, setHistoryPage] = useState(1);
  const [historyTotal, setHistoryTotal] = useState(0);
  const request = useRef<AbortController | null>(null);
  const baselineUrl = useImageUrl(baseline);
  const currentUrl = useImageUrl(current);
  useEffect(() => () => { const controller = request.current; request.current = null; controller?.abort(); }, []);
  const refreshHistory = useCallback((signal: AbortSignal) => {
    setHistoryBusy(true); setHistoryError("");
    return listHistoryComparisons(token, signal, historyPage).then((data) => { if (!signal.aborted) { setComparisons(data.comparisons); setHistoryTotal(data.total); const lastPage = Math.max(1, Math.ceil(data.total / 20)); if (historyPage > lastPage) setHistoryPage(lastPage); } }).catch(() => { if (!signal.aborted) setHistoryError("暂时无法读取历史记录，可手动刷新。"); }).finally(() => { if (!signal.aborted) setHistoryBusy(false); });
  }, [token, historyPage]);
  useEffect(() => { const controller = new AbortController(); void refreshHistory(controller.signal); return () => controller.abort(); }, [refreshHistory, historyEpoch]);

  function clearResult() {
    request.current?.abort(); request.current = null;
    setBusy(false); setResult(null); setSaved(null); setError("");
  }
  function selectFile(file: File | null, setter: (value: File | null) => void) {
    clearResult();
    if (file && file.size > 20 * 1024 * 1024) { setError("单张图片不能超过20MB"); return; }
    setter(file);
  }
  async function run() {
    if (!baseline || !current) return;
    clearResult();
    const controller = new AbortController(); request.current = controller; setBusy(true);
    try {
      const data = await createHistoryComparison(token, baseline, current, controller.signal);
      if (request.current === controller) { setResult(data.result); setSaved(data); setHistoryPage(1); setHistoryEpoch((value) => value + 1); }
    } catch (reason) {
      if (!controller.signal.aborted) setError(reason instanceof Error ? reason.message : "对比失败");
    } finally {
      if (request.current === controller) { setBusy(false); request.current = null; }
    }
  }
  async function openComparison(id: string) {
    clearResult(); setBaseline(null); setCurrent(null);
    const controller = new AbortController(); request.current = controller; setBusy(true);
    try {
      const data = await getHistoryComparison(token, id, controller.signal);
      if (request.current === controller) { setSaved(data); setResult(data.result); }
    } catch (reason) { if (!controller.signal.aborted && request.current === controller) setError(reason instanceof Error ? reason.message : "读取历史对比失败"); }
    finally { if (request.current === controller) { setBusy(false); request.current = null; } }
  }
  function download() {
    if (!result) return;
    const url = URL.createObjectURL(new Blob([JSON.stringify(result, null, 2)], { type: "application/json" }));
    const anchor = document.createElement("a"); anchor.href = url; anchor.download = "history-compare.json"; anchor.click();
    setTimeout(() => URL.revokeObjectURL(url), 1000);
  }
  const panels = [
    { title: "历史照片", file: baseline, url: baselineUrl || saved?.baseline_url, setter: setBaseline, target: false },
    { title: "当前照片", file: current, url: currentUrl || saved?.current_url, setter: setCurrent, target: true },
  ];
  return <main className="workspace history-workspace">
    <div className="history-heading"><h1>历史对比</h1><div className="history-actions">
      <button className="icon-button" title="交换历史与当前照片" disabled={busy || !baseline || !current} onClick={() => { clearResult(); setBaseline(current); setCurrent(baseline); }}><ArrowLeftRight size={18} /></button>
      <button className="icon-button" title="清空照片" disabled={!baseline && !current && !saved} onClick={() => { clearResult(); setBaseline(null); setCurrent(null); }}><X size={18} /></button>
      <button className="primary-button" disabled={busy || !baseline || !current} onClick={run}>{busy ? <LoaderCircle size={16} className="spin" /> : <Play size={16} />}{busy ? "对比中" : "开始对比"}</button>
    </div></div>
    {error && <div className="notice error" role="alert">{error}</div>}
    <section className="history-saved"><header><h2>已保存的对比</h2><button className="secondary-button" disabled={historyBusy} onClick={() => setHistoryEpoch((value) => value + 1)}>刷新对比记录</button></header>{historyError && <p className="form-error" role="alert">{historyError}</p>}<div className="history-saved-list">{comparisons.map((item) => <button key={item.comparison_id} className={saved?.comparison_id === item.comparison_id ? "selected" : ""} onClick={() => void openComparison(item.comparison_id)}><strong>{item.diff_summary || (item.status === "matched" ? `${item.change_count} 处变化` : "无法比较")}</strong><small>{new Date(item.created_at).toLocaleString("zh-CN")}</small></button>)}{!historyBusy && comparisons.length === 0 && <p>尚无保存的对比；完成一次对比后将自动保存。</p>}</div><RecordPagination label="对比记录" page={historyPage} total={historyTotal} busy={historyBusy} onChange={setHistoryPage} /></section>
    {busy && <p role="status">正在处理对比，请稍候…</p>}
    <div className="history-panes">{panels.map((panel) => <section className="history-pane" key={panel.title}>
      <header><h2>{panel.title}</h2><label className="secondary-button history-upload"><ImagePlus size={16} />选择照片<input aria-label={`选择${panel.title}`} type="file" accept="image/jpeg,image/png,image/webp" onChange={(event) => { selectFile(event.target.files?.[0] ?? null, panel.setter); event.target.value = ""; }} /></label></header>
      <div className="history-image-stage">{panel.url ? <div className="history-image-wrap"><img src={panel.url} alt={panel.title} onError={() => setError(`${panel.title}无法预览，请选择有效图片`)} />
        {panel.target && showRegions && result && <svg className="history-regions" viewBox="0 0 1000 1000" preserveAspectRatio="none" aria-label="当前照片变化区域">{result.polygons.map((polygon, index) => <polygon key={index} points={polygon.map(([x, y]) => `${x * 1000},${y * 1000}`).join(" ")} vectorEffect="non-scaling-stroke"><title>变化区域 {index + 1}</title></polygon>)}</svg>}
      </div> : <div className="history-empty"><ImagePlus size={28} /><span>尚未选择{panel.title}</span></div>}</div>
      <div className="history-filename" title={panel.file?.name}>{panel.file?.name ?? (saved ? "已保存照片" : "未选择照片")}</div>
    </section>)}</div>
    {result && <section className="history-result" aria-live="polite"><div className="history-result-heading"><h2>{result.status === "matched" ? result.change_count ? `${result.change_count} 处变化` : "无变化" : "无法比较"}</h2><div className="history-actions"><label><input type="checkbox" checked={showRegions} onChange={(event) => setShowRegions(event.target.checked)} />变化标注</label><button className="icon-button" title="下载对比结果" onClick={download}><Download size={18} /></button></div></div>
      <p>{result.diff_summary}</p>{result.semantic.summary && <div><h3 className="history-semantic-scope">局部变化描述 · {result.semantic.region_count} 处</h3><p className="history-semantic">{result.semantic.summary}</p></div>}
      {result.semantic.error && <p className="history-warning" role="status">{result.semantic.error}</p>}
      <div className="history-metadata"><span>当前图 {result.current_width} × {result.current_height}</span><span>历史图 {result.baseline_width} × {result.baseline_height}</span><span>重叠范围 {(result.alignment.overlap * 100).toFixed(1)}%</span></div>
    </section>}
  </main>;
}
