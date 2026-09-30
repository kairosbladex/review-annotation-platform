import { Crosshair, ImageOff, ZoomIn, ZoomOut } from "lucide-react";
import { useState } from "react";
import type { Candidate } from "../types";

interface ReviewCanvasProps {
  imageUrl: string | null;
  cleanImageUrl?: string | null;
  cleanBboxes?: Record<string, [number,number,number,number]>;
  candidates: Candidate[];
  selectedId: string | null;
  onSelect: (id: string) => void;
}

export function ReviewCanvas({ imageUrl, cleanImageUrl, cleanBboxes, candidates, selectedId, onSelect }: ReviewCanvasProps) {
  const [scale, setScale] = useState(1);
  const [showClean, setShowClean] = useState(false);
  const activeUrl = showClean && cleanImageUrl ? cleanImageUrl : imageUrl;
  return <section className="canvas-panel panel">
    <div className="canvas-toolbar"><div className="canvas-label"><Crosshair size={15} /><span>模型候选框</span><span className="canvas-count">{selectedId ? `${candidates.findIndex((item) => item.candidate_id === selectedId) + 1}/${candidates.length}` : `${candidates.length} targets`}</span></div><div className="zoom-actions">{cleanImageUrl && <button onClick={() => setShowClean((v) => !v)}>{showClean ? "查看接入截图" : "查看干净原图"}</button>}<button onClick={() => setScale((value) => Math.max(0.5, Number((value - 0.25).toFixed(2))))} title="缩小"><ZoomOut size={15} /></button><span>{Math.round(scale * 100)}%</span><button onClick={() => setScale((value) => Math.min(4, Number((value + 0.25).toFixed(2))))} title="放大"><ZoomIn size={15} /></button></div></div>
    <div className="image-stage">
      {activeUrl ? <div className="image-zoom" style={{ transform: `scale(${scale})` }}><img src={activeUrl} alt="待复核原图" />{candidates.map((candidate) => {
        const [x1, y1, x2, y2] = showClean && cleanImageUrl ? cleanBboxes?.[String(candidate.target_index)] ?? candidate.bbox : candidate.bbox;
        const selected = candidate.candidate_id === selectedId;
        const tiny = x2 - x1 < 0.04 || y2 - y1 < 0.04;
        const labelBelow = y1 < 0.04;
        return <button key={candidate.candidate_id} className={`bbox ${selected ? "selected" : ""} ${candidate.annotation.label ?? "pending"} ${candidate.dataset_tag === "v2" ? "v2" : ""} ${tiny ? "tiny" : ""} ${labelBelow ? "label-below" : ""}`} style={{ left: `${x1 * 100}%`, top: `${y1 * 100}%`, width: `${(x2 - x1) * 100}%`, height: `${(y2 - y1) * 100}%` }} onClick={() => onSelect(candidate.candidate_id)} title={`${candidate.dataset_tag === "v2" ? "V2 · " : ""}${candidate.target_index + 1} · ${candidate.class_name}`}><span>{selected ? `${candidate.target_index + 1} ${candidate.class_name}` : candidate.target_index + 1}</span></button>;
      })}</div> : <div className="image-missing"><ImageOff size={34} /><strong>原图暂不可用</strong><span>资源缺失或尚未接入 MinIO</span></div>}
    </div>
    <div className="canvas-legend"><span><i className="legend-box uncertain" />存疑候选</span><span><i className="legend-box true" />真目标</span><span><i className="legend-box false" />假目标</span><span className="canvas-hint">点击红框查看详情 · 快捷键 1 / 2 标注</span></div>
  </section>;
}
