import { Crosshair, ImageOff, ZoomIn, ZoomOut } from "lucide-react";
import { useState } from "react";
import type { Candidate } from "../types";

interface ReviewCanvasProps {
  imageUrl: string | null;
  candidates: Candidate[];
  selectedId: string | null;
  onSelect: (id: string) => void;
}

export function ReviewCanvas({ imageUrl, candidates, selectedId, onSelect }: ReviewCanvasProps) {
  const [scale, setScale] = useState(1);
  return <section className="canvas-panel panel">
    <div className="canvas-toolbar"><div className="canvas-label"><Crosshair size={15} /><span>模型候选框</span><span className="canvas-count">{candidates.length} targets</span></div><div className="zoom-actions"><button onClick={() => setScale((value) => Math.max(0.75, value - 0.1))} title="缩小"><ZoomOut size={15} /></button><span>{Math.round(scale * 100)}%</span><button onClick={() => setScale((value) => Math.min(1.5, value + 0.1))} title="放大"><ZoomIn size={15} /></button></div></div>
    <div className="image-stage">
      {imageUrl ? <div className="image-zoom" style={{ transform: `scale(${scale})` }}><img src={imageUrl} alt="待复核原图" />{candidates.map((candidate) => {
        const [x1, y1, x2, y2] = candidate.bbox;
        const selected = candidate.candidate_id === selectedId;
        return <button key={candidate.candidate_id} className={`bbox ${selected ? "selected" : ""} ${candidate.annotation.label ?? "pending"}`} style={{ left: `${x1 * 100}%`, top: `${y1 * 100}%`, width: `${(x2 - x1) * 100}%`, height: `${(y2 - y1) * 100}%` }} onClick={() => onSelect(candidate.candidate_id)} title={`目标 ${candidate.target_index + 1}`}><span>{candidate.target_index + 1} · {candidate.class_name}</span></button>;
      })}</div> : <div className="image-missing"><ImageOff size={34} /><strong>原图暂不可用</strong><span>资源缺失或尚未接入 MinIO</span></div>}
    </div>
    <div className="canvas-legend"><span><i className="legend-box uncertain" />存疑候选</span><span><i className="legend-box true" />真目标</span><span><i className="legend-box false" />假目标</span><span className="canvas-hint">点击红框查看详情 · 快捷键 1 / 2 标注</span></div>
  </section>;
}
