import { ArrowLeft, ArrowRight, Check, CircleAlert, FileText, RotateCcw, Save, Tag, X } from "lucide-react";
import { useEffect, useMemo, useState } from "react";
import type { Candidate } from "../types";

interface InspectorProps {
  candidate: Candidate | null;
  candidates: Candidate[];
  onSelect: (id: string) => void;
  onLabel: (label: "true" | "false" | null, note: string, options?: { advance?: boolean }) => Promise<void>;
  saving: boolean;
}

function confidence(value: number | null) {
  return value == null ? "—" : `${Math.round(value * 100)}%`;
}

export function Inspector({ candidate, candidates, onSelect, onLabel, saving }: InspectorProps) {
  const [note, setNote] = useState("");
  useEffect(() => setNote(candidate?.annotation.note ?? ""), [candidate?.candidate_id, candidate?.annotation.note]);
  const index = candidate ? candidates.findIndex((item) => item.candidate_id === candidate.candidate_id) : -1;
  const hasPrev = index > 0;
  const hasNext = index >= 0 && index < candidates.length - 1;
  const labelText = useMemo(() => candidate?.annotation.label === "true" ? "真目标" : candidate?.annotation.label === "false" ? "假目标" : "待判断", [candidate?.annotation.label]);

  if (!candidate) return <aside className="inspector panel"><div className="empty-inspector"><Tag size={30} /><strong>选择一个目标</strong><span>点击中间画布上的红框，开始人工判断</span></div></aside>;

  const save = (label: "true" | "false" | null, advance = true) => onLabel(label, note, { advance });
  return <aside className="inspector panel">
    <div className="inspector-heading"><div><span className="eyebrow">当前目标 {index + 1} / {candidates.length}</span><h2>{candidate.class_name}</h2></div><div className="inspector-tags">{candidate.dataset_tag === "v2" && <span className="dataset-badge">V2</span>}<span className={`label-chip ${candidate.annotation.label ?? "pending"}`}>{labelText}</span></div></div>
    <div className="target-index"><button disabled={!hasPrev} onClick={() => hasPrev && onSelect(candidates[index - 1].candidate_id)}><ArrowLeft size={15} />上一个</button><span>{index + 1} / {candidates.length}</span><button disabled={!hasNext} onClick={() => hasNext && onSelect(candidates[index + 1].candidate_id)}>下一个<ArrowRight size={15} /></button></div>
    <div className="decision-actions"><button className={`decision true ${candidate.annotation.label === "true" ? "active" : ""}`} disabled={saving} onClick={() => save("true")}><Check size={18} /><span>真目标</span><kbd>1</kbd></button><button className={`decision false ${candidate.annotation.label === "false" ? "active" : ""}`} disabled={saving} onClick={() => save("false")}><X size={18} /><span>假目标</span><kbd>2</kbd></button></div>
    <div className="detail-grid"><div><span>置信度</span><strong>{confidence(candidate.confidence)}</strong></div><div><span>目标索引</span><strong>#{candidate.target_index}</strong></div><div><span>原始判定</span><strong>{candidate.raw_decision ?? "—"}</strong></div><div><span>原因码</span><strong>{candidate.reason_code ?? "—"}</strong></div></div>
    <div className="inspector-block"><div className="block-label"><CircleAlert size={14} />模型判断链</div><dl><div><dt>YOLOE</dt><dd>{candidate.yolo_decision ?? "—"}</dd></div><div><dt>VLM / LLM</dt><dd>{candidate.llm_decision ?? "未返回"}</dd></div><div><dt>错误码</dt><dd>{candidate.error_code ?? "—"}</dd></div></dl></div>
    <div className="inspector-block"><div className="block-label"><FileText size={14} />模型说明</div><p className="reasoning">{candidate.reasoning || "模型未提供说明"}</p></div>
    <div className="inspector-block note-block"><div className="block-label"><FileText size={14} />人工备注</div><textarea value={note} onChange={(event) => setNote(event.target.value)} placeholder="记录误检原因、场景或后续训练建议…" /><button className="secondary-button" disabled={saving} onClick={() => save(candidate.annotation.label, false)}><Save size={15} />保存备注</button></div>
    <div className="inspector-footer"><span>revision {candidate.annotation.revision}</span><button className="ghost-button" disabled={saving || candidate.annotation.label === null} onClick={() => save(null, false)}><RotateCcw size={14} />撤销标注</button></div>
  </aside>;
}
