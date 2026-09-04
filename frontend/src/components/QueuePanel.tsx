import { ArrowLeft, ArrowRight, Check, CircleDot, Filter, LoaderCircle, Search, Upload } from "lucide-react";
import type { Sample, SampleStatus } from "../types";

interface QueuePanelProps {
  samples: Sample[];
  selectedId: string | null;
  statusFilter: string;
  classFilter: string;
  datasetFilter: string;
  classOptions: string[];
  query: string;
  pageOffset: number;
  pageSize: number;
  total: number;
  onSelect: (id: string) => void;
  onFilter: (value: string) => void;
  onClassFilter: (value: string) => void;
  onDatasetFilter: (value: string) => void;
  onQuery: (value: string) => void;
  onPageChange: (page: number) => void;
  onUpload: (files: File[]) => void;
  uploading: boolean;
}

const statusLabels: Record<SampleStatus, string> = {
  imported: "待配置",
  pending: "待标注",
  in_progress: "进行中",
  completed: "已完成",
  no_candidates: "无目标",
};

function formatTime(value: string) {
  return new Intl.DateTimeFormat("zh-CN", { month: "2-digit", day: "2-digit", hour: "2-digit", minute: "2-digit" }).format(new Date(value));
}

export function QueuePanel({ samples, selectedId, statusFilter, classFilter, datasetFilter, classOptions, query, pageOffset, pageSize, total, onSelect, onFilter, onClassFilter, onDatasetFilter, onQuery, onPageChange, onUpload, uploading }: QueuePanelProps) {
  const filtered = samples.filter((sample) => {
    const text = `${sample.session_id ?? ""} ${sample.source_event_id ?? ""} ${sample.rule_code ?? ""}`.toLowerCase();
    return text.includes(query.toLowerCase());
  });

  return (
    <aside className="queue-panel panel">
      <div className="panel-heading">
        <div>
          <span className="eyebrow">REVIEW QUEUE</span>
          <h2>复核队列</h2>
        </div>
        <label className="icon-button upload-button" title="上传图片">
          {uploading ? <LoaderCircle className="spin" size={17} /> : <Upload size={17} />}
          <input type="file" accept="image/*" multiple hidden onChange={(event) => {
            if (event.target.files?.length) onUpload(Array.from(event.target.files));
            event.currentTarget.value = "";
          }} />
        </label>
      </div>

      <div className="queue-tools">
        <div className="search-field"><Search size={15} /><input value={query} onChange={(event) => onQuery(event.target.value)} placeholder="搜索会话、事件或规则" /></div>
        <div className="filter-row"><Filter size={14} /><select value={statusFilter} onChange={(event) => onFilter(event.target.value)} aria-label="按状态筛选"><option value="needs_review">待复核</option><option value="all">全部状态</option><option value="pending">未开始</option><option value="in_progress">进行中</option><option value="completed">已完成</option><option value="imported">待配置</option></select><select value={classFilter} onChange={(event) => onClassFilter(event.target.value)} aria-label="按类别筛选">{classOptions.map((name) => <option key={name || "all"} value={name}>{name || "全部类别"}</option>)}</select><select value={datasetFilter} onChange={(event) => onDatasetFilter(event.target.value)} aria-label="按数据集筛选"><option value="all">全部数据集</option><option value="v2">V2 待标注</option><option value="other">非 V2</option></select></div>
      </div>

      <div className="queue-list">
        {filtered.length === 0 && <div className="empty-state">暂无匹配样本<br /><span>可以从右上角上传复核图片</span></div>}
        {filtered.map((sample) => {
          const isSelected = sample.sample_id === selectedId;
          const targetCount = datasetFilter === "v2" ? sample.v2_target_count : sample.counts.total;
          const pendingCount = datasetFilter === "v2" ? sample.v2_pending_count : sample.counts.pending;
          const labeledCount = targetCount - pendingCount;
          const progress = targetCount ? Math.round((labeledCount / targetCount) * 100) : 0;
          return <button className={`queue-item ${isSelected ? "selected" : ""}`} key={sample.sample_id} onClick={() => onSelect(sample.sample_id)}>
            <div className="queue-thumb"><span className="thumb-placeholder">{sample.source_type === "review_service" ? "RV" : "UP"}</span><span className={`status-dot ${sample.status}`} /></div>
            <div className="queue-copy"><div className="queue-title"><strong>{sample.session_id ?? sample.sample_id.slice(0, 8)}</strong><span className={`status-text ${sample.status}`}>{statusLabels[sample.status] ?? sample.status}</span></div><div className="queue-meta">{sample.rule_code ?? "未指定规则"} · {formatTime(sample.created_at)}</div><div className="queue-badges">{sample.dataset_tags.includes("v2") && <span className="dataset-badge">V2 {sample.v2_pending_count}/{sample.v2_target_count}</span>}</div><div className="progress-line"><span style={{ width: `${progress}%` }} /><small>{labeledCount}/{targetCount}</small></div></div>
            {sample.status === "completed" ? <Check className="queue-check" size={16} /> : <CircleDot className="queue-dot" size={16} />}
          </button>;
        })}
      </div>
      <div className="queue-footer"><span>{total === 0 ? "0" : `${pageOffset + 1}-${Math.min(pageOffset + pageSize, total)}`} / {total} 个样本</span><div className="queue-pagination"><button type="button" aria-label="上一页" title="上一页" disabled={pageOffset === 0} onClick={() => onPageChange(Math.max(0, pageOffset / pageSize - 1))}><ArrowLeft size={13} /></button><span>{Math.floor(pageOffset / pageSize) + 1} / {Math.max(1, Math.ceil(total / pageSize))}</span><button type="button" aria-label="下一页" title="下一页" disabled={pageOffset + pageSize >= total} onClick={() => onPageChange(pageOffset / pageSize + 1)}><ArrowRight size={13} /></button></div><span className="live-indicator"><i />实时队列</span></div>
    </aside>
  );
}
