export function RecordPagination({ label, page, total, busy, onChange }: { label: string; page: number; total: number; busy: boolean; onChange: (page: number) => void }) {
  const pages = Math.max(1, Math.ceil(total / 20));
  return <nav className="record-pagination" aria-label={`${label}分页`}>
    <button className="secondary-button" aria-label={`${label}上一页`} disabled={busy || page <= 1} onClick={() => onChange(page - 1)}>上一页</button>
    <span>第 {page} / {pages} 页 · 共 {total} 条</span>
    <button className="secondary-button" aria-label={`${label}下一页`} disabled={busy || page >= pages} onClick={() => onChange(page + 1)}>下一页</button>
  </nav>;
}
