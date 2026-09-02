export function nextPendingCandidateId(
  candidates: { candidate_id: string; annotation: { label: string | null } }[],
  currentId: string | null,
): string | null {
  if (candidates.length === 0) return null;
  const currentIndex = candidates.findIndex((item) => item.candidate_id === currentId);
  const start = currentIndex >= 0 ? currentIndex + 1 : 0;
  for (let offset = 0; offset < candidates.length; offset += 1) {
    const item = candidates[(start + offset) % candidates.length];
    if (item.annotation.label === null) return item.candidate_id;
  }
  return null;
}

export function nextPendingSampleId(
  samples: { sample_id: string; counts: { pending: number } }[],
  currentId: string | null,
): string | null {
  if (samples.length === 0) return null;
  const currentIndex = samples.findIndex((item) => item.sample_id === currentId);
  const start = currentIndex >= 0 ? currentIndex + 1 : 0;
  for (let offset = 0; offset < samples.length; offset += 1) {
    const item = samples[(start + offset) % samples.length];
    if (item.sample_id !== currentId && item.counts.pending > 0) return item.sample_id;
  }
  return null;
}
