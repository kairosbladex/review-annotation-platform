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

export type AdvanceAfterLabelResult = {
  sampleId: string;
  candidateId: string | null;
  kind: "next_box" | "next_sample" | "queue_done";
};

export function advanceAfterLabel(args: {
  candidates: { candidate_id: string; annotation: { label: string | null } }[];
  labeledId: string;
  currentSampleId: string;
  queueSamples: { sample_id: string; counts: { pending: number } }[];
}): AdvanceAfterLabelResult {
  const nextBox = nextPendingCandidateId(args.candidates, args.labeledId);
  if (nextBox) {
    return { sampleId: args.currentSampleId, candidateId: nextBox, kind: "next_box" };
  }
  const nextSample = nextPendingSampleId(args.queueSamples, args.currentSampleId);
  if (nextSample) {
    return { sampleId: nextSample, candidateId: null, kind: "next_sample" };
  }
  return { sampleId: args.currentSampleId, candidateId: args.labeledId, kind: "queue_done" };
}

export function keepQueueSelection(
  currentId: string | null,
  samples: { sample_id: string }[],
  options?: { stealIfMissing?: boolean },
): string | null {
  if (currentId && samples.some((item) => item.sample_id === currentId)) return currentId;
  if (currentId && options?.stealIfMissing !== true) return currentId;
  return samples[0]?.sample_id ?? null;
}
