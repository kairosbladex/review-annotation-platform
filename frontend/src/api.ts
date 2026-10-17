import type { AutoAnnotationJob, Job, ModelProfile, Sample } from "./types";

const API_BASE = import.meta.env.VITE_API_BASE ?? "/api/v1";

export type HistoryCompareResult = {
  status: "matched" | "not_found";
  polygons: [number, number][][];
  change_count: number;
  current_width: number;
  current_height: number;
  baseline_width: number;
  baseline_height: number;
  coordinate_space: "normalized_current";
  alignment: { method: string; inliers: number; overlap: number };
  diff_summary: string;
  semantic: { status: string; summary: string | null; error: string | null; region_count: number };
};

export async function compareHistoryImages(token: string, baseline: File, current: File, signal: AbortSignal) {
  const form = new FormData(); form.append("baseline", baseline); form.append("current", current);
  const response = await fetch(`${API_BASE}/history-compare`, { method: "POST", headers: { Authorization: `Bearer ${token}` }, body: form, signal });
  const body = await response.json().catch(() => ({}));
  if (!response.ok) throw new Error(typeof body.detail === "string" ? body.detail : `对比失败（${response.status}）`);
  return body as HistoryCompareResult;
}

async function request<T>(path: string, init: RequestInit = {}, token?: string): Promise<T> {
  const headers = new Headers(init.headers);
  headers.set("Content-Type", "application/json");
  if (token) headers.set("Authorization", `Bearer ${token}`);
  const response = await fetch(`${API_BASE}${path}`, { ...init, headers });
  const body = await response.json().catch(() => ({}));
  if (!response.ok) {
    throw new Error(body.detail ?? `请求失败（${response.status}）`);
  }
  return body as T;
}

export async function login(username: string, password: string) {
  return request<{ access_token: string; user: { id: string; username: string } }>("/auth/login", {
    method: "POST",
    body: JSON.stringify({ username, password }),
  });
}

export async function listSamples(token: string, status = "all", offset = 0, limit = 100, className = "", datasetFilter = "all") {
  const params = new URLSearchParams({ status_filter: status, limit: String(limit), offset: String(offset) });
  if (className) params.set("class_name", className);
  if (datasetFilter !== "all") params.set("dataset_filter", datasetFilter);
  return request<{ samples: Sample[]; pagination: { limit: number; offset: number; total: number } }>(`/samples?${params.toString()}`, {}, token);
}

export async function getSample(token: string, sampleId: string) {
  return request<Sample>(`/samples/${sampleId}`, {}, token);
}

export async function getAssetUrl(token: string, assetId: string) {
  return request<{ asset_id: string; url: string; expires_in: number }>(`/assets/${assetId}/url`, {}, token);
}

export async function labelCandidate(token: string, candidateId: string, payload: { label: "true" | "false" | null; note: string; expected_revision?: number }) {
  return request<{ sample: Sample }>(`/candidates/${candidateId}/annotation`, {
    method: "PATCH",
    body: JSON.stringify(payload),
  }, token);
}

export async function listProfiles(token: string) {
  return request<{ profiles: ModelProfile[] }>("/model-profiles", {}, token);
}

export async function createProfile(token: string, payload: { profile_key: string; name: string; model_name: string; model_version: string; adapter_type: string; config: Record<string, unknown> }) {
  return request<ModelProfile>("/model-profiles", { method: "POST", body: JSON.stringify(payload) }, token);
}

export async function createModelRun(token: string, sampleIds: string[], modelProfileId: string) {
  return request<{ run_id: string; job_id: string; status: string }>("/model-runs", {
    method: "POST",
    body: JSON.stringify({ sample_ids: sampleIds, model_profile_id: modelProfileId, options: {} }),
  }, token);
}

export async function createAutoAnnotations(token: string, sampleIds: string[], options: Record<string, unknown> = {}) {
  return request<AutoAnnotationJob>("/auto-annotations", {
    method: "POST",
    body: JSON.stringify({ sample_ids: sampleIds, options }),
  }, token);
}

export async function getJob(token: string, jobId: string) {
  return request<Job>(`/jobs/${jobId}`, {}, token);
}

export async function createExport(token: string, runId: string, formats: string[] = ["jsonl", "yolo"]) {
  return request<{ export_id: string; job_id: string; status: string }>("/exports", {
    method: "POST",
    body: JSON.stringify({ run_id: runId, formats, filters: {} }),
  }, token);
}

export async function uploadImages(token: string, files: File[]) {
  const summary = { imported: 0, skipped: 0, failed: 0, errors: [] as string[] };
  for (const file of files) {
    const data = new FormData(); data.append("files", file);
    try {
      const response = await fetch(`${API_BASE}/imports/uploads`, { method: "POST", headers: { Authorization: `Bearer ${token}` }, body: data });
      const body = await response.json().catch(() => ({}));
      if (!response.ok) { summary.failed++; summary.errors.push(`HTTP ${response.status}`); continue; }
      summary.imported += body.imported ?? 0; summary.skipped += body.skipped ?? 0; summary.failed += body.failed ?? 0;
      if (body.failed > 0) summary.errors.push("图片格式不支持或无法读取");
    } catch { summary.failed++; summary.errors.push("网络连接失败"); }
  }
  return summary;
}

export type HistoryComparison = { comparison_id: string; created_at: string; baseline_url: string; current_url: string; result: HistoryCompareResult };
export type HistoryComparisonSummary = { comparison_id: string; created_at: string; status: string; change_count: number; diff_summary: string };
export function listHistoryComparisons(token: string, signal?: AbortSignal, page = 1, limit = 20) {
  return request<{ comparisons: HistoryComparisonSummary[]; total: number }>(`/history-comparisons?page=${page}&limit=${limit}`, { signal }, token);
}
export function getHistoryComparison(token: string, id: string, signal?: AbortSignal) {
  return request<HistoryComparison>(`/history-comparisons/${encodeURIComponent(id)}`, { signal }, token);
}
export async function createHistoryComparison(token: string, baseline: File, current: File, signal: AbortSignal) {
  const form = new FormData(); form.append("baseline", baseline); form.append("current", current);
  const response = await fetch(`${API_BASE}/history-comparisons`, { method: "POST", headers: { Authorization: `Bearer ${token}` }, body: form, signal });
  const body = await response.json().catch(() => ({}));
  if (!response.ok) throw new Error(typeof body.detail === "string" ? body.detail : `对比失败（${response.status}）`);
  return body as HistoryComparison;
}

export type RetrievalMedia = { media_id: string; filename: string; media_type: string; size_bytes: number };
export type RetrievalDetection = { class_name: string; confidence: number; bbox: [number, number, number, number] };
export type RetrievalResult = { file_index: number | null; frame_index?: number | null; timestamp_seconds?: number | null; image_url?: string | null; thumbnail_url?: string | null; detections: RetrievalDetection[] };
export type RetrievalTask = {
  task_id: string; status: string; query: string; message?: string | null; media_count: number; total_hits: number;
  processed_count?: number; total_count?: number; video_progress?: number | { progress?: number; current_seconds?: number; duration_seconds?: number; processed_frames?: number; sampled_frames?: number; total_frames?: number; processed_samples?: number; sample_count?: number; estimated_total_samples?: number; phase?: string } | null; error_code?: string | null;
  sync_error?: string | null; created_at: string; finished_at?: string | null; results?: RetrievalResult[];
  file_errors?: { file_index?: number | null; message?: string; error?: string }[];
};
export function getCapabilities(token: string, signal?: AbortSignal) {
  return request<{ retrieval_enabled: boolean }>("/capabilities", { signal }, token);
}
export async function uploadRetrievalMedia(token: string, file: File, signal: AbortSignal) {
  const body = new FormData(); body.append("file", file);
  const response = await fetch(`${API_BASE}/retrieval/media`, { method: "POST", headers: { Authorization: `Bearer ${token}` }, body, signal });
  const data = await response.json().catch(() => ({}));
  if (!response.ok) throw new Error(typeof data.detail === "string" ? data.detail : `上传失败（${response.status}）`);
  return data as RetrievalMedia;
}
export function createRetrievalTask(token: string, payload: { query: string; media_ids: string[]; prompt_media_ids?: string[]; confidence: number; sample_interval_seconds: number }, idempotencyKey: string, signal?: AbortSignal) {
  return request<RetrievalTask>("/retrieval/tasks", { method: "POST", headers: { "Idempotency-Key": idempotencyKey }, body: JSON.stringify(payload), signal }, token);
}
export function listRetrievalTasks(token: string, signal?: AbortSignal, offset = 0, limit = 20) {
  return request<{ tasks: RetrievalTask[]; total: number }>(`/retrieval/tasks?offset=${offset}&limit=${limit}`, { signal }, token);
}
export function getRetrievalTask(token: string, id: string, signal?: AbortSignal) {
  return request<RetrievalTask>(`/retrieval/tasks/${encodeURIComponent(id)}`, { signal }, token);
}
export function cancelRetrievalTask(token: string, id: string, signal?: AbortSignal) {
  return request<RetrievalTask>(`/retrieval/tasks/${encodeURIComponent(id)}/cancel`, { method: "POST", signal }, token);
}
