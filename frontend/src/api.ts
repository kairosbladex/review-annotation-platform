import type { Job, ModelProfile, Sample } from "./types";

const API_BASE = import.meta.env.VITE_API_BASE ?? "/api/v1";

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

export async function listSamples(token: string, status = "all") {
  return request<{ samples: Sample[]; pagination: { total: number } }>(`/samples?status_filter=${encodeURIComponent(status)}&limit=100`, {}, token);
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
  const data = new FormData();
  files.forEach((file) => data.append("files", file));
  const response = await fetch(`${API_BASE}/imports/uploads`, {
    method: "POST",
    headers: { Authorization: `Bearer ${token}` },
    body: data,
  });
  const body = await response.json().catch(() => ({}));
  if (!response.ok) throw new Error(body.detail ?? `上传失败（${response.status}）`);
  return body as { imported: number; skipped: number; failed: number };
}
