export type SampleStatus = "imported" | "pending" | "in_progress" | "completed" | "no_candidates";
export type AnnotationLabel = "true" | "false" | null;

export interface Counts {
  total: number;
  labeled: number;
  pending: number;
  true: number;
  false: number;
}

export interface Annotation {
  annotation_id: string | null;
  label: AnnotationLabel;
  note: string | null;
  labeled_by: string | null;
  labeled_at: string | null;
  revision: number;
}

export interface Candidate {
  candidate_id: string;
  canonical_id: string | null;
  dataset_tag: "v2" | null;
  target_index: number;
  class_id: number | null;
  class_name: string;
  confidence: number | null;
  bbox: [number, number, number, number];
  raw_decision: string | null;
  yolo_decision: string | null;
  llm_decision: string | null;
  reason_code: string | null;
  error_code: string | null;
  reasoning: string;
  annotation: Annotation;
}

export interface Asset {
  asset_id: string;
  kind: "original" | "comparison" | string;
  bucket: string;
  object_key: string;
  sha256: string | null;
  status: string;
}

export interface AnnotationTask {
  annotation_task_id: string;
  model_run_id: string;
  status: SampleStatus;
  counts: Counts;
  model_run: {
    model_name: string | null;
    model_version: string | null;
    source_type: string | null;
    status: string | null;
    parameters: Record<string, unknown>;
  };
  candidates: Candidate[];
}

export interface Sample {
  sample_id: string;
  source_type: string;
  source_event_id: string | null;
  source_service: string | null;
  session_id: string | null;
  capture_time: string | null;
  camera_id: string | null;
  rule_code: string | null;
  metadata: Record<string, unknown>;
  asset_status: string;
  created_at: string;
  status: SampleStatus;
  counts: Counts;
  dataset_tags: string[];
  v2_target_count: number;
  v2_pending_count: number;
  annotation_task_id: string | null;
  assets: Asset[];
  runs: AnnotationTask[];
  candidates: Candidate[];
}

export interface ModelProfile {
  model_profile_id: string;
  profile_key: string;
  name: string;
  model_name: string;
  model_version: string;
  adapter_type: string;
}

export interface Job {
  job_id: string;
  status: string;
  total: number;
  completed: number;
  failed: number;
  error_summary: string | null;
}

/**
 * Progress returned by the AI auto-annotation endpoint.  The create response
 * intentionally contains only the public progress fields; the job detail
 * endpoint may add an error summary once processing has failed.
 */
export type AutoAnnotationJob = Omit<Job, "error_summary"> & {
  error_summary?: string | null;
};
