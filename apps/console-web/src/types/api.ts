export type RunState =
  | "idle"
  | "starting"
  | "running"
  | "stopping"
  | "stopped"
  | "completed"
  | "failed";

export interface ModelSummary {
  id: string;
  model_name: string;
  model_type: string;
  head_type: string;
  subject_id: string | null;
  dataset_name: string;
  task: string;
  window_sec: number;
  step_sec: number;
  sample_rate: number;
  target_channels: number;
  schema_version: number;
  runtime_verified: boolean;
  live_verified: boolean;
  package_version: string;
  balanced_accuracy: number | null;
  macro_f1: number | null;
  warning_message: string | null;
}

export interface DatasetSummary {
  id: string;
  name: string;
  subject_id: string;
  sessions: string[];
  trial_count: number;
  channel_count: number;
  sample_rate: number;
  unit: string;
  class_names: string[];
  qc_status: string;
}

export interface PredictionPayload {
  window_id: number;
  trial_id: number | null;
  predicted_class: number;
  predicted_name: string;
  command: string;
  confidence: number;
  probabilities: number[];
  expected_class_id: number | null;
  expected_class_name: string | null;
}

export interface LatencyPayload {
  prepare_ms: number;
  inference_ms: number;
  total_ms: number;
  p50_ms: number | null;
  p95_ms: number | null;
}

export interface RuntimeHealthPayload {
  successful_windows: number;
  failed_windows: number;
  expected_windows: number | null;
}

export interface InputContractPayload {
  safe: boolean;
  source_channels: number;
  target_channels: number;
  valid_channels: number;
  window_sec: number;
  target_sample_rate: number;
  reason?: string;
}

export interface RunSummary {
  id: string;
  run_type: string;
  state: RunState;
  dataset_id: string | null;
  subject_id: string | null;
  session: string | null;
  model_id: string;
  created_at: number;
  successful_windows: number;
  failed_windows: number;
  expected_windows: number | null;
}

export interface SystemStatus {
  device: { status: string; source: string | null; health?: Record<string, unknown> };
  runtime: { state: string; run_id?: string; model_id?: string };
  compute: { cuda_available: boolean };
}

export type RunEvent = {
  type: "state" | "prediction" | "latency" | "runtime_health" | "device_health" | "input_contract" | "trigger" | "window" | "error";
  run_id: string;
  timestamp: number;
  payload: Record<string, unknown>;
};
