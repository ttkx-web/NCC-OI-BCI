import type { DatasetSummary, ModelSummary } from "@/types/api";

export const API_BASE = process.env.NEXT_PUBLIC_CONSOLE_API_URL ?? "http://127.0.0.1:8000";

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(`${API_BASE}${path}`, {
    ...init,
    headers: { "Content-Type": "application/json", ...init?.headers },
  });
  if (!response.ok) {
    const body = (await response.json().catch(() => null)) as { detail?: string } | null;
    throw new Error(body?.detail ?? `API 请求失败（${response.status}）`);
  }
  return response.json() as Promise<T>;
}

export const consoleApi = {
  models: async () => (await request<{ items: ModelSummary[] }>("/api/v1/models")).items,
  datasets: async () => (await request<{ items: DatasetSummary[] }>("/api/v1/datasets")).items,
  startReplay: (payload: Record<string, unknown>) =>
    request<{ run_id: string; state: string }>("/api/v1/runs/replay", {
      method: "POST",
      body: JSON.stringify(payload),
    }),
  stopRun: (runId: string) => request(`/api/v1/runs/${runId}/stop`, { method: "POST" }),
  restartRun: (runId: string) => request(`/api/v1/runs/${runId}/restart`, { method: "POST" }),
};

