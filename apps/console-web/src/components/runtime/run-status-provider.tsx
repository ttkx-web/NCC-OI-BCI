"use client";

import { createContext, useCallback, useContext, useEffect, useMemo, useRef, useState, type ReactNode } from "react";

import { consoleApi } from "@/lib/api/client";
import { connectRun } from "@/lib/websocket/run-stream";
import type { InputContractPayload, LatencyPayload, ModelSummary, RunEvent, RunState, RunSummary, RuntimeHealthPayload, SystemStatus } from "@/types/api";

const activeStates: RunState[] = ["starting", "running", "stopping"];
const stateLabels: Record<RunState, string> = {
  idle: "空闲", starting: "启动中", running: "运行中", stopping: "停止中",
  stopped: "已停止", completed: "已完成", failed: "失败",
};

type RuntimeStatus = {
  runs: RunSummary[];
  models: ModelSummary[];
  system: SystemStatus | null;
  activeRun: RunSummary | null;
  activeModel: ModelSummary | null;
  deviceHealth: Record<string, unknown>;
  latency: LatencyPayload | null;
  runtimeHealth: RuntimeHealthPayload | null;
  inputContract: InputContractPayload | null;
  recentEvents: RunEvent[];
  fatalReason: string | null;
  refresh: () => Promise<void>;
  registerActiveRun: (runId: string) => void;
};

const RuntimeStatusContext = createContext<RuntimeStatus | null>(null);

export const runStateLabel = (state: RunState | null | undefined) => state ? stateLabels[state] : "空闲";

export function RuntimeStatusProvider({ children }: { children: ReactNode }) {
  const [runs, setRuns] = useState<RunSummary[]>([]);
  const [models, setModels] = useState<ModelSummary[]>([]);
  const [system, setSystem] = useState<SystemStatus | null>(null);
  const [activeRunId, setActiveRunId] = useState<string | null>(null);
  const [deviceHealth, setDeviceHealth] = useState<Record<string, unknown>>({});
  const [latency, setLatency] = useState<LatencyPayload | null>(null);
  const [runtimeHealth, setRuntimeHealth] = useState<RuntimeHealthPayload | null>(null);
  const [inputContract, setInputContract] = useState<InputContractPayload | null>(null);
  const [recentEvents, setRecentEvents] = useState<RunEvent[]>([]);
  const [fatalReason, setFatalReason] = useState<string | null>(null);
  const socketRef = useRef<WebSocket | null>(null);

  const refresh = useCallback(async () => {
    const [nextRuns, nextSystem, nextModels] = await Promise.all([
      consoleApi.runs(), consoleApi.systemStatus(), consoleApi.models(),
    ]);
    setRuns(nextRuns); setSystem(nextSystem); setModels(nextModels);
    const active = nextRuns.find(run => activeStates.includes(run.state));
    setActiveRunId(active?.id ?? null);
    if (!active) {
      setDeviceHealth({}); setLatency(null); setRuntimeHealth(null); setInputContract(null); setFatalReason(null); setRecentEvents([]);
    }
  }, []);

  useEffect(() => {
    const refreshTimer = window.setTimeout(() => { void refresh(); }, 0);
    return () => window.clearTimeout(refreshTimer);
  }, [refresh]);

  const activeRun = useMemo(() => runs.find(run => run.id === activeRunId && activeStates.includes(run.state)) ?? null, [runs, activeRunId]);
  const activeModel = useMemo(() => models.find(model => model.id === activeRun?.model_id) ?? null, [models, activeRun]);

  useEffect(() => {
    socketRef.current?.close(); socketRef.current = null;
    if (!activeRunId) return;
    const socket = connectRun(activeRunId, event => {
      setRecentEvents(items => [event, ...items].slice(0, 8));
      if (event.type === "state") {
        const state = event.payload.state as RunState;
        setRuns(items => items.map(item => item.id === event.run_id ? { ...item, state } : item));
        if (!activeStates.includes(state)) window.setTimeout(() => void refresh(), 0);
      }
      if (event.type === "device_health") setDeviceHealth(event.payload);
      if (event.type === "latency") setLatency(event.payload as unknown as LatencyPayload);
      if (event.type === "runtime_health") {
        const health = event.payload as unknown as RuntimeHealthPayload;
        setRuntimeHealth(health);
        setRuns(items => items.map(item => item.id === event.run_id ? { ...item, successful_windows: health.successful_windows, failed_windows: health.failed_windows, expected_windows: health.expected_windows } : item));
      }
      if (event.type === "input_contract") setInputContract(event.payload as unknown as InputContractPayload);
      if (event.type === "error" && event.payload.fatal === true) {
        setFatalReason(String(event.payload.message ?? "运行已阻断"));
        setInputContract(current => current ? { ...current, safe: false } : { safe: false, source_channels: 0, target_channels: 0, valid_channels: 0, window_sec: 0, target_sample_rate: 0 });
      }
    });
    socketRef.current = socket;
    return () => socket.close();
  }, [activeRunId, refresh]);

  const registerActiveRun = useCallback((runId: string) => {
    setActiveRunId(runId); setFatalReason(null); setDeviceHealth({}); setLatency(null); setRuntimeHealth(null); setInputContract(null); setRecentEvents([]);
    void consoleApi.run(runId).then(run => setRuns(items => [run, ...items.filter(item => item.id !== run.id)])).catch(() => void refresh());
  }, [refresh]);

  const value = useMemo(() => ({ runs, models, system, activeRun, activeModel, deviceHealth, latency, runtimeHealth, inputContract, recentEvents, fatalReason, refresh, registerActiveRun }), [runs, models, system, activeRun, activeModel, deviceHealth, latency, runtimeHealth, inputContract, recentEvents, fatalReason, refresh, registerActiveRun]);
  return <RuntimeStatusContext.Provider value={value}>{children}</RuntimeStatusContext.Provider>;
}

export function useRuntimeStatus(): RuntimeStatus {
  const context = useContext(RuntimeStatusContext);
  if (!context) throw new Error("useRuntimeStatus must be used within RuntimeStatusProvider");
  return context;
}
