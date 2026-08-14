"use client";

import { useEffect, useRef, useState } from "react";

import { PredictionCard } from "@/components/status/prediction-card";
import { KeyValue, MetricCard, PageHeader, SectionCard, StatusBadge } from "@/components/ui/design-system";
import { consoleApi } from "@/lib/api/client";
import { connectRun } from "@/lib/websocket/run-stream";
import { useRuntimeStatus } from "@/components/runtime/run-status-provider";
import type { ModelSummary, RunEvent, RunState } from "@/types/api";

type LiveState = {
  state: RunState | "idle";
  runId: string | null;
  blocked: boolean;
  reason: string | null;
  prediction?: { name: string; confidence: number; probabilities: number[]; command: string };
  health: Record<string, unknown>;
  inputSafe: boolean;
  events: string[];
};

const initial: LiveState = { state: "idle", runId: null, blocked: false, reason: null, health: {}, inputSafe: false, events: [] };
const tone = (state: string) => state === "running" ? "running" : state === "failed" ? "danger" : state === "stopped" ? "idle" : "warning" as const;

export default function LivePage() {
  const { registerActiveRun } = useRuntimeStatus();
  const [models, setModels] = useState<ModelSummary[]>([]);
  const [modelId, setModelId] = useState("");
  const [device, setDevice] = useState<"cpu" | "cuda">("cuda");
  const [live, setLive] = useState<LiveState>(initial);
  const [error, setError] = useState("");
  const socket = useRef<WebSocket | null>(null);
  const running = live.state === "starting" || live.state === "running" || live.state === "stopping";

  useEffect(() => {
    consoleApi.models().then(items => {
      const verified = items.filter(item =>
        item.runtime_verified
        && item.live_verified
        && item.window_sec === 4
        && item.step_sec === 0.5
        && ["model_50m", "labram", "cbramod"].includes(item.model_type)
      );
      setModels(verified); setModelId(verified[0]?.id ?? "");
    }).catch(value => setError(value instanceof Error ? value.message : "无法读取模型"));
    return () => socket.current?.close();
  }, []);

  function subscribe(runId: string) {
    socket.current?.close();
    socket.current = connectRun(runId, (event: RunEvent) => setLive(current => {
      const next = { ...current, events: [`${event.type}`, ...current.events].slice(0, 8) };
      const payload = event.payload;
      if (event.type === "state") return { ...next, state: String(payload.state) as RunState };
      if (event.type === "device_health") return { ...next, health: payload };
      if (event.type === "input_contract") return { ...next, inputSafe: payload.safe === true, blocked: payload.safe === false && current.blocked, reason: typeof payload.reason === "string" ? payload.reason : current.reason };
      if (event.type === "prediction" && next.inputSafe && !next.blocked) return { ...next, prediction: { name: String(payload.predicted_name), confidence: Number(payload.confidence), probabilities: Array.isArray(payload.probabilities) ? payload.probabilities.map(Number) : [], command: String(payload.command) } };
      if (event.type === "error" && payload.fatal === true) return { ...next, blocked: true, inputSafe: false, prediction: undefined, reason: String(payload.message ?? "实时运行已阻断") };
      return next;
    }));
  }

  async function start() {
    setError("");
    try {
      const created = await consoleApi.startLive({ model_id: modelId, source: "neuracle_jellyfish", compute_device: device, confidence_threshold: 0.55 });
      setLive({ ...initial, state: "starting", runId: created.run_id }); subscribe(created.run_id);
      registerActiveRun(created.run_id);
    } catch (value) { setError(value instanceof Error ? value.message : "无法启动 Live 运行"); }
  }
  async function stop() { if (live.runId) { await consoleApi.stopRun(live.runId); } }
  async function restart() { if (live.runId) { await consoleApi.restartRun(live.runId); registerActiveRun(live.runId); } else { await start(); } }
  const h = live.health;

  return <>
    <PageHeader title="实时运行" description="Neuracle JellyFish 的受控实时推理；控制台不接收 EEG 波形或原始样本。" action={<StatusBadge tone={tone(live.state)}>{live.blocked ? "已阻断" : live.state}</StatusBadge>} />
    {error && <div className="error-state"><strong>操作失败</strong><span>{error}</span></div>}
    <SectionCard title="运行控制" eyebrow="LIVE DEVICE">
      <div className="filter-row"><div className="field"><label>模型</label><select value={modelId} disabled={running} onChange={event => setModelId(event.target.value)}>{models.map(model => <option value={model.id} key={model.id}>{model.model_name} · {model.window_sec}s</option>)}</select></div><div className="field"><label>计算设备</label><select value={device} disabled={running} onChange={event => setDevice(event.target.value as "cpu" | "cuda")}><option value="cuda">CUDA</option><option value="cpu">CPU</option></select></div><div className="field"><label>置信度阈值</label><input value="0.55" disabled /></div><div className="inspector-actions"><button className="button button-primary" disabled={running || !modelId} onClick={start}>开始</button><button className="button button-secondary" disabled={!running} onClick={stop}>停止</button><button className="button button-secondary" disabled={running} onClick={restart}>重新开始</button></div></div>
    </SectionCard>
    {live.blocked && <div className="notice warning">已阻断：{live.reason ?? "实时安全 Gate 失败"}。旧 prediction 已失效，命令为 STOP。</div>}
    <div className="dashboard-grid">
      <MetricCard label="1. 设备连接" value={<StatusBadge tone={h.connected === true ? "success" : "danger"}>{h.connected === true ? "已连接" : "未连接"}</StatusBadge>} />
      <MetricCard label="2. EEG 通道" value={h.channel_count == null ? "—" : String(h.channel_count)} detail="仅验证 EEG 元数据" />
      <MetricCard label="3. 采样率 / 单位" value={h.sampling_rate == null ? "—" : `${String(h.sampling_rate)} Hz`} detail={String(h.eeg_unit ?? "—")} />
      <MetricCard label="4. Trigger" value={live.events.includes("trigger") ? "已接收" : "—"} detail="仅事件元数据" />
      <MetricCard label="5. Packet continuity" value={<StatusBadge tone={Number(h.missing_packets ?? 0) + Number(h.duplicate_packets ?? 0) + Number(h.out_of_order_packets ?? 0) === 0 ? "success" : "danger"}>{live.blocked ? "已阻断" : "正常"}</StatusBadge>} />
      <MetricCard label="6. Model Input Contract" value={<StatusBadge tone={live.inputSafe ? "success" : live.blocked ? "danger" : "idle"}>{live.inputSafe ? "SAFE" : live.blocked ? "BLOCKED" : "等待验证"}</StatusBadge>} />
      <div className="span-7"><PredictionCard invalid={live.blocked || !live.inputSafe} prediction={live.prediction?.name} confidence={live.prediction?.confidence} probabilities={live.prediction?.probabilities} command={live.inputSafe && !live.blocked ? live.prediction?.command : "STOP"} /></div>
      <SectionCard title="输出命令" eyebrow="FAIL-CLOSED" className="span-5"><div className="prediction-hero"><span>{live.inputSafe && !live.blocked ? live.prediction?.command ?? "STOP" : "STOP"}</span><strong>{live.inputSafe && !live.blocked ? "有效" : "已阻断"}</strong><small>{live.reason ?? "仅在 Model Input SAFE 后生效"}</small></div></SectionCard>
      <SectionCard title="设备与运行健康" className="span-6"><div className="key-value-list"><KeyValue label="收到 Packet" value={String(h.received_packets ?? "—")} /><KeyValue label="丢失 / 重复 / 乱序" value={`${String(h.missing_packets ?? "—")} / ${String(h.duplicate_packets ?? "—")} / ${String(h.out_of_order_packets ?? "—")}`} /><KeyValue label="运行 ID" value={live.runId ?? "—"} /></div></SectionCard>
      <SectionCard title="事件时间线" className="span-6"><div className="timeline-list">{live.events.length ? live.events.map((item, index) => <div className="timeline-item" key={`${item}-${index}`}><span className="event-chip">{item}</span><span>实时事件</span></div>) : <span>暂无运行</span>}</div></SectionCard>
    </div>
  </>;
}
