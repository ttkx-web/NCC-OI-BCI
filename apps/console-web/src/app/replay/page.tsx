"use client";

import { useEffect, useMemo, useRef, useState } from "react";

import { LatencyCard } from "@/components/status/latency-card";
import { PredictionCard } from "@/components/status/prediction-card";
import { RuntimeHealthCard } from "@/components/status/runtime-health-card";
import { ErrorState, KeyValue, PageHeader, ProgressBar, SectionCard, StatusBadge } from "@/components/ui/design-system";
import { consoleApi } from "@/lib/api/client";
import { milliseconds, percent } from "@/lib/format/value";
import { connectRun } from "@/lib/websocket/run-stream";
import type { DatasetSummary, InputContractPayload, LatencyPayload, ModelSummary, PredictionPayload, RunEvent, RunState, RuntimeHealthPayload } from "@/types/api";

const stateNames: Record<RunState, string> = { idle: "空闲", starting: "启动中", running: "运行中", stopping: "停止中", stopped: "已停止", completed: "已完成", failed: "失败" };
const activeStates: RunState[] = ["starting", "running", "stopping"];

export default function ReplayPage() {
  const [datasets, setDatasets] = useState<DatasetSummary[]>([]);
  const [models, setModels] = useState<ModelSummary[]>([]);
  const [datasetKey, setDatasetKey] = useState("");
  const [session, setSession] = useState("");
  const [modelId, setModelId] = useState("");
  const [device, setDevice] = useState("cpu");
  const [speed, setSpeed] = useState(1);
  const [maximum, setMaximum] = useState(100);
  const [threshold, setThreshold] = useState(0.55);
  const [runId, setRunId] = useState("");
  const [state, setState] = useState<RunState>("idle");
  const [prediction, setPrediction] = useState<PredictionPayload | null>(null);
  const [latency, setLatency] = useState<LatencyPayload | null>(null);
  const [health, setHealth] = useState<RuntimeHealthPayload | null>(null);
  const [contract, setContract] = useState<InputContractPayload | null>(null);
  const [history, setHistory] = useState<PredictionPayload[]>([]);
  const [blockedReason, setBlockedReason] = useState("");
  const [error, setError] = useState("");
  const socketRef = useRef<WebSocket | null>(null);
  const blockedRef = useRef(false);

  useEffect(() => {
    Promise.all([consoleApi.datasets(), consoleApi.models()]).then(([datasetItems, modelItems]) => {
      setDatasets(datasetItems);
      setModels(modelItems.filter(item => item.runtime_verified));
      if (datasetItems[0]) { setDatasetKey(`${datasetItems[0].id}|${datasetItems[0].subject_id}`); setSession(datasetItems[0].sessions[0] ?? ""); }
      if (modelItems[0]) { setModelId(modelItems.find(item => item.runtime_verified)?.id ?? ""); setThreshold(modelItems[0].warning_message ? 0.55 : 0.55); }
    }).catch(errorValue => setError(errorValue instanceof Error ? errorValue.message : "初始化失败"));
    return () => socketRef.current?.close();
  }, []);

  const selectedDataset = useMemo(() => datasets.find(item => `${item.id}|${item.subject_id}` === datasetKey), [datasets, datasetKey]);
  const selectedModel = models.find(item => item.id === modelId);
  const configurationDisabled = activeStates.includes(state);

  function handleEvent(event: RunEvent) {
    if (event.type === "state") setState(event.payload.state as RunState);
    if (event.type === "input_contract") {
      const value = event.payload as unknown as InputContractPayload;
      setContract(value);
      if (!value.safe) { blockedRef.current = true; setPrediction(null); setBlockedReason("模型输入合同不安全，推理已禁用。"); }
    }
    if (event.type === "prediction" && !blockedRef.current) {
      const value = event.payload as unknown as PredictionPayload;
      setPrediction(value);
      setHistory(items => [...items, value].slice(-200));
    }
    if (event.type === "latency") setLatency(event.payload as unknown as LatencyPayload);
    if (event.type === "runtime_health") setHealth(event.payload as unknown as RuntimeHealthPayload);
    if (event.type === "error") {
      const fatal = event.payload.fatal === true;
      const message = String(event.payload.message ?? "运行发生错误");
      setError(message);
      if (fatal) { blockedRef.current = true; setPrediction(null); setBlockedReason(message); }
    }
  }

  async function start() {
    if (!selectedDataset || !selectedModel || !session) { setError("请选择完整的数据集、Session 和模型。"); return; }
    setError(""); setBlockedReason(""); blockedRef.current = false; setPrediction(null); setHistory([]); setLatency(null); setHealth(null); setContract(null); setState("starting");
    try {
      const result = await consoleApi.startReplay({ dataset_id: selectedDataset.id, subject_id: selectedDataset.subject_id, session, model_id: selectedModel.id, compute_device: device, replay_speed: speed, max_windows: maximum, confidence_threshold: threshold });
      setRunId(result.run_id);
      socketRef.current?.close();
      socketRef.current = connectRun(result.run_id, handleEvent);
      socketRef.current.onerror = () => setError("实时事件连接中断，请检查 Console API。");
    } catch (errorValue) { setState("failed"); setError(errorValue instanceof Error ? errorValue.message : "启动失败"); }
  }

  async function stop() { if (!runId) return; try { await consoleApi.stopRun(runId); } catch (errorValue) { setError(errorValue instanceof Error ? errorValue.message : "停止失败"); } }
  async function restart() { if (!runId) return; setError(""); setBlockedReason(""); blockedRef.current = false; setPrediction(null); setHistory([]); try { await consoleApi.restartRun(runId); } catch (errorValue) { setError(errorValue instanceof Error ? errorValue.message : "重新开始失败"); } }

  const expectedName = prediction?.expected_class_name?.toUpperCase() ?? "—";
  const correct = prediction?.expected_class_id == null ? null : prediction.expected_class_id === prediction.predicted_class;
  const progress = health?.expected_windows ? (health.successful_windows / health.expected_windows) * 100 : 0;
  return <>
    <PageHeader title="离线回放" description="使用现有 Runtime Package 和 PipelineController 进行可控的伪实时推理。" action={runId && <span className="compute-chip">{runId}</span>} />
    {error && <div style={{marginBottom: 16}}><ErrorState message={error} /></div>}
    {blockedReason && <div className="notice danger" style={{marginBottom: 16}}><strong>已阻断：</strong>{blockedReason} 当前预测与输出命令均已失效。</div>}
    <div className="dashboard-grid">
      <SectionCard title="离线回放配置" eyebrow="REPLAY CONFIGURATION" className="span-9">
        <div className="field-grid">
          <div className="field"><label>数据集</label><select value={datasetKey} disabled={configurationDisabled} onChange={event => { const value = event.target.value; setDatasetKey(value); const item = datasets.find(dataset => `${dataset.id}|${dataset.subject_id}` === value); setSession(item?.sessions[0] ?? ""); }}><option value="">请选择</option>{datasets.map(item => <option key={`${item.id}-${item.subject_id}`} value={`${item.id}|${item.subject_id}`}>{item.name}</option>)}</select></div>
          <div className="field"><label>被试</label><select value={datasetKey} disabled={configurationDisabled} onChange={event => { const value = event.target.value; setDatasetKey(value); const item = datasets.find(dataset => `${dataset.id}|${dataset.subject_id}` === value); setSession(item?.sessions[0] ?? ""); }}>{datasets.map(item => <option key={`${item.id}-${item.subject_id}`} value={`${item.id}|${item.subject_id}`}>被试 {item.subject_id}</option>)}</select></div>
          <div className="field"><label>Session</label><select value={session} disabled={configurationDisabled} onChange={event => setSession(event.target.value)}>{selectedDataset?.sessions.map(item => <option key={item}>{item}</option>)}</select></div>
          <div className="field"><label>模型</label><select value={modelId} disabled={configurationDisabled} onChange={event => setModelId(event.target.value)}>{models.map(item => <option key={item.id} value={item.id}>{item.model_name} · {item.head_type} · {item.window_sec}s</option>)}</select></div>
          <div className="field"><label>计算设备</label><select value={device} disabled={configurationDisabled} onChange={event => setDevice(event.target.value)}><option value="cpu">CPU</option><option value="cuda">CUDA</option></select></div>
          <div className="field"><label>回放速度</label><input type="number" min="0.1" max="100" step="0.1" value={speed} disabled={configurationDisabled} onChange={event => setSpeed(Number(event.target.value))} /></div>
          <div className="field"><label>最大窗口数</label><input type="number" min="1" max="10000" value={maximum} disabled={configurationDisabled} onChange={event => setMaximum(Number(event.target.value))} /></div>
          <div className="field"><label>置信度阈值 · {threshold.toFixed(2)}</label><input type="range" min="0" max="1" step="0.01" value={threshold} disabled={configurationDisabled} onChange={event => setThreshold(Number(event.target.value))} /></div>
        </div>
        <div className="readonly-strip"><span>窗口长度<strong>{selectedModel?.window_sec.toFixed(1) ?? "—"} s</strong></span><span>滑窗步长<strong>{selectedModel?.step_sec.toFixed(1) ?? "—"} s</strong></span><span>目标采样率<strong>{selectedModel?.sample_rate ?? "—"} Hz</strong></span></div>
      </SectionCard>
      <SectionCard title="运行控制" eyebrow="RUN CONTROL" className="span-3">
        <div className="control-state"><span>状态</span><StatusBadge tone={state === "running" ? "running" : state === "failed" ? "danger" : "idle"}>{stateNames[state]}</StatusBadge></div>
        <div className="control-stack"><button className="button button-primary control-button" disabled={configurationDisabled} onClick={start}>▶ 开始</button><button className="button button-danger" disabled={!activeStates.includes(state)} onClick={stop}>■ 停止</button><button className="button button-secondary" disabled={!runId || activeStates.includes(state)} onClick={restart}>↻ 重新开始</button></div>
      </SectionCard>

      <SectionCard title="当前窗口" eyebrow="CURRENT WINDOW" className="span-7" action={<StatusBadge tone={contract?.safe === false ? "danger" : "success"}>{contract?.safe === false ? "已阻断" : "数据正常"}</StatusBadge>}>
        <div className="key-value-list"><KeyValue label="Window" value={prediction ? `#${prediction.window_id}` : "—"} /><KeyValue label="Trial" value={prediction?.trial_id ? `#${prediction.trial_id}` : "—"} /><KeyValue label="Session" value={session || "—"} /><KeyValue label="时间位置" value={prediction ? `${(prediction.window_id * (selectedModel?.step_sec ?? 0.5)).toFixed(1)} s` : "—"} /><KeyValue label="数据状态" value="✓ 正常" /><KeyValue label="模型输入" value={contract?.safe === false ? "已阻断" : "✓ SAFE"} /><KeyValue label="真实类别" value={expectedName} /></div>
      </SectionCard>
      <div className="span-5"><PredictionCard prediction={prediction?.predicted_name} confidence={prediction?.confidence} probabilities={prediction?.probabilities} invalid={Boolean(blockedReason)} />{prediction && <div className={`notice ${correct === false ? "danger" : ""}`} style={{marginTop: 10}}>真实类别：<strong>{expectedName}</strong>　结果：<strong>{correct == null ? "无标签" : correct ? "✓ 正确" : "✕ 错误"}</strong></div>}</div>

      <SectionCard title="窗口进度" className="span-4"><div style={{display:"flex",justifyContent:"space-between",marginBottom:14,fontSize:12}}><strong>{health?.successful_windows ?? 0} / {health?.expected_windows ?? maximum}</strong><span className="skeleton-label">{progress.toFixed(0)}%</span></div><ProgressBar value={progress} /></SectionCard>
      <div className="span-4"><LatencyCard value={latency} /></div>
      <div className="span-4"><RuntimeHealthCard value={health} /></div>

      <SectionCard title="预测历史" eyebrow="PREDICTION HISTORY" className="span-12">
        <div className="table-wrap"><table className="data-table"><thead><tr><th>窗口</th><th>Trial</th><th>真实类别</th><th>预测类别</th><th>置信度</th><th>预处理延迟</th><th>模型延迟</th><th>总延迟</th></tr></thead><tbody>{history.slice().reverse().map((item, index) => <tr key={`${item.window_id}-${index}`}><td>#{item.window_id}</td><td>{item.trial_id ?? "—"}</td><td>{item.expected_class_name?.toUpperCase() ?? "—"}</td><td><strong>{item.predicted_name.toUpperCase()}</strong></td><td>{percent(item.confidence)}</td><td>{index === 0 ? milliseconds(latency?.prepare_ms) : "—"}</td><td>{index === 0 ? milliseconds(latency?.inference_ms) : "—"}</td><td>{index === 0 ? milliseconds(latency?.total_ms) : "—"}</td></tr>)}</tbody></table>{history.length === 0 && <p className="skeleton-label" style={{textAlign:"center",padding:30}}>开始回放后，预测事件会在这里实时追加。</p>}</div>
      </SectionCard>
    </div>
  </>;
}
