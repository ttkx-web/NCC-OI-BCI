"use client";

import Link from "next/link";
import { useEffect, useState } from "react";
import { KeyValue, MetricCard, PageHeader, SectionCard, StatusBadge } from "@/components/ui/design-system";
import { consoleApi } from "@/lib/api/client";
import { connectRun } from "@/lib/websocket/run-stream";
import type { RunEvent, RunSummary, SystemStatus } from "@/types/api";

export default function OverviewPage() {
  const [runs, setRuns] = useState<RunSummary[]>([]); const [system, setSystem] = useState<SystemStatus | null>(null); const [events, setEvents] = useState<RunEvent[]>([]);
  useEffect(() => { Promise.all([consoleApi.runs(), consoleApi.systemStatus()]).then(([items, value]) => { setRuns(items); setSystem(value); const active = items.find(item => item.state === "running"); if (active) { const socket = connectRun(active.id, event => setEvents(current => [event, ...current].slice(0, 8))); return () => socket.close(); } }).catch(() => undefined); }, []);
  const active = runs.find(run => run.state === "running") ?? null; const health = system?.device.health;
  return <><PageHeader title="运行总览" description="当前 Console API 的真实 Live / Replay 运行状态。" action={<Link className="button button-primary" href="/live">开始实时运行</Link>} /><div className="dashboard-grid"><MetricCard label="设备状态" value={<StatusBadge tone={system?.device.status === "connected" ? "success" : "idle"}>{system?.device.status ?? "—"}</StatusBadge>} detail={system?.device.source ?? "暂无运行"} /><MetricCard label="运行状态" value={active?.state ?? "—"} detail={active ? active.id : "暂无运行"} /><MetricCard label="当前模型" value={active?.model_id ?? "—"} /><MetricCard label="模型输入" value={<StatusBadge tone="idle">{events.find(event => event.type === "input_contract")?.payload.safe === true ? "SAFE" : "—"}</StatusBadge>} /><SectionCard title="当前运行" className="span-5"><div className="key-value-list"><KeyValue label="Run ID" value={active?.id ?? "—"} /><KeyValue label="成功窗口" value={active?.successful_windows ?? "—"} /><KeyValue label="收到 Packet" value={String(health?.received_packets ?? "—")} /></div></SectionCard><SectionCard title="最近事件" className="span-7"><div className="timeline-list">{events.length ? events.map((event, index) => <div className="timeline-item" key={`${event.timestamp}-${index}`}><span className="event-chip">{event.type}</span><span>{event.run_id}</span></div>) : <span>暂无运行</span>}</div></SectionCard><SectionCard title="最近运行" className="span-12"><div className="table-wrap"><table className="data-table"><thead><tr><th>Run ID</th><th>类型</th><th>状态</th><th>成功窗口</th><th>失败窗口</th></tr></thead><tbody>{runs.length ? runs.map(run => <tr key={run.id}><td>{run.id}</td><td>{run.run_type}</td><td>{run.state}</td><td>{run.successful_windows}</td><td>{run.failed_windows}</td></tr>) : <tr><td colSpan={5}>暂无运行</td></tr>}</tbody></table></div></SectionCard></div></>;
}
