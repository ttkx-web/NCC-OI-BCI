"use client";

import Link from "next/link";
import { KeyValue, MetricCard, PageHeader, SectionCard, StatusBadge } from "@/components/ui/design-system";
import { runStateLabel, useRuntimeStatus } from "@/components/runtime/run-status-provider";

export default function OverviewPage() {
  const { activeRun, activeModel, deviceHealth, inputContract, recentEvents, runs } = useRuntimeStatus();
  const connected = deviceHealth.connected === true;
  return <>
    <PageHeader title="运行总览" description="当前 Console API 的真实 Live / Replay 运行状态。" action={<Link className="button button-primary" href="/live">开始实时运行</Link>} />
    <div className="dashboard-grid">
      <MetricCard label="设备状态" value={<StatusBadge tone={connected ? "success" : "idle"}>{connected ? "已连接" : "未连接"}</StatusBadge>} detail={connected ? String(deviceHealth.state ?? "已连接") : "暂无设备连接"} />
      <MetricCard label="运行状态" value={runStateLabel(activeRun?.state)} detail={activeRun?.id ?? "暂无运行"} />
      <MetricCard label="当前模型" value={activeModel?.model_name ?? "—"} detail={activeRun?.subject_id ?? "当前被试：—"} />
      <MetricCard label="模型输入" value={<StatusBadge tone={inputContract?.safe === true ? "success" : inputContract?.safe === false ? "danger" : "idle"}>{inputContract?.safe === true ? "SAFE" : inputContract?.safe === false ? "BLOCKED" : "—"}</StatusBadge>} />
      <SectionCard title="当前运行" className="span-5"><div className="key-value-list"><KeyValue label="Run ID" value={activeRun?.id ?? "—"} /><KeyValue label="当前被试" value={activeRun?.subject_id ?? "—"} /><KeyValue label="成功窗口" value={activeRun?.successful_windows ?? "—"} /><KeyValue label="收到 Packet" value={String(deviceHealth.received_packets ?? "—")} /></div></SectionCard>
      <SectionCard title="最近事件" className="span-7"><div className="timeline-list">{recentEvents.length ? recentEvents.map((event, index) => <div className="timeline-item" key={`${event.timestamp}-${index}`}><span className="event-chip">{event.type}</span><span>{event.run_id}</span></div>) : <span>暂无运行</span>}</div></SectionCard>
      <SectionCard title="最近运行" className="span-12"><div className="table-wrap"><table className="data-table"><thead><tr><th>Run ID</th><th>类型</th><th>状态</th><th>成功窗口</th><th>失败窗口</th></tr></thead><tbody>{runs.length ? runs.map(run => <tr key={run.id}><td>{run.id}</td><td>{run.run_type}</td><td>{runStateLabel(run.state)}</td><td>{run.successful_windows}</td><td>{run.failed_windows}</td></tr>) : <tr><td colSpan={5}>暂无运行</td></tr>}</tbody></table></div></SectionCard>
    </div>
  </>;
}
