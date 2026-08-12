"use client";
import { useEffect, useState } from "react";
import { PageHeader, SectionCard, StatusBadge } from "@/components/ui/design-system";
import { consoleApi } from "@/lib/api/client";
import type { RunSummary } from "@/types/api";
const statusTone = (state: string) => state === "running" ? "running" : state === "completed" ? "success" : state === "failed" ? "danger" : "idle" as const;
export default function RunsPage() { const [runs, setRuns] = useState<RunSummary[]>([]); useEffect(() => { consoleApi.runs().then(setRuns).catch(() => setRuns([])); }, []); return <><PageHeader title="运行记录" description="读取 Console Run Registry 中的真实运行。" /><SectionCard title="最近运行" eyebrow="RUN REGISTRY"><div className="table-wrap"><table className="data-table"><thead><tr><th>Run ID</th><th>类型</th><th>模型 ID</th><th>成功窗口</th><th>失败窗口</th><th>状态</th></tr></thead><tbody>{runs.length ? runs.map(run => <tr key={run.id}><td><strong>{run.id}</strong></td><td>{run.run_type === "live" ? "实时运行" : "离线回放"}</td><td>{run.model_id}</td><td>{run.successful_windows}</td><td>{run.failed_windows}</td><td><StatusBadge tone={statusTone(run.state)}>{run.state}</StatusBadge></td></tr>) : <tr><td colSpan={6}>暂无运行</td></tr>}</tbody></table></div></SectionCard></>; }
