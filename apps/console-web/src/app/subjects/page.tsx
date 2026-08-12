"use client";

import { useRuntimeStatus } from "@/components/runtime/run-status-provider";
import { PageHeader, SectionCard } from "@/components/ui/design-system";

export default function SubjectsPage() {
  const { runs } = useRuntimeStatus();
  const subjects = [...new Set(runs.map(run => run.subject_id).filter((value): value is string => Boolean(value)))];
  return <><PageHeader title="被试管理" description="使用匿名被试 ID 关联真实数据与运行记录。" /><SectionCard title="被试列表" eyebrow="PRIVACY SAFE"><div className="table-wrap"><table className="data-table"><thead><tr><th>被试 ID</th><th>最近运行</th><th>状态</th></tr></thead><tbody>{subjects.length ? subjects.map(subject => { const latest = runs.find(run => run.subject_id === subject); return <tr key={subject}><td><strong>{subject}</strong></td><td>{latest?.id ?? "—"}</td><td>{latest?.state ?? "—"}</td></tr>; }) : <tr><td colSpan={3}>暂无运行</td></tr>}</tbody></table></div></SectionCard></>;
}
