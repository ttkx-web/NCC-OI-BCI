"use client";

import { useEffect, useState } from "react";

import { EmptyState, ErrorState, PageHeader, SectionCard, StatusBadge } from "@/components/ui/design-system";
import { consoleApi } from "@/lib/api/client";
import type { DatasetSummary } from "@/types/api";

export default function DataPage() {
  const [items, setItems] = useState<DatasetSummary[]>([]); const [error, setError] = useState("");
  useEffect(() => { consoleApi.datasets().then(setItems).catch(value => setError(value instanceof Error ? value.message : "读取失败")); }, []);
  return <><PageHeader title="数据管理" description="浏览已处理数据集的匿名元数据与质量状态。" />{error && <ErrorState message={error} />}<SectionCard title="已处理数据集" eyebrow={`${items.length} SUBJECT DATASETS`}><div className="table-wrap"><table className="data-table"><thead><tr><th>数据集</th><th>被试</th><th>Session</th><th>Trial</th><th>通道</th><th>采样率</th><th>单位</th><th>QC</th></tr></thead><tbody>{items.map(item => <tr key={`${item.id}-${item.subject_id}`}><td><strong>{item.name}</strong></td><td>{item.subject_id}</td><td>{item.sessions.join("、")}</td><td>{item.trial_count}</td><td>{item.channel_count}</td><td>{item.sample_rate} Hz</td><td>{item.unit}</td><td><StatusBadge tone="success">PASSED</StatusBadge></td></tr>)}</tbody></table></div>{!error && items.length === 0 && <EmptyState title="暂无可用数据" description="将经过现有数据流水线处理的 HDF5 数据放入 data/processed 后即可发现。" />}</SectionCard></>;
}

