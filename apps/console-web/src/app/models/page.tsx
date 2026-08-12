"use client";

import { useEffect, useMemo, useState } from "react";
import { ErrorState, KeyValue, PageHeader, SectionCard, StatusBadge } from "@/components/ui/design-system";
import { consoleApi } from "@/lib/api/client";
import { percent } from "@/lib/format/value";
import type { ModelSummary } from "@/types/api";

export default function ModelsPage() {
  const [models, setModels] = useState<ModelSummary[]>([]);
  const [selectedId, setSelectedId] = useState("");
  const [backbone, setBackbone] = useState("all");
  const [query, setQuery] = useState("");
  const [error, setError] = useState("");
  useEffect(() => { consoleApi.models().then(items => { setModels(items); setSelectedId(items[0]?.id ?? ""); }).catch(value => setError(value instanceof Error ? value.message : "无法读取模型")); }, []);
  const filtered = useMemo(() => models.filter(model => (backbone === "all" || model.model_type === backbone) && `${model.model_name} ${model.subject_id ?? ""}`.toLowerCase().includes(query.toLowerCase())), [models, backbone, query]);
  const selected = models.find(model => model.id === selectedId) ?? filtered[0];
  return <>
    <PageHeader title="模型管理" description="只读展示真实 Runtime Package 元数据；验证状态来自统一 Runtime loader。" />
    {error && <ErrorState message={error} />}
    <div className="filter-row"><div className="field"><label>模型</label><select value={backbone} onChange={event => setBackbone(event.target.value)}><option value="all">全部</option><option value="model_50m">50M</option><option value="labram">LaBraM</option><option value="cbramod">CBraMod</option></select></div><div className="field"><label>搜索</label><input value={query} onChange={event => setQuery(event.target.value)} placeholder="模型名或被试 ID" /></div></div>
    <div className="split-layout"><SectionCard title="模型列表" eyebrow={`${filtered.length} RUNTIME PACKAGES`}><div className="model-list">{filtered.length ? filtered.map(model => <button className={`model-row ${selected?.id === model.id ? "selected" : ""}`} key={model.id} onClick={() => setSelectedId(model.id)}><strong>{model.model_name}</strong><span>Window <b>{model.window_sec.toFixed(1)} s</b></span><span>BAcc <b>{percent(model.balanced_accuracy)}</b></span><StatusBadge tone={model.runtime_verified ? "success" : "danger"}>{model.runtime_verified ? "已验证" : "未验证"}</StatusBadge></button>) : <span>暂无模型</span>}</div></SectionCard>
      <SectionCard title="模型详情" eyebrow="PACKAGE METADATA">{selected ? <div className="key-value-list"><KeyValue label="模型" value={selected.model_name} /><KeyValue label="Window" value={`${selected.window_sec.toFixed(1)} s`} /><KeyValue label="Step" value={`${selected.step_sec.toFixed(1)} s`} /><KeyValue label="Sample Rate" value={`${selected.sample_rate} Hz`} /><KeyValue label="Channels" value={selected.target_channels} /><KeyValue label="Schema" value={`v${selected.schema_version}`} /><KeyValue label="BAcc" value={percent(selected.balanced_accuracy)} /><KeyValue label="Macro-F1" value={percent(selected.macro_f1)} /><KeyValue label="Runtime 状态" value={selected.runtime_verified ? "runtime_verified=true" : "runtime_verified=false"} /></div> : <span>暂无模型</span>}</SectionCard></div>
  </>;
}
