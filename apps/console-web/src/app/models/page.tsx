"use client";

import { useEffect, useMemo, useState } from "react";

import { ErrorState, KeyValue, PageHeader, SectionCard, StatusBadge } from "@/components/ui/design-system";
import { consoleApi } from "@/lib/api/client";
import { percent } from "@/lib/format/value";
import type { ModelSummary } from "@/types/api";

export default function ModelsPage() {
  const [models, setModels] = useState<ModelSummary[]>([]);
  const [selectedId, setSelectedId] = useState<string>("");
  const [query, setQuery] = useState("");
  const [type, setType] = useState("all");
  const [error, setError] = useState("");
  useEffect(() => { consoleApi.models().then(items => { setModels(items); setSelectedId(items[0]?.id ?? ""); }).catch(errorValue => setError(errorValue instanceof Error ? errorValue.message : "未知错误")); }, []);
  const filtered = useMemo(() => models.filter(model => (type === "all" || model.head_type === type) && `${model.model_name} ${model.subject_id ?? ""}`.toLowerCase().includes(query.toLowerCase())), [models, query, type]);
  const selected = models.find(model => model.id === selectedId) ?? filtered[0];
  return <>
    <PageHeader title="模型管理" description="只读浏览 Runtime Package schema v2；模型参数与 checkpoint 不在控制台中修改。" />
    {error && <ErrorState message={error} />}
    <div className="filter-row">
      <div className="field"><label>Backbone</label><select><option>全部</option><option>50M</option><option>LaBraM</option></select></div>
      <div className="field"><label>被试</label><select><option>全部</option><option>S01</option></select></div>
      <div className="field"><label>类型</label><select value={type} onChange={event => setType(event.target.value)}><option value="all">全部</option><option value="population">Population</option><option value="personal">Personal</option></select></div>
      <div className="field"><label>窗口</label><select><option>全部</option><option>4.0 s</option></select></div>
      <div className="field"><label>搜索</label><input value={query} onChange={event => setQuery(event.target.value)} placeholder="模型名称或被试 ID" /></div>
    </div>
    <div className="split-layout">
      <SectionCard title="模型列表" eyebrow={`${filtered.length} RUNTIME PACKAGES`}>
        <div className="model-list">{filtered.map(model => <button className={`model-row ${selected?.id === model.id ? "selected" : ""}`} key={model.id} onClick={() => setSelectedId(model.id)}><strong>{model.model_name}</strong><span>分类头<b>{model.head_type === "personal" ? `Personal · ${model.subject_id}` : "Population"}</b></span><span>窗口<b>{model.window_sec.toFixed(1)} s</b></span><span>Balanced Accuracy<b>{percent(model.balanced_accuracy)}</b></span><StatusBadge tone={model.runtime_verified ? "success" : "warning"}>{model.runtime_verified ? "已验证" : "未支持"}</StatusBadge></button>)}</div>
      </SectionCard>
      <SectionCard title="模型详情" eyebrow="INSPECTOR">
        {selected ? <><div className="key-value-list"><KeyValue label="Backbone" value={selected.model_name} /><KeyValue label="分类头" value={selected.head_type} /><KeyValue label="被试" value={selected.subject_id ?? "Population"} /><KeyValue label="任务" value="四分类运动想象" /><KeyValue label="窗口" value={`${selected.window_sec.toFixed(1)} s`} /><KeyValue label="步长" value={`${selected.step_sec.toFixed(1)} s`} /><KeyValue label="采样率" value={`${selected.sample_rate} Hz`} /><KeyValue label="目标通道" value={selected.target_channels} /><KeyValue label="Schema" value={`v${selected.schema_version}`} /><KeyValue label="Runtime" value={selected.runtime_verified ? "✓ 已验证" : "暂不支持"} /></div><div className="inspector-actions"><button className="button button-primary">用于离线回放</button><button className="button button-secondary">运行评估</button><button className="button button-secondary">查看 Package 信息</button></div></> : <p className="skeleton-label">没有发现可用的 Runtime Package。</p>}
      </SectionCard>
    </div>
  </>;
}
