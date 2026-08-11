import { EmptyState, PageHeader, SectionCard } from "@/components/ui/design-system";

export default function ExperimentsPage() {
  return <>
    <PageHeader title="实验评估" description="配置跨模型与窗口长度的标准化评估；执行能力将在后续版本开放。" />
    <div className="dashboard-grid">
      <SectionCard title="新建实验" eyebrow="EXPERIMENT SETUP" className="span-4">
        <div className="field-grid" style={{gridTemplateColumns:"1fr"}}><div className="field"><label>任务</label><select disabled><option>BNCI2014_001</option></select></div><div className="field"><label>模型</label><select disabled><option>50M · LaBraM · CBraMod</option></select></div><div className="field"><label>窗口</label><select disabled><option>4 s</option></select></div><div className="field"><label>被试</label><select disabled><option>01 — 09</option></select></div><div className="field"><label>指标</label><input disabled value="Accuracy · BAcc · Macro-F1 · P50 · P95" readOnly /></div></div>
        <button className="button button-primary" disabled style={{width:"100%",marginTop:16}}>开始实验</button><p className="skeleton-label">实验执行将在后续版本开放。</p>
      </SectionCard>
      <SectionCard title="实验结果" eyebrow="RESULTS" className="span-8"><div className="table-wrap"><table className="data-table"><thead><tr><th>模型</th><th>窗口</th><th>Accuracy</th><th>BAcc</th><th>Macro-F1</th><th>P50</th><th>P95</th></tr></thead></table></div><EmptyState title="暂无实验结果" description="P0 只建立页面结构和结果类型，不触发评估任务。" /></SectionCard>
    </div>
  </>;
}
