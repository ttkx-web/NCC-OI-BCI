import { PageHeader, SectionCard, StatusBadge } from "@/components/ui/design-system";

export default function RunsPage() {
  return <><PageHeader title="运行记录" description="统一查看 Replay、Live 与 Evaluation 运行。" /><SectionCard title="最近运行" eyebrow="RUN REGISTRY"><div className="table-wrap"><table className="data-table"><thead><tr><th>Run ID</th><th>类型</th><th>被试</th><th>模型</th><th>成功窗口</th><th>P50</th><th>状态</th></tr></thead><tbody><tr><td><strong>#842</strong></td><td>实时运行</td><td>S01</td><td>50M Personal</td><td>105</td><td>52 ms</td><td><StatusBadge tone="running">运行中</StatusBadge></td></tr><tr><td><strong>#841</strong></td><td>离线回放</td><td>S01</td><td>50M Population</td><td>100</td><td>55 ms</td><td><StatusBadge tone="success">已完成</StatusBadge></td></tr></tbody></table></div></SectionCard></>;
}
