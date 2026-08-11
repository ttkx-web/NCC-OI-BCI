import { PageHeader, SectionCard, StatusBadge } from "@/components/ui/design-system";

export default function SubjectsPage() {
  return <><PageHeader title="被试管理" description="使用匿名被试 ID 关联数据、模型与运行记录。" /><SectionCard title="被试列表" eyebrow="PRIVACY SAFE"><div className="table-wrap"><table className="data-table"><thead><tr><th>被试 ID</th><th>数据集</th><th>模型</th><th>最近运行</th><th>状态</th></tr></thead><tbody>{["S01","S02","S03","S04"].map((id,index) => <tr key={id}><td><strong>{id}</strong></td><td>BNCI2014_001</td><td>{index === 0 ? "3" : "1"}</td><td>{index === 0 ? "Run #842" : "—"}</td><td><StatusBadge tone="success">可用</StatusBadge></td></tr>)}</tbody></table></div></SectionCard></>;
}

