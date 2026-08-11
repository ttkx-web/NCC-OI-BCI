import Link from "next/link";

import { LatencyCard } from "@/components/status/latency-card";
import { PredictionCard } from "@/components/status/prediction-card";
import { KeyValue, MetricCard, PageHeader, SectionCard, StatusBadge } from "@/components/ui/design-system";

const predictions = [
  ["#101", "LEFT", "81%"], ["#102", "LEFT", "79%"], ["#103", "RIGHT", "72%"], ["#104", "RIGHT", "76%"], ["#105", "LEFT", "82%"],
];

export default function OverviewPage() {
  return <>
    <PageHeader title="运行总览" description="集中查看设备、Runtime、模型输入合同与最近预测状态。" action={<Link className="button button-primary" href="/replay">开始离线回放 <span>→</span></Link>} />
    <div className="dashboard-grid">
      <MetricCard label="设备状态" value={<StatusBadge tone="success">已连接</StatusBadge>} detail={<>Neuracle JellyFish<br />59 通道 · 1000 Hz · Trigger 正常</>} tone="success" />
      <MetricCard label="运行状态" value={<StatusBadge tone="running">运行中</StatusBadge>} detail={<>窗口 4.0 s · 步长 0.5 s<br />计算设备 CUDA</>} tone="running" />
      <MetricCard label="当前模型" value="50M" detail={<>Personal · S01<br />Runtime Package · Schema v2</>} tone="brand" />
      <MetricCard label="模型输入" value={<StatusBadge tone="success">SAFE</StatusBadge>} detail={<>有效目标通道 57 / 64 · Zero-fill 7<br />目标采样率 100 Hz</>} tone="success" />

      <div className="span-8"><PredictionCard /></div>
      <div className="span-4"><LatencyCard /></div>

      <SectionCard title="最近预测" eyebrow="RECENT PREDICTIONS" className="span-7">
        <div className="timeline-list">{predictions.map(([id, result, score], index) => <div className="timeline-item" key={id}><strong>{id}</strong><span className="event-chip">窗口</span><strong>{result}</strong><span>{score} · {index % 2 ? "0.5 秒前" : "刚刚"}</span></div>)}</div>
      </SectionCard>
      <SectionCard title="当前运行" eyebrow="ACTIVE RUN" className="span-5" action={<StatusBadge tone="running">运行中</StatusBadge>}>
        <div className="key-value-list"><KeyValue label="Run ID" value="#842" /><KeyValue label="模式" value="实时运行" /><KeyValue label="被试" value="S01" /><KeyValue label="模型" value="50M Personal" /><KeyValue label="运行时间" value="00:14:32" /><KeyValue label="窗口" value="105" /></div>
        <div className="inspector-actions"><Link className="button button-secondary" href="/runs">查看运行详情</Link></div>
      </SectionCard>

      <SectionCard title="最近运行" eyebrow="RECENT RUNS" className="span-8">
        <div className="table-wrap"><table className="data-table"><thead><tr><th>Run ID</th><th>类型</th><th>模型</th><th>被试</th><th>状态</th></tr></thead><tbody><tr><td><strong>#842</strong></td><td>实时运行</td><td>50M Personal</td><td>S01</td><td><StatusBadge tone="running">运行中</StatusBadge></td></tr><tr><td><strong>#841</strong></td><td>离线回放</td><td>50M Population</td><td>S01</td><td><StatusBadge tone="success">已完成</StatusBadge></td></tr><tr><td><strong>#840</strong></td><td>实验评估</td><td>LaBraM</td><td>S01</td><td><StatusBadge tone="success">已完成</StatusBadge></td></tr></tbody></table></div>
      </SectionCard>
      <SectionCard title="系统检查" eyebrow="SYSTEM CHECK" className="span-4">
        <div className="check-list">{["设备连接", "Runtime", "模型加载", "输入 Contract", "Packet continuity"].map(item => <div className="check-item" key={item}><span>{item}</span><span className="check-ok">✓ 正常</span></div>)}</div><div className="notice" style={{marginTop: 16}}>当前无异常，系统可以继续运行。</div>
      </SectionCard>
    </div>
  </>;
}
