import { PredictionCard } from "@/components/status/prediction-card";
import { KeyValue, MetricCard, PageHeader, SectionCard, StatusBadge } from "@/components/ui/design-system";

export default function LivePage() {
  return <>
    <PageHeader title="实时运行" description="P0 预留实时 Gate、预测与事件时间线；当前版本不连接真实 JellyFish 设备。" action={<StatusBadge tone="idle">尚未启用</StatusBadge>} />
    <div className="notice warning" style={{marginBottom:16}}>实时设备接入属于 P1。本页仅定义显示合同，不启动采集、不改写 Stage 2B Runtime。</div>
    <div className="dashboard-grid">
      <MetricCard label="连接状态" value={<StatusBadge tone="idle">未连接</StatusBadge>} detail="等待 P1 设备适配" />
      <MetricCard label="EEG 通道" value="—" detail="仅显示通道元数据" />
      <MetricCard label="采样率" value="— Hz" detail="等待设备握手" />
      <MetricCard label="Trigger" value={<StatusBadge tone="idle">待检查</StatusBadge>} />
      <MetricCard label="数据连续性" value={<StatusBadge tone="idle">待检查</StatusBadge>} />
      <MetricCard label="模型输入" value={<StatusBadge tone="idle">未验证</StatusBadge>} detail="未通过 Gate 前不推理" />
      <div className="span-7"><PredictionCard invalid /></div>
      <SectionCard title="输出命令" eyebrow="DEMO OUTPUT" className="span-5"><div className="prediction-hero"><span>STOP</span><strong style={{color:"var(--text-secondary)"}}>—</strong><small>未达到可运行条件</small></div><div className="notice">只有输入合同为 SAFE 且置信度达标时才允许输出命令。</div></SectionCard>
      <SectionCard title="运行时间线" eyebrow="EVENT TIMELINE" className="span-8"><div className="timeline-list">{["Trigger", "Window", "Prediction", "Command"].map(item => <div className="timeline-item" key={item}><strong>—</strong><span className="event-chip">{item}</span><span>等待实时事件</span><span>—</span></div>)}</div></SectionCard>
      <SectionCard title="设备与运行健康" className="span-4"><div className="key-value-list"><KeyValue label="收到 Packet" value="0" /><KeyValue label="丢失" value="0" /><KeyValue label="重复" value="0" /><KeyValue label="乱序" value="0" /><KeyValue label="Gap" value="0" /><KeyValue label="模型输入失败" value="0" /><KeyValue label="预测失败" value="0" /></div></SectionCard>
    </div>
  </>;
}
