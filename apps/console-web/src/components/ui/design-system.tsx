import type { ReactNode } from "react";

type Tone = "success" | "running" | "warning" | "danger" | "idle" | "brand";

export function StatusBadge({ tone, children }: { tone: Tone; children: ReactNode }) {
  return <span className={`status-badge status-${tone}`}><span className="status-dot" />{children}</span>;
}

export function SectionCard({
  title,
  eyebrow,
  action,
  className = "",
  children,
}: {
  title?: string;
  eyebrow?: string;
  action?: ReactNode;
  className?: string;
  children: ReactNode;
}) {
  return (
    <section className={`section-card ${className}`}>
      {(title || eyebrow || action) && (
        <div className="card-heading">
          <div>{eyebrow && <span className="eyebrow">{eyebrow}</span>}{title && <h2>{title}</h2>}</div>
          {action}
        </div>
      )}
      {children}
    </section>
  );
}

export function MetricCard({ label, value, detail, tone = "idle" }: { label: string; value: ReactNode; detail?: ReactNode; tone?: Tone }) {
  return (
    <SectionCard className="metric-card">
      <div className="metric-label">{label}</div>
      <div className={`metric-value metric-${tone}`}>{value}</div>
      {detail && <div className="metric-detail">{detail}</div>}
    </SectionCard>
  );
}

export function PageHeader({ title, description, action }: { title: string; description: string; action?: ReactNode }) {
  return (
    <div className="page-header">
      <div><span className="page-kicker">NCC BCI CONSOLE</span><h1>{title}</h1><p>{description}</p></div>
      {action}
    </div>
  );
}

export function EmptyState({ title, description }: { title: string; description: string }) {
  return <div className="empty-state"><div className="empty-mark">◇</div><strong>{title}</strong><p>{description}</p></div>;
}

export function ErrorState({ message }: { message: string }) {
  return <div className="error-state"><strong>暂时无法读取</strong><span>{message}</span></div>;
}

export function KeyValue({ label, value }: { label: string; value: ReactNode }) {
  return <div className="key-value"><span>{label}</span><strong>{value}</strong></div>;
}

export function ProgressBar({ value, tone = "brand" }: { value: number; tone?: Tone }) {
  return <div className="progress-track"><span className={`progress-fill progress-${tone}`} style={{ width: `${Math.max(0, Math.min(100, value))}%` }} /></div>;
}
