import { ProgressBar, SectionCard } from "@/components/ui/design-system";
import { percent } from "@/lib/format/value";

const labels = ["LEFT", "RIGHT", "FEET", "TONGUE"];
const names: Record<string, string> = { left_hand: "左手", right_hand: "右手", feet: "双脚", tongue: "舌部" };

export function PredictionCard({
  prediction,
  confidence,
  probabilities,
  command = "STOP",
  invalid = false,
}: { prediction?: string; confidence?: number; probabilities?: number[]; command?: string; invalid?: boolean }) {
  if (invalid) {
    return <SectionCard title="当前预测" className="prediction-card blocked-card"><div className="blocked-symbol">!</div><h3>已阻断</h3><p>模型输入契约不安全，旧预测已失效。</p></SectionCard>;
  }
  if (!prediction || confidence == null || !probabilities) {
    return <SectionCard title="当前预测" className="prediction-card"><div className="prediction-hero"><span>—</span><strong>—</strong><small>暂无运行</small></div><div className="command-line"><span>输出命令</span><strong>STOP</strong></div></SectionCard>;
  }
  return <SectionCard title="当前预测" className="prediction-card">
    <div className="prediction-hero"><span>{labels[["left_hand", "right_hand", "feet", "tongue"].indexOf(prediction)] ?? prediction.toUpperCase()}</span><strong>{percent(confidence)}</strong><small>运动想象 · {names[prediction] ?? prediction}</small></div>
    <div className="probability-list">{labels.map((label, index) => <div className="probability-row" key={label}><span>{label}</span><ProgressBar value={(probabilities[index] ?? 0) * 100} /><strong>{percent(probabilities[index] ?? 0, 0)}</strong></div>)}</div>
    <div className="command-line"><span>输出命令</span><strong>{command}</strong></div>
  </SectionCard>;
}
