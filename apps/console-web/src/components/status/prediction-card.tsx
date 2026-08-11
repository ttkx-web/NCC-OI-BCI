import { ProgressBar, SectionCard } from "@/components/ui/design-system";
import { percent } from "@/lib/format/value";

const labels = ["LEFT", "RIGHT", "FEET", "TONGUE"];
const names: Record<string, string> = { left_hand: "左手", right_hand: "右手", feet: "双脚", tongue: "舌部" };

export function PredictionCard({ prediction = "left_hand", confidence = 0.824, probabilities = [0.824, 0.1, 0.046, 0.03], invalid = false }: { prediction?: string; confidence?: number; probabilities?: number[]; invalid?: boolean }) {
  if (invalid) {
    return <SectionCard title="当前预测" className="prediction-card blocked-card"><div className="blocked-symbol">!</div><h3>推理已阻断</h3><p>模型输入合同不安全，旧预测已失效。</p></SectionCard>;
  }
  return (
    <SectionCard title="当前预测" className="prediction-card">
      <div className="prediction-hero"><span>{labels[["left_hand", "right_hand", "feet", "tongue"].indexOf(prediction)] ?? prediction.toUpperCase()}</span><strong>{percent(confidence)}</strong><small>运动想象 · {names[prediction] ?? prediction}</small></div>
      <div className="probability-list">
        {labels.map((label, index) => <div className="probability-row" key={label}><span>{label}</span><ProgressBar value={(probabilities[index] ?? 0) * 100} /><strong>{percent(probabilities[index] ?? 0, 0)}</strong></div>)}
      </div>
      <div className="command-line"><span>输出命令</span><strong>{labels[["left_hand", "right_hand", "feet", "tongue"].indexOf(prediction)] ?? "STOP"}</strong></div>
    </SectionCard>
  );
}
