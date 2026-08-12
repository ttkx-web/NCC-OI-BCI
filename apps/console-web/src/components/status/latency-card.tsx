import { KeyValue, SectionCard } from "@/components/ui/design-system";
import { milliseconds } from "@/lib/format/value";
import type { LatencyPayload } from "@/types/api";

export function LatencyCard({ value }: { value?: LatencyPayload | null }) {
  return <SectionCard title="运行性能"><div className="key-value-list"><KeyValue label="预处理" value={milliseconds(value?.prepare_ms)} /><KeyValue label="模型推理" value={milliseconds(value?.inference_ms)} /><KeyValue label="当前总延迟" value={milliseconds(value?.total_ms)} /><KeyValue label="P50" value={milliseconds(value?.p50_ms)} /><KeyValue label="P95" value={milliseconds(value?.p95_ms)} /></div></SectionCard>;
}
