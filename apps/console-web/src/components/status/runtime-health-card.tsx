import { KeyValue, SectionCard, StatusBadge } from "@/components/ui/design-system";
import type { RuntimeHealthPayload } from "@/types/api";

export function RuntimeHealthCard({ value }: { value?: RuntimeHealthPayload | null }) {
  const hasValue = value != null;
  return <SectionCard title="运行健康" action={<StatusBadge tone={hasValue && value.failed_windows === 0 ? "success" : "idle"}>{hasValue ? (value.failed_windows === 0 ? "正常" : "异常") : "—"}</StatusBadge>}><div className="key-value-list"><KeyValue label="成功窗口" value={value?.successful_windows ?? "—"} /><KeyValue label="失败窗口" value={value?.failed_windows ?? "—"} /><KeyValue label="预计窗口" value={value?.expected_windows ?? "—"} /></div></SectionCard>;
}
