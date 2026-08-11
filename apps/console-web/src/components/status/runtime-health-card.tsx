import { KeyValue, SectionCard, StatusBadge } from "@/components/ui/design-system";
import type { RuntimeHealthPayload } from "@/types/api";

export function RuntimeHealthCard({ value }: { value?: RuntimeHealthPayload | null }) {
  return <SectionCard title="运行健康" action={<StatusBadge tone="success">正常</StatusBadge>}><div className="key-value-list"><KeyValue label="成功窗口" value={value?.successful_windows ?? 105} /><KeyValue label="失败窗口" value={value?.failed_windows ?? 0} /><KeyValue label="预计窗口" value={value?.expected_windows ?? 200} /></div></SectionCard>;
}
