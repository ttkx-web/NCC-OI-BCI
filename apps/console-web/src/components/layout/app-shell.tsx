"use client";

import Image from "next/image";
import Link from "next/link";
import { usePathname } from "next/navigation";
import type { ReactNode } from "react";

import { runStateLabel, useRuntimeStatus } from "@/components/runtime/run-status-provider";

const navigation = [
  { href: "/", label: "总览", icon: "总" }, { section: "运行" },
  { href: "/replay", label: "离线回放", icon: "回" }, { href: "/live", label: "实时运行", icon: "实" },
  { href: "/experiments", label: "实验评估", icon: "评" }, { section: "资源" },
  { href: "/data", label: "数据管理", icon: "数" }, { href: "/models", label: "模型管理", icon: "模" },
  { href: "/subjects", label: "被试管理", icon: "被" }, { href: "/runs", label: "运行记录", icon: "记" },
  { href: "/system", label: "系统状态", icon: "系" },
] as const;

export function AppShell({ children }: { children: ReactNode }) {
  const pathname = usePathname();
  const { activeRun, activeModel, deviceHealth, latency, runtimeHealth, system } = useRuntimeStatus();
  const deviceConnected = deviceHealth.connected === true || (activeRun?.run_type === "live" && system?.device.status === "connected");
  const subject = activeRun?.subject_id ?? "—";
  const model = activeModel?.model_name ?? "—";
  const state = runStateLabel(activeRun?.state);
  const windows = activeRun ? `${runtimeHealth?.successful_windows ?? activeRun.successful_windows} / ${runtimeHealth?.expected_windows ?? activeRun.expected_windows ?? "—"}` : "—";
  const p50 = latency?.p50_ms == null ? "—" : `${latency.p50_ms.toFixed(1)} ms`;
  return <div className="app-shell">
    <header className="topbar">
      <div className="brand-lockup"><Image className="brand-logo" src="/brand/omni-intelligence.jpg" alt="全域智能 Omni-Intelligence" width={251} height={90} priority /><div><strong>NCC BCI Console</strong><small>脑机接口运行与实验控制台</small></div></div>
      <div className="header-context"><div><span>当前被试</span><strong>{subject}</strong></div><div><span>当前模型</span><strong>{model}</strong></div><div><span>运行状态</span><strong className={activeRun?.state === "running" ? "live-state" : ""}>{activeRun?.state === "running" && <i />}{state}</strong></div></div>
    </header>
    <aside className="sidebar"><nav>{navigation.map((item, index) => "section" in item ? <div className="nav-section" key={`${item.section}-${index}`}>{item.section}</div> : <Link href={item.href} className={pathname === item.href ? "nav-item active" : "nav-item"} key={item.href}><span className="nav-icon">{item.icon}</span>{item.label}<span className="nav-arrow">›</span></Link>)}</nav><div className="sidebar-note"><span>控制台版本</span><strong>Web Console</strong><small>Runtime Schema v2</small></div></aside>
    <main className="main-content">{children}</main>
    <footer className="statusbar"><span><i className={`dot ${deviceConnected ? "success" : "idle"}`} />设备 <strong>{deviceConnected ? "已连接" : "未连接"}</strong></span><span><i className={`dot ${activeRun?.state === "running" ? "running" : "idle"}`} />运行 <strong>{state}</strong></span><span>窗口 <strong>{windows}</strong></span><span>P50 <strong>{p50}</strong></span><span className="status-spacer" /><span>Run ID <strong>{activeRun?.id ?? "—"}</strong></span></footer>
  </div>;
}
