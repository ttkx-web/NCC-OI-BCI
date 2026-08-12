import type { Metadata } from "next";
import type { ReactNode } from "react";

import { AppShell } from "@/components/layout/app-shell";
import { RuntimeStatusProvider } from "@/components/runtime/run-status-provider";
import "./globals.css";

export const metadata: Metadata = {
  title: "NCC BCI Console",
  description: "脑机接口运行与实验控制台",
};

export default function RootLayout({ children }: Readonly<{ children: ReactNode }>) {
  return <html lang="zh-CN"><body><RuntimeStatusProvider><AppShell>{children}</AppShell></RuntimeStatusProvider></body></html>;
}
