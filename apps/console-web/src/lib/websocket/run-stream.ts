import { API_BASE } from "@/lib/api/client";
import type { RunEvent } from "@/types/api";

export function connectRun(runId: string, onEvent: (event: RunEvent) => void): WebSocket {
  const base = new URL(API_BASE);
  const protocol = base.protocol === "https:" ? "wss:" : "ws:";
  const socket = new WebSocket(`${protocol}//${base.host}/ws/v1/runs/${runId}`);
  socket.onmessage = (message) => onEvent(JSON.parse(message.data) as RunEvent);
  return socket;
}

