import type { StatusKind } from "./types.js";

export const STATUS_LABELS: Record<string, string> = {
  verified: "Proven",
  conditional: "Conditional",
  open: "Open",
  pending: "Pending",
  diagnostic: "Development",
  withheld: "Withheld",
  failed: "Blocked",
  no_harm: "No-harm",
};

export function esc(s: unknown): string {
  if (s == null) return "—";
  return String(s)
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;");
}

export function badge(status: StatusKind | string, text?: string): string {
  const label = text || STATUS_LABELS[status] || status;
  return `<span class="badge badge--${status}">${esc(label)}</span>`;
}

export function fmtNum(v: unknown, digits = 3): string {
  if (v == null || Number.isNaN(Number(v))) return "—";
  const n = Number(v);
  if (Math.abs(n) >= 10) return n.toFixed(2);
  if (Math.abs(n) >= 1) return n.toFixed(3);
  return n.toFixed(digits);
}

export function fmtPct(v: unknown): string {
  if (v == null) return "—";
  return `${(Number(v) * 100).toFixed(1)}%`;
}

export function fmtMs(v: unknown): string {
  if (v == null) return "—";
  return `${Number(v).toFixed(0)} ms`;
}

export function cmdBlock(label: string, text: string): string {
  const id = `cmd-${Math.random().toString(36).slice(2, 9)}`;
  return `<div class="cmd-block">
    <div class="cmd-block-header"><span>${esc(label)}</span><button type="button" class="btn-copy" data-copy-target="${id}">Copy</button></div>
    <pre id="${id}">${esc(text)}</pre>
  </div>`;
}
