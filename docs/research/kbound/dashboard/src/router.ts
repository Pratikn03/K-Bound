import type { RouteDef, Snapshot } from "./types.js";

export const ROUTES: RouteDef[] = [
  { path: "/", id: "overview", label: "Overview" },
  { path: "/theory", id: "theory", label: "Theory Ledger" },
  { path: "/evidence", id: "evidence", label: "Evidence Map" },
  { path: "/experiments", id: "experiments", label: "Experiments" },
  { path: "/edge", id: "edge", label: "Edge Validation" },
  { path: "/safety", id: "safety", label: "Safety Boundary" },
  { path: "/reproduce", id: "reproduce", label: "Reproduce" },
  { path: "/artifacts", id: "artifacts", label: "Artifacts" },
];

export function parseRoute(): string {
  const hash = window.location.hash.replace(/^#/, "") || "/";
  const path = hash.startsWith("/") ? hash.split("?")[0] : `/${hash}`;
  const known = ROUTES.some((r) => r.path === path);
  return known ? path : "/";
}

export function navigate(path: string): void {
  window.location.hash = path;
}

export function onRouteChange(cb: (path: string) => void): void {
  const handler = () => cb(parseRoute());
  window.addEventListener("hashchange", handler);
  handler();
}

export function routeLabel(path: string): string {
  return ROUTES.find((r) => r.path === path)?.label ?? "Overview";
}

export type PageRenderer = (data: Snapshot) => string;

export function buildPageRenderers(renderers: Record<string, PageRenderer>): Map<string, PageRenderer> {
  const map = new Map<string, PageRenderer>();
  for (const r of ROUTES) {
    map.set(r.path, renderers[r.id] ?? renderers.overview);
  }
  return map;
}
