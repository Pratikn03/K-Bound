export const ROUTES = [
    { path: "/", id: "overview", label: "Overview" },
    { path: "/theory", id: "theory", label: "Theory Ledger" },
    { path: "/evidence", id: "evidence", label: "Evidence Map" },
    { path: "/experiments", id: "experiments", label: "Experiments" },
    { path: "/edge", id: "edge", label: "Edge Validation" },
    { path: "/safety", id: "safety", label: "Safety Boundary" },
    { path: "/reproduce", id: "reproduce", label: "Reproduce" },
    { path: "/artifacts", id: "artifacts", label: "Artifacts" },
];
export function parseRoute() {
    const hash = window.location.hash.replace(/^#/, "") || "/";
    const path = hash.startsWith("/") ? hash.split("?")[0] : `/${hash}`;
    const known = ROUTES.some((r) => r.path === path);
    return known ? path : "/";
}
export function navigate(path) {
    window.location.hash = path;
}
export function onRouteChange(cb) {
    const handler = () => cb(parseRoute());
    window.addEventListener("hashchange", handler);
    handler();
}
export function routeLabel(path) {
    return ROUTES.find((r) => r.path === path)?.label ?? "Overview";
}
export function buildPageRenderers(renderers) {
    const map = new Map();
    for (const r of ROUTES) {
        map.set(r.path, renderers[r.id] ?? renderers.overview);
    }
    return map;
}
//# sourceMappingURL=router.js.map