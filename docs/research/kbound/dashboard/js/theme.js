const STORAGE_KEY = "kb-theme";
export function initTheme() {
    const stored = localStorage.getItem(STORAGE_KEY) ?? "auto";
    applyTheme(stored);
}
export function currentTheme() {
    return document.documentElement.dataset.theme || "auto";
}
export function cycleTheme() {
    const order = ["auto", "light", "dark"];
    const idx = order.indexOf(currentTheme());
    const next = order[(idx + 1) % order.length];
    applyTheme(next);
    return next;
}
export function applyTheme(mode) {
    document.documentElement.dataset.theme = mode;
    localStorage.setItem(STORAGE_KEY, mode);
    const btn = document.getElementById("theme-toggle");
    if (btn) {
        const labels = {
            auto: "Theme: Auto",
            light: "Theme: Light",
            dark: "Theme: Dark",
        };
        btn.textContent = labels[mode];
        btn.setAttribute("aria-label", `Color theme ${mode}`);
    }
}
export function bindThemeToggle() {
    document.getElementById("theme-toggle")?.addEventListener("click", () => cycleTheme());
}
//# sourceMappingURL=theme.js.map