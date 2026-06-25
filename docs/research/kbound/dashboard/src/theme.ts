export type ThemeMode = "auto" | "light" | "dark";

const STORAGE_KEY = "kb-theme";

export function initTheme(): void {
  const stored = (localStorage.getItem(STORAGE_KEY) as ThemeMode | null) ?? "auto";
  applyTheme(stored);
}

export function currentTheme(): ThemeMode {
  return (document.documentElement.dataset.theme as ThemeMode) || "auto";
}

export function cycleTheme(): ThemeMode {
  const order: ThemeMode[] = ["auto", "light", "dark"];
  const idx = order.indexOf(currentTheme());
  const next = order[(idx + 1) % order.length];
  applyTheme(next);
  return next;
}

export function applyTheme(mode: ThemeMode): void {
  document.documentElement.dataset.theme = mode;
  localStorage.setItem(STORAGE_KEY, mode);
  const btn = document.getElementById("theme-toggle");
  if (btn) {
    const labels: Record<ThemeMode, string> = {
      auto: "Theme: Auto",
      light: "Theme: Light",
      dark: "Theme: Dark",
    };
    btn.textContent = labels[mode];
    btn.setAttribute("aria-label", `Color theme ${mode}`);
  }
}

export function bindThemeToggle(): void {
  document.getElementById("theme-toggle")?.addEventListener("click", () => cycleTheme());
}
