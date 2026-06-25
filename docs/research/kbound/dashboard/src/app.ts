import {
  architectureSection,
  artifactFooter,
  edgeStudyStatus,
  evidenceBoard,
  evidenceStrip,
  experimentsPage,
  regimeMap,
  reproducePanel,
  researchHeader,
  safetyBoundary,
  theoryLedger,
} from "./components.js";
import {
  ROUTES,
  buildPageRenderers,
  navigate,
  onRouteChange,
  parseRoute,
  routeLabel,
} from "./router.js";
import type { Snapshot } from "./types.js";
import { bindThemeToggle, initTheme } from "./theme.js";

let snapshot: Snapshot | null = null;

async function loadSnapshot(): Promise<Snapshot> {
  const res = await fetch("dashboard/data/snapshot.json", { cache: "no-store" });
  if (!res.ok) throw new Error(`snapshot.json ${res.status}`);
  return res.json() as Promise<Snapshot>;
}

function renderNav(activePath: string): string {
  return ROUTES.map((r) => {
    const href = `#${r.path}`;
    const active = r.path === activePath ? " is-active" : "";
    return `<li><a class="nav-link${active}" href="${href}" data-route="${r.path}">${r.label}</a></li>`;
  }).join("");
}

function pageContent(path: string, data: Snapshot): string {
  const pages = buildPageRenderers({
    overview: (d) => `${researchHeader(d)}${evidenceStrip(d.evidence_strip)}${regimeMap(d.regime_map)}`,
    theory: (d) => theoryLedger(d.theory_ledger),
    evidence: (d) => evidenceBoard(d.evidence_board, d.headline_controlled),
    experiments: (d) => experimentsPage(d.evidence_board, d.headline_controlled),
    edge: (d) => edgeStudyStatus(d.edge_validation),
    safety: (d) => safetyBoundary(d.safety),
    reproduce: (d) => `${reproducePanel(d.reproduce)}${architectureSection()}`,
    artifacts: (d) => artifactFooter(d.meta, d.provenance),
  });
  const render = pages.get(path) ?? pages.get("/")!;
  return `<div class="route-view" data-route="${path}">${render(data)}</div>`;
}

function renderShell(data: Snapshot, path: string): void {
  const meta = data.meta ?? {};
  const root = document.getElementById("app-root");
  if (!root) return;
  root.className = "app-shell";
  root.innerHTML = `
    <div class="sidebar-overlay" id="sidebar-overlay"></div>
    <aside class="sidebar" id="sidebar" aria-label="Primary">
      <div class="sidebar-brand">Research console</div>
      <ul class="nav-list" id="nav-list">${renderNav(path)}</ul>
    </aside>
    <div class="main">
      <header class="topbar">
        <div class="topbar-left">
          <button type="button" class="mobile-nav-toggle" id="nav-toggle" aria-label="Open navigation">Menu</button>
          <span class="wordmark">K-BOUND</span>
          <span class="route-title">${routeLabel(path)}</span>
        </div>
        <div class="topbar-meta">
          <button type="button" class="btn-theme" id="theme-toggle" aria-label="Color theme">Theme</button>
          <span class="build-id">build ${meta.build_id ?? "—"}</span>
          <span class="badge badge--verified">Artifact-backed</span>
        </div>
      </header>
      <main class="content" id="page-content">
        ${pageContent(path, data)}
      </main>
    </div>`;
  bindInteractions(path);
  initTheme();
  bindThemeToggle();
}

function bindInteractions(path: string): void {
  const toggle = document.getElementById("nav-toggle");
  const sidebar = document.getElementById("sidebar");
  const overlay = document.getElementById("sidebar-overlay");
  const close = () => {
    sidebar?.classList.remove("is-open");
    overlay?.classList.remove("is-visible");
  };
  toggle?.addEventListener("click", () => {
    sidebar?.classList.toggle("is-open");
    overlay?.classList.toggle("is-visible");
  });
  overlay?.addEventListener("click", close);

  document.querySelectorAll(".nav-link").forEach((a) => {
    a.addEventListener("click", (e) => {
      e.preventDefault();
      const route = (a as HTMLAnchorElement).dataset.route ?? "/";
      navigate(route);
      close();
    });
  });

  document.querySelectorAll(".btn-copy").forEach((btn) => {
    btn.addEventListener("click", async () => {
      const id = btn.getAttribute("data-copy-target");
      const el = id ? document.getElementById(id) : null;
      if (!el) return;
      try {
        await navigator.clipboard.writeText(el.textContent ?? "");
        btn.textContent = "Copied";
        window.setTimeout(() => {
          btn.textContent = "Copy";
        }, 1500);
      } catch {
        btn.textContent = "Failed";
      }
    });
  });

  if (path === "/edge") {
    bindVideoTabs();
    bindLiveCameraPreview();
  }
}

let liveCameraStream: MediaStream | null = null;

async function bindLiveCameraPreview(): Promise<void> {
  const video = document.getElementById("live-camera-preview") as HTMLVideoElement | null;
  const select = document.getElementById("live-camera-device") as HTMLSelectElement | null;
  const status = document.getElementById("live-camera-status");
  if (!video || !select || !status) return;

  const setStatus = (msg: string) => {
    status.textContent = msg;
  };

  const stopStream = () => {
    if (liveCameraStream) {
      liveCameraStream.getTracks().forEach((t) => t.stop());
      liveCameraStream = null;
    }
    video.srcObject = null;
  };

  const startDevice = async (deviceId: string) => {
    stopStream();
    setStatus("Starting camera…");
    try {
      const constraints: MediaStreamConstraints = {
        video: deviceId ? { deviceId: { exact: deviceId } } : { facingMode: "environment" },
        audio: false,
      };
      liveCameraStream = await navigator.mediaDevices.getUserMedia(constraints);
      video.srcObject = liveCameraStream;
      await video.play().catch(() => undefined);
      const track = liveCameraStream.getVideoTracks()[0];
      const label = track?.label || "camera";
      setStatus(`Live: ${label}`);
    } catch (err) {
      setStatus(`Camera blocked or unavailable: ${err instanceof Error ? err.message : String(err)}. Allow Chrome/Safari in System Settings → Privacy → Camera.`);
    }
  };

  const screenBtn = document.getElementById("live-screen-share-btn") as HTMLButtonElement | null;
  screenBtn?.addEventListener("click", async () => {
    stopStream();
    setStatus("Requesting screen share permission...");
    try {
      if (!navigator.mediaDevices?.getDisplayMedia) {
        setStatus("Screen sharing is not supported by your browser.");
        return;
      }
      liveCameraStream = await navigator.mediaDevices.getDisplayMedia({
        video: true,
        audio: false
      });
      video.srcObject = liveCameraStream;
      await video.play().catch(() => undefined);
      setStatus("Live: Computer Screen Sharing");
      
      liveCameraStream.getVideoTracks()[0].addEventListener("ended", () => {
        setStatus("Screen share ended.");
      });
    } catch (err) {
      setStatus(`Screen share failed: ${err instanceof Error ? err.message : String(err)}`);
    }
  });

  if (!navigator.mediaDevices?.getUserMedia) {
    setStatus("Browser camera API not available (use Chrome/Safari on localhost).");
    return;
  }

  try {
    await startDevice("");
    const devices = (await navigator.mediaDevices.enumerateDevices()).filter((d) => d.kind === "videoinput");
    select.innerHTML = "";
    for (const d of devices) {
      const opt = document.createElement("option");
      opt.value = d.deviceId;
      opt.textContent = d.label || `Camera ${select.length + 1}`;
      select.appendChild(opt);
    }
    const iphone = devices.find((d) => /iphone|continuity/i.test(d.label));
    if (iphone) {
      select.value = iphone.deviceId;
      await startDevice(iphone.deviceId);
    } else if (devices[0]) {
      select.value = devices[0].deviceId;
    }
    select.addEventListener("change", () => {
      void startDevice(select.value);
    });
  } catch (err) {
    setStatus(`Camera error: ${err instanceof Error ? err.message : String(err)}`);
  }
}

function bindVideoTabs(): void {
  const player = document.getElementById("demo-player") as HTMLVideoElement | null;
  document.querySelectorAll(".video-tab").forEach((tab) => {
    tab.addEventListener("click", () => {
      document.querySelectorAll(".video-tab").forEach((t) => t.classList.remove("is-active"));
      tab.classList.add("is-active");
      const src = tab.getAttribute("data-video");
      if (player && src) {
        player.src = src;
        player.play().catch(() => undefined);
      }
    });
  });
}

function renderRoute(path: string): void {
  if (!snapshot) return;
  const content = document.getElementById("page-content");
  if (!content) {
    renderShell(snapshot, path);
    return;
  }
  content.innerHTML = pageContent(path, snapshot);
  document.querySelectorAll(".nav-link").forEach((a) => {
    a.classList.toggle("is-active", (a as HTMLAnchorElement).dataset.route === path);
  });
  const title = document.querySelector(".route-title");
  if (title) title.textContent = routeLabel(path);
  bindInteractions(path);
  window.scrollTo({ top: 0, behavior: "smooth" });
}

async function main(): Promise<void> {
  const root = document.getElementById("app-root");
  if (!root) return;
  root.innerHTML = `<div class="loading">Loading artifact snapshot…</div>`;
  try {
    snapshot = await loadSnapshot();
    const path = parseRoute();
    renderShell(snapshot, path);
    onRouteChange(renderRoute);
  } catch (err) {
    const msg = err instanceof Error ? err.message : String(err);
    root.innerHTML = `<div class="error-state">Failed to load dashboard: ${msg}. Run <code>bash docs/research/kbound/scripts/build_dashboard.sh</code>.</div>`;
  }
}

main();
