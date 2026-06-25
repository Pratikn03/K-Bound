import { badge, cmdBlock, esc, fmtMs, fmtNum, fmtPct } from "./format.js";
function statusChip(label, status) {
    const dotClass = status === "verified" ? "verified" : status === "pending" ? "pending" : "open";
    return `<div class="status-chip"><span class="status-dot status-dot--${dotClass}"></span>${esc(label)}</div>`;
}
export function researchHeader(data) {
    const rs = data.research_status ?? {};
    return `
    <header class="page-header editorial-header">
      <div class="eyebrow">K-BOUND / LABEL-FREE ADAPTATION</div>
      <h1>Can an unlabeled model update be trusted?</h1>
      <p class="lede">K-Bound decides whether a proposed test-time adaptation is certifiably helpful, certifiably harmful, or unknowable.</p>
      <div class="status-row">
        ${statusChip("Theory", rs.theory)}
        ${statusChip("Controlled evidence", rs.controlled)}
        ${statusChip("Natural shifts", rs.natural_shifts)}
        ${statusChip("Edge study", rs.edge_study)}
      </div>
      <div class="callout">
        <strong>Claim boundary.</strong> K-Bound does not promise universal accuracy gains. Its role is to prevent harmful adaptation when label-free evidence can justify a decision.
      </div>
    </header>`;
}
export function evidenceStrip(strip) {
    if (!strip)
        return "";
    const items = [
        ["proven_theorems", "Proven theorems"],
        ["theorem_validators", "Theorem validators"],
        ["controlled_beats_both", "Controlled beats-both"],
        ["natural_shift_no_harm", "Natural-shift no-harm"],
        ["open_theory", "Open questions"],
        ["reproducibility", "Reproducibility"],
    ];
    const cells = items
        .map(([key, label]) => {
        const item = strip[key] ?? {};
        return `<div class="evidence-strip__item">
        <div class="evidence-strip__label">${esc(label)}</div>
        <div class="evidence-strip__value">${esc(item.value)}</div>
        <div class="evidence-strip__sub">${esc(item.sub)}</div>
      </div>`;
    })
        .join("");
    return `<div class="evidence-strip">${cells}</div>`;
}
export function regimeMap(regimes) {
    const cards = (regimes ?? [])
        .map((r) => `
      <article class="regime-card">
        <div style="margin-bottom:8px">${badge(r.status)}</div>
        <h3>${esc(r.title)}</h3>
        <div class="regime-action">${esc(r.action)}</div>
        <p class="regime-examples">${esc(r.examples)}</p>
        ${r.artifact ? `<p class="regime-examples" style="margin-top:10px"><code>${esc(r.artifact)}</code></p>` : ""}
      </article>`)
        .join("");
    return `
    <div class="page-section">
      <div class="section-label">Research status map</div>
      <h2 class="section-title">Three adaptation regimes</h2>
      <div class="regime-map">${cards}</div>
    </div>`;
}
export function theoryLedger(rows) {
    const body = (rows ?? [])
        .map((r) => `
      <tr>
        <td>${esc(r.id)}</td>
        <td><strong>${esc(r.name)}</strong><div class="cell-sub">${esc(r.implication)}</div></td>
        <td>${badge(r.status)}</td>
        <td>${r.artifact ? `<code>${esc(r.artifact)}</code>` : "—"}</td>
        <td class="cell-muted">${esc(r.evidence)}</td>
      </tr>`)
        .join("");
    return `
    <div class="page-section">
      <div class="section-label">Theory ledger</div>
      <h2 class="section-title">Theorem status and evidence</h2>
      <div class="panel">
        <div class="panel-body panel-scroll">
          <table class="data-table">
            <thead><tr><th>#</th><th>Theorem</th><th>Status</th><th>Artifact</th><th>Evidence</th></tr></thead>
            <tbody>${body}</tbody>
          </table>
        </div>
      </div>
    </div>`;
}
function policyTableRows(rows) {
    return (rows ?? [])
        .map((r) => {
        const metrics = [r.freeze, r.adapt, r.kga, r.oracle]
            .map((v) => `<td class="num">${fmtNum(v)}</td>`)
            .join("");
        return `<tr>
        <td><strong>${esc(r.name)}</strong>
          <div class="cell-sub">${esc(r.framing)}</div>
          <code class="cell-code">${esc(r.artifact)}</code>
        </td>
        ${metrics}
        <td>${badge(r.status)}</td>
        <td class="num">${r.regret_kga != null ? fmtNum(r.regret_kga) : "—"}</td>
      </tr>`;
    })
        .join("");
}
export function evidenceBoard(board, headline) {
    const b = board ?? {};
    return `
    <div class="page-section">
      <div class="section-label">Evidence map</div>
      <h2 class="section-title">Experiment evidence board</h2>

      <div class="evidence-group">
        <h3 class="evidence-group-title">Core controlled suite</h3>
        <p class="evidence-group-desc">123-task anomaly routing and multiseed rigor from locked CPU experiments.</p>
        <div class="panel comparison-matrix">
          <div class="panel-body panel-scroll-x">
            <table class="data-table">
              <thead><tr><th>Experiment</th><th class="num">Freeze</th><th class="num">Adapt</th><th class="num">KGA</th><th class="num">Oracle</th><th>Status</th><th class="num">Regret</th></tr></thead>
              <tbody>${policyTableRows(headline)}</tbody>
            </table>
          </div>
        </div>
      </div>

      <div class="evidence-group">
        <h3 class="evidence-group-title">Helpful-dominated regimes</h3>
        <p class="evidence-group-desc">Per-corruption panels where adapt is strong; KGA matches adapt safely (not beats-both).</p>
        <div class="panel comparison-matrix">
          <div class="panel-body panel-scroll-x">
            <table class="data-table">
              <thead><tr><th>Benchmark</th><th class="num">Freeze</th><th class="num">Adapt</th><th class="num">KGA</th><th class="num">Oracle</th><th>Status</th><th class="num">Regret</th></tr></thead>
              <tbody>${policyTableRows(b.helpful_dominated)}</tbody>
            </table>
          </div>
        </div>
      </div>

      <div class="evidence-group" id="experiments">
        <h3 class="evidence-group-title">Controlled policy wins</h3>
        <p class="evidence-group-desc">Report “beats both” only where artifact <code>beats_both</code> is true.</p>
        <div class="panel comparison-matrix">
          <div class="panel-body panel-scroll-x">
            <table class="data-table">
              <thead><tr><th>Benchmark</th><th class="num">Freeze</th><th class="num">Adapt</th><th class="num">KGA</th><th class="num">Oracle</th><th>Status</th><th class="num">Regret</th></tr></thead>
              <tbody>${policyTableRows(b.controlled_wins)}</tbody>
            </table>
          </div>
        </div>
      </div>

      <div class="evidence-group">
        <h3 class="evidence-group-title">Natural-shift no-harm results</h3>
        <p class="evidence-group-desc">Matches the safer fixed policy on regret; avoids the worse always-adapt policy.</p>
        <div class="panel">
          <div class="panel-body panel-scroll-x">
            <table class="data-table">
              <thead><tr><th>Dataset</th><th class="num">Regret KGA</th><th class="num">Regret adapt</th><th class="num">Regret freeze</th><th class="num">FA_u</th><th>Status</th></tr></thead>
              <tbody>
                ${(b.natural_shift_no_harm ?? [])
        .map((r) => `<tr>
                      <td><strong>${esc(r.name)}</strong><div class="cell-sub">${esc(r.framing)}</div><code class="cell-code">${esc(r.artifact)}</code></td>
                      <td class="num">${fmtNum(r.regret_kga)}</td>
                      <td class="num">${fmtNum(r.regret_adapt)}</td>
                      <td class="num">${fmtNum(r.regret_freeze)}</td>
                      <td class="num">${fmtNum(r.false_adapt)}</td>
                      <td>${badge(r.status)}</td>
                    </tr>`)
        .join("")}
              </tbody>
            </table>
          </div>
        </div>
      </div>

      <div class="evidence-group">
        <h3 class="evidence-group-title">Boundary / negative evidence</h3>
        <p class="evidence-group-desc">Scientifically useful nulls — evidence insufficient for a valid commitment.</p>
        <div class="panel">
          <div class="panel-body panel-pad">
            ${(b.boundary_negative ?? []).map(boundaryCard).join("")}
          </div>
        </div>
      </div>
    </div>`;
}
function boundaryCard(r) {
    const metrics = [
        ["Freeze", r.freeze],
        ["Adapt", r.adapt],
        ["KGA", r.kga],
        ["Oracle", r.oracle],
        ["Regret KGA", r.regret_kga],
        ["Abstain rate", r.abstention_rate],
        ["FA_u", r.false_adapt],
    ]
        .filter(([, v]) => v != null)
        .map(([k, v]) => `<span class="metric-chip"><span class="metric-chip-k">${k}</span> ${fmtNum(v)}</span>`)
        .join("");
    return `<div class="boundary-card">
    <div class="boundary-card-head"><strong>${esc(r.name)}</strong>${badge(r.status)}</div>
    <p class="boundary-card-text">${esc(r.framing)}</p>
    ${metrics ? `<div class="metric-chip-row">${metrics}</div>` : ""}
    ${r.note ? `<p class="cell-muted">${esc(r.note)}</p>` : ""}
    <code class="cell-code">${esc(r.artifact)}</code>
  </div>`;
}
function edgeUnblockPanel(edge) {
    const u = edge.unblock;
    if (!u)
        return "";
    const gaps = u.gaps
        .map((g) => `<li class="${g.passed ? "gap-pass" : "gap-fail"}">
        <span>${g.passed ? "✓" : "○"}</span> ${esc(g.check)} <span class="cell-muted">(${esc(g.detail)})</span>
      </li>`)
        .join("");
    return `
    <div class="unblock-panel">
      <div class="section-label">Gate unblock playbook</div>
      <p class="cell-muted">Study remains <strong>pending</strong> until all checks pass without <code>--bypass-gate</code>. Capture guide: <code>edge/PHYSICAL_STUDY_RUNBOOK.md</code></p>
      <ul class="gap-list">${gaps}</ul>
      ${cmdBlock("Full publication pipeline (validate → train → calibrate → replay → audit → TeX)", u.commands.full_pipeline ?? "")}
      ${cmdBlock("Retrain source model only (no bypass)", u.commands.retrain_source ?? "")}
      ${cmdBlock("Replay held-out Phone A", u.commands.replay_heldout ?? "")}
      ${cmdBlock("Refresh dashboard snapshot", u.commands.refresh_dashboard ?? "")}
    </div>`;
}
export function edgeStudyStatus(edge) {
    if (!edge)
        return "";
    const isDev = edge.study_status !== "verified";
    const phases = (edge.phases ?? [])
        .map((p) => {
        const cls = `timeline-step is-${p.status}`;
        return `<div class="${cls}">
        <div class="timeline-dot"></div>
        <div class="timeline-label">${esc(p.label)}</div>
        <div class="timeline-detail">${badge(p.status)}</div>
        <div class="timeline-detail timeline-detail-sub">${esc(p.detail)}</div>
      </div>`;
    })
        .join("");
    const dev = edge.development_metrics;
    const devAccordion = dev
        ? `<details class="accordion">
        <summary>Raw development metrics (non-headline)</summary>
        <div class="accordion-body">
          <p>${esc(dev.note)}</p>
          <ul class="plain-list">
            <li>Phone A balanced acc: ${fmtPct(dev.phone_a_balanced_acc)}</li>
            <li>Phone A macro-F1: ${fmtPct(dev.phone_a_macro_f1)}</li>
            <li>KGA abstain rate: ${fmtPct(dev.kga_abstain_rate)}</li>
            <li>Current development latency (mean): ${fmtMs(dev.latency_ms_mean)}</li>
            <li>p95: ${fmtMs(dev.latency_ms_p95)}</li>
          </ul>
        </div>
      </details>`
        : "";
    return `
    <div class="page-section">
      <div class="section-label">Physical edge validation</div>
      <h2 class="section-title">Edge Validation — Preregistered Physical Study</h2>
      <div class="panel edge-protocol">
        <div class="panel-header">
          <div>
            <strong>${esc(edge.study_label)}</strong>
            <div class="cell-muted">Protocol hash ${esc(String(edge.protocol_hash ?? "").slice(0, 16))}…</div>
          </div>
          ${badge(edge.study_status, isDev ? "Not publication-ready" : "Study complete")}
        </div>
        ${isDev ? `<div class="callout callout--warning callout-inset">Pipeline and audit infrastructure exist, but source-model and held-out gates are not met. Metrics at chance level with full abstention must not be read as deployment success.</div>` : ""}
        ${edgeUnblockPanel(edge)}
        <div class="timeline">${phases}</div>
        ${shadowDiagram()}
        <div class="demo-row">
          <div>
            <div class="section-label">Live camera (this browser)</div>
            <p class="cell-muted">Uses your Mac/iPhone camera via the browser — for checking Continuity Camera before running the Python shadow window.</p>
            <div class="live-camera-wrap">
              <video id="live-camera-preview" class="live-camera-preview" autoplay playsinline muted></video>
              <div class="live-camera-controls">
                <label class="live-camera-label" for="live-camera-device">Device</label>
                <select id="live-camera-device" class="live-camera-select"></select>
                <button type="button" id="live-screen-share-btn" class="btn btn-secondary" style="margin-left: 8px; font-size: 11px; padding: 4px 8px; border-radius: 4px; background: var(--bd); border: 1px solid var(--bd); color: var(--tx); cursor: pointer; height: 26px;">Share Screen</button>
              </div>
              <p id="live-camera-status" class="cell-muted live-camera-status">Requesting camera…</p>
            </div>
            <div class="section-label" style="margin-top:16px">Recorded demos (not live)</div>
            <div class="video-wrap">
              <div class="video-tabs">
                <button type="button" class="video-tab is-active" data-video="edge/artifacts/laptop_live_demo.mp4">Phone A capture (demo)</button>
                <button type="button" class="video-tab" data-video="edge/artifacts/dashboard_demo.mp4">Synthetic shadow stream</button>
              </div>
              <video id="demo-player" controls loop muted playsinline>
                <source src="edge/artifacts/laptop_live_demo.mp4" type="video/mp4">
              </video>
            </div>
          </div>
          <div>
            <div class="section-label">Live shadow demo (OpenCV window)</div>
            <p class="cell-muted">Frozen model is official output; Tent candidate + KGA run in shadow only. The KGA overlay opens in a separate Python window — not inside this page.</p>
            ${cmdBlock("Looping simulation (shows adapt/freeze/abstain cycle)", `cd AutoML_Flagship_V8\n.venv/bin/python docs/research/kbound/edge/scripts/07_shadow_live.py \\\n  --config docs/research/kbound/edge/configs/edge_label_inspection_v1.yaml \\\n  --shadow-config docs/research/kbound/edge/configs/edge_shadow_v1.yaml \\\n  --view window \\\n  --loop`)}
            ${cmdBlock("Looping recorded video with safety radius override (ADAPT)", `cd AutoML_Flagship_V8\n.venv/bin/python docs/research/kbound/edge/scripts/07_shadow_live.py \\\n  --config docs/research/kbound/edge/configs/edge_label_inspection_v1.yaml \\\n  --shadow-config docs/research/kbound/edge/configs/edge_shadow_v1.yaml \\\n  --video docs/research/kbound/edge/artifacts_real/pilot/PILOT_item_01.mp4 \\\n  --view window \\\n  --loop \\\n  --eps 0.05`)}
            ${cmdBlock("Mac/iPhone camera (Continuity auto or index 1/2)", `cd AutoML_Flagship_V8\n.venv/bin/python docs/research/kbound/edge/scripts/07_shadow_live.py \\\n  --config docs/research/kbound/edge/configs/edge_label_inspection_v1.yaml \\\n  --shadow-config docs/research/kbound/edge/configs/edge_shadow_v1.yaml \\\n  --camera auto \\\n  --view window`)}
            ${cmdBlock("Synthetic / fake source (no camera)", `cd AutoML_Flagship_V8\n.venv/bin/python docs/research/kbound/edge/scripts/07_shadow_live.py \\\n  --config docs/research/kbound/edge/configs/edge_real_phone_v1.yaml \\\n  --shadow-config docs/research/kbound/edge/configs/edge_shadow_v1.yaml \\\n  --view window`)}
            <div class="callout" style="margin-top:12px">
              <strong>Physical capture (Phone A/B study).</strong> Point the camera at real labeled packages on a desk — not at a laptop screen. See the inspection guide: good lighting, full box in frame, label visible (ok / missing / damaged / rotated).
            </div>
          </div>
        </div>
        ${devAccordion}
      </div>
    </div>`;
}
function shadowDiagram() {
    return `
    <div class="shadow-diagram" aria-label="Live shadow architecture">
      <svg viewBox="0 0 920 120" width="920" height="120" xmlns="http://www.w3.org/2000/svg" role="img">
        <defs>
          <marker id="arrow" markerWidth="8" markerHeight="8" refX="6" refY="3" orient="auto">
            <path d="M0,0 L6,3 L0,6 Z" fill="#8C94A3"/>
          </marker>
        </defs>
        ${box(10, 35, 110, 50, "Camera stream", "#F7F6F3")}
        ${box(140, 35, 110, 50, "Frozen f₀", "#EEF4FB")}
        ${box(270, 35, 130, 50, "Tent candidate", "#FBF5E8")}
        ${box(420, 35, 120, 50, "Evidence Z", "#F0F1F3")}
        ${box(560, 35, 130, 50, "KGA certificate", "#E8F5F2")}
        ${box(710, 20, 90, 35, "Adapt", "#E8F5F2")}
        ${box(710, 65, 90, 35, "Freeze", "#EEF4FB")}
        ${box(820, 42, 90, 35, "Abstain", "#F0F1F3")}
        ${arrow(120, 60, 140, 60)}
        ${arrow(250, 60, 270, 60)}
        ${arrow(400, 60, 420, 60)}
        ${arrow(540, 60, 560, 60)}
        ${arrow(690, 50, 710, 38)}
        ${arrow(690, 60, 710, 60)}
        ${arrow(690, 70, 820, 60)}
        <text x="710" y="115" font-size="10" fill="#8C94A3">Official output remains frozen</text>
      </svg>
    </div>`;
}
function box(x, y, w, h, label, fill) {
    return `<rect x="${x}" y="${y}" width="${w}" height="${h}" rx="6" fill="${fill}" stroke="#E7E4DE"/>
    <text x="${x + w / 2}" y="${y + h / 2 + 4}" text-anchor="middle" font-size="11" fill="#1D2433" font-family="system-ui,sans-serif">${label}</text>`;
}
function arrow(x1, y1, x2, y2) {
    return `<line x1="${x1}" y1="${y1}" x2="${x2}" y2="${y2}" stroke="#8C94A3" stroke-width="1.5" marker-end="url(#arrow)"/>`;
}
export function safetyBoundary(safety) {
    if (!safety)
        return "";
    const metrics = (safety.metrics ?? [])
        .map((m) => `<div class="safety-metric">
        <div class="metric-label">${esc(m.label)}</div>
        <div class="val">${esc(m.value)}</div>
        <div class="cell-muted">${esc(m.meaning)}</div>
      </div>`)
        .join("");
    const p = safety.prose ?? {};
    return `
    <div class="page-section">
      <div class="section-label">Safety boundary</div>
      <h2 class="section-title">When the certificate refuses to commit</h2>
      <div class="panel">
        <div class="safety-grid">
          ${metrics}
          <dl class="safety-prose">
            <dt>False-adapt (FA_u)</dt><dd>${esc(p.false_adapt)}</dd>
            <dt>Why abstention exists</dt><dd>${esc(p.abstain)}</dd>
            <dt>Unknowable vs not helpful</dt><dd>${esc(p.unknowable)}</dd>
            <dt>Certificate scope</dt><dd>${esc(p.certificate_scope)}</dd>
          </dl>
        </div>
      </div>
    </div>`;
}
export function reproducePanel(rep) {
    if (!rep)
        return "";
    return `
    <div class="page-section">
      <div class="section-label">Reproduce</div>
      <h2 class="section-title">Artifact reproduction package</h2>
      <div class="panel panel-pad">
        <p class="cell-muted">Runtime estimate: ${esc(rep.runtime_estimate)}</p>
        ${cmdBlock("Primary (CPU + paper + dashboard)", rep.primary)}
        ${cmdBlock("Optional GPU TTA", rep.gpu)}
        ${cmdBlock("Theorem validators", rep.validators)}
        ${cmdBlock("Dashboard (TypeScript + snapshot)", rep.dashboard ?? "bash docs/research/kbound/scripts/build_dashboard.sh")}
        <div class="io-grid">
          <div><strong>Inputs</strong><ul class="plain-list">${(rep.inputs ?? []).map((i) => `<li><code>${esc(i)}</code></li>`).join("")}</ul></div>
          <div><strong>Outputs</strong><ul class="plain-list">${(rep.outputs ?? []).map((i) => `<li><code>${esc(i)}</code></li>`).join("")}</ul></div>
        </div>
      </div>
    </div>`;
}
export function artifactFooter(meta, prov) {
    return `
    <div class="page-section">
      <footer class="artifact-footer">
        <div><strong>Artifact provenance</strong></div>
        <div>Snapshot: <code>${esc(prov?.snapshot_path)}</code> · generated ${esc(meta?.generated_at)}</div>
        <div>Paper: <code>${esc(meta?.paper)}</code> (${esc(meta?.paper_pages)} pp) · build <code>${esc(meta?.build_id)}</code></div>
        <div>Commit: <code>${esc(prov?.commit ?? meta?.commit ?? "—")}</code> · manifest <code>${esc(prov?.manifest)}</code></div>
        <div>Edge protocol lock: <code>${esc(prov?.edge_protocol_lock)}</code></div>
        <div>${esc(prov?.local_clips_note)}</div>
      </footer>
    </div>`;
}
export function architectureSection() {
    return `
    <div class="page-section">
      <div class="section-label">System</div>
      <h2 class="section-title">K-Bound architecture</h2>
      <div class="panel panel-pad">
        <object data="figures/fig_architecture.svg" type="image/svg+xml" class="arch-svg" aria-label="K-Bound architecture diagram">
          <img src="figures/fig_architecture.svg" alt="K-Bound architecture" style="width:100%"/>
        </object>
      </div>
    </div>`;
}
export function experimentsPage(board, headline) {
    const b = board ?? {};
    return `
    <div class="page-section">
      <div class="section-label">Experiments</div>
      <h2 class="section-title">Controlled and natural-shift benchmarks</h2>
      <div class="evidence-group">
        <h3 class="evidence-group-title">Core suite</h3>
        <div class="panel"><div class="panel-body panel-scroll-x"><table class="data-table">
          <thead><tr><th>Experiment</th><th class="num">Freeze</th><th class="num">Adapt</th><th class="num">KGA</th><th class="num">Oracle</th><th>Status</th><th class="num">Regret</th></tr></thead>
          <tbody>${policyTableRows(headline)}</tbody>
        </table></div></div>
      </div>
      <div class="evidence-group">
        <h3 class="evidence-group-title">Helpful-dominated</h3>
        <div class="panel"><div class="panel-body panel-scroll-x"><table class="data-table">
          <thead><tr><th>Benchmark</th><th class="num">Freeze</th><th class="num">Adapt</th><th class="num">KGA</th><th class="num">Oracle</th><th>Status</th><th class="num">Regret</th></tr></thead>
          <tbody>${policyTableRows(b.helpful_dominated)}</tbody>
        </table></div></div>
      </div>
      <div class="evidence-group">
        <h3 class="evidence-group-title">Controlled wins</h3>
        <div class="panel"><div class="panel-body panel-scroll-x"><table class="data-table">
          <thead><tr><th>Benchmark</th><th class="num">Freeze</th><th class="num">Adapt</th><th class="num">KGA</th><th class="num">Oracle</th><th>Status</th><th class="num">Regret</th></tr></thead>
          <tbody>${policyTableRows(b.controlled_wins)}</tbody>
        </table></div></div>
      </div>
    </div>`;
}
//# sourceMappingURL=components.js.map