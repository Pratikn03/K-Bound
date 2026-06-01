const GATE_LABELS = {
  gate_a: "Gate A",
  gate_b: "Gate B",
  gate_c: "Gate C",
  gate_d: "Gate D",
  gate_e: "Gate E",
  gate_e_bounded_v3: "Gate E (bounded)",
  gate_e_positive_transfer: "Gate E (D13)",
  gate_f: "Gate F",
  gate_f_scientific: "Gate F (scientific)",
  gate_f_bounded_v3: "Gate F (bounded)",
  gate_f_positive_transfer: "Gate F (D13)",
};

const STORAGE_SETTINGS = "elara_dash_settings";
const STORAGE_PROFILE = "elara_dash_profile";

const PAGE_TITLES = {
  home: ["Home", "Overview & quick actions"],
  analytics: ["Analytics", "Trends & checklist progress"],
  gates: ["Gates", "Gate A–F — click for evidence"],
  confirmatory: ["Confirmatory", "M1 / M2 sealed statistics"],
  research: ["Research profile", "Your program & claim context"],
  codebase: ["Codebase", "Python catalog summary"],
  settings: ["Settings", "Display & data refresh"],
};

let chartInstances = {};
let liveTimer = null;
let lastSnapshot = null;
let researchProfile = {};
let dashSettings = { refresh_seconds: 5, repo_base: "/", live_enabled: false };

function loadSettings() {
  try {
    const raw = localStorage.getItem(STORAGE_SETTINGS);
    if (raw) dashSettings = { ...dashSettings, ...JSON.parse(raw) };
  } catch (_) {
    /* ignore */
  }
  const params = new URLSearchParams(window.location.search);
  if (params.get("repo")) dashSettings.repo_base = params.get("repo");
  if (params.get("live") === "1") dashSettings.live_enabled = true;
}

function saveSettings() {
  localStorage.setItem(STORAGE_SETTINGS, JSON.stringify(dashSettings));
}

async function loadProfileDefaults() {
  try {
    const res = await fetch("data/research_profile.default.json");
    if (res.ok) return res.json();
  } catch (_) {
    /* ignore */
  }
  return {
    researcher_name: "Researcher",
    role: "",
    program: "ELARA Scenario C",
    repo: "AutoML_Flagship_V8",
    focus: "",
    claim_status: "",
    advisor_notes: "",
  };
}

async function loadProfile() {
  const defaults = await loadProfileDefaults();
  try {
    const raw = localStorage.getItem(STORAGE_PROFILE);
    researchProfile = raw ? { ...defaults, ...JSON.parse(raw) } : defaults;
  } catch (_) {
    researchProfile = defaults;
  }
}

function saveProfile() {
  localStorage.setItem(STORAGE_PROFILE, JSON.stringify(researchProfile));
}

function repoBaseUrl() {
  const base = dashSettings.repo_base || "/";
  return base.replace(/\/$/, "") || "/";
}

function navigateTo(pageId) {
  document.querySelectorAll(".page").forEach((el) => {
    const on = el.id === `page-${pageId}`;
    el.classList.toggle("active", on);
    el.hidden = !on;
  });
  document.querySelectorAll(".nav-btn").forEach((btn) => {
    btn.classList.toggle("active", btn.dataset.page === pageId);
  });
  const titles = PAGE_TITLES[pageId] || ["Dashboard", ""];
  document.getElementById("page-title").textContent = titles[0];
  document.getElementById("page-subtitle").textContent = titles[1];
  location.hash = pageId;
}

function setupNavigation() {
  document.querySelectorAll(".nav-btn").forEach((btn) => {
    btn.addEventListener("click", () => navigateTo(btn.dataset.page));
  });
  document.querySelectorAll("[data-goto]").forEach((btn) => {
    btn.addEventListener("click", () => navigateTo(btn.dataset.goto));
  });
  const hash = location.hash.replace("#", "");
  if (hash && PAGE_TITLES[hash]) navigateTo(hash);
}

function evidenceUrl(relPath) {
  const base = repoBaseUrl();
  const clean = relPath.replace(/^\//, "");
  return `${base}${clean}`;
}

function fmtPct(n) {
  if (n == null || Number.isNaN(n)) return "—";
  return `${Number(n).toFixed(1)}%`;
}

function fmtDelta(n) {
  if (n == null || Number.isNaN(n)) return "—";
  const sign = n >= 0 ? "+" : "";
  return `${sign}${Number(n).toFixed(4)}`;
}

function fmtCi(ci) {
  if (!ci || ci.low == null || ci.high == null) return "—";
  return `[${Number(ci.low).toFixed(4)}, ${Number(ci.high).toFixed(4)}]`;
}

function badge(ok, passLabel = "PASS", failLabel = "FAIL") {
  const cls = ok ? "pass" : "fail";
  const text = ok ? passLabel : failLabel;
  return `<span class="badge ${cls}">${text}</span>`;
}

function destroyCharts() {
  Object.values(chartInstances).forEach((c) => c.destroy());
  chartInstances = {};
}

function showError(message) {
  let banner = document.querySelector(".error-banner");
  if (!banner) {
    banner = document.createElement("div");
    banner.className = "error-banner";
    document.body.prepend(banner);
  }
  banner.textContent = message;
}

function clearError() {
  const banner = document.querySelector(".error-banner");
  if (banner) banner.remove();
}

async function loadSnapshot() {
  const res = await fetch("data/snapshot.json", { cache: "no-store" });
  if (!res.ok) {
    throw new Error(
      `Failed to load data/snapshot.json (${res.status}). Run elara_research_snapshot first.`
    );
  }
  return res.json();
}

function renderDiff(data) {
  const section = document.getElementById("diff-section");
  const el = document.getElementById("diff-content");
  const diff = data.diff_vs_previous || {};
  if (!diff.has_previous) {
    section.hidden = true;
    return;
  }
  section.hidden = false;
  const parts = [];
  const pct = diff.checklist_percent_delta;
  if (pct != null && pct !== 0) {
    parts.push(
      `<p>Checklist: <strong class="${pct > 0 ? "up" : "down"}">${pct > 0 ? "+" : ""}${pct}%</strong></p>`
    );
  }
  const exec = diff.execution_percent_delta;
  if (exec != null && exec !== 0) {
    parts.push(
      `<p>Execution: <strong class="${exec > 0 ? "up" : "down"}">${exec > 0 ? "+" : ""}${exec}%</strong></p>`
    );
  }
  if ((diff.gate_changes || []).length) {
    parts.push("<ul class='diff-gates'>");
    diff.gate_changes.forEach((g) => {
      parts.push(
        `<li><code>${g.id}</code>: ${g.from ? "PASS" : "FAIL"} → ${g.to ? "PASS" : "FAIL"}</li>`
      );
    });
    parts.push("</ul>");
  }
  if ((diff.blockers_added || []).length) {
    parts.push(`<p>Blockers added: ${diff.blockers_added.map((b) => `<code>${b}</code>`).join(", ")}</p>`);
  }
  if ((diff.blockers_removed || []).length) {
    parts.push(
      `<p>Blockers cleared: ${diff.blockers_removed.map((b) => `<code>${b}</code>`).join(", ")}</p>`
    );
  }
  if ((diff.confirmatory_cell_changes || []).length) {
    parts.push("<table class='mini-table'><tr><th>Family</th><th>Δ change</th></tr>");
    diff.confirmatory_cell_changes.forEach((c) => {
      parts.push(
        `<tr><td>${c.family}</td><td class="num">${fmtDelta(c.previous_delta)} → ${fmtDelta(c.current_delta)}</td></tr>`
      );
    });
    parts.push("</table>");
  }
  if (!parts.length) {
    parts.push("<p>No metric changes since previous snapshot.</p>");
  }
  parts.push(`<p class="muted">Previous: ${diff.previous_at || "—"}</p>`);
  el.innerHTML = parts.join("");
}

function renderCharts(data) {
  if (typeof Chart === "undefined") return;
  destroyCharts();

  const timeline = data.history_timeline || [];
  const labels = timeline.map((p) => {
    const t = p.generated_at || "";
    return t.length >= 16 ? t.slice(5, 16).replace("T", " ") : t;
  });

  const ctxTrend = document.getElementById("chart-checklist-trend");
  if (ctxTrend && labels.length) {
    chartInstances.trend = new Chart(ctxTrend, {
      type: "line",
      data: {
        labels,
        datasets: [
          {
            label: "Checklist %",
            data: timeline.map((p) => p.percent_complete),
            borderColor: "#5b9fd4",
            tension: 0.2,
          },
          {
            label: "Execution %",
            data: timeline.map((p) => p.execution_percent),
            borderColor: "#3d9970",
            tension: 0.2,
          },
        ],
      },
      options: {
        responsive: true,
        scales: { y: { min: 0, max: 100, title: { display: true, text: "%" } } },
      },
    });
  }

  const cells = (data.confirmatory || {}).cells || [];
  const ctxDelta = document.getElementById("chart-delta-bars");
  if (ctxDelta && cells.length) {
    chartInstances.delta = new Chart(ctxDelta, {
      type: "bar",
      data: {
        labels: cells.map((c) => c.family || "?"),
        datasets: [
          {
            label: "Δ vs SAR",
            data: cells.map((c) => c.mean_delta_roc_auc),
            backgroundColor: cells.map((c) =>
              (c.mean_delta_roc_auc || 0) >= 0 ? "rgba(61,153,112,0.7)" : "rgba(196,78,82,0.7)"
            ),
          },
        ],
      },
      options: {
        responsive: true,
        scales: { y: { title: { display: true, text: "ROC-AUC Δ" } } },
      },
    });
  }

  const stages = (data.checklist || {}).items_by_stage || {};
  const stageNames = Object.keys(stages).sort();
  const ctxStages = document.getElementById("chart-stages");
  if (ctxStages && stageNames.length) {
    chartInstances.stages = new Chart(ctxStages, {
      type: "bar",
      data: {
        labels: stageNames,
        datasets: [
          {
            label: "Done",
            data: stageNames.map((s) => stages[s].done),
            backgroundColor: "rgba(91,159,212,0.8)",
          },
          {
            label: "Remaining",
            data: stageNames.map((s) => stages[s].total - stages[s].done),
            backgroundColor: "rgba(46,61,82,0.8)",
          },
        ],
      },
      options: {
        responsive: true,
        scales: { x: { stacked: true }, y: { stacked: true } },
      },
    });
  }
}

async function openEvidence(gateId, gate) {
  const dialog = document.getElementById("evidence-dialog");
  const title = document.getElementById("dialog-title");
  const pathsEl = document.getElementById("dialog-paths");
  const jsonEl = document.getElementById("dialog-json");

  title.textContent = `${GATE_LABELS[gateId] || gateId} — evidence`;
  const paths = gate.evidence_paths || [];
  pathsEl.innerHTML = paths.length
    ? paths
        .map(
          (p) =>
            `<a href="${evidenceUrl(p)}" target="_blank" rel="noopener"><code>${p}</code></a>`
        )
        .join("<br>")
    : "<p class='muted'>No evidence paths mapped.</p>";

  jsonEl.textContent = "Loading…";
  dialog.showModal();

  const primary = paths[0];
  if (!primary) {
    jsonEl.textContent = JSON.stringify(gate, null, 2);
    return;
  }
  try {
    const res = await fetch(evidenceUrl(primary), { cache: "no-store" });
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    const doc = await res.json();
    jsonEl.textContent = JSON.stringify(doc, null, 2);
  } catch (err) {
    jsonEl.textContent =
      `Could not fetch ${primary}.\n\n` +
      `Serve from repo root: ./research_dashboard/serve.sh\n` +
      `Error: ${err.message}\n\n` +
      `Gate record:\n${JSON.stringify(gate, null, 2)}`;
  }
}

function pillarStatusClass(status) {
  return (status || "unknown").toLowerCase().replace(/\s+/g, "-");
}

function renderPillarRows(pillars, compact) {
  if (!pillars?.length) {
    return '<p class="hint">No claim data in snapshot — rebuild with latest aggregator.</p>';
  }
  return pillars
    .map((p) => {
      const pass = p.pass ? " pass" : "";
      const cls = pillarStatusClass(p.status);
      const evidence = compact ? "" : `<span class="pillar-evidence">${p.required_evidence || ""}</span>`;
      return `<div class="pillar-row${pass}">
        <span class="pillar-id">${p.id}</span>
        <span>${p.name}${evidence}</span>
        <span class="pillar-status ${cls}">${p.status}</span>
      </div>`;
    })
    .join("");
}

function syncProfileClaim(data) {
  const claim = data.claim;
  if (!claim?.one_sentence_claim) return;
  researchProfile.claim_status = claim.one_sentence_claim;
  const field = document.getElementById("profile-form")?.elements?.claim_status;
  if (field) field.value = claim.one_sentence_claim;
}

function renderClaim(data) {
  const claim = data.claim || {};
  const section = document.getElementById("claim-section");
  if (!section) return;

  const tier = claim.readiness_tier || "tier_1_bounded";
  const tierBadge = document.getElementById("claim-tier-badge");
  if (tierBadge) {
    tierBadge.textContent = tier.replace(/_/g, " ");
    tierBadge.className = `tier-badge ${tier}`;
  }

  const sentence = document.getElementById("claim-one-sentence");
  if (sentence) sentence.textContent = claim.one_sentence_claim || "—";

  const pillarsCount = document.getElementById("claim-pillars-count");
  if (pillarsCount) {
    pillarsCount.textContent = `Pillars pass: ${claim.pillars_pass_count ?? 0}/${claim.pillars_total ?? 6}`;
  }

  const sci = document.getElementById("claim-scientific-ready");
  if (sci) {
    sci.textContent = claim.scientific_ready ? "Scientific: ready" : "Scientific: not ready";
    sci.className = claim.scientific_ready ? "claim-flag pass" : "claim-flag fail";
  }

  const stop = document.getElementById("claim-flagship-stop");
  const fsr = claim.flagship_stop_rule || {};
  if (stop) {
    if (!fsr.found) {
      stop.textContent = "Flagship stop rule: no sweep data";
      stop.className = "claim-flag warn";
    } else {
      const delta = Number(fsr.best_mean_delta ?? 0).toFixed(3);
      const min = Number(fsr.min_delta_vs_sar ?? 0.01).toFixed(2);
      stop.textContent = fsr.passed
        ? `Flagship stop: PASS (Δ ${delta} ≥ ${min})`
        : `Flagship stop: FAIL (Δ ${delta} < ${min})`;
      stop.className = fsr.passed ? "claim-flag pass" : "claim-flag fail";
    }
  }

  const preview = document.getElementById("home-pillars-preview");
  if (preview) preview.innerHTML = renderPillarRows(claim.pillars, true);

  section.classList.toggle("pass", !!claim.scientific_ready);
  section.classList.toggle("fail", !claim.scientific_ready);

  const contractHint = document.getElementById("claim-contract-hint");
  if (contractHint && claim.contract_path) {
    contractHint.innerHTML = `Derived from gates + checklist. Contract: <code>${claim.contract_path}</code>`;
  }

  const pillars = document.getElementById("pillars-table");
  if (pillars) pillars.innerHTML = renderPillarRows(claim.pillars, false);

  const tiersEl = document.getElementById("readiness-tiers");
  const checklists = claim.readiness_checklists || {};
  if (tiersEl) {
    tiersEl.innerHTML = Object.entries(checklists)
      .map(([key, block]) => {
        const items = (block.items || [])
          .map(
            (it) =>
              `<li class="${it.done ? "done" : "open"}"><span>${it.label}</span>${it.done ? " ✓" : ""}</li>`
          )
          .join("");
        return `<article class="tier-block">
          <h3>${key.replace(/_/g, " ")} <span class="tier-progress">${block.done ?? 0}/${block.total ?? 0}</span></h3>
          <ul>${items}</ul>
        </article>`;
      })
      .join("");
  }

  syncProfileClaim(data);
}

function renderVerdict(data) {
  const cl = data.checklist || {};
  const el = document.getElementById("verdict-text");
  const section = document.getElementById("verdict-section");
  el.textContent = cl.verdict || "No verdict in snapshot.";
  const claim = data.claim || {};
  const ready = claim.scientific_ready ?? (cl.scientific_scenario_c_ready && cl.m2_transfer_confirmed);
  section.classList.toggle("pass", !!ready);
  section.classList.toggle("fail", !ready);
}

function renderChecklist(data) {
  const cl = data.checklist || {};
  document.getElementById("checklist-percent").textContent = fmtPct(cl.percent_complete);
  document.getElementById("checklist-count").textContent = `${cl.done ?? "—"}/${cl.total ?? "—"}`;
  document.getElementById("execution-percent").textContent = fmtPct(cl.execution_percent);
  document.getElementById("checklist-bar").style.width = `${cl.percent_complete ?? 0}%`;

  const stageEl = document.getElementById("stage-breakdown");
  stageEl.innerHTML = "";
  const stages = cl.items_by_stage || {};
  Object.entries(stages)
    .sort(([a], [b]) => a.localeCompare(b))
    .forEach(([stage, info]) => {
      const chip = document.createElement("span");
      chip.className = "stage-chip";
      chip.innerHTML = `<strong>${stage}</strong> ${info.done}/${info.total} (${fmtPct(info.percent)})`;
      stageEl.appendChild(chip);
    });
}

function renderProtocol(data) {
  const p = data.protocol || {};
  const dl = document.getElementById("protocol-kv");
  const rows = [
    ["Path", p.path || "—"],
    ["Version", p.version || "—"],
    ["Status", p.status || "—"],
    ["Ratified", p.ratified || "—"],
    ["Confirmatory blocked", p.confirmatory_blocked ? "yes (stop rule)" : "no"],
  ];
  dl.innerHTML = rows.map(([k, v]) => `<dt>${k}</dt><dd>${v}</dd>`).join("");
}

function renderGates(data) {
  const grid = document.getElementById("gate-grid");
  grid.innerHTML = "";
  const gates = data.gates || {};
  Object.entries(gates).forEach(([id, gate]) => {
    const card = document.createElement("button");
    card.type = "button";
    card.className = `gate-card ${gate.done ? "pass" : "fail"}`;
    card.innerHTML = `
      <div class="gate-id">${GATE_LABELS[id] || id}</div>
      <div class="gate-status">${gate.done ? "PASS" : "FAIL"}</div>
      <p class="gate-desc">${gate.description || ""}</p>
      <p class="gate-evidence-hint">View evidence →</p>
    `;
    card.addEventListener("click", () => openEvidence(id, gate));
    grid.appendChild(card);
  });
}

function renderConfirmatory(data) {
  const c = data.confirmatory || {};
  const flagsEl = document.getElementById("confirmatory-flags");
  const flags = [
    ["gate_d_m1", c.gate_d_m1],
    ["gate_d_m2_external", c.gate_d_m2_external],
    ["gate_e_strict", c.gate_e_m2_transfer_confirmed_strict ?? c.gate_e_m2_transfer_confirmed],
    ["gate_e_bounded_v3", c.gate_e_m2_bounded_v3_pass],
    ["gate_e_positive_transfer", c.gate_e_positive_transfer_confirmed],
    ["gate_e_positive_official", c.gate_e_positive_transfer_official],
    ["t5_m1", c.t5_m1],
    ["t5_m2_ran", c.t5_m2_ran],
    ["gate_f_strict", c.gate_f_scenario_c_scientific],
    ["gate_f_bounded_v3", c.gate_f_bounded_v3],
    ["gate_f_positive_transfer", c.gate_f_positive_transfer_track],
  ];
  flagsEl.innerHTML = flags
    .map(
      ([name, val]) =>
        `<span class="flag ${val ? "true" : "false"}">${name}: ${val ? "true" : "false"}</span>`
    )
    .join("");

  const tbody = document.querySelector("#cells-table tbody");
  tbody.innerHTML = "";
  (c.cells || []).forEach((cell) => {
    const tr = document.createElement("tr");
    tr.innerHTML = `
      <td>${cell.family || "—"}</td>
      <td>${cell.benchmark || "—"}</td>
      <td class="num">${fmtDelta(cell.mean_delta_roc_auc)}</td>
      <td class="num">${fmtCi(cell.bootstrap_95_ci)}</td>
      <td>${badge(cell.gate_d_pass)}</td>
      <td>${badge(cell.gate_e_pass)}</td>
      <td>${badge(cell.cell_valid, "OK", "INVALID")}</td>
    `;
    tbody.appendChild(tr);
  });
}

function blockersHtml(blockers) {
  if (!blockers.length) return "<li>No remaining blockers.</li>";
  return blockers
    .map((b) => `<li><code>${b.id || "unknown"}</code> — ${b.description || ""}</li>`)
    .join("");
}

function renderBlockers(data) {
  const blockers = data.blockers || [];
  const html = blockersHtml(blockers);
  const list = document.getElementById("blocker-list");
  const homeList = document.getElementById("home-blocker-list");
  if (list) list.innerHTML = html;
  if (homeList) homeList.innerHTML = html;
}

function renderHome(data) {
  const cl = data.checklist || {};
  const gates = data.gates || {};
  const blockers = data.blockers || [];

  document.getElementById("home-checklist-pct").textContent = fmtPct(cl.percent_complete);
  document.getElementById("home-blockers").textContent = String(blockers.length);

  let pass = 0;
  let total = 0;
  const mini = document.getElementById("home-mini-gates");
  mini.innerHTML = "";
  Object.entries(gates).forEach(([id, gate]) => {
    total += 1;
    if (gate.done) pass += 1;
    const chip = document.createElement("button");
    chip.type = "button";
    chip.className = `mini-gate ${gate.done ? "pass" : "fail"}`;
    chip.textContent = `${GATE_LABELS[id] || id}: ${gate.done ? "PASS" : "FAIL"}`;
    chip.addEventListener("click", () => {
      navigateTo("gates");
      setTimeout(() => openEvidence(id, gate), 100);
    });
    mini.appendChild(chip);
  });
  document.getElementById("home-gates-pass").textContent = `${pass}/${total}`;

  const p = researchProfile;
  const homeKv = document.getElementById("home-profile-kv");
  homeKv.innerHTML = [
    ["Researcher", p.researcher_name],
    ["Program", p.program],
    ["Focus", p.focus],
    ["Claim", p.claim_status || data.claim?.one_sentence_claim],
  ]
    .map(([k, v]) => `<dt>${k}</dt><dd>${v || "—"}</dd>`)
    .join("");

  const spName = document.querySelector("#sidebar-profile .sidebar-profile-name");
  const spProg = document.querySelector("#sidebar-profile .sidebar-profile-program");
  if (spName) spName.textContent = p.researcher_name || "—";
  if (spProg) spProg.textContent = p.program || "—";
}

function renderResearchPage(data) {
  const form = document.getElementById("profile-form");
  Object.entries(researchProfile).forEach(([key, val]) => {
    const field = form.elements[key];
    if (field) field.value = val ?? "";
  });

  const cl = data.checklist || {};
  const c = data.confirmatory || {};
  const claim = data.claim || {};
  const live = document.getElementById("research-live-kv");
  live.innerHTML = [
    ["Checklist", `${cl.done}/${cl.total} (${fmtPct(cl.percent_complete)})`],
    ["Readiness tier", (claim.readiness_tier || "—").replace(/_/g, " ")],
    ["Scientific ready", claim.scientific_ready ? "yes" : "no"],
    ["Execution ready", claim.execution_ready ? "yes" : "no"],
    ["M2 transfer", c.gate_e_m2_transfer_confirmed ? "confirmed" : "not confirmed"],
    ["D13 positive transfer", c.gate_e_positive_transfer_confirmed ? "confirmed" : (c.gate_e_positive_transfer_status || "pending")],
    ["Central claim ratified", claim.central_claim_ratified ? "yes" : "no"],
    ["Repo root", data.repo_root || "—"],
    ["Snapshot", data.generated_at || "—"],
  ]
    .map(([k, v]) => `<dt>${k}</dt><dd>${v}</dd>`)
    .join("");

  renderClaim(data);
}

function bindProfileForm() {
  document.getElementById("profile-form").addEventListener("submit", (e) => {
    e.preventDefault();
    const fd = new FormData(e.target);
    researchProfile = Object.fromEntries(fd.entries());
    saveProfile();
    if (lastSnapshot) {
      renderHome(lastSnapshot);
      renderResearchPage(lastSnapshot);
    }
    e.target.querySelector(".btn-primary").textContent = "Saved ✓";
    setTimeout(() => {
      e.target.querySelector(".btn-primary").textContent = "Save profile";
    }, 1500);
  });

  document.getElementById("btn-profile-reset").addEventListener("click", async () => {
    researchProfile = await loadProfileDefaults();
    saveProfile();
    if (lastSnapshot) {
      renderHome(lastSnapshot);
      renderResearchPage(lastSnapshot);
    }
  });
}

function bindSettingsForm() {
  const form = document.getElementById("settings-form");
  form.elements.refresh_seconds.value = dashSettings.refresh_seconds;
  form.elements.repo_base.value = dashSettings.repo_base;
  form.elements.live_enabled.checked = !!dashSettings.live_enabled;

  form.addEventListener("submit", (e) => {
    e.preventDefault();
    const fd = new FormData(form);
    dashSettings.refresh_seconds = Math.max(3, Number(fd.get("refresh_seconds")) || 5);
    dashSettings.repo_base = fd.get("repo_base") || "/";
    dashSettings.live_enabled = fd.get("live_enabled") === "on";
    saveSettings();
    document.getElementById("live-refresh").checked = dashSettings.live_enabled;
    setupLiveRefresh();
  });

  document.getElementById("btn-export-profile").addEventListener("click", () => {
    const blob = new Blob([JSON.stringify(researchProfile, null, 2)], { type: "application/json" });
    const a = document.createElement("a");
    a.href = URL.createObjectURL(blob);
    a.download = "elara_research_profile.json";
    a.click();
  });

  document.getElementById("import-profile").addEventListener("change", async (e) => {
    const file = e.target.files?.[0];
    if (!file) return;
    researchProfile = { ...researchProfile, ...JSON.parse(await file.text()) };
    saveProfile();
    if (lastSnapshot) {
      renderHome(lastSnapshot);
      renderResearchPage(lastSnapshot);
    }
    const formProfile = document.getElementById("profile-form");
    Object.entries(researchProfile).forEach(([key, val]) => {
      if (formProfile.elements[key]) formProfile.elements[key].value = val ?? "";
    });
  });
}

function renderCatalog(data) {
  const cat = data.python_catalog || {};
  document.getElementById("py-total").textContent = cat.total ?? "—";
  document.getElementById("py-used").textContent = cat.used_count ?? "—";
  document.getElementById("py-unused").textContent = cat.unused_count ?? "—";

  const catEl = document.getElementById("py-categories");
  const byCat = cat.by_category || {};
  catEl.innerHTML = Object.entries(byCat)
    .sort(([a], [b]) => a.localeCompare(b))
    .map(
      ([name, info]) =>
        `<div class="category-row"><span>${name}</span><span>${info.used}/${info.total}</span></div>`
    )
    .join("");
}

function renderMeta(data) {
  document.getElementById("last-updated").textContent = `Updated: ${data.generated_at || "unknown"}`;
  document.getElementById("repo-root").textContent = data.repo_root || "";
}

function renderAll(data) {
  lastSnapshot = data;
  clearError();
  renderMeta(data);
  renderVerdict(data);
  renderClaim(data);
  renderDiff(data);
  renderCharts(data);
  renderChecklist(data);
  renderProtocol(data);
  renderGates(data);
  renderConfirmatory(data);
  renderBlockers(data);
  renderCatalog(data);
  renderHome(data);
  renderResearchPage(data);
}

async function refresh() {
  try {
    const data = await loadSnapshot();
    renderAll(data);
  } catch (err) {
    showError(err.message);
    console.error(err);
  }
}

function setupLiveRefresh() {
  const cb = document.getElementById("live-refresh");
  cb.checked = !!dashSettings.live_enabled;

  const toggle = () => {
    dashSettings.live_enabled = cb.checked;
    saveSettings();
    if (liveTimer) {
      clearInterval(liveTimer);
      liveTimer = null;
    }
    if (cb.checked) {
      const ms = (Number(dashSettings.refresh_seconds) || 5) * 1000;
      liveTimer = setInterval(refresh, ms);
    }
  };
  cb.onchange = toggle;
  toggle();
}

async function init() {
  loadSettings();
  await loadProfile();
  setupNavigation();
  bindProfileForm();
  bindSettingsForm();
  document.getElementById("btn-reload").addEventListener("click", refresh);
  setupLiveRefresh();
  await refresh();
}

init();
