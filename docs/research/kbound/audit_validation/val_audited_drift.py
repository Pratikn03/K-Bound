#!/usr/bin/env python3
"""Validation for the auditable-drift-budget appendix (Theorems Aud-A..Aud-H).

Single-paper artifact: numbers cited in paper/sections/auditable_budgets.tex and the
short-paper appendix trace to this script's JSON output. Pure numpy + matplotlib,
seeded, CPU. No fabrication: ground truth on a dense grid; finite-sample claims tested
by actual sampling.

World (bounded 1-d representation, support [-4,4], mu(D)=1 WLOG):
  s(x)  = 0.5 + 0.38*tanh(x)                 (observable score; M_T = E_T[s]-1/2)
  u(x)  = eta_a(x)-s(x) = -0.08 - 0.04*tanh(x)   (latent drift; Lip(u)=L*=0.04)
  gamma_T = E_T[u];  sign(Delta) = sign(M_T + gamma_T)   (Lemma 1)
  domains = truncated N(theta,1) on [-4,4]; calibration theta=0, deployment theta swept.

Blocks:
  A. impossibility panel (exact, Thm Aud-A)
  B. label-budget audit (Thm Aud-B): UCB width, coverage, audited-frontier FA vs n
  C. coverage of beta_hat over random Lipschitz worlds + 3L robustness +
     directed 30x adversary (assumption is load-bearing)
  D. theta sweep (Thms Aud-C/D/E): audited + fully-empirical rules vs beta=0 plug-in
  F. Aud-F unknown-direction index worlds: net MSW1 certificate coverage vs
     gradient-ascent heuristic; failure under non-index high-d Lip drift
Run: STAGE=0 python3 val_audited_drift.py   (stages 1/2/3 for slow machines)
"""
import json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

rng = np.random.default_rng(20260714)
HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "results_audited_drift.json")
B_SUP = 4.0
GRID = np.linspace(-B_SUP, B_SUP, 4001)

def trunc_norm_pdf(x, theta):
    z = np.exp(-0.5 * (x - theta) ** 2)
    w = np.trapz(np.exp(-0.5 * (GRID - theta) ** 2), GRID)
    return z / w

def expect(f, theta):
    return float(np.trapz(f(GRID) * trunc_norm_pdf(GRID, theta), GRID))

def cdf_grid(theta):
    p = trunc_norm_pdf(GRID, theta)
    c = np.cumsum((p[1:] + p[:-1]) * 0.5 * np.diff(GRID))
    return np.concatenate([[0.0], c])

def w1_true(t1, t2):
    return float(np.trapz(np.abs(cdf_grid(t1) - cdf_grid(t2)), GRID))

def sample(theta, m):
    xs = rng.normal(theta, 1.0, size=int(m * 1.6))
    xs = xs[(xs >= -B_SUP) & (xs <= B_SUP)]
    while len(xs) < m:
        extra = rng.normal(theta, 1.0, size=m)
        xs = np.concatenate([xs, extra[(extra >= -B_SUP) & (extra <= B_SUP)]])
    return xs[:m]

def w1_hat(xa, xb):
    n = min(len(xa), len(xb), 1200)
    qa = np.quantile(xa, np.linspace(0, 1, n))
    qb = np.quantile(xb, np.linspace(0, 1, n))
    return float(np.mean(np.abs(qa - qb)))

s_fn = lambda x: 0.5 + 0.38 * np.tanh(x)
u_fn = lambda x: -0.08 - 0.04 * np.tanh(x)
L_TRUE = 0.04
STAGE = int(os.environ.get("STAGE", "0"))   # 0=all, 1=cal+D, 2=C, 3=B+figs
STATE = "/tmp/aud_state.json"
if STAGE > 1 and os.path.exists(STATE):
    res = json.load(open(STATE))
else:
    res = {"seed": 20260714, "L_true": L_TRUE, "support": [-B_SUP, B_SUP]}

# ---------------- A. impossibility (exact, Thm Aud-A) ----------------
Ms = np.linspace(-0.45, 0.45, 19)
res["A_impossibility"] = {
    "M_grid": Ms.tolist(),
    "gamma_lo": (-(Ms + 0.5)).tolist(),
    "gamma_hi": (0.5 - Ms).tolist(),
    "max_abs_gamma": (0.5 + np.abs(Ms)).tolist(),
    "note": "any label-free UCB valid on the unrestricted class must exceed 1/2+|M| >= |M|; frontier abstains",
}

# ---------------- shared: calibration estimates ----------------
DELTA = 0.10
N_CAL_LAB, M_CAL_UNLAB, M_DEP_UNLAB = 1500, 4000, 4000
theta_cal = 0.0
gamma_cal_true = expect(u_fn, theta_cal)
x_cal_lab = sample(theta_cal, N_CAL_LAB)
y_is_fa = rng.random(N_CAL_LAB) < (s_fn(x_cal_lab) + u_fn(x_cal_lab))
u_hat_i = y_is_fa.astype(float) - s_fn(x_cal_lab)
gamma_cal_hat = float(np.mean(u_hat_i))
t_hoeff = float(np.sqrt(2 * np.log(4 / DELTA) / N_CAL_LAB))
x_cal_unlab = sample(theta_cal, M_CAL_UNLAB)
eps_w = 2 * B_SUP * (np.sqrt(np.log(8 / DELTA) / (2 * M_CAL_UNLAB)) +
                     np.sqrt(np.log(8 / DELTA) / (2 * M_DEP_UNLAB)))
res["calibration"] = {"gamma_cal_true": gamma_cal_true, "gamma_cal_hat": gamma_cal_hat,
                      "t_hoeffding": t_hoeff, "eps_W": float(eps_w),
                      "n_cal_labeled": N_CAL_LAB, "m_unlabeled": M_CAL_UNLAB}

def beta_hat_for(theta_dep, L_declared):
    x_dep = sample(theta_dep, M_DEP_UNLAB)
    w = w1_hat(x_cal_unlab, x_dep)
    return abs(gamma_cal_hat) + t_hoeff + L_declared * (w + eps_w), w

# ---------------- D. theta sweep (Thms Aud-C/D/E) ----------------
thetas = np.linspace(0.0, 2.5, 21)
sweep = {"theta": thetas.tolist(), "M": [], "gamma": [], "delta_sign": [],
         "beta_hat": [], "W1_true": [], "trivial_bound": [],
         "audited_action": [], "plugin_action": []}
for th in thetas:
    M = expect(s_fn, th) - 0.5
    g = expect(u_fn, th)
    bh, _ = beta_hat_for(th, L_TRUE)
    sweep["M"].append(M); sweep["gamma"].append(g)
    sweep["delta_sign"].append(int(np.sign(M + g)))
    sweep["beta_hat"].append(bh); sweep["W1_true"].append(w1_true(theta_cal, th))
    sweep["trivial_bound"].append(0.5 + abs(M))
    sweep["audited_action"].append("adapt" if M > bh else ("freeze" if M < -bh else "abstain"))
    sweep["plugin_action"].append("adapt" if M > 0 else ("freeze" if M < 0 else "abstain"))
    # Thm Aud-E: fully empirical rule (estimated margin from the unlabeled batch)
    DPRIME = 0.05
    x_m = sample(th, M_DEP_UNLAB)
    M_hat = float(np.mean(s_fn(x_m))) - 0.5
    t_M = float(np.sqrt(np.log(2 / DPRIME) / (2 * M_DEP_UNLAB)))
    sweep.setdefault("M_hat", []).append(M_hat)
    sweep.setdefault("t_M", []).append(t_M)
    sweep.setdefault("empirical_action", []).append(
        "adapt" if M_hat - t_M > bh else ("freeze" if M_hat + t_M < -bh else "abstain"))

def fa(actions, Ms_, gs_):
    bad = 0; committed = 0
    for a, M, g in zip(actions, Ms_, gs_):
        if a == "abstain":
            continue
        committed += 1
        d = M + g
        if (a == "adapt" and d <= 0) or (a == "freeze" and d >= 0):
            bad += 1
    return bad, committed

fa_emp, com_emp = fa(sweep["empirical_action"], sweep["M"], sweep["gamma"])
sweep["false_commits_empirical"] = fa_emp; sweep["commits_empirical"] = com_emp
fa_aud, com_aud = fa(sweep["audited_action"], sweep["M"], sweep["gamma"])
fa_plg, com_plg = fa(sweep["plugin_action"], sweep["M"], sweep["gamma"])
sweep["false_commits_audited"] = fa_aud; sweep["commits_audited"] = com_aud
sweep["false_commits_plugin"] = fa_plg; sweep["commits_plugin"] = com_plg
res["D_sweep"] = sweep

if STAGE == 1:
    json.dump(res, open(STATE, "w")); print("stage1 done"); raise SystemExit

# ---------------- C. coverage over random Lipschitz worlds ----------------
def random_lip_u(L, k=6):
    freqs = rng.uniform(0.2, 1.2, k); phases = rng.uniform(0, 2 * np.pi, k)
    amps = rng.normal(0, 1, k)
    scale = np.sum(np.abs(amps) * freqs)
    amps = amps * (L / scale)
    off = rng.uniform(-0.06, 0.06)
    def u(x):
        return off + sum(a * np.sin(f * x + p) for a, f, p in zip(amps, freqs, phases))
    return u

N_TRIALS = 120
cov_ok, cov_viol = 0, 0
for tr in range(N_TRIALS):
    u_r = random_lip_u(L_TRUE)
    th = rng.uniform(0.2, 2.5)
    g_dep_r = expect(u_r, th)
    xs = sample(theta_cal, N_CAL_LAB)
    eta = np.clip(s_fn(xs) + u_r(xs), 0, 1)
    uh = (rng.random(N_CAL_LAB) < eta).astype(float) - s_fn(xs)
    g_cal_hat_r = float(np.mean(uh))
    x_dep = sample(th, M_DEP_UNLAB // 4)
    eps_w_r = 2 * B_SUP * (np.sqrt(np.log(8 / DELTA) / (2 * M_CAL_UNLAB)) +
                           np.sqrt(np.log(8 / DELTA) / (2 * (M_DEP_UNLAB // 4))))
    bh = abs(g_cal_hat_r) + t_hoeff + L_TRUE * (w1_hat(x_cal_unlab, x_dep) + eps_w_r)
    cov_ok += int(abs(g_dep_r) <= bh)
    # robustness: true Lipschitz 3x declared (random worlds)
    u_a = random_lip_u(3 * L_TRUE)
    g_dep_a = expect(u_a, th)
    g_cal_a = expect(u_a, theta_cal)
    bh_a = abs(g_cal_a) + t_hoeff + L_TRUE * (w1_hat(x_cal_unlab, x_dep) + eps_w_r)
    cov_viol += int(abs(g_dep_a) <= bh_a)
res["C_coverage"] = {"n_trials": N_TRIALS, "target": 1 - DELTA,
                     "coverage_declaredL": cov_ok / N_TRIALS,
                     "coverage_violatedL_3x": cov_viol / N_TRIALS}

# directed adversary: near-zero calibration drift, steep deployment drift (Lip = 30x declared)
c0 = expect(lambda x: np.tanh(4 * (x - 2)), 0.0)
u_adv = lambda x: 0.3 * (np.tanh(4 * (x - 2)) - c0)
th_dep = 2.5
g_cal_adv = expect(u_adv, 0.0); g_dep_adv = expect(u_adv, th_dep)
W1_adv = w1_true(0.0, th_dep)
bh_adv = abs(res["calibration"]["gamma_cal_hat"]) + t_hoeff + L_TRUE * (W1_adv + eps_w)
res["C_adversarial_step"] = {"true_lipschitz": 1.2, "declared_L": L_TRUE, "lip_ratio": 30,
                             "gamma_cal": g_cal_adv, "gamma_dep": g_dep_adv,
                             "beta_hat": bh_adv, "covered": bool(abs(g_dep_adv) <= bh_adv),
                             "note": "directed adversary breaks validity once (Lip) fails, as Thm Aud-A requires"}

if STAGE == 2:
    json.dump(res, open(STATE, "w")); print("stage2 done"); raise SystemExit

# ---------------- B. label-budget audit (Thm Aud-B) ----------------
theta_b = 0.6
g_true_b = expect(u_fn, theta_b); M_b = expect(s_fn, theta_b) - 0.5
budget = {"theta": theta_b, "gamma_true": g_true_b, "M": M_b, "n": [], "ucb_width": [],
          "coverage": [], "fa_rate_audited": []}
for n in [25, 50, 100, 200, 400, 800, 1600, 3200]:
    widths, covers, fas = [], [], 0
    reps = 150
    for _ in range(reps):
        xs = sample(theta_b, n)
        eta = np.clip(s_fn(xs) + u_fn(xs), 0, 1)
        uh = (rng.random(n) < eta).astype(float) - s_fn(xs)
        t = np.sqrt(2 * np.log(2 / DELTA) / n)
        bh = abs(float(np.mean(uh))) + t
        widths.append(t); covers.append(abs(g_true_b) <= bh)
        act = "adapt" if M_b > bh else ("freeze" if M_b < -bh else "abstain")
        if act != "abstain" and ((act == "adapt" and M_b + g_true_b <= 0) or
                                 (act == "freeze" and M_b + g_true_b >= 0)):
            fas += 1
    budget["n"].append(n); budget["ucb_width"].append(float(np.mean(widths)))
    budget["coverage"].append(float(np.mean(covers))); budget["fa_rate_audited"].append(fas / reps)
res["B_label_budget"] = budget

# ---------------- F. Aud-F: unknown-direction index drift (net MSW1) ----------------
# Multi-d features psi ~ N(0,I_d) truncated to ||psi||_inf <= B_F.
# Latent drift u = L_F * tanh(theta_* · psi); gamma = E[u].
# Certificate uses a finite epsilon-net on S^{d-1} (admissible in Thm Aud-F);
# gradient ascent on the sphere is recorded as a heuristic (not a certificate).
D_F, B_F, L_F = 4, 1.5, 0.10
N_LAB_F, N_CAL_F, N_DEP_F = 1000, 2000, 2000
N_TRIALS_F = 40 if STAGE == 0 else 20
EPS_NET = 0.35  # net inflation 2B*eps; denser net => smaller eps, more dirs


def _sample_ball(n, d=D_F, b=B_F):
    x = rng.normal(0.0, 0.7, size=(n, d)).astype(np.float64)
    return np.clip(x, -b, b)


def _w1_1d(a, b):
    a = np.asarray(a, dtype=np.float64).ravel()
    b = np.asarray(b, dtype=np.float64).ravel()
    n = min(len(a), len(b), 800)
    qa = np.quantile(a, np.linspace(0, 1, n))
    qb = np.quantile(b, np.linspace(0, 1, n))
    return float(np.mean(np.abs(qa - qb)))


def _sphere_net(d, eps, rng_local):
    # Fibonacci-ish + random fill: admissible over-approximation of an eps-net.
    target = int(min(250, max(48, (2.5 / eps) ** (d - 1))))
    dirs = []
    for _ in range(target * 8):
        if len(dirs) >= target:
            break
        v = rng_local.normal(0.0, 1.0, size=d).astype(np.float64)
        nrm = float(np.linalg.norm(v))
        if nrm < 1e-12:
            continue
        v = v / nrm
        if not dirs or min(float(np.linalg.norm(v - u)) for u in dirs) >= 0.4 * eps:
            dirs.append(v)
    while len(dirs) < target:
        v = rng_local.normal(0.0, 1.0, size=d).astype(np.float64)
        v = v / (float(np.linalg.norm(v)) + 1e-12)
        dirs.append(v)
    return np.asarray(dirs[:target], dtype=np.float64)


def _proj(x, th):
    return np.dot(x, th)


def _ms_net(x_cal, x_dep, net):
    best = 0.0
    for th in net:
        best = max(best, _w1_1d(_proj(x_cal, th), _proj(x_dep, th)))
    return float(best)


def _ms_grad_ascent(x_cal, x_dep, steps=30, restarts=3):
    """Heuristic local ascent of sliced W1 on the sphere (NOT a certificate)."""
    best = 0.0
    for _ in range(restarts):
        th = rng.normal(0.0, 1.0, size=D_F).astype(np.float64)
        th = th / (float(np.linalg.norm(th)) + 1e-12)
        for _s in range(steps):
            g = np.zeros(D_F, dtype=np.float64)
            base = _w1_1d(_proj(x_cal, th), _proj(x_dep, th))
            for j in range(D_F):
                e = np.zeros(D_F, dtype=np.float64); e[j] = 1e-3
                th2 = th + e; th2 = th2 / (float(np.linalg.norm(th2)) + 1e-12)
                g[j] = (_w1_1d(_proj(x_cal, th2), _proj(x_dep, th2)) - base) / 1e-3
            th = th + 0.35 * g
            th = th / (float(np.linalg.norm(th)) + 1e-12)
            best = max(best, _w1_1d(_proj(x_cal, th), _proj(x_dep, th)))
    return float(best)


def _eps_vc(n, d, delta):
    return float(np.sqrt(2 * ((d + 1) * np.log(2 * n) + np.log(8 / delta)) / n))


def _beta_F(gamma_cal_hat_f, t_f, ms, eps_vc):
    return float(abs(gamma_cal_hat_f) + t_f
                 + L_F * (ms + 2 * B_F * EPS_NET + 2 * B_F * eps_vc))


net_F = _sphere_net(D_F, EPS_NET, rng)
cov_f, cov_f_ascent, cov_f_fail = 0, 0, 0
beta_f_vals, ms_net_vals, ms_asc_vals = [], [], []
gamma_abs_vals, beta_null_vals = [], []
eps_vc = _eps_vc(N_CAL_F, D_F, DELTA / 4) + _eps_vc(N_DEP_F, D_F, DELTA / 4)
t_f = float(np.sqrt(2 * np.log(8 / DELTA) / N_LAB_F))
for _tr in range(N_TRIALS_F):
    theta_star = rng.normal(0.0, 1.0, size=D_F).astype(np.float64)
    theta_star = theta_star / (float(np.linalg.norm(theta_star)) + 1e-12)
    shift = rng.normal(0.0, 1.0, size=D_F).astype(np.float64)
    shift = 0.7 * shift / (float(np.linalg.norm(shift)) + 1e-12)
    x_cal = _sample_ball(N_CAL_F)
    x_dep = np.clip(rng.normal(0.0, 0.7, size=(N_DEP_F, D_F)) + shift, -B_F, B_F).astype(np.float64)
    x_lab = _sample_ball(N_LAB_F)
    u_lab = L_F * np.tanh(_proj(x_lab, theta_star))
    gamma_cal_hat_f = float(np.mean(u_lab))
    gamma_dep_true = float(np.mean(L_F * np.tanh(_proj(x_dep, theta_star))))
    gamma_abs_vals.append(abs(gamma_dep_true))
    ms_n = _ms_net(x_cal, x_dep, net_F)
    ms_a = _ms_grad_ascent(x_cal, x_dep)
    ms_net_vals.append(ms_n); ms_asc_vals.append(ms_a)
    beta_f = _beta_F(gamma_cal_hat_f, t_f, ms_n, eps_vc)
    beta_f_vals.append(beta_f)
    cov_f += int(abs(gamma_dep_true) <= beta_f)
    beta_a = _beta_F(gamma_cal_hat_f, t_f, ms_a, eps_vc)
    cov_f_ascent += int(abs(gamma_dep_true) <= beta_a)
    # Non-index misspecification: drift depends on all coordinates jointly
    u_fail_dep = float(np.mean(L_F * np.tanh(x_dep.sum(1) / np.sqrt(D_F))))
    cov_f_fail += int(abs(u_fail_dep) <= beta_f)
    # Null usefulness: identical cal/dep laws (shift=0), report beta size
    x_null = _sample_ball(N_DEP_F)
    ms_null = _ms_net(x_cal, x_null, net_F)
    beta_null_vals.append(_beta_F(gamma_cal_hat_f, t_f, ms_null, eps_vc))

res["F_index_msw"] = {
    "d": D_F, "L": L_F, "B": B_F, "eps_net": EPS_NET, "n_net": int(len(net_F)),
    "n_trials": N_TRIALS_F, "target": 1 - DELTA,
    "coverage_net_certificate": cov_f / N_TRIALS_F,
    "coverage_grad_ascent_heuristic": cov_f_ascent / N_TRIALS_F,
    "coverage_nonindex_failure_world": cov_f_fail / N_TRIALS_F,
    "mean_abs_gamma": float(np.mean(gamma_abs_vals)),
    "mean_beta_F": float(np.mean(beta_f_vals)),
    "mean_beta_null_identical_laws": float(np.mean(beta_null_vals)),
    "trivial_vacuity_floor": 0.5,
    "useful_vs_vacuity": bool(float(np.mean(beta_null_vals)) < 0.5),
    "mean_MS_net": float(np.mean(ms_net_vals)),
    "mean_MS_ascent": float(np.mean(ms_asc_vals)),
    "eps_vc_sum": float(eps_vc),
    "note": (
        "Net certificate is the Aud-F-admissible upper bound; gradient ascent is a "
        "heuristic (Conjecture aud-computational). Null beta << 1/2 demonstrates "
        "usefulness (Def. useful-audit) vs Aud-A vacuity. Non-index world applies the "
        "index certificate under misspecification."
    ),
}

with open(OUT, "w") as f:
    json.dump(res, f, indent=1)

# ---------------- figures ----------------
th = np.array(sweep["theta"]); Mv = np.array(sweep["M"]); gv = np.array(sweep["gamma"])
bh = np.array(sweep["beta_hat"]); triv = np.array(sweep["trivial_bound"])
fig, ax = plt.subplots(figsize=(7, 4.2))
ax.plot(th, np.abs(Mv), label="|M(θ)| (observable margin)", lw=2)
ax.plot(th, np.abs(gv), label="|γ(θ)| (true drift, latent)", lw=2, ls="--")
ax.plot(th, bh, label="β̂(θ) (audited budget)", lw=2)
ax.plot(th, triv, label="trivial bound ½+|M| (vacuity floor)", lw=1.5, ls=":")
ax.set_xlabel("deployment shift θ"); ax.set_ylabel("magnitude")
ax.set_title("Audited budget: valid (β̂ ≥ |γ|) and non-vacuous (β̂ ≪ ½+|M|)")
ax.legend(fontsize=8); fig.tight_layout()
fig.savefig(os.path.join(HERE, "fig_aud_budget_curves.png"), dpi=150); plt.close(fig)

fig, ax = plt.subplots(figsize=(7, 3.6))
amap = {"adapt": 1, "abstain": 0, "freeze": -1}
aud = np.array([amap[a] for a in sweep["audited_action"]])
emp = np.array([amap[a] for a in sweep["empirical_action"]])
plg = np.array([amap[a] for a in sweep["plugin_action"]])
tru = np.array(sweep["delta_sign"])
ax.step(th, tru, where="mid", label="true sign(Δ)", lw=2)
ax.step(th, plg + 0.05, where="mid", label="β=0 plug-in", lw=1.5, ls="--")
ax.step(th, aud - 0.05, where="mid", label="audited (population M)", lw=1.5)
ax.step(th, emp - 0.10, where="mid", label="audited (empirical M̂, Thm Aud-E)", lw=1.5, ls="-.")
ax.set_yticks([-1, 0, 1], ["freeze", "abstain", "adapt"])
ax.set_xlabel("deployment shift θ")
ax.set_title(f"False strict commits — plug-in {fa_plg}/{com_plg}, audited {fa_aud}/{com_aud}, empirical {fa_emp}/{com_emp}")
ax.legend(fontsize=8); fig.tight_layout()
fig.savefig(os.path.join(HERE, "fig_aud_decisions.png"), dpi=150); plt.close(fig)

fig, ax = plt.subplots(figsize=(6.4, 3.6))
ax.semilogx(budget["n"], budget["ucb_width"], "o-", label="UCB width t(n, δ)")
ax2 = ax.twinx()
ax2.semilogx(budget["n"], budget["coverage"], "s--", color="tab:green", label="coverage")
ax2.axhline(1 - DELTA, color="gray", lw=0.8, ls=":")
ax2.set_ylim(0.8, 1.02); ax.set_xlabel("audit labels n"); ax.set_ylabel("width")
ax2.set_ylabel("empirical coverage")
ax.set_title("Label-budget audit: width shrinks, validity holds")
fig.tight_layout(); fig.savefig(os.path.join(HERE, "fig_aud_label_budget.png"), dpi=150); plt.close(fig)

print(json.dumps({
    "coverage_declaredL": res["C_coverage"]["coverage_declaredL"],
    "coverage_violatedL_3x": res["C_coverage"]["coverage_violatedL_3x"],
    "adversarial_covered": res["C_adversarial_step"]["covered"],
    "sweep_false_commits": {"plugin": f"{fa_plg}/{com_plg}", "audited": f"{fa_aud}/{com_aud}",
                            "empirical": f"{fa_emp}/{com_emp}"},
    "budget_coverage_min": min(budget["coverage"]),
    "F_index_msw": {
        "coverage_net": res["F_index_msw"]["coverage_net_certificate"],
        "coverage_ascent_heuristic": res["F_index_msw"]["coverage_grad_ascent_heuristic"],
        "coverage_nonindex_fail": res["F_index_msw"]["coverage_nonindex_failure_world"],
        "mean_beta_F": res["F_index_msw"]["mean_beta_F"],
        "mean_beta_null": res["F_index_msw"]["mean_beta_null_identical_laws"],
        "useful_vs_vacuity": res["F_index_msw"]["useful_vs_vacuity"],
        "n_net": res["F_index_msw"]["n_net"],
    },
}, indent=1))
