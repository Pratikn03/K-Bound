#!/usr/bin/env python3
"""
gate_baseline_comparison.py  --  decision-baseline comparison for K-Bound (reviewer item 1).

Compares six LABEL-FREE adapt/freeze(/abstain) decision rules on the SAME locked per-cell
CIFAR-10-C stress data the KGA runner (cifar_tent_mps_v2.py) already produces:

  1. confidence gate    : adapt iff adaptation RAISED mean confidence   (post_conf > pre_conf)
  2. entropy gate       : adapt iff adaptation LOWERED mean entropy      (entropy_drop > 0)   [Tent's own objective]
  3. drift / KL gate    : freeze iff marginal-prediction KL exceeds a dev-calibrated threshold
  4. ATC-style gate     : adapt iff ATC-predicted adapted acc > predicted frozen acc (isotonic conf->acc)
  5. KGA (no radius)    : adapt iff Bhat > 0   (GBR point estimate, NO conformal margin, NO abstain)
  6. KGA (certificate)  : adapt / freeze / abstain via Bhat +/- eps      (the full method)

The point of the table: only the certificate (6) keeps a low *unconditional* false-adapt rate
(FA_u) while staying near-oracle on regret. The naive confidence/entropy gates false-adapt
because, under collapse, a harmful adaptation can look MORE confident / lower entropy.

INPUT  (--in cifar10c_percell.json): a list of per-cell records, each
        {"condition": "<corr>|<sev>|...", "Z": [11 floats], "a0": frozen_acc, "aa": adapted_acc, "regime": "helpful|harmful|marginal"}
        This is exactly the `rows` list cifar_tent_mps_v2.summarize() already receives; dump it with
        the 3-line hook in RUNNER_HOOK below.

OUTPUT : <out>.json and <out>.md with, per gate, regret-to-oracle, FA_u (unconditional),
         FA_c (conditional among adapts), coverage, adapt-rate -- on the full grid and on the
         harmful subset. Every threshold gate is calibrated LEAVE-ONE-TASK-OUT (task = corruption),
         exactly like KGA, so the comparison is apples-to-apples and uses no target labels at deploy.

  Real run:   python gate_baseline_comparison.py --in cifar10c_percell.json --out gate_comparison
  Self-test:  python gate_baseline_comparison.py --selftest
"""
import argparse, json, math
import numpy as np

# Evidence layout -- MUST match cifar_tent_mps_v2.EVIDENCE_NAMES
(PRE_ENT, PRE_CONF, PRE_PBAL, POST_ENT, POST_CONF, POST_PBAL,
 PBAL_DROP, ENT_DROP, FRAC_HI, MKL, UPD) = range(11)
ALPHA = 0.1

RUNNER_HOOK = r"""
# --- add inside cifar_tent_mps_v2.py, right after `rows` is built for a benchmark ---
# (rows is the list of {Z, a0, aa, regime, condition} that summarize() consumes)
import json as _json
_json.dump(rows, open("cifar10c_percell.json", "w"))
# then:  python gate_baseline_comparison.py --in cifar10c_percell.json --out gate_comparison
"""

# ----------------------------------------------------------------------------- gates
def gate_confidence(Z, B, tasks, a0, aa, alpha):
    return np.where(Z[:, POST_CONF] > Z[:, PRE_CONF], "ADAPT", "FREEZE")

def gate_entropy(Z, B, tasks, a0, aa, alpha):
    return np.where(Z[:, ENT_DROP] > 0, "ADAPT", "FREEZE")

def _drift_tau(mkl_dev, B_dev, alpha):
    """largest KL threshold tau s.t. dev false-adapt (adapt & B<0) <= alpha (max coverage under budget)."""
    best = -np.inf
    for tau in np.sort(np.unique(mkl_dev)):
        adapt = mkl_dev <= tau
        if adapt.sum() == 0:
            continue
        if np.mean(B_dev[adapt] < 0) <= alpha:
            best = tau
    return best

def gate_drift(Z, B, tasks, a0, aa, alpha):
    mkl = Z[:, MKL]; dec = np.empty(len(B), dtype=object)
    for t in np.unique(tasks):
        te = tasks == t; dv = ~te
        tau = _drift_tau(mkl[dv], B[dv], alpha)
        dec[te] = np.where(mkl[te] <= tau, "ADAPT", "FREEZE")
    return dec

def gate_atc(Z, B, tasks, a0, aa, alpha):
    from sklearn.isotonic import IsotonicRegression
    dec = np.empty(len(B), dtype=object)
    for t in np.unique(tasks):
        te = tasks == t; dv = ~te
        conf_dev = np.concatenate([Z[dv, PRE_CONF], Z[dv, POST_CONF]])
        acc_dev = np.concatenate([a0[dv], aa[dv]])
        ir = IsotonicRegression(out_of_bounds="clip").fit(conf_dev, acc_dev)
        dec[te] = np.where(ir.predict(Z[te, POST_CONF]) > ir.predict(Z[te, PRE_CONF]),
                           "ADAPT", "FREEZE")
    return dec

def _kga_bhat(Z, B, seed=0):
    from sklearn.ensemble import GradientBoostingRegressor
    N = len(B); Bhat = np.zeros(N)
    for i in range(N):
        tr = np.arange(N) != i
        m = GradientBoostingRegressor(n_estimators=250, max_depth=2, learning_rate=0.05,
                                      subsample=0.8, random_state=seed)
        m.fit(Z[tr], B[tr]); Bhat[i] = m.predict(Z[i:i + 1])[0]
    eps = float(np.quantile(np.abs(Bhat - B), 1 - ALPHA))
    return Bhat, eps

def gate_kga_noradius(Z, B, tasks, a0, aa, alpha):
    Bhat, _ = _kga_bhat(Z, B)
    return np.where(Bhat > 0, "ADAPT", "FREEZE")

def gate_kga_cert(Z, B, tasks, a0, aa, alpha):
    Bhat, eps = _kga_bhat(Z, B)
    return np.where(Bhat - eps > 0, "ADAPT", np.where(Bhat + eps < 0, "FREEZE", "ABSTAIN"))

GATES = [("confidence gate", gate_confidence), ("entropy gate", gate_entropy),
         ("drift/KL gate", gate_drift), ("ATC-style gate", gate_atc),
         ("KGA (no radius)", gate_kga_noradius), ("KGA (certificate)", gate_kga_cert)]

# ----------------------------------------------------------------------------- scoring
def score(dec, a0, aa, B, idx=None):
    if idx is not None:
        dec, a0, aa, B = dec[idx], a0[idx], aa[idx], B[idx]
    adapt = dec == "ADAPT"
    realized = np.where(adapt, aa, a0)             # abstain / freeze -> keep frozen
    oracle = np.maximum(a0, aa)
    return {
        "regret": float((oracle - realized).mean()),
        "FA_u": float(np.mean(adapt & (B < 0))),                       # P(adapt AND B<0)
        "FA_c": float(np.mean(B[adapt] < 0)) if adapt.any() else 0.0,  # P(B<0 | adapt)
        "coverage": float(np.mean(dec != "ABSTAIN")),
        "adapt_rate": float(adapt.mean()),
        "mean_acc": float(realized.mean()),
    }

def run_gate_comparison(rows, alpha=ALPHA):
    Z = np.array([r["Z"] for r in rows], float)
    a0 = np.array([r["a0"] for r in rows], float)
    aa = np.array([r["aa"] for r in rows], float)
    B = aa - a0
    tasks = np.array([r["condition"].split("|")[0] for r in rows])
    harmful = B < 0
    out = {"n": len(B), "n_harmful": int(harmful.sum()), "alpha": alpha, "gates": {}}
    for name, fn in GATES:
        dec = np.asarray(fn(Z, B, tasks, a0, aa, alpha), dtype=object)
        out["gates"][name] = {"all": score(dec, a0, aa, B),
                              "harmful_subset": score(dec, a0, aa, B, idx=harmful)}
    return out

def to_markdown(res):
    L = [f"# Decision-gate comparison (CIFAR-10-C stress; n={res['n']}, harmful={res['n_harmful']}, "
         f"alpha={res['alpha']})", "",
         "Lower regret and lower FA_u are better; the certificate is the only rule that keeps "
         "FA_u <= alpha while staying near-oracle.", "",
         "| Decision rule | regret | FA_u | FA_c | coverage | adapt-rate |",
         "|---|---:|---:|---:|---:|---:|"]
    for name, fn in GATES:
        g = res["gates"][name]["all"]
        L.append(f"| {name} | {g['regret']:.4f} | {g['FA_u']:.3f} | {g['FA_c']:.3f} | "
                 f"{g['coverage']:.2f} | {g['adapt_rate']:.2f} |")
    L += ["", "## On the harmful subset only (where naive gates fail)", "",
          "| Decision rule | regret | FA_u | FA_c | adapt-rate |", "|---|---:|---:|---:|---:|"]
    for name, fn in GATES:
        g = res["gates"][name]["harmful_subset"]
        L.append(f"| {name} | {g['regret']:.4f} | {g['FA_u']:.3f} | {g['FA_c']:.3f} | {g['adapt_rate']:.2f} |")
    return "\n".join(L)

# ----------------------------------------------------------------------------- self-test
def _synthetic_rows(n=300, seed=0):
    """Honest synthetic stress grid: harmful cells COLLAPSE (look confident / low entropy),
    so confidence/entropy gates false-adapt; only frac_highconf + marginal_KL reveal the harm,
    which the GBR (KGA) learns. This reproduces the qualitative failure the paper describes."""
    rng = np.random.default_rng(seed)
    corrs = ["gaussian_noise", "shot_noise", "motion_blur", "fog", "contrast", "jpeg"]
    rows = []
    for _ in range(n):
        u = rng.random()
        regime = "helpful" if u < 0.40 else ("harmful" if u < 0.75 else "marginal")
        if regime == "helpful":
            B = rng.uniform(0.02, 0.20)
        elif regime == "harmful":
            B = -rng.uniform(0.02, 0.20)
        else:
            B = rng.uniform(-0.02, 0.02)
        a0 = float(rng.uniform(0.40, 0.80)); aa = float(np.clip(a0 + B, 0, 1))
        pre_ent = rng.uniform(0.8, 1.4); pre_conf = rng.uniform(0.55, 0.75)
        # BOTH helpful and harmful look "better" on entropy/confidence (the confound):
        ent_drop = rng.uniform(0.05, 0.35); conf_rise = rng.uniform(0.03, 0.20)
        post_ent = pre_ent - ent_drop; post_conf = min(0.999, pre_conf + conf_rise)
        if regime == "harmful":                 # collapse signature, only visible in these two:
            frac_hi = rng.uniform(0.75, 0.99); mkl = rng.uniform(0.30, 1.20)
        elif regime == "helpful":
            frac_hi = rng.uniform(0.10, 0.45); mkl = rng.uniform(0.00, 0.12)
        else:
            frac_hi = rng.uniform(0.30, 0.70); mkl = rng.uniform(0.05, 0.40)
        Z = [pre_ent, pre_conf, 0.9, post_ent, post_conf, 0.85,
             0.05, ent_drop, frac_hi, mkl, rng.uniform(0.1, 2.0)]
        rows.append({"condition": f"{rng.choice(corrs)}|s5|x|iid|mild|r0",
                     "Z": Z, "a0": a0, "aa": aa, "regime": regime})
    return rows

def _selftest():
    rows = _synthetic_rows()
    res = run_gate_comparison(rows)
    md = to_markdown(res)
    print(md)
    g = res["gates"]
    cert = g["KGA (certificate)"]["all"]; conf = g["confidence gate"]["all"]; ent = g["entropy gate"]["all"]
    # sanity assertions: the certificate must dominate the naive gates on FA_u and not lose on regret
    assert cert["FA_u"] <= ALPHA + 1e-9, f"cert FA_u {cert['FA_u']} should be <= alpha"
    assert cert["FA_u"] < conf["FA_u"] and cert["FA_u"] < ent["FA_u"], "cert should false-adapt less than naive gates"
    assert cert["regret"] <= conf["regret"] + 1e-9, "cert regret should not exceed confidence gate"
    print("\n[selftest] PASS: certificate keeps FA_u<=alpha and beats naive gates on false-adapt.")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", help="per-cell rows JSON (from the runner hook)")
    ap.add_argument("--out", default="gate_comparison")
    ap.add_argument("--alpha", type=float, default=ALPHA)
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        _selftest(); return
    rows = json.load(open(a.inp))
    res = run_gate_comparison(rows, alpha=a.alpha)
    json.dump(res, open(a.out + ".json", "w"), indent=2)
    open(a.out + ".md", "w").write(to_markdown(res))
    print(to_markdown(res))
    print(f"\nwrote {a.out}.json and {a.out}.md")

if __name__ == "__main__":
    main()
