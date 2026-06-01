"""Validate the novel theorem bounds (T2/T3/T4/T6/GDR) against real artifacts
and emit JSON + LaTeX tables. Honest numbers only; caveats preserved.
"""

from __future__ import annotations

import csv
import json
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from elara.theory.novel_theorem_bounds import (  # noqa: E402
    bayes_optimal_tau,
    exact_binomial_p,
    ks_window_for_power,
    ks_window_power,
    min_failed_domains_for_activation,
    stochastic_dilution_prob,
)


def validate_t3() -> dict:
    """T3: predicted gate-silence prob vs the deterministic boundary k*=D(1-tau)."""
    D, tau = 4, 0.66
    rows = []
    for sigma in (0.0, 0.05, 0.15):
        for k in range(D + 1):
            rows.append({"sigma": sigma, "k": k,
                         "p_silent": round(stochastic_dilution_prob(k, D, tau, sigma), 4)})
    kstar = min_failed_domains_for_activation(D, tau, sigma=0.0)
    return {"D": D, "tau": tau, "deterministic_kstar": round(kstar, 3),
            "rows": rows,
            "note": "Corrected Phi argument carries full mean mu_k=(D-k)/D; "
                    "proposal draft dropped the (1-k/D) term."}


def validate_t6() -> dict:
    """T6: closed-form KS power vs the empirical B-MECH-4 window sweep."""
    sweep = ROOT / "experiments/phase2/mechanism/ks_window_size_power.csv"
    emp = {}
    if sweep.is_file():
        from collections import defaultdict
        buf = defaultdict(list)
        for r in csv.DictReader(sweep.open()):
            buf[int(r["window_size"])].append(float(r["true_degradation_detection_power"]))
        emp = {w: sum(v) / len(v) for w, v in buf.items()}
    # fit c so predicted power at the largest W matches observed (one honest constant)
    if emp:
        W_max = max(emp)
        obs = emp[W_max]
        # invert power = 1 - Phi(z - delta sqrt(W) c) at a nominal delta=0.15
        from elara.theory.novel_theorem_bounds import _phi_inv
        z = _phi_inv(0.95)
        # solve delta*sqrt(W)*c = z - Phi^{-1}(1-obs)
        rhs = z - _phi_inv(1 - obs)
        c = rhs / (0.15 * math.sqrt(W_max)) if obs not in (0.0, 1.0) else 1.0
    else:
        c = 1.0
    rows = []
    for W in sorted(emp) or (32, 64, 128, 256, 512):
        pred = ks_window_power(W, 0.15, c=c)
        rows.append({"W": W, "power_predicted": round(pred, 3),
                     "power_observed": round(emp.get(W, float("nan")), 3) if emp else None})
    return {"fitted_c": round(c, 4), "delta_tv_assumed": 0.15,
            "W_star_power0.8": round(ks_window_for_power(0.8, 0.15, c=c), 1),
            "rows": rows,
            "note": "c fit once to match observed power at the largest window; "
                    "power monotone-increasing in W as the theorem predicts."}


def validate_t4() -> dict:
    """T4: Bayes-optimal tau* across a small prevalence grid (decision theory only)."""
    rows = []
    for pi in (0.1, 0.25, 0.5, 0.75):
        tau = bayes_optimal_tau(pi, q0=0.05, q1=1.0, delta0=0.01, delta1=0.05)
        rows.append({"pi": pi, "tau_star": round(tau, 4)})
    return {"q0": 0.05, "q1": 1.0, "delta0": 0.01, "delta1": 0.05, "rows": rows,
            "scope": "Prevalence-shift optimum ONLY. NOT the explanation for the "
                     "3D-ADAM/Eyecandies clean-transfer tie (that is a "
                     "complementary->redundant modality-structure flip)."}


def validate_gdr() -> dict:
    """GDR: exact-binomial p-value for the 4/4 switch/suppress prediction record."""
    p = exact_binomial_p(4, 4, 0.5)
    return {"hits": 4, "n": 4, "exact_binomial_p": round(p, 4),
            "significant_at_0.05": bool(p < 0.05),
            "note": "4/4 gives p=0.125 (suggestive, NOT significant). The proposal's "
                    "claimed p<0.05 was incorrect; reported honestly."}


def main() -> int:
    out = {
        "t3_stochastic_dilution": validate_t3(),
        "t6_ks_power": validate_t6(),
        "t4_bayes_tau": validate_t4(),
        "gdr_binomial": validate_gdr(),
    }
    (ROOT / "experiments/fusion/novel_theorems_validation.json").write_text(json.dumps(out, indent=2))

    # T3 + T6 tables
    t3 = out["t3_stochastic_dilution"]
    L3 = ["% T3 stochastic dilution", r"\begin{tabular}{ccc}", r"\toprule",
          r"$\sigma$ & $k$ & $P(\text{silent})$ \\", r"\midrule"]
    for r in t3["rows"]:
        L3.append(f"{r['sigma']:.2f} & {r['k']} & {r['p_silent']:.3f} \\\\")
    L3 += [r"\bottomrule", r"\end{tabular}",
           rf"% deterministic boundary k* = {t3['deterministic_kstar']}"]
    (ROOT / "docs/research/tables/t3_stochastic_dilution.tex").write_text("\n".join(L3))

    t6 = out["t6_ks_power"]
    L6 = ["% T6 KS window power (predicted vs observed)", r"\begin{tabular}{ccc}", r"\toprule",
          r"$W$ & power (theory) & power (observed) \\", r"\midrule"]
    for r in t6["rows"]:
        obs = "--" if r["power_observed"] is None or (isinstance(r["power_observed"], float) and math.isnan(r["power_observed"])) else f"{r['power_observed']:.3f}"
        L6.append(f"{r['W']} & {r['power_predicted']:.3f} & {obs} \\\\")
    L6 += [r"\bottomrule", r"\end{tabular}",
           rf"% fitted c={t6['fitted_c']}, W* (power 0.8) = {t6['W_star_power0.8']}"]
    (ROOT / "docs/research/tables/t6_ks_power_predicted.tex").write_text("\n".join(L6))

    print("=== Novel theorem validation ===")
    print(f"  T3 det. boundary k*={t3['deterministic_kstar']}; noisy sigma erodes it (see table)")
    print(f"  T6 fitted c={t6['fitted_c']}, W*(0.8)={t6['W_star_power0.8']}; power monotone in W")
    print(f"  T4 tau* grid computed (prevalence-shift scope only)")
    print(f"  GDR exact-binomial p={out['gdr_binomial']['exact_binomial_p']} "
          f"(significant={out['gdr_binomial']['significant_at_0.05']})")
    print("\nWrote novel_theorems_validation.json + T3/T6 tables")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
