"""NATURAL_WIN_PROTOCOL_v1 — pre-committed analysis (frozen with the protocol).

Consumes per_condition_<dataset>_<method>_seed<S>.json files produced by the
Wave-5-patched runners (c_ij / n_D / Z_ev2 present) and applies the frozen
decision rule:

  gate:   self-normalized tau' (tau_selfnorm.py); multicandidate routing only
          where the gate does NOT reject H.
  radius: V3 = leave-one-seed cross-fitted GBR + signed asymmetric conformal
          + Mondrian terciles (radius_v2.py), alpha = 0.10 per direction.
  Z:      runner Z concatenated with Z_ev2 where present.

ImageNet-R secondary arm (diverse 10-backbone panel): scores ONCE per
(seed, condition) — not once per backbone. Prefer per_panel_<dataset>_seed<S>.json
when present; otherwise collapse the per-candidate files.

Outputs regret vs always-adapt / always-freeze with per-condition paired
bootstrap (10^4) + Holm over the 2-comparison family, FA_u, and the verdict
WIN / CI-ROBUST WIN / NO-HARM / FAIL. Fabricates nothing; errors loudly on
missing fields.

Usage:
  python3 natural_win_analysis.py --run-dir <dir> --dataset camelyon17
  python3 natural_win_analysis.py --run-dir <dir> --dataset imagenet-r --panel
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from collections import defaultdict

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", "..", "..", ".."))
WILDS = os.path.join(REPO, "experiments", "kbound", "wilds")
TV = os.path.join(REPO, "experiments", "kbound", "theory_validation")
for p in (HERE, WILDS, TV):
    if p not in sys.path:
        sys.path.insert(0, p)

from radius_v2 import crossfit_oof, mondrian_bounds  # noqa: E402
from tau_selfnorm import tau_selfnorm  # noqa: E402

try:
    import val_multicandidate_residual as vmc  # noqa: E402
except Exception as _e:  # pragma: no cover
    vmc = None
    _VMC_ERR = repr(_e)

ALPHA = 0.10
KAPPA = 2.5
MIN_D = 8
TAU_SIM = 300

# Protocol D backbone order (must match run_imagenetr_kbound preds_mat row order).
DIVERSE_BACKBONES = [
    "resnet101", "resnet152", "resnext101_32x8d", "efficientnet_b0",
    "efficientnet_b3", "convnext_tiny", "convnext_base", "vit_b_16",
    "swin_t", "swin_b",
]


def load_per_candidate_records(run_dir: str, dataset: str, method: str | None):
    pat = os.path.join(run_dir, f"per_condition_{dataset}_*_seed*.json")
    files = [f for f in sorted(glob.glob(pat))
             if not os.path.basename(f).startswith("._")]
    if method:
        files = [f for f in files if f"_{method}_" in os.path.basename(f)]
    if not files:
        print(f"SCHEMA ERROR: no per_condition files match {pat}", file=sys.stderr)
        sys.exit(3)
    rows = []
    for f in files:
        d = json.load(open(f))
        for r in d["records"]:
            rows.append(r)
    return rows


def load_panel_files(run_dir: str, dataset: str):
    pat = os.path.join(run_dir, f"per_panel_{dataset}_seed*.json")
    files = [f for f in sorted(glob.glob(pat))
             if not os.path.basename(f).startswith("._")]
    rows = []
    for f in files:
        d = json.load(open(f))
        rows.extend(d.get("records", []))
    return rows


def collapse_imagenetr_panel(run_dir: str, dataset: str):
    """One row per (seed, condition) from per-candidate backbone files."""
    raw = load_per_candidate_records(run_dir, dataset, method=None)
    groups: dict[tuple, list] = defaultdict(list)
    for r in raw:
        groups[(int(r["seed"]), r["condition"])].append(r)
    if not groups:
        print("SCHEMA ERROR: no records to collapse for panel scoring", file=sys.stderr)
        sys.exit(3)
    rows = []
    for (seed, cond), rs in sorted(groups.items()):
        a0 = float(rs[0]["a0"])
        by_method = {r["method"]: r for r in rs}
        missing = [b for b in DIVERSE_BACKBONES if b not in by_method]
        if missing:
            print(f"SCHEMA ERROR: panel cell {seed}/{cond} missing backbones "
                  f"{missing}", file=sys.stderr)
            sys.exit(3)
        aa_all = [a0] + [float(by_method[b]["a_adapted"]) for b in DIVERSE_BACKBONES]
        best_aa = float(max(aa_all))
        best_r = max(rs, key=lambda r: float(r["a_adapted"]))
        row = dict(
            seed=int(seed), condition=cond, a0=a0,
            a_adapted=best_aa, B=float(best_aa - a0),
            aa_all=aa_all,
            cand_names=["freeze_f0"] + list(DIVERSE_BACKBONES),
            Z=list(map(float, best_r["Z"])),
            c_ij=rs[0].get("c_ij"), n_D=rs[0].get("n_D"),
            Z_ev2=best_r.get("Z_ev2"),
        )
        rows.append(row)
    return rows


def load_records(run_dir: str, dataset: str, method: str | None, panel: bool):
    if panel:
        rows = load_panel_files(run_dir, dataset)
        if rows:
            return rows
        if dataset in ("imagenet-r", "imagenet_r"):
            return collapse_imagenetr_panel(run_dir, dataset)
        print("SCHEMA ERROR: --panel requested but no per_panel files and "
              f"no collapse rule for {dataset}", file=sys.stderr)
        sys.exit(3)
    return load_per_candidate_records(run_dir, dataset, method)


def route_realized(route: dict, aa_all: list[float]) -> float:
    if route.get("decision") == "ADAPT" and route.get("choice") is not None:
        return float(aa_all[int(route["choice"])])
    return float(aa_all[0])


def multicandidate_from_agreement(C: np.ndarray, n_D: int,
                                  kappa: float = KAPPA) -> dict:
    """Theorem-1A style route from a stored agreement matrix (τ′ already passed)."""
    if vmc is None:
        return {"decision": "ERROR", "choice": None, "reason": _VMC_ERR}
    C = np.asarray(C, float).copy()
    M = C.shape[0]
    if M < 4:
        return {"decision": "ABSTAIN", "choice": None, "n_D": int(n_D),
                "reason": f"need M>=4; got M={M}"}
    if int(n_D) < MIN_D:
        return {"decision": "FREEZE", "choice": None, "n_D": int(n_D),
                "reason": f"|D|={n_D} < min_D={MIN_D}"}
    np.fill_diagonal(C, 0.0)
    b_hat, _tau = vmc.rankone_fit_offdiag(C)
    off = ~np.eye(M, dtype=bool)
    h_hat = float(np.abs(C - np.outer(b_hat, b_hat))[off].max())
    margin = kappa * h_hat + 2.0 / np.sqrt(max(int(n_D), 1))
    adv = b_hat[1:] - b_hat[0]
    committed = [i + 1 for i in range(M - 1)
                 if adv[i] > margin and b_hat[i + 1] > 0]
    if not committed:
        return {"decision": "FREEZE", "choice": None, "n_D": int(n_D),
                "reason": "no candidate beats anchor by margin"}
    choice = int(max(committed, key=lambda i: b_hat[i]))
    return {"decision": "ADAPT", "choice": choice, "n_D": int(n_D),
            "committed": committed}


def route_decision_code(route: dict) -> int:
    d = route.get("decision")
    if d == "ADAPT":
        return 1
    if d == "FREEZE":
        return -1
    return 0


def paired_bootstrap(diff_by_cond: np.ndarray, nboot: int, rng) -> tuple:
    n = len(diff_by_cond)
    idx = rng.integers(0, n, size=(nboot, n))
    means = diff_by_cond[idx].mean(axis=1)
    lo, hi = np.quantile(means, [0.025, 0.975])
    p = 2.0 * min((means >= 0).mean(), (means <= 0).mean())
    return float(diff_by_cond.mean()), float(lo), float(hi), float(max(p, 1.0 / nboot))


def score_rows(rows: list[dict], nboot: int, panel: bool) -> dict:
    need = ("Z", "B", "a0", "a_adapted", "seed", "condition")
    for k in need:
        if any(r.get(k) is None for r in rows):
            print(f"SCHEMA ERROR: field {k} missing in some records", file=sys.stderr)
            sys.exit(3)

    has_ev2 = any(r.get("Z_ev2") for r in rows)
    dim_ev2 = len(next((r["Z_ev2"] for r in rows if r.get("Z_ev2")), [])) if has_ev2 else 0
    Z = np.array([list(map(float, r["Z"]))
                  + (list(map(float, r["Z_ev2"])) if r.get("Z_ev2")
                     else [0.0] * dim_ev2) for r in rows])
    B = np.array([float(r["B"]) for r in rows])
    g = np.array([int(r["seed"]) for r in rows])
    a0 = np.array([float(r["a0"]) for r in rows])
    aa = np.array([float(r["a_adapted"]) for r in rows])

    n = len(rows)
    gate_pass = np.zeros(n, dtype=bool)
    n_gated = n_reject = n_multicand = 0
    tau_cache: dict = {}
    for i, r in enumerate(rows):
        C = r.get("c_ij")
        m = r.get("n_D")
        if C is None or not m or int(m) < 20:
            continue
        key = (r["condition"], int(r["seed"]))
        if key not in tau_cache:
            res = tau_selfnorm(np.array(C, float), int(m), alpha=0.05,
                               n_sim=TAU_SIM, seed=abs(hash(key)) % (1 << 30))
            tau_cache[key] = bool(res["reject_H"])
        n_gated += 1
        gate_pass[i] = not tau_cache[key]
        if tau_cache[key]:
            n_reject += 1

    # V3 radius (single-candidate fallback when τ′ rejects H)
    Bhat = crossfit_oof(Z, B, g)
    resid = B - Bhat
    dec_v3 = np.zeros(n, dtype=int)
    for s in np.unique(g):
        cal, te = g != s, g == s
        lo, hi = mondrian_bounds(Bhat[cal], resid[cal], Bhat[te], ALPHA)
        d = np.zeros(int(te.sum()), dtype=int)
        d[Bhat[te] + lo > 0] = 1
        d[Bhat[te] + hi < 0] = -1
        dec_v3[te] = d

    acc_kga = np.zeros(n, float)
    dec = np.zeros(n, dtype=int)
    for i, r in enumerate(rows):
        if panel and gate_pass[i] and r.get("aa_all"):
            route = multicandidate_from_agreement(np.array(r["c_ij"], float),
                                                  int(r["n_D"]))
            acc_kga[i] = route_realized(route, r["aa_all"])
            dec[i] = route_decision_code(route)
            n_multicand += 1
        else:
            acc_kga[i] = aa[i] if dec_v3[i] == 1 else a0[i]
            dec[i] = dec_v3[i]

    oracle = np.maximum(a0, aa)
    reg = dict(kga=float(np.mean(oracle - acc_kga)),
               adapt=float(np.mean(oracle - aa)),
               freeze=float(np.mean(oracle - a0)))
    fa_u = float(np.mean((dec == 1) & (B <= 0)))

    rng = np.random.default_rng(20260702)
    d_adapt = (oracle - acc_kga) - (oracle - aa)
    d_freeze = (oracle - acc_kga) - (oracle - a0)
    m_a, lo_a, hi_a, p_a = paired_bootstrap(d_adapt, nboot, rng)
    m_f, lo_f, hi_f, p_f = paired_bootstrap(d_freeze, nboot, rng)
    ps = sorted([("adapt", p_a), ("freeze", p_f)], key=lambda t: t[1])
    holm = {ps[0][0]: min(1.0, 2 * ps[0][1]), ps[1][0]: min(1.0, ps[1][1])}

    win = reg["kga"] < reg["adapt"] and reg["kga"] < reg["freeze"] and fa_u <= ALPHA
    ci_robust = win and hi_a < 0 and hi_f < 0
    better, worse = (("freeze", "adapt") if reg["freeze"] <= reg["adapt"]
                     else ("adapt", "freeze"))
    no_harm = (not win and fa_u <= ALPHA
               and reg["kga"] <= reg[better] + 1e-3
               and reg["kga"] < reg[worse])
    verdict = ("CI_ROBUST_WIN" if ci_robust else
               "WIN" if win else "NO_HARM" if no_harm else "FAIL")

    gate_note = ""
    if n_gated == 0:
        gate_note = "c_ij absent on all records"
    elif panel:
        gate_note = (f"panel scoring: {n_multicand}/{n} cells used multicandidate "
                     f"(τ′ pass); {n_reject} fell back to V3")

    return dict(
        n_records=len(rows), n_seeds=len(set(g.tolist())),
        evidence_dims=int(Z.shape[1]), used_ev2=bool(has_ev2),
        gate=dict(n_gated=n_gated, n_reject_H=n_reject, n_multicandidate=n_multicand,
                  note=gate_note),
        decisions=dict(adapt=float((dec == 1).mean()),
                         freeze=float((dec == -1).mean()),
                         abstain=float((dec == 0).mean())),
        regret=reg, FA_u=fa_u,
        vs_adapt=dict(mean=m_a, ci=[lo_a, hi_a], p_holm=holm["adapt"]),
        vs_freeze=dict(mean=m_f, ci=[lo_f, hi_f], p_holm=holm["freeze"]),
        VERDICT=verdict,
        scoring_mode="panel" if panel else "per_candidate",
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--method", default=None,
                    help="restrict to one method/candidate (per-candidate mode only)")
    ap.add_argument("--panel", action="store_true",
                    help="score once per (seed, condition) panel cell "
                         "(ImageNet-R diverse-backbone secondary arm)")
    ap.add_argument("--nboot", type=int, default=10000)
    args = ap.parse_args()

    rows = load_records(args.run_dir, args.dataset, args.method, panel=args.panel)
    scored = score_rows(rows, args.nboot, panel=args.panel)

    out = dict(protocol="NATURAL_WIN_PROTOCOL_v1", dataset=args.dataset,
               run_dir=args.run_dir, method=args.method, alpha=ALPHA, **scored)
    print(json.dumps(out, indent=1))
    tag = args.method or "all"
    if args.panel:
        tag = "panel"
    with open(os.path.join(args.run_dir,
                           f"NATURAL_WIN_v1_{args.dataset}_{tag}.json"), "w") as f:
        json.dump(out, f, indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
