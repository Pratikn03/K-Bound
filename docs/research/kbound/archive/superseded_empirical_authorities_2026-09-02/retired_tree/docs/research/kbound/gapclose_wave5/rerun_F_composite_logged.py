"""Arm F (WIN_HUNT_v4) — KGA-v2 COMPOSITE on natural protocols, logged re-analysis.

Frozen per research_lock/WIN_HUNT_v4_PROTOCOL.yaml (arm_F_composite_natural):
"arms A+B instruments applied jointly to G/H/M v2 and the pooled universal-gate
stream, same frozen splits; scored once."

Composite definition (auditable, validity-preserving — the literal reading of
"jk+ radius + estimator v2"):
  * ESTIMATOR = arm B, estimator_v2.py: the frozen 3-config GBR ensemble
    (INCUMBENT_CFG + DEEPER_CFG + SHALLOW_CFG, equal weights). No search.
  * RADIUS    = arm A, radius_jackknife_plus.py: the jackknife+ / CV+ interval
    of Barber et al. (2021), which has finite-sample distribution-free coverage
    >= 1 - 2*alpha for ANY base learner. We plug the estimator-v2 ensemble in as
    that base learner (JackknifePlusGate(make_learner=EnsembleV2Regressor)).
  So the composite is a SINGLE gate: the arm-B estimator with an arm-A jackknife+
  interval built on its own leave-one-seed predictions. Both upgrades, jointly.

Everything else is IDENTICAL to arm A's logged re-analysis (rerun_A_jkplus_logged.py):
same per_condition/per_panel loader, same leave-one-seed split, same INCUMBENT
(radius_v2 crossfit_oof + Mondrian signed conformal), same regret / FA_u / paired
10^4 bootstrap of the per-cell accuracy improvement. CPU, LOGGED DATA ONLY; no new
fitting on held-out labels; fabricates nothing (schema-defensive, exits 3).

Bar (arm_F): FA_u <= alpha everywhere; per-protocol CI_ROBUST_IMPROVEMENT iff the
regret-gap CI vs the incumbent KGA excludes zero. Pooled universal-gate stream
(--pool over several run-dirs): additionally WIN iff the composite beats BOTH fixed
policies (always-adapt, always-freeze) with CIs excluding zero AND composite regret
<= 0.0183 (the existing universal-gate regret) with the improvement CI excluding zero.

Run (from repo root):
  .venv/bin/python docs/research/kbound/gapclose_wave5/rerun_F_composite_logged.py \
      --run-dir experiments/kbound/results/natural_win_v2_camelyon --dataset camelyon17 --alpha 0.10
  .venv/bin/python docs/research/kbound/gapclose_wave5/rerun_F_composite_logged.py \
      --run-dir experiments/kbound/results/natural_win_v1_imagenetr --dataset imagenet-r --panel --alpha 0.10
  # pooled universal-gate stream (repeat --run-dir/--dataset[/--panel] per source):
  .venv/bin/python docs/research/kbound/gapclose_wave5/rerun_F_composite_logged.py --pool \
      --source natural_win_v2_camelyon:camelyon17 \
      --source natural_win_v1_imagenetr:imagenet-r:panel --alpha 0.10
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", "..", "..", ".."))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

# Reuse the FROZEN arm-A machinery verbatim (loader, incumbent, bootstrap) so the
# only thing that changes vs arm A is the composite gate's base learner.
from rerun_A_jkplus_logged import (  # noqa: E402
    load_records, build_arrays, incumbent_v3, paired_bootstrap,
)
from radius_jackknife_plus import JackknifePlusGate  # noqa: E402
from estimator_v2 import _ensemble_fit, _ensemble_pred  # noqa: E402

UNIVERSAL_GATE_EXISTING_REGRET = 0.0183  # frozen reference in arm_F bar


class EnsembleV2Regressor:
    """sklearn-style wrapper around the arm-B frozen 3-config GBR ensemble, so it can
    be dropped into JackknifePlusGate as the jackknife+ base learner (arm A x arm B)."""

    def __init__(self):
        self.models_ = None

    def fit(self, Z, B):
        self.models_ = _ensemble_fit(np.asarray(Z, float), np.asarray(B, float))
        return self

    def predict(self, Z):
        return _ensemble_pred(self.models_, np.asarray(Z, float))


def composite_leave_one_seed(Z, B, g, alpha):
    """KGA-v2 composite decision under leave-one-seed: for each held-out seed,
    calibrate a jackknife+/CV+ gate whose base learner is the estimator-v2 ensemble
    on the OTHER seeds, then decide the held-out cells. 1=ADAPT, -1=FREEZE, 0=ABSTAIN."""
    dec = np.zeros(B.shape[0], dtype=int)
    for s in np.unique(g):
        te = g == s
        other = ~te
        go = g[other]
        gate = JackknifePlusGate(alpha, make_learner=EnsembleV2Regressor)
        if np.unique(go).size >= 2:
            gate.fit_grouped(Z[other], B[other], go)
        else:  # only one remaining group -> per-point LOO fallback
            gate.fit(Z[other], B[other])
        codes, _lo, _hi = gate.decide_batch(Z[te])
        dec[te] = codes
    return dec


def score_one(Z, B, g, a0, aa, alpha, nboot, has_ev2, tag):
    dec_inc = incumbent_v3(Z, B, g, alpha)
    dec_comp = composite_leave_one_seed(Z, B, g, alpha)

    oracle = np.maximum(a0, aa)
    acc_inc = np.where(dec_inc == 1, aa, a0)
    acc_comp = np.where(dec_comp == 1, aa, a0)
    reg_inc = float(np.mean(oracle - acc_inc))
    reg_comp = float(np.mean(oracle - acc_comp))
    fa_inc = float(np.mean((dec_inc == 1) & (B <= 0)))
    fa_comp = float(np.mean((dec_comp == 1) & (B <= 0)))

    # regret vs the two fixed policies (needed for the pooled universal-gate WIN bar)
    reg_always_adapt = float(np.mean(oracle - aa))
    reg_always_freeze = float(np.mean(oracle - a0))

    rng = np.random.default_rng(20260705)
    # improvement vs incumbent KGA (per-protocol CI-robust-improvement bar)
    imp_vs_inc = acc_comp - acc_inc
    mean_imp, lo_imp, hi_imp, p_imp = paired_bootstrap(imp_vs_inc, nboot, rng)
    # improvement vs fixed policies (universal-gate beats-both bar)
    gap_adapt = acc_comp - aa      # >0 => composite recovers more accuracy than always-adapt
    gap_freeze = acc_comp - a0
    m_a, lo_a, hi_a, p_a = paired_bootstrap(gap_adapt, nboot, rng)
    m_f, lo_f, hi_f, p_f = paired_bootstrap(gap_freeze, nboot, rng)

    ci_robust = bool(mean_imp > 0.0 and lo_imp > 0.0 and fa_comp <= alpha)
    beats_both_ci = bool(lo_a > 0.0 and lo_f > 0.0)
    return dict(
        tag=tag, alpha=alpha, n_records=int(B.shape[0]),
        n_seeds=int(np.unique(g).size), evidence_dims=int(Z.shape[1]), used_ev2=bool(has_ev2),
        incumbent=dict(regret=reg_inc, FA_u=fa_inc,
                       adapt=float((dec_inc == 1).mean()),
                       freeze=float((dec_inc == -1).mean()),
                       abstain=float((dec_inc == 0).mean())),
        composite=dict(regret=reg_comp, FA_u=fa_comp,
                       adapt=float((dec_comp == 1).mean()),
                       freeze=float((dec_comp == -1).mean()),
                       abstain=float((dec_comp == 0).mean())),
        fixed_policies=dict(regret_always_adapt=reg_always_adapt,
                            regret_always_freeze=reg_always_freeze),
        regret_improvement_vs_incumbent=dict(mean=mean_imp, ci=[lo_imp, hi_imp], p=p_imp),
        gap_vs_always_adapt=dict(mean=m_a, ci=[lo_a, hi_a], p=p_a),
        gap_vs_always_freeze=dict(mean=m_f, ci=[lo_f, hi_f], p=p_f),
        nboot=int(nboot),
        FA_u_le_alpha=bool(fa_comp <= alpha),
        CI_ROBUST_IMPROVEMENT=ci_robust,
        beats_both_CI=beats_both_ci,
    )


def _load(run_dir, dataset, method, panel):
    rows = load_records(run_dir, dataset, method, panel)
    return build_arrays(rows)  # Z, B, g, a0, aa, has_ev2


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir")
    ap.add_argument("--dataset")
    ap.add_argument("--method", default=None)
    ap.add_argument("--panel", action="store_true")
    ap.add_argument("--alpha", type=float, default=0.10)
    ap.add_argument("--nboot", type=int, default=10000)
    ap.add_argument("--pool", action="store_true",
                    help="pooled universal-gate stream over several --source entries")
    ap.add_argument("--source", action="append", default=[],
                    help="pooled source 'RESULTDIR:dataset[:panel]' (dir relative to experiments/kbound/results or absolute)")
    args = ap.parse_args()

    out_dir = os.path.join(REPO, "research_lock")
    os.makedirs(out_dir, exist_ok=True)

    if args.pool:
        if not args.source:
            print("SCHEMA ERROR: --pool needs >=1 --source", file=sys.stderr)
            return 3
        Zs, Bs, gs, a0s, aas = [], [], [], [], []
        used_ev2_any = False
        base = os.path.join(REPO, "experiments/kbound/results")
        for k, spec in enumerate(args.source):
            parts = spec.split(":")
            rd = parts[0]
            ds = parts[1]
            pn = (len(parts) > 2 and parts[2] == "panel")
            rd_abs = rd if os.path.isabs(rd) else os.path.join(base, rd)
            Z, B, g, a0, aa, has_ev2 = _load(rd_abs, ds, None, pn)
            used_ev2_any = used_ev2_any or has_ev2
            # pad/truncate Z to a common width across sources (evidence dims can differ);
            # the composite gate is fit PER SOURCE-independent leave-one-seed below, but for
            # ONE universal gate we align to the min common evidence width (base-11 shared Z).
            Zs.append(Z); Bs.append(B)
            gs.append(g + 100 * k)  # disjoint seed-group ids per source
            a0s.append(a0); aas.append(aa)
        wmin = min(z.shape[1] for z in Zs)
        Z = np.vstack([z[:, :wmin] for z in Zs])
        B = np.concatenate(Bs); g = np.concatenate(gs)
        a0 = np.concatenate(a0s); aa = np.concatenate(aas)
        res = score_one(Z, B, g, a0, aa, args.alpha, args.nboot, used_ev2_any,
                        tag="pooled_universal_gate")
        res["protocol"] = "WIN_HUNT_v4"; res["arm"] = "arm_F_composite_natural"
        res["mode"] = "pooled_universal_gate"
        res["sources"] = args.source
        res["universal_gate_bar"] = dict(
            existing_regret=UNIVERSAL_GATE_EXISTING_REGRET,
            regret_le_existing=bool(res["composite"]["regret"] <= UNIVERSAL_GATE_EXISTING_REGRET),
            beats_both_CI=res["beats_both_CI"],
            improvement_ci_excludes_zero=bool(res["regret_improvement_vs_incumbent"]["ci"][0] > 0.0),
            WIN=bool(res["beats_both_CI"] and
                     res["composite"]["regret"] <= UNIVERSAL_GATE_EXISTING_REGRET and
                     res["regret_improvement_vs_incumbent"]["ci"][0] > 0.0 and
                     res["FA_u_le_alpha"]))
        print(json.dumps(res, indent=1))
        with open(os.path.join(out_dir, "WIN_HUNT_v4_ARM_F_pooled_result.json"), "w") as fh:
            json.dump(res, fh, indent=1)
        return 0

    if not (args.run_dir and args.dataset):
        print("SCHEMA ERROR: need --run-dir and --dataset (or --pool)", file=sys.stderr)
        return 3
    Z, B, g, a0, aa, has_ev2 = _load(args.run_dir, args.dataset, args.method, args.panel)
    res = score_one(Z, B, g, a0, aa, args.alpha, args.nboot, has_ev2, tag=args.dataset)
    res["protocol"] = "WIN_HUNT_v4"; res["arm"] = "arm_F_composite_natural"
    res["dataset"] = args.dataset; res["run_dir"] = args.run_dir
    res["scoring_mode"] = "panel" if args.panel else "per_candidate"
    print(json.dumps(res, indent=1))
    with open(os.path.join(out_dir, f"WIN_HUNT_v4_ARM_F_{args.dataset}_result.json"), "w") as fh:
        json.dump(res, fh, indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
