#!/usr/bin/env python3
"""
verify_headtohead.py - TORCH-FREE CPU verification of the mixed head-to-head apparatus
on clearly-labeled SYNTHETIC per-condition signals.

WHAT THIS PROVES (and ONLY this):
  STAGE 1 (decision rules on synthetic Z): the six decision rules (always-adapt,
    always-freeze, AETTA, POEM, KGA, oracle) execute and return valid {ADAPT,FREEZE,
    ABSTAIN} decisions on synthetic label-free signals, and the regret + false-adapt
    metrics compute for all six.
  STAGE 2 (verdict machinery, NOT hard-wired to WIN): the paired-bootstrap + Holm +
    pre-registered WIN/TIE/LOSE verdict machinery (headtohead_analysis) is fed THREE
    synthetic regret structures and must return WIN, TIE, and LOSE respectively. If the
    code always returned WIN, the TIE and LOSE assertions would fail.
  STAGE 3 (schema round-trip): synthetic decisions serialize via the harness serializer
    and re-load through multiseed_paired_ci.analyze (the SAME analysis the locked stress
    grid uses), proving the on-disk schema is compatible.

WHAT THIS DOES *NOT* PROVE:
  NOTHING about the real winner on real data. Every signal here is SYNTHETIC and LABELED
  (manifest fields carry SYNTHETIC_ prefixes; every record condition starts with
  "SYNTHETIC::"). The real KGA-vs-POEM/AETTA outcome is decided ONLY by
  run_mixed_headtohead.py on the real cached/fresh records (protocol sec 7,
  RUN_ON_MAC_POEM_AETTA.md).

Pure numpy (+ scipy via baselines' CDF). No torch, no sklearn.

RUN:  python experiments/kbound/poem_aetta/verify_headtohead.py
"""
from __future__ import annotations
import os
import sys
import json
import tempfile
import shutil
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_WILDS = os.path.join(os.path.dirname(_HERE), "wilds")
for _p in (_HERE, _WILDS):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import baselines as BL                         # noqa: E402
import run_mixed_headtohead as H2H             # noqa: E402
import multiseed_paired_ci as MPC              # noqa: E402

ZN = BL.Z_NAMES
LOGK = float(np.log(BL.DEFAULT_NUM_CLASSES))


# =========================================================================== #
#  Synthetic signal generator (CLEARLY LABELED, used only for STAGE 1 + 3)     #
# =========================================================================== #
def _synth_Z(true_B, seed_rng):
    """SYNTHETIC label-free Z vector loosely correlated with the synthetic true benefit.
    NOT real model outputs. Returns a length-11 list in Z_NAMES order."""
    b = float(true_B)
    helpful = b > 0
    nz = lambda s: float(seed_rng.normal(0, s))
    pre_entropy = 0.55 + nz(0.05)
    entropy_drop = (0.10 if helpful else -0.02) + nz(0.03)
    post_entropy = max(0.05, pre_entropy - max(entropy_drop, 0.0))
    pre_conf = 0.80 + nz(0.03)
    post_conf = float(np.clip(pre_conf + (0.05 if helpful else -0.04) + nz(0.03), 0.5, 0.99))
    pre_pbal = 0.95 + nz(0.02)
    post_pbal = float(np.clip((0.99 if helpful else 0.75) + nz(0.03), 0.4, 0.9999))
    pbal_drop = post_pbal - pre_pbal
    frac_highconf = float(np.clip((0.70 if helpful else 0.55) + nz(0.05), 0.3, 0.95))
    marginal_KL = max(0.0, LOGK - LOGK * post_pbal)
    update_norm = 3.0 + nz(0.5)
    z = {"pre_entropy": pre_entropy, "pre_conf": pre_conf, "pre_pbal": pre_pbal,
         "post_entropy": post_entropy, "post_conf": post_conf, "post_pbal": post_pbal,
         "pbal_drop": pbal_drop, "entropy_drop": entropy_drop,
         "frac_highconf": frac_highconf, "marginal_KL": marginal_KL,
         "update_norm": update_norm}
    return [z[name] for name in ZN]


def make_synthetic_records(n, harmful_frac, seed_rng):
    """n SYNTHETIC per-condition records with synthetic kga_decision = near-oracle."""
    recs = []
    for i in range(n):
        harmful = (seed_rng.uniform() < harmful_frac)
        true_B = (-abs(float(seed_rng.uniform(0.03, 0.10))) if harmful
                  else abs(float(seed_rng.uniform(0.03, 0.20))))
        a0 = float(seed_rng.uniform(0.55, 0.75))
        a_adapted = float(np.clip(a0 + true_B, 0.0, 1.0))
        Z = _synth_Z(true_B, seed_rng)
        if abs(true_B) < 0.01:
            kdec = "ABSTAIN"
        else:
            kdec = "ADAPT" if true_B > 0 else "FREEZE"
            if seed_rng.uniform() < 0.12:
                kdec = "ABSTAIN"
        recs.append({
            "condition": f"SYNTHETIC::cond{i:04d}",
            "B": true_B, "a0": a0, "a_adapted": a_adapted,
            "oracle_action": "ADAPT" if true_B > 0 else "FREEZE",
            "kga_decision": kdec, "eps_conformal": 0.02,
            "b_hat": true_B + float(seed_rng.normal(0, 0.01)),
            "Z": Z, "Z_names": list(ZN), "SYNTHETIC": True,
        })
    return recs


# =========================================================================== #
#  STAGE 1: real decision rules on synthetic Z                                 #
# =========================================================================== #
def stage1_decision_rules():
    print("-" * 80)
    print("STAGE 1: real decision rules (POEM, AETTA, KGA, oracle, trivials) on")
    print("         SYNTHETIC label-free signals -> valid decisions + metrics.")
    fails = []
    srng = np.random.default_rng(1234)
    recs = make_synthetic_records(300, harmful_frac=0.3, seed_rng=srng)
    dec = H2H.policy_decisions(recs, recompute_kga=False)
    for p in H2H.POLICIES:
        d = dec[p]
        if len(d) != len(recs):
            fails.append(f"{p}: wrong length")
        if not set(np.unique(d)) <= {"ADAPT", "FREEZE", "ABSTAIN"}:
            fails.append(f"{p}: invalid decision labels {set(np.unique(d))}")
        reg, acc = H2H.policy_regret(recs, d)
        if reg.shape[0] != len(recs):
            fails.append(f"{p}: regret wrong shape")
        if np.any(reg < -1e-9):
            fails.append(f"{p}: negative regret (oracle should dominate)")
        print(f"  {p:14s} decisions={ {k:int((d==k).sum()) for k in ['ADAPT','FREEZE','ABSTAIN'] } } "
              f"mean_regret={reg.mean():.4f}")
    # POEM and AETTA must produce a non-degenerate split on this synthetic mix
    for p in ("poem", "aetta"):
        ad = float(np.mean(dec[p] == "ADAPT"))
        if not (0.02 < ad < 0.98):
            fails.append(f"{p}: degenerate adapt fraction {ad:.3f} on synthetic mix "
                         f"(would indicate a broken/strawman rule)")
        else:
            print(f"  [{p} adapt fraction = {ad:.3f} -> non-degenerate, OK]")
    return fails


# =========================================================================== #
#  STAGE 2: verdict machinery on controlled synthetic regret arrays            #
#  (the purest test that the WIN/TIE/LOSE logic is not hard-wired to WIN)      #
# =========================================================================== #
def _build_world(kga_advantage_over_competitor, n=240, noise=0.004, rng=None,
                 tie_per_cond_noise=0.006):
    """Construct pooled per-condition regret arrays for all six policies such that
    mean(regret_KGA - regret_X) == -advantage for X in {poem, aetta}, i.e. a POSITIVE
    'advantage' means KGA has LOWER regret (KGA better) by that amount, so the signed
    diff(KGA,X)=regret_KGA-regret_X is NEGATIVE (the WIN direction).
    Everything here is a SYNTHETIC regret array used ONLY to exercise the CI + verdict
    logic. oracle regret = 0 by construction; trivial floors are set high so KGA beats
    them (isolating the POEM/AETTA verdict).

    For a genuine TIE (advantage == 0) we add LARGE independent per-condition noise to
    that competitor's regret so the 0-mean paired difference yields a CI that straddles
    0 (a true statistical tie), rather than a razor-thin spurious significance."""
    rng = rng or np.random.default_rng(0)
    # base per-condition regret for KGA. We use a base comfortably above 0 so that the
    # competitors' regret (base +/- small amounts) NEVER needs clipping at 0 -- clipping
    # would shift a mean and manufacture spurious significance in the TIE case.
    base = np.abs(rng.normal(0.040, noise, size=n)) + 0.010
    pooled = {}
    pooled["kga"] = base
    for X, adv in kga_advantage_over_competitor.items():
        if abs(adv) < 1e-12:
            # genuine tie: regret_X = regret_KGA + EXACTLY-mean-zero per-condition jitter
            # (subtract the sample mean so the empirical paired-diff mean is 0 to machine
            # precision; base >> jitter guarantees no clip). The paired-bootstrap CI is
            # then centered on 0 and straddles it -> a true statistical tie, deterministic.
            jitter = rng.normal(0.0, tie_per_cond_noise, size=n)
            jitter = jitter - jitter.mean()             # empirical mean exactly 0
            cand = base + jitter
            assert cand.min() > 0, "tie construction clipped (would bias the mean)"
            pooled[X] = cand
        else:
            # KGA advantage 'adv' (>0 => KGA better) => regret_X = regret_KGA + adv.
            cand = base + adv + rng.normal(0, noise / 2, size=n)
            assert cand.min() > 0, "advantage construction clipped"
            pooled[X] = cand
    # trivial floors KGA clearly beats; oracle = 0
    pooled["always_adapt"] = base + 0.05 + np.abs(rng.normal(0, noise, size=n))
    pooled["always_freeze"] = base + 0.10 + np.abs(rng.normal(0, noise, size=n))
    pooled["oracle"] = np.zeros(n)
    # meta: KGA false-adapt set to 0 (<= alpha) so WIN is reachable when CIs allow.
    meta = {p: {"mean_regret": float(np.mean(pooled[p])),
                "false_adapt_rate": 0.0} for p in H2H.POLICIES}
    return pooled, meta


def stage2_verdict_machinery():
    print("-" * 80)
    print("STAGE 2: WIN/TIE/LOSE verdict machinery on controlled SYNTHETIC regret")
    print("         arrays (proves it is NOT hard-wired to WIN).")
    fails = []
    rng = np.random.default_rng(42)

    cases = [
        # (name, dict of KGA ADVANTAGE over competitor [>0 => KGA better], expected, seed)
        # WIN: KGA lower regret than BOTH POEM and AETTA (adv>0 for both)
        ("WIN", {"poem": +0.006, "aetta": +0.006}, "WIN", 101),
        # LOSE: KGA HIGHER regret than POEM (adv<0 -> diff CI above 0 -> KGA loses)
        ("LOSE", {"poem": -0.006, "aetta": +0.006}, "LOSE", 202),
        # TIE: KGA ~ POEM (adv ~0 -> CI straddles 0), and KGA better than AETTA (no loss)
        ("TIE", {"poem": 0.0, "aetta": +0.006}, "TIE", 303),
    ]
    for name, diffs, expected, world_seed in cases:
        pooled, meta = _build_world(diffs, rng=np.random.default_rng(world_seed))
        h = H2H.headtohead_analysis(pooled, meta, f"SYNTHETIC_{name}")
        comps = {r["competitor"]: r for r in h["comparisons"]}
        print(f"\n  [{name}] expected={expected} got={h['VERDICT']}")
        for c in ("poem", "aetta"):
            r = comps[c]
            print(f"     KGA vs {c:6s} diff={r['mean_diff_kga_minus_competitor']:+.5f} "
                  f"CI[{r['ci95_lo']:+.5f},{r['ci95_hi']:+.5f}] p_holm={r['p_holm']:.2e} "
                  f"beats={r['kga_beats']} loses={r['kga_loses']}")
        if h["VERDICT"] != expected:
            fails.append(f"verdict {name}: expected {expected}, got {h['VERDICT']}")
        # structural: obs inside its own CI; p_holm in [0,1]
        for c, r in comps.items():
            if not (r["ci95_lo"] - 1e-9 <= r["mean_diff_kga_minus_competitor"] <= r["ci95_hi"] + 1e-9):
                fails.append(f"{name}/{c}: obs outside CI")
            if not (0.0 <= r["p_holm"] <= 1.0):
                fails.append(f"{name}/{c}: p_holm out of range")
    return fails


# =========================================================================== #
#  STAGE 3: schema round-trip through multiseed_paired_ci.analyze              #
# =========================================================================== #
def stage3_schema_roundtrip():
    print("-" * 80)
    print("STAGE 3: serialize SYNTHETIC decisions -> re-load via")
    print("         multiseed_paired_ci.analyze (the locked stress-grid analysis).")
    fails = []
    seeds = [0, 1, 2]
    per_seed_records, per_seed_dec = [], {p: [] for p in H2H.POLICIES}
    for s in seeds:
        srng = np.random.default_rng(2000 + s)
        recs = make_synthetic_records(150, harmful_frac=0.25, seed_rng=srng)
        per_seed_records.append(recs)
        dec = H2H.policy_decisions(recs, recompute_kga=False)
        for p in H2H.POLICIES:
            per_seed_dec[p].append(dec[p])
    tmp = tempfile.mkdtemp(prefix="synth_h2h_")
    try:
        tag = "SYNTHETIC"
        methods = ["kga", "poem", "aetta"]
        for p in methods:
            for s, recs, d in zip(seeds, per_seed_records, per_seed_dec[p]):
                H2H.serialize_policy(tmp, tag, p, s, recs, d)
        res = MPC.analyze(tmp, tag, methods, seeds, nboot=2000)
        if res["n_conditions"] != len(per_seed_records[0]):
            fails.append("n_conditions mismatch after round-trip")
        if set(res["candidates"].keys()) != set(methods):
            fails.append("candidate set mismatch after round-trip")
        if len(res["comparisons"]) != 2 * len(methods):
            fails.append("comparison count mismatch after round-trip")
        print(f"  round-trip ok: n_conditions={res['n_conditions']} "
              f"candidates={sorted(res['candidates'].keys())} "
              f"comparisons={len(res['comparisons'])}")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    return fails


def main():
    print("=" * 80)
    print("SYNTHETIC head-to-head apparatus verification (torch-free).")
    print("ALL DATA IS SYNTHETIC AND LABELED. It decides NOTHING about the real")
    print("KGA-vs-POEM/AETTA outcome — that is the real GPU/Mac run's job (protocol sec 7).")
    print("=" * 80)

    fails = []
    fails += stage1_decision_rules()
    fails += stage2_verdict_machinery()
    fails += stage3_schema_roundtrip()

    print("\n" + "=" * 80)
    if fails:
        print("VERIFICATION FAILED:")
        for f in fails:
            print("  -", f)
        print("=" * 80)
        sys.exit(1)
    print("VERIFICATION PASSED. The apparatus:")
    print("  STAGE 1: runs all six decision rules on synthetic Z with valid decisions")
    print("           and non-degenerate POEM/AETTA splits (not strawmen);")
    print("  STAGE 2: classifies SYNTHETIC regret structures as WIN, TIE, and LOSE")
    print("           respectively -> the verdict logic is NOT hard-wired to WIN;")
    print("  STAGE 3: round-trips the per-condition schema through the locked analysis.")
    print("This verifies the COMPARISON MACHINERY only. The real winner is decided by")
    print("run_mixed_headtohead.py on the real records (protocol sec 7).")
    print("=" * 80)


if __name__ == "__main__":
    main()
