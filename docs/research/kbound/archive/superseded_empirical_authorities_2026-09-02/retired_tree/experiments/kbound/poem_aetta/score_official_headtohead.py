#!/usr/bin/env python3
"""
score_official_headtohead.py - WIN_HUNT_v4 arm_D scorer (CPU, torch-free).

Scores the OFFICIAL per-sample POEM (poem_official.py) and dropout-AETTA
(aetta_dropout.py) against the logged KGA decisions on the NEW per-sample stress run
(stress_persample_v1), and writes the pre-registered verdict to
research_lock/WIN_HUNT_v4_ARM_D_result.json.

Inputs (all produced by cifar_tent_mps_v2.py --log-samples), read from --run-dir:
  * per_condition_<dataset>_<method>_seed<S>.json     (a0, a_adapted, B, oracle_action,
                                                        kga_decision, condition)
  * samples_<dataset>_<method>_seed<S>__<cond>.npz    (frozen/adapted per-sample entropy,
                                                        aetta_acc_est[_frozen])
  * samples_source_<dataset>_seed<S>.npz              (POEM source-entropy CDF; optional,
                                                        faithful fallback if absent)

Pipeline (identical regret + paired-bootstrap machinery as the locked stress-grid
analysis, reusing multiseed_paired_ci.paired_boot / holm):
  regret_pol = oracle - realized;  oracle = max(a0, a_adapted);
               realized = a_adapted if decision==ADAPT else a0 (ABSTAIN/FREEZE -> a0);
  per-condition regret pooled (paired) across seeds; diff(KGA,X) = regret_KGA - regret_X;
  95% paired bootstrap (nboot=1e4) CI; Holm over the TWO comparisons {POEM, AETTA};
  false-adapt rate FA = mean(decision==ADAPT & B<=0) per policy.

Pre-registered arm_D bar (research_lock/WIN_HUNT_v4_PROTOCOL.yaml):
  "WIN iff both regret-gap CIs below zero with Holm; TIE/LOSS reported as-is."
  -> WIN  : KGA beats POEM and AETTA (each 95% CI of diff entirely < 0 AND survives Holm)
     LOSE : KGA loses to POEM or AETTA (each CI entirely > 0 AND survives Holm)
     TIE  : otherwise.
  FA is REPORTED (secondary); per the frozen headline-replacement policy a WIN only
  replaces a headline row when KGA FA <= alpha too (recorded as replacement_eligible).

Schema-defensive: any structural mismatch (missing npz, missing field, condition-order
or a0/a_adapted disagreement between JSON and npz) -> clear message + exit code 3.

Pure numpy. No torch, no sklearn.
"""
from __future__ import annotations
import os
import sys
import json
import glob
import time
import argparse
import subprocess
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_WILDS = os.path.join(os.path.dirname(_HERE), "wilds")
for _p in (_HERE, _WILDS):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import multiseed_paired_ci as MPC          # paired_boot, holm, load_cell (reused)
import poem_official as POEM
import aetta_dropout as AETTA

EXIT_SCHEMA = 3
A0_TOL = 5e-3                              # JSON vs npz a0/a_adapted agreement tolerance


class SchemaError(Exception):
    pass


# --------------------------------------------------------------------------- #
def _npz_scalar_str(v):
    return v.item() if getattr(v, "ndim", None) == 0 else str(v)


def _load_source_entropy(run_dir, dataset, seed):
    p = os.path.join(run_dir, f"samples_source_{dataset}_seed{seed}.npz")
    if not os.path.exists(p):
        return None
    try:
        with np.load(p, allow_pickle=True) as d:
            return np.asarray(d["source_entropy"], float)
    except Exception:
        return None


def _index_sample_npzs(run_dir, dataset, method, seed):
    """condition-string -> loaded npz dict, for one (dataset, method, seed)."""
    pat = os.path.join(run_dir, f"samples_{dataset}_{method}_seed{seed}__*.npz")
    out = {}
    for path in glob.glob(pat):
        try:
            with np.load(path, allow_pickle=True) as d:
                cond = _npz_scalar_str(d["condition"])
                out[cond] = {
                    "condition": cond,
                    "frozen_entropy": np.asarray(d["frozen_entropy"], float),
                    "adapted_entropy": np.asarray(d["adapted_entropy"], float),
                    "aetta_acc_est": float(d["aetta_acc_est"]),
                    "aetta_acc_est_frozen": float(d["aetta_acc_est_frozen"])
                    if "aetta_acc_est_frozen" in d.files else float("nan"),
                    "a0": float(d["a0"]) if "a0" in d.files else float("nan"),
                    "a_adapted": float(d["a_adapted"]) if "a_adapted" in d.files else float("nan"),
                }
        except Exception as e:
            raise SchemaError(f"unreadable npz {os.path.basename(path)}: {e}")
    return out


def _align_seed(run_dir, dataset, method, seed):
    """Return (records_json, streams_in_json_order, source_entropy) with strict schema
    checks. records_json is the per_condition list; streams[i] carries the per-sample
    arrays + AETTA estimates for records_json[i]['condition']."""
    try:
        cell = MPC.load_cell(run_dir, dataset, method, seed)
    except FileNotFoundError as e:
        raise SchemaError(f"missing per_condition file for seed{seed}: {e}")
    recs = cell.get("records")
    if not recs:
        raise SchemaError(f"per_condition seed{seed}: no 'records'")
    npz = _index_sample_npzs(run_dir, dataset, method, seed)
    if not npz:
        raise SchemaError(
            f"no samples_{dataset}_{method}_seed{seed}__*.npz in {run_dir} "
            f"(was the run launched with --log-samples?)")
    streams, missing, mism = [], [], []
    for r in recs:
        cond = r["condition"]
        s = npz.get(cond)
        if s is None:
            missing.append(cond)
            continue
        for k in ("frozen_entropy", "adapted_entropy", "aetta_acc_est"):
            if k not in s or (np.ndim(s[k]) and np.asarray(s[k]).size == 0):
                missing.append(f"{cond}:{k}")
        # sanity: npz correctness means must reproduce JSON a0 / a_adapted
        if np.isfinite(s["a0"]) and abs(s["a0"] - float(r["a0"])) > A0_TOL:
            mism.append(f"{cond} a0 json={float(r['a0']):.4f} npz={s['a0']:.4f}")
        if np.isfinite(s["a_adapted"]) and abs(s["a_adapted"] - float(r["a_adapted"])) > A0_TOL:
            mism.append(f"{cond} a_adapted json={float(r['a_adapted']):.4f} npz={s['a_adapted']:.4f}")
        streams.append(s)
    if missing:
        raise SchemaError(f"seed{seed}: {len(missing)} condition(s) missing npz/fields, "
                          f"e.g. {missing[:3]}")
    if mism:
        raise SchemaError(f"seed{seed}: {len(mism)} a0/a_adapted JSON-vs-npz mismatch(es), "
                          f"e.g. {mism[:3]} (npz and per_condition JSON are from different runs?)")
    src = _load_source_entropy(run_dir, dataset, seed)
    return recs, streams, src


# --------------------------------------------------------------------------- #
def _regret_and_fa(recs, decisions):
    """(per-condition regret array, false-adapt count, n) for a decision array.
    Regret convention identical to multiseed_paired_ci / the locked stress grid."""
    a0 = np.array([float(r["a0"]) for r in recs], float)
    aa = np.array([float(r["a_adapted"]) for r in recs], float)
    B = np.array([float(r["B"]) for r in recs], float)
    oracle = np.maximum(a0, aa)
    adapt = np.asarray(decisions) == "ADAPT"
    realized = np.where(adapt, aa, a0)
    fa = int(np.sum(adapt & (B <= 0.0)))
    return oracle - realized, fa, len(recs)


def score(run_dir, dataset, method, seeds, alpha_poem, poem_variant,
          aetta_floor, aetta_margin, alpha_falseadapt, nboot, boot_seed, holm_alpha):
    policies = ["kga", "poem", "aetta"]
    per_seed_reg = {p: [] for p in policies}
    fa_num = {p: 0 for p in policies}
    fa_den = {p: 0 for p in policies}
    cond_order = None
    poem_meta = aetta_meta = None
    harmful_rates = []

    for s in seeds:
        recs, streams, src = _align_seed(run_dir, dataset, method, s)
        keys = [r["condition"] for r in recs]
        if cond_order is None:
            cond_order = keys
        elif keys != cond_order:
            raise SchemaError(f"condition order differs at seed{s}; paired CIs require the "
                              f"same condition order across seeds")
        harmful_rates.append(float(np.mean([float(r["B"]) < 0 for r in recs])))

        dec = {}
        dec["kga"] = np.array([r["kga_decision"] for r in recs], dtype=object)
        dec["poem"], poem_meta = POEM.poem_official_decision(
            streams, source_entropy=src, alpha=alpha_poem, variant=poem_variant,
            return_detail=True)
        dec["aetta"], aetta_meta = AETTA.aetta_dropout_decision(
            streams, floor=aetta_floor, margin=aetta_margin, return_detail=True)

        for p in policies:
            reg, fa, n = _regret_and_fa(recs, dec[p])
            per_seed_reg[p].append(reg)
            fa_num[p] += fa
            fa_den[p] += n

    pooled = {p: np.mean(np.vstack(per_seed_reg[p]), axis=0) for p in policies}
    fa_rate = {p: (fa_num[p] / fa_den[p]) if fa_den[p] else None for p in policies}

    rng = np.random.default_rng(boot_seed)
    comps, pvals, labels = [], [], []
    for X in ("poem", "aetta"):
        diff = pooled["kga"] - pooled[X]            # <0 => KGA lower regret (better)
        obs, lo, hi, p = MPC.paired_boot(diff, nboot, rng)
        lab = f"kga_vs_{X}"
        pvals.append(p); labels.append(lab)
        comps.append({"competitor": X, "label": lab,
                      "kga_mean_regret": float(np.mean(pooled["kga"])),
                      "competitor_mean_regret": float(np.mean(pooled[X])),
                      "mean_diff_kga_minus_competitor": obs,
                      "ci95_lo": lo, "ci95_hi": hi, "p_raw": p})
    holm_adj = MPC.holm(pvals, labels)
    for row in comps:
        row["p_holm"] = holm_adj[row["label"]]
        row["ci_below_zero"] = bool(row["ci95_hi"] < 0.0)
        row["ci_above_zero"] = bool(row["ci95_lo"] > 0.0)
        row["kga_beats"] = bool(row["ci_below_zero"] and row["p_holm"] < holm_alpha)
        row["kga_loses"] = bool(row["ci_above_zero"] and row["p_holm"] < holm_alpha)

    by = {r["competitor"]: r for r in comps}
    beats_poem, beats_aetta = by["poem"]["kga_beats"], by["aetta"]["kga_beats"]
    loses_poem, loses_aetta = by["poem"]["kga_loses"], by["aetta"]["kga_loses"]
    if loses_poem or loses_aetta:
        verdict = "LOSE"
    elif beats_poem and beats_aetta:
        verdict = "WIN"
    else:
        verdict = "TIE"
    fa_ok = (fa_rate["kga"] is not None) and (fa_rate["kga"] <= alpha_falseadapt)

    return {
        "arm": "arm_D_official_persample_headtohead",
        "protocol": "research_lock/WIN_HUNT_v4_PROTOCOL.yaml (registered 2026-07-04)",
        "dataset": dataset, "adapter": method, "seeds": list(seeds),
        "n_conditions_per_seed": len(cond_order),
        "harmful_base_rate_range": [min(harmful_rates), max(harmful_rates)],
        "nboot": nboot, "boot_seed": boot_seed, "holm_alpha": holm_alpha,
        "alpha_falseadapt": alpha_falseadapt,
        "poem_config": {"alpha_detect": alpha_poem, "variant": poem_variant,
                        "source_cdf": poem_meta["cdf_source"] if poem_meta else None,
                        "alarm_threshold": poem_meta["alarm_threshold"] if poem_meta else None},
        "aetta_config": {"floor": aetta_floor, "margin": aetta_margin,
                         "N_dropout_passes": 10, "alpha_entropy_ratio": 3.0,
                         "dropout_rate": "dataset-specific (CIFAR-10 0.4)"},
        "policy_mean_regret": {p: float(np.mean(pooled[p])) for p in policies},
        "policy_false_adapt_rate": fa_rate,
        "comparisons": comps,
        "beats_poem": beats_poem, "beats_aetta": beats_aetta,
        "loses_to_poem": loses_poem, "loses_to_aetta": loses_aetta,
        "kga_false_adapt_rate": fa_rate["kga"], "kga_false_adapt_le_alpha": fa_ok,
        "VERDICT": verdict,
        "replacement_eligible": bool(verdict == "WIN" and fa_ok),
        "verdict_criterion": ("WIN iff KGA beats POEM AND AETTA (each 95% paired-bootstrap "
                              "CI of regret_KGA-regret_X entirely < 0 AND survives Holm at "
                              "family-wise alpha=0.05); LOSE if KGA loses to either; else TIE. "
                              "arm_D bar does not gate the VERDICT on FA; replacement_eligible "
                              "adds the frozen headline policy's FA<=alpha requirement."),
        "verdict_legend": {
            "WIN": "official per-sample POEM and dropout-AETTA are both beaten by KGA on regret",
            "TIE": "KGA matches at least one of the official baselines on regret (no loss)",
            "LOSE": "KGA is beaten by official POEM and/or dropout-AETTA on regret"},
    }


def _git_hash():
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=_HERE,
                                       stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        return "unknown"


def _print_summary(res):
    print("=" * 80)
    print(f"ARM D official head-to-head  dataset={res['dataset']}  adapter={res['adapter']}  "
          f"seeds={res['seeds']}  n_cond/seed={res['n_conditions_per_seed']}")
    print(f"harmful base-rate range: {res['harmful_base_rate_range']}")
    print("-" * 80)
    print("policy   mean_regret   false_adapt")
    for p in ("kga", "poem", "aetta"):
        fa = res["policy_false_adapt_rate"][p]
        print(f"  {p:6s}  {res['policy_mean_regret'][p]:10.5f}   "
              f"{'n/a' if fa is None else f'{fa:.4f}'}")
    print("-" * 80)
    for r in res["comparisons"]:
        print(f"  {r['label']:12s} diff={r['mean_diff_kga_minus_competitor']:+.5f} "
              f"CI[{r['ci95_lo']:+.5f},{r['ci95_hi']:+.5f}] p_holm={r['p_holm']:.2e} "
              f"beats={r['kga_beats']} loses={r['kga_loses']}")
    print("-" * 80)
    print(f"  KGA FA={res['kga_false_adapt_rate']} <= alpha({res['alpha_falseadapt']}): "
          f"{res['kga_false_adapt_le_alpha']}")
    print(f"  >>> VERDICT: {res['VERDICT']}  (replacement_eligible={res['replacement_eligible']})")
    print("=" * 80)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run-dir", required=True,
                    help="dir with per_condition_*.json + samples_*.npz (stress_persample_v1)")
    ap.add_argument("--dataset", default="cifar10c")
    ap.add_argument("--adapter", "--method", dest="adapter", default="tent")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--alpha-poem", type=float, default=POEM.DEFAULT_ALPHA,
                    help="POEM detection level (alarm at wealth >= 1/alpha). Default 0.05.")
    ap.add_argument("--poem-variant", default="matching", choices=["matching", "frozen_only"])
    ap.add_argument("--aetta-floor", type=float, default=AETTA.DEFAULT_FLOOR)
    ap.add_argument("--aetta-margin", type=float, default=AETTA.DEFAULT_MARGIN)
    ap.add_argument("--alpha-falseadapt", type=float, default=0.10)
    ap.add_argument("--nboot", type=int, default=10000)
    ap.add_argument("--boot-seed", type=int, default=20260704)   # registration date; frozen
    ap.add_argument("--holm-alpha", type=float, default=0.05)
    ap.add_argument("--out", default=None,
                    help="verdict JSON (default research_lock/WIN_HUNT_v4_ARM_D_result.json "
                         "resolved from repo root, i.e. two levels above experiments/kbound)")
    a = ap.parse_args(argv)

    try:
        res = score(a.run_dir, a.dataset, a.adapter, a.seeds, a.alpha_poem, a.poem_variant,
                    a.aetta_floor, a.aetta_margin, a.alpha_falseadapt, a.nboot,
                    a.boot_seed, a.holm_alpha)
    except SchemaError as e:
        print(f"[arm-D][SCHEMA] {e}", file=sys.stderr)
        sys.exit(EXIT_SCHEMA)

    res["git_hash"] = _git_hash()
    res["python"] = sys.version.split()[0]
    res["scored"] = time.strftime("%Y-%m-%d %H:%M:%S")
    res["run_dir"] = os.path.abspath(a.run_dir)

    if a.out:
        out_path = a.out
    else:
        repo_root = os.path.dirname(os.path.dirname(os.path.dirname(_HERE)))  # the repository root
        out_path = os.path.join(repo_root, "research_lock", "WIN_HUNT_v4_ARM_D_result.json")
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(res, f, indent=2)
    _print_summary(res)
    print("WROTE", os.path.abspath(out_path))
    return out_path


if __name__ == "__main__":
    main()
