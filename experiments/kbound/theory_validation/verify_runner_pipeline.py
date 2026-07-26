#!/usr/bin/env python3
"""
verify_runner_pipeline.py - torch-free CPU verification of the gap-relevant runner logic.

WHAT THIS PROVES (without any GPU, torch, or sklearn)
  The two open K-Bound gaps -- ImageNet-R multi-seed and Camelyon17-with-SAR -- need the
  user's Mac GPU only for *model inference*.  Everything downstream of inference is
  torch-free and is what actually has to be correct for the multi-seed paired-CI result:
      (1) per-condition SERIALIZATION into the stress_grid_multiseed schema
          (experiments/kbound/wilds/per_condition_serialize.py),
      (2) the single-candidate KGA DECISION RULE over those conditions,
      (3) multi-seed AGGREGATION + PAIRED bootstrap CIs + Holm
          (experiments/kbound/wilds/multiseed_paired_ci.py).
  This harness feeds SYNTHETIC per-condition score/loss arrays through the EXACT same code
  the GPU runners call, writes real-schema output JSON, and asserts the contract:
      * per-condition arrays present for every (method, seed) cell,
      * all requested seeds present,
      * the SAR column present (Camelyon17),
      * paired CIs computable (finite lo/hi, p-values in (0,1]).

INTEGRITY
  Every synthetic artifact is stamped "_synthetic_smoke": true and lives ONLY under the
  caller-supplied --out-dir (default: a *_SMOKE_VERIFY dir).  No real result JSON is read
  or written.  The benefit estimator here is the documented numpy fallback
  (kga_backend="numpy_knn_fallback"); the production sklearn gradient-boosted estimator is
  exercised on the Mac.  The serialization layout, decision rule, conformal radius, and the
  paired-bootstrap/Holm machinery are the SAME objects used in production.

USAGE (sandbox)
  PYTHONPATH=<repo>:<repo>/src:<repo>/experiments/kbound/wilds \
      python3 experiments/kbound/theory_validation/verify_runner_pipeline.py --smoke
"""
from __future__ import annotations
import os
import sys
import json
import argparse
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))            # .../experiments/kbound/theory_validation
REPO = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))  # the repository root
WILDS = os.path.join(REPO, "experiments", "kbound", "wilds")
for _p in (WILDS, REPO, os.path.join(REPO, "src")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import per_condition_serialize as pcs   # noqa: E402  torch-free
import multiseed_paired_ci as mc        # noqa: E402  torch-free


def _label_regime(B, thr=0.02):
    return "helpful" if B > thr else ("harmful" if B < -thr else "marginal")


# ---- realistic synthetic grids (axes mirror the real runners) -------------- #
CAMELYON = dict(
    dataset="camelyon17",
    methods=["tent", "eata", "sar"],          # <-- SAR must appear
    domains=["test", "val", "id_val"],
    comps=["iid", "imbalanced", "single_class"],
    regimes=["small"],
    aggrs=["mild", "aggressive"],
)
IMAGENETR = dict(
    dataset="imagenet-r",
    methods=["tent", "eata", "sar"],
    domains=["imagenet_r"],
    comps=["iid", "imbalanced", "single_class"],
    regimes=["small", "tiny"],
    aggrs=["mild", "aggressive"],
)


def synth_records(spec, seeds, rng_master):
    """Build SYNTHETIC runner-style records[] for a dataset spec.

    Each record mimics what the GPU runner emits per (seed,domain,comp,regime,aggr,method):
    a0 (frozen acc), aa (adapted acc), B=aa-a0, and an 11-dim label-free Z.  The synthetic
    benefit is correlated with Z so the KGA estimator has signal to certify (this is a
    *plumbing* test, not a claim about real effect sizes).  Method-specific harmful rates
    are imposed so SAR looks different from tent/eata, exercising the column plumbing.
    """
    # per-method synthetic harmful tendency (purely to differentiate columns)
    harm_bias = {"tent": -0.02, "eata": -0.01, "sar": +0.01}
    recs = []
    for seed in seeds:
        rng = np.random.default_rng(rng_master + 1000 * int(seed))
        for dom in spec["domains"]:
            for comp in spec["comps"]:
                for regime in spec["regimes"]:
                    for aggr in spec["aggrs"]:
                        # latent "shift difficulty" drives both Z and benefit
                        diff = rng.uniform(0.0, 1.0)
                        a0 = float(np.clip(0.80 - 0.25 * diff + rng.normal(scale=0.01), 0.4, 0.98))
                        for method in spec["methods"]:
                            # emit BOTH adaptation modes per method (as the real WILDS
                            # runners do: {online,episodic}); this exercises the unique-key
                            # dedup path in per_condition_serialize.
                            for mode in ("online", "episodic"):
                                base = harm_bias[method] + 0.06 * (0.5 - diff)
                                base += 0.005 if mode == "episodic" else 0.0
                                B = float(base + rng.normal(scale=0.015))
                                aa = float(np.clip(a0 + B, 0.30, 0.99))
                                B = aa - a0
                                # 11-dim Z: first dims correlate with benefit sign + difficulty
                                z = rng.normal(scale=0.05, size=11)
                                z[0] = 0.5 - 0.4 * diff            # pre_entropy-ish
                                z[1] = 0.6 + 0.3 * (1 - diff)      # pre_conf-ish
                                z[7] = B * 4.0 + rng.normal(scale=0.02)   # entropy_drop ~ benefit
                                z[10] = float(aggr == "aggressive")  # update_norm-ish proxy
                                recs.append(dict(
                                    seed=int(seed), domain=dom, comp=comp, regime=regime, aggr=aggr,
                                    method=method, mode=mode, candidate=f"{method}_{mode}",
                                    a0=a0, aa=aa, B=B, upd_norm=float(abs(z[10])),
                                    Z=[float(v) for v in z], regime_label=_label_regime(B),
                                    _synthetic_smoke=True))
    return recs


def verify_dataset(spec, seeds, out_root, nboot, rng_master=20260619):
    dataset = spec["dataset"]
    out_dir = os.path.join(out_root, f"{dataset.replace('-', '')}_SMOKE_VERIFY")
    os.makedirs(out_dir, exist_ok=True)
    recs = synth_records(spec, seeds, rng_master)

    # (1)+(2) per-condition serialization through the SAME production module,
    # numpy backend (sklearn-free sandbox).
    ser = pcs.serialize_run(
        recs, dataset=dataset, out_dir=out_dir, seeds=seeds, methods=spec["methods"],
        prefer="numpy",
        extra_top={"_synthetic_smoke": True,
                   "_synthetic_note": "CPU verification of serialization/decision/CI plumbing; "
                                      "scores are synthetic, estimator is numpy fallback."})

    # (3) multi-seed aggregation + paired bootstrap CIs + Holm through the SAME module.
    res = mc.analyze(out_dir, dataset, spec["methods"], seeds, nboot=nboot)
    res["_synthetic_smoke"] = True
    res["_synthetic_note"] = ("Synthetic scores. Proves serialization + multi-seed paired-CI "
                              "plumbing only; not a real K-Bound result.")
    ana_path = os.path.join(out_dir, "MULTISEED_ANALYSIS_RESULTS.SYNTHETIC.json")
    json.dump(res, open(ana_path, "w"), indent=2)

    # -------- ASSERTIONS (the actual contract) --------
    checks = {}
    n_seeds = len(seeds)
    expected_cells = len(spec["methods"]) * n_seeds
    # per-condition arrays present for every (method, seed)
    files = {}
    for m in spec["methods"]:
        for s in seeds:
            fp = os.path.join(out_dir, f"per_condition_{dataset}_{m}_seed{s}.json")
            assert os.path.exists(fp), f"missing per-condition file {fp}"
            d = json.load(open(fp))
            assert isinstance(d.get("records"), list) and len(d["records"]) > 0, \
                f"empty per-condition records in {fp}"
            r0 = d["records"][0]
            for key in ("B", "a0", "a_adapted", "regime", "oracle_action", "Z", "Z_names",
                        "b_hat", "eps_conformal", "kga_decision"):
                assert key in r0, f"per-condition record missing '{key}' in {fp}"
            assert len(r0["Z"]) == len(pcs.EVIDENCE_NAMES), "Z dimensionality mismatch"
            conds = [rec["condition"] for rec in d["records"]]
            assert len(conds) == len(set(conds)), \
                f"duplicate condition keys in {fp} (online/episodic collision?)"
            files[f"{m}_seed{s}"] = len(d["records"])
    checks["per_condition_files_written"] = len(files)
    checks["expected_cells"] = expected_cells
    assert len(files) == expected_cells, "not all (method,seed) cells serialized"

    # all requested seeds present
    seeds_seen = sorted({int(k.split("seed")[-1]) for k in files})
    assert seeds_seen == sorted(int(s) for s in seeds), \
        f"seed set mismatch: got {seeds_seen}, want {sorted(seeds)}"
    checks["all_seeds_present"] = seeds_seen

    # SAR column present (the Camelyon17 gap) -- assert for any spec that lists it
    if "sar" in spec["methods"]:
        assert any(k.startswith("sar_seed") for k in files), "SAR per-condition column missing"
        assert "sar" in res["candidates"], "SAR absent from multi-seed analysis candidates"
        checks["sar_column_present"] = True

    # paired CIs computable: finite lo/hi and p in (0,1] for every comparison
    assert res["comparisons"], "no paired comparisons produced"
    for c in res["comparisons"]:
        assert np.isfinite(c["ci95_lo"]) and np.isfinite(c["ci95_hi"]), \
            f"non-finite CI for {c['label']}"
        assert c["ci95_lo"] <= c["mean_diff_kga_minus_trivial"] <= c["ci95_hi"] + 1e-9, \
            f"observed diff outside CI for {c['label']}"
        assert 0.0 < c["p_raw"] <= 1.0, f"p_raw out of range for {c['label']}"
        assert 0.0 <= c["p_holm"] <= 1.0, f"p_holm out of range for {c['label']}"
    checks["paired_CIs_computable"] = len(res["comparisons"])
    checks["n_conditions_per_cell"] = res["n_conditions"]
    checks["kga_backend"] = ser["kga_backend"]

    return {
        "dataset": dataset, "out_dir": out_dir,
        "per_condition_manifest": ser["cells"],
        "analysis_file": ana_path,
        "comparisons": [
            {"label": c["label"], "mean_diff": round(c["mean_diff_kga_minus_trivial"], 6),
             "ci95": [round(c["ci95_lo"], 6), round(c["ci95_hi"], 6)],
             "p_raw": round(c["p_raw"], 5), "p_holm": round(c["p_holm"], 5),
             "survives_holm": c["survives_holm"]}
            for c in res["comparisons"]],
        "checks": checks,
    }


def main(argv=None):
    p = argparse.ArgumentParser(description="Torch-free CPU verification of runner pipeline")
    p.add_argument("--out-dir", default=os.path.join(REPO, "experiments", "kbound", "results",
                                                     "_pipeline_smoke_verify"),
                   help="where to write SYNTHETIC verification artifacts (never a real run dir)")
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2],
                   help=">=3 seeds proves the multi-seed plumbing (default 0 1 2)")
    p.add_argument("--nboot", type=int, default=2000,
                   help="bootstrap resamples for the paired CIs in the smoke (prod uses 1e4)")
    p.add_argument("--smoke", action="store_true", help="alias; this harness is always a smoke")
    a = p.parse_args(argv)
    os.makedirs(a.out_dir, exist_ok=True)

    report = {"_synthetic_smoke": True,
              "_what_this_verifies": ("torch-free per-condition serialization + single-candidate "
                                      "KGA decision rule + multi-seed paired-bootstrap CIs + Holm, "
                                      "exactly as the GPU runners call them. Synthetic scores only."),
              "seeds": a.seeds, "nboot": a.nboot, "datasets": {}}
    ok = True
    for spec in (CAMELYON, IMAGENETR):
        try:
            report["datasets"][spec["dataset"]] = verify_dataset(spec, a.seeds, a.out_dir, a.nboot)
        except AssertionError as e:
            ok = False
            report["datasets"][spec["dataset"]] = {"PASSED": False, "assertion_error": str(e)}
    report["ALL_ASSERTIONS_PASSED"] = ok
    rep_path = os.path.join(a.out_dir, "VERIFY_RUNNER_PIPELINE_REPORT.SYNTHETIC.json")
    json.dump(report, open(rep_path, "w"), indent=2)

    # human-readable summary
    print("=" * 74)
    print("CPU VERIFICATION OF RUNNER PIPELINE  (SYNTHETIC scores; torch-free, sklearn-free)")
    print("=" * 74)
    for ds, r in report["datasets"].items():
        if r.get("PASSED") is False:
            print(f"[{ds}] FAILED: {r['assertion_error']}")
            continue
        ch = r["checks"]
        print(f"\n[{ds}]  cells={ch['per_condition_files_written']}/{ch['expected_cells']}  "
              f"seeds={ch['all_seeds_present']}  conds/cell={ch['n_conditions_per_cell']}  "
              f"kga_backend={ch['kga_backend']}")
        print(f"        SAR column present : {ch.get('sar_column_present', 'n/a')}")
        print(f"        paired CIs computable for {ch['paired_CIs_computable']} comparisons")
        for c in r["comparisons"]:
            print(f"        {c['label']:24s} diff={c['mean_diff']:+.6f} "
                  f"CI[{c['ci95'][0]:+.6f},{c['ci95'][1]:+.6f}] "
                  f"p_raw={c['p_raw']:.4f} p_holm={c['p_holm']:.4f} survive={c['survives_holm']}")
    print("\n" + "-" * 74)
    print(f"ALL_ASSERTIONS_PASSED = {ok}")
    print(f"report -> {rep_path}")
    print("NOTE: all numbers above are SYNTHETIC (plumbing proof). Real metrics come from "
          "the GPU runs on the Mac.")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
