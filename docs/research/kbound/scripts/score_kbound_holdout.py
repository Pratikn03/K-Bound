#!/usr/bin/env python3
"""Score a selected K-Bound config on a separate held-out record file.

Unlike analyze_F.py, this script supports different calibration and test JSONs.
Use it after a dev-only finder chooses one candidate/config.  Do not sweep over
the held-out test with this script and then claim the best row as pre-registered.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / "docs/research/kbound/scripts"))
import analyze_F as af  # noqa: E402


def load_records(patterns: list[str], candidate: str) -> list[dict]:
    out = []
    for pat in patterns:
        paths = sorted(glob.glob(pat))
        if not paths:
            raise FileNotFoundError(f"no files match {pat}")
        for path in paths:
            data = json.loads(Path(path).read_text())
            top_method = data.get("method")
            cand_arg = candidate
            if candidate == "auto" and top_method:
                cand_arg = str(top_method)
            recs, _ = af.load_records(path, candidate=None if cand_arg == "auto" else cand_arg)
            for r in recs:
                if candidate != "auto" and str(r.get("candidate")) != candidate:
                    continue
                try:
                    z = np.asarray(r["Z"], dtype=float)
                    vals = [float(r["B"]), float(r["a0"]), float(r["aa"])]
                    if not np.isfinite(z).all() or not np.isfinite(vals).all():
                        continue
                    rr = dict(r)
                    rr["Z"] = [float(v) for v in z]
                    rr["B"], rr["a0"], rr["aa"] = vals
                    rr["seed"] = int(rr["seed"])
                    out.append(rr)
                except Exception:
                    continue
    return out


def arrays(records):
    Z = np.array([r["Z"] for r in records], float)
    B = np.array([r["B"] for r in records], float)
    a0 = np.array([r["a0"] for r in records], float)
    aa = np.array([r["aa"] for r in records], float)
    comp = np.array([r.get("comp", "unknown") for r in records])
    return Z, B, a0, aa, comp


def score_transfer(cal_records, test_records, estimator: str, conformal: str, frozen_eps=None):
    Zc, Bc, _a0c, _aac, compc = arrays(cal_records)
    Zt, Bt, a0t, aat, compt = arrays(test_records)
    m = af.fit_point(Zc, Bc)
    Bhat_c = m.predict(Zc)
    Bhat_t = m.predict(Zt)
    if estimator == "ppi_debias":
        Bhat_c, Bhat_t = af.ppi_debias(Bhat_c, Bc, Zc, Zt, Bhat_t)
    elif estimator != "gbr":
        raise ValueError(estimator)

    # Leave-one-calibration-record-out residuals remove in-sample residual bias.
    # Deployment Bhat_t still comes from a separately refit full-calibration model,
    # so this is empirical cross-fitted calibration unless estimator stability is
    # established; it is not exact split conformal or jackknife+.
    if conformal == "frozen":
        resid_c = None  # eps supplied externally (frozen dev radius); skip the LOO refit
    elif estimator == "gbr":
        _loo = np.empty(len(Bc))
        for _i in range(len(Bc)):
            _tr = np.arange(len(Bc)) != _i
            _loo[_i] = af.fit_point(Zc[_tr], Bc[_tr]).predict(Zc[_i:_i + 1])[0]
        resid_c = np.abs(_loo - Bc)
    else:
        resid_c = np.abs(Bhat_c - Bc)  # ppi_debias variant (non-headline); unchanged

    if conformal == "global":
        eps = af.conformal_rank_radius(resid_c, af.ALPHA)
        dec = af.decide_global(Bhat_t, eps)
    elif conformal == "mondrian":
        eps_glob = af.conformal_rank_radius(resid_c, af.ALPHA)
        dec = np.array(["ABSTAIN"] * len(Bhat_t), dtype=object)
        groups = set(compc.tolist())
        for g in groups:
            mc = compc == g
            epsg = (
                af.conformal_rank_radius(resid_c[mc], af.ALPHA)
                if mc.sum() >= 5 else eps_glob
            )
            mt = compt == g
            dec[mt] = af.decide_global(Bhat_t[mt], epsg)
        unseen = ~np.isin(compt, list(groups))
        dec[unseen] = af.decide_global(Bhat_t[unseen], eps_glob)
        eps = eps_glob
    elif conformal == "frozen":
        if frozen_eps is None:
            raise ValueError("--frozen-eps required for frozen conformal")
        eps = float(frozen_eps)
        dec = af.decide_global(Bhat_t, eps)
    else:
        raise ValueError(conformal)

    out = af.metrics(dec, Bt, a0t, aat)
    out["eps_global"] = eps
    out["beats_both"] = bool(out["regret_kga"] < out["regret_adapt"] and out["regret_kga"] < out["regret_freeze"])
    out["fa_ok"] = bool(out["false_adapt"] <= af.ALPHA)
    out["verdict_win"] = bool(out["beats_both"] and out["fa_ok"])
    out["margin"] = float(min(out["regret_adapt"], out["regret_freeze"]) - out["regret_kga"])
    return out


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--cal-records", nargs="+", required=True)
    p.add_argument("--test-records", nargs="+", required=True)
    p.add_argument("--candidate", required=True)
    p.add_argument("--estimator", choices=["gbr", "ppi_debias"], default="gbr")
    p.add_argument("--conformal", choices=["global", "mondrian", "frozen"], default="global")
    p.add_argument("--frozen-eps", type=float, default=None)
    p.add_argument("--output-dir", default=None)
    args = p.parse_args()

    cal = load_records(args.cal_records, args.candidate)
    test = load_records(args.test_records, args.candidate)
    if len(cal) < 5 or len(test) < 1:
        raise SystemExit(f"not enough records: cal={len(cal)} test={len(test)}")
    out = {
        "alpha": af.ALPHA,
        "candidate": args.candidate,
        "estimator": args.estimator,
        "conformal": args.conformal,
        "cal_records": args.cal_records,
        "test_records": args.test_records,
        "n_cal": len(cal),
        "n_test_records": len(test),
        "cal_seeds": sorted(set(r["seed"] for r in cal)),
        "test_seeds": sorted(set(r["seed"] for r in test)),
        "test_locked": score_transfer(cal, test, args.estimator, args.conformal, args.frozen_eps),
    }
    print(json.dumps(out, indent=2))
    if args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)
        op = Path(args.output_dir) / "holdout_score.json"
        op.write_text(json.dumps(out, indent=2))
        print(f"Saved {op}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
