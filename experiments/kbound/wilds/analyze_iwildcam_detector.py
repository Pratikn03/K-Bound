"""
analyze_iwildcam_detector.py - compare label-free HARM DETECTORS for K-Bound iWildCam.

Same source-calibrated certificate machinery for every detector (fit B-hat on SOURCE +
conformal eps, apply to TARGET, decide ADAPT/FREEZE/ABSTAIN); ONLY the label-free input
signal changes:
   entropy_Z11      : the 11-d entropy/confidence evidence vector  (BASELINE)
   entropy_marginalKL: best single entropy feature
   aetta_dacc       : AETTA-style dropout accuracy-estimate drop  (NEW detector)
   frozen_ref_dacc  : frozen-reference disagreement accuracy proxy (NEW detector)
   aetta_plus_Z     : entropy Z augmented with aetta_dacc
Deployed adapter is chosen on SOURCE.  Detector threshold (conformal eps) is calibrated
on SOURCE.  Target labels touch ONLY final scoring.  -> honest detector swap, no test fit.

Reports per detector: target harm-AUC (signal + certificate), harmful-recall (the
detectability rate on KNOWN-HARMFUL target conditions), KGA regret vs always-adapt /
always-freeze, false-adapt count, beats_both.
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
import sys
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import analyze_iwildcam_kbound as AZ

EV = ["pre_entropy", "pre_conf", "pre_pbal", "post_entropy", "post_conf", "post_pbal",
      "pbal_drop", "entropy_drop", "frac_highconf", "marginal_KL", "update_norm"]
MKL = EV.index("marginal_KL")


def feats(recs, name):
    Z = np.array([r["Z"] for r in recs], float)
    da = np.array([[r["dacc_aetta"]] for r in recs], float)
    dr = np.array([[r["dacc_ref"]] for r in recs], float)
    if name == "entropy_Z11":
        return Z
    if name == "entropy_marginalKL":
        return Z[:, MKL:MKL + 1]
    if name == "aetta_dacc":
        return da
    if name == "frozen_ref_dacc":
        return dr
    if name == "aetta_plus_Z":
        return np.hstack([Z, da])
    raise ValueError(name)


def run_detector(name, src, tgt, alpha):
    Xs, Bs = feats(src, name), np.array([r["B"] for r in src])
    Xt, Bt = feats(tgt, name), np.array([r["B"] for r in tgt])
    a0t = np.array([r["a0"] for r in tgt]); aat = np.array([r["aa"] for r in tgt])
    model, eps = AZ.fit_certificate(Xs, Bs, alpha)
    Bhat = model.predict(Xt)
    dec = np.where(Bhat - eps > 0, "ADAPT", np.where(Bhat + eps < 0, "FREEZE", "ABSTAIN"))
    pm = AZ.policy(dec, a0t, aat, Bt)
    harm = (Bt < 0).astype(int)
    pm["certificate_harm_AUC_target"] = AZ._auc(-Bhat, harm) if harm.sum() not in (0, len(harm)) else None
    # raw-signal harm AUC (source-oriented): use predicted Bhat as the benefit estimate
    pm["harmful_recall_flagged"] = float(np.mean(Bhat[harm == 1] < 0)) if harm.sum() else None
    pm["helpful_recall_flagged"] = float(np.mean(Bhat[harm == 0] > 0)) if (harm == 0).sum() else None
    # source detectability (legitimate, calibration side)
    harm_s = (Bs < 0).astype(int)
    from sklearn.model_selection import KFold
    Bhat_s = np.zeros(len(Bs))
    if len(Bs) >= 4:
        for tr, te in KFold(n_splits=min(5, len(Bs)), shuffle=True, random_state=0).split(Xs):
            from sklearn.ensemble import GradientBoostingRegressor
            mm = GradientBoostingRegressor(n_estimators=250, max_depth=2, learning_rate=0.05,
                                           subsample=0.8, random_state=0).fit(Xs[tr], Bs[tr])
            Bhat_s[te] = mm.predict(Xs[te])
    pm["certificate_harm_AUC_source"] = AZ._auc(-Bhat_s, harm_s) if harm_s.sum() not in (0, len(harm_s)) else None
    pm["eps_source"] = float(eps)
    return pm


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", required=True)
    ap.add_argument("--target", required=True)
    ap.add_argument("--alpha", type=float, default=0.10)
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    so, sr, sc = AZ.load_manifest(args.source)
    to, tr, tc = AZ.load_manifest(args.target)
    src = AZ.recompute(sr, sc, "macro_f1")
    tgt = AZ.recompute(tr, tc, "macro_f1")
    cand_src = {c: float(np.mean([r["aa"] for r in src if r["candidate"] == c]))
                for c in sorted(set(r["candidate"] for r in src))}
    deployed = max(cand_src, key=cand_src.get)
    s_dep = [r for r in src if r["candidate"] == deployed]
    t_dep = [r for r in tgt if r["candidate"] == deployed]
    Bt = np.array([r["B"] for r in t_dep])
    base = {"deployed_adapter": deployed, "source_mean_F1_by_candidate": cand_src,
            "n_source_dep": len(s_dep), "n_target_dep": len(t_dep),
            "target_base_rate_harmful": float(np.mean(Bt < 0)),
            "target_frac_helpful_B>0.02": float(np.mean(Bt > 0.02)),
            "target_mean_B": float(Bt.mean()), "target_B_range": [float(Bt.min()), float(Bt.max())]}

    detectors = ["entropy_Z11", "entropy_marginalKL", "aetta_dacc", "frozen_ref_dacc", "aetta_plus_Z"]
    results = {d: run_detector(d, s_dep, t_dep, args.alpha) for d in detectors}

    print("=" * 96)
    print(f"DEPLOYED (source-chosen) = {deployed} | target harmful={base['target_base_rate_harmful']:.3f} "
          f"helpful={base['target_frac_helpful_B>0.02']:.3f} meanB={base['target_mean_B']:+.4f} "
          f"B_range=[{Bt.min():+.3f},{Bt.max():+.3f}]")
    print(f"{'detector':18s} {'srcAUC':>7s} {'tgtAUC':>7s} {'harmRec':>7s} {'KGAreg':>7s} {'adaptReg':>8s} "
          f"{'freezeReg':>9s} {'falseAdp':>8s} {'beats_both':>10s}")
    for d in detectors:
        r = results[d]
        rv = r["regret_vs_oracle"]
        print(f"{d:18s} {str(round(r['certificate_harm_AUC_source'] or 0,3)):>7s} "
              f"{str(round(r['certificate_harm_AUC_target'] or 0,3)):>7s} "
              f"{str(round(r['harmful_recall_flagged'] or 0,3)):>7s} "
              f"{rv['K_Bound']:7.4f} {rv['always_adapt']:8.4f} {rv['always_freeze']:9.4f} "
              f"{r['false_adapt_count']:8d} {str(r['beats_both']):>10s}")

    verdict = {"schema": "kbound_iwildcam_detector_compare_v1", "metric": "macro_f1", "alpha": args.alpha,
               "source": args.source, "target": args.target, **base, "detectors": results}
    out = args.out or str(Path(args.target).parent / "VERDICT_detector.json")
    json.dump(verdict, open(out, "w"), indent=2)
    print(f"verdict -> {out}")


if __name__ == "__main__":
    main()
