#!/usr/bin/env python3
"""kga_elara_demo.py -- KGA x ELARA on REAL cached multimodal anomaly scores.

ELARA = reliability-gated multimodal score fusion (the mechanism / ceiling-raiser).
KGA   = the K-Bound adapt/freeze/abstain certificate (the safety floor) -- using the
        ACTUAL kga/ package: kga.certificate.empirical_bernstein + kga.policy.decide.

For each multimodal track we load the real per-category .npz caches
  experiments/fusion/<track>_score_cache/*.npz
each holding  Sval[n,M], yval[n], Stest[n,M], ytest[n], valauc[M]  -- the inference
artifacts behind ELARA's published AUROCs (M = co-observed modalities). We then:
  (1) per-modality test AUROC (are the modalities complementary?),
  (2) ELARA fusion AUROC -- confidence-weighted (CW) and reliability-gated -- the CEILING,
  (3) KGA certificate on the per-sample placement-benefit of fusion over the best
      VAL-chosen single modality, pooled across categories -> ADAPT/FREEZE/ABSTAIN,
  (4) a selective (risk-coverage) curve for the KGA abstain action.

Honest by construction: every AUROC is computed from the real cached scores with
sklearn; the best single modality is chosen by VALIDATION auc (no test-label peeking);
nothing is tuned; tracks that do not reach 90 are reported as such; KGA abstains where
the benefit is not certifiable at level alpha.
"""
import os, sys, glob, json
import numpy as np
from sklearn.metrics import roc_auc_score
from scipy.stats import ks_2samp

REPO = "/Volumes/T9/uav/AutoML_Flagship_V8"
sys.path.insert(0, REPO)
from kga.certificate import empirical_bernstein      # the user's actual KGA code
from kga.policy import decide
from kga import KGA

ALPHA = 0.10
PROBE_K = 32  # Protocol D24 default deployment probe size
TRACKS = {
    "3D-ADAM":         ("experiments/fusion/3d_adam_score_cache",        "*.npz"),
    "Real-IAD-D3":     ("experiments/fusion/realiad_d3_score_cache",     "*_v2_binpcd.npz"),
    "MulSen-AD":       ("experiments/fusion/mulsen_score_cache",         "*.npz"),
    "MVTec-3D":        ("experiments/fusion/mvtec3d_score_cache",        "*.npz"),
    "Real-IAD-NatDeg": ("experiments/fusion/realiad_natdeg_score_cache", "*.npz"),
}


def auroc(y, s):
    y = np.asarray(y); s = np.asarray(s)
    if len(np.unique(y)) < 2:           # one-class category -> AUROC undefined
        return np.nan
    return float(roc_auc_score(y, s))

def cw_fuse(S):                          # ELARA confidence-weighted fusion (parameter-free)
    w = 2.0 * np.abs(S - 0.5)
    return (S * w).sum(1) / np.clip(w.sum(1), 1e-9, None)

def relgate_fuse(Sval, Stest, valauc):   # ELARA reliability-gated weighted fusion
    val_ok = (valauc >= 0.55) & (Sval.std(0) >= 0.02)
    drift = np.array([ks_2samp(Sval[:, m], Stest[:, m]).statistic for m in range(Sval.shape[1])])
    keep = val_ok & (drift <= 0.25)
    if keep.sum() == 0: keep = val_ok.copy()
    if keep.sum() == 0: keep = np.ones(Sval.shape[1], bool)
    w = np.clip(valauc[keep] - 0.5, 1e-6, None)
    return Stest[:, keep] @ (w / w.sum())

def placements(y, s):
    """Per-positive placement = frac of negatives it outranks (+0.5 for ties).
       mean(placements) == AUROC exactly, so placement-difference is a per-sample benefit
       whose mean equals the AUROC improvement (Mann-Whitney decomposition)."""
    y = np.asarray(y); s = np.asarray(s)
    pos, neg = s[y == 1], s[y == 0]
    if len(pos) == 0 or len(neg) == 0:
        return np.array([])
    ns = np.sort(neg)
    below = np.searchsorted(ns, pos, side="left")
    ties = np.searchsorted(ns, pos, side="right") - below
    return (below + 0.5 * ties) / len(neg)

def selective_auc(y, fused, coverages):
    """KGA abstain action at sample level: keep the most-confident fraction (confidence =
       |rank-normalised fused score - 0.5|), report AUROC on the retained region."""
    y = np.asarray(y); fused = np.asarray(fused); n = len(y)
    order = np.argsort(fused); rank = np.empty(n); rank[order] = np.arange(n)
    conf = np.abs(rank / max(n - 1, 1) - 0.5)
    keep_order = np.argsort(-conf)                  # most confident first
    out = {}
    for c in coverages:
        k = max(2, int(round(c * n)))
        idx = keep_order[:k]
        out[c] = auroc(y[idx], fused[idx])
    return out


def run_track(name, cache_dir, pattern):
    files = sorted(glob.glob(os.path.join(REPO, cache_dir, pattern)))
    rows = []
    for f in files:
        z = np.load(f)
        Sval, yval, Stest, ytest, valauc = z["Sval"], z["yval"], z["Stest"], z["ytest"], z["valauc"]
        if len(np.unique(ytest)) < 2:               # skip degenerate (single-class) cats
            continue
        M = Sval.shape[1]
        permod = [auroc(ytest, Stest[:, m]) for m in range(M)]
        best_val_m = int(np.nanargmax(valauc))       # deployable: pick modality by VAL auc
        base = Stest[:, best_val_m]
        cw = cw_fuse(Stest); rg = relgate_fuse(Sval, Stest, valauc)
        rows.append(dict(cat=os.path.basename(f), M=M, permod=permod,
                         best_single=auroc(ytest, base), cw=auroc(ytest, cw), rg=auroc(ytest, rg),
                         deployable=bool((valauc > 0.6).sum() >= 2),
                         pb=(placements(ytest, cw) - placements(ytest, base)),
                         sel=selective_auc(ytest, cw, [1.0, 0.9, 0.8])))
    dep = [r for r in rows if r["deployable"]]       # ELARA protocol: >=2 informative modalities
    use = dep if dep else rows
    def mean(key): return float(np.nanmean([r[key] for r in use]))
    cw_all = float(np.nanmean([r["cw"] for r in rows]))     # unfiltered, for honesty
    pool = np.concatenate([r["pb"] for r in use]) if use else np.array([])
    cert_lf = empirical_bernstein(pool, alpha=ALPHA, benefit_range=2.0) if pool.size >= 2 else None
    dec_lf = str(decide(cert_lf)) if cert_lf is not None else "ABSTAIN(no-data)"
    # Target-label-light probe path (Protocol D24): subsample k labels from test pool
    cert_probe = None
    dec_probe = None
    if pool.size >= 2:
        kga = KGA(alpha=ALPHA)
        cert_probe = kga.certify_probe(pool, k=min(PROBE_K, pool.size), seed=20260615, benefit_range=2.0)
        dec_probe = str(kga.decide(cert_probe))
    cert = cert_probe or cert_lf
    dec = dec_probe or dec_lf
    sel = {c: float(np.nanmean([r["sel"][c] for r in use])) for c in [1.0, 0.9, 0.8]}
    return dict(name=name, n_all=len(rows), n_dep=len(dep), M=use[0]["M"] if use else 0,
                best_single=mean("best_single"), cw=mean("cw"), rg=mean("rg"), cw_allcats=cw_all,
                cert=dict(n=int(cert.n), delta=cert.delta_hat, eps=cert.epsilon,
                          lo=cert.lower, hi=cert.upper) if cert else None,
                cert_labelfree=dict(n=int(cert_lf.n), delta=cert_lf.delta_hat, eps=cert_lf.epsilon,
                                    lo=cert_lf.lower, hi=cert_lf.upper) if cert_lf else None,
                decision=dec, decision_labelfree=dec_lf, probe_k=PROBE_K, selective=sel)

def main():
    print("KGA x ELARA -- real cached multimodal anomaly scores (alpha=%.2f)" % ALPHA)
    print("ELARA = reliability-gated fusion (ceiling); KGA = kga/ certificate (floor).\n")
    OUT = {}
    for name, (d, p) in TRACKS.items():
        r = run_track(name, d, p); OUT[name] = r
        c = r["cert"]
        print(f"== {name} ==  ({r['n_dep']}/{r['n_all']} deployable cats, M={r['M']} modalities)")
        print(f"   per-modality best single (val-chosen) : {r['best_single']:.4f}")
        print(f"   ELARA fusion  CW (deployable cats)     : {r['cw']:.4f}   [all cats: {r['cw_allcats']:.4f}]")
        print(f"   ELARA fusion  reliability-gated        : {r['rg']:.4f}")
        print(f"   >= 0.90 ?  {'YES' if r['cw'] >= 0.90 or r['rg'] >= 0.90 else 'no'}")
        if c:
            print(f"   KGA certificate (fusion vs best single modality, n={c['n']} samples):")
            print(f"      delta_hat={c['delta']:+.4f}  eps={c['eps']:.4f}  "
                  f"LB={c['lo']:+.4f}  UB={c['hi']:+.4f}  ->  {r['decision']}")
        print(f"   selective AUROC @cov 1.0/0.9/0.8 = "
              f"{r['selective'][1.0]:.4f}/{r['selective'][0.9]:.4f}/{r['selective'][0.8]:.4f}\n")
    # headline
    print("=" * 70)
    print("HEADLINE: tracks where KGA x ELARA fusion reaches 90+ AUROC")
    for name, r in OUT.items():
        ceil = max(r["cw"], r["rg"]); ok90 = ceil >= 0.90
        print(f"  {name:16s} fusion AUROC={ceil:.4f}  {'>=90 ' if ok90 else ' <90 '}"
              f" KGA={r['decision']}")
    with open(os.path.join(REPO, "experiments/kbound/results_kga_elara.json"), "w") as fh:
        json.dump(OUT, fh, indent=2, default=float)
    print("\nwrote experiments/kbound/results_kga_elara.json")

if __name__ == "__main__":
    main()
