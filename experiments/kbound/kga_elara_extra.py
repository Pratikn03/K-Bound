#!/usr/bin/env python3
"""kga_elara_extra.py -- two ADDITIONAL real tracks for the KGA x ELARA demo,
loaded from CSV score files (the .npz caches are covered by kga_elara_demo.py).

Same pipeline, same kga/ certificate, same comparator (fusion vs best VAL-chosen
single modality). Honest labels:
  * healthcare   = genuine MULTIMODAL (4 physiological channels), real but weak (~0.70).
  * realiad-2det = same RGB modality, TWO detectors (CW + SAR): multi-DETECTOR, not
                   multimodal; included for completeness and labeled as such.
No dataset is selected to force a decision; decisions are whatever the data gives.
"""
import os, sys, json
import numpy as np, pandas as pd
REPO = "/Volumes/T9/uav/AutoML_Flagship_V8"
sys.path.insert(0, REPO); sys.path.insert(0, os.path.join(REPO, "experiments/kbound"))
from kga_elara_demo import auroc, cw_fuse, relgate_fuse, placements, selective_auc, ALPHA
from kga.certificate import empirical_bernstein
from kga.policy import decide

def ranknorm(fit, x):
    """label-free per-modality map to [0,1] via the empirical CDF of the fit (val) scores."""
    s = np.sort(np.asarray(fit, float))
    return np.searchsorted(s, np.asarray(x, float), side="right") / max(len(s), 1)

def _binlabel(v):
    v = np.asarray(v)
    return v if set(np.unique(v)) <= {0, 1} else (v != 0).astype(int)


def load_healthcare():
    df = pd.read_csv(os.path.join(REPO, "experiments/fusion/healthcare_paired_inputs.csv"))
    doms = sorted(df["domain"].unique())
    def pivot(split):
        g = df[df.fusion_split == split]
        piv = g.pivot_table(index="sample_id", columns="domain", values="score", aggfunc="mean").reindex(columns=doms)
        lab = g.groupby("sample_id")["label"].first().reindex(piv.index)
        keep = piv.notna().all(axis=1) & lab.notna()
        piv, lab = piv[keep], lab[keep]
        return piv.values.astype(float), _binlabel(lab.values)
    Sv_raw, yv = pivot("validation"); St_raw, yt = pivot("test")
    Sv = np.column_stack([ranknorm(Sv_raw[:, m], Sv_raw[:, m]) for m in range(Sv_raw.shape[1])])
    St = np.column_stack([ranknorm(Sv_raw[:, m], St_raw[:, m]) for m in range(St_raw.shape[1])])
    vauc = np.array([auroc(yv, Sv[:, m]) for m in range(Sv.shape[1])])
    return ("healthcare (MULTIMODAL: %d physio channels)" % len(doms), Sv, yv, St, yt, vauc)

def load_realiad_2det():
    a = pd.read_csv(os.path.join(REPO, "experiments/fusion/realiad_256_c1_c2_d13_cw_scores.csv"))
    b = pd.read_csv(os.path.join(REPO, "experiments/fusion/realiad_256_c1_c2_d13_sar_scores.csv"))
    m = a.merge(b, on="sample_id", suffixes=("_cw", "_sar"))
    y = m["sample_id"].str.contains("__NG__").astype(int).values
    S = np.column_stack([m["raw_score_cw"].values, m["raw_score_sar"].values]).astype(float)
    # deterministic, label-free 50/50 val/test split by a stable hash of sample_id
    h = (m["sample_id"].apply(lambda s: hash(s) % 1000)).values
    val = h < 500; te = ~val
    Sv_raw, St_raw = S[val], S[te]; yv, yt = y[val], y[te]
    Sv = np.column_stack([ranknorm(Sv_raw[:, k], Sv_raw[:, k]) for k in range(2)])
    St = np.column_stack([ranknorm(Sv_raw[:, k], St_raw[:, k]) for k in range(2)])
    vauc = np.array([auroc(yv, Sv[:, k]) for k in range(2)])
    return ("realiad-2det (multi-DETECTOR: CW+SAR, same RGB modality)", Sv, yv, St, yt, vauc)


def run_one(loader):
    name, Sval, yval, Stest, ytest, valauc = loader()
    M = Sval.shape[1]
    permod = [auroc(ytest, Stest[:, m]) for m in range(M)]
    best_val_m = int(np.nanargmax(valauc)); base = Stest[:, best_val_m]
    cw = cw_fuse(Stest); rg = relgate_fuse(Sval, Stest, valauc)
    cw_auc, rg_auc, base_auc = auroc(ytest, cw), auroc(ytest, rg), auroc(ytest, base)
    pb = placements(ytest, cw) - placements(ytest, base)          # per-sample benefit of fusion
    cert = empirical_bernstein(pb, alpha=ALPHA, benefit_range=2.0) if pb.size >= 2 else None
    dec = str(decide(cert)) if cert is not None else "ABSTAIN(no-data)"
    sel = selective_auc(ytest, cw, [1.0, 0.9, 0.8])
    print("== %s ==  (M=%d, n_test=%d)" % (name, M, len(ytest)))
    print("   per-modality test AUROC : " + ", ".join("%.3f" % a for a in permod))
    print("   best single (val-chosen): %.4f" % base_auc)
    print("   ELARA fusion CW / relgate: %.4f / %.4f   (>=0.90? %s)"
          % (cw_auc, rg_auc, "YES" if max(cw_auc, rg_auc) >= 0.90 else "no"))
    if cert:
        print("   KGA cert (fusion vs best single, n=%d): delta=%+.4f eps=%.4f LB=%+.4f UB=%+.4f -> %s"
              % (cert.n, cert.delta_hat, cert.epsilon, cert.lower, cert.upper, dec))
    print("   selective AUROC @1.0/0.9/0.8 = %.3f/%.3f/%.3f\n" % (sel[1.0], sel[0.9], sel[0.8]))
    return dict(name=name, M=M, n_test=int(len(ytest)), permod=permod, best_single=base_auc,
                cw=cw_auc, rg=rg_auc, decision=dec,
                cert=dict(n=int(cert.n), delta=cert.delta_hat, eps=cert.epsilon,
                          lo=cert.lower, hi=cert.upper) if cert else None, selective=sel)

def main():
    print("KGA x ELARA -- ADDITIONAL real tracks (alpha=%.2f); decisions are whatever the data gives.\n" % ALPHA)
    OUT = {}
    for loader in (load_healthcare, load_realiad_2det):
        r = run_one(loader); OUT[r["name"].split()[0]] = r
    with open(os.path.join(REPO, "experiments/kbound/results_kga_elara_extra.json"), "w") as fh:
        json.dump(OUT, fh, indent=2, default=float)
    print("wrote experiments/kbound/results_kga_elara_extra.json")

if __name__ == "__main__":
    main()
