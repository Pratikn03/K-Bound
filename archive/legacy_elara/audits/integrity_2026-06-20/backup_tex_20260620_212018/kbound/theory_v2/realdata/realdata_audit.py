#!/usr/bin/env python3
"""
realdata_audit.py  --  First REAL-DATA validation of K-Bound Theory V2.

P1  Anomaly-bank panels (123-task ADBench-style score archive, 6 detectors):
    binarize detector scores -> predictions; define region D where designated
    candidate f_a disagrees with best-val f_0; on D compute pairwise agreements
    c_ij, the >=4-minor spread tau, product-ratio advantage recovery b_hat with
    majority anchor; compare recovered sign(b_a - b_0) and |b| against TRUE values
    from test labels (labels used ONLY to score, never to fit).

P2  Per-condition paired bootstrap + Holm on the logged CIFAR-10-C grid
    (cifar10c_65cells.csv: per-cell accuracy for frozen/tent/eata/sar/kga/oracle),
    replacing the flagged synthetic-stream Pareto bootstrap.

Honest integrity: report whatever comes out, including failures. Detector
correlation violating H is EXPECTED -- it is the diagnostic (tau, gamma) working.

Author: K-Bound theory_v2 real-data agent. CPU-only. Seeds fixed.
"""
import json, os, glob, csv
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ARCHIVE = "/Volumes/T9/uav/AutoML_Flagship_V8/experiments/elara_u/score_archive"
TTA_CSV = "/Volumes/T9/uav/AutoML_Flagship_V8/experiments/kbound/results/cifar10c_65cells.csv"
TTA_CIS = "/Volumes/T9/uav/AutoML_Flagship_V8/experiments/kbound/results/decisive_tta_cis.json"
OUT_JSON = os.path.join(HERE, "realdata_audit_results.json")
RNG = np.random.default_rng(20260610)

# ----------------------------------------------------------------- panel helpers
def agreements(F):
    K = F.shape[0]
    A = np.empty((K, K))
    for i in range(K):
        for j in range(K):
            A[i, j] = np.mean(F[i] == F[j])
    return A

def recover_b_up_to_flip(c, K, anchor_sign=+1):
    """Product-ratio |b| with median over valid (k,l); relative signs from sign(c[0,i]);
    global flip anchored by majority-above-chance (sum b > 0). Mirrors reference impl."""
    b2 = np.zeros(K)
    for i in range(K):
        vals = []
        for k in range(K):
            for l in range(K):
                if len({i, k, l}) == 3 and abs(c[k, l]) > 1e-9:
                    vals.append(c[i, k] * c[i, l] / c[k, l])
        b2[i] = np.median(vals) if vals else np.nan
    b2 = np.clip(b2, 0.0, None)
    mag = np.sqrt(b2)
    s = np.ones(K)
    for i in range(1, K):
        s[i] = np.sign(c[0, i]) if abs(c[0, i]) > 0 else 1.0
    b = s * mag
    if np.sign(np.sum(b)) != anchor_sign and np.sum(b) != 0:
        b = -b
    return b

def tau_minors(c):
    """Spread of the three 2x2-minor products over a chosen 4-subset (indices 0,1,2,3).
    tau = max(prods) - min(prods) where prods are the three c_ij c_kl pairings.
    Falsifies H (rank-1) when > 0 beyond sampling noise (Cor 0.3 / T-II)."""
    pr = [c[0, 1] * c[2, 3], c[0, 2] * c[1, 3], c[0, 3] * c[1, 2]]
    return float(max(pr) - min(pr)), pr

def binarize(S, thresholds):
    """predictions: 1 (anomaly) if score >= threshold (per detector)."""
    return (S >= thresholds[None, :]).astype(np.int8).T   # -> (K, N)

def choose_thresholds(Sval, yval, mode="val_opt"):
    """Per detector: validation-optimal Youden-J threshold (maximize TPR-FPR) if val_opt,
    else median of validation scores."""
    K = Sval.shape[1]
    th = np.zeros(K)
    for j in range(K):
        s = Sval[:, j]
        if mode == "median" or yval is None or len(np.unique(yval)) < 2:
            th[j] = np.median(s)
            continue
        order = np.argsort(s)
        ss = s[order]; yy = yval[order]
        P = yy.sum(); N = len(yy) - P
        # candidate thresholds = unique score values; sweep
        uniq = np.unique(ss)
        best_j, best_t = -1.0, np.median(s)
        # cumulative counts of positives/negatives at score>=t
        for t in uniq:
            pred = (s >= t)
            tp = np.sum(pred & (yval == 1)); fp = np.sum(pred & (yval == 0))
            tpr = tp / P if P > 0 else 0.0
            fpr = fp / N if N > 0 else 0.0
            J = tpr - fpr
            if J > best_j:
                best_j, best_t = J, t
        th[j] = best_t
    return th

# ----------------------------------------------------------------- P1 per task
def audit_task(path, thr_mode="val_opt", min_D=40, boot=300):
    d = np.load(path, allow_pickle=True)
    Sval, yval = d["Sval"], d["yval"].astype(int)
    Stest, ytest = d["Stest"], d["ytest"].astype(int)
    det = [str(x) for x in d["det_names"]]
    val_auc = np.asarray(d["val_auc"], float)
    K = Sval.shape[1]
    name = os.path.basename(path).replace(".npz", "")

    if len(np.unique(ytest)) < 2 or len(np.unique(yval)) < 2:
        return {"task": name, "skipped": "test/val single-class"}

    th = choose_thresholds(Sval, yval, thr_mode)
    Ftest = binarize(Stest, th)             # (K, N) predictions on TEST region (unlabeled use)
    # f0 = best-val-AUC detector; reorder so index 0 = f0
    i0 = int(np.argmax(val_auc))
    # f_a = designated candidate = SECOND-best val detector (a genuine alternative predictor)
    order = np.argsort(val_auc)[::-1]
    ia = int(order[1])
    # Build panel index order: [f0, f_a, then the rest] so recover anchor index0=f0
    rest = [j for j in range(K) if j not in (i0, ia)]
    perm = [i0, ia] + rest
    Fp = Ftest[perm]                        # reordered predictions (K,N)
    detp = [det[j] for j in perm]
    aucp = val_auc[perm]

    # region D = where f_a disagrees with f0 (indices 1 vs 0 in permuted panel)
    Dmask = (Fp[1] != Fp[0])
    nD = int(Dmask.sum())
    if nD < min_D:
        return {"task": name, "skipped": f"|D|={nD} < {min_D}"}

    FD = Fp[:, Dmask]                        # predictions restricted to D
    yD = ytest[Dmask]                        # TRUE labels on D  (GROUND-TRUTH SCORING ONLY)
    piD = float(yD.mean())

    # ---- label-free estimands on D (no labels) ----
    A = agreements(FD); c = 2 * A - 1
    tau, prods = tau_minors(c) if K >= 4 else (np.nan, None)
    b_hat = recover_b_up_to_flip(c, K, anchor_sign=+1)   # majority-above-chance anchor
    sign_rec = int(np.sign(b_hat[1] - b_hat[0]))         # recovered sign(b_a - b_0)

    # ---- TRUE advantages on D from labels (scoring only) ----
    Ctrue = (FD == yD[None, :]).astype(np.int8)          # correctness vs true labels
    a_true = Ctrue.mean(1)                                # marginal accuracy on D
    b_true = 2 * a_true - 1                               # true advantage b_j on D
    sign_true = int(np.sign(b_true[1] - b_true[0]))

    # |b| recovery error (identification is up to global flip: compare both signs, take min)
    err_b = float(np.max(np.abs(np.abs(b_hat) - np.abs(b_true))))
    # sign recovery correctness (the bit): theory recovers sign only up to the global flip;
    # we score the RELATIVE sign sign(b_a - b_0), which the anchor is meant to fix.
    sign_ok = bool(sign_rec == sign_true) if sign_true != 0 else None

    # ---- H falsification: bootstrap threshold on tau ----
    # Null reference: resample D indices, recompute tau; H-reject if observed tau exceeds
    # the (1-alpha) quantile of a CEI-consistent null. We build the null by the rank-1
    # PROJECTION surrogate: simulate K independent CEI predictors with the recovered |b|
    # (per-class symmetric), i.e. the best-fitting H-model, and bootstrap its tau.
    H_reject = None; tau_null_q95 = None; tau_boot_p = None
    if K >= 4 and np.all(np.isfinite(b_hat)):
        # fit per-class-symmetric q_j = (1+|b_hat_j|)/2 ; CEI panel of size nD ; many draws
        q = np.clip((1 + np.abs(b_hat)) / 2.0, 0.51, 0.99)
        taus_null = np.empty(boot)
        for bi in range(boot):
            rng = np.random.default_rng(1000 + bi)
            Yb = (rng.random(nD) < piD).astype(np.int8)
            Cb = (rng.random((K, nD)) < q[:, None]).astype(np.int8)  # CEI by construction
            Fb = np.where(Cb == 1, Yb[None, :], 1 - Yb[None, :])
            Ab = agreements(Fb); cb = 2 * Ab - 1
            taus_null[bi], _ = tau_minors(cb)
        tau_null_q95 = float(np.quantile(taus_null, 0.95))
        tau_boot_p = float((np.sum(taus_null >= tau) + 1) / (boot + 1))
        H_reject = bool(tau > tau_null_q95)

    # ---- gamma audit calibration ----
    # gamma_hat = b_hat_a/2 - M_hat ; here the observable surrogate s is the candidate's own
    # correctness proxy. With no separate drift surrogate available on D, we set
    # M_hat := bar_a_a - 1/2 estimated WITHOUT labels via b_hat: M_hat = b_hat_a/2  => gamma_hat=0
    # is the trivial identity. To make gamma INFORMATIVE we instead define the audit's true
    # drift as the gap between recovered and true candidate advantage on D:
    #   gamma_true := b_true_a/2 - b_hat_a/2   (the part of the budget the evidence channel misses)
    # and report whether |gamma_true| is small (good calibration when true drift small).
    gamma_true = float(b_true[1] / 2.0 - np.abs(b_hat[1]) / 2.0 * sign_rec * sign_rec)
    # (kept simple: difference of true vs recovered candidate half-advantage)
    gamma_recoverable_gap = float(abs(b_true[1] - sign_rec * abs(b_hat[1])) / 2.0)

    return {
        "task": name, "domain": str(d["domain"]), "K": K, "detectors": detp,
        "val_auc_perm": aucp.round(4).tolist(),
        "n_test": int(len(ytest)), "n_D": nD, "pi_D": round(piD, 4),
        "f0": detp[0], "f_a": detp[1],
        "tau": (round(tau, 6) if np.isfinite(tau) else None),
        "tau_null_q95": (round(tau_null_q95, 6) if tau_null_q95 is not None else None),
        "tau_boot_p": (round(tau_boot_p, 4) if tau_boot_p is not None else None),
        "H_reject": H_reject,
        "b_hat": b_hat.round(4).tolist(), "b_true_D": b_true.round(4).tolist(),
        "sign_rec_ba_minus_b0": sign_rec, "sign_true_ba_minus_b0": sign_true,
        "sign_ok": sign_ok,
        "err_abs_b_max": round(err_b, 4),
        "gamma_recoverable_gap": round(gamma_recoverable_gap, 4),
    }

if __name__ == "__main__":
    print("realdata_audit module ready")
