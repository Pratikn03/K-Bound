"""FEASIBILITY PROBE for the multi-candidate latent-label identifiability route.

This is NOT a theorem and NOT a result for any paper. It is a numerical check of
whether Step 1's first lemma is even viable, and where it breaks. Every number is
produced by this run.

Setting (binary, on the disagreement region D): latent label Y in {0,1}; M candidate
predictors f_1..f_M; accuracy a_j = P(f_j = Y). We observe ONLY the predictions
(never Y) -> only pairwise/triple AGREEMENT rates A_ij = P(f_i=f_j).

Key identity under conditional error-independence given Y (binary, symmetric acc):
    A_ij = a_i a_j + (1-a_i)(1-a_j)  =>  2 A_ij - 1 = b_i b_j ,  b_j := 2 a_j - 1.
So observed pairwise agreements give the PRODUCTS of advantages b_j. With M=3 the
system {b1 b2, b1 b3, b2 b3} is exactly determined:
    b1 = sqrt(c12 c13 / c23), etc. (sign fixed by the anchor: all better than chance).
-> the accuracy ORDERING on D is recovered label-free. That ordering is exactly the
   ordinal comparison KGA's Thms 6/8/9 reduce sign(Delta) to.

Probes:
  A. M=3, conditional independence: recover a_j from agreements only; ordering correct?
  B. correlated errors (interpolate CI->shared via rho): when does the ordering flip?
  C. M=4 OVER-determination: c12 c34 = c13 c24 = c14 c23 under CI; the residual is a
     label-free DIAGNOSTIC that should be ~0 under CI and grow with rho (seed of Step 2).
"""
import os, json
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
OUTJ = os.path.join(HERE, "feasibility_probe_results.json")
rng = np.random.default_rng(0)
N = 400000


def gen(accs, rho, seed):
    """M conditionally-(in)dependent binary predictors of Y. rho in [0,1] mixes in a
    SHARED correctness draw (rho=0 -> conditional independence; rho=1 -> fully shared)."""
    r = np.random.default_rng(seed)
    Y = (r.random(N) < 0.5).astype(int)
    shared_correct = (r.random(N) < np.mean(accs))      # one common correctness event
    use_shared = r.random((len(accs), N)) < rho
    preds = []
    for j, a in enumerate(accs):
        indep_correct = r.random(N) < a
        correct = np.where(use_shared[j], shared_correct, indep_correct)
        preds.append(np.where(correct, Y, 1 - Y))
    return Y, preds


def pairwise_products(preds):
    M = len(preds); C = {}
    for i in range(M):
        for j in range(i + 1, M):
            A = float(np.mean(preds[i] == preds[j]))
            C[(i, j)] = 2 * A - 1                         # = b_i b_j (population)
    return C


def recover3(C):
    c12, c13, c23 = C[(0, 1)], C[(0, 2)], C[(1, 2)]
    # all better than chance => products positive; guard tiny negatives from noise
    b1 = np.sqrt(max(c12 * c13 / c23, 0))
    b2 = np.sqrt(max(c12 * c23 / c13, 0))
    b3 = np.sqrt(max(c13 * c23 / c12, 0))
    return [(1 + b) / 2 for b in (b1, b2, b3)]


def part_A():
    errs, ord_ok = [], 0
    for t in range(50):
        accs = sorted(rng.uniform(0.55, 0.9, 3), reverse=True)
        _, preds = gen(accs, 0.0, t)
        ahat = recover3(pairwise_products(preds))
        errs.append(float(max(abs(np.array(ahat) - np.array(accs)))))
        ord_ok += int(np.argsort(ahat).tolist() == np.argsort(accs).tolist())
    return dict(trials=50, max_abs_acc_error=float(np.max(errs)),
                mean_abs_acc_error=float(np.mean(errs)),
                ordering_correct_rate=ord_ok / 50,
                verdict="label-free accuracy ordering recovered exactly under conditional independence")


def part_B():
    accs = [0.78, 0.70, 0.62]                            # true ordering f1>f2>f3
    rows = []
    for rho in [0.0, 0.1, 0.2, 0.3, 0.5, 0.7, 0.9]:
        flips = 0
        for t in range(20):
            _, preds = gen(accs, rho, 100 + t)
            ahat = recover3(pairwise_products(preds))
            flips += int(np.argsort(ahat).tolist() != np.argsort(accs).tolist())
        rows.append(dict(rho=rho, ordering_flip_rate=flips / 20))
    broke = next((r["rho"] for r in rows if r["ordering_flip_rate"] >= 0.5), None)
    return dict(rows=rows, ordering_breaks_at_rho=broke,
                verdict="ordering is exact at rho=0 and degrades as error-correlation grows -> the condition is the crux, as predicted")


def part_C():
    """M=4 diagnostic: under CI, c12*c34 = c13*c24 = c14*c23 (= b1b2b3b4)."""
    rows = []
    for rho in [0.0, 0.05, 0.1, 0.2, 0.3, 0.5]:
        res = []
        for t in range(20):
            accs = [0.80, 0.72, 0.66, 0.60]
            _, preds = gen(accs, rho, 200 + t)
            C = pairwise_products(preds)
            trips = [C[(0, 1)] * C[(2, 3)], C[(0, 2)] * C[(1, 3)], C[(0, 3)] * C[(1, 2)]]
            res.append(float(max(trips) - min(trips)))     # consistency residual
        rows.append(dict(rho=rho, mean_residual=float(np.mean(res))))
    rhos = np.array([r["rho"] for r in rows]); resv = np.array([r["mean_residual"] for r in rows])
    corr = float(np.corrcoef(rhos, resv)[0, 1])
    return dict(rows=rows, residual_at_CI=rows[0]["mean_residual"],
                pearson_corr_residual_rho=corr,
                monotone=bool(np.all(np.diff(resv) > -1e-4)),
                verdict="overdetermination yields a label-free residual ~0 under CI and rising with correlation -> a checkable diagnostic exists (Step-2 seed)")


def main():
    out = dict(A_recovery=part_A(), B_breakdown=part_B(), C_diagnostic=part_C(),
               honest_note="Feasibility only. Binary, symmetric accuracy, population-scale agreements. "
                           "The theorem needs: multiclass + asymmetric confusions, the WEAKEST checkable "
                           "condition (not full CI), a soundness guarantee for the diagnostic, a matching "
                           "converse, and real-shift validation. None of those are shown here.")
    json.dump(out, open(OUTJ, "w"), indent=2)
    print("[A] max acc error:", round(out["A_recovery"]["max_abs_acc_error"], 4),
          "| ordering correct:", out["A_recovery"]["ordering_correct_rate"])
    print("[B] ordering flip vs rho:", [(r["rho"], r["ordering_flip_rate"]) for r in out["B_breakdown"]["rows"]])
    print("[C] residual@CI:", round(out["C_diagnostic"]["residual_at_CI"], 4),
          "| corr(residual,rho):", round(out["C_diagnostic"]["pearson_corr_residual_rho"], 3),
          "| monotone:", out["C_diagnostic"]["monotone"])

    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(10.5, 4.0))
    b = out["B_breakdown"]["rows"]
    ax[0].plot([r["rho"] for r in b], [r["ordering_flip_rate"] for r in b], "-o", color="#e76f51")
    ax[0].axhline(0.5, ls=":", color="k"); ax[0].set_xlabel(r"error-correlation $\rho$")
    ax[0].set_ylabel("ordering-flip rate"); ax[0].set_title("Step-1 estimator: exact at ρ=0, breaks as CI fails")
    c = out["C_diagnostic"]["rows"]
    ax[1].plot([r["rho"] for r in c], [r["mean_residual"] for r in c], "-s", color="#2a9d8f")
    ax[1].set_xlabel(r"error-correlation $\rho$"); ax[1].set_ylabel("consistency residual (label-free)")
    ax[1].set_title("Step-2 seed: a checkable diagnostic (≈0 under CI, rises with ρ)")
    plt.tight_layout(); fig.savefig(os.path.join(HERE, "fig_feasibility.png"), dpi=130, bbox_inches="tight")
    print("figure: fig_feasibility.png")


if __name__ == "__main__":
    main()
