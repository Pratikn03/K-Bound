"""Conjecture 1 under bounded calibration transfer -- numerical validation.

Validates Theorem ``thm:caltransfer`` in
docs/research/kbound/paper/sections/theory_extensions_v2.tex. Does NOT modify the paper.

------------------------------------------------------------------------------
SETTING (from the host paper, Thm thm:disagree-mc / conj:gen)
------------------------------------------------------------------------------
On the observable disagreement region D = {x : f0(x) != fa(x)}, the paper reduces
sign(Delta) to an ordinal accuracy comparison

        Delta = P_T(D) * (p_a - p_0),   sign(Delta) = sign(p_a - p_0),
        p_a = P_T(fa = Y | D),   p_0 = P_T(f0 = Y | D).

For K >= 3 this is two-sided (p_0 != 1 - p_a). The OPEN piece (Conjecture 1) is to
bracket p_a vs p_0 WITHOUT target labels.

------------------------------------------------------------------------------
CALIBRATION-TRANSFER ASSUMPTION (the conditional hypothesis we test)
------------------------------------------------------------------------------
A confidence map conf_h(x) in [0,1] is attached to each predictor h in {f0, fa}
(e.g. a source-calibrated softmax-max / reliability score). On D the map is
eta-miscalibrated on the target:

        | P_T( h(X) = Y | conf_h(X) = c, X in D ) - c | <= eta   for all c.        (CT)

Define the label-free confidence accuracy estimates

        q_h := E_T[ conf_h(X) | X in D ]   (purely observable: average target
                                            confidence on D, no labels).

CLAIM (Thm thm:caltransfer):
    Under (CT),   | q_h - p_h | <= eta   for h in {f0, fa},   hence
        (p_a - p_0)  in  [ (q_a - q_0) - 2 eta ,  (q_a - q_0) + 2 eta ].
    Therefore the confidence gap brackets the accuracy gap, and
        gap := q_a - q_0,    |gap| > 2 eta   ==>   sign(p_a - p_0) = sign(gap),
    so sign(Delta) is LABEL-FREE IDENTIFIABLE.

TIGHTNESS (ties to thm:imp): when eta >= |gap| the bracket straddles 0; the regime
re-enters the unknowable region and the certificate must abstain. We expect
sign-recovery ~ 100% while |gap| > 2 eta and decay to chance (50%) as eta grows past
|gap|.

------------------------------------------------------------------------------
SIMULATION DESIGN
------------------------------------------------------------------------------
We synthesize, per trial, a K-class disagreement region with controllable true
accuracies (p_a, p_0) and a CONFIDENCE MAP whose target miscalibration is bounded by
eta (we draw a per-sample calibration error uniformly in [-eta, eta] and add it to the
true conditional accuracy, clipped to [0,1]; this saturates the (CT) bound). We then
form the LABEL-FREE estimate gap = q_a - q_0 from confidences only, decide sign by the
rule above, and compare to the true sign(p_a - p_0). Monte-Carlo over trials gives the
sign-recovery rate as a function of eta (relative to a fixed accuracy gap) and as a
function of the gap (at fixed eta).
"""

import json
import os
import numpy as np

RNG = np.random.default_rng(20260606)
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                   "results_conj1_caltransfer.json")


def simulate_region(n_D, K, p_a, p_0, eta, rng):
    """One disagreement-region sample of size n_D.

    Returns label-free estimates (q_a, q_0) and the empirical true accuracies.

    Construction. On D both predictors may be right or wrong (K >= 3). We draw the
    correctness of fa as Bernoulli(p_a) and of f0 as Bernoulli(p_0) (independently --
    the joint coupling does not affect the marginal accuracy gap that drives sign).
    For each predictor we attach a confidence c that is calibrated up to eta: we set
    the *true* conditional accuracy at that point to the predictor's accuracy and let
        c = clip( accuracy + U(-eta, eta), 0, 1 ),
    so that E[1{correct} | c] = accuracy and |E[1{correct}|c] - c| <= eta, saturating
    (CT). The label-free statistic q_h = mean(c_h) then satisfies |q_h - p_h| <= eta
    in expectation.
    """
    # true correctness indicators on D
    correct_a = (rng.random(n_D) < p_a).astype(float)
    correct_0 = (rng.random(n_D) < p_0).astype(float)
    # confidence maps: mean = predictor accuracy, miscalibration bounded by eta
    conf_a = np.clip(p_a + rng.uniform(-eta, eta, n_D), 0.0, 1.0)
    conf_0 = np.clip(p_0 + rng.uniform(-eta, eta, n_D), 0.0, 1.0)
    q_a = float(conf_a.mean())          # label-free
    q_0 = float(conf_0.mean())          # label-free
    pa_emp = float(correct_a.mean())    # needs labels (only for ground truth)
    p0_emp = float(correct_0.mean())
    return q_a, q_0, pa_emp, p0_emp


def sign_decision(q_a, q_0, eta):
    """Label-free certified sign with abstention (thm:caltransfer rule).

    Returns +1 (adapt), -1 (freeze), or 0 (abstain).
    Certify only when the confidence gap exceeds the 2*eta bracket half-width.
    """
    gap = q_a - q_0
    if gap > 2 * eta:
        return +1
    if gap < -2 * eta:
        return -1
    return 0


def true_sign(p_a, p_0):
    d = p_a - p_0
    if d > 0:
        return +1
    if d < 0:
        return -1
    return 0


def run_eta_sweep(n_trials=2000, n_D=400, K=5, gap_true=0.20, etas=None):
    """Fix a true accuracy gap; sweep eta. Expect ~100% recovery while gap>2eta,
    decaying toward chance as eta grows past gap/2."""
    if etas is None:
        etas = [0.0, 0.02, 0.05, 0.08, 0.10, 0.15, 0.20, 0.30, 0.40]
    # symmetric accuracies around 0.5 with the target gap
    p_a = 0.5 + gap_true / 2.0
    p_0 = 0.5 - gap_true / 2.0
    tsign = true_sign(p_a, p_0)
    rows = []
    for eta in etas:
        n_correct = 0       # certified AND correct sign
        n_committed = 0     # certified (non-abstain)
        n_wrong_commit = 0  # certified but WRONG sign (the dangerous error)
        for _ in range(n_trials):
            q_a, q_0, _, _ = simulate_region(n_D, K, p_a, p_0, eta, RNG)
            d = sign_decision(q_a, q_0, eta)
            if d != 0:
                n_committed += 1
                if d == tsign:
                    n_correct += 1
                else:
                    n_wrong_commit += 1
        rows.append({
            "eta": eta,
            "gap_true": gap_true,
            "gap_gt_2eta": bool(gap_true > 2 * eta),
            "p_a": p_a, "p_0": p_0,
            "commit_rate": n_committed / n_trials,
            "sign_recovery_among_committed":
                (n_correct / n_committed) if n_committed else float("nan"),
            "sign_recovery_overall": n_correct / n_trials,
            "wrong_commit_rate": n_wrong_commit / n_trials,
        })
    return rows


def run_gap_sweep(n_trials=2000, n_D=400, K=5, eta=0.05, gaps=None):
    """Fix eta; sweep the true accuracy gap. Expect recovery to switch on as the gap
    crosses 2*eta."""
    if gaps is None:
        gaps = [0.0, 0.05, 0.08, 0.10, 0.12, 0.15, 0.20, 0.30, 0.40]
    rows = []
    for gap_true in gaps:
        p_a = 0.5 + gap_true / 2.0
        p_0 = 0.5 - gap_true / 2.0
        tsign = true_sign(p_a, p_0)
        n_correct = 0
        n_committed = 0
        n_wrong_commit = 0
        for _ in range(n_trials):
            q_a, q_0, _, _ = simulate_region(n_D, K, p_a, p_0, eta, RNG)
            d = sign_decision(q_a, q_0, eta)
            if d != 0:
                n_committed += 1
                if d == tsign:
                    n_correct += 1
                elif tsign != 0:
                    n_wrong_commit += 1
        rows.append({
            "gap_true": gap_true,
            "eta": eta,
            "gap_gt_2eta": bool(gap_true > 2 * eta),
            "commit_rate": n_committed / n_trials,
            "sign_recovery_among_committed":
                (n_correct / n_committed) if n_committed else float("nan"),
            "wrong_commit_rate": n_wrong_commit / n_trials,
        })
    return rows


def bracket_check(n_trials=3000, n_D=600, K=5, eta=0.08, gap_true=0.15):
    """Directly verify the bracketing claim |q_h - p_h| <= eta (in MC mean) and that
    (p_a - p_0) lies inside [gap - 2eta, gap + 2eta]."""
    p_a = 0.5 + gap_true / 2.0
    p_0 = 0.5 - gap_true / 2.0
    max_abs_dev_a = 0.0
    max_abs_dev_0 = 0.0
    n_inside = 0
    for _ in range(n_trials):
        q_a, q_0, pa_emp, p0_emp = simulate_region(n_D, K, p_a, p_0, eta, RNG)
        max_abs_dev_a = max(max_abs_dev_a, abs(q_a - pa_emp))
        max_abs_dev_0 = max(max_abs_dev_0, abs(q_0 - p0_emp))
        gap = q_a - q_0
        if (gap - 2 * eta) <= (pa_emp - p0_emp) <= (gap + 2 * eta):
            n_inside += 1
    # finite-sample sampling noise inflates the deviation slightly beyond eta;
    # report both the raw max and the eta budget for honesty.
    return {
        "eta": eta, "gap_true": gap_true, "n_D": n_D,
        "max_abs_dev_q_a_vs_p_a": max_abs_dev_a,
        "max_abs_dev_q_0_vs_p_0": max_abs_dev_0,
        "bracket_contains_truth_rate": n_inside / n_trials,
        "note": ("max deviation may exceed eta by O(1/sqrt(n_D)) finite-sample "
                 "noise; the population claim is |E[q_h]-p_h|<=eta"),
    }


def main():
    results = {
        "description": "Conjecture 1 under bounded calibration transfer (thm:caltransfer)",
        "seed": 20260606,
        "eta_sweep": run_eta_sweep(),
        "gap_sweep": run_gap_sweep(),
        "bracket_check": bracket_check(),
    }
    # headline numbers
    es = results["eta_sweep"]
    certifiable = [r for r in es if r["gap_gt_2eta"]]
    unknowable = [r for r in es if not r["gap_gt_2eta"]]
    rec_cert = np.mean([r["sign_recovery_among_committed"] for r in certifiable])
    # overall recovery in the deep-unknowable tail (eta >= 2*gap, fully straddled)
    deep = [r for r in es if r["eta"] >= 2 * r["gap_true"]]
    results["headline"] = {
        "min_sign_recovery_when_gap_gt_2eta":
            float(min(r["sign_recovery_among_committed"] for r in certifiable)),
        "mean_sign_recovery_when_gap_gt_2eta": float(rec_cert),
        "max_wrong_commit_rate_when_gap_gt_2eta":
            float(max(r["wrong_commit_rate"] for r in certifiable)),
        "max_commit_rate_in_unknowable_region":
            float(max(r["commit_rate"] for r in unknowable)) if unknowable else 0.0,
        "overall_recovery_in_deep_unknowable_tail":
            float(np.mean([r["sign_recovery_overall"] for r in deep])) if deep else None,
        "PASS_recovery_high_when_certifiable":
            bool(min(r["sign_recovery_among_committed"] for r in certifiable) >= 0.99),
        "PASS_no_wrong_commit_when_certifiable":
            bool(max(r["wrong_commit_rate"] for r in certifiable) == 0.0),
        "PASS_abstains_in_unknowable":
            bool((max(r["commit_rate"] for r in unknowable) if unknowable else 0.0) <= 0.05),
    }
    with open(OUT, "w") as f:
        json.dump(results, f, indent=2)
    print(json.dumps(results["headline"], indent=2))
    print("\n-- eta sweep (gap_true=0.20) --")
    for r in es:
        print(f"eta={r['eta']:.2f} gap>2eta={int(r['gap_gt_2eta'])} "
              f"commit={r['commit_rate']:.3f} "
              f"recov(committed)={r['sign_recovery_among_committed']:.4f} "
              f"wrong_commit={r['wrong_commit_rate']:.4f}")
    print("\n-- gap sweep (eta=0.05) --")
    for r in results["gap_sweep"]:
        print(f"gap={r['gap_true']:.2f} gap>2eta={int(r['gap_gt_2eta'])} "
              f"commit={r['commit_rate']:.3f} "
              f"recov(committed)={r['sign_recovery_among_committed']:.4f}")
    print("\n-- bracket check --")
    print(json.dumps(results["bracket_check"], indent=2))
    print(f"\nWrote {OUT}")


if __name__ == "__main__":
    main()
