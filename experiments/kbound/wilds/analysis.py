"""
analysis.py - torch-free K-Bound analysis core for the WILDS Camelyon17 pipeline.

Routing variants implemented:
  (a) SINGLE-candidate diagnostic -> decide_kga (label-disjoint cross-fitted
      gradient-boosted B_hat(Z) + split-conformal radius eps;
      ADAPT/FREEZE/ABSTAIN). It is not a held-out confirmation unless an
      independently locked validation-to-test scorer is used.
  (b) MULTI-candidate route [Theorem 1A, tau-residual] -> multicandidate_route.
      The release path contains a small NumPy-only implementation of the three
      estimators it needs.  Keeping route math here avoids importing the plotting
      stack from the numerical-validation script at runtime.
  Route C is intentionally absent.  The retired prototype mixed a binary Brier
  bracket with accuracy/F1 runner objectives and selected the compared adapter
  using target labels.  Current runners emit an explicit UNSUPPORTED state.

DETECTABILITY: detectability_analysis correlates each label-free Z feature (and the
LOO B_hat) with the TRUE benefit sign -> tells us whether harm, if it occurs, is
detectable label-free.  INTEGRITY: labels are used ONLY to compute B / oracle /
these correlations for evaluation; the router only ever sees Z or agreements.
"""
from __future__ import annotations
import os, sys
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor  # noqa: F401 (used by callers/tests)

ALPHA = 0.10
SEED = 0
CALIBRATION = "crossfit_split"

# ---- the ONE K-Bound decision path (fix-queue items 4 + 15) -----------------
# `docs/research/kbound/scripts/kbound_decide.py` wraps `kga.certificate` /
# `kga.policy`; importing it here removes decide_kga fork #2 of seven.
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), *[os.pardir] * 3))
_KB_SCRIPTS = os.path.join(_REPO_ROOT, "docs", "research", "kbound", "scripts")
if _KB_SCRIPTS not in sys.path:
    sys.path.insert(0, _KB_SCRIPTS)
import kbound_decide as _kb  # noqa: E402

# ============================ (a) single-candidate KGA =======================
def decide_kga(Z, B, alpha=ALPHA, n_estimators=250, max_depth=2, lr=0.05, seed=SEED,
               calibration=CALIBRATION, sample_ids=None):
    """Label-disjoint cross-fitted estimator + exact-rank conformal radius.

    Returns ``(Bhat, eps, decisions)`` where **eps is a per-cell ndarray**, not a
    scalar.

    FIX-QUEUE ITEM 4.  The old body ended with::

        eps = float(np.quantile(np.abs(Bhat - B), 1 - alpha))
        dec = np.where(Bhat - eps > 0, "ADAPT", ...)

    -- one interpolated ``np.quantile`` over ALL N residuals (including cell i's
    own) then used to decide cell i, so eps was a function of the very test
    labels the FA_u <= alpha guarantee attaches to.  The current path instead
    cross-fits on identity-stable folds: a scored cell's label is used in
    neither its estimator fit nor its disjoint exact-rank calibration subset.
    The radius uses ``k = ceil((n+1)(1-alpha))`` with no interpolation.

    FIX-QUEUE ITEM 15.  The body is gone; this is a thin signature-preserving
    shim over ``kbound_decide.decide_kga_crossfit``.  This file was fork #2 of
    seven.
    """
    return _kb.decide_kga_crossfit(
        Z,
        B,
        alpha=alpha,
        n_estimators=n_estimators,
        max_depth=max_depth,
        lr=lr,
        seed=seed,
        sample_ids=sample_ids,
    )


def policy_metrics(dec, a0, aa, B=None, alpha=ALPHA):
    """Realized accuracy + regret vs oracle for each policy. ABSTAIN/FREEZE -> frozen.
    beats_both REQUIRES the pre-registered false-adapt budget, not regret alone.

    FIX-QUEUE ITEM 28 -- ONE definition of false-adapt.  This function used to
    emit only ``false_adapt_rate_B<0 = mean(B[adapt] < 0)``: *conditional* on
    ADAPT and *strict*.  Two things were wrong with using it to gate
    ``beats_both``.  (i) ``thm:certificate`` bounds the MARGINAL rate
    ``Pr[ADAPT and B <= 0]``, which is what ``_locked_analysis_script.py:43``
    computes -- the conditional rate is a different, larger quantity and gating
    on it is neither the guarantee nor conservative in general.  (ii) The strict
    inequality exempts ties: 500 archived cells have ``B`` exactly 0.0 and 102 of
    them ADAPT, and an ADAPT on ``B == 0`` is a false adapt (no strict benefit
    was obtained, yet the certificate committed).
    ``fa_u`` and ``fa_c`` are now separate named fields, both computed by
    ``kbound_decide.false_adapt``; the legacy field is retained, marked
    deprecated, and gates nothing.
    """
    dec = np.asarray(dec)
    a0 = np.asarray(a0, float); aa = np.asarray(aa, float)
    if dec.ndim != 1 or a0.ndim != 1 or aa.ndim != 1 or not (len(dec) == len(a0) == len(aa)):
        raise ValueError("dec, a0, and aa must be one-dimensional arrays of equal length")
    allowed = {"ADAPT", "FREEZE", "ABSTAIN"}
    unknown = sorted({str(x) for x in dec.tolist()} - allowed)
    if unknown:
        raise ValueError(
            "unscorable routing decision(s): " + ", ".join(unknown)
            + "; ERROR/UNSUPPORTED states must be resolved, not scored as FREEZE"
        )
    if not (np.isfinite(a0).all() and np.isfinite(aa).all()):
        raise ValueError("a0 and aa must contain only finite values")
    adapt = dec == "ADAPT"
    kga = np.where(adapt, aa, a0)
    oracle = np.maximum(a0, aa)
    if B is None:
        B = aa - a0
    B = np.asarray(B, float)
    if B.shape != a0.shape or not np.isfinite(B).all():
        raise ValueError("B must match a0 and contain only finite values")
    _fa = _kb.false_adapt(dec, B)
    return {
        "n": int(len(a0)),
        "decision_counts": {d: int((dec == d).sum()) for d in ["ADAPT", "FREEZE", "ABSTAIN"]},
        "coverage": float(np.mean(dec != "ABSTAIN")),
        "abstention_rate": float(np.mean(dec == "ABSTAIN")),
        "adapt_precision_B>0": float(np.mean(B[adapt] > 0)) if adapt.any() else None,
        # ---- fix-queue item 28: the two rates, named, never interchanged ----
        "false_adapt_unconditional": _fa["fa_u"],   # Pr[ADAPT and B <= 0]  (thm:certificate)
        "false_adapt_conditional": _fa["fa_c"],     # Pr[B <= 0 | ADAPT]
        "n_false_adapt": _fa["n_false_adapt"],
        "false_adapt_definition":
            "false adapt := ADAPT and B <= 0; fa_u marginal (bounded by thm:certificate), "
            "fa_c conditional",
        # DEPRECATED (conditional AND strict). Kept only so pre-fix artifacts stay
        # diffable. Gates nothing.
        "false_adapt_rate_B<0": float(np.mean(B[adapt] < 0)) if adapt.any() else None,
        "mean_acc": {
            "always_adapt": float(aa.mean()), "always_freeze": float(a0.mean()),
            "K_Bound": float(kga.mean()), "oracle": float(oracle.mean()),
        },
        "regret_vs_oracle": {
            "always_adapt": float((oracle - aa).mean()),
            "always_freeze": float((oracle - a0).mean()),
            "K_Bound": float((oracle - kga).mean()),
        },
        "worst_case_acc": {"always_adapt": float(aa.min()), "always_freeze": float(a0.min()),
                           "K_Bound": float(kga.min())},
        "alpha_false_adapt_budget": float(alpha),
        "beats_both_regret_only": bool((oracle - kga).mean() < (oracle - aa).mean() - 1e-9 and
                                       (oracle - kga).mean() < (oracle - a0).mean() - 1e-9),
        # fix-queue item 28: the budget gate is now the MARGINAL rate the theorem
        # bounds, with the weak inequality. The old gate was
        #   float(np.mean(B[adapt] < 0)) <= alpha
        # i.e. conditional + strict, which is neither the guaranteed quantity nor
        # tie-safe.
        "beats_both": bool((oracle - kga).mean() < (oracle - aa).mean() - 1e-9 and
                           (oracle - kga).mean() < (oracle - a0).mean() - 1e-9 and
                           adapt.any() and float(_fa["fa_u"]) <= alpha),
        "beats_both_gate": "regret vs both fixed policies AND fa_u <= alpha "
                           "(fa_u = Pr[ADAPT and B <= 0])",
    }


def label_regime(B, thr=0.02):
    return "helpful" if B > thr else ("harmful" if B < -thr else "marginal")


# ============================ detectability ==================================
def _auc(score, label):
    """AUC via Mann-Whitney U (prob. that a positive outranks a negative).
    label: 1 = harmful (B<0) [the event we want to detect], 0 = not harmful."""
    score = np.asarray(score, float); label = np.asarray(label, int)
    pos = score[label == 1]; neg = score[label == 0]
    if len(pos) == 0 or len(neg) == 0:
        return None
    order = np.argsort(score, kind="mergesort")
    ranks = np.empty(len(score), float); ranks[order] = np.arange(1, len(score) + 1)
    # average ties
    s = score[order]; i = 0
    while i < len(s):
        j = i
        while j + 1 < len(s) and s[j + 1] == s[i]:
            j += 1
        if j > i:
            ranks[order[i:j + 1]] = (i + 1 + j + 1) / 2.0
        i = j + 1
    r_pos = ranks[label == 1].sum()
    auc = (r_pos - len(pos) * (len(pos) + 1) / 2.0) / (len(pos) * len(neg))
    return float(auc)


def detectability_analysis(records, evidence_names, alpha=ALPHA):
    """Does the label-free evidence Z reveal the TRUE benefit sign?

    records: list of dicts with keys 'Z' (list), 'B' (float).  Per Z-feature we report
    Pearson corr with B, point-biserial corr with the harmful event (B<0), and the
    single-feature AUC for detecting harm (the feature is sign-flipped so AUC>=.5 means
    'higher value => more harmful').  We also report the AUC of the LOO B_hat(Z) harm
    predictor (-B_hat as harm score) -> the operational detectability of the certificate
    only when the requested exact-rank cross-fit design is feasible.
    """
    Z = np.array([r["Z"] for r in records], float)
    B = np.array([r["B"] for r in records], float)
    harmful = (B < 0).astype(int)
    n, d = Z.shape
    out = {"n_cells": int(n), "n_harmful": int(harmful.sum()),
           "base_rate_harmful": float(harmful.mean()), "mean_B": float(B.mean()),
           "per_feature": {}}
    for k in range(d):
        zk = Z[:, k]
        pear = float(np.corrcoef(zk, B)[0, 1]) if np.std(zk) > 1e-12 else 0.0
        # orient harm-score so that higher => more harmful, then AUC
        a_pos = _auc(zk, harmful); a_neg = _auc(-zk, harmful)
        best = max([a for a in (a_pos, a_neg) if a is not None], default=None)
        name = evidence_names[k] if k < len(evidence_names) else f"z{k}"
        out["per_feature"][name] = {"pearson_corr_B": pear, "harm_AUC": best}
    # Operational cross-fitted B_hat as a harm detector.  At alpha=.10 the
    # disjoint fit/calibrate/score construction needs 12 cells.  The old n>=4
    # guard called the exact-rank routine for n=4..11 and then serialized an
    # infinite radius.  An infeasible finite-sample certificate is now explicit
    # JSON null/status, never an Infinity token.
    minimum_cells = int(_kb.minimum_crossfit_size(alpha))
    out["certificate_minimum_cells"] = minimum_cells
    if harmful.sum() in (0, n):
        out["certificate_calibration_status"] = "NOT_APPLICABLE_NO_REGIME_VARIATION"
        out["certificate_eps"] = None
        out["certificate_eps_min"] = None
        out["certificate_eps_max"] = None
    elif n < minimum_cells:
        out["certificate_calibration_status"] = "INFEASIBLE_UNDERSIZED_EXACT_RANK"
        out["certificate_calibration_feasible"] = False
        out["certificate_infeasible_reason"] = (
            f"need at least {minimum_cells} cells for disjoint estimator-fit, "
            f"calibration, and scoring at alpha={float(alpha):g}; observed {n}"
        )
        out["certificate_harm_AUC_negBhat"] = None
        out["certificate_eps"] = None
        out["certificate_eps_is"] = "unavailable: exact-rank calibration infeasible"
        out["certificate_eps_min"] = None
        out["certificate_eps_max"] = None
    else:
        Bhat, eps, dec = decide_kga(Z, B, alpha=alpha)
        out["certificate_harm_AUC_negBhat"] = _auc(-Bhat, harmful)
        # eps is now one radius PER CELL (fix-queue item 4), so a single scalar
        # here would silently be a summary. Say which summary it is.
        eps = np.asarray(eps, float)
        if not np.isfinite(eps).all():
            raise ValueError("feasible detectability calibration returned a non-finite radius")
        out["certificate_calibration_status"] = "FINITE"
        out["certificate_calibration_feasible"] = True
        out["certificate_eps"] = float(np.mean(eps))
        out["certificate_eps_is"] = "mean of label-disjoint cross-fitted split-conformal radii"
        out["certificate_eps_min"] = float(np.min(eps))
        out["certificate_eps_max"] = float(np.max(eps))
    # headline: is harm detectable at all?
    aucs = [v["harm_AUC"] for v in out["per_feature"].values() if v["harm_AUC"] is not None]
    out["best_single_feature_harm_AUC"] = float(max(aucs)) if aucs else None
    if out["best_single_feature_harm_AUC"] is not None:
        out["detectability_verdict"] = (
            "detectable" if out["best_single_feature_harm_AUC"] >= 0.75 else
            ("weak" if out["best_single_feature_harm_AUC"] >= 0.6 else "undetectable"))
    else:
        out["detectability_verdict"] = "n/a (no regime variation)"
    return out


# ================= (b) multi-candidate route [Theorem 1A, tau] ===============
_BINARY_TASK_TYPES = frozenset({"binary", "binary_classification"})
_ACCURACY_OBJECTIVES = frozenset({"accuracy", "top1_accuracy", "top_1_accuracy"})


class _OrientationError(ValueError):
    """The agreement system cannot be oriented from the trusted anchor."""


def _route_state(decision, status, reason, *, scorable, **extra):
    """Build one explicit route state; non-OK states are never silently scorable."""
    out = {
        "decision": decision,
        "status": status,
        "scorable": bool(scorable),
        "choice": None,
        "reason": reason,
    }
    out.update(extra)
    return out


def _normalise_token(value):
    return str(value).strip().lower().replace("-", "_").replace(" ", "_")


def _prediction_labels(preds_all):
    """Return the observed label set, rejecting missing or non-finite labels."""
    if preds_all.dtype.kind in "biuf":
        if not np.isfinite(preds_all).all():
            raise ValueError("predictions contain NaN or infinity")
    elif preds_all.dtype.kind == "c":
        raise ValueError("complex-valued class labels are unsupported")
    else:
        for value in preds_all.ravel().tolist():
            if value is None:
                raise ValueError("predictions contain missing labels")
            try:
                missing = bool(np.isnan(value))
            except (TypeError, ValueError):
                missing = False
            if missing:
                raise ValueError("predictions contain NaN labels")
    try:
        return np.unique(preds_all)
    except (TypeError, ValueError) as exc:
        raise ValueError("prediction labels must have one consistent comparable type") from exc


def _candidate_geometry(preds_all, labels, rank_rtol, rank_atol):
    """Audit exact duplicates and numerical row rank in a coding-invariant way.

    The route is binary-only, so mapping either observed label to 0/1 loses no
    information.  Centering and row-normalising then makes the SVD invariant to
    swapping the two label names and prevents nominally duplicated/complementary
    predictors from satisfying the M>=4 identifiability premise.
    """
    M = preds_all.shape[0]
    duplicates = []
    for i in range(M):
        for j in range(i + 1, M):
            if np.array_equal(preds_all[i], preds_all[j]):
                duplicates.append([int(i), int(j)])

    if len(labels) == 2:
        encoded = (preds_all == labels[1]).astype(float)
    else:
        # A declared binary task may contain a one-class batch.  Such a batch has
        # no candidate geometry and must fail the rank check rather than commit.
        encoded = np.zeros(preds_all.shape, dtype=float)
    centered = encoded - encoded.mean(axis=1, keepdims=True)
    norms = np.linalg.norm(centered, axis=1)
    normalised = np.divide(
        centered,
        norms[:, None],
        out=np.zeros_like(centered),
        where=norms[:, None] > rank_atol,
    )
    singular_values = np.linalg.svd(normalised, compute_uv=False)
    if len(singular_values) == 0 or singular_values[0] <= rank_atol:
        effective_rank = 0
    else:
        threshold = max(float(rank_atol), float(rank_rtol) * float(singular_values[0]))
        effective_rank = int(np.sum(singular_values > threshold))
    return duplicates, effective_rank, [float(x) for x in singular_values]


def agreement_matrix(preds_all, D):
    """Centered pairwise prediction-agreement matrix C on region D (label-free).

    ``preds_all`` has shape (M, N), row 0 is the frozen anchor, and
    C_ij = 2 Pr(f_i=f_j | D) - 1.  Input validation belongs to
    :func:`multicandidate_route`; this helper intentionally performs only math.
    """
    P = preds_all[:, D]
    M = P.shape[0]
    eq = (P[:, None, :] == P[None, :, :]).mean(axis=2)
    C = 2.0 * eq - 1.0
    np.fill_diagonal(C, 0.0)
    return C


def _rankone_fit_offdiag(C, iters=90, tol=1e-12):
    """NumPy-only off-diagonal rank-one fit used solely to compute tau.

    The fitted vector has an arbitrary global sign and can be outside [-1, 1].
    It is therefore deliberately *not returned by the route* and is never used
    for orientation, ranking, a margin, or a decision.
    """
    M = C.shape[0]
    W = np.array(C, dtype=float, copy=True)
    d = np.sqrt(np.clip((C ** 2).sum(axis=1) / max(M - 1, 1), 1e-12, None))
    previous = None
    b_fit = np.zeros(M, dtype=float)
    for _ in range(int(iters)):
        np.fill_diagonal(W, d ** 2)
        values, vectors = np.linalg.eigh(W)
        lam = max(float(values[-1]), 0.0)
        b_fit = np.sqrt(lam) * vectors[:, -1]
        if not np.isfinite(b_fit).all():
            raise FloatingPointError("non-finite spectral rank-one fit")
        d = np.abs(b_fit)
        if previous is not None and min(
            np.linalg.norm(b_fit - previous), np.linalg.norm(b_fit + previous)
        ) < tol:
            break
        previous = b_fit.copy()
    off = ~np.eye(M, dtype=bool)
    residual = C[off] - np.outer(b_fit, b_fit)[off]
    tau = float(np.sqrt(np.dot(residual, residual)))
    if not np.isfinite(tau):
        raise FloatingPointError("non-finite tau residual")
    return b_fit, tau


def _minor_estimator(C, orientation_tol=1e-10):
    """Anchor-oriented, bounded median-of-minors advantage estimator.

    Candidate 0 is declared above chance by the caller.  Magnitudes are clipped
    to [0, 1] before the signs are oriented by C[0,j].  An unobservable sign for
    any non-zero estimate is an explicit failure, never a spectral fallback.
    """
    M = C.shape[0]
    b2 = np.zeros(M, dtype=float)
    for i in range(M):
        others = [k for k in range(M) if k != i]
        ratios = []
        for left in range(len(others)):
            for right in range(left + 1, len(others)):
                k, ell = others[left], others[right]
                denominator = float(C[k, ell])
                if abs(denominator) <= orientation_tol:
                    continue
                ratio = float(C[i, k] * C[i, ell] / denominator)
                if np.isfinite(ratio):
                    ratios.append(ratio)
        if not ratios:
            raise _OrientationError(f"candidate {i} has no finite orientable minor")
        b2[i] = float(np.median(ratios))
    magnitude = np.sqrt(np.clip(b2, 0.0, 1.0))
    b = magnitude.copy()  # trusted anchor fixes b[0] >= 0
    for j in range(1, M):
        if magnitude[j] <= orientation_tol:
            b[j] = 0.0
            continue
        anchor_agreement = float(C[0, j])
        if abs(anchor_agreement) <= orientation_tol:
            raise _OrientationError(f"candidate {j} sign is not orientable from the anchor")
        b[j] *= np.sign(anchor_agreement)
    if not np.isfinite(b).all() or np.any(np.abs(b) > 1.0 + 1e-12):
        raise _OrientationError("bounded advantage estimator failed its finite/range check")
    if b[0] <= orientation_tol:
        raise _OrientationError("anchor advantage is not strictly positive/orientable")
    return np.clip(b, -1.0, 1.0)


def _overdet_residual(C):
    """Mean spread of the three rank-one pairings over all 4-subsets."""
    from itertools import combinations

    M = C.shape[0]
    if M < 4:
        return 0.0
    spreads = []
    for i, j, k, ell in combinations(range(M), 4):
        products = [C[i, j] * C[k, ell], C[i, k] * C[j, ell], C[i, ell] * C[j, k]]
        spreads.append(float(max(products) - min(products)))
    value = float(np.mean(spreads))
    if not np.isfinite(value):
        raise FloatingPointError("non-finite overdetermination residual")
    return value


def multicandidate_route(
    preds_all,
    tau_star=0.08,
    kappa=2.5,
    min_D=8,
    *,
    task_type=None,
    n_classes=None,
    objective=None,
    anchor_above_chance=None,
    effective_rank_rtol=1e-6,
    effective_rank_atol=1e-12,
):
    """Fail-closed Theorem-1A route for binary accuracy only.

    The pairwise-agreement identity used by this route is valid only for binary
    correctness and an accuracy objective.  ``objective`` and the trusted
    ``anchor_above_chance`` premise must therefore be explicit.  Binary task
    metadata may be supplied as ``task_type='binary_classification'`` or
    ``n_classes=2``; when both are omitted, exactly two observed prediction
    labels are accepted as a documented inference.  Any multiclass label set or
    non-accuracy objective returns an unscorable ``ABSTAIN/UNSUPPORTED`` state.

    Every decision quantity -- anchor, ranking, h_hat, margin, and selected
    candidate -- uses the same anchor-oriented, [-1, 1]-bounded minor estimator.
    The arbitrary-sign spectral vector is used only inside the scalar tau
    residual calculation and is never exposed or used for a decision.
    """
    try:
        predictions = np.asarray(preds_all)
    except Exception as exc:
        return _route_state("ERROR", "ERROR", f"could not read prediction matrix: {exc}", scorable=False)
    if predictions.ndim != 2:
        return _route_state(
            "ERROR", "ERROR", f"preds_all must have shape (M, N); got ndim={predictions.ndim}",
            scorable=False,
        )
    M, N = predictions.shape
    base = {"M": int(M), "N": int(N), "n_D": 0}
    if M == 0 or N == 0:
        return _route_state("ERROR", "ERROR", "prediction matrix must be non-empty", scorable=False, **base)

    scalar_parameters = {
        "tau_star": tau_star,
        "kappa": kappa,
        "effective_rank_rtol": effective_rank_rtol,
        "effective_rank_atol": effective_rank_atol,
    }
    try:
        parsed = {name: float(value) for name, value in scalar_parameters.items()}
    except (TypeError, ValueError):
        return _route_state("ERROR", "ERROR", "route thresholds must be numeric", scorable=False, **base)
    if not all(np.isfinite(value) and value >= 0.0 for value in parsed.values()):
        return _route_state(
            "ERROR", "ERROR", "route thresholds must be finite and non-negative", scorable=False, **base
        )
    tau_star = parsed["tau_star"]
    kappa = parsed["kappa"]
    rank_rtol = parsed["effective_rank_rtol"]
    rank_atol = parsed["effective_rank_atol"]
    if isinstance(min_D, (bool, np.bool_)) or not isinstance(min_D, (int, np.integer)) or int(min_D) < 1:
        return _route_state("ERROR", "ERROR", "min_D must be a positive integer", scorable=False, **base)
    min_D = int(min_D)

    try:
        labels = _prediction_labels(predictions)
    except ValueError as exc:
        return _route_state("ERROR", "ERROR", str(exc), scorable=False, **base)

    objective_token = None if objective is None else _normalise_token(objective)
    if objective_token not in _ACCURACY_OBJECTIVES:
        shown = "missing" if objective is None else repr(objective)
        return _route_state(
            "ABSTAIN", "UNSUPPORTED",
            f"UNSUPPORTED objective {shown}; Route B is valid only for binary accuracy",
            scorable=False, observed_n_classes=int(len(labels)), **base,
        )

    if task_type is not None and _normalise_token(task_type) not in _BINARY_TASK_TYPES:
        return _route_state(
            "ABSTAIN", "UNSUPPORTED",
            f"UNSUPPORTED task_type {task_type!r}; Route B requires binary classification",
            scorable=False, observed_n_classes=int(len(labels)), **base,
        )
    if n_classes is not None:
        if isinstance(n_classes, (bool, np.bool_)):
            declared_n_classes = None
        else:
            try:
                declared_n_classes = float(n_classes)
            except (TypeError, ValueError):
                declared_n_classes = None
        if declared_n_classes != 2.0:
            return _route_state(
                "ABSTAIN", "UNSUPPORTED",
                f"UNSUPPORTED n_classes={n_classes!r}; Route B requires exactly two classes",
                scorable=False, observed_n_classes=int(len(labels)), **base,
            )
    if len(labels) > 2:
        return _route_state(
            "ABSTAIN", "UNSUPPORTED",
            f"UNSUPPORTED observed label set has {len(labels)} classes; binary identity does not hold",
            scorable=False, observed_n_classes=int(len(labels)), **base,
        )
    declared_binary = task_type is not None or n_classes is not None
    if not declared_binary and len(labels) != 2:
        return _route_state(
            "ABSTAIN", "UNSUPPORTED",
            "UNSUPPORTED task is not verifiably binary; pass task_type or n_classes metadata",
            scorable=False, observed_n_classes=int(len(labels)), **base,
        )
    if not isinstance(anchor_above_chance, (bool, np.bool_)) or not bool(anchor_above_chance):
        return _route_state(
            "ABSTAIN", "UNSUPPORTED",
            "UNSUPPORTED missing/false anchor_above_chance premise; global sign is unidentified",
            scorable=False, observed_n_classes=int(len(labels)), **base,
        )

    try:
        duplicates, effective_rank, singular_values = _candidate_geometry(
            predictions, labels, rank_rtol, rank_atol
        )
    except (FloatingPointError, np.linalg.LinAlgError, ValueError) as exc:
        return _route_state(
            "ERROR", "ERROR", f"candidate geometry check failed: {exc}", scorable=False, **base
        )
    geometry = {
        "duplicate_candidate_pairs": duplicates,
        "effective_candidate_rank": int(effective_rank),
        "candidate_singular_values": singular_values,
    }
    if duplicates or effective_rank < M:
        detail = f"exact duplicate pairs={duplicates}" if duplicates else f"effective rank={effective_rank} < M={M}"
        return _route_state(
            "ABSTAIN", "DEGENERATE_CANDIDATES",
            f"candidate panel is not identifiable: {detail}",
            scorable=False, observed_n_classes=int(len(labels)), **base, **geometry,
        )
    if M < 4:
        return _route_state(
            "ABSTAIN", "INSUFFICIENT_CANDIDATES",
            f"need four distinct full-rank predictors including the anchor; got M={M}",
            scorable=False, observed_n_classes=int(len(labels)), **base, **geometry,
        )

    unanimous = (predictions == predictions[0:1, :]).all(axis=0)
    D = np.where(~unanimous)[0]
    route_context = {
        **base,
        **geometry,
        "n_D": int(len(D)),
        "observed_n_classes": int(len(labels)),
        "task_type": "binary_classification",
        "objective": "accuracy",
        "anchor_above_chance": True,
    }
    if len(D) < min_D:
        return _route_state(
            "FREEZE", "OK",
            f"|D|={len(D)} < min_D={min_D}; candidates nearly agree, so freeze",
            scorable=True, **route_context,
        )

    try:
        C = agreement_matrix(predictions, D)
        if not np.isfinite(C).all():
            raise FloatingPointError("agreement matrix is non-finite")
        spectral_fit, tau = _rankone_fit_offdiag(C)
        if not np.isfinite(np.asarray(spectral_fit, dtype=float)).all():
            raise FloatingPointError("spectral tau fit is non-finite")
        b_decision = np.asarray(_minor_estimator(C), dtype=float)
        if b_decision.shape != (M,):
            raise ValueError(f"decision estimator returned shape {b_decision.shape}; expected {(M,)}")
        off = ~np.eye(M, dtype=bool)
        h_hat = float(np.max(np.abs(C - np.outer(b_decision, b_decision))[off]))
        margin = float(kappa * h_hat + 2.0 / np.sqrt(len(D)))
        overdet = _overdet_residual(C)
        numeric = np.asarray([tau, h_hat, margin, overdet, *b_decision], dtype=float)
        if not np.isfinite(numeric).all() or np.any(np.abs(b_decision) > 1.0 + 1e-12):
            raise FloatingPointError("route produced a non-finite or unbounded statistic")
    except _OrientationError as exc:
        return _route_state(
            "ABSTAIN", "ORIENTATION_FAILED", f"anchor orientation failed: {exc}",
            scorable=False, **route_context,
        )
    except (FloatingPointError, np.linalg.LinAlgError, ValueError) as exc:
        return _route_state(
            "ERROR", "ERROR", f"route numerical failure: {exc}", scorable=False, **route_context
        )

    gate = bool(tau <= tau_star)
    bounded = [float(x) for x in b_decision]
    result = {
        **route_context,
        "status": "OK",
        "scorable": True,
        "tau": float(tau),
        "tau_star": float(tau_star),
        "gate_pass": gate,
        "overdet_residual": float(overdet),
        "h_hat": float(h_hat),
        "margin": float(margin),
        "advantage_estimator": "anchor_oriented_bounded_median_of_minors",
        "b_decision": bounded,
        "b_tilde": bounded,
        # Backward-compatible field: no spectral vector is emitted.  The alias
        # is bounded and is exactly the vector used for every decision quantity.
        "b_hat": bounded,
        "b_hat_semantics": "bounded decision estimator (alias of b_tilde), not spectral fit",
        "anchor_b0": float(b_decision[0]),
    }
    if not gate:
        result.update({
            "decision": "ABSTAIN",
            "choice": None,
            "reason": f"tau={tau:.4f} > tau*={tau_star:.4f} certifies the rank-one premise is violated",
        })
        return result

    advantage_over_anchor = b_decision[1:] - b_decision[0]
    committed = [
        i + 1 for i in range(M - 1)
        if advantage_over_anchor[i] > margin and b_decision[i + 1] > 0.0
    ]
    if not committed:
        result.update({
            "decision": "FREEZE",
            "choice": None,
            "reason": "no bounded, anchor-oriented candidate estimate beats the anchor by the margin",
        })
        return result
    choice = int(max(committed, key=lambda i: b_decision[i]))
    result.update({
        "decision": "ADAPT",
        "choice": choice,
        "committed": committed,
        "reason": (
            f"candidate {choice} commits (b={b_decision[choice]:.3f} > "
            f"anchor {b_decision[0]:.3f} + margin {margin:.3f})"
        ),
    })
    return result
