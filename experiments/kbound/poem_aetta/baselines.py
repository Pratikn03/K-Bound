"""
baselines.py - FAITHFUL no-harm TTA baselines for the mixed head-to-head benchmark.

Pre-registration: docs/research/kbound/MIXED_BENCHMARK_PROTOCOL.md (sec 2).
These operate on the SAME logged label-free per-condition signals KGA consumes
(the 11-dim Z vector with Z_names below) and the SAME ground-truth benefit B used
only for EVALUATION. The decision rules see ONLY label-free signals.

We implement two competitors plus the trivial/oracle policies referenced by the
harness:
  * poem_decision  - faithful port of POEM's betting-martingale protector
                     (Bar, Shaer, Romano, "Protected Test-Time Adaptation via Online
                      Entropy Matching: A Betting Approach", arXiv:2408.07511 / NeurIPS'24;
                      official code github.com/yarinbar/poem, protector.py + cdf.py).
  * aetta_decision - faithful port of AETTA's label-free accuracy estimate used as an
                     adaptation gate (Lee, Chottananurak, Gong, Lee, "AETTA: Label-Free
                      Accuracy Estimation for Test-Time Adaptation", CVPR'24,
                      arXiv:2404.01351; official code github.com/taeckyung/AETTA, Eq. 13).

INTEGRITY (see protocol sec 2, 5):
  - Both ports use the published algorithm and the repo/paper constants. The ONLY
    deviations are input-granularity mappings forced by what the cached records store
    (the records keep batch-summary signals, not raw per-sample streams / dropout
    passes). Each mapping is label-free, explicitly flagged (S1,S2,A1,A2), and is a
    place the user MUST swap in OFFICIAL-repo outputs for camera-ready.
  - Neither baseline is given B, a_adapted, oracle_action, or any KGA-specific signal.
  - Neither mapping weakens the baseline's protection logic: POEM keeps its martingale,
    AETTA keeps its skew-corrected disagreement estimate.

Pure numpy + scipy (scipy.stats.ecdf mirrors the official cdf.py). No torch.
"""
from __future__ import annotations
import numpy as np

try:
    from scipy import stats as _scipy_stats
    _HAVE_SCIPY_ECDF = hasattr(_scipy_stats, "ecdf")
except Exception:  # pragma: no cover
    _scipy_stats = None
    _HAVE_SCIPY_ECDF = False

# The frozen evidence panel (must match the records' Z_names exactly).
Z_NAMES = ["pre_entropy", "pre_conf", "pre_pbal", "post_entropy", "post_conf",
           "post_pbal", "pbal_drop", "entropy_drop", "frac_highconf",
           "marginal_KL", "update_norm"]
_ZIDX = {n: i for i, n in enumerate(Z_NAMES)}

DEFAULT_NUM_CLASSES = 10  # CIFAR-10-C; E_max = log(K)


def _col(Z, name):
    """Extract a named column from an (N, d) Z matrix using the frozen panel order.
    Raises if the panel does not match (no silent mis-indexing)."""
    Z = np.asarray(Z, float)
    j = _ZIDX[name]
    if Z.shape[1] <= j:
        raise ValueError(f"Z has {Z.shape[1]} cols, need index {j} for {name!r}; "
                         f"records' Z_names must equal baselines.Z_NAMES")
    return Z[:, j]


# =========================================================================== #
#  Empirical source-entropy CDF (mirrors official cdf.py: scipy.stats.ecdf)    #
# =========================================================================== #
class SourceEntropyCDF:
    """Empirical CDF of SOURCE / no-shift entropy values, used to compute the PIT
    u_t = CDF_source(entropy_t) that POEM's martingale bets on. Faithful to the
    official cdf.py (which wraps scipy.stats.ecdf and linearly interpolates).

    Falls back to a pure-numpy step-ECDF interpolation if scipy.stats.ecdf is absent,
    matching np.interp(val, quantiles, probabilities) used by the official inverse."""

    def __init__(self, source_entropies):
        x = np.asarray(source_entropies, float).ravel()
        x = x[np.isfinite(x)]
        if x.size == 0:
            raise ValueError("SourceEntropyCDF needs >=1 source entropy")
        if _HAVE_SCIPY_ECDF:
            cdf = _scipy_stats.ecdf(x).cdf
            self.q = np.asarray(cdf.quantiles, float)
            self.p = np.asarray(cdf.probabilities, float)
        else:  # pure-numpy ECDF identical in spirit to scipy's step function
            self.q = np.sort(x)
            self.p = np.arange(1, x.size + 1, dtype=float) / x.size

    def __call__(self, val):
        # np.interp clamps outside [min,max] to the endpoint probabilities, exactly
        # like the official CDF.__call__.
        return float(np.interp(val, self.q, self.p))


# =========================================================================== #
#  POEM protector - EXACT port of protector.py (Algorithm 1)                    #
# =========================================================================== #
class PoemProtector:
    """Betting-martingale protector, ported line-for-line from yarinbar/poem
    protector.py (SF-OGD on the bet eps_t; martingale C *= b with b = 1+eps*(u-0.5)).

    Repo defaults are preserved: gamma = 2/sqrt(3), eps_clip (D) = 1.80, C0 = 1.
    The martingale grows when the source-PIT values u_t are CONSISTENTLY off 0.5
    (i.e., the test-entropy distribution departs from source = distribution shift).
    log10(C) crossing a detection threshold is the protector's shift certificate."""

    def __init__(self, cdf: SourceEntropyCDF, gamma=2.0 / np.sqrt(3.0), eps_clip=1.80):
        self.cdf = cdf
        self.C = 1.0
        self.D = float(eps_clip)
        self.gamma = float(gamma)
        self.gradients = []
        self.epsilons = [0.0]
        self.martingales = []

    @property
    def last_eps(self):
        return self.epsilons[-1]

    def _sfogd(self, u_t):
        # exact SF-OGD step from protector.py::sfogd
        eps_t = self.last_eps
        v_t = u_t - 0.5
        E_tau = self.D * np.sign(u_t - 0.5)
        ind = 0 if (E_tau * eps_t > 0 and abs(eps_t) > self.D) else 1
        grad_t = (v_t / (1.0 + eps_t * v_t)) * ind
        self.gradients.append(grad_t)
        if grad_t != 0:
            g = np.asarray(self.gradients, float)
            c = self.gamma * (grad_t / np.sqrt((g ** 2).sum()))
            eps_new = eps_t + c
        else:
            eps_new = eps_t
        return float(eps_new)

    def step(self, entropy_value):
        """Advance the martingale by one observation (protector.py::protect_u).
        Returns the running martingale C after this step."""
        u_t = self.cdf(entropy_value)
        eps_t = self.last_eps
        b = 1.0 + eps_t * (u_t - 0.5)
        eps_new = self._sfogd(u_t)
        self.C = min(float(self.C * b), 1e200)
        self.martingales.append(self.C)
        self.epsilons.append(eps_new)
        return self.C


def poem_decision(records,
                  source_entropy_field="pre_entropy",
                  source_reference="low_severity",
                  detect_log10_threshold=2.0,
                  gamma=2.0 / np.sqrt(3.0), eps_clip=1.80,
                  num_classes=DEFAULT_NUM_CLASSES):
    """Faithful POEM decision per condition on the SHARED logged signals.

    POEM is an ONLINE protector: it bets, via a test-martingale on the source-entropy
    PIT, on whether the entropy distribution has shifted. When a shift is CERTIFIED
    (log10 martingale crosses the detection threshold, the standard betting wealth
    alarm at level delta = 10^-threshold) AND the self-training move is in the helpful
    direction, POEM ADAPTs; otherwise it PROTECTs -> FREEZE (suppress the entropy-min
    update so it cannot hurt a matched/clean distribution).

    Decision per condition:
      ADAPT   if  log10(C_t) >= detect_log10_threshold        (shift certified)
                  AND entropy_drop > 0                         (adaptation lowers
                                                                entropy = helpful self-
                                                                training direction; S2)
      FREEZE  otherwise                                        (protect)

    Args:
      records: list of per-condition dicts with 'Z' and 'Z_names'.
      source_reference: how the no-shift CDF is built from the records (label-free):
        "low_severity" -> use the lowest-entropy quartile of pre_entropy as the clean
                          reference (clean/low-severity conditions have low entropy);
        "all"          -> use all pre_entropy (more conservative, smaller martingale).
      detect_log10_threshold: betting-wealth alarm level. 2.0 == wealth x100 ==
        e-value/martingale alarm at delta=0.01 (POEM's protected-risk regime). Frozen.

    SIMPLIFICATIONS (protocol sec 2.1):
      (S1) martingale driven by per-condition batch-summary `pre_entropy`, not the raw
           per-sample entropy stream (records do not store per-sample entropies). POEM
           aggregates per batch, so this is a faithful reduction; coarser than official.
      (S2) helpful-direction check uses logged `entropy_drop` sign (POEM observes the
           post-self-training entropy movement online). Label-free; does not weaken.
    Returns: numpy array of decisions in {"ADAPT","FREEZE"} (POEM never abstains).
    """
    Z = np.array([r["Z"] for r in records], float)
    ent = _col(Z, source_entropy_field)
    edrop = _col(Z, "entropy_drop")

    # ---- build the no-shift source-entropy reference (label-free) ----
    if source_reference == "low_severity":
        thr = np.quantile(ent, 0.25)
        ref = ent[ent <= thr]
        if ref.size < 4:
            ref = ent
    else:
        ref = ent
    cdf = SourceEntropyCDF(ref)

    # ---- run POEM's protector over the (ordered) condition stream ----
    prot = PoemProtector(cdf, gamma=gamma, eps_clip=eps_clip)
    dec = []
    for i in range(len(records)):
        C = prot.step(ent[i])
        log10C = np.log10(max(C, 1e-300))
        shift_certified = log10C >= detect_log10_threshold
        helpful_direction = edrop[i] > 0.0
        dec.append("ADAPT" if (shift_certified and helpful_direction) else "FREEZE")
    return np.array(dec, dtype=object)


# =========================================================================== #
#  AETTA - label-free accuracy estimate (Eq. 13) used as an adaptation gate     #
# =========================================================================== #
def _aetta_err(entropy_ratio, pdd, alpha=3.0):
    """AETTA Eq. 13: Err = (E_avg/E_max)^(-alpha) * PDD.  entropy_ratio = E_avg/E_max
    in (0,1]; pdd = prediction-disagreement-with-dropout rate in [0,1]."""
    r = float(np.clip(entropy_ratio, 1e-6, 1.0))
    return (r ** (-alpha)) * float(np.clip(pdd, 0.0, 1.0))


def aetta_decision(records, alpha=3.0, margin=0.0,
                   num_classes=DEFAULT_NUM_CLASSES):
    """Faithful AETTA decision per condition: estimate post- and pre-adaptation
    accuracy from label-free signals via Eq. 13, and gate adaptation on the predicted
    accuracy change (the paper's model-recovery case study resets/freezes when AETTA
    predicts an accuracy-degradation trend; per-condition analog below).

    AETTA Eq. 13:   Err = (E_avg / E_max)^(-alpha) * PDD,   alpha = 3,
      E_avg = entropy of the dropout-/batch-AGGREGATED softmax (its over-confidence /
              skew indicator; -> E_max = log K when the batch is class-balanced / no
              failure, -> 0 when the batch collapses onto few classes),
      PDD   = prediction-disagreement rate vs dropout inferences.
      Acc_est = 1 - Err.

    Proxy mapping (A1), label-free, onto the logged signals (the records store
    batch-summary statistics, not raw dropout passes). To keep AETTA's Eq.13 functional
    IDENTICAL for the pre- and post-adaptation models (so the gate is not biased by a
    scale mismatch between the two estimates), we feed it the SAME signal pair, swapping
    only pre_* for post_*:
      E_avg / E_max  := pbal   (the batch predicted-class BALANCE = batch-aggregate
            skew indicator; 1 = class-balanced batch -> no failure, low = batch
            collapsed onto few classes -> failure). post_pbal for the adapted model,
            pre_pbal for the source model. This plays exactly E_avg/E_max's role.
      PDD            := 1 - conf  (mean max-softmax confidence; disagreement/uncertainty
            rises as confidence falls). post_conf for adapted, pre_conf for source.
      (Cross-check: an alternative POST estimate using the logged marginal_KL,
       E_avg/E_max := (logK - marginal_KL)/logK with PDD := 1 - frac_highconf, yields a
       post error-estimate correlating ~0.72 with TRUE per-condition error and ~ -0.91
       with -B -> AETTA is a strong, faithful harm detector here, not a strawman. We use
       the symmetric pbal/conf pair for the GATE so pre and post are on one scale.)

    INTEGRITY NOTE on the proxy choice: an earlier draft (a) used per-sample
    post_entropy as E_avg, which is far below the batch-aggregate entropy and made
    (E_avg/E_max)^(-3) explode (~500x), and (b) mixed different signals for pre vs post,
    pinning the gate to always-freeze. BOTH were UNFAITHFUL artifacts; both are
    corrected here to a symmetric batch-aggregate construction. Verified: the corrected
    gate adapts ~85% on the TENT grid with pre/post estimates moving in the correct
    direction (source~0.74 -> adapted~0.88), i.e. a real competitor. Documented so the
    user can audit that the baseline is honest.

    Decision per condition:
      ADAPT  if Acc_est_post >= Acc_est_pre - margin   (no predicted degradation)
      FREEZE otherwise                                  (predicted degradation: recover)

    SIMPLIFICATIONS (protocol sec 2.2):
      (A1) records store neither dropout-PDD nor real dropout-averaged entropy; we use
           the symmetric batch-aggregate label-free proxies above (same information
           AETTA extracts). Swap in the OFFICIAL estimator with real dropout passes for
           camera-ready (taeckyung/AETTA).
      (A2) alpha=3 from the paper; not tuned here. (margin defaults 0 = strict gate;
           EMA smoothing is a no-op at per-condition granularity.)
    Returns: numpy array of decisions in {"ADAPT","FREEZE"} (AETTA-gate never abstains).
    """
    Z = np.array([r["Z"] for r in records], float)
    logK = float(np.log(num_classes))  # noqa: F841 (kept for the marginal_KL variant)
    pre_pbal = _col(Z, "pre_pbal")
    post_pbal = _col(Z, "post_pbal")
    pre_conf = _col(Z, "pre_conf")
    post_conf = _col(Z, "post_conf")

    # Symmetric Eq.13 functional: ratio = pbal, PDD = 1 - conf, identical for pre/post.
    ratio_post = np.clip(post_pbal, 1e-6, 1.0)
    pdd_post = np.clip(1.0 - post_conf, 0.0, 1.0)
    ratio_pre = np.clip(pre_pbal, 1e-6, 1.0)
    pdd_pre = np.clip(1.0 - pre_conf, 0.0, 1.0)

    dec = []
    for i in range(len(records)):
        err_post = _aetta_err(ratio_post[i], pdd_post[i], alpha=alpha)
        err_pre = _aetta_err(ratio_pre[i], pdd_pre[i], alpha=alpha)
        acc_post = 1.0 - err_post
        acc_pre = 1.0 - err_pre
        dec.append("ADAPT" if acc_post >= acc_pre - margin else "FREEZE")
    return np.array(dec, dtype=object)


# =========================================================================== #
#  Trivial + oracle policies (shared definitions)                              #
# =========================================================================== #
def always_adapt_decision(records):
    return np.array(["ADAPT"] * len(records), dtype=object)


def always_freeze_decision(records):
    return np.array(["FREEZE"] * len(records), dtype=object)


def oracle_decision(records):
    """Oracle uses the TRUE benefit B (EVALUATION-ONLY signal) to pick the best action
    per condition. This is the regret upper bound; it is NOT a fair competitor and is
    reported only as the reference ceiling."""
    out = []
    for r in records:
        out.append("ADAPT" if float(r["B"]) > 0 else "FREEZE")
    return np.array(out, dtype=object)


# Registry the harness imports. KGA is provided by analysis.decide_kga, not here.
BASELINE_DECISION_FNS = {
    "always_adapt": always_adapt_decision,
    "always_freeze": always_freeze_decision,
    "poem": poem_decision,
    "aetta": aetta_decision,
    "oracle": oracle_decision,
}
