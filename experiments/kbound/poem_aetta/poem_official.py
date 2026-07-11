"""
poem_official.py - FAITHFUL per-sample POEM decision for the WIN_HUNT_v4 arm_D
head-to-head (closes the "protocol-matched ports" caveat for POEM).

POEM  =  Protected Online Entropy Matching
         Bar, Shaer, Romano, "Protected Test-Time Adaptation via Online Entropy
         Matching: A Betting Approach", NeurIPS 2024, arXiv:2408.07511.
         Official code: github.com/yarinbar/poem (protector.py + cdf.py).

Unlike experiments/kbound/poem_aetta/baselines.py::poem_decision (which drives the
official betting martingale with a single BATCH-SUMMARY entropy per condition -- the
(S1) simplification the paper's port caveat is about), THIS module drives the *exact
same* official betting martingale over the RAW PER-SAMPLE entropy stream logged by the
patched stress runner (cifar_tent_mps_v2.py --log-samples). That is the only change
required to make POEM here per-sample-faithful; the betting function, the SF-OGD update
of the betting fraction, the wealth recursion, and the source-entropy CDF are all reused
verbatim from the official-code port in baselines.py.

--------------------------------------------------------------------------------------
WHAT THE PAPER FIXES vs WHAT IT LEAVES OPEN (every open choice is pinned here)
--------------------------------------------------------------------------------------
Fixed by the official code (reused from baselines.py, do NOT re-tune):
  * betting factor           b_t = 1 + eps_t * (u_t - 0.5)          (protector.py::protect_u)
  * wealth / test-martingale C_t = C_{t-1} * b_t,  C_0 = 1          (protector.py)
  * betting fraction eps_t   learned online by scale-free OGD       (protector.py::sfogd)
  * SF-OGD constants         gamma = 2/sqrt(3),  eps_clip D = 1.80  (repo defaults)
  * PIT statistic            u_t = CDF_source(entropy_t)            (cdf.py, scipy ecdf + interp)
  * detection / alarm        reject H0 (shift) when wealth >= 1/alpha  (Ville's inequality;
                             the standard betting-wealth e-value alarm at level alpha)

Left open by the paper -> PINNED here (documented, frozen before scoring):
  (P1) DECISION GRANULARITY. POEM is an online adapter, not a per-condition binary
       router; to feed the existing head-to-head scorer we must emit one ADAPT/FREEZE
       label per condition. We run ONE martingale per condition over that condition's
       per-sample entropy stream (reset C=1 between conditions -- each condition is an
       independent deployment). The stream order is the logged eval-pool order (a fixed
       i.i.d. shuffle per condition).
  (P2) SHIFT-DETECTION vs MODEL-PROTECTION, made explicit as a TWO-SIDED entropy-match
       test (the faithful reading of POEM's "match test entropy to source, no-harm when
       matched"):
         shift_in   = martingale on the FROZEN model's per-sample entropies vs source CDF
                      fires (>= 1/alpha)  ==  the incoming stream is OOD -> adaptation is warranted.
         overshoot  = the ADAPTED model OVER-minimized entropy, i.e. it drove the entropy
                      distribution significantly BELOW source (POEM's explicit failure mode:
                      over-confidence / collapse). Detected as a LOWER-TAIL betting alarm:
                      the martingale on the adapted per-sample entropies fires (>= 1/alpha)
                      AND the mean source-PIT of the adapted entropies is < 0.5 (they sit in
                      the lower part of the source distribution). A BENIGN adaptation moves
                      entropies TOWARD source (PIT ~ 0.5, or above if still uncertain) and is
                      NOT flagged -- only over-minimization below source is.
       Decision:  ADAPT iff (shift_in and not overshoot)  else FREEZE (protect).
         - matched / clean condition (no shift_in)                    -> FREEZE  (POEM no-harm)
         - shifted + entropy moved toward source (no overshoot)       -> ADAPT   (POEM success)
         - shifted + entropy over-minimized below source (overshoot)  -> FREEZE  (POEM protects)
       IMPORTANT (why the lower-tail, not a generic "mismatch"): the ADAPTED model here is
       produced by the stress runner's Tent/EATA/SAR (entropy MINIMIZATION), NOT by POEM's
       own entropy-MATCHING transport, so its entropies never match source exactly; a
       two-sided "differs-from-source" martingale would fire on essentially every stream
       (and freeze even helpful cells) over a long per-sample stream. POEM's protection is
       specifically against OVER-minimization (entropies pushed below source -> over-
       confidence / Tent-collapse), which is exactly the lower-tail alarm above and is the
       discriminating signal for the Tent-collapse cells the stress grid targets (a
       collapsed model has entropies ~0 -> PIT ~ 0 << 0.5 -> lower-tail fires strongly).
       Note: on an all-corrupted grid shift_in is typically satisfied for every condition,
       so the discriminative work is done by the overshoot protection -- the faithful POEM
       behaviour of "adapt under shift, protect against over-minimization".
       Variant "frozen_only" (baselines.py's original rule: shift_in and mean-entropy_drop>0)
       is provided for cross-reference but is NOT the default.
  (P3) SOURCE CDF. POEM's F_source is the empirical CDF of SOURCE-DOMAIN entropies of the
       ORIGINAL model. We build it from samples_source_<bench>_seed<k>.npz (frozen model's
       per-sample entropy on the CLEAN eval pool) written by the runner. If that file is
       absent we fall back, label-free, to the lowest-mean-frozen-entropy quartile of the
       run's conditions (the same "low_severity" reference discipline as baselines.py).
  (P4) DETECTION LEVEL alpha. Default alpha = 0.05 => alarm at wealth >= 20. Frozen here;
       exposed only so the scorer can record it. (baselines.py used log10>=2, i.e. alpha=0.01;
       0.05 is the conventional betting-test level and is fixed before scoring.)

Pure numpy (+ scipy via baselines' CDF). No torch. Consumes only the logged .npz streams.
"""
from __future__ import annotations
import os
import sys
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import baselines as BL   # official-code port: PoemProtector (protector.py), SourceEntropyCDF (cdf.py)

DEFAULT_ALPHA = 0.05     # (P4) detection level; alarm threshold = 1/alpha
GAMMA = 2.0 / np.sqrt(3.0)
EPS_CLIP = 1.80


def _peak_wealth(cdf, entropy_stream, gamma=GAMMA, eps_clip=EPS_CLIP):
    """Run the OFFICIAL POEM betting martingale (baselines.PoemProtector, i.e.
    protector.py::protect_u + sfogd) over a per-sample entropy stream, return the
    PEAK wealth sup_t C_t. Betting martingales are anytime-valid, so the shift is
    certified the first time C_t crosses 1/alpha -> peak >= 1/alpha iff it ever fires."""
    stream = np.asarray(entropy_stream, float).ravel()
    stream = stream[np.isfinite(stream)]
    if stream.size == 0:
        return 1.0
    prot = BL.PoemProtector(cdf, gamma=gamma, eps_clip=eps_clip)
    peak = 1.0
    for z in stream:
        c = prot.step(float(z))
        if c > peak:
            peak = c
    return float(peak)


def _mean_pit(cdf, stream):
    """Mean source-PIT u = CDF_source(entropy) over a stream. u<0.5 => the entropies
    sit in the LOWER part of the source distribution (over-minimized / over-confident)."""
    s = np.asarray(stream, float).ravel()
    s = s[np.isfinite(s)]
    if s.size == 0:
        return 0.5
    return float(np.mean([cdf(float(z)) for z in s]))


def build_source_cdf(records, source_entropy=None):
    """(P3) Build POEM's source-entropy CDF. Prefer the logged clean source stream;
    else fall back label-free to the lowest-mean-frozen-entropy quartile of conditions."""
    if source_entropy is not None:
        arr = np.asarray(source_entropy, float).ravel()
        arr = arr[np.isfinite(arr)]
        if arr.size >= 4:
            return BL.SourceEntropyCDF(arr), "logged_clean_source"
    # fallback: pool frozen per-sample entropies from the low-entropy quartile of conditions
    means = np.array([float(np.nanmean(r["frozen_entropy"])) for r in records], float)
    thr = np.quantile(means, 0.25)
    pool = np.concatenate([np.asarray(r["frozen_entropy"], float).ravel()
                           for r, m in zip(records, means) if m <= thr]) if len(records) else np.array([])
    pool = pool[np.isfinite(pool)]
    if pool.size < 4:  # last resort: everything
        pool = np.concatenate([np.asarray(r["frozen_entropy"], float).ravel() for r in records])
        pool = pool[np.isfinite(pool)]
    return BL.SourceEntropyCDF(pool), "low_entropy_quartile_fallback"


def poem_official_decision(records, source_entropy=None, alpha=DEFAULT_ALPHA,
                           variant="matching", gamma=GAMMA, eps_clip=EPS_CLIP,
                           return_detail=False):
    """Per-condition ADAPT/FREEZE via the faithful per-sample POEM (P1-P4).

    Args:
      records: list of per-condition dicts, each with numpy arrays
               'frozen_entropy' and 'adapted_entropy' (the logged per-sample streams).
      source_entropy: 1-D array of clean SOURCE per-sample entropies (P3); may be None.
      alpha: detection level; martingale alarm at wealth >= 1/alpha (P4).
      variant: 'matching' (default, two-sided entropy-match test, P2) or 'frozen_only'
               (baselines.py's original shift_in and mean-entropy_drop>0 rule).
    Returns: np.array(dtype=object) of {"ADAPT","FREEZE"} (POEM never abstains);
             if return_detail, also a list of per-condition diagnostic dicts.
    """
    cdf, cdf_src = build_source_cdf(records, source_entropy)
    thresh = 1.0 / float(alpha)
    dec, detail = [], []
    for r in records:
        fe = np.asarray(r["frozen_entropy"], float)
        ae = np.asarray(r["adapted_entropy"], float)
        w_in = _peak_wealth(cdf, fe, gamma, eps_clip)
        shift_in = w_in >= thresh
        if variant == "frozen_only":
            edrop = float(np.nanmean(fe) - np.nanmean(ae))   # mean entropy_drop (adapted lower => >0)
            adapt = bool(shift_in and edrop > 0.0)
            w_out = pit_out = overshoot = None
        else:  # "matching" (default, P2): protect against OVER-minimization (lower-tail alarm)
            w_out = _peak_wealth(cdf, ae, gamma, eps_clip)
            pit_out = _mean_pit(cdf, ae)
            overshoot = bool(w_out >= thresh and pit_out < 0.5)
            adapt = bool(shift_in and not overshoot)
        dec.append("ADAPT" if adapt else "FREEZE")
        detail.append({"condition": r.get("condition"),
                       "peak_wealth_frozen": w_in, "peak_wealth_adapted": w_out,
                       "mean_pit_adapted": pit_out, "shift_in": bool(shift_in),
                       "overshoot": overshoot, "alarm_threshold": thresh,
                       "decision": dec[-1]})
    out = np.array(dec, dtype=object)
    if return_detail:
        return out, {"cdf_source": cdf_src, "alpha": alpha, "variant": variant,
                     "alarm_threshold": thresh, "per_condition": detail}
    return out
