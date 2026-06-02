"""DEVELOPMENT re-analysis of the D18 held-out cache with the degenerate-channel guard.

NOT a confirmation. Real-IAD-D3 is opened (D17/D18); this re-uses the opened
held-out score cache to answer a development question only: once degenerate
channels are removed *fairly from both the CW comparator and the gated rule*,
how much of the apparent reliability-gating advantage survives?

Finding (2026-06-02): the unguarded pooled gated-vs-CW delta (+0.045) is almost
entirely the lego_propeller degenerate-channel artifact. With the guard applied
to both sides, the pooled delta collapses to ~+0.005 (below the +0.010 bar) and
survives only in categories with >=2 genuinely reliable channels. The guard's
main effect is to make CW *robust* (0.700 -> 0.730 pooled), not to validate the
gate. See docs/research/phase3/D18_HELDOUT_CONFIRMATION_AUDIT_2026_06_02.md.

Threshold caveat: the guard uses general degeneracy thresholds (DEFAULT_* in
elara.evaluation.degenerate_channel_guard) motivated by validation-distribution
reasoning, not tuned to maximise this delta. No test labels are used for any
selection here.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

from elara.evaluation.degenerate_channel_guard import guarded_channel_mask

ROOT = Path(__file__).resolve().parents[3]
CACHE = ROOT / "experiments/fusion/realiad_d3_score_cache"
SUFFIX = "70_90_130_4096_v2_binpcd"
HELDOUT = [
    "connector_housing_female", "crimp_st_cable_mount_box", "fork_crimp_terminal",
    "headphone_jack_socket", "lattice_block_plug", "lego_pin_connector_plate",
    "lego_propeller", "limit_switch", "miniature_lifting_motor", "purple_clay_pot",
    "telephone_spring_switch",
]


def _cw(s):
    w = 2.0 * np.abs(s - 0.5)
    return (s * w).sum(1) / np.clip(w.sum(1), 1e-9, None)


def _rel_x_sharp(s, va, floor=0.05):
    rel = np.clip(va - 0.5, 0.0, None) + floor * (va > 0.5)
    w = rel[None, :] * (2.0 * np.abs(s - 0.5))
    return (s * w).sum(1) / np.clip(w.sum(1), 1e-9, None)


def _guard(s, va, mask):
    s2, va2 = s.copy(), va.copy()
    s2[:, ~mask] = 0.5   # neutral score -> zero sharpness vote
    va2[~mask] = 0.5     # zero reliability
    return s2, va2


def main() -> int:
    per_cat = {}
    pool = {"y": [], "cw_u": [], "g_u": [], "cw_g": [], "g_g": []}
    for cat in HELDOUT:
        z = np.load(CACHE / f"{cat}_{SUFFIX}.npz")
        sval, yval, stest, ytest, va = (z["Sval"], z["yval"], z["Stest"],
                                        z["ytest"], z["valauc"])
        mask = guarded_channel_mask(sval, yval)
        sg, vag = _guard(stest, va, mask)
        cw_u, g_u = _cw(stest), _rel_x_sharp(stest, va)
        cw_g, g_g = _cw(sg), _rel_x_sharp(sg, vag)
        per_cat[cat] = {
            "channels_kept": int(mask.sum()),
            "unguarded": {"cw": float(roc_auc_score(ytest, cw_u)),
                          "gated": float(roc_auc_score(ytest, g_u))},
            "guarded": {"cw": float(roc_auc_score(ytest, cw_g)),
                        "gated": float(roc_auc_score(ytest, g_g))},
        }
        per_cat[cat]["delta_unguarded"] = (per_cat[cat]["unguarded"]["gated"]
                                           - per_cat[cat]["unguarded"]["cw"])
        per_cat[cat]["delta_guarded"] = (per_cat[cat]["guarded"]["gated"]
                                         - per_cat[cat]["guarded"]["cw"])
        pool["y"].append(ytest)
        pool["cw_u"].append(cw_u); pool["g_u"].append(g_u)
        pool["cw_g"].append(cw_g); pool["g_g"].append(g_g)

    y = np.concatenate(pool["y"])

    def auc(key):
        return float(roc_auc_score(y, np.concatenate(pool[key])))

    dG = [per_cat[c]["delta_guarded"] for c in HELDOUT]
    result = {
        "protocol": "GUARDED_CHANNEL_DEV_REANALYSIS_v1",
        "status": "DEVELOPMENT_ONLY_OPENED_DATA_NOT_A_CONFIRMATION",
        "pooled": {
            "unguarded": {"cw": auc("cw_u"), "gated": auc("g_u"),
                          "delta": auc("g_u") - auc("cw_u")},
            "guarded": {"cw": auc("cw_g"), "gated": auc("g_g"),
                        "delta": auc("g_g") - auc("cw_g")},
        },
        "within_category_mean_delta": {
            "unguarded": float(np.mean([per_cat[c]["delta_unguarded"] for c in HELDOUT])),
            "guarded": float(np.mean(dG)),
        },
        "guarded_categories_gated_beats_cw": int(sum(d > 0.005 for d in dG)),
        "guarded_categories_gated_loses": int(sum(d < -0.005 for d in dG)),
        "per_category": per_cat,
        "interpretation": (
            "Most of the unguarded +0.045 pooled delta was the degenerate-channel "
            "artifact. Guarding both sides raises CW (robustness) and shrinks the "
            "gating advantage to ~+0.005 pooled, surviving only where >=2 channels "
            "are genuinely reliable. Detector quality (more non-degenerate channels "
            "per category) is the gating headroom bottleneck."
        ),
    }
    out = ROOT / "experiments/fusion/guarded_channel_dev_analysis.json"
    out.write_text(json.dumps(result, indent=2))
    print(f"pooled  unguarded delta={result['pooled']['unguarded']['delta']:+.4f}  "
          f"guarded delta={result['pooled']['guarded']['delta']:+.4f}")
    print(f"within-cat mean delta  unguarded="
          f"{result['within_category_mean_delta']['unguarded']:+.4f}  "
          f"guarded={result['within_category_mean_delta']['guarded']:+.4f}")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
