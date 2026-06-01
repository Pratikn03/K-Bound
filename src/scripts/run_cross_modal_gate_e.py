"""Gate-E research run: can a stronger CROSS-MODAL fusion beat the
confidence-weighted baseline (CW) on CLEAN external transfer?

Honest protocol:
  - Fit/select fusion rules on the 3D-ADAM VALIDATION split only (labels present).
  - Report the validation-selected rule ONCE on the 3D-ADAM TEST split.
  - Compare the selected rule to CW (the parameter-free baseline that previously
    beat RGA) AND to SAR (the frozen strongest baseline that defines Gate E),
    each with a per-sample paired bootstrap 95% CI.

Gate-E (contract) passes iff the method beats SAR with CI lower bound > 0.
The stronger bar we also report: beating CW (the actual best clean method).

Writes experiments/fusion/cross_modal_gate_e_result.json.
"""

from __future__ import annotations

import glob
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from uais.fusion.attention.cross_modal_fusion import select_and_eval  # noqa: E402

CSV = ROOT / "experiments/fusion/m2_external_3d_adam_v3_inputs.csv"
TRANSFER_ARCHIVE = ROOT / "elara_master_c/predictions/v3_transfer"


def _pivot(df, split):
    s = df[df["split"] == split]
    r = s[s.domain == "rgb"].set_index("sample_id")
    d = s[s.domain == "depth_or_xyz"].set_index("sample_id")
    ids = r.index.intersection(d.index)
    return (r.loc[ids, "score"].to_numpy(), d.loc[ids, "score"].to_numpy(),
            r.loc[ids, "label"].to_numpy().astype(int), ids)


def _sar_test_scores(ids):
    """Load the frozen SAR baseline's per-sample test scores from the archive."""
    fs = [f for f in glob.glob(f"{TRANSFER_ARCHIVE}/*/sar_score_adapter/test/seed_42.parquet")
          if not Path(f).name.startswith("._")]
    if not fs:
        return None
    sar = pd.read_parquet(fs[0]).set_index("sample_id")
    common = [i for i in ids if i in sar.index]
    return sar.loc[common, "raw_score"].to_numpy(), common


def _boot(y, a, b, n_iter=10000, seed=0):
    rng = np.random.default_rng(seed)
    n = len(y)
    ds = []
    for _ in range(n_iter):
        i = rng.integers(0, n, n)
        if len(np.unique(y[i])) < 2:
            continue
        ds.append(roc_auc_score(y[i], a[i]) - roc_auc_score(y[i], b[i]))
    return (float(roc_auc_score(y, a) - roc_auc_score(y, b)),
            float(np.percentile(ds, 2.5)), float(np.percentile(ds, 97.5)))


def main() -> int:
    df = pd.read_csv(CSV)
    va, vb, vy, _ = _pivot(df, "validation")
    ta, tb, ty, tids = _pivot(df, "test")

    res = select_and_eval(va, vb, vy, ta, tb, ty)

    # selected-rule test scores (recompute to get the per-sample vector)
    from uais.fusion.attention.cross_modal_fusion import (FUSION_RULES,
                                                          _fit_copula,
                                                          _fit_logistic)
    rules = dict(FUSION_RULES)
    rules["logistic_xmodal"] = _fit_logistic(va, vb, vy)
    rules["copula_lite"] = _fit_copula(va, vb, vy)
    sel_scores = rules[res.selected_rule](ta, tb)
    cw_scores = rules["cw"](ta, tb)

    out = {
        "protocol": "3D-ADAM clean external transfer; fusion selected on VALIDATION only",
        "selected_rule": res.selected_rule,
        "val_auroc_selected": res.val_auroc_selected,
        "test_auroc_selected": res.test_auroc_selected,
        "test_auroc_cw": res.test_auroc_cw,
        "all_val_auroc": res.all_val_auroc,
        "all_test_auroc": res.all_test_auroc,
    }

    # vs CW (the strong clean baseline)
    d_cw, lo_cw, hi_cw = _boot(ty, sel_scores, cw_scores)
    out["selected_vs_cw"] = {"delta": d_cw, "ci95": [lo_cw, hi_cw], "beats_cw": bool(lo_cw > 0)}

    # vs SAR (the Gate-E contract comparator)
    sar = _sar_test_scores(list(tids))
    if sar is not None:
        sar_scores, common = sar
        # align selected scores to the SAR-common ids
        id_to_idx = {i: k for k, i in enumerate(tids)}
        idx = [id_to_idx[i] for i in common]
        sel_c = sel_scores[idx]
        y_c = ty[idx]
        d_sar, lo_sar, hi_sar = _boot(y_c, sel_c, sar_scores)
        out["selected_vs_sar"] = {"delta": d_sar, "ci95": [lo_sar, hi_sar],
                                  "gate_e_pass": bool(lo_sar > 0)}

    (ROOT / "experiments/fusion/cross_modal_gate_e_result.json").write_text(json.dumps(out, indent=2))

    print("=== Cross-modal Gate-E research run (validation-selected) ===")
    print(f"  selected rule: {res.selected_rule}  (val AUROC {res.val_auroc_selected:.4f})")
    print("  test AUROC by rule:")
    for k, v in sorted(res.all_test_auroc.items(), key=lambda kv: -kv[1]):
        mark = " <- selected" if k == res.selected_rule else (" (CW baseline)" if k == "cw" else "")
        print(f"    {k:18s} {v:.4f}{mark}")
    print(f"\n  selected vs CW:  delta={d_cw:+.4f} CI=[{lo_cw:+.4f},{hi_cw:+.4f}]  beats_CW={out['selected_vs_cw']['beats_cw']}")
    if "selected_vs_sar" in out:
        s = out["selected_vs_sar"]
        print(f"  selected vs SAR: delta={s['delta']:+.4f} CI={s['ci95']}  GATE-E PASS={s['gate_e_pass']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
