"""Fusion test A v2 (DEVELOPMENT) — refined reliability-gated fusion, with a
weight floor + reliability x per-sample-sharpness blend, and VALIDATION-ONLY
rule selection (no test labels used for any choice).

Fixes the v1 failure mode (the simple "weight by validation AUROC" rule
over-commits to a single modality and loses the blend, e.g. ferrite_bead). The
refined family keeps a floor so reliable-but-not-dominant modalities still
contribute, and blends per-category reliability with per-sample sharpness (the
actual RGA philosophy: trust a modality that is BOTH reliable on validation AND
confident on this sample).

Integrity:
  - Per-modality scores: strong detectors (deep PatchCore rgb/ps, PCD geometry xyz).
  - Reliability weights: per-category VALIDATION AUROC (val labels only).
  - Rule FORM (rel_only / rel_floor / rel_x_sharp) selected by pooled VALIDATION
    AUROC, then evaluated once on test. No test-label selection.
  - Holdout is OPENED -> DEVELOPMENT evidence; cannot set the official gate flag.
  - Per-category scores cached to npz so rule iteration needs no re-scoring.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import zipfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from sklearn.metrics import roc_auc_score  # noqa: E402

from scripts.scenario_c.prepare_realiad_d3_headroom_inputs import _load_rows  # noqa: E402
from uais.fusion.attention.realiad_3d_detector import (  # noqa: E402
    load_modality_image,
    score_one_class_patchcore,
)

MODALITIES = ("rgb", "ps", "xyz")
CACHE_DIR = ROOT / "experiments/fusion/realiad_d3_score_cache"
# Bump when detector scoring changes so stale caches can't silently persist
# (the 2026-06-02 binary-PCD audit fix is why this guard exists).
DETECTOR_VERSION = "v2_binpcd"


def _zsig(raw, ref):
    mu = float(np.mean(ref)); sd = float(np.std(ref) + 1e-6)
    return 1.0 / (1.0 + np.exp(-(raw - mu) / sd))


def _strat(rs, cap):
    pos = [r for r in rs if r.label == 1][: cap // 2]
    neg = [r for r in rs if r.label == 0][: cap // 2]
    return pos + neg


def score_category(cat, coreset, max_train, max_val, max_test):
    """Return aligned (S_val[N,3], yval, S_test[M,3], ytest, val_auc[3]) with caching."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache = CACHE_DIR / f"{cat}_{max_train}_{max_val}_{max_test}_{coreset}_{DETECTOR_VERSION}.npz"
    if cache.exists():
        z = np.load(cache)
        return z["Sval"], z["yval"], z["Stest"], z["ytest"], z["valauc"]
    rows, _ = _load_rows(ROOT / "data/raw/realiad_d3", [cat])
    zf = zipfile.ZipFile(ROOT / "data/raw/realiad_d3/realiad_d3_raw" / f"{cat}.zip")
    val_s, test_s, val_auc = {}, {}, {}
    val_lab = test_lab = None
    for m in MODALITIES:
        mrows = [r for r in rows if r.modality == m]
        train = [r for r in mrows if r.split == "train" and r.label == 0][:max_train]
        vrows = _strat([r for r in mrows if r.split == "validation"], max_val)
        trows = _strat([r for r in mrows if r.split == "test"], max_test)
        if len(train) < 10 or len(vrows) < 10 or len(trows) < 10:
            return None
        bank = [load_modality_image(zf, r.zip_member, m) for r in train]
        ref = score_one_class_patchcore(bank[: min(40, len(bank))], bank, coreset_size=coreset)
        v = _zsig(score_one_class_patchcore(bank, [load_modality_image(zf, r.zip_member, m) for r in vrows], coreset_size=coreset), ref)
        t = _zsig(score_one_class_patchcore(bank, [load_modality_image(zf, r.zip_member, m) for r in trows], coreset_size=coreset), ref)
        val_s[m] = {r.sample_id: s for r, s in zip(vrows, v)}
        test_s[m] = {r.sample_id: s for r, s in zip(trows, t)}
        if val_lab is None:
            val_lab = {r.sample_id: r.label for r in vrows}
            test_lab = {r.sample_id: r.label for r in trows}
        val_auc[m] = float(roc_auc_score([val_lab[i] for i in val_s[m]], [val_s[m][i] for i in val_s[m]]))
    vcom = sorted(set.intersection(*[set(val_s[m]) for m in MODALITIES]))
    tcom = sorted(set.intersection(*[set(test_s[m]) for m in MODALITIES]))
    Sval = np.column_stack([[val_s[m][i] for i in vcom] for m in MODALITIES])
    Stest = np.column_stack([[test_s[m][i] for i in tcom] for m in MODALITIES])
    yval = np.array([val_lab[i] for i in vcom], int)
    ytest = np.array([test_lab[i] for i in tcom], int)
    valauc = np.array([val_auc[m] for m in MODALITIES], float)
    np.savez(cache, Sval=Sval, yval=yval, Stest=Stest, ytest=ytest, valauc=valauc)
    return Sval, yval, Stest, ytest, valauc


# ---- fusion rules (each maps S[N,3] + val_auc[3] -> fused score[N]) ----
def _cw(S, valauc):
    w = 2.0 * np.abs(S - 0.5)
    return (S * w).sum(1) / np.clip(w.sum(1), 1e-9, None)


def _rel(valauc):
    r = np.clip(valauc - 0.5, 0.0, None)
    return r / r.sum() if r.sum() > 1e-9 else np.ones(len(valauc)) / len(valauc)


def _rel_only(S, valauc):
    return S @ _rel(valauc)


def _rel_floor(S, valauc, floor=0.15):
    r = np.clip(valauc - 0.5, 0.0, None) + floor * (valauc > 0.5)
    r = r / r.sum() if r.sum() > 1e-9 else np.ones(len(valauc)) / len(valauc)
    return S @ r


def _rel_x_sharp(S, valauc, floor=0.05):
    rel = np.clip(valauc - 0.5, 0.0, None) + floor * (valauc > 0.5)
    w = (rel[None, :]) * (2.0 * np.abs(S - 0.5))  # per-sample reliability x sharpness
    return (S * w).sum(1) / np.clip(w.sum(1), 1e-9, None)


GATED_RULES = {"rel_only": _rel_only, "rel_floor": _rel_floor, "rel_x_sharp": _rel_x_sharp}


def _boot(y, cand, comp, n_iter=10000, seed=0):
    rng = np.random.default_rng(seed); n = len(y); d = []
    for _ in range(n_iter):
        i = rng.integers(0, n, n)
        if len(np.unique(y[i])) < 2:
            continue
        d.append(roc_auc_score(y[i], cand[i]) - roc_auc_score(y[i], comp[i]))
    return {"delta": float(roc_auc_score(y, cand) - roc_auc_score(y, comp)),
            "ci95": [float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))],
            "ci_low_gt_0": bool(np.percentile(d, 2.5) > 0)}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--categories", nargs="*",
                    default=["knob_cap", "ferrite_bead", "fuse_holder",
                             "audio_jack_socket", "dc_power_connector", "ethernet_connector"])
    ap.add_argument("--coreset", type=int, default=4096)
    ap.add_argument("--max-train", type=int, default=70)
    ap.add_argument("--max-val", type=int, default=90)
    ap.add_argument("--max-test", type=int, default=130)
    ap.add_argument("--out", type=Path, default=ROOT / "experiments/fusion/realiad_d3_fusion_test_a_v2.json")
    ap.add_argument("--fixed-rule", choices=list(GATED_RULES), default=None,
                    help="Freeze the fusion rule (skip val re-selection); use for the held-out confirmation.")
    args = ap.parse_args()
    print(f"=== Fusion test A v2 (development, refined+val-selected): {args.categories} ===", flush=True)

    per_cat = {}
    val_blocks, test_blocks = [], []  # (S, y, valauc) per category
    for cat in args.categories:
        t0 = time.time()
        out = score_category(cat, args.coreset, args.max_train, args.max_val, args.max_test)
        if out is None:
            per_cat[cat] = {"skipped": True}; continue
        Sval, yval, Stest, ytest, valauc = out
        val_blocks.append((Sval, yval, valauc)); test_blocks.append((Stest, ytest, valauc))
        per_cat[cat] = {"val_auc": {m: round(float(a), 4) for m, a in zip(MODALITIES, valauc)},
                        "n_val": int(len(yval)), "n_test": int(len(ytest)), "secs": round(time.time() - t0, 1)}
        print(f"[{cat}] valAUC={per_cat[cat]['val_auc']} (n_test {len(ytest)}, {time.time()-t0:.0f}s)", flush=True)

    # build pooled val/test fused scores per rule
    def pooled(blocks, rule_fn):
        ys, fs, cws = [], [], []
        for S, y, va in blocks:
            ys.append(y); fs.append(rule_fn(S, va)); cws.append(_cw(S, va))
        return np.concatenate(ys), np.concatenate(fs), np.concatenate(cws)

    # VALIDATION-ONLY rule selection (or a FROZEN pre-registered rule for held-out)
    val_scores = {}
    for name, fn in GATED_RULES.items():
        yv, fv, _ = pooled(val_blocks, fn)
        val_scores[name] = float(roc_auc_score(yv, fv))
    if args.fixed_rule:
        selected = args.fixed_rule  # frozen: no re-selection on held-out data
    else:
        selected = max(val_scores, key=val_scores.get)

    # evaluate selected rule on TEST vs CW (+ report all rules on test for transparency)
    yt, ft_sel, cw_t = pooled(test_blocks, GATED_RULES[selected])
    test_auc = {name: float(roc_auc_score(*pooled(test_blocks, fn)[:2])) for name, fn in GATED_RULES.items()}
    report = {
        "protocol": "FUSION_TEST_A_v2_development",
        "holdout_status": "OPENED_DEVELOPMENT_ONLY",
        "rule_selection": "validation pooled AUROC (val labels only)",
        "val_auc_by_rule": {k: round(v, 4) for k, v in val_scores.items()},
        "selected_rule": selected,
        "per_category": per_cat,
        "pooled": {
            "n_test": int(len(yt)),
            "auc_cw": round(float(roc_auc_score(yt, cw_t)), 4),
            "auc_gated_selected": round(float(roc_auc_score(yt, ft_sel)), 4),
            "test_auc_all_rules": {k: round(v, 4) for k, v in test_auc.items()},
            "gated_vs_cw": _boot(yt, ft_sel, cw_t),
        },
    }
    # per-category selected-rule deltas
    for (S, y, va), cat in zip(test_blocks, [c for c in args.categories if not per_cat.get(c, {}).get("skipped")]):
        g = GATED_RULES[selected](S, va); c = _cw(S, va)
        per_cat[cat]["cw"] = round(float(roc_auc_score(y, c)), 4)
        per_cat[cat]["gated"] = round(float(roc_auc_score(y, g)), 4)
        per_cat[cat]["delta"] = round(per_cat[cat]["gated"] - per_cat[cat]["cw"], 4)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2))

    print("\n=== VALIDATION rule selection (val labels only) ===")
    for k, v in val_scores.items():
        print(f"  {k}: val AUROC {v:.4f}" + ("  <- selected" if k == selected else ""))
    p = report["pooled"]; g = p["gated_vs_cw"]
    print("\n=== POOLED TEST (development) ===")
    print(f"  CW={p['auc_cw']}  GATED[{selected}]={p['auc_gated_selected']}  all_rules={p['test_auc_all_rules']}")
    print(f"  GATED vs CW: delta={g['delta']:+.4f}  CI95={[round(x,4) for x in g['ci95']]}  CI_low>0={g['ci_low_gt_0']}")
    print("  per-category:")
    for cat in args.categories:
        d = per_cat.get(cat, {})
        if "delta" in d:
            print(f"    {cat:<22} CW={d['cw']} gated={d['gated']} delta={d['delta']:+.4f}")
    print("  VERDICT:", "GATE BEATS CW (dev)" if g["ci_low_gt_0"] else "tie (dev)")
    print(f"Wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
