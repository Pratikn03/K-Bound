"""
build_results_manifest.py - consolidate the iWildCam K-Bound result into ONE manifest
under experiments/kbound/results/, fusing the source-calibrated verdict(s) with the
f0 provenance so every headline number traces to a file.
"""
from __future__ import annotations
import argparse, json, time
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
RES = REPO / "experiments/kbound/results"


def load(p):
    try:
        return json.load(open(p))
    except Exception as e:
        return {"_load_error": repr(e), "_path": str(p)}


def thin_verdict(v):
    """Keep headline + routing summaries; drop nothing essential but avoid records."""
    return v  # verdicts are already compact (no per-sample records)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--val-verdict", default=str(RES / "iwildcam_full_val/VERDICT_val.json"))
    ap.add_argument("--test-verdict", default=str(RES / "iwildcam_full_test/VERDICT_test.json"))
    ap.add_argument("--f0-log", default=str(RES / "iwildcam_f0_erm/trainlog_seed0.json"))
    ap.add_argument("--source-manifest", default=str(RES / "iwildcam_full_idval"))
    ap.add_argument("--out", default=str(RES / "iwildcam_kbound_RESULTS.json"))
    args = ap.parse_args()

    f0log = load(args.f0_log)
    best = f0log.get("best") if isinstance(f0log, dict) else None
    val_v = load(args.val_verdict)
    test_v = load(args.test_verdict) if Path(args.test_verdict).exists() else {"_note": "test target not run"}

    # locate source manifest file
    srcdir = Path(args.source_manifest)
    src_files = sorted(srcdir.glob("result_*.json")) if srcdir.is_dir() else []
    src_meta = {}
    if src_files:
        sm = load(src_files[0])
        src_meta = {"path": str(src_files[0]), "config": sm.get("config"),
                    "f0_quick_eval": sm.get("f0_quick_eval"), "host": sm.get("host"),
                    "n_records": len(sm.get("records", [])), "n_conditions": len(sm.get("conditions", []))}

    def headline(v):
        if not isinstance(v, dict) or "regime" not in v:
            return v
        return {
            "metric": v.get("metric"), "regime": v.get("regime"),
            "goldilocks": v.get("goldilocks"), "beats_both": v.get("beats_both"),
            "beats_both_route_a_deployed": v.get("beats_both_route_a_deployed"),
            "beats_both_route_b_multicand": v.get("beats_both_route_b_multicand"),
            "deployed_adapter_chosen_on_source": v.get("deployed_adapter_chosen_on_source"),
            "target_benefit": v.get("target_benefit"),
            "detectability_target": {k: v.get("detectability_target", {}).get(k) for k in
                                     ("base_rate_harmful", "best_single_feature_harm_AUC",
                                      "certificate_harm_AUC_sourcefit_on_target", "detectability_verdict")},
            "n_target_conditions": v.get("n_target_conditions"),
        }

    manifest = {
        "schema": "kbound_iwildcam_RESULTS_v1",
        "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "dataset": "wilds-iwildcam-v2.0",
        "metric": "macro_f1 (WILDS-standard)",
        "candidate_number": 5,
        "prior_real_shift_candidates": ["Camelyon17", "scale-invariant tau_hat", "ACDC", "structural"],
        "integrity": {
            "f0": "ResNet-50 ERM full fine-tune from ImageNet; plain shuffle; "
                  "model selection on id_val macro-F1; OOD val/test never seen in training/selection.",
            "calibration": "Source = id_val. KGA certificate (Z->B), conformal eps, route-(b) tau*, "
                           "and the deployed adapter are ALL fit on SOURCE only. Target labels touch "
                           "ONLY final scoring. No test-set tuning.",
            "metric_recompute": "all a0/aa/B recomputed offline from stored per-sample preds + eval_y.",
        },
        "f0_provenance": {
            "checkpoint": str(RES / "iwildcam_f0_erm/f0_resnet50_erm_seed0.pt"),
            "best_epoch": best, "trainlog": args.f0_log,
            "epochs_logged": (f0log.get("epochs") if isinstance(f0log, dict) else None),
        },
        "source_manifest": src_meta,
        "headline_val": headline(val_v),
        "headline_test": headline(test_v),
        "verdict_val_full": val_v,
        "verdict_test_full": test_v,
        "artifacts": {
            "runner": "experiments/kbound/wilds/run_iwildcam_kbound.py",
            "f0_trainer": "experiments/kbound/wilds/train_iwildcam_f0.py",
            "analyzer": "experiments/kbound/wilds/analyze_iwildcam_kbound.py",
            "val_verdict": args.val_verdict, "test_verdict": args.test_verdict,
        },
    }
    json.dump(manifest, open(args.out, "w"), indent=2)
    print(f"results manifest -> {args.out}")
    hv = manifest["headline_val"]
    print(json.dumps(hv, indent=1) if isinstance(hv, dict) else str(hv))


if __name__ == "__main__":
    main()
