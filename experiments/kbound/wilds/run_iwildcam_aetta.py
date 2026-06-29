"""
run_iwildcam_aetta.py - K-Bound iWildCam run with a STRONGER label-free harm detector.

Identical TTA / conditions / metric / eval to run_iwildcam_kbound.py, but additionally
computes, per condition, label-free ACCURACY estimates (AETTA dropout + frozen-reference
disagreement) for the frozen model and each adapted candidate. Stores the entropy Z too,
so the analyzer can compare entropy-baseline vs accuracy-estimator detectors apples-to-apples.

Online candidates only (fast preview); episodic deferred to a validated full run.
INTEGRITY: detector + threshold are calibrated on SOURCE in the analyzer, never on test.
"""
from __future__ import annotations
import argparse, gc, hashlib, json, os, platform, sys, time
from pathlib import Path
import numpy as np
import torch

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
import run_iwildcam_kbound as R
import tta_methods as tm
import aetta as AE

NUM_CLASSES = R.NUM_CLASSES
AGGR = R.AGGR


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", default=str(R.REPO / "experiments/kbound/data/wilds"))
    p.add_argument("--results-root", default=str(R.REPO / "experiments/kbound/results"))
    p.add_argument("--run-name", default="iwildcam_aetta_preview")
    p.add_argument("--ckpt", required=True)
    p.add_argument("--backbone", default="resnet50")
    p.add_argument("--split", default="val")
    p.add_argument("--seeds", type=int, nargs="+", default=[0])
    p.add_argument("--max-locations", type=int, default=3)
    p.add_argument("--compositions", nargs="+", default=["iid", "single_class"])
    p.add_argument("--batch-regimes", nargs="+", default=["tiny"], dest="batch_regimes")
    p.add_argument("--aggressiveness", nargs="+", default=["mild", "aggressive"])
    p.add_argument("--candidates", nargs="+", default=["tent_online", "eata_online", "sar_online"])
    p.add_argument("--n-eval", type=int, default=48, dest="n_eval")
    p.add_argument("--n-batches", type=int, default=2, dest="n_batches")
    p.add_argument("--eval-bs", type=int, default=48, dest="eval_bs")
    p.add_argument("--mc-M", type=int, default=8, dest="mc_M")
    p.add_argument("--mc-p", type=float, default=0.4, dest="mc_p")
    p.add_argument("--device", default="mps")
    args = p.parse_args(argv)

    device = tm.pick_device(args.device)
    out_dir = Path(args.results_root) / args.run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[aetta] split={args.split} device={device} M={args.mc_M} p={args.mc_p}", flush=True)
    f0 = R.make_model(args.backbone, device)
    obj = torch.load(args.ckpt, map_location=device, weights_only=False)
    state = obj["model"] if isinstance(obj, dict) and "model" in obj else obj
    f0.load_state_dict(state, strict=True); f0.eval()
    print(f"[f0] loaded {args.ckpt}", flush=True)

    _, sub, y, locations = R.get_iwildcam(args.data_root, args.split, train_tf=False)
    min_count = args.n_eval + max(R.BATCH_REGIMES[r] for r in args.batch_regimes) * args.n_batches
    loc_rows = R.select_locations(y, locations, args.max_locations, min_count)
    print("[locs] " + ", ".join(f"{l}(n={n},c={c})" for l, n, c in loc_rows), flush=True)
    cands = R.parse_candidates(args.candidates)

    records, conditions = [], []
    t0 = time.time()
    ci = 0
    ncell = len(args.seeds) * len(loc_rows) * len(args.compositions) * len(args.batch_regimes) * len(args.aggressiveness)
    for seed in args.seeds:
        for loc, loc_n, loc_classes in loc_rows:
            for comp in args.compositions:
                for regime in args.batch_regimes:
                    bs = R.BATCH_REGIMES[regime]
                    for aggr in args.aggressiveness:
                        ci += 1
                        tag = f"s{seed}/loc{loc}/{comp}/{regime}/{aggr}"
                        cseed = int(hashlib.sha256(tag.encode()).hexdigest()[:8], 16)
                        rng = np.random.default_rng(cseed); torch.manual_seed(cseed)
                        try:
                            stream, eval_x, eval_y = R.build_condition(
                                sub, y, locations, loc, comp, bs, args.n_eval, args.n_batches, rng, device)
                            steps = AGGR[aggr]["steps"]; lr = AGGR[aggr]["lr"]
                            p0 = tm._predict(f0, eval_x, train_mode=False, bs=args.eval_bs)
                            a0 = R.macro_f1(eval_y, p0)
                            est_frozen = AE.aetta_estacc(f0, eval_x, M=args.mc_M, p=args.mc_p, bn_train=False, bs=args.eval_bs)
                            for method, mode in cands:
                                fa, upd = tm._adapt(method, f0, stream, steps, lr, NUM_CLASSES)
                                preds = tm.predict_logits(fa, eval_x, train_mode=True, bs=args.eval_bs).argmax(1)
                                aa = R.macro_f1(eval_y, preds)
                                Z = tm.evidence_vector(f0, fa, stream[0], NUM_CLASSES, upd)
                                est_adapt = AE.aetta_estacc(fa, eval_x, M=args.mc_M, p=args.mc_p, bn_train=True, bs=args.eval_bs)
                                dacc_ref = AE.disagree_ref_estacc(preds, p0) - 1.0   # = -(disagree from frozen)
                                B = float(aa - a0)
                                records.append({
                                    "seed": int(seed), "location": int(loc), "comp": comp, "regime": regime,
                                    "aggr": aggr, "candidate": f"{method}_{mode}", "metric": "macro_f1",
                                    "a0": float(a0), "aa": float(aa), "B": B,
                                    "Z": [float(v) for v in Z],
                                    "est_frozen_aetta": float(est_frozen), "est_adapt_aetta": float(est_adapt),
                                    "dacc_aetta": float(est_adapt - est_frozen),
                                    "dacc_ref": float(dacc_ref),
                                    "preds": [int(v) for v in preds],
                                    "regime_label": ("helpful" if B > 0.02 else "harmful" if B < -0.02 else "marginal"),
                                })
                                del fa; tm.mps_free(); gc.collect()
                            conditions.append({"seed": int(seed), "location": int(loc), "comp": comp,
                                               "regime": regime, "aggr": aggr, "a0": float(a0),
                                               "eval_y": [int(v) for v in eval_y],
                                               "preds_frozen": [int(v) for v in p0]})
                            print(f"  [{ci}/{ncell}] {tag} a0={a0:.3f} estF={est_frozen:.3f} "
                                  f"B*={max(r['B'] for r in records[-len(cands):]):+.3f} "
                                  f"dacc*={max(r['dacc_aetta'] for r in records[-len(cands):]):+.3f}", flush=True)
                        except Exception as e:
                            print(f"  [{ci}/{ncell}] {tag} ERROR {repr(e)[:160]}", flush=True)
                        finally:
                            R.atomic_dump({"records": records, "conditions": conditions}, out_dir / "_partial.json")
                            tm.mps_free(); gc.collect()
    manifest = {
        "schema": "kbound_iwildcam_aetta_v1", "dataset": "wilds-iwildcam", "metric": "macro_f1",
        "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "host": {"node": platform.node(), "torch": torch.__version__, "mps": bool(torch.backends.mps.is_available())},
        "config": {k: getattr(args, k) for k in vars(args)},
        "detectors": ["aetta_dropout_dacc", "frozen_ref_dacc", "entropy_Z(11d)"],
        "evidence_names": tm.EVIDENCE_NAMES,
        "n_records": len(records), "n_conditions": len(conditions),
        "records": records, "conditions": conditions,
    }
    sha = hashlib.sha256(json.dumps(manifest["config"], sort_keys=True, default=str).encode()).hexdigest()[:8]
    out = out_dir / f"result_{sha}.json"
    json.dump(manifest, open(out, "w"), indent=2)
    print(f"\nrecords={len(records)} conditions={len(conditions)} wall={time.time()-t0:.1f}s\nmanifest -> {out}", flush=True)
    return str(out)


if __name__ == "__main__":
    main()
