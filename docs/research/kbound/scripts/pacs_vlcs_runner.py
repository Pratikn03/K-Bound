#!/usr/bin/env python3
"""
pacs_vlcs_runner.py -- PACS / VLCS domain-shift confirmation for K-Bound (item-4 bonus).
Pre-registered protocol: research_lock/PACS_VLCS_PREREG_PROTOCOL_v1.md

This REUSES the locked decision machinery from cifar_tent_mps_v2.py (the {tent,eata,sar}
adapters, evidence_vector, policy_metrics) so the KGA decision is identical to the headline
protocols. It adds only: a DomainBed image loader, ERM source training, and an honest
leave-one-domain-out calibration (GBR + conformal radius fit on SOURCE-val cells, applied to the
held-out TEST domain -- no test labels ever touch the gate).

For each held-out test domain d:
  * f0 = ResNet-18 (ImageNet-pretrained) ERM-trained on the OTHER domains' train split;
  * generate per-batch "cells" (batch-size x aggressiveness x composition x repeats) on a held-out
    SOURCE-val split (calibration, true B known) and on the TEST domain (evaluation);
  * for each cell: a0 = frozen acc, run each adapter -> aa, Z = label-free evidence, B = aa - a0;
  * fit GBR(Z->B) + eps on the calibration cells, decide adapt/freeze/abstain on the test cells;
  * report regret, FA_u, FA_c, coverage, adapt-rate + paired bootstrap CIs (vs adapt / vs freeze).

DomainBed layout expected:
  <root>/PACS/{art_painting,cartoon,photo,sketch}/<class>/*.jpg
  <root>/VLCS/{Caltech101,LabelMe,SUN09,VOC2007}/<class>/*.jpg

Smoke (tiny, validates end-to-end in a couple min on CPU):
  python scripts/pacs_vlcs_runner.py --dataset PACS --root <root> --device cpu --smoke
Full run:
  python scripts/pacs_vlcs_runner.py --dataset PACS --root <root> --device mps --out pacs_result.json
"""
import argparse, hashlib, json, os, sys, tempfile, time
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
import torchvision as tv
from torchvision import transforms
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cifar_tent_mps_v2 import (tent_adapt, eata_adapt, sar_adapt, evidence_vector,
                               policy_metrics, label_regime, acc_on, ALPHA, SEED,
                               EVIDENCE_NAMES)
# The ONE radius rule and the ONE false-adapt definition (fix-queue items 4, 15, 28).
import kbound_decide as _kb  # noqa: E402

DOMAINS = {"PACS": ["art_painting", "cartoon", "photo", "sketch"],
           "VLCS": ["Caltech101", "LabelMe", "SUN09", "VOC2007"]}
IMN_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
IMN_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
BATCH_REGIMES = {"large": 128, "small": 48, "tiny": 16}
AGGR = {"mild": dict(steps=10, lr=1e-3), "aggressive": dict(steps=50, lr=2.5e-3)}
COMPS = ["iid", "imbalanced", "single_class"]
TF = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()])  # [0,1], norm later


def pick_device(flag):
    if flag == "cpu": return torch.device("cpu")
    if flag == "mps" and torch.backends.mps.is_available(): return torch.device("mps")
    if flag == "cuda" and torch.cuda.is_available(): return torch.device("cuda")
    if torch.cuda.is_available(): return torch.device("cuda")
    if torch.backends.mps.is_available(): return torch.device("mps")
    return torch.device("cpu")


def _norm(x):  # x: (N,3,224,224) in [0,1] on device
    return (x - IMN_MEAN.to(x.device)) / IMN_STD.to(x.device)


def load_domain(root, dataset, domain):
    """Return (samples list[(path,label)], classes) without loading pixels.
    Filters macOS AppleDouble '._' sidecar files that exFAT drives create (not real images)."""
    ds = tv.datasets.ImageFolder(os.path.join(root, dataset, domain))
    samples = [(p, l) for (p, l) in ds.samples if not os.path.basename(p).startswith("._")]
    return samples, ds.classes


def load_batch(samples, idx, dev):
    xs = []
    for i in idx:
        with Image.open(samples[i][0]) as im:
            xs.append(TF(im.convert("RGB")))
    x = torch.stack(xs).to(dev)
    y = torch.tensor([samples[i][1] for i in idx])
    return _norm(x).contiguous(), y


def class_indices(samples, nC):
    labs = np.array([s[1] for s in samples])
    return [np.where(labs == c)[0] for c in range(nC)]


def sample_cell_idx(cls_idx, comp, bs, rng, nstream=4):
    """Return (stream_idx, eval_idx) mirroring build_stream_and_eval's composition logic."""
    nC = len(cls_idx)
    ev = np.concatenate([rng.choice(ci, min(max(1, 64 // nC), len(ci)), replace=False)
                         for ci in cls_idx if len(ci)])
    pool = np.setdiff1d(np.concatenate(cls_idx), ev)
    if comp == "iid":
        s = rng.choice(pool, min(bs * nstream, len(pool)), replace=len(pool) < bs * nstream)
    elif comp == "imbalanced":
        maj = rng.integers(nC); m = np.intersect1d(cls_idx[maj], pool); o = np.setdiff1d(pool, m)
        nM = int(bs * nstream * 0.85)
        s = np.concatenate([rng.choice(m, min(nM, len(m)) or 1, replace=True),
                            rng.choice(o, max(1, bs * nstream - min(nM, len(m))), replace=True)])
    else:  # single_class
        maj = rng.integers(nC); m = np.intersect1d(cls_idx[maj], pool)
        s = rng.choice(m if len(m) else pool, bs * nstream, replace=True)
    rng.shuffle(s)
    return s, ev


def make_resnet18(nC, dev):
    try:
        m = tv.models.resnet18(weights=tv.models.ResNet18_Weights.IMAGENET1K_V1)
    except Exception:
        m = tv.models.resnet18(pretrained=True)  # older torchvision
    m.fc = nn.Linear(m.fc.in_features, nC)
    return m.to(dev)


def erm_train(samples_list, nC, dev, steps, bs=64, lr=1e-3, seed=SEED):
    """ERM fine-tune on concatenated source-domain TRAIN samples."""
    torch.manual_seed(seed); np.random.seed(seed)
    m = make_resnet18(nC, dev); m.train()
    opt = torch.optim.Adam(m.parameters(), lr=lr)
    alls = [s for samp in samples_list for s in samp]
    rng = np.random.default_rng(seed)
    for t in range(steps):
        idx = rng.choice(len(alls), bs, replace=len(alls) < bs)
        x, y = load_batch(alls, idx, dev)
        loss = F.cross_entropy(m(x), y.to(dev))
        opt.zero_grad(); loss.backward(); opt.step()
        if (t + 1) % max(1, steps // 5) == 0:
            print(f"   [erm] step {t+1}/{steps} loss={loss.item():.3f}", flush=True)
    m.eval()
    return m


def _adapt(mth, m, stream, a, nC):
    if mth == "tent": return tent_adapt(m, stream, a["steps"], a["lr"])
    if mth == "eata": return eata_adapt(m, stream, a["steps"], a["lr"], nC)
    if mth == "sar":  return sar_adapt(m, stream, a["steps"], a["lr"], nC)
    raise ValueError(f"unknown method {mth}")


def gen_cells(f0, samples, nC, dev, rng, methods, cell_cfgs, adapt_lr=None):
    """Per-cell (Z, a0, aa, B, regime, method) for one domain's data.
    WIN_HUNT_v5: adapt_lr overrides the AGGR cell lr (None = per-cell lr, byte-identical)."""
    cls_idx = class_indices(samples, nC)
    rows = []
    for (brn, comp, agn, rep) in cell_cfgs:
        bs = BATCH_REGIMES[brn]; a = dict(AGGR[agn])
        if adapt_lr is not None:
            a["lr"] = adapt_lr   # WIN_HUNT_v5 aggressive op-point (0.004 = 4x the 1e-3 baseline)
        s_idx, e_idx = sample_cell_idx(cls_idx, comp, bs, rng)
        ex, ey = load_batch(samples, e_idx, dev)
        stream = []
        for i in range(0, len(s_idx), bs):
            sx, _ = load_batch(samples, s_idx[i:i + bs], dev); stream.append(sx)
        a0 = acc_on(f0, ex, ey, train_mode=False)
        for mth in methods:
            adapted, un = _adapt(mth, f0, stream, a, nC)
            aa = acc_on(adapted, ex, ey, train_mode=True)
            Z = evidence_vector(f0, adapted, ex, nC, un)
            rows.append(dict(condition=f"{comp}|{brn}|{agn}|r{rep}|{mth}",
                             candidate=mth,
                             Z=Z, a0=a0, aa=aa, B=aa - a0, regime=label_regime(aa - a0)))
    return rows


def decide_transfer(cal, test, alpha):
    """GBR fit on CALIBRATION cells, conformal eps from cal LOO residuals, applied to TEST cells.

    FIX-QUEUE ITEM 4 does not bite here: this is the *genuine held-out calibration
    split* the fix-queue offers as the alternative to leave-one-out-of-pool.  The
    calibration cells come from a held-out SOURCE domain and the scored cells from
    the held-out TEST domain, so eps is never a function of a label it protects.

    What DID need fixing is the RULE (fix-queue items 2 + 4): eps was
    ``float(np.quantile(np.abs(loo - Bc), 1 - alpha))``, numpy's linear
    interpolation between order statistics.  That is not an observed residual and
    does not satisfy the finite-sample rank argument, and having two rules inside
    one paper is panel finding F1-2 / F4-10 / F5-2.  eps is now the exact
    split-conformal rank quantile ``r_(k)``, ``k = ceil((n+1)(1-alpha))``, from
    ``kbound_decide.conformal_radius`` -> ``kga.certificate`` (fix-queue item 15),
    and the trichotomy comes from ``kga.policy`` rather than a local ``np.where``.

    NOTE for the paper: this changes the PACS/VLCS numbers by the interpolated ->
    exact-rank delta (NUMBERS_PACK.md sec. 0.4 prices the same change on five other
    published rows). The committed PACS artifacts carry no ``b_hat``, so they
    cannot be re-scored offline (NUMBERS_PACK.md sec. 8.2) -- this track must be
    re-run to be reported under the declared rule.
    """
    from sklearn.ensemble import GradientBoostingRegressor
    Zc = np.array([r["Z"] for r in cal]); Bc = np.array([r["B"] for r in cal])
    Zt = np.array([r["Z"] for r in test])
    gbr = GradientBoostingRegressor(n_estimators=250, max_depth=2, learning_rate=0.05,
                                    subsample=0.8, random_state=SEED).fit(Zc, Bc)
    loo = np.zeros(len(Bc))
    for i in range(len(Bc)):
        tr = np.arange(len(Bc)) != i
        loo[i] = GradientBoostingRegressor(n_estimators=250, max_depth=2, learning_rate=0.05,
                                           subsample=0.8, random_state=SEED).fit(Zc[tr], Bc[tr]).predict(Zc[i:i+1])[0]
    eps = float(_kb.conformal_radius(np.abs(loo - Bc), alpha))
    Bhat = gbr.predict(Zt)
    dec = _kb.decide(Bhat, eps, alpha=alpha)
    residual_payload = np.asarray(np.sort(np.abs(loo - Bc)), dtype="<f8").tobytes()
    return dec, {
        "epsilon": eps,
        "b_hat": Bhat,
        "residual_pool_sha256": hashlib.sha256(residual_payload).hexdigest(),
        "n_calibration_residuals": len(Bc),
    }


def _model_sha256(model):
    digest = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        array = tensor.detach().cpu().contiguous().numpy()
        digest.update(name.encode("utf-8"))
        digest.update(str(array.dtype).encode("ascii"))
        digest.update(str(array.shape).encode("ascii"))
        digest.update(array.tobytes())
    return digest.hexdigest()


def _canonical_sha256(document):
    payload = json.dumps(document, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _atomic_json(path, document):
    """Write a complete artifact atomically; partial seed files are never publishable."""
    path = os.path.abspath(path)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=f".{os.path.basename(path)}.", dir=os.path.dirname(path))
    try:
        with os.fdopen(fd, "w") as handle:
            json.dump(document, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, path)
    except Exception:
        try:
            os.unlink(tmp)
        except FileNotFoundError:
            pass
        raise


def _sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def boot_ci(dec, a0, aa, n_boot=3000, seed=SEED):
    rng = np.random.default_rng(seed)
    adapt = dec == "ADAPT"; kga = np.where(adapt, aa, a0); orc = np.maximum(a0, aa)
    rk, ra, rf = orc - kga, orc - aa, orc - a0
    N = len(a0); da, dfz = ra - rk, rf - rk  # positive => KGA better
    bA = [da[rng.integers(0, N, N)].mean() for _ in range(n_boot)]
    bF = [dfz[rng.integers(0, N, N)].mean() for _ in range(n_boot)]
    ci = lambda b: [float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))]
    return {"vs_adapt_mean": float(da.mean()), "vs_adapt_ci": ci(bA),
            "vs_freeze_mean": float(dfz.mean()), "vs_freeze_ci": ci(bF)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["PACS", "VLCS"], required=True)
    ap.add_argument("--root", required=True, help="parent dir containing PACS/ and/or VLCS/")
    ap.add_argument("--device", default="auto", choices=["auto", "cpu", "mps", "cuda"])
    ap.add_argument("--methods", nargs="+", default=["tent", "eata", "sar"])
    ap.add_argument("--erm-steps", type=int, default=600)
    ap.add_argument("--alpha", type=float, default=ALPHA)
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--smoke", action="store_true", help="tiny: 1 test domain, few cells, few ERM steps")
    ap.add_argument("--out", default="pacs_vlcs_result.json")
    # ---- WIN_HUNT_v5 aggressive-regime wave operating-point overrides (opt-in) ----
    ap.add_argument("--adapt-lr", type=float, default=None, dest="adapt_lr",
                    help="WIN_HUNT_v5: absolute adapter LR override for tent/eata/sar (AGGR cell lr "
                         "ignored when set). DEFAULT None = per-cell lr (byte-identical). v5 sets 0.004 "
                         "(= 4x the 1e-3 shared-baseline lr).")
    ap.add_argument("--batch-regimes", nargs="+", default=None, dest="batch_regimes",
                    choices=list(BATCH_REGIMES.keys()),
                    help="WIN_HUNT_v5: restrict the batch-regime sweep (DEFAULT None = all; v5 uses "
                         "'tiny' = 16). Ignored under --smoke.")
    ap.add_argument("--aggressiveness", nargs="+", default=None, dest="aggressiveness",
                    choices=list(AGGR.keys()),
                    help="WIN_HUNT_v5: restrict the aggressiveness sweep (DEFAULT None = all; v5 uses "
                         "'aggressive' = 50 steps with --adapt-lr 0.004). Ignored under --smoke.")
    args = ap.parse_args()
    dev = pick_device(args.device); print("device:", dev, "dataset:", args.dataset, flush=True)

    domains = DOMAINS[args.dataset]
    # cell grid (held back in smoke)
    # Exact split conformal at alpha=.10 requires at least nine calibration
    # residuals. Smoke uses 2 compositions x 1 mild setting x 5 repeats = 10
    # cells per candidate, while avoiding the expensive 50-step setting.
    reps = list(range(5)) if args.smoke else [0, 1]
    # WIN_HUNT_v5: optional batch/aggr subsetting (None -> full sweep, byte-identical). Smoke unchanged.
    _brs = (["small"] if args.smoke else (args.batch_regimes or list(BATCH_REGIMES)))
    _ags = (["mild"] if args.smoke else (args.aggressiveness or list(AGGR)))
    _comps = (["iid", "single_class"] if args.smoke else COMPS)
    cell_cfgs = [(b, c, a, r) for b in _brs for c in _comps for a in _ags for r in reps]
    erm_steps = 30 if args.smoke else args.erm_steps
    test_domains = domains[:1] if args.smoke else domains

    # discover classes from the first domain
    s0, classes = load_domain(args.root, args.dataset, domains[0]); nC = len(classes)
    print(f"classes={nC} domains={domains}", flush=True)
    cache = {d: load_domain(args.root, args.dataset, d)[0] for d in domains}

    results = {"schema": "kbound_pacs_seed_v1", "dataset": args.dataset,
               "seed": args.seed, "alpha": args.alpha, "n_classes": nC,
               "protocol": {"erm_steps": erm_steps, "methods": list(args.methods),
                            "cell_grid": {"batch_regimes": list(_brs),
                                          "compositions": list(_comps),
                                          "aggressiveness": list(_ags),
                                          "repetitions": list(reps)}},
               "win_hunt_v5_override": {"adapt_lr": args.adapt_lr,
                                        "batch_regimes": args.batch_regimes,
                                        "aggressiveness": args.aggressiveness},
               "per_domain": {}}
    for d_test in test_domains:
        t0 = time.time()
        src = [d for d in domains if d != d_test]
        print(f"\n=== held-out test domain: {d_test}  (sources {src}) ===", flush=True)
        f0 = erm_train([cache[s] for s in src], nC, dev, erm_steps, seed=args.seed)
        checkpoint_sha256 = _model_sha256(f0)
        domain_config = {
            "dataset": args.dataset,
            "seed": args.seed,
            "source_domains": src,
            "target_domain": d_test,
            "calibration_domain": src[-1],
            "methods": list(args.methods),
            "alpha": args.alpha,
            "erm_steps": erm_steps,
            "cell_grid": results["protocol"]["cell_grid"],
            "adapt_lr": args.adapt_lr,
            "evidence_schema": "kbound_evidence_v1_11d",
        }
        config_sha256 = _canonical_sha256(domain_config)
        rng = np.random.default_rng(args.seed)
        # calibration cells: a held-out SOURCE domain (last source) -- no test labels used
        cal_dom = src[-1]
        cal = gen_cells(f0, cache[cal_dom], nC, dev, rng, args.methods, cell_cfgs, adapt_lr=args.adapt_lr)
        test = gen_cells(f0, cache[d_test], nC, dev, rng, args.methods, cell_cfgs, adapt_lr=args.adapt_lr)
        dec, decision_fit = decide_transfer(cal, test, args.alpha)
        eps = float(decision_fit["epsilon"])
        bhat = decision_fit["b_hat"]
        a0 = np.array([r["a0"] for r in test]); aa = np.array([r["aa"] for r in test])
        B = aa - a0
        pm = policy_metrics(dec, a0, aa, B)
        adapt = dec == "ADAPT"
        # fix-queue item 28: ONE definition -- a false adapt is ADAPT on B <= 0.
        # The old line used the STRICT `B < 0` on both rates, which exempts ties
        # (500 archived cells have B exactly 0.0 and 102 of them ADAPT), and
        # silently reported fa_c = 0.0 rather than "undefined" when nothing adapted.
        _fa = _kb.false_adapt(dec, B)
        fa_u = float(_fa["fa_u"])
        fa_c = _fa["fa_c"]   # None when there are no ADAPT decisions
        cis = boot_ci(dec, a0, aa)
        win = (cis["vs_adapt_ci"][0] > 0 and cis["vs_freeze_ci"][0] > 0 and fa_u <= args.alpha)
        verdict = "WIN (beats-both, CI-robust)" if win else (
            "SAFETY (no-harm)" if fa_u <= args.alpha and pm["regret_vs_oracle"]["K_Bound"] <=
            min(pm["regret_vs_oracle"]["always_adapt"], pm["regret_vs_oracle"]["always_freeze"]) + 1e-6
            else "NULL")
        radius_is_finite = bool(np.isfinite(eps))
        serialized_radius = float(eps) if radius_is_finite else None
        percell_records = []
        for row, prediction, action in zip(test, bhat, dec, strict=True):
            benefit = float(row["B"])
            percell_records.append(
                {
                    "dataset": args.dataset,
                    "domain": d_test,
                    "calibration_domain": cal_dom,
                    "seed": args.seed,
                    "split": "test",
                    "condition": row["condition"],
                    "candidate": row["candidate"],
                    "metric": "accuracy",
                    "Z": [float(value) for value in row["Z"]],
                    "Z_names": list(EVIDENCE_NAMES),
                    "evidence_schema_version": "kbound_evidence_v1_11d",
                    "a0": float(row["a0"]),
                    "aa": float(row["aa"]),
                    "loss_frozen": float(1.0 - row["a0"]),
                    "loss_adapted": float(1.0 - row["aa"]),
                    "B": benefit,
                    "b_hat": float(prediction),
                    "eps_conformal": serialized_radius,
                    "radius_status": (
                        "FINITE" if radius_is_finite else "INFINITE_INSUFFICIENT_CALIBRATION"
                    ),
                    "kga_decision": str(action),
                    "oracle_action": "ADAPT" if benefit > 0 else "FREEZE",
                    "source_checkpoint_sha256": checkpoint_sha256,
                    "run_config_sha256": config_sha256,
                    "residual_pool_sha256": decision_fit["residual_pool_sha256"],
                }
            )
            percell_records[-1]["record_id"] = _canonical_sha256(
                {
                    "run_config_sha256": config_sha256,
                    "condition": row["condition"],
                    "candidate": row["candidate"],
                }
            )

        percell_dir = os.path.join(os.path.dirname(os.path.abspath(args.out)), "per_cell")
        percell_path = os.path.join(
            percell_dir, f"{args.dataset.lower()}_{d_test}_seed{args.seed}_percell.json")
        percell_document = {
            "schema": "kbound_pacs_percell_v2",
            "dataset": args.dataset,
            "domain": d_test,
            "calibration_domain": cal_dom,
            "source_domains": src,
            "seed": args.seed,
            "alpha": args.alpha,
            "source_checkpoint_sha256": checkpoint_sha256,
            "run_config": domain_config,
            "run_config_sha256": config_sha256,
            "decision_rule": "ADAPT iff b_hat-eps>0; FREEZE iff b_hat+eps<0; else ABSTAIN",
            "calibration": {
                "design": "held-out source-domain calibration to held-out target domain",
                "radius": "LOO residuals with exact split-conformal order statistic",
                "n_calibration_cells": len(cal),
                "residual_pool_sha256": decision_fit["residual_pool_sha256"],
            },
            "records": percell_records,
        }
        _atomic_json(percell_path, percell_document)

        results["per_domain"][d_test] = {
            "calibration_domain": cal_dom, "n_test_cells": len(test),
            "eps": serialized_radius,
            "radius_status": (
                "FINITE" if radius_is_finite else "INFINITE_INSUFFICIENT_CALIBRATION"
            ),
            "regret": pm["regret_vs_oracle"], "FA_u": fa_u, "FA_c": fa_c,
            "coverage": pm["coverage"], "adapt_rate": float(adapt.mean()),
            "base_rate_harmful": float(np.mean(B < 0)), "cis": cis, "verdict": verdict,
            "per_cell_artifact": os.path.relpath(percell_path, os.path.dirname(os.path.abspath(args.out))),
            "per_cell_sha256": _sha256(percell_path),
            "source_checkpoint_sha256": checkpoint_sha256,
            "run_config_sha256": config_sha256,
            "residual_pool_sha256": decision_fit["residual_pool_sha256"],
            "wall_sec": round(time.time() - t0, 1)}
        print(f"  {d_test}: {verdict} | regret KGA {pm['regret_vs_oracle']['K_Bound']:.4f} "
              f"vs adapt {pm['regret_vs_oracle']['always_adapt']:.4f} / freeze "
              f"{pm['regret_vs_oracle']['always_freeze']:.4f} | FA_u {fa_u:.3f} cov {pm['coverage']:.2f}",
              flush=True)
    _atomic_json(args.out, results)
    nwin = sum(v["verdict"].startswith("WIN") for v in results["per_domain"].values())
    print(f"\n==== {args.dataset}: {nwin}/{len(results['per_domain'])} domains are CI-robust beats-both."
          f"  wrote {args.out} ====")


if __name__ == "__main__":
    main()
