"""
run_rxrx1_kbound.py - K-Bound TTA sweep on WILDS RxRx1 (experimental-batch shift, 1139-class).

Models the ImageNet-R multi-class runner (run_imagenetr_kbound.py) on the REAL WILDS RxRx1 setup:
  frozen f0   : torchvision resnet50(num_classes=1139) loaded from the OFFICIAL WILDS ERM
                checkpoint (rxrx1_seed:0_epoch:best_model.pth; 'model.'-prefixed state_dict
                -> 0 missing/0 unexpected). NO training. In-dist acc ~35.9% (verified ~33% on n=256).
  data        : WILDS RxRx1 v1.0, OOD 'test' split (14 unseen experiments) = the single target
                domain. WILDS 'rxrx1' eval transform (ToTensor + per-image standardize) - REQUIRED
                to reproduce the checkpoint's accuracy.
  candidates  : {tent,eata,sar} x {online,episodic}        (reused VERBATIM from tta_methods)
  routing     : (a) single-cand KGA, (b) multi-cand tau-route, (c) smooth-drift (max-prob Brier)
                (reused from analysis); aggregation/manifest/kbound_summary reused from run_camelyon17.
  conditions  : composition x batch_regime x aggressiveness x seed (single domain).

SURVIVAL HARNESS (16 GB MPS, has OOM-killed heavy sweeps): per-condition memory hygiene
(del + gc.collect + torch.mps.empty_cache + torch.mps.synchronize), resume-from-_partial
(skip completed cells, APPEND, atomic flush, never truncate), per-cell deterministic rng so
resume is bit-identical, plus an external auto-restart supervisor (supervise_rxrx1.sh).

INTEGRITY: real runs only; honest helpful/harmful/mixed classification from measured B; tau*
fixed at 0.52 (NEVER tuned to force collapse); per-condition tau stored for re-pick.
"""
from __future__ import annotations
import os, sys, gc, re, json, time, argparse, platform, hashlib
import numpy as np
import torch
import torchvision.models as M
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))
if HERE not in sys.path:
    sys.path.insert(0, HERE)
import tta_methods as tm            # noqa: E402
import analysis as an              # noqa: E402
import run_camelyon17_kbound as rc  # noqa: E402  (AGGR, CANDIDATES, aggregate_*, kbound_summary, route_realized)

NUM_CLASSES = 1139
BATCH_REGIMES = {"large_iid": 200, "small": 16, "tiny": 8}
DOMAIN = "rxrx1"


def _standardize(x):
    mean = x.mean(dim=(1, 2)); std = x.std(dim=(1, 2)); std[std == 0.] = 1.
    return TF.normalize(x, mean, std)


def rxrx1_eval_transform():
    """WILDS 'rxrx1' eval transform: ToTensor then per-image per-channel standardize."""
    return transforms.Compose([transforms.ToTensor(), transforms.Lambda(_standardize)])


def ensure_rxrx1_patch():
    """wilds 2.0.0 RxRx1Dataset: the split_array in-place write needs a WRITABLE numpy array
    (numpy>=1.24 returns read-only .values -> 'assignment destination is read-only'). Ensure the
    installed source uses .values.copy() on that line. Idempotent (no-op once patched)."""
    try:
        import wilds.datasets.rxrx1_dataset as rd
        p = rd.__file__
        with open(p) as f:
            src = f.read()
        new = re.sub(r"(self\._split_dict\.get\)\.values)(?!\.copy)", r"\1.copy()", src)
        if new != src:
            with open(p, "w") as f:
                f.write(new)
            return "patched"
        return "already_patched"
    except Exception as e:
        return f"skip:{e!r}"


def load_f0(ckpt, device, num_classes=NUM_CLASSES):
    """Frozen f0 = torchvision resnet50(num_classes) <- official WILDS ERM checkpoint.
    Strips the 'model.' prefix; asserts 0 missing / 0 unexpected keys. NO training."""
    obj = torch.load(ckpt, map_location="cpu", weights_only=False)
    state = obj["algorithm"] if isinstance(obj, dict) and "algorithm" in obj else obj
    new = {(k[len("model."):] if k.startswith("model.") else k): v for k, v in state.items()}
    model = M.resnet50(weights=None, num_classes=num_classes)
    res = model.load_state_dict(new, strict=False)
    assert not res.missing_keys and not res.unexpected_keys, \
        f"checkpoint key mismatch: missing={res.missing_keys[:4]} unexpected={res.unexpected_keys[:4]}"
    model.to(device).eval()
    return model


def load_rxrx1(data_root, split, device):
    """Build the OOD target subset with the WILDS rxrx1 eval transform; disk-filter to
    present images (honest dropped-count) so a partial internal copy is handled correctly."""
    from wilds import get_dataset
    ds = get_dataset(dataset="rxrx1", download=False, root_dir=data_root)
    sub = ds.get_subset(split, transform=rxrx1_eval_transform())
    idx = np.asarray(sub.indices)
    data_dir = str(ds.data_dir)
    inp = ds._input_array
    keep = np.fromiter((os.path.exists(os.path.join(data_dir, str(inp[i]))) for i in idx),
                       dtype=bool, count=len(idx))
    n_total = int(len(idx)); n_present = int(keep.sum())
    sub.indices = idx[keep]
    y = ds.y_array[np.asarray(sub.indices)].numpy().astype(int)
    return ds, sub, y, n_present, n_total


def _load_x(sub, pos):
    x, _, _ = sub[int(pos)]
    return x


def build_condition(sub, y, comp, bs, n_eval, rng, device, n_batches=4, tries=20):
    """Class-balanced (CAPPED at n_eval) held-out eval + composition-controlled adaptation stream.
    RxRx1 has 1139 classes; the per-class eval pool is capped at n_eval so eval_x stays MPS-tractable
    (one example per sampled class -> balanced_acc == standard accuracy). single_class / imbalanced +
    tiny batches are the natural collapse-prone cells; harm arises from the DATA, never tuned HPs."""
    pos_all = np.arange(len(sub))
    classes = np.unique(y)
    per = max(1, n_eval // max(1, len(classes)))
    ev = []
    for c in classes:
        ci = pos_all[y == c]
        if len(ci):
            ev.append(rng.choice(ci, min(per, len(ci)), replace=False))
    ev = np.concatenate(ev)
    if len(ev) > n_eval:                                  # 1139 classes >> n_eval -> cap
        ev = rng.choice(ev, n_eval, replace=False)
    rng.shuffle(ev)
    remain = np.setdiff1d(pos_all, ev)
    if len(remain) == 0:
        remain = pos_all
    n_stream = max(bs, bs * n_batches)
    if comp == "iid":
        s = rng.choice(remain, n_stream, replace=len(remain) < n_stream)
    elif comp == "imbalanced":
        maj = int(rng.choice(classes))
        mp = np.intersect1d(pos_all[y == maj], remain); op = np.setdiff1d(remain, mp)
        if len(mp) and len(op):
            nM = int(n_stream * 0.85)
            s = np.concatenate([rng.choice(mp, nM, replace=len(mp) < nM),
                                rng.choice(op, n_stream - nM, replace=len(op) < (n_stream - nM))])
        else:
            s = rng.choice(remain, n_stream, replace=len(remain) < n_stream)
    else:  # single_class label shift (collapse-prone)
        maj = int(rng.choice(classes))
        mp = np.intersect1d(pos_all[y == maj], remain)
        pool = mp if len(mp) else remain
        s = rng.choice(pool, n_stream, replace=len(pool) < n_stream)
    rng.shuffle(s)

    def _load(positions, same_class=None):
        cand = pos_all[y == same_class] if same_class is not None else pos_all
        xs = []
        for p in positions:
            try:
                xs.append(_load_x(sub, int(p))); continue
            except Exception:
                pass
            ok = False
            for q in rng.permutation(cand)[:tries]:      # fallback only on read error
                try:
                    xs.append(_load_x(sub, int(q))); ok = True; break
                except Exception:
                    continue
            if not ok:
                raise RuntimeError("no readable image in condition")
        return torch.stack(xs).to(device)

    stream_x = _load(s)
    stream = [stream_x[i:i + bs] for i in range(0, len(stream_x), bs)]
    ex, ey = [], []
    for p in ev:
        c = int(y[p])
        ex.append(_load([p], same_class=c)); ey.append(c)
    eval_x = torch.cat(ex, 0).to(device); eval_y = np.array(ey, dtype=int)
    return stream, eval_x, eval_y


# ----------------------------- survival helpers ------------------------------
def deep_free(device):
    """Per-condition memory hygiene for MPS: collect python garbage, then empty + sync MPS."""
    gc.collect()
    try:
        if getattr(device, "type", None) == "mps" and torch.backends.mps.is_available():
            torch.mps.empty_cache(); torch.mps.synchronize()
    except Exception:
        pass


def _tag(seed, comp, regime, aggr):
    return f"s{seed}/{comp}/{regime}/{aggr}"


def _atomic_dump(obj, path):
    """Write _partial.json via tmp + os.replace so a crash mid-write never corrupts the file."""
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(obj, f)
    os.replace(tmp, path)


def load_partial(partial_path):
    """Resume: load prior records/conditions; rebuild the completed-cell set from conditions
    (a cell is COMPLETE iff its condition exists); drop any orphan records from a half-done cell."""
    if not os.path.exists(partial_path):
        return [], [], set()
    try:
        with open(partial_path) as f:
            d = json.load(f)
    except Exception:
        return [], [], set()
    conditions = d.get("conditions", [])
    done = set((int(c["seed"]), c["comp"], c["regime"], c["aggr"]) for c in conditions)
    records = [r for r in d.get("records", [])
               if (int(r["seed"]), r["comp"], r["regime"], r["aggr"]) in done]
    return records, conditions, done


# --------------------------------- the sweep ---------------------------------
def run(args, out_dir):
    t0 = time.time()
    device = tm.pick_device(args.device)
    partial = os.path.join(out_dir, "_partial.json")
    records, conditions, done = load_partial(partial) if args.resume else ([], [], set())
    ds, sub, y, n_present, n_total = load_rxrx1(args.data_root, args.split, device)
    print(f"[rxrx1] split={args.split} present={n_present}/{n_total} "
          f"classes={len(np.unique(y))} device={device}", flush=True)
    print(f"[resume] {len(done)} completed cells loaded, {len(records)} records carried", flush=True)
    f0 = load_f0(args.ckpt, device)
    n_cells = (len(args.seeds) * len(args.compositions) * len(args.batch_regimes) * len(args.aggressiveness))
    ci = 0
    # WIN_HUNT_v5: online-only candidate pool (the "continual" no-episodic-reset op-point) when
    # --online-only is set; default keeps all six online+episodic candidates (byte-identical).
    _cands = [(m, md) for (m, md) in rc.CANDIDATES
              if (not getattr(args, "online_only", False)) or md == "online"]
    for seed in args.seeds:
        for comp in args.compositions:
            for regime in args.batch_regimes:
                for aggr in args.aggressiveness:
                    ci += 1
                    key = (int(seed), comp, regime, aggr); tag = _tag(seed, comp, regime, aggr)
                    if key in done:
                        print(f"  [{ci}/{n_cells}] {tag} SKIP (already done)", flush=True)
                        continue
                    cell_seed = int(hashlib.sha256(tag.encode()).hexdigest()[:8], 16)
                    torch.manual_seed(cell_seed); np.random.seed(cell_seed % (2 ** 31))
                    rng = np.random.default_rng(cell_seed)         # per-cell -> resume is bit-identical
                    bs = BATCH_REGIMES[regime]
                    steps = args.steps_override or rc.AGGR[aggr]["steps"]; lr = args.adapt_lr if getattr(args, "adapt_lr", None) is not None else rc.AGGR[aggr]["lr"]
                    try:
                        stream, eval_x, eval_y = build_condition(
                            sub, y, comp, bs, args.n_eval, rng, device, n_batches=args.n_batches)
                    except Exception as e:
                        print(f"  [{ci}/{n_cells}] {tag} SKIP build: {e}", flush=True); continue
                    try:
                        a0, p0, p0_pos = tm.eval_frozen(f0, eval_x, eval_y, prob_mode="max")
                        stream_f0_pos = tm._predict_prob(f0, torch.cat(stream, 0), train_mode=False, bs=128, mode="max")
                        preds_all = [p0]; aa_all = [a0]; cand_names = ["freeze_f0"]; best_pa = p0_pos; best_aa_c = float(a0)
                        for (method, mode) in _cands:
                            aa, Z, upd, preds, pa_pos = tm.run_candidate(
                                method, mode, f0, stream, eval_x, eval_y, NUM_CLASSES, steps, lr,
                                eval_bs=args.episodic_batch, prob_mode="max", episodic_steps=args.episodic_steps)
                            records.append(dict(seed=int(seed), domain=DOMAIN, comp=comp, regime=regime, aggr=aggr,
                                                method=method, mode=mode, candidate=f"{method}_{mode}",
                                                a0=float(a0), aa=float(aa), B=float(aa - a0), upd_norm=float(upd),
                                                Z=[float(z) for z in Z], regime_label=an.label_regime(float(aa - a0))))
                            preds_all.append(preds); aa_all.append(float(aa)); cand_names.append(f"{method}_{mode}")
                            if float(aa) > best_aa_c:
                                best_aa_c = float(aa); best_pa = pa_pos
                            tm.mps_free(); gc.collect()
                        preds_mat = np.stack(preds_all, 0)
                        route = an.multicandidate_route(preds_mat, tau_star=args.tau_star, kappa=args.kappa)
                        realized = rc.route_realized(route, aa_all)
                        oracle = float(max(aa_all)); best_adapt = float(max(aa_all[1:]))
                        try:
                            route_c = an.smooth_drift_route(p0_pos, best_pa, stream_f0_pos, L=args.sd_L)
                            if route_c.get("implemented") and "bracket" in route_c:
                                trueB = best_adapt - a0
                                route_c["true_B_best"] = float(trueB)
                                route_c["bracket_covers_trueB"] = bool(route_c["bracket"][0] <= trueB <= route_c["bracket"][1])
                        except Exception as e:
                            route_c = {"decision": "ERROR", "implemented": False, "reason": repr(e)}
                        conditions.append(dict(seed=int(seed), domain=DOMAIN, comp=comp, regime=regime, aggr=aggr,
                                               cand_names=cand_names, aa_all=[float(a) for a in aa_all], a0=float(a0),
                                               oracle=oracle, best_adapt=best_adapt,
                                               true_best=cand_names[int(np.argmax(aa_all))], route=route,
                                               route_c=route_c, realized=realized,
                                               regime_label=an.label_regime(best_adapt - a0)))
                        done.add(key)
                        print(f"  [{ci}/{n_cells}] {tag} a0={a0:.3f} best_aa={best_adapt:.3f} "
                              f"oracle={oracle:.3f} route={route.get('decision')} "
                              f"tau={route.get('tau', float('nan')):.3f} sd_c={route_c.get('decision')}", flush=True)
                    except Exception as e:
                        print(f"  [{ci}/{n_cells}] {tag} ERROR: {repr(e)[:140]}", flush=True)
                    # ---- survival: atomic partial flush (never truncate) + memory hygiene ----
                    try:
                        _atomic_dump({"progress": f"{len(done)}/{n_cells}",
                                      "elapsed_sec": round(time.time() - t0, 1),
                                      "done_tags": sorted("/".join(map(str, k)) for k in done),
                                      "records": records, "conditions": conditions}, partial)
                    except Exception as e:
                        print(f"  [warn] partial flush failed: {e!r}", flush=True)
                    try:
                        del stream, eval_x
                    except Exception:
                        pass
                    try:
                        del stream_f0_pos, preds_all, preds_mat, p0_pos, best_pa
                    except Exception:
                        pass
                    deep_free(device)
    del f0; deep_free(device)
    all_done = (len(done) == n_cells)
    return records, conditions, {"n_present": n_present, "n_total": n_total,
                                 "n_classes": int(len(np.unique(y))), "split": args.split,
                                 "wall_sec": time.time() - t0, "all_done": all_done,
                                 "n_cells_done": len(done), "n_cells_total": n_cells}


def build_manifest(args, records, conditions, meta):
    cfg = {k: getattr(args, k) for k in (
        "data_root", "ckpt", "split", "seeds", "compositions", "batch_regimes", "aggressiveness",
        "n_eval", "n_batches", "tau_star", "kappa", "sd_L", "delta", "device", "steps_override",
        "episodic_steps", "episodic_batch", "smoke",
        "adapt_lr", "online_only")}   # WIN_HUNT_v5 operating-point overrides enter the config hash
    sha = hashlib.sha256(json.dumps(cfg, sort_keys=True, default=str).encode()).hexdigest()[:8]
    return {
        "schema": "kbound_rxrx1_v0.5", "dataset": "wilds-rxrx1",
        "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "host": {"node": platform.node(), "platform": platform.platform(),
                 "python": platform.python_version(), "torch": torch.__version__,
                 "mps": bool(torch.backends.mps.is_available())},
        "config": cfg, "config_sha8": sha,
        "f0": ("torchvision resnet50(num_classes=1139) <- official WILDS RxRx1 ERM checkpoint "
               "(rxrx1_seed:0_epoch:best_model.pth, 'model.'-stripped, 0 missing/0 unexpected); frozen; "
               "WILDS rxrx1 eval transform (ToTensor + per-image standardize); in-dist acc ~35.9%"),
        "num_classes": NUM_CLASSES,
        "candidates": [f"{m}_{md}" for (m, md) in rc.CANDIDATES
                       if (not getattr(args, "online_only", False)) or md == "online"],
        "domain": f"{DOMAIN} (OOD '{args.split}' split = 14 unseen experiments)",
        "multiclass_caveat": ("multi-candidate tau-route (b) uses prediction agreement on a 1139-class "
                              "label space; the binary-Y advantage recovery (Thm 1A) is heuristic here, so "
                              "per-condition tau is stored for re-pick. (a) KGA and (c) smooth-drift "
                              "(max-prob Brier) generalize directly."),
        "eval_pool_note": ("eval pool is class-balanced then CAPPED at n_eval (1139 classes >> n_eval), so "
                           "balanced_acc over the sampled singleton-classes equals standard accuracy."),
        "data": {"data_root": args.data_root, "split": args.split, "n_present": meta["n_present"],
                 "n_total": meta["n_total"], "n_dropped_disk_filter": meta["n_total"] - meta["n_present"],
                 "n_classes": meta["n_classes"], "wall_sec": round(meta["wall_sec"], 1),
                 "cells_done": meta["n_cells_done"], "cells_total": meta["n_cells_total"]},
        "baselines": {
            "always_freeze_mean_acc": float(np.mean([r["a0"] for r in records])) if records else None,
            "per_candidate_always_adapt_mean_acc": {
                c: float(np.mean([r["aa"] for r in records if r["candidate"] == c]))
                for c in sorted(set(r["candidate"] for r in records))},
            "per_condition_oracle_mean_acc": float(np.mean([c["oracle"] for c in conditions])) if conditions else None},
        "routing_a_single_candidate": rc.aggregate_single_candidate(records),
        "routing_b_multicandidate": rc.aggregate_multicandidate(conditions),
        "routing_c_smooth_drift": rc.aggregate_smoothdrift(conditions),
        "detectability": an.detectability_analysis(records, tm.EVIDENCE_NAMES) if len(records) >= 4 else {"note": "need>=4"},
        "kbound_summary": rc.kbound_summary(records, conditions, delta=args.delta),
        "tau_distribution": sorted([float(c["route"]["tau"]) for c in conditions if c["route"].get("tau") is not None]),
        "records": records, "conditions": conditions,
    }


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="K-Bound TTA sweep on WILDS RxRx1 (1139-class, MPS survival)")
    p.add_argument("--data-root", default=os.path.expanduser("~/kbound_rxrx1_data"), dest="data_root",
                   help="dir containing rxrx1_v1.0 (INTERNAL copy; T9 exFAT reads stall MPS)")
    p.add_argument("--ckpt", default=os.path.expanduser("~/kbound_rxrx1_ckpt/rxrx1_seed:0_epoch:best_model.pth"))
    p.add_argument("--split", default="test", help="OOD target domain (default 'test' = 14 unseen experiments)")
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3])
    p.add_argument("--compositions", nargs="+", default=["iid", "imbalanced", "single_class"])
    p.add_argument("--batch-regimes", nargs="+", default=["small", "tiny"], dest="batch_regimes",
                   help="LIGHT recipe: large_iid DROPPED (it OOM-killed heavy sweeps)")
    p.add_argument("--aggressiveness", nargs="+", default=["mild", "aggressive"])
    p.add_argument("--n-eval", type=int, default=256, dest="n_eval")
    p.add_argument("--n-batches", type=int, default=4, dest="n_batches")
    p.add_argument("--episodic-steps", type=int, default=5, dest="episodic_steps")
    p.add_argument("--episodic-batch", type=int, default=64, dest="episodic_batch")
    p.add_argument("--tau-star", type=float, default=0.52, dest="tau_star")
    p.add_argument("--kappa", type=float, default=2.5)
    p.add_argument("--sd-L", type=float, default=0.6, dest="sd_L")
    p.add_argument("--delta", type=float, default=0.05)
    p.add_argument("--device", default="auto", choices=["auto", "cpu", "mps", "cuda"])
    p.add_argument("--steps-override", type=int, default=0, dest="steps_override")
    p.add_argument("--results-root", default=os.path.expanduser("~/kbound_rxrx1_results"), dest="results_root")
    p.add_argument("--run-name", default="rxrx1_kbound_light_mps_internal", dest="run_name")
    p.add_argument("--out", default="")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--smoke", action="store_true")
    # ---- WIN_HUNT_v5 aggressive-regime wave operating-point overrides (opt-in) ----
    p.add_argument("--adapt-lr", type=float, default=None, dest="adapt_lr",
                   help="WIN_HUNT_v5: absolute adapter LR override for tent/eata/sar (rc.AGGR cell "
                        "lr ignored when set). DEFAULT None = per-cell lr (byte-identical). v5 sets "
                        "0.004 (= 4x the 1e-3 shared-baseline lr).")
    p.add_argument("--online-only", action="store_true", dest="online_only",
                   help="WIN_HUNT_v5: restrict the candidate pool to online (no-episodic-reset) "
                        "adapters -- the 'continual' operating point. DEFAULT off (byte-identical).")
    a = p.parse_args(argv)
    if a.smoke:
        a.compositions = ["iid", "single_class"]; a.batch_regimes = ["tiny"]; a.aggressiveness = ["mild"]
        a.seeds = [0, 1]; a.n_eval = 32; a.n_batches = 2; a.steps_override = 4
    return a


def main(argv=None):
    a = parse_args(argv)
    print("[patch]", ensure_rxrx1_patch(), flush=True)
    out_dir = os.path.join(a.results_root, "rxrx1_kbound_smoke" if a.smoke else a.run_name)
    os.makedirs(out_dir, exist_ok=True)
    records, conditions, meta = run(a, out_dir)
    if not meta["all_done"]:
        print(f"[incomplete] {meta['n_cells_done']}/{meta['n_cells_total']} cells done; "
              f"exiting (supervisor will resume)", flush=True)
        return None
    man = build_manifest(a, records, conditions, meta)
    out = a.out or os.path.join(out_dir, f"result_{man['config_sha8']}.json")
    with open(out, "w") as f:
        json.dump(man, f, indent=2)
    with open(os.path.join(out_dir, ".done"), "w") as f:
        f.write(time.strftime("%Y-%m-%dT%H:%M:%S") + f"  {out}\n")
    ks = man["kbound_summary"]; mb = man["routing_b_multicandidate"]; td = man["tau_distribution"]
    print("\n" + "=" * 70, flush=True)
    print(f"records={len(records)} conditions={len(conditions)} wall={meta['wall_sec']:.1f}s", flush=True)
    print(f"classification : {ks['classification']}  base_harmful={ks['base_rate_harmful_B<0']:.3f}  mean_B={ks['mean_B']:+.4f}", flush=True)
    print(f"detectability  : {ks['detectability_verdict']} (best harm-AUC={ks['best_single_feature_harm_AUC']})", flush=True)
    print(f"multicand route: mean_tau={mb.get('mean_tau')} abstain={mb.get('abstention_rate')} breakdown={mb.get('routing_breakdown')}", flush=True)
    print((f"tau range      : [{td[0]:.3f}..{td[-1]:.3f}] (tau*={a.tau_star})") if td else "tau range: n/a", flush=True)
    print(f"manifest -> {out}", flush=True)
    return out


if __name__ == "__main__":
    main()
