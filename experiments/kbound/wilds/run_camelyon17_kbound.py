"""
run_camelyon17_kbound.py - K-Bound natural-shift pipeline on WILDS Camelyon17.

PROTOCOL
  frozen f0       : standard WILDS DenseNet-121 (loaded from results/wilds/f0_seed{S}.pt)
  candidates (6)  : {tent,eata,sar} x {online,episodic}
  baselines       : always-adapt (per candidate), always-freeze, per-condition oracle
  routing         : (a) single-candidate KGA certificate  [analysis.decide_kga]
                    (b) multi-candidate tau-residual        [Theorem 1A; analysis.multicandidate_route
                        -> reuses theory_validation/val_multicandidate_residual.py]
                    (c) smooth-drift                        [Theorem 1B; TODO STUB]
  metrics         : mean acc, regret-to-oracle, false-adapt rate, coverage, abstention,
                    per-condition routing breakdown
  detectability   : per-condition label-free Z vs TRUE benefit sign  [analysis.detectability_analysis]

CONDITION = (domain, composition, batch_regime) x (aggressiveness) x seed.
INTEGRITY: every cell is run for real; labels are used ONLY for B/oracle/detectability
eval; the routers see only Z (a) or label-free agreements (b).  Every reported number
traces to records[] / conditions[] in the output JSON manifest.  Run for real; never
fabricate; report null/negative results as-is.
"""
from __future__ import annotations
import os, sys, json, time, argparse, platform, hashlib
import numpy as np
import torch
import torch.nn as nn

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)
import tta_methods as tm        # noqa: E402
import analysis as an           # noqa: E402
import cam_data as cd           # noqa: E402
import per_condition_serialize as pcs  # noqa: E402  (torch-free per-condition serializer)
# integrity: the duplicated EVIDENCE_NAMES in the serializer must stay in lock-step
assert list(pcs.EVIDENCE_NAMES) == list(tm.EVIDENCE_NAMES), "EVIDENCE_NAMES drift"

CANDIDATES = [("tent", "online"), ("tent", "episodic"),
              ("eata", "online"), ("eata", "episodic"),
              ("sar", "online"), ("sar", "episodic")]
AGGR = {"mild": dict(steps=10, lr=1e-3), "aggressive": dict(steps=50, lr=2.5e-3)}
NUM_CLASSES = 2


def make_model(device):
    import torchvision.models as tv
    m = tv.densenet121(weights=None)            # arch only; weights come from the f0 ckpt
    m.classifier = nn.Linear(m.classifier.in_features, NUM_CLASSES)
    return m.to(device)


def load_f0(ckpt, device):
    m = make_model(device)
    sd = torch.load(ckpt, map_location=device)
    m.load_state_dict(sd)
    m.eval()
    return m


def route_realized(route, aa_all):
    """Realized balanced acc of the multi-candidate router on a condition."""
    if route.get("decision") == "ADAPT" and route.get("choice") is not None:
        return float(aa_all[route["choice"]])
    return float(aa_all[0])                      # FREEZE / ABSTAIN / ERROR -> frozen f0


def _cell_key(seed, dom, comp, regime, aggr):
    return (int(seed), dom, comp, regime, aggr)


def load_partial(partial_path):
    if not partial_path or not os.path.exists(partial_path):
        return [], [], set()
    with open(partial_path) as f:
        d = json.load(f)
    records = d.get("records", [])
    conditions = d.get("conditions", [])
    done = {_cell_key(c["seed"], c["domain"], c["comp"], c["regime"], c["aggr"])
            for c in conditions}
    return records, conditions, done


def run(args, partial_path=None):
    t_start = time.time()
    device = tm.pick_device(args.device)
    print(f"[device] {device}")
    ds, transform, keep_full, n_present, n_total = cd.load_camelyon(args.data_root)
    drop = n_total - n_present
    print(f"[disk-filter] {n_present}/{n_total} patches present "
          f"({drop} dropped, {100.0*drop/max(n_total,1):.2f}%)")
    dom_cache = {dn: cd.make_domain(ds, transform, keep_full, dn) for dn in args.domains}
    source_cache = None
    if args.evidence_panel == "rich":
        source_cache = cd.make_domain(ds, transform, keep_full, "train")
    for dn in args.domains:
        print(f"[domain] {dn}: {len(dom_cache[dn][0])} present patches")
    if source_cache is not None:
        print(f"[source] train: {len(source_cache[0])} present patches for source-calibrated rich evidence")

    records, conditions, done = ([], [], set())
    if partial_path and args.resume:
        records, conditions, done = load_partial(partial_path)
        if done:
            print(f"[resume] {len(done)} cells loaded from {partial_path}", flush=True)
    n_cells_total = (len(args.seeds) * len(args.domains) * len(args.compositions)
                     * len(args.batch_regimes) * len(args.aggressiveness))
    cell_i = 0
    for seed in args.seeds:
        torch.manual_seed(seed); np.random.seed(seed)
        rng = np.random.default_rng(seed)
        f0_ckpt = args.f0_template.format(seed=seed)
        if not os.path.exists(f0_ckpt):
            raise FileNotFoundError(f"f0 checkpoint missing: {f0_ckpt}")
        f0 = load_f0(f0_ckpt, device)
        logits_src = y_src = None
        if args.evidence_panel == "rich":
            src_rng = np.random.default_rng(100000 + int(seed))
            _, src_x, y_src = cd.build_condition(
                source_cache[0], source_cache[1], "iid", 64,
                min(int(args.n_eval), 1024), src_rng, device, n_batches=1)
            logits_src = tm.predict_logits(f0, src_x, train_mode=False, bs=256)
        for dom in args.domains:
            sub, y = dom_cache[dom]
            for comp in args.compositions:
                for regime in args.batch_regimes:
                    bs = cd.BATCH_REGIMES[regime]
                    for aggr in args.aggressiveness:
                        steps = args.steps_override or AGGR[aggr]["steps"]
                        lr = AGGR[aggr]["lr"]
                        cell_i += 1
                        tag = f"s{seed}/{dom}/{comp}/{regime}/{aggr}"
                        if _cell_key(seed, dom, comp, regime, aggr) in done:
                            print(f"  [{cell_i}/{n_cells_total}] {tag}  SKIP (resume)", flush=True)
                            continue
                        try:
                            stream, eval_x, eval_y = cd.build_condition(
                                sub, y, comp, bs, args.n_eval, rng, device, n_batches=args.n_batches)
                        except Exception as e:
                            print(f"  [{cell_i}/{n_cells_total}] {tag}  SKIP build: {e}")
                            continue
                        a0, p0, p0_pos = tm.eval_frozen(f0, eval_x, eval_y)
                        logits_f0_eval = None
                        bn_kl = 0.0
                        rich_note = None
                        if args.evidence_panel == "rich":
                            logits_f0_eval = tm.predict_logits(f0, eval_x, train_mode=False, bs=256)
                            rm, rv = tm.bn_running_stats(f0)
                            bm, bv = tm.bn_batch_stats(f0, stream[0])
                            if rm is None or bm is None:
                                rich_note = "BN stats unavailable; bn_kl set to 0"
                            bn_kl = tm.bn_stat_kl_drift(rm, rv, bm, bv)
                        stream_f0_pos = tm._predict_prob(f0, torch.cat(stream, 0), train_mode=False, bs=256)
                        preds_all = [p0]; aa_all = [a0]; cand_names = ["freeze_f0"]
                        best_pa = p0_pos; best_aa_c = float(a0)
                        for (method, mode) in CANDIDATES:
                            cand_out = tm.run_candidate(
                                method, mode, f0, stream, eval_x, eval_y, NUM_CLASSES,
                                steps, lr, eval_bs=min(bs, 64),
                                return_details=(args.evidence_panel == "rich"))
                            if args.evidence_panel == "rich":
                                aa, Z_base, upd, preds, pa_pos, details = cand_out
                                rich = tm.rich_evidence_vector(
                                    logits_f0_eval, details["logits_eval"], logits_src, y_src, bn_kl)
                                Z = list(Z_base) + [float(x) for x in rich]
                            else:
                                aa, Z, upd, preds, pa_pos = cand_out
                                Z_base = Z
                                rich = None
                            B = float(aa - a0)
                            rec = dict(
                                seed=int(seed), domain=dom, comp=comp, regime=regime, aggr=aggr,
                                method=method, mode=mode, candidate=f"{method}_{mode}",
                                a0=float(a0), aa=float(aa), B=B, upd_norm=float(upd),
                                Z=[float(z) for z in Z], Z_base=[float(z) for z in Z_base],
                                evidence_panel=args.evidence_panel,
                                regime_label=an.label_regime(B))
                            if args.evidence_panel == "rich":
                                rec.update({
                                    "Z_rich": [float(x) for x in rich],
                                    "bn_kl": float(bn_kl),
                                    "rich_note": rich_note,
                                })
                            records.append(rec)
                            preds_all.append(preds); aa_all.append(float(aa))
                            cand_names.append(f"{method}_{mode}")
                            if float(aa) > best_aa_c:
                                best_aa_c = float(aa); best_pa = pa_pos
                            tm.mps_free()
                        preds_mat = np.stack(preds_all, 0)
                        route = an.multicandidate_route(preds_mat, tau_star=args.tau_star, kappa=args.kappa)
                        realized = route_realized(route, aa_all)
                        oracle = float(max(aa_all)); best_adapt = float(max(aa_all[1:]))
                        try:
                            route_c = an.smooth_drift_route(p0_pos, best_pa, stream_f0_pos, L=args.sd_L)
                            if route_c.get("implemented") and "bracket" in route_c:
                                trueB = best_adapt - a0
                                route_c["true_B_best"] = float(trueB)
                                route_c["bracket_covers_trueB"] = bool(
                                    route_c["bracket"][0] <= trueB <= route_c["bracket"][1])
                        except Exception as e:
                            route_c = {"decision": "ERROR", "implemented": False, "reason": repr(e)}
                        conditions.append(dict(
                            seed=int(seed), domain=dom, comp=comp, regime=regime, aggr=aggr,
                            cand_names=cand_names, aa_all=[float(a) for a in aa_all], a0=float(a0),
                            oracle=oracle, best_adapt=best_adapt,
                            true_best=cand_names[int(np.argmax(aa_all))],
                            route=route, route_c=route_c, realized=realized,
                            regime_label=an.label_regime(best_adapt - a0)))
                        print(f"  [{cell_i}/{n_cells_total}] {tag}  a0={a0:.3f} "
                              f"best_aa={best_adapt:.3f} oracle={oracle:.3f} "
                              f"route={route.get('decision')}->{route.get('choice')} "
                              f"tau={route.get('tau', float('nan')):.4f} "
                              f"sd_c={route_c.get('decision')}")
                        # lightweight partial flush: raw records recoverable if killed
                        if partial_path:
                            try:
                                with open(partial_path, "w") as pf:
                                    json.dump({"progress": f"{cell_i}/{n_cells_total}",
                                               "elapsed_sec": round(time.time() - t_start, 1),
                                               "records": records, "conditions": conditions}, pf)
                            except Exception:
                                pass
        del f0; tm.mps_free()
    return records, conditions, {
        "n_present": n_present, "n_total": n_total, "wall_sec": time.time() - t_start}


def aggregate_single_candidate(records):
    """(a) per-candidate single-candidate KGA certificate + policy metrics."""
    out = {}
    cands = sorted(set(r["candidate"] for r in records))
    for c in cands:
        rs = [r for r in records if r["candidate"] == c]
        a0 = np.array([r["a0"] for r in rs]); aa = np.array([r["aa"] for r in rs])
        B = aa - a0; Z = np.array([r["Z"] for r in rs])
        entry = {"n_cells": len(rs), "mean_B": float(B.mean()),
                 "base_rate_harmful_B<0": float(np.mean(B < 0)),
                 "mean_acc": {"always_adapt": float(aa.mean()), "always_freeze": float(a0.mean())}}
        if len(rs) >= 2 and len(np.unique(B)) > 1:
            Bhat, eps, dec = an.decide_kga(Z, B)
            pm = an.policy_metrics(dec, a0, aa, B)
            pm["eps_conformal"] = float(eps)
            entry["kga"] = pm
        else:
            entry["kga"] = {"note": "need >=2 cells with B variation for the cross-cell certificate"}
        out[c] = entry
    return out


def aggregate_multicandidate(conditions, alpha=0.10):
    """(b) multi-candidate tau-route metrics + routing breakdown.
    beats_both REQUIRES the pre-registered false-adapt budget FA<=alpha, not regret alone."""
    if not conditions:
        return {"note": "no conditions"}
    a0 = np.array([c["a0"] for c in conditions])
    oracle = np.array([c["oracle"] for c in conditions])
    realized = np.array([c["realized"] for c in conditions])
    dec = np.array([c["route"].get("decision", "ERROR") for c in conditions])
    tau = np.array([c["route"].get("tau", np.nan) for c in conditions], float)
    adapt = dec == "ADAPT"
    # fixed best always-adapt candidate (max mean aa across conditions)
    names = conditions[0]["cand_names"][1:]
    aa_mat = np.array([c["aa_all"][1:] for c in conditions])     # (cells, K)
    fixed_idx = int(np.argmax(aa_mat.mean(0)))
    fixed_aa = aa_mat[:, fixed_idx]
    breakdown = {}
    for c in conditions:
        ch = c["route"].get("choice")
        nm = c["cand_names"][ch] if ch is not None else c["route"].get("decision", "ERROR")
        breakdown[nm] = breakdown.get(nm, 0) + 1
    by_regime = {}
    for c in conditions:
        g = c["regime_label"]; d = c["route"].get("decision", "ERROR")
        by_regime.setdefault(g, {}).setdefault(d, 0)
        by_regime[g][d] += 1
    by_domain = {}
    for c in conditions:
        d0 = c["domain"]; d = c["route"].get("decision", "ERROR")
        by_domain.setdefault(d0, {}).setdefault(d, 0)
        by_domain[d0][d] += 1
    return {
        "n_conditions": len(conditions),
        "mean_acc": {"router": float(realized.mean()), "always_freeze": float(a0.mean()),
                     "best_fixed_always_adapt": float(fixed_aa.mean()),
                     "per_condition_oracle": float(oracle.mean())},
        "regret_vs_oracle": {"router": float((oracle - realized).mean()),
                             "always_freeze": float((oracle - a0).mean()),
                             "best_fixed_always_adapt": float((oracle - fixed_aa).mean())},
        "coverage": float(np.mean((dec == "ADAPT") | (dec == "FREEZE"))),
        "abstention_rate": float(np.mean(dec == "ABSTAIN")),
        "false_adapt_rate": (float(np.mean((realized < a0 - 1e-9)[adapt])) if adapt.any() else None),
        "mean_tau": float(np.nanmean(tau)) if np.isfinite(tau).any() else None,
        "gate_pass_rate": float(np.mean([bool(c["route"].get("gate_pass", False)) for c in conditions])),
        "fixed_best_candidate": names[fixed_idx],
        "routing_breakdown": breakdown,
        "decisions_by_regime": by_regime,
        "decisions_by_domain": by_domain,
        "alpha_false_adapt_budget": float(alpha),
        "beats_both_regret_only": bool((oracle - realized).mean() < (oracle - a0).mean() - 1e-9 and
                                       (oracle - realized).mean() < (oracle - fixed_aa).mean() - 1e-9),
        "beats_both": bool((oracle - realized).mean() < (oracle - a0).mean() - 1e-9 and
                           (oracle - realized).mean() < (oracle - fixed_aa).mean() - 1e-9 and
                           adapt.any() and float(np.mean((realized < a0 - 1e-9)[adapt])) <= alpha),
    }


def kbound_summary(records, conditions, delta=0.05, evidence_names=None):
    """Camelyon17 regime classification + K-Bound gamma_S / gamma_T / tau (debug proxies)."""
    B = np.array([r["B"] for r in records])
    base_h = float(np.mean(B < 0)); meanB = float(B.mean())
    det = an.detectability_analysis(records, evidence_names or tm.EVIDENCE_NAMES)
    verdict = det.get("detectability_verdict", "n/a")
    if base_h < 0.10 and meanB > 0:
        klass = "helpful-dominated"
    elif base_h > 0.60:
        klass = "harmful-dominated"
    else:
        klass = "mixed+detectable" if verdict == "detectable" else "mixed+undetectable"
    id_conds = [c for c in conditions if c["domain"] == "id_val"]
    tg_conds = [c for c in conditions if c["domain"] in ("test", "val")]
    gamma_S = float(np.mean([c["best_adapt"] - c["a0"] for c in id_conds])) if id_conds else None
    gamma_T = float(np.mean([c["oracle"] - c["a0"] for c in tg_conds])) if tg_conds else None
    taus = [c["route"].get("tau") for c in conditions if c["route"].get("tau") is not None]
    return {
        "classification": klass, "base_rate_harmful_B<0": base_h, "mean_B": meanB,
        "detectability_verdict": verdict,
        "best_single_feature_harm_AUC": det.get("best_single_feature_harm_AUC"),
        "gamma_S_proxy_indist_advantage": gamma_S,
        "gamma_T_proxy_oracle_advantage": gamma_T,
        "delta": delta,
        "abs_gammaT_minus_gammaS": (abs(gamma_T - gamma_S) if (gamma_S is not None and gamma_T is not None) else None),
        "gamma_gap_within_delta": (bool(abs(gamma_T - gamma_S) <= delta)
                                   if (gamma_S is not None and gamma_T is not None) else None),
        "real_data_multicandidate_tau_mean": (float(np.mean(taus)) if taus else None),
        "_note": "gamma_S/gamma_T are operational debug-scale proxies (in-dist vs OOD adaptation "
                 "advantage); final defs follow the paper. Classification honors the integrity policy: "
                 "reported from measured B, never tuned to a target.",
    }


def aggregate_smoothdrift(conditions):
    """(c) Theorem-1B smooth-drift route summary across conditions (diagnostic surrogate)."""
    rcs = [c.get("route_c") for c in conditions if c.get("route_c")]
    impl = [r for r in rcs if r.get("implemented")]
    if not impl:
        return {"implemented": False, "theorem": "1B",
                "reason": (rcs[0].get("reason") if rcs else "no route_c computed"),
                "note": "smooth-drift route not executed"}
    from collections import Counter
    dec = [r["decision"] for r in impl]
    cov = [r["bracket_covers_trueB"] for r in impl if r.get("bracket_covers_trueB") is not None]
    return {
        "implemented": True, "theorem": "1B", "view": "brier_squared_loss",
        "gS_estimate": "f0_surrogate(conservative)", "n_conditions": len(impl),
        "decision_counts": dict(Counter(dec)),
        "mean_center_c": float(np.mean([r["center_c"] for r in impl])),
        "mean_reach": float(np.mean([r["reach"] for r in impl])),
        "mean_d_obs": float(np.mean([r["d_obs"] for r in impl])),
        "bracket_coverage_trueB": (float(np.mean(cov)) if cov else None),
        "status": "DIAGNOSTIC (wired to val_smooth_drift.py; conservative g_S~f0 surrogate)",
        "note": impl[0].get("note"),
    }


REPO = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))   # .../AutoML_Flagship_V8


def build_manifest(args, records, conditions, meta):
    cfg = {k: getattr(args, k) for k in (
        "data_root", "f0_template", "seeds", "domains", "compositions", "batch_regimes",
        "aggressiveness", "n_eval", "n_batches", "tau_star", "kappa", "device",
        "steps_override", "delta", "sd_L", "evidence_panel", "smoke")}
    cfg_sha = hashlib.sha256(json.dumps(cfg, sort_keys=True).encode()).hexdigest()[:8]
    evidence_names = list(tm.EVIDENCE_NAMES)
    if args.evidence_panel == "rich":
        evidence_names += list(tm.RICH_EVIDENCE_NAMES)
    return {
        "schema": "kbound_wilds_camelyon17_v0.6",
        "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "host": {"node": platform.node(), "platform": platform.platform(),
                 "python": platform.python_version(), "torch": torch.__version__,
                 "mps": bool(torch.backends.mps.is_available())},
        "config": cfg, "config_sha8": cfg_sha,
        "evidence_panel": args.evidence_panel,
        "evidence_names": evidence_names,
        "f0_checkpoints": {int(s): args.f0_template.format(seed=s) for s in args.seeds},
        "data": {"data_root": args.data_root, "n_present": meta["n_present"],
                 "n_total": meta["n_total"], "n_dropped_disk_filter": meta["n_total"] - meta["n_present"],
                 "wall_sec": round(meta["wall_sec"], 1)},
        "candidates": [f"{m}_{md}" for (m, md) in CANDIDATES],
        "baselines": {
            "always_freeze_mean_acc": float(np.mean([r["a0"] for r in records])) if records else None,
            "per_candidate_always_adapt_mean_acc": {
                c: float(np.mean([r["aa"] for r in records if r["candidate"] == c]))
                for c in sorted(set(r["candidate"] for r in records))},
            "per_condition_oracle_mean_acc": float(np.mean([c["oracle"] for c in conditions])) if conditions else None,
        },
        "routing_a_single_candidate": aggregate_single_candidate(records),
        "routing_b_multicandidate": aggregate_multicandidate(conditions),
        "routing_c_smooth_drift": aggregate_smoothdrift(conditions),
        "detectability": an.detectability_analysis(records, evidence_names) if len(records) >= 4 else {"note": "need >=4 records"},
        "kbound_summary": kbound_summary(records, conditions, delta=args.delta, evidence_names=evidence_names),
        "records": records,
        "conditions": conditions,
    }


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="K-Bound natural-shift pipeline on WILDS Camelyon17")
    p.add_argument("--data-root", default=os.path.expanduser("~/kbound_cam/wilds"),
                   help="dir containing camelyon17_v1.0 (internal copy for speed)")
    p.add_argument("--f0-template",
                   default=os.path.join(REPO, "experiments/kbound/results/wilds/f0_seed{seed}.pt"))
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3])
    p.add_argument("--domains", nargs="+", default=list(cd.DOMAINS))
    p.add_argument("--compositions", nargs="+", default=list(cd.COMPOSITIONS))
    p.add_argument("--batch-regimes", nargs="+", default=["small"], dest="batch_regimes")
    p.add_argument("--aggressiveness", nargs="+", default=["mild", "aggressive"])
    p.add_argument("--n-eval", type=int, default=256, dest="n_eval")
    p.add_argument("--n-batches", type=int, default=4, dest="n_batches")
    p.add_argument("--tau-star", type=float, default=0.08, dest="tau_star")
    p.add_argument("--kappa", type=float, default=2.5)
    p.add_argument("--delta", type=float, default=0.05)
    p.add_argument("--sd-L", type=float, default=0.6, dest="sd_L",
                   help="Theorem-1B drift-smoothness modulus L (variant c)")
    p.add_argument("--evidence-panel", choices=["base", "rich"], default="base",
                   dest="evidence_panel",
                   help="base = legacy 11-dim Z; rich = append Protocol-F drift-aware evidence")
    p.add_argument("--device", default="auto", choices=["auto", "cpu", "mps", "cuda"])
    p.add_argument("--steps-override", type=int, default=0, dest="steps_override")
    p.add_argument("--out", default="")
    p.add_argument("--run-name", default="wilds_kbound", dest="run_name",
                   help="results subdir name under experiments/kbound/results/")
    p.add_argument("--smoke", action="store_true", help="tiny CPU end-to-end smoke")
    p.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True,
                   help="skip cells already in _partial.json (default: on)")
    p.add_argument("--serialize-per-condition", action=argparse.BooleanOptionalAction,
                   default=True, dest="serialize_per_condition",
                   help="also write per_condition_camelyon17_<method>_seed<S>.json files "
                        "(stress_grid_multiseed schema; default: on)")
    args = p.parse_args(argv)
    if args.smoke:
        args.domains = ["test"]; args.compositions = ["iid", "single_class"]
        args.batch_regimes = ["tiny"]; args.aggressiveness = ["mild"]
        args.seeds = [0, 1]; args.n_eval = 32; args.n_batches = 2
        args.steps_override = 4
        if args.device == "auto":
            args.device = "cpu"
    return args


def main(argv=None):
    args = parse_args(argv)
    out_dir = os.path.join(REPO, "experiments/kbound/results",
                           "wilds_kbound_smoke" if args.smoke else args.run_name)
    os.makedirs(out_dir, exist_ok=True)
    partial = os.path.join(out_dir, "_partial.json")
    records, conditions, meta = run(args, partial_path=partial)
    manifest = build_manifest(args, records, conditions, meta)
    out = args.out or os.path.join(out_dir, f"result_{manifest['config_sha8']}.json")
    with open(out, "w") as f:
        json.dump(manifest, f, indent=2)
    # ---- per-condition serialization (stress_grid_multiseed schema) ----------
    if getattr(args, "serialize_per_condition", True) and records:
        methods = sorted({r["method"] for r in records})       # tent, eata, sar
        seeds = [int(s) for s in args.seeds]
        ser = pcs.serialize_run(records, dataset="camelyon17", out_dir=out_dir,
                                seeds=seeds, methods=methods)
        print(f"[serialize] wrote {len(ser['written'])} per-condition files "
              f"(methods={methods}, seeds={seeds}, kga_backend={ser['kga_backend']}) -> {out_dir}")
    ks = manifest["kbound_summary"]; mb = manifest["routing_b_multicandidate"]
    print("\n" + "=" * 70)
    print(f"records={len(records)}  conditions={len(conditions)}  wall={meta['wall_sec']:.1f}s")
    print(f"classification        : {ks['classification']}")
    print(f"base_rate_harmful B<0 : {ks['base_rate_harmful_B<0']:.3f}   mean_B={ks['mean_B']:+.4f}")
    print(f"detectability verdict : {ks['detectability_verdict']} "
          f"(best harm-AUC={ks['best_single_feature_harm_AUC']})")
    print(f"multicand route       : mean_tau={mb.get('mean_tau')}  "
          f"abstain={mb.get('abstention_rate')}  breakdown={mb.get('routing_breakdown')}")
    print(f"gamma_S={ks['gamma_S_proxy_indist_advantage']}  gamma_T={ks['gamma_T_proxy_oracle_advantage']}  "
          f"|dT-dS|<=delta: {ks['gamma_gap_within_delta']}")
    rc = manifest["routing_c_smooth_drift"]
    print(f"smooth-drift (c)      : implemented={rc.get('implemented')} "
          f"decisions={rc.get('decision_counts')} bracket_cov={rc.get('bracket_coverage_trueB')}")
    print(f"\nmanifest -> {out}")
    return out


if __name__ == "__main__":
    main()
