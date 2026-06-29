"""
analyze_iwildcam_kbound.py - SOURCE-CALIBRATED K-Bound verdict for WILDS iWildCam.

LEAKAGE-FREE PROTOCOL
  SOURCE = id_val conditions (in-distribution).  Used ONLY to (i) fit the route-(a)
    KGA certificate  Z -> B  and its split/cross-conformal radius eps, and (ii) pick
    the route-(b) tau* = (1-alpha) quantile of SOURCE tau.  The deployed adapter is
    also chosen on SOURCE (best source always-adapt macro-F1).
  TARGET = val (and/or test) OOD conditions.  The source-calibrated certificate
    DECIDES adapt/freeze/abstain, and ONLY THEN are decisions scored against the
    held-out target labels.  No target label ever touches calibration -> the prior
    "tau tuned on the eval set" integrity bug cannot occur here.

METRIC  WILDS macro-F1, recomputed offline from stored per-sample preds + eval_y
  (every number is reproducible from the manifest).  B = F1(adapted) - F1(frozen).

VERDICT
  goldilocks  = helpful on some target conditions AND harmful on others AND the harm
                is detectable label-free (harm-AUC >= 0.75).
  beats_both  = KGA regret-to-oracle < BOTH always-adapt and always-freeze on TARGET.
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import KFold
from sklearn.metrics import f1_score, balanced_accuracy_score

ALPHA = 0.10
THR = 0.02


def load_manifest(path):
    obj = json.load(open(path))
    recs = obj.get("records", [])
    conds = obj.get("conditions", [])
    return obj, recs, conds


def cond_key(d):
    return (int(d["seed"]), int(d["location"]), d["comp"], d["regime"], d["aggr"])


def metric_of(y, preds, metric):
    y = np.asarray(y, int); preds = np.asarray(preds, int)
    if metric == "macro_f1":
        return float(f1_score(y, preds, average="macro"))
    if metric == "balanced_acc":
        return float(balanced_accuracy_score(y, preds))
    if metric == "accuracy":
        return float((y == preds).mean())
    raise ValueError(metric)


def recompute(recs, conds, metric):
    """Recompute a0/aa/B for every record from stored preds + eval_y (audit-grade)."""
    ey = {cond_key(c): np.asarray(c["eval_y"], int) for c in conds}
    fz = {cond_key(c): np.asarray(c["preds_frozen"], int) for c in conds if "preds_frozen" in c}
    out = []
    for r in recs:
        k = cond_key(r)
        if k not in ey:
            continue
        y = ey[k]
        a0 = metric_of(y, fz[k], metric) if k in fz else float(r["a0"])
        aa = metric_of(y, r["preds"], metric) if "preds" in r else float(r["aa"])
        rr = dict(r)
        rr["a0"] = a0; rr["aa"] = aa; rr["B"] = float(aa - a0)
        out.append(rr)
    return out


def _auc(score, label):
    score = np.asarray(score, float); label = np.asarray(label, int)
    pos = score[label == 1]; neg = score[label == 0]
    if len(pos) == 0 or len(neg) == 0:
        return None
    order = np.argsort(score, kind="mergesort")
    ranks = np.empty(len(score), float); ranks[order] = np.arange(1, len(score) + 1)
    s = score[order]; i = 0
    while i < len(s):
        j = i
        while j + 1 < len(s) and s[j + 1] == s[i]:
            j += 1
        if j > i:
            ranks[order[i:j + 1]] = (i + 1 + j + 1) / 2.0
        i = j + 1
    rp = ranks[label == 1].sum()
    return float((rp - len(pos) * (len(pos) + 1) / 2.0) / (len(pos) * len(neg)))


def fit_certificate(Z, B, alpha=ALPHA, seed=0):
    """Fit Z->B on SOURCE; cross-conformal eps from source CV residuals."""
    Z = np.asarray(Z, float); B = np.asarray(B, float); n = len(B)
    res = np.zeros(n)
    if n >= 4:
        kf = KFold(n_splits=min(5, n), shuffle=True, random_state=seed)
        for tr, te in kf.split(Z):
            m = GradientBoostingRegressor(n_estimators=250, max_depth=2, learning_rate=0.05,
                                          subsample=0.8, random_state=seed).fit(Z[tr], B[tr])
            res[te] = np.abs(m.predict(Z[te]) - B[te])
        eps = float(np.quantile(res, 1 - alpha))
    else:
        eps = float(np.std(B) + 1e-6)
    full = GradientBoostingRegressor(n_estimators=250, max_depth=2, learning_rate=0.05,
                                     subsample=0.8, random_state=seed).fit(Z, B)
    return full, eps


def policy(dec, a0, aa, B, alpha=ALPHA):
    a0 = np.asarray(a0, float); aa = np.asarray(aa, float); B = np.asarray(B, float)
    dec = np.asarray(dec)
    adapt = dec == "ADAPT"
    realized = np.where(adapt, aa, a0)
    oracle = np.maximum(a0, aa)
    rk = float((oracle - realized).mean()); ra = float((oracle - aa).mean()); rf = float((oracle - a0).mean())
    return {
        "n": int(len(a0)),
        "decision_counts": {d: int((dec == d).sum()) for d in ["ADAPT", "FREEZE", "ABSTAIN"]},
        "mean_F1": {"always_adapt": float(aa.mean()), "always_freeze": float(a0.mean()),
                    "K_Bound": float(realized.mean()), "oracle": float(oracle.mean())},
        "regret_vs_oracle": {"always_adapt": ra, "always_freeze": rf, "K_Bound": rk},
        "false_adapt_count": int(np.sum(adapt & (B < 0))),
        "adapt_count": int(adapt.sum()),
        "false_adapt_rate_among_adapt": (float(np.mean(B[adapt] < 0)) if adapt.any() else None),
        "worst_case_F1": {"always_adapt": float(aa.min()), "always_freeze": float(a0.min()),
                          "K_Bound": float(realized.min())},
        "alpha_false_adapt_budget": float(alpha),
        "beats_both_regret_only": bool(rk < ra - 1e-9 and rk < rf - 1e-9),
        "beats_both": bool(rk < ra - 1e-9 and rk < rf - 1e-9
                           and adapt.any() and float(np.mean(B[adapt] < 0)) <= alpha),
    }


def route_a(src, tgt, alpha=ALPHA):
    """Single-candidate KGA, source-fit -> target-applied, per candidate."""
    out = {}
    cands = sorted(set(r["candidate"] for r in tgt))
    for c in cands:
        rs = [r for r in src if r["candidate"] == c]
        rt = [r for r in tgt if r["candidate"] == c]
        if len(rs) < 4 or len(rt) < 2:
            out[c] = {"note": f"insufficient cells src={len(rs)} tgt={len(rt)}"}
            continue
        Zs = np.array([r["Z"] for r in rs]); Bs = np.array([r["B"] for r in rs])
        Zt = np.array([r["Z"] for r in rt]); Bt = np.array([r["B"] for r in rt])
        a0t = np.array([r["a0"] for r in rt]); aat = np.array([r["aa"] for r in rt])
        model, eps = fit_certificate(Zs, Bs, alpha)
        Bhat = model.predict(Zt)
        dec = np.where(Bhat - eps > 0, "ADAPT", np.where(Bhat + eps < 0, "FREEZE", "ABSTAIN"))
        pm = policy(dec, a0t, aat, Bt)
        pm["eps_conformal_source"] = float(eps)
        pm["src_mean_B"] = float(Bs.mean()); pm["src_base_rate_harmful"] = float(np.mean(Bs < 0))
        pm["tgt_mean_B"] = float(Bt.mean()); pm["tgt_base_rate_harmful"] = float(np.mean(Bt < 0))
        pm["tgt_B_range"] = [float(Bt.min()), float(Bt.max())]
        harm = (Bt < 0).astype(int)
        pm["certificate_harm_AUC"] = _auc(-Bhat, harm) if harm.sum() not in (0, len(harm)) else None
        out[c] = pm
    return out


def recompute_committed(b_hat, margin):
    b = np.asarray(b_hat, float); M = len(b)
    adv = b[1:] - b[0]
    committed = [i + 1 for i in range(M - 1) if adv[i] > margin and b[i + 1] > 0]
    choice = int(max(committed, key=lambda i: b[i])) if committed else None
    return committed, choice


def route_b_decisions(conds, tau_star):
    """Recompute route-(b) decision per condition for a given tau* from stored fields."""
    decs, choices = [], []
    for c in conds:
        rt = c.get("route", {})
        tau = rt.get("tau"); b_hat = rt.get("b_hat"); margin = rt.get("margin")
        if b_hat is None or margin is None:
            # |D|<min_D (candidates agree) or M<4 -> stored decision (FREEZE/ABSTAIN); outcome=frozen
            decs.append(rt.get("decision", "ABSTAIN") if rt.get("decision") in ("FREEZE", "ABSTAIN") else "ABSTAIN")
            choices.append(None); continue
        if tau is None or tau > tau_star:
            decs.append("ABSTAIN"); choices.append(None); continue
        committed, choice = recompute_committed(b_hat, margin)
        if choice is None:
            decs.append("FREEZE"); choices.append(None)
        else:
            decs.append("ADAPT"); choices.append(choice)
    return decs, choices


def route_b_score(conds, decs, choices):
    a0 = np.array([c["a0"] for c in conds])
    aa_all = [c["aa_all"] for c in conds]
    oracle = np.array([max(a) for a in aa_all])
    K = len(aa_all[0]) - 1
    aa_mat = np.array([a[1:] for a in aa_all])
    fixed_idx = int(np.argmax(aa_mat.mean(0)))
    fixed_aa = aa_mat[:, fixed_idx]
    realized = np.array([aa_all[i][choices[i]] if (decs[i] == "ADAPT" and choices[i] is not None)
                         else a0[i] for i in range(len(conds))])
    rk = float((oracle - realized).mean()); rf = float((oracle - a0).mean()); rfa = float((oracle - fixed_aa).mean())
    adapt = np.array([d == "ADAPT" for d in decs])
    false_adapt = int(np.sum(adapt & (realized < a0 - 1e-9)))
    return {
        "n_conditions": len(conds),
        "decision_counts": {d: int(sum(1 for x in decs if x == d)) for d in ["ADAPT", "FREEZE", "ABSTAIN"]},
        "fixed_best_always_adapt_idx": fixed_idx,
        "mean_F1": {"router": float(realized.mean()), "always_freeze": float(a0.mean()),
                    "best_fixed_always_adapt": float(fixed_aa.mean()), "oracle": float(oracle.mean())},
        "regret_vs_oracle": {"router": rk, "always_freeze": rf, "best_fixed_always_adapt": rfa},
        "false_adapt_count": false_adapt,
        "beats_both": bool(rk < rf - 1e-9 and rk < rfa - 1e-9),
    }


def route_b(src_conds, tgt_conds, alpha=ALPHA):
    src_tau = np.array([c["route"].get("tau") for c in src_conds
                        if c.get("route", {}).get("tau") is not None], float)
    if len(src_tau) == 0:
        return {"note": "no source tau available"}
    tau_star = float(np.quantile(src_tau, 1 - alpha))
    decs, choices = route_b_decisions(tgt_conds, tau_star)
    res = route_b_score(tgt_conds, decs, choices)
    res["tau_star_source_calibrated"] = tau_star
    res["tau_star_rule"] = f"(1-alpha)={1-alpha:.2f} quantile of SOURCE tau"
    res["source_tau_summary"] = {"n": int(len(src_tau)), "median": float(np.median(src_tau)),
                                 "min": float(src_tau.min()), "max": float(src_tau.max())}
    # target tau* sensitivity sweep (transparency; NOT used to pick the headline)
    sweep = {}
    for q in (0.5, 0.75, 0.9, 0.95):
        ts = float(np.quantile(src_tau, q))
        d2, c2 = route_b_decisions(tgt_conds, ts)
        sweep[f"src_q{q}"] = {"tau_star": ts, **{k: route_b_score(tgt_conds, d2, c2)[k]
                                                  for k in ("regret_vs_oracle", "beats_both")}}
    res["tau_star_sensitivity"] = sweep
    return res


def detectability(tgt, src, evidence_names, alpha=ALPHA):
    Zt = np.array([r["Z"] for r in tgt]); Bt = np.array([r["B"] for r in tgt])
    harm = (Bt < 0).astype(int)
    out = {"n_cells": len(tgt), "n_harmful": int(harm.sum()), "base_rate_harmful": float(harm.mean()),
           "mean_B": float(Bt.mean()), "per_feature_harm_AUC": {}}
    if harm.sum() in (0, len(harm)):
        out["detectability_verdict"] = "n/a (no regime variation on target)"
        out["best_single_feature_harm_AUC"] = None
        return out
    for k in range(Zt.shape[1]):
        zk = Zt[:, k]
        a = max([x for x in (_auc(zk, harm), _auc(-zk, harm)) if x is not None], default=None)
        nm = evidence_names[k] if k < len(evidence_names) else f"z{k}"
        out["per_feature_harm_AUC"][nm] = a
    aucs = [v for v in out["per_feature_harm_AUC"].values() if v is not None]
    out["best_single_feature_harm_AUC"] = float(max(aucs)) if aucs else None
    # source-fit certificate (pooled across candidates) harm-AUC on target = operational detectability
    Zs = np.array([r["Z"] for r in src]); Bs = np.array([r["B"] for r in src])
    if len(Bs) >= 4:
        model, eps = fit_certificate(Zs, Bs, alpha)
        out["certificate_harm_AUC_sourcefit_on_target"] = _auc(-model.predict(Zt), harm)
        out["certificate_eps_source"] = float(eps)
    b = out["best_single_feature_harm_AUC"]
    out["detectability_verdict"] = ("detectable" if (b or 0) >= 0.75 else
                                    "weak" if (b or 0) >= 0.6 else "undetectable")
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--source", required=True, help="manifest for SOURCE (id_val) conditions")
    p.add_argument("--target", required=True, help="manifest for TARGET (val/test) conditions")
    p.add_argument("--metric", default="macro_f1", choices=["macro_f1", "balanced_acc", "accuracy"])
    p.add_argument("--alpha", type=float, default=ALPHA)
    p.add_argument("--out", default="")
    args = p.parse_args()

    EV = ["pre_entropy", "pre_conf", "pre_pbal", "post_entropy", "post_conf", "post_pbal",
          "pbal_drop", "entropy_drop", "frac_highconf", "marginal_KL", "update_norm"]
    so, sr, sc = load_manifest(args.source)
    to, tr, tc = load_manifest(args.target)
    src = recompute(sr, sc, args.metric)
    tgt = recompute(tr, tc, args.metric)

    # deployed adapter chosen on SOURCE (best source always-adapt F1) -> leakage-free
    cand_src_mean = {c: float(np.mean([r["aa"] for r in src if r["candidate"] == c]))
                     for c in sorted(set(r["candidate"] for r in src))}
    deployed = max(cand_src_mean, key=cand_src_mean.get) if cand_src_mean else None

    ra = route_a(src, tgt, args.alpha)
    rb = route_b(sc, tc, args.alpha)
    det = detectability(tgt, src, EV, args.alpha)

    Bt = np.array([r["B"] for r in tgt])
    base_h = float(np.mean(Bt < 0)); meanB = float(Bt.mean())
    frac_help = float(np.mean(Bt > THR))
    detv = det.get("detectability_verdict")
    if base_h < 0.15 and meanB > 0:
        regime = "helpful-dominated"
    elif base_h > 0.85:
        regime = "harmful-dominated" + ("+detectable" if detv == "detectable" else "+undetectable")
    else:
        regime = "mixed+detectable" if detv == "detectable" else "mixed+undetectable"
    goldilocks = bool(frac_help >= 0.05 and base_h >= 0.15 and detv == "detectable")
    dep = ra.get(deployed, {}) if deployed else {}
    beats_both_a = bool(dep.get("beats_both", False))
    beats_both_b = bool(rb.get("beats_both", False))

    verdict = {
        "schema": "kbound_iwildcam_sourcecalibrated_verdict_v1",
        "metric": args.metric, "alpha": args.alpha,
        "source_manifest": str(args.source), "target_manifest": str(args.target),
        "n_source_records": len(src), "n_target_records": len(tgt),
        "n_source_conditions": len(sc), "n_target_conditions": len(tc),
        "deployed_adapter_chosen_on_source": deployed, "source_mean_F1_by_candidate": cand_src_mean,
        "target_benefit": {"mean_B": meanB, "base_rate_harmful_B<0": base_h,
                           "frac_helpful_B>thr": frac_help, "thr": THR,
                           "B_range": [float(Bt.min()), float(Bt.max())]},
        "detectability_target": det,
        "route_a_single_candidate_sourcecal": ra,
        "route_b_multicandidate_sourcecal": rb,
        "regime": regime,
        "goldilocks": goldilocks,
        "beats_both_route_a_deployed": beats_both_a,
        "beats_both_route_b_multicand": beats_both_b,
        "beats_both": bool(beats_both_a or beats_both_b) and goldilocks,
    }
    out = args.out or str(Path(args.target).parent / f"VERDICT_{args.metric}.json")
    json.dump(verdict, open(out, "w"), indent=2)
    print("=" * 80)
    print(f"METRIC={args.metric}  deployed_adapter(source-chosen)={deployed}")
    print(f"TARGET benefit: mean_B={meanB:+.4f} base_rate_harmful={base_h:.3f} "
          f"frac_helpful={frac_help:.3f} B_range=[{Bt.min():+.4f},{Bt.max():+.4f}]")
    print(f"detectability: verdict={detv} best_feature_harm_AUC={det.get('best_single_feature_harm_AUC')} "
          f"certificate_harm_AUC={det.get('certificate_harm_AUC_sourcefit_on_target')}")
    if isinstance(dep, dict) and "regret_vs_oracle" in dep:
        print(f"route_a[{deployed}] regret KGA={dep['regret_vs_oracle']['K_Bound']:.4f} "
              f"adapt={dep['regret_vs_oracle']['always_adapt']:.4f} "
              f"freeze={dep['regret_vs_oracle']['always_freeze']:.4f} beats_both={dep.get('beats_both')} "
              f"false_adapt={dep.get('false_adapt_count')}")
    if "regret_vs_oracle" in rb:
        print(f"route_b tau*={rb.get('tau_star_source_calibrated'):.4f} regret router={rb['regret_vs_oracle']['router']:.4f} "
              f"freeze={rb['regret_vs_oracle']['always_freeze']:.4f} "
              f"best_fixed_adapt={rb['regret_vs_oracle']['best_fixed_always_adapt']:.4f} beats_both={rb.get('beats_both')}")
    print(f"REGIME={regime}  GOLDILOCKS={goldilocks}  BEATS_BOTH={verdict['beats_both']}")
    print(f"verdict -> {out}")
    return out


if __name__ == "__main__":
    main()
