"""
oh_analyze.py - SOURCE-CALIBRATED, leakage-free K-Bound analysis for Office-Home.

Modes:
  regime-scan : per target domain x candidate -> B=Delta acc, regime (helpful/harmful/
                mixed), label-free harm-AUC; per-domain rollups; the WIN PRECONDITION
                = any domain mixed+detectable (and is the DEPLOYED adapter mixed+detectable?).
                Calibrates the harm detector on SOURCE/target-VAL only.  [STEP 1 checkpoint]
  verdict     : full source-calibrated route-a (deployed single candidate) + route-b
                (multi-candidate tau* router) + goldilocks + beats_both.  SOURCE fits the
                certificate Z->B, conformal eps, and tau* = (1-alpha) quantile of SOURCE tau;
                the deployed adapter is the best-SOURCE-accuracy candidate.  TARGET labels
                only score decisions -> tau is NEVER tuned on the eval set.

B = acc(adapted) - acc(frozen), recomputed audit-grade from stored preds + eval_y.
"""
from __future__ import annotations
import argparse, json
from collections import defaultdict
from pathlib import Path
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import KFold

ALPHA = 0.10
THR = 0.02
DET_AUC = 0.75
PRIOR_CANDS = {"labelshift", "conservative"}   # inference-only prior-correction (NOT gradient TTA)


def is_grad(c):
    """Gradient TTA candidate = Tent/EATA/SAR (excludes the prior-correction pair)."""
    return c not in PRIOR_CANDS


def load(path):
    o = json.load(open(path))
    return o, o.get("records", []), o.get("conditions", []), o.get("evidence_names", [])


def ckey(d):
    return (int(d["seed"]), d["domain"], d["split"], d["comp"], d["regime"])


def recompute(records, conditions):
    """Audit-grade a0/aa/B from stored preds + eval_y."""
    ey = {ckey(c): np.asarray(c["eval_y"], int) for c in conditions}
    fz = {ckey(c): np.asarray(c["preds_frozen"], int) for c in conditions}
    out = []
    for r in records:
        k = ckey(r)
        if k not in ey:
            continue
        y = ey[k]
        a0 = float((fz[k] == y).mean())
        aa = float((np.asarray(r["preds"], int) == y).mean()) if "preds" in r else float(r["aa"])
        rr = dict(r); rr["a0"] = a0; rr["aa"] = aa; rr["B"] = float(aa - a0)
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


def best_feature_auc(Z, harm, names):
    aucs = {}
    for k in range(Z.shape[1]):
        a = [x for x in (_auc(Z[:, k], harm), _auc(-Z[:, k], harm)) if x is not None]
        aucs[names[k] if k < len(names) else f"z{k}"] = max(a) if a else None
    vals = [v for v in aucs.values() if v is not None]
    return (float(max(vals)) if vals else None), aucs


def fit_certificate(Z, B, alpha=ALPHA, seed=0):
    Z = np.asarray(Z, float); B = np.asarray(B, float); n = len(B)
    res = np.zeros(n)
    if n >= 4:
        kf = KFold(n_splits=min(5, n), shuffle=True, random_state=seed)
        for tr, te in kf.split(Z):
            m = GradientBoostingRegressor(n_estimators=250, max_depth=2, learning_rate=0.05,
                                          subsample=0.8, random_state=seed).fit(Z[tr], B[tr])
            res[te] = np.abs(m.predict(Z[te]) - B[te])
        # Exact-rank conformal quantile, k = ceil((n+1)(1-alpha)), matching the
        # canonical rule in kga.policy / THEORY_TO_CODE_MAP.md S1a.  Was
        # np.quantile(res, 1-alpha) -- linear interpolation between order
        # statistics, which is anti-conservative at small n and was a THIRD
        # radius rule in a promoted track.  Fixed 2026-07-26 (fix-queue item 25).
        # NOTE: calibration here is on SOURCE cells and scoring is on TARGET
        # cells (see route_a), so the scored cell is never in its own pool;
        # this is a rule-consistency fix, not a leakage fix.
        k = int(np.ceil((n + 1) * (1 - alpha)))
        eps = float(np.inf) if k > n else float(np.sort(res)[k - 1])
    else:
        eps = float(np.std(B) + 1e-6)
    full = GradientBoostingRegressor(n_estimators=250, max_depth=2, learning_rate=0.05,
                                     subsample=0.8, random_state=seed).fit(Z, B)
    return full, eps


def regime_of(B, harm_auc):
    B = np.asarray(B, float); base_h = float(np.mean(B < 0)); meanB = float(B.mean())
    det = "detectable" if (harm_auc or 0) >= DET_AUC else ("weak" if (harm_auc or 0) >= 0.6 else "undetectable")
    if base_h < 0.15 and meanB > 0:
        reg = "helpful-dominated"
    elif base_h > 0.85:
        reg = "harmful-dominated"
    else:
        reg = "mixed"
    return reg, det, base_h, meanB


def grad_detectability(src, tgt, names, alpha=ALPHA):
    """DECISIVE detectability: harm-AUC over GRADIENT-TTA candidates only (Tent/EATA/SAR),
    excluding the trivially-detectable label-shift collapse. Per domain + pooled, plus the
    source-fit certificate transfer-AUC (does the source-calibrated certificate detect harm
    on the OOD target?)."""
    sg = [r for r in src if is_grad(r["candidate"])]
    out = {"_def": "gradient-TTA only = Tent/EATA/SAR x {online,episodic} x {mild,aggressive}; "
                   "labelshift + conservative EXCLUDED", "per_domain": {}}
    model = eps = None
    if len(sg) >= 4 and len(np.unique([r["B"] for r in sg])) > 1:
        model, eps = fit_certificate(np.array([r["Z"] for r in sg]), np.array([r["B"] for r in sg]), alpha)
    def block(rs):
        B = np.array([r["B"] for r in rs]); Z = np.array([r["Z"] for r in rs]); harm = (B < 0).astype(int)
        d = {"n": len(rs), "base_rate_harmful": float(harm.mean()), "mean_B": float(B.mean()),
             "frac_helpful": float(np.mean(B > THR))}
        if harm.sum() in (0, len(harm)):
            d["best_harm_AUC"] = None; d["top_features"] = {}; d["regime"] = (
                "harmful-dominated" if harm.all() else "helpful-dominated"); d["detect"] = "n/a (no harm variation)"
            d["certificate_transfer_AUC"] = None
            return d
        hauc, feats = best_feature_auc(Z, harm, names)
        reg, det, _, _ = regime_of(B, hauc)
        d["best_harm_AUC"] = hauc; d["regime"] = reg; d["detect"] = det
        d["top_features"] = dict(sorted({k: v for k, v in feats.items() if v is not None}.items(),
                                        key=lambda kv: -kv[1])[:6])
        d["certificate_transfer_AUC"] = (_auc(-model.predict(Z), harm) if model is not None else None)
        return d
    for dom in sorted(set(r["domain"] for r in tgt)):
        rs = [r for r in tgt if r["domain"] == dom and is_grad(r["candidate"])]
        if rs:
            out["per_domain"][dom] = block(rs)
    pooled = [r for r in tgt if is_grad(r["candidate"])]
    out["pooled_all_targets"] = block(pooled) if pooled else {}
    out["per_domain_mixed_detectable"] = {d: bool(v.get("regime") == "mixed" and v.get("detect") == "detectable")
                                          for d, v in out["per_domain"].items()}
    out["ANY_domain_gradTTA_mixed_detectable"] = any(out["per_domain_mixed_detectable"].values())
    return out


# ----------------------------- regime scan (STEP 1) -----------------------------
def regime_scan(src, tgt, names):
    domains = sorted(set(r["domain"] for r in tgt))
    cands = sorted(set(r["candidate"] for r in tgt))
    # deployed adapter = best SOURCE accuracy
    src_mean = {c: float(np.mean([r["aa"] for r in src if r["candidate"] == c])) for c in cands}
    deployed = max(src_mean, key=src_mean.get) if src_mean else None

    per_dc = {}
    for dom in domains:
        for c in cands:
            rs = [r for r in tgt if r["domain"] == dom and r["candidate"] == c]
            if not rs:
                continue
            B = np.array([r["B"] for r in rs]); Z = np.array([r["Z"] for r in rs])
            harm = (B < 0).astype(int)
            hauc = None
            if harm.sum() not in (0, len(harm)):
                hauc, _ = best_feature_auc(Z, harm, names)
            reg, det, base_h, meanB = regime_of(B, hauc)
            per_dc[f"{dom}|{c}"] = {"n": len(rs), "mean_B": meanB, "base_rate_harmful": base_h,
                                    "frac_helpful": float(np.mean(B > THR)), "regime": reg,
                                    "harm_AUC": hauc, "detect": det}
    per_dom = {}
    for dom in domains:
        rs = [r for r in tgt if r["domain"] == dom]
        B = np.array([r["B"] for r in rs]); Z = np.array([r["Z"] for r in rs])
        harm = (B < 0).astype(int)
        hauc, featauc = (best_feature_auc(Z, harm, names) if harm.sum() not in (0, len(harm)) else (None, {}))
        reg, det, base_h, meanB = regime_of(B, hauc)
        dep = per_dc.get(f"{dom}|{deployed}", {})
        per_dom[dom] = {
            "n_records": len(rs), "regime_pooled": reg, "detect_pooled": det,
            "base_rate_harmful": base_h, "mean_B": meanB, "frac_helpful": float(np.mean(B > THR)),
            "best_harm_AUC": hauc, "top_features": dict(sorted(
                {k: v for k, v in featauc.items() if v is not None}.items(),
                key=lambda kv: -kv[1])[:5]),
            "deployed_candidate": deployed, "deployed_regime": dep.get("regime"),
            "deployed_mean_B": dep.get("mean_B"), "deployed_harm_AUC": dep.get("harm_AUC"),
            "deployed_detect": dep.get("detect"),
            "mixed_and_detectable": bool(reg == "mixed" and det == "detectable"),
            "deployed_mixed_and_detectable": bool(dep.get("regime") == "mixed" and dep.get("detect") == "detectable"),
        }
    any_mixed_det = any(d["mixed_and_detectable"] for d in per_dom.values())
    any_dep_mixed_det = any(d["deployed_mixed_and_detectable"] for d in per_dom.values())
    grad = grad_detectability(src, tgt, names)
    return {"deployed_adapter_source_chosen": deployed, "source_mean_acc_by_candidate": src_mean,
            "per_domain": per_dom, "per_domain_candidate": per_dc,
            "gradient_TTA_detectability_DECISIVE": grad,
            "ANY_domain_allcand_mixed_detectable": any_mixed_det,
            "ANY_domain_deployed_mixed_detectable": any_dep_mixed_det,
            "ANY_domain_gradTTA_mixed_detectable": grad["ANY_domain_gradTTA_mixed_detectable"],
            "win_precondition_met": bool(grad["ANY_domain_gradTTA_mixed_detectable"])}


# ----------------------------- verdict (route a/b) -----------------------------
def policy(dec, a0, aa, B):
    a0 = np.asarray(a0, float); aa = np.asarray(aa, float); B = np.asarray(B, float); dec = np.asarray(dec)
    adapt = dec == "ADAPT"; realized = np.where(adapt, aa, a0); oracle = np.maximum(a0, aa)
    rk = float((oracle - realized).mean()); ra = float((oracle - aa).mean()); rf = float((oracle - a0).mean())
    return {"n": int(len(a0)), "decision_counts": {d: int((dec == d).sum()) for d in ["ADAPT", "FREEZE", "ABSTAIN"]},
            "regret_vs_oracle": {"always_adapt": ra, "always_freeze": rf, "K_Bound": rk},
            "false_adapt_count": int(np.sum(adapt & (B < 0))), "adapt_count": int(adapt.sum()),
            "beats_both": bool(rk < ra - 1e-9 and rk < rf - 1e-9)}


def route_a(src, tgt, alpha=ALPHA):
    out = {}
    for c in sorted(set(r["candidate"] for r in tgt)):
        rs = [r for r in src if r["candidate"] == c]; rt = [r for r in tgt if r["candidate"] == c]
        if len(rs) < 4 or len(rt) < 2:
            out[c] = {"note": f"insufficient cells src={len(rs)} tgt={len(rt)}"}; continue
        Zs = np.array([r["Z"] for r in rs]); Bs = np.array([r["B"] for r in rs])
        Zt = np.array([r["Z"] for r in rt]); Bt = np.array([r["B"] for r in rt])
        a0t = np.array([r["a0"] for r in rt]); aat = np.array([r["aa"] for r in rt])
        model, eps = fit_certificate(Zs, Bs, alpha)
        Bhat = model.predict(Zt)
        dec = np.where(Bhat - eps > 0, "ADAPT", np.where(Bhat + eps < 0, "FREEZE", "ABSTAIN"))
        pm = policy(dec, a0t, aat, Bt)
        pm["eps_conformal_source"] = float(eps); pm["src_mean_B"] = float(Bs.mean())
        pm["tgt_mean_B"] = float(Bt.mean()); pm["tgt_base_rate_harmful"] = float(np.mean(Bt < 0))
        harm = (Bt < 0).astype(int)
        pm["certificate_harm_AUC"] = _auc(-Bhat, harm) if harm.sum() not in (0, len(harm)) else None
        out[c] = pm
    return out


def recompute_committed(b_hat, margin):
    b = np.asarray(b_hat, float); M = len(b); adv = b[1:] - b[0]
    committed = [i + 1 for i in range(M - 1) if adv[i] > margin and b[i + 1] > 0]
    return (int(max(committed, key=lambda i: b[i])) if committed else None)


def route_b_decisions(conds, tau_star):
    decs, choices = [], []
    for c in conds:
        rt = c.get("route", {}); tau = rt.get("tau"); b_hat = rt.get("b_hat"); margin = rt.get("margin")
        if b_hat is None or margin is None:
            decs.append(rt.get("decision") if rt.get("decision") in ("FREEZE", "ABSTAIN") else "ABSTAIN")
            choices.append(None); continue
        if tau is None or tau > tau_star:
            decs.append("ABSTAIN"); choices.append(None); continue
        ch = recompute_committed(b_hat, margin)
        decs.append("ADAPT" if ch is not None else "FREEZE"); choices.append(ch)
    return decs, choices


def route_b_score(conds, decs, choices):
    a0 = np.array([c["a0"] for c in conds]); aa_all = [c["aa_all"] for c in conds]
    oracle = np.array([max(a) for a in aa_all]); aa_mat = np.array([a[1:] for a in aa_all])
    fixed_idx = int(np.argmax(aa_mat.mean(0))); fixed_aa = aa_mat[:, fixed_idx]
    realized = np.array([aa_all[i][choices[i]] if (decs[i] == "ADAPT" and choices[i] is not None) else a0[i]
                         for i in range(len(conds))])
    rk = float((oracle - realized).mean()); rf = float((oracle - a0).mean()); rfa = float((oracle - fixed_aa).mean())
    adapt = np.array([d == "ADAPT" for d in decs])
    return {"n_conditions": len(conds),
            "decision_counts": {d: int(sum(1 for x in decs if x == d)) for d in ["ADAPT", "FREEZE", "ABSTAIN"]},
            "fixed_best_always_adapt_idx": fixed_idx,
            "regret_vs_oracle": {"router": rk, "always_freeze": rf, "best_fixed_always_adapt": rfa},
            "false_adapt_count": int(np.sum(adapt & (realized < a0 - 1e-9))),
            "beats_both": bool(rk < rf - 1e-9 and rk < rfa - 1e-9)}


def route_b(src_conds, tgt_conds, alpha=ALPHA):
    src_tau = np.array([c["route"].get("tau") for c in src_conds if c.get("route", {}).get("tau") is not None], float)
    if len(src_tau) == 0:
        return {"note": "no source tau"}
    tau_star = float(np.quantile(src_tau, 1 - alpha))
    decs, choices = route_b_decisions(tgt_conds, tau_star)
    res = route_b_score(tgt_conds, decs, choices)
    res["tau_star_source_calibrated"] = tau_star
    res["source_tau_summary"] = {"n": int(len(src_tau)), "median": float(np.median(src_tau)),
                                 "min": float(src_tau.min()), "max": float(src_tau.max())}
    sweep = {}
    for q in (0.5, 0.75, 0.9, 0.95):
        ts = float(np.quantile(src_tau, q)); d2, c2 = route_b_decisions(tgt_conds, ts)
        sweep[f"src_q{q}"] = {"tau_star": ts, **{k: route_b_score(tgt_conds, d2, c2)[k] for k in ("regret_vs_oracle", "beats_both")}}
    res["tau_star_sensitivity"] = sweep
    return res


def detectability(tgt, src, names, alpha=ALPHA):
    Zt = np.array([r["Z"] for r in tgt]); Bt = np.array([r["B"] for r in tgt]); harm = (Bt < 0).astype(int)
    out = {"n_cells": len(tgt), "base_rate_harmful": float(harm.mean()), "mean_B": float(Bt.mean())}
    if harm.sum() in (0, len(harm)):
        out["detectability_verdict"] = "n/a"; out["best_single_feature_harm_AUC"] = None; return out
    hauc, featauc = best_feature_auc(Zt, harm, names)
    out["per_feature_harm_AUC"] = featauc; out["best_single_feature_harm_AUC"] = hauc
    Zs = np.array([r["Z"] for r in src]); Bs = np.array([r["B"] for r in src])
    if len(Bs) >= 4:
        model, eps = fit_certificate(Zs, Bs, alpha)
        out["certificate_harm_AUC_sourcefit_on_target"] = _auc(-model.predict(Zt), harm)
    out["detectability_verdict"] = "detectable" if (hauc or 0) >= DET_AUC else ("weak" if (hauc or 0) >= 0.6 else "undetectable")
    return out


def verdict(src, tgt, sc, tc, names, alpha=ALPHA):
    cand_src = {c: float(np.mean([r["aa"] for r in src if r["candidate"] == c]))
                for c in sorted(set(r["candidate"] for r in src))}
    deployed = max(cand_src, key=cand_src.get) if cand_src else None
    ra = route_a(src, tgt, alpha); rb = route_b(sc, tc, alpha); det = detectability(tgt, src, names, alpha)
    Bt = np.array([r["B"] for r in tgt]); base_h = float(np.mean(Bt < 0)); meanB = float(Bt.mean())
    frac_help = float(np.mean(Bt > THR)); detv = det.get("detectability_verdict")
    if base_h < 0.15 and meanB > 0:
        regime = "helpful-dominated"
    elif base_h > 0.85:
        regime = "harmful-dominated+" + ("detectable" if detv == "detectable" else "undetectable")
    else:
        regime = "mixed+" + ("detectable" if detv == "detectable" else "undetectable")
    goldilocks = bool(frac_help >= 0.05 and base_h >= 0.15 and detv == "detectable")
    dep = ra.get(deployed, {})
    bb_a = bool(dep.get("beats_both", False)); bb_b = bool(rb.get("beats_both", False))
    return {"deployed_adapter_source_chosen": deployed, "source_mean_acc_by_candidate": cand_src,
            "target_benefit": {"mean_B": meanB, "base_rate_harmful": base_h, "frac_helpful": frac_help,
                               "B_range": [float(Bt.min()), float(Bt.max())]},
            "detectability_target": det, "route_a_deployed": dep, "route_a_all": ra,
            "route_b_multicandidate": rb, "regime": regime, "goldilocks": goldilocks,
            "beats_both_route_a_deployed": bb_a, "beats_both_route_b_multicand": bb_b,
            "beats_both": bool((bb_a or bb_b) and goldilocks)}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["regime-scan", "verdict"], required=True)
    p.add_argument("--source", required=True)
    p.add_argument("--target", required=True)
    p.add_argument("--out", default="")
    p.add_argument("--alpha", type=float, default=ALPHA)
    args = p.parse_args()
    so, sr, scd, names = load(args.source)
    to, tr, tcd, names_t = load(args.target)
    names = names or names_t
    src = recompute(sr, scd); tgt = recompute(tr, tcd)
    if args.mode == "regime-scan":
        res = {"schema": "officehome_regime_scan_v1", "n_source": len(src), "n_target": len(tgt),
               "evidence_names": names, **regime_scan(src, tgt, names)}
        out = args.out or str(Path(args.target).parent / "REGIME_SCAN.json")
    else:
        res = {"schema": "officehome_verdict_sourcecal_v1", "alpha": args.alpha,
               "n_source": len(src), "n_target": len(tgt),
               **verdict(src, tgt, scd, tcd, names, args.alpha)}
        out = args.out or str(Path(args.target).parent / "VERDICT.json")
    json.dump(res, open(out, "w"), indent=2, default=float)
    print(json.dumps(res, indent=2, default=float)[:4000])
    print(f"\n-> {out}")


if __name__ == "__main__":
    main()
