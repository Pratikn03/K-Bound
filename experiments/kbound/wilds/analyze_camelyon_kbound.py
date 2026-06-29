"""
analyze_camelyon_kbound.py — source-calibrated route audit for WILDS Camelyon17.

Re-scores stored Protocol F (or debug) manifests WITHOUT GPU:
  route-(a) single-candidate KGA with SOURCE-fit certificate applied to TARGET
  route-(b) multicandidate with SOURCE-calibrated tau* (fixes frozen synthetic tau*=0.52)

SOURCE = id_val (in-distribution hospitals).  TARGET = test (OOD held-out).
This is the leakage-free multicandidate fix; headline policy wins remain Protocol G/H.
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import KFold

ALPHA = 0.10
THR = 0.02
GBR_KW = dict(n_estimators=250, max_depth=2, learning_rate=0.05, subsample=0.8)


def load_manifest(path):
    obj = json.load(open(path))
    recs = obj.get("records", [])
    conds = obj.get("conditions", [])
    return obj, recs, conds


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
    Z = np.asarray(Z, float); B = np.asarray(B, float); n = len(B)
    res = np.zeros(n)
    if n >= 4:
        kf = KFold(n_splits=min(5, n), shuffle=True, random_state=seed)
        for tr, te in kf.split(Z):
            m = GradientBoostingRegressor(**GBR_KW, random_state=seed).fit(Z[tr], B[tr])
            res[te] = np.abs(m.predict(Z[te]) - B[te])
        eps = float(np.quantile(res, 1 - alpha))
    else:
        eps = float(np.std(B) + 1e-6)
    full = GradientBoostingRegressor(**GBR_KW, random_state=seed).fit(Z, B)
    return full, eps


def policy(dec, a0, aa, B, alpha=ALPHA):
    a0 = np.asarray(a0, float); aa = np.asarray(aa, float); B = np.asarray(B, float)
    dec = np.asarray(dec)
    adapt = dec == "ADAPT"
    realized = np.where(adapt, aa, a0)
    oracle = np.maximum(a0, aa)
    rk = float((oracle - realized).mean()); ra = float((oracle - aa).mean()); rf = float((oracle - a0).mean())
    fa = float(np.mean(B[adapt] < 0)) if adapt.any() else None
    return {
        "n": int(len(a0)),
        "decision_counts": {d: int((dec == d).sum()) for d in ["ADAPT", "FREEZE", "ABSTAIN"]},
        "regret_vs_oracle": {"always_adapt": ra, "always_freeze": rf, "K_Bound": rk},
        "false_adapt_count": int(np.sum(adapt & (B < 0))),
        "false_adapt_rate_among_adapt": fa,
        "beats_both": bool(rk < ra - 1e-9 and rk < rf - 1e-9 and (fa is None or fa <= alpha)),
    }


def route_a(src, tgt, alpha=ALPHA):
    out = {}
    for c in sorted(set(r["candidate"] for r in tgt)):
        rs = [r for r in src if r["candidate"] == c]
        rt = [r for r in tgt if r["candidate"] == c]
        if len(rs) < 4 or len(rt) < 2:
            out[c] = {"note": f"insufficient src={len(rs)} tgt={len(rt)}"}
            continue
        Zs = np.array([r["Z"] for r in rs]); Bs = np.array([r["B"] for r in rs])
        Zt = np.array([r["Z"] for r in rt]); Bt = np.array([r["B"] for r in rt])
        a0t = np.array([r["a0"] for r in rt]); aat = np.array([r["aa"] for r in rt])
        model, eps = fit_certificate(Zs, Bs, alpha)
        Bhat = model.predict(Zt)
        dec = np.where(Bhat - eps > 0, "ADAPT", np.where(Bhat + eps < 0, "FREEZE", "ABSTAIN"))
        pm = policy(dec, a0t, aat, Bt, alpha)
        pm["eps_conformal_source"] = float(eps)
        harm = (Bt < 0).astype(int)
        pm["certificate_harm_AUC"] = _auc(-Bhat, harm) if harm.sum() not in (0, len(harm)) else None
        out[c] = pm
    return out


def recompute_committed(b_hat, margin):
    b = np.asarray(b_hat, float)
    adv = b[1:] - b[0]
    committed = [i + 1 for i in range(len(b) - 1) if adv[i] > margin and b[i + 1] > 0]
    choice = int(max(committed, key=lambda i: b[i])) if committed else None
    return committed, choice


def route_b_decisions(conds, tau_star):
    decs, choices = [], []
    for c in conds:
        rt = c.get("route", {})
        tau = rt.get("tau"); b_hat = rt.get("b_hat"); margin = rt.get("margin")
        if b_hat is None or margin is None:
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


def route_b_score(conds, decs, choices, alpha=ALPHA):
    a0 = np.array([c["a0"] for c in conds])
    aa_all = [c["aa_all"] for c in conds]
    oracle = np.array([max(a) for a in aa_all])
    aa_mat = np.array([a[1:] for a in aa_all])
    fixed_idx = int(np.argmax(aa_mat.mean(0)))
    fixed_aa = aa_mat[:, fixed_idx]
    realized = np.array([aa_all[i][choices[i]] if (decs[i] == "ADAPT" and choices[i] is not None)
                         else a0[i] for i in range(len(conds))])
    rk = float((oracle - realized).mean()); rf = float((oracle - a0).mean()); rfa = float((oracle - fixed_aa).mean())
    adapt = np.array([d == "ADAPT" for d in decs])
    fa = float(np.mean(realized[adapt] < a0[adapt] - 1e-9)) if adapt.any() else None
    return {
        "n_conditions": len(conds),
        "decision_counts": {d: int(sum(1 for x in decs if x == d)) for d in ["ADAPT", "FREEZE", "ABSTAIN"]},
        "regret_vs_oracle": {"router": rk, "always_freeze": rf, "best_fixed_always_adapt": rfa},
        "false_adapt_count": int(np.sum(adapt & (realized < a0 - 1e-9))),
        "false_adapt_rate_among_adapt": fa,
        "beats_both": bool(rk < rf - 1e-9 and rk < rfa - 1e-9 and (fa is None or fa <= alpha)),
    }


def route_b(src_conds, tgt_conds, tau_star=None, alpha=ALPHA):
    if tau_star is None:
        src_tau = np.array([c["route"].get("tau") for c in src_conds
                            if c.get("route", {}).get("tau") is not None], float)
        if len(src_tau) == 0:
            return {"note": "no source tau"}
        tau_star = float(np.quantile(src_tau, 1 - alpha))
        rule = f"(1-alpha) quantile of SOURCE tau"
    else:
        rule = f"frozen tau_star={tau_star}"
    decs, choices = route_b_decisions(tgt_conds, tau_star)
    res = route_b_score(tgt_conds, decs, choices, alpha)
    res["tau_star"] = float(tau_star)
    res["tau_star_rule"] = rule
    return res


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", required=True)
    p.add_argument("--source-domain", default="id_val")
    p.add_argument("--target-domains", nargs="+", default=["test"])
    p.add_argument("--alpha", type=float, default=ALPHA)
    p.add_argument("--out", default="")
    args = p.parse_args()

    obj, recs, conds = load_manifest(args.manifest)
    src_recs = [r for r in recs if r["domain"] == args.source_domain]
    tgt_recs = [r for r in recs if r["domain"] in args.target_domains]
    src_conds = [c for c in conds if c["domain"] == args.source_domain]
    tgt_conds = [c for c in conds if c["domain"] in args.target_domains]

    cand_src = {c: float(np.mean([r["aa"] for r in src_recs if r["candidate"] == c]))
                for c in sorted(set(r["candidate"] for r in src_recs))}
    deployed = max(cand_src, key=cand_src.get) if cand_src else None

    ra = route_a(src_recs, tgt_recs, args.alpha)
    rb_frozen = route_b(src_conds, tgt_conds, tau_star=0.52, alpha=args.alpha)
    rb_src = route_b(src_conds, tgt_conds, tau_star=None, alpha=args.alpha)

    Bt = np.array([r["B"] for r in tgt_recs])
    verdict = {
        "schema": "kbound_camelyon_sourcecalibrated_verdict_v1",
        "manifest": str(args.manifest),
        "source_domain": args.source_domain,
        "target_domains": args.target_domains,
        "alpha": args.alpha,
        "n_source_records": len(src_recs),
        "n_target_records": len(tgt_recs),
        "deployed_adapter_source_chosen": deployed,
        "source_mean_acc_by_candidate": cand_src,
        "target_benefit": {
            "mean_B": float(Bt.mean()),
            "base_rate_harmful": float(np.mean(Bt < 0)),
        },
        "route_a_single_candidate_sourcecal": ra,
        "route_b_multicandidate_frozen_tau052": rb_frozen,
        "route_b_multicandidate_source_calibrated": rb_src,
        "embedded_frozen_tau_from_gpu": obj.get("routing_b_multicandidate"),
    }
    dep = ra.get(deployed, {})
    verdict["beats_both_route_a_deployed"] = bool(dep.get("beats_both", False))
    verdict["beats_both_route_b_frozen"] = bool(rb_frozen.get("beats_both", False))
    verdict["beats_both_route_b_sourcecal"] = bool(rb_src.get("beats_both", False))

    out = args.out or str(Path(args.manifest).parent / "VERDICT_sourcecal.json")
    json.dump(verdict, open(out, "w"), indent=2)
    print(f"deployed={deployed} target_harm={verdict['target_benefit']['base_rate_harmful']:.3f}")
    if dep.get("regret_vs_oracle"):
        print(f"route_a[{deployed}] KGA={dep['regret_vs_oracle']['K_Bound']:.4f} "
              f"adapt={dep['regret_vs_oracle']['always_adapt']:.4f} "
              f"freeze={dep['regret_vs_oracle']['always_freeze']:.4f} beats={dep.get('beats_both')}")
    print(f"route_b frozen tau=0.52 beats={rb_frozen.get('beats_both')} "
          f"decisions={rb_frozen.get('decision_counts')}")
    print(f"route_b source-cal tau*={rb_src.get('tau_star'):.4f} beats={rb_src.get('beats_both')} "
          f"decisions={rb_src.get('decision_counts')}")
    print(f"wrote {out}")
    return out


if __name__ == "__main__":
    main()
