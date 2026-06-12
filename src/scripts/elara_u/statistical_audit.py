"""Statistical audit: Holm-Bonferroni correction over the primary endpoints and the
consolidated reliability ablation (4 deployment regimes + the stack gate).

p-values are derived from the verified paired-bootstrap 95% CIs via the standard
Altman--Bland CI->p transform (SE = (hi-lo)/(2*1.96); z = mean/SE; two-sided normal
p), then Holm-adjusted within each pre-specified family. This needs no re-running of
the bootstraps and is fully reproducible from the committed result files.

Family A (primary positive claims): stack>auto-select, stack>best-fixed, auto>fixed.
Family B (reliability ablation): i.i.d., uniform shift, heterogeneous missingness,
natural shift (D22), and the stack reliability gate. The tested effect is "reliability
helps" (Delta>0); a Holm-significant Delta<0 means reliability significantly HURTS.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy.stats import norm

ROOT = Path(__file__).resolve().parents[3]
EXP = ROOT / "experiments/elara_u"
OUT = EXP / "statistical_audit.json"


def load(name):
    return json.loads((EXP / name).read_text())


def p_from_ci(mean, lo, hi):
    """Altman--Bland: two-sided p from a 95% CI."""
    se = (hi - lo) / (2 * 1.959964)
    if se <= 0:
        return 0.0 if mean != 0 else 1.0
    z = abs(mean) / se
    return float(2 * norm.sf(z))


def holm(entries, alpha=0.05):
    """entries: list of dicts with 'p'. Adds 'p_holm' and 'reject' (Holm step-down)."""
    order = sorted(range(len(entries)), key=lambda i: entries[i]["p"])
    m = len(entries)
    running = 0.0
    for rank, i in enumerate(order):
        adj = min(1.0, (m - rank) * entries[i]["p"])
        running = max(running, adj)            # enforce monotonicity
        entries[i]["p_holm"] = running
        entries[i]["reject"] = running < alpha
    return entries


def main():
    H = load("honest_benchmark.json")["contrasts"]
    lr = load("learned_router_ablation.json")["decisive_contrasts"]
    ss = load("shift_stress_ablation.json")["severity_3.0"]["contrasts"]
    hg = load("heterogeneous_degradation_ablation.json")["missing_0.7"]["contrasts"]
    ns = load("natural_shift_results.json")

    def E(label, c):
        m, lo, hi = c["mean"], c["ci95"][0], c["ci95"][1]
        return {"label": label, "delta": m, "ci95": [lo, hi], "p": p_from_ci(m, lo, hi)}

    # Family A: primary positive claims
    famA = holm([
        E("Stack > auto-select", H["stack_vs_auto_select"]),
        E("Stack > best fixed", H["stack_vs_best_fixed"]),
        E("Auto-select > best fixed", H["auto_select_vs_best_fixed"]),
    ])

    # Family B: consolidated reliability ablation (4 single-input regimes + gate + D23 multimodal)
    famB_raw = [
        E("i.i.d. (learned router)", lr["rel_vs_norel_ABLATION"]),
        E("uniform shift (sev. 3)", ss["rel_vs_norel_ABLATION"]),
        E("heterogeneous missingness (0.7)", hg["rel_vs_norel_ABLATION"]),
        E("natural shift D22 (drift vs plain)", ns["drift_stack_vs_plain_stack"]),
        E("stack reliability gate", H["stack_rel_vs_stack_ABLATION"]),
    ]
    for fname, label in [("multimodal_reliability_results.json", "multimodal D23 Real-IAD-D3 (gate vs val-only)"),
                         ("multimodal_reliability_results_mvtec3d.json", "multimodal D23 MVTec-3D (gate vs val-only)"),
                         ("multimodal_reliability_results_3d_adam.json", "multimodal D23 3D-ADAM (gate vs val-only)"),
                         ("multimodal_reliability_results_mulsen.json", "multimodal D23 MulSen-AD (gate vs val-only)")]:
        if not (EXP / fname).exists():
            continue
        d = load(fname)
        # tolerate both schema keys (script was co-edited): old + new hypotheses key
        hyp = d.get("hypotheses_failure_regime") or d.get("hypotheses_modality_failure_regime")
        if hyp:
            famB_raw.append(E(label, hyp["H3_vs_no_reliability"]))
    famB = holm(famB_raw)
    for e in famB:
        e["reliability_helps"] = bool(e["delta"] > 0 and e["reject"])
        e["reliability_hurts"] = bool(e["delta"] < 0 and e["reject"])

    result = {
        "protocol": "ELARA_U_STATISTICAL_AUDIT_v1 (Holm-Bonferroni, CI->p Altman-Bland)",
        "family_A_primary_positive": famA,
        "family_B_reliability_ablation": famB,
        "summary": {
            "all_primary_claims_hold_after_holm": all(e["reject"] for e in famA),
            "any_regime_reliability_helps_after_holm": any(e["reliability_helps"] for e in famB),
            "n_regimes_reliability_hurts_after_holm": sum(e["reliability_hurts"] for e in famB),
        },
    }
    OUT.write_text(json.dumps(result, indent=2))

    print("=== Family A: primary positive claims (Holm) ===")
    for e in famA:
        print(f"  {e['label']:26} d={e['delta']:+.3f} p={e['p']:.2e} p_holm={e['p_holm']:.2e} "
              f"{'REJECT(sig)' if e['reject'] else 'ns'}")
    print("\n=== Family B: consolidated reliability ablation (Holm) ===")
    for e in famB:
        tag = "HELPS" if e["reliability_helps"] else ("HURTS" if e["reliability_hurts"] else "ns")
        print(f"  {e['label']:34} d={e['delta']:+.3f} CI{[round(x,3) for x in e['ci95']]} "
              f"p_holm={e['p_holm']:.2e} -> reliability {tag}")
    s = result["summary"]
    print(f"\nprimary claims hold after Holm: {s['all_primary_claims_hold_after_holm']}")
    print(f"reliability helps in any regime after Holm: {s['any_regime_reliability_helps_after_holm']}")
    print(f"regimes where reliability significantly HURTS after Holm: {s['n_regimes_reliability_hurts_after_holm']}")
    print(f"wrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
