"""KGA adapt/freeze/abstain is correct and beats both trivial policies on a mixed stream."""
import numpy as np
import cifar_tent_mps_v2 as K

def _planted(n_help=60, n_harm=60, n_marg=30, seed=1):
    rng = np.random.default_rng(seed); rows = []
    def Z(g):
        if g == "helpful":  return [0.6,0.78,0.3,0.05,0.4] + rng.normal(0,0.02,5)
        if g == "harmful":  return [0.05,0.12,0.96,0.9,2.5] + rng.normal(0,0.03,5)
        return [0.6,0.7,0.5,0.2,0.8] + rng.normal(0,0.03,5)
    for g in ["helpful"]*n_help + ["harmful"]*n_harm + ["marginal"]*n_marg:
        a0 = float(np.clip(rng.normal(0.55,0.05),0.2,0.8))
        B = rng.normal(0.10,0.02) if g=="helpful" else (rng.normal(-0.40,0.05) if g=="harmful" else rng.normal(0,0.008))
        rows.append(dict(Z=list(np.asarray(Z(g),float)), a0=a0, aa=float(np.clip(a0+B,0,1)),
                         regime=K.label_regime(a0+B - a0)))
    return rows

def test_decisions_and_beats_both():
    m, _ = K.summarize(_planted(), alpha=0.10)
    assert m["false_adapt_rate_B<0"] == 0.0          # never adapts into harm
    assert m["adapt_precision_B>0"] >= 0.95
    assert m["beats_both"] is True                   # lower regret than adapt AND freeze
    r = m["regret_vs_oracle"]
    assert r["K_Bound"] < r["always_adapt"] and r["K_Bound"] < r["always_freeze"]

def test_pareto_crossover_exists():
    m, _ = K.summarize(_planted(), alpha=0.10)
    pa = m["pareto"]
    assert pa.get("p_where_KGA_beats_both") is not None
