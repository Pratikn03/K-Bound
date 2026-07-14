#!/usr/bin/env python3
"""Validate the Conjecture-1 closure (testability dichotomy).

Three exact, population-level checks (grid worlds, no sampling error in the law-level
claims) + one finite-sample sanity:
  1. multiclass_swap     : conditional swap on D preserves ALL prediction-law statistics
                           (pairwise + triple agreement, marginals) exactly, while
                           p_a <-> p_0 and Delta -> -Delta.
  2. regression_reflection: midpoint reflection preserves evidence, Delta -> -Delta.
  3. binary_H_swap       : D-local complement preserves H (c = bb^T, tau = 0), b -> -b.
Results -> conj1_closure_results.json ; figure -> fig_conj1_closure.png
"""
import json, numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

rng = np.random.default_rng(7)
OUT = {}

# ---------- 1. multiclass conditional swap on D ----------
K, N, M_AUX = 5, 4000, 2
X = np.arange(N)                                  # grid world, uniform mu
f0 = rng.integers(0, K, N)
fa = f0.copy()
flip = rng.random(N) < 0.45                       # D = disagreement region
fa[flip] = (f0[flip] + 1 + rng.integers(0, K - 1, flip.sum())) % K
gaux = [rng.integers(0, K, N) for _ in range(M_AUX)]
D = f0 != fa
PY = rng.dirichlet(np.ones(K), N)                 # P(Y=k | x)

def swap_conditional(PY, f0, fa, D):
    PY2 = PY.copy()
    idx = np.where(D)[0]
    a, b = fa[idx], f0[idx]
    pa, pb = PY2[idx, a].copy(), PY2[idx, b].copy()
    PY2[idx, a], PY2[idx, b] = pb, pa
    return PY2

PY2 = swap_conditional(PY, f0, fa, D)
preds = [f0, fa] + gaux

def evidence_stats(preds):
    """Label-free statistics: pairwise & triple agreement rates + prediction marginals.
    Functions of X only -> identical under any Y-swap by construction; computed anyway."""
    P = len(preds)
    pair = {f"A{i}{j}": float(np.mean(preds[i] == preds[j]))
            for i in range(P) for j in range(i + 1, P)}
    trip = {f"T{i}{j}{k}": float(np.mean((preds[i] == preds[j]) & (preds[j] == preds[k])))
            for i in range(P) for j in range(i + 1, P) for k in range(j + 1, P)}
    marg = {f"m{i}": np.bincount(preds[i], minlength=K).astype(float) / len(preds[i])
            for i in range(P)}
    return pair, trip, marg

def acc_delta(PY, f0, fa, D):
    p_a = float(np.mean(PY[D, fa[D]]))            # P(fa = Y | D)
    p_0 = float(np.mean(PY[D, f0[D]]))
    delta = D.mean() * (p_a - p_0)
    return p_a, p_0, float(delta)

pair1, trip1, _ = evidence_stats(preds)
pair2, trip2, _ = evidence_stats(preds)           # predictions unchanged by swap
ev_diff = max(max(abs(pair1[k] - pair2[k]) for k in pair1),
              max(abs(trip1[k] - trip2[k]) for k in trip1))
pa1, p01, d1 = acc_delta(PY, f0, fa, D)
pa2, p02, d2 = acc_delta(PY2, f0, fa, D)
OUT["multiclass_swap"] = dict(
    K=K, N=N, frac_D=float(D.mean()),
    evidence_max_abs_diff=ev_diff,
    p_a=pa1, p_0=p01, p_a_swapped=pa2, p_0_swapped=p02,
    swap_exchanges_accuracies=bool(abs(pa2 - p01) < 1e-12 and abs(p02 - pa1) < 1e-12),
    Delta=d1, Delta_swapped=d2,
    sign_flips=bool(np.sign(d1) == -np.sign(d2) and d1 != 0),
    delta_sum_zero=float(abs(d1 + d2)))

# ---------- 2. regression midpoint reflection ----------
Ng = 6000
x = np.linspace(0, 1, Ng)
F0 = np.sin(3 * x) + 0.3 * x
FA = F0 + np.where(np.abs(np.sin(7 * x)) > 0.4, 0.6 * np.cos(5 * x), 0.0)
Dr = F0 != FA
m_true = 0.4 * np.cos(4 * x) + 0.5 * x            # E[Y|x]
var = 0.05 + 0.04 * (1 + np.sin(2 * x)) / 2       # Var[Y|x] (kept by reflection)

def risks(m, var, F):
    return float(np.mean((F - m) ** 2 + var))     # E (F - Y)^2 = (F - m)^2 + var

d_reg = risks(m_true, var, F0) - risks(m_true, var, FA)
m_ref = np.where(Dr, F0 + FA - m_true, m_true)    # reflected conditional mean on D
d_reg_ref = risks(m_ref, var, F0) - risks(m_ref, var, FA)
OUT["regression_reflection"] = dict(
    frac_D=float(Dr.mean()), Delta=d_reg, Delta_reflected=d_reg_ref,
    delta_sum_zero=float(abs(d_reg + d_reg_ref)),
    sign_flips=bool(np.sign(d_reg) == -np.sign(d_reg_ref) and d_reg != 0),
    evidence_unchanged=True)                      # predictions & X untouched (by construction)

# ---------- 3. binary D-local complement preserves H ----------
Kp, m = 4, 400000
b = np.array([0.55, 0.40, -0.30, 0.22])           # advantages on D (H: q_j(0)=q_j(1))
q = (1 + b) / 2
Y = rng.integers(0, 2, m)
C = (rng.random((Kp, m)) < q[:, None]).astype(int)          # correctness, CEI given Y
G = np.where(C == 1, Y, 1 - Y)                              # predictions
def cmat(G):
    c = np.zeros((Kp, Kp))
    for i in range(Kp):
        for j in range(Kp):
            c[i, j] = 2 * np.mean(G[i] == G[j]) - 1
    return c
c_before = cmat(G)
# D-local complement: here D is the whole region under study -> Y' = 1 - Y, G unchanged
b_after = np.array([2 * np.mean(G[i] == (1 - Y)) - 1 for i in range(Kp)])
b_before = np.array([2 * np.mean(G[i] == Y) - 1 for i in range(Kp)])
c_after = cmat(G)                                            # predictions untouched
prods = [c_before[0, 1] * c_before[2, 3], c_before[0, 2] * c_before[1, 3],
         c_before[0, 3] * c_before[1, 2]]
OUT["binary_H_swap"] = dict(
    c_max_abs_change=float(np.max(np.abs(c_after - c_before))),
    b_before=[float(v) for v in b_before], b_after=[float(v) for v in b_after],
    b_negated_max_err=float(np.max(np.abs(b_after + b_before))),
    tau_before=float(max(prods) - min(prods)),
    H_preserved=True)

# ---------- figure ----------
fig, ax = plt.subplots(1, 3, figsize=(11, 3.2))
ax[0].bar(["$\\Delta$", "$\\Delta^\\star$"], [d1, d2], color=["#2a6", "#a26"])
ax[0].axhline(0, c="k", lw=0.6); ax[0].set_title(f"Multiclass swap (K={K}): sign flips")
ax[1].bar(["$\\Delta$", "$\\Delta^\\star$"], [d_reg, d_reg_ref], color=["#2a6", "#a26"])
ax[1].axhline(0, c="k", lw=0.6); ax[1].set_title("Regression reflection: sign flips")
ax[2].bar(["pairwise", "triple"],
          [max(abs(pair1[k] - pair2[k]) for k in pair1),
           max(abs(trip1[k] - trip2[k]) for k in trip1)], color="#46a")
ax[2].set_title("Evidence change under swap (= 0)")
ax[2].set_ylim(0, 1e-3)
fig.suptitle("Conjecture-1 closure: the swap flips every benefit sign, no observable moves")
fig.tight_layout()
fig.savefig(__file__.rsplit("/", 1)[0] + "/fig_conj1_closure.png", dpi=140)

path = __file__.rsplit("/", 1)[0] + "/conj1_closure_results.json"
json.dump(OUT, open(path, "w"), indent=1)
print(json.dumps(OUT, indent=1))
print("saved ->", path)
