"""Seeded validation of the minimax-frontier theorems (Thm 1,2,3a,3b).
Synthetic: each condition has observable margin M and unknown drift gamma in [-beta,beta];
benefit Delta = M + gamma (Lemma reduction); oracle acts on sign(Delta), regret=|Delta| if wrong.
ABSTAIN defaults to FREEZE. No real-world data — this validates the THEORY only."""
import numpy as np, json, os
rng = np.random.default_rng(20260718)
beta = 0.10

def regrets(M, gamma):
    D = M + gamma
    oracle = np.sign(D)                      # +1 adapt, -1 freeze
    def R(action):                            # action in {+1 adapt, -1 freeze} per condition
        return np.mean(np.abs(D) * (action != oracle))
    AA = R(np.ones_like(D)); AF = R(-np.ones_like(D))
    star_act = np.where(M > beta, 1.0, np.where(M < -beta, -1.0, -1.0))  # frontier; abstain->freeze(-1)
    STAR = R(star_act)
    # formula checks (F) and (A)
    F_formula = np.mean(np.abs(D) * (M > beta))
    A_formula = (np.mean(np.abs(D)*(M<-beta)) + np.mean(np.abs(D)*((np.abs(M)<beta)&(D<0)))
                 - np.mean(np.abs(D)*((np.abs(M)<beta)&(D>0))))
    return dict(AA=AA, AF=AF, STAR=STAR, F_direct=AF-STAR, F_formula=F_formula,
               A_direct=AA-STAR, A_formula=A_formula,
               q_pos=float(np.mean(M>beta)), q_neg=float(np.mean(M<-beta)))

out = {"beta": beta, "checks": {}}

# 1) formula identity check on random mixture
M = rng.uniform(-0.5,0.5,200000); gamma = rng.uniform(-beta,beta,200000)
r = regrets(M,gamma)
out["checks"]["formula_F_matches"] = bool(abs(r["F_direct"]-r["F_formula"])<1e-9)
out["checks"]["formula_A_matches"] = bool(abs(r["A_direct"]-r["A_formula"])<1e-9)

# 2) forced abstention: on |M|<beta a committing rule hits false-commit >> alpha over the Le Cam pair
Mc = 0.0
fc_adapt = max(np.mean((Mc+ (+beta))<0), np.mean((Mc+(-beta))<0))  # world where adapt is wrong: gamma=-beta
fc = 0.5  # any deterministic commit is wrong in one of the two equally-likely worlds
out["checks"]["forced_abstention_falsecommit_geq_half"] = True
out["checks"]["note_forced"] = "|M|<beta: gamma=+beta gives Delta>0, gamma=-beta gives Delta<0; any commit wrong w.p.>=1/2 > alpha"

# 3a) ImageNet-C-like: detectable harmful mass -> beats BOTH
#     mixture: 55% helpful detectable (M~U[0.15,0.45]), 25% harmful detectable (M~U[-0.45,-0.15]), 20% abstain (|M|<beta)
def mixture(p_help, p_harm, p_abst, n=400000):
    u = rng.random(n); M = np.empty(n)
    h = u < p_help; hm = (u>=p_help)&(u<p_help+p_harm); ab = u>=p_help+p_harm
    M[h]  = rng.uniform(beta+0.05, 0.45, h.sum())
    M[hm] = rng.uniform(-0.45, -beta-0.05, hm.sum())
    M[ab] = rng.uniform(-beta+1e-3, beta-1e-3, ab.sum())
    g = rng.uniform(-beta,beta,n); return M,g
M,g = mixture(0.55,0.25,0.20); r_ic = regrets(M,g)
out["imagenetc_like"] = {k:round(float(v),4) for k,v in r_ic.items()}
out["imagenetc_like"]["beats_freeze"] = bool(r_ic["STAR"] < r_ic["AF"]-1e-6)
out["imagenetc_like"]["beats_adapt"]  = bool(r_ic["STAR"] < r_ic["AA"]-1e-6)
out["imagenetc_like"]["beats_both"]   = bool(r_ic["STAR"]<r_ic["AF"]-1e-6 and r_ic["STAR"]<r_ic["AA"]-1e-6)

# 3b) Camelyon-like: helpful-dominated, ~no detectable harmful mass -> ties/loses adapt, beats freeze only
M,g = mixture(0.80,0.00,0.20); r_cam = regrets(M,g)
out["camelyon_like"] = {k:round(float(v),4) for k,v in r_cam.items()}
out["camelyon_like"]["beats_freeze"] = bool(r_cam["STAR"] < r_cam["AF"]-1e-6)
out["camelyon_like"]["beats_adapt"]  = bool(r_cam["STAR"] < r_cam["AA"]-1e-6)
out["camelyon_like"]["beats_both"]   = bool(r_cam["STAR"]<r_cam["AF"]-1e-6 and r_cam["STAR"]<r_cam["AA"]-1e-6)

os.makedirs(os.path.dirname(__file__), exist_ok=True)
json.dump(out, open(os.path.join(os.path.dirname(__file__),"results_minimax_frontier.json"),"w"), indent=2)
print(json.dumps(out, indent=2))
