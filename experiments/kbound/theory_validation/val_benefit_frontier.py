"""val_benefit_frontier.py -- validates Lemma + Thm(frontier i/ii/iii) + Thm(minimax)
 + Cor(irreducible) + Prop(beta=0), each with EXPLICIT synthetic distributions on the
 disagreement region D (a finite set of n points with observable scores s_i and the
 unobservable target correctness eta_i = P_T(Y=f_a|x_i)). mu|D uniform.

   M     = mean(s) - 1/2          (observable)
   gamma = mean(eta - s)          (unobservable)
   Delta ∝ mean(eta) - 1/2 = M + gamma   (Lemma)
"""
import numpy as np, json, os
rng = np.random.default_rng(7)
OUT = {}

def signDelta(eta): return int(np.sign(eta.mean() - 0.5))

# ---- Lemma + Thm(i) scalar sufficiency: same (M,gamma) => same signDelta ----
viol = 0
for _ in range(20000):
    n = rng.integers(4, 200)
    s = rng.uniform(0, 1, n)
    eta = np.clip(s + rng.uniform(-0.45, 0.45), 0, 1)
    M = s.mean() - 0.5; g = (eta - s).mean()
    # a DIFFERENT eta' with the same mean (hence same gamma given same s): permute residual
    eta2 = np.clip(s + (eta - s).mean() + rng.normal(0, 0.05, n), 0, 1)
    # force exact same gamma by shifting to match mean
    eta2 = eta2 + (eta.mean() - eta2.mean())
    if abs((M + g) - (eta.mean() - 0.5)) > 1e-12: viol += 1
    if signDelta(eta) != np.sign(M + g) and (eta.mean() != 0.5): viol += 1
    if abs(eta2.mean() - eta.mean()) < 1e-9 and signDelta(eta2) != signDelta(eta): viol += 1
OUT["lemma_sufficiency_violations"] = int(viol)
print(f"[Lemma+i] reduction & scalar-sufficiency: {20000-viol}/20000 ok -> {'PASS' if viol==0 else 'FAIL'}")

# ---- Thm(ii)+(Thm minimax): explicit two-point at |M|<=beta, identical observables ----
# Build P+ and P- sharing the SAME observable scores s (=> same M, same mu, same everything
# observable) but eta differing to realize gamma=+beta and gamma=-beta -> opposite Delta.
def two_point(M, beta, n=400):
    s = np.clip(0.5 + M + rng.normal(0, 0.02, n), 0, 1)   # mean(s)-1/2 ~ M (observable)
    s = s + (0.5 + M - s.mean())                          # force mean(s)=1/2+M exactly
    s = np.clip(s, 1e-6, 1-1e-6)
    etaP = np.clip(s + beta, 0, 1); etaP = etaP + (s.mean() + beta - etaP.mean())  # gamma=+beta
    etaM = np.clip(s - beta, 0, 1); etaM = etaM + (s.mean() - beta - etaM.mean())  # gamma=-beta
    return s, etaP, etaM

beta = 0.15
mmx = {}
for M in [-0.10, 0.0, 0.10, 0.20, 0.30]:
    s, etaP, etaM = two_point(M, beta)
    same_obs = abs((s.mean()-0.5) - M) < 1e-9            # identical observable margin
    dP, dM = signDelta(etaP), signDelta(etaM)
    if abs(M) <= beta:
        # unknowable: opposite signs achievable with identical observables -> minimax err 1/2
        ok = same_obs and (dP >= 0) and (dM <= 0) and (dP != dM)
        mmx[f"M={M}"] = dict(regime="unknowable", signP=dP, signM=dM, identical_observables=bool(same_obs),
                             minimax_error=0.5, pass_=bool(ok))
    else:
        # knowable: both forced to sign(M)
        ok = same_obs and (dP == np.sign(M)) and (dM == np.sign(M))
        mmx[f"M={M}"] = dict(regime="knowable", signP=dP, signM=dM, forced_sign=int(np.sign(M)),
                             minimax_error=0.0, pass_=bool(ok))
    print(f"[minimax] beta={beta} M={M:+.2f} [{mmx[f'M={M}']['regime']}] "
          f"signs(P+,P-)=({dP:+d},{dM:+d}) identical-obs={same_obs} -> {'PASS' if ok else 'FAIL'}")
OUT["minimax_two_point"] = mmx

# ---- Cor(irreducible): gamma is NOT a function of the observables ------------
s, etaP, etaM = two_point(0.0, 0.2)
gP, gM = (etaP - s).mean(), (etaM - s).mean()
irre = abs((s.mean()-0.5)) < 1e-9 and abs(gP - gM) > 0.1   # same observables, different gamma
OUT["irreducible"] = dict(M=float(s.mean()-0.5), gamma_P=float(gP), gamma_M=float(gM), pass_=bool(irre))
print(f"[Cor] identical observables (M={s.mean()-0.5:+.3f}) but gamma={gP:+.3f} vs {gM:+.3f} "
      f"-> gamma uncheckable: {'PASS' if irre else 'FAIL'}")

# ---- Thm(iii) necessity: signDelta constant on a class  <=>  gamma one-sided about -M
nec_ok = True
for _ in range(20000):
    M = rng.uniform(-0.4, 0.4)
    gam = rng.uniform(-0.6, 0.6, rng.integers(2, 8))      # a class: fixed M, several gammas
    constant = len(set(np.sign(M + gam))) == 1            # signDelta constant on the class?
    one_sided = (np.all(gam >= -M)) or (np.all(gam <= -M))  # all gamma on one side of -M
    if constant != one_sided: nec_ok = False
OUT["necessity_one_sided"] = bool(nec_ok)
print(f"[Thm iii] constant signDelta  <=>  gamma one-sided about -M : {'PASS' if nec_ok else 'FAIL'}")

# ---- Prop(beta=0): ATC=sign(M) error == frontier event, vanishes as drift->0 ----
N = 400000; prop = []
for sd in [0.0, 0.05, 0.1, 0.2]:
    M = rng.uniform(-0.5, 0.5, N); G = rng.normal(0, sd, N)
    err = float(np.mean(np.sign(M) != np.sign(M + G)))
    ev = float(np.mean((np.abs(G) > np.abs(M)) & (np.sign(G) != np.sign(M))))
    prop.append(dict(drift_sd=sd, atc_error=err, event=ev, match=abs(err-ev) < 1e-9))
    print(f"[Prop] drift={sd}: ATC err={err:.4f} == event={ev:.4f} -> {abs(err-ev)<1e-9}")
OUT["prop_beta0"] = prop

allp = (OUT["lemma_sufficiency_violations"] == 0
        and all(v["pass_"] for v in mmx.values()) and OUT["irreducible"]["pass_"]
        and nec_ok and all(p["match"] for p in prop))
OUT["ALL_PASS"] = bool(allp)
open(os.path.join(os.path.dirname(__file__), "results_benefit_frontier.json"), "w").write(json.dumps(OUT, indent=2))
print("\nALL_PASS:", allp)
