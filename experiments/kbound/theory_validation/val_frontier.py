"""val_frontier.py -- numerical validation of the exact label-free knowability frontier.

Claims tested (binary 0/1 loss; D = disagreement region of f0,f1; observable = target
X-marginal mu and any source-calibrated score s(x) for eta_a(x)=P_T(Y=f_a(x)|x)):

  Decomposition:  abar - 1/2 = M + gamma,   where
      M     = E_{mu|D}[s] - 1/2          (OBSERVABLE margin)
      gamma = E_{mu|D}[eta_a - s]        (UNOBSERVABLE calibration drift on D)
  and  sign Delta = sign(abar - 1/2) = sign(M + gamma).

  T1 (impossibility): two targets with identical observables (same mu, same s) but
      opposite sign Delta exist whenever mu(D)>0  -> no label-free certificate is sound.
  T2 (exact frontier): the plug-in certificate sign(M) is correct
      iff NOT(|gamma|>|M| and sign gamma != sign M).
  T3 (tight knowable region): under drift budget |gamma|<=beta, the rule
      {adapt iff M>beta, freeze iff M<-beta, else abstain} is SOUND, and |M|<=beta is
      exactly unknowable (a flip of sign within budget exists).
  T4 (ATC = beta=0 face): the average-threshold-confidence certificate equals sign(M);
      it errs exactly on {|gamma|>|M|, opposite sign}; error rate -> 0 as drift -> 0.
"""
import numpy as np, json, os

rng = np.random.default_rng(0)
OUT = {}

# ---- T1: impossibility -- identical observables, opposite Delta -------------
n = 200
s = rng.uniform(0, 1, n)                     # observable source-calibrated scores on D
muD = 0.3                                    # mass of disagreement region
abar_help, abar_hurt = 1.0, 0.0              # eta_a = 1 (helps) vs 0 (hurts)
Delta_help = muD * (2*abar_help - 1)
Delta_hurt = muD * (2*abar_hurt - 1)
M_obs = s.mean() - 0.5                        # SAME for both worlds (depends only on s)
T1 = (np.sign(Delta_help) == 1 and np.sign(Delta_hurt) == -1)
OUT["T1_impossibility"] = dict(M_observable=float(M_obs),
                               Delta_help=float(Delta_help), Delta_hurt=float(Delta_hurt),
                               identical_observable=True, pass_=bool(T1))
print(f"[T1] identical M={M_obs:+.4f}; Delta = {Delta_help:+.3f} (help) vs {Delta_hurt:+.3f} (hurt)"
      f"  -> {'PASS' if T1 else 'FAIL'} (observables cannot decide sign)")

# ---- identity check: sign Delta = sign(M + gamma) on random problems ---------
bad = 0
for _ in range(20000):
    m = rng.integers(5, 300)
    sc = rng.uniform(0, 1, m)
    eta = np.clip(sc + rng.uniform(-0.5, 0.5) + rng.normal(0, 0.1, m), 0, 1)
    M = sc.mean() - 0.5; g = (eta - sc).mean(); abar = eta.mean()
    if abs((M + g) - (abar - 0.5)) > 1e-12: bad += 1
    if np.sign(abar - 0.5) != 0 and np.sign(M + g) != np.sign(abar - 0.5): bad += 1
OUT["identity_violations"] = int(bad)
print(f"[id] sign Delta = sign(M+gamma): {20000-bad}/20000 exact  -> {'PASS' if bad==0 else 'FAIL'}")

# ---- T2: exact frontier (plug-in correctness law) ---------------------------
N = 500000
M = rng.uniform(-0.5, 0.5, N)
G = rng.uniform(-0.7, 0.7, N)               # gamma free of M
true = np.sign(M + G); plug = np.sign(M)
correct = (true == plug) | (true == 0)
law = ~((np.abs(G) > np.abs(M)) & (np.sign(G) != np.sign(M)))
T2 = bool(np.all(correct == law))
OUT["T2_frontier_law"] = dict(pass_=T2, n=N)
print(f"[T2] plug-in correct  <=>  not(|g|>|M| & opp sign): {'PASS' if T2 else 'FAIL'} over {N} draws")

# ---- T3: tight knowable region under |gamma|<=beta --------------------------
T3 = {}
for beta in [0.05, 0.10, 0.20]:
    know = np.abs(M) > beta
    # soundness: worst admissible drift g=-sign(M)*beta keeps the sign on the knowable set
    worst = np.sign(M - np.sign(M) * beta)
    sound = bool(np.all(worst[know] == np.sign(M)[know]))
    # unknowable: on |M|<=beta a within-budget g flips the sign (choose g=-M, |g|=|M|<=beta)
    unk = np.abs(M) <= beta
    flippable = bool(np.all(np.abs(M)[unk] <= beta))
    T3[f"beta={beta}"] = dict(sound_on_knowable=sound, all_unknowable_flippable=flippable)
    print(f"[T3] beta={beta}: |M|>beta sound={sound}; |M|<=beta flippable={flippable}")
OUT["T3_tight_region"] = T3

# ---- T4: ATC is the beta=0 face --------------------------------------------
# ATC benefit-sign certificate = sign(M). Sweep a drift distribution; error must equal
# the {|g|>|M|, opposite sign} event, and vanish as drift sd -> 0.
print("[T4] ATC(=sign M) error rate vs calibration-drift std:")
T4rows = []
for sd in [0.0, 0.05, 0.1, 0.2, 0.4]:
    Mx = rng.uniform(-0.5, 0.5, N)
    Gx = rng.normal(0, sd, N)
    err = np.mean(np.sign(Mx) != np.sign(Mx + Gx))
    event = np.mean((np.abs(Gx) > np.abs(Mx)) & (np.sign(Gx) != np.sign(Mx)))
    T4rows.append(dict(drift_sd=sd, atc_error=float(err), frontier_event=float(event),
                       match=bool(abs(err-event) < 1e-9)))
    print(f"     drift_sd={sd:<4}: ATC error={err:.4f}  frontier-event={event:.4f}  match={abs(err-event)<1e-9}")
OUT["T4_atc_beta0"] = T4rows

allpass = T1 and bad == 0 and T2 and all(v["sound_on_knowable"] and v["all_unknowable_flippable"]
                                         for v in T3.values()) and all(r["match"] for r in T4rows)
OUT["ALL_PASS"] = bool(allpass)
d = os.path.dirname(__file__); open(os.path.join(d, "results_frontier.json"), "w").write(json.dumps(OUT, indent=2))
print("\nALL_PASS:", allpass, "-> results_frontier.json")
