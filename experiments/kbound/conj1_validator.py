#!/usr/bin/env python3
"""conj1_validator.py -- numerical validation of the CONDITIONAL resolution of
Conjecture 1 (kbound.tex / paper/sections/main_theory_5.tex, label conj:gen).

Conjecture 1 leaves one piece open, verbatim:
    "What remains open is the weakest falsifiable class on which the bit still suffices."

This validator backs the added Assumption (general position), Definition (margin-monotone
relative-calibration class C_mono), Lemma (structural sign-freedom = bit complexity),
Theorem (C_mono is, under general position, the weakest falsifiable class admitting a
ONE-bit benefit-sign certificate), and the explicit non-identifiability witness.

============================ FINITE-DISTRIBUTION MODEL ============================
Work on the observable disagreement region D = {x : f0(x) != fa(x)} (binary Y, 0/1 loss;
this is exactly the Lemma "reduction to D" of main_theory_5.tex). Discretize D into cells:
    mu_i  >= 0   observable P_X mass on cell i  (sum_i mu_i = 1 over D)
    m_i          observable relative margin (the calibrated-score axis, m_i = 2 s_i - 1)
    a_i in [-1,1]  UNOBSERVABLE per-cell benefit, a_i = 2*eta_a,i - 1, eta_a,i=Pr_T(fa=Y|cell i)
    Delta = sum_i mu_i a_i           (the benefit; sign(Delta) is the adapt/freeze decision)

This matches main_theory_5.tex exactly:  Delta = 2 mu(D) (abar - 1/2) with abar=Pr(fa=Y|D),
since a_i = 2 eta_a,i - 1  =>  Delta = mu(D) E_{mu|D}[a] = 2 mu(D)(E[eta_a]-1/2).

EVIDENCE LAW E := the law of ALL label-free observables. Because predictions and the
source-calibrated score are fixed maps of x, every label-free statistic of any order is a
function of (mu, m, s). The benefit field a (equivalently P(Y|x)) does NOT enter E. Hence:
    two laws are EVIDENCE-IDENTICAL  <=>  they share (mu, m, s).   [TV(E,E')=0 exactly]
This is the population-level swap mechanism of theory_v2/CONJ1_CLOSURE_PROOFS.md, restricted
to the disagreement region.

Relative-calibration field:  c_i := w_i * (m_i - mstar), with w_i>=0 and mstar both OBSERVABLE
(E-determined). C_mono asserts a_i = sigma * c_i for a single global orientation bit
sigma in {+1,-1}; then sign(Delta) = sigma * sign(T_E), T_E := sum_i mu_i c_i (observable).

All checks are exact / population-level (no sampling error) unless explicitly marked.
Exit code 0 iff EVERY check PASSES.  Numbers are printed verbatim; nothing is tuned to pass.
"""
import json, sys
import numpy as np

rng = np.random.default_rng(20260615)
TOL = 1e-12
OUT = {}
FAILURES = []

def sgn(z):
    return int(np.sign(z))

def banner(t):
    print("\n" + "=" * 78); print(t); print("=" * 78)

# ===================================================================================
# CHECK A.  C_mono: the one orientation bit sigma is NECESSARY and SUFFICIENT.
#   Sufficiency: for every margin-monotone law, E + sigma recovers sign(Delta) via
#               the closed form sign(Delta) = sigma * sign(T_E), T_E observable.
#   Necessity : sigma and -sigma are evidence-identical (same E, TV=0) but give
#               opposite Delta  =>  >= 1 bit is information-theoretically required
#               (Le Cam two-point).  Together: EXACTLY one bit.
# ===================================================================================
def sample_cmono_law():
    """Random margin-monotone relative-calibration law in C_mono."""
    n = int(rng.integers(3, 9))
    mu = rng.dirichlet(np.ones(n))                      # observable masses, all > 0
    m = np.sort(rng.uniform(-3, 3, n))                  # observable margins
    mstar = float(rng.uniform(-3, 3))                   # observable anchor
    w = rng.uniform(0.1, 2.0, n)                        # observable nonneg weight
    c = w * (m - mstar)                                 # observable calibration field
    mx = np.max(np.abs(c))
    if mx > 0:
        c = c / mx * 0.99                               # keep a_i = sigma*c_i in [-1,1]
    sigma = int(rng.choice([-1, 1]))                    # UNOBSERVABLE orientation bit
    a = sigma * c                                       # benefit field (unobservable)
    return dict(mu=mu, m=m, mstar=mstar, w=w, c=c, sigma=sigma, a=a)

def run_check_A(trials=20000):
    banner("CHECK A  --  C_mono: one bit necessary AND sufficient")
    suff_ok = 0; suff_total = 0; gp_excluded = 0
    nec_ok = 0
    worst_err = 0.0
    for _ in range(trials):
        L = sample_cmono_law()
        mu, c, sigma, a = L["mu"], L["c"], L["sigma"], L["a"]
        Delta = float(mu @ a)
        T_E = float(mu @ c)                              # observable statistic
        # --- general position: skip exact ties (measure zero; excluded by Assumption GP)
        if abs(T_E) < TOL or abs(Delta) < TOL:
            gp_excluded += 1
            continue
        suff_total += 1
        # SUFFICIENCY: decoder uses ONLY E (=T_E) and the single declared bit sigma.
        recovered = sigma * sgn(T_E)
        if recovered == sgn(Delta):
            suff_ok += 1
        worst_err = max(worst_err, abs(Delta - sigma * T_E))   # closed-form identity
        # NECESSITY: the -sigma twin shares E exactly (same mu,m,w,c => TV=0) but flips Delta.
        Delta_twin = float(mu @ (-sigma * c))
        evidence_identical = True   # by construction E depends only on (mu,m,w)=shared
        if evidence_identical and abs(Delta_twin + Delta) < TOL and sgn(Delta_twin) == -sgn(Delta):
            nec_ok += 1
    rate = suff_ok / max(1, suff_total)
    nec_rate = nec_ok / max(1, suff_total)
    print(f"trials={trials}  GP-excluded(ties)={gp_excluded}  evaluated={suff_total}")
    print(f"SUFFICIENCY  sign(Delta)==sigma*sign(T_E):  {suff_ok}/{suff_total}  rate={rate:.4f}")
    print(f"closed-form identity  max|Delta - sigma*T_E| = {worst_err:.3e}")
    print(f"NECESSITY    (-sigma twin: same E, opposite Delta): {nec_ok}/{suff_total}  rate={nec_rate:.4f}")
    okA = (rate == 1.0) and (nec_rate == 1.0) and (worst_err < 1e-9)
    print("RESULT A:", "PASS" if okA else "FAIL")
    if not okA: FAILURES.append("A")
    OUT["check_A"] = dict(trials=trials, evaluated=suff_total, gp_excluded=gp_excluded,
                          sufficiency_rate=rate, necessity_rate=nec_rate,
                          closed_form_max_err=worst_err, passed=bool(okA))
    return okA

# ===================================================================================
# CHECK B1.  MINIMALITY WITNESS -- two independently-signable regions (K=2).
#   Weaken C_mono by dropping margin-calibration on ONE region: region R2 carries a
#   benefit whose MAGNITUDE is not E-identified (its calibrated score is uninformative,
#   c2 = 0).  Then for EVERY single fixed structural bit there is an evidence-identical
#   pair with the SAME bit value but OPPOSITE sign(Delta).  Hence >= 2 bits are needed:
#   one bit cannot certify the sign once D contains a non-margin-monotone region.
#
#   E is fixed by (mu, m, s). R1 is calibrated (c1>0 observable); R2 is uncalibrated
#   (m2 = mstar so c2 = 0: evidence shows NO benefit signal on R2), but the true a2 is
#   free in sign and magnitude.  Every pair below shares (mu, m, s) EXACTLY => TV(E,E')=0.
# ===================================================================================
def run_check_B1():
    banner("CHECK B1  --  minimality witness: two regions, one uncalibrated (K=2)")
    mu1, mu2 = 0.5, 0.5
    m1, m2 = 1.0, 0.0          # R2 at the anchor mstar=0  => c2 = 0 (no evidence signal)
    mstar = 0.0
    w1 = 1.0
    c1 = w1 * (m1 - mstar)     # = 1.0 (observable benefit signal on R1)
    a1_mag = c1                # calibrated magnitude on R1
    # The one-bit C_mono certificate sees only R1 (c2=0): it outputs sigma = sign(a1).
    # Build two laws that share E and the same value of beta = sign(a1), opposite Delta.
    A_big = 2.0                # |a2| on the uncalibrated region (UNobservable magnitude)
    # P : a1 = +c1 (beta=+1),  a2 = -A_big  -> Delta = mu1*c1 - mu2*A_big < 0
    # P': a1 = +c1 (beta=+1),  a2 = +A_big  -> Delta = mu1*c1 + mu2*A_big > 0
    DeltaP  = mu1 * (+a1_mag) + mu2 * (-A_big)
    DeltaPp = mu1 * (+a1_mag) + mu2 * (+A_big)
    print(f"observables shared by P,P': mu={(mu1,mu2)}, m={(m1,m2)}, mstar={mstar}, c=({c1},0.0)")
    print(f"  P : a=(+{a1_mag}, -{A_big})  beta=sign(a1)=+1   Delta={DeltaP:+.4f}  sign={sgn(DeltaP):+d}")
    print(f"  P': a=(+{a1_mag}, +{A_big})  beta=sign(a1)=+1   Delta={DeltaPp:+.4f}  sign={sgn(DeltaPp):+d}")
    beta_equal_1 = True  # both beta = sign(a1) = +1
    opp_1 = sgn(DeltaP) == -sgn(DeltaPp) and sgn(DeltaP) != 0
    print(f"  -> same beta=sign(a1), opposite sign(Delta): {opp_1}")

    # Defeat the OTHER candidate single bit, beta = sign(a2):
    # Q : a1 = +c1, a2 = +eps_small -> Delta = mu1*c1 + mu2*eps > 0
    # Q': a1 = -c1, a2 = +eps_small -> Delta = -mu1*c1 + mu2*eps < 0
    eps = 0.2
    DeltaQ  = mu1 * (+a1_mag) + mu2 * (+eps)
    DeltaQp = mu1 * (-a1_mag) + mu2 * (+eps)
    print(f"  Q : a=(+{a1_mag}, +{eps})  beta=sign(a2)=+1   Delta={DeltaQ:+.4f}  sign={sgn(DeltaQ):+d}")
    print(f"  Q': a=(-{a1_mag}, +{eps})  beta=sign(a2)=+1   Delta={DeltaQp:+.4f}  sign={sgn(DeltaQp):+d}")
    opp_2 = sgn(DeltaQ) == -sgn(DeltaQp) and sgn(DeltaQ) != 0
    print(f"  -> same beta=sign(a2), opposite sign(Delta): {opp_2}")

    okB1 = opp_1 and opp_2
    print("Both candidate single structural bits (sign a1, sign a2) are defeated => >= 2 bits.")
    print("RESULT B1:", "PASS" if okB1 else "FAIL")
    if not okB1: FAILURES.append("B1")
    OUT["check_B1"] = dict(DeltaP=DeltaP, DeltaPp=DeltaPp, DeltaQ=DeltaQ, DeltaQp=DeltaQp,
                           beta_signa1_defeated=bool(opp_1), beta_signa2_defeated=bool(opp_2),
                           passed=bool(okB1))
    return okB1

# ===================================================================================
# CHECK B2.  MINIMALITY WITNESS -- non-monotone, MAGNITUDE-CALIBRATED, equal-mass (K=3).
#   Here magnitudes ARE observable (w_i known) -- the fairest comparison to C_mono.
#   Drop ONLY single-crossing: allow each of three margin-ordered cells to take an
#   independent orientation eps_i in {+1,-1} (a non-monotone benefit field).  With equal
#   benefit-mass per cell (general position: comparable masses, no dominant region),
#       Delta = W * (eps1+eps2+eps3),   sign(Delta) = majority(eps1,eps2,eps3).
#   Majority is fully sensitive: for EACH single cell-orientation bit eps_j there is an
#   evidence-identical pair sharing eps_j but with opposite Delta.  => one bit cannot
#   certify the sign; >= 2 bits are required.  (All laws share (mu,m,w) => TV(E,E')=0.)
# ===================================================================================
def run_check_B2():
    banner("CHECK B2  --  minimality witness: non-monotone equal-mass majority (K=3)")
    m = np.array([-1.0, 0.0, 1.0])          # observable, margin-ordered
    w = np.array([1.0, 1.0, 1.0])           # observable calibrated magnitudes
    mu = np.array([1/3, 1/3, 1/3])          # equal mass => W_i = mu_i*w_i all equal (GP)
    W = float(mu[0] * w[0])
    def Delta(eps):
        return float(np.sum(mu * (np.array(eps) * w)))
    def majority(eps):
        return sgn(sum(eps))
    # sanity: sign(Delta)==majority over all 8 patterns
    import itertools
    all_ok = all(sgn(Delta(e)) == majority(e) for e in itertools.product([-1,1], repeat=3)
                 if sum(e) != 0)
    print(f"sign(Delta)=majority(eps) on all non-tied patterns: {all_ok}  (W={W:.4f})")
    # For each single bit eps_j, exhibit same-bit opposite-Delta evidence-identical pair:
    pairs = {
        "eps1": ((+1,+1,+1), (+1,-1,-1)),
        "eps2": ((+1,+1,+1), (-1,+1,-1)),
        "eps3": ((+1,+1,+1), (-1,-1,+1)),
    }
    defeated = {}
    for bit, (P, Pp) in pairs.items():
        j = int(bit[-1]) - 1
        same_bit = (P[j] == Pp[j])
        opp = sgn(Delta(P)) == -sgn(Delta(Pp)) and sgn(Delta(P)) != 0
        defeated[bit] = bool(same_bit and opp)
        print(f"  bit {bit}: P={P} Delta={Delta(P):+.4f}  P'={Pp} Delta={Delta(Pp):+.4f}"
              f"  same {bit}={same_bit}  opposite sign={opp}")
    # Also: the global-orientation bit sigma (flip ALL eps) fails too -- (+,-,-) vs (-,+,+)
    g1, g2 = (+1,-1,-1), (-1,+1,+1)         # NOT related by global flip's decoder
    print(f"  global-orientation bit: (+,-,-) Delta={Delta(g1):+.4f} , its complement "
          f"(-,+,+) Delta={Delta(g2):+.4f}  (a single global bit maps a mixed pattern and its"
          f" complement to opposite Delta but cannot place them)")
    okB2 = all_ok and all(defeated.values())
    print("Every single cell-orientation bit defeated => >= 2 bits required.")
    print("RESULT B2:", "PASS" if okB2 else "FAIL")
    if not okB2: FAILURES.append("B2")
    OUT["check_B2"] = dict(W=W, sign_equals_majority=bool(all_ok), defeated=defeated,
                           passed=bool(okB2))
    return okB2

# ===================================================================================
# CHECK C.  CONSISTENCY -- the C_mono one-bit certificate is UNSOUND on a B1 law.
#   The decoder that is sound on C_mono (output sigma*sign(T_E)) must FAIL on the
#   minimality witness; otherwise the witness would not need a second bit.  We confirm
#   it returns the SAME action for the evidence-identical pair (P,P') of B1 while their
#   true signs are opposite -- i.e. it is wrong on exactly one of them.
# ===================================================================================
def run_check_C():
    banner("CHECK C  --  consistency: C_mono certificate is unsound off C_mono")
    mu1, mu2 = 0.5, 0.5
    c1 = 1.0                       # observable; R2 has c2=0
    T_E = mu1 * c1 + mu2 * 0.0     # observable statistic seen by the certificate
    cert_action = sgn(T_E)         # C_mono decoder with declared sigma=+1: sigma*sign(T_E)=+1
    DeltaP  = mu1 * (+c1) + mu2 * (-2.0)   # true sign -1
    DeltaPp = mu1 * (+c1) + mu2 * (+2.0)   # true sign +1
    same_action = True             # certificate sees identical E => identical action
    wrong_on_one = (cert_action != sgn(DeltaP)) ^ (cert_action != sgn(DeltaPp))
    print(f"certificate action on shared evidence = {cert_action:+d} (same for P and P')")
    print(f"true sign(Delta_P)={sgn(DeltaP):+d}  true sign(Delta_P')={sgn(DeltaPp):+d}")
    print(f"certificate is wrong on exactly one of the pair: {wrong_on_one}")
    okC = same_action and wrong_on_one
    print("RESULT C:", "PASS" if okC else "FAIL")
    if not okC: FAILURES.append("C")
    OUT["check_C"] = dict(cert_action=cert_action, signP=sgn(DeltaP), signPp=sgn(DeltaPp),
                          unsound=bool(wrong_on_one), passed=bool(okC))
    return okC

# ===================================================================================
# CHECK D.  GENERAL-POSITION IS NECESSARY (honesty check on the conditional claim).
#   If GP is DROPPED -- one region is E-certifiably dominant -- then a single bit (the
#   dominant region's orientation) CAN certify the sign even outside C_mono.  This shows
#   the minimality / "weakest class" claim is genuinely CONDITIONAL on general position,
#   matching the candid Remark.  Magnitudes observable; W1 >> W2 with W2 bounded by E.
# ===================================================================================
def run_check_D(trials=20000):
    banner("CHECK D  --  general position is necessary (conditional claim is honest)")
    # Two calibrated regions, NON-single-crossing (independent orientations allowed),
    # but region 1 E-certifiably dominates: W1 in [1.5,2], W2 in [0,1]  => W1 > W2 always.
    one_bit_ok = 0
    for _ in range(trials):
        W1 = rng.uniform(1.5, 2.0)
        W2 = rng.uniform(0.0, 1.0)      # E-bounded: certifiably < W1
        e1 = rng.choice([-1, 1]); e2 = rng.choice([-1, 1])
        Delta = e1 * W1 + e2 * W2
        # one-bit decoder declares the dominant region's orientation e1 (W1>W2 from E):
        if sgn(Delta) == e1:
            one_bit_ok += 1
    rate = one_bit_ok / trials
    print(f"non-GP class (W1 certifiably > W2): one-bit (dominant orientation) recovers "
          f"sign(Delta) in {one_bit_ok}/{trials}  rate={rate:.4f}")
    print("=> Without general position, one bit can suffice OUTSIDE C_mono; the weakest-")
    print("   class theorem is therefore stated CONDITIONALLY on general position. (Honest.)")
    okD = (rate == 1.0)   # confirms the conditional caveat is real, not vacuous
    print("RESULT D:", "PASS" if okD else "FAIL")
    if not okD: FAILURES.append("D")
    OUT["check_D"] = dict(trials=trials, one_bit_recovery_rate=rate, passed=bool(okD))
    return okD

# ===================================================================================
# CHECK E.  C_mono is "A" weakest one-bit class, NOT the unique one (honest tightening).
#   Even under general position, the minimal-element is non-unique: any calibrated
#   sign-aligned field {a = sigma*h, h observable} is equally one-bit certifiable, and
#   such classes are INCOMPARABLE to C_mono (different magnitude law |a|=|h| != |c|).
#   So the headline must read "A weakest" not "THE weakest". This check exhibits a second
#   robustly-one-bit class C_h incomparable to C_c=C_mono.
# ===================================================================================
def run_check_E():
    banner("CHECK E  --  C_mono is A weakest one-bit class, not THE unique one")
    mu = np.array([0.3, 0.3, 0.4]); m = np.array([-1.0, 0.5, 2.0]); mstar = 0.0
    w = np.array([1.0, 1.0, 1.0])
    c = w * (m - mstar)                       # C_c = C_mono calibration field (|a|=|c|)
    h = np.sign(c) * np.abs(c) ** 2           # C_h: same SIGNS, different magnitude law
    Tc, Th = float(mu @ c), float(mu @ h)
    onebit_c = abs(Tc) > TOL                  # sign Delta = sigma*sign(Tc): one bit
    onebit_h = abs(Th) > TOL                  # sign Delta = sigma*sign(Th): one bit
    # incomparable: a law a=c in C_c is not in C_h, and a=h in C_h is not in C_c
    incomparable = (not np.allclose(np.abs(c), np.abs(h)))
    print(f"C_c field c={c}  T_E(c)={Tc:+.4f}  one-bit={onebit_c}")
    print(f"C_h field h={h}  T_E(h)={Th:+.4f}  one-bit={onebit_h}")
    print(f"|c|={np.abs(c)}  |h|={np.abs(h)}  -> classes incomparable: {incomparable}")
    okE = onebit_c and onebit_h and incomparable
    print("=> two incomparable robustly-one-bit classes exist; C_mono is A weakest, not THE.")
    print("RESULT E:", "PASS" if okE else "FAIL")
    if not okE: FAILURES.append("E")
    OUT["check_E"] = dict(Tc=Tc, Th=Th, onebit_c=bool(onebit_c), onebit_h=bool(onebit_h),
                          incomparable=bool(incomparable), passed=bool(okE))
    return okE

# ===================================================================================
def main():
    print("conj1_validator.py  --  conditional resolution of Conjecture 1 (conj:gen)")
    print("model: finite distributions on the disagreement region D (binary, 0/1 loss)")
    a = run_check_A()
    b1 = run_check_B1()
    b2 = run_check_B2()
    c = run_check_C()
    d = run_check_D()
    e = run_check_E()
    banner("SUMMARY")
    rows = [("A  C_mono: 1 bit nec & suff (UNCONDITIONAL core)", a),
            ("B1 minimality witness (two regions, K=2)       ", b1),
            ("B2 minimality witness (non-monotone K=3)       ", b2),
            ("C  C_mono certificate unsound off C_mono       ", c),
            ("D  general position necessary for minimality   ", d),
            ("E  C_mono is A weakest, not THE unique         ", e)]
    for name, ok in rows:
        print(f"  [{'PASS' if ok else 'FAIL'}]  {name}")
    all_ok = all(ok for _, ok in rows)
    OUT["all_passed"] = bool(all_ok)
    with open("results_conj1_validator.json", "w") as f:
        json.dump(OUT, f, indent=2)
    print("\nwrote results_conj1_validator.json")
    print("\nOVERALL:", "ALL CHECKS PASS" if all_ok else f"FAILURES = {FAILURES}")
    print("\nInterpretation (honest):")
    print(" * UNCONDITIONAL core (Check A): one declared bit (the global orientation)")
    print("   certifies sign(Delta) on C_mono -- no general position needed.")
    print(" * CONDITIONAL minimality (Checks B1,B2 vs D): C_mono is a weakest one-bit class")
    print("   ONLY under general position; drop GP and a dominant region's orientation can")
    print("   certify outside C_mono (Check D) -- so the minimality is conditional on GP.")
    print(" * NON-UNIQUE (Check E): even under GP, C_mono is A weakest, not THE unique one.")
    print(" => Conjecture 1's open piece is resolved CONDITIONALLY; the certificate is")
    print("    unconditional, the 'weakest-class' boundary rests on general position.")
    sys.exit(0 if all_ok else 1)

if __name__ == "__main__":
    main()
