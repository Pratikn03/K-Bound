import numpy as np
rng = np.random.default_rng(7)

# ============================================================
# (A) FLIP WITNESS under H (per-class symmetric accuracy).
# Y' = 1-Y. Predictions f_j UNCHANGED (fixed maps of X).
# Under flip: correctness C_j' = 1[f_j=Y'] = 1[f_j != Y] = 1 - C_j.
# So a_j' = 1 - a_j, b_j' = -b_j. Agreements A_ij = P(f_i=f_j) UNCHANGED (no Y in it).
# Check H is preserved: q_j'(y') = P(C_j'=1|Y'=y') = P(1-C_j=1 | 1-Y=y') = P(C_j=0|Y=1-y')=1-q_j(1-y').
# If q_j(0)=q_j(1)=q_j (H), then q_j'(0)=1-q_j(1)=1-q_j, q_j'(1)=1-q_j(0)=1-q_j => symmetric. H preserved. Good.
# Evidence law = joint dist of prediction patterns (f_1..f_K) -- unchanged. TV=0 between P and P'.
# ============================================================
K=4; n=2_000_000
qsym = rng.uniform(0.6,0.9,size=K)
pi=0.45
Y=(rng.random(n)<pi).astype(int)
q=np.stack([qsym,qsym],axis=1)[:,Y]
C=(rng.random((K,n))<q).astype(int)
F=np.where(C==1,Y[None,:],1-Y[None,:])    # predictions, fixed maps
# original
a=C.mean(1); b=2*a-1
# pattern law (joint freq of (f1..fK)) -- the full label-free evidence law
codes = F[0]*8+F[1]*4+F[2]*2+F[3]
patt = np.bincount(codes, minlength=16)/n
# flipped: Y'=1-Y, SAME predictions F, recompute correctness
Yp=1-Y
Cp=(F==Yp[None,:]).astype(int)
ap=Cp.mean(1); bp=2*ap-1
codes_p = F[0]*8+F[1]*4+F[2]*2+F[3]   # predictions identical
patt_p = np.bincount(codes_p, minlength=16)/n
print("=== FLIP WITNESS (under H) ===")
print("b      :", np.round(b,4))
print("b' flip:", np.round(bp,4), " (should be -b)")
print("b' + b :", np.round(bp+b,4), " (should be ~0)")
print("max|pattern_law - pattern_law'| =", np.format_float_scientific(np.max(np.abs(patt-patt_p)),precision=2),
      " (TV=0 by construction; predictions identical)")
# agreements unchanged
A=np.zeros((K,K))
for i in range(K):
  for j in range(K): A[i,j]=(F[i]==F[j]).mean()
od=~np.eye(K,dtype=bool); c=2*A-1
print("c_ij=b_i b_j check (H): max offdiag", np.format_float_scientific(np.max(np.abs((c-np.outer(b,b))[od])),precision=2))
print("c_ij=b'_i b'_j check  : max offdiag", np.format_float_scientific(np.max(np.abs((c-np.outer(bp,bp))[od])),precision=2),
      " (products invariant: (-b)(-b)=bb)")

# ============================================================
# (B) AoL win-rate / drift flip relation.
# On D_theta = {f_0 != f_theta}, win rate w = P(f_theta=Y | f_0!=f_theta).
# slope 1-2 wbar. Under flip Y->1-Y: on same D_theta (predictions fixed), w -> P(f_theta=1-Y|...) = 1-w.
# slope 1-2(1-w) = -(1-2w). VERIFY.
# Also gamma on disagreement region relates: gamma = E[eta_a - s], with flip eta_a->1-eta_a if s fixed... 
# but the cleaner audited claim: w -> 1-w, slope sign flips. Check numerically.
# ============================================================
print()
print("=== AoL: win rate & slope under flip ===")
# build f_0 and a family f_theta; compute w and slope on a target P; flip; recompute
n2=1_500_000
X=rng.normal(size=n2)
# true label depends on X
eta = 1/(1+np.exp(-1.5*X))   # P(Y=1|X)
Y2=(rng.random(n2)<eta).astype(int)
f0=(X>0.0).astype(int)
a0=(f0==Y2).mean()
slopes=[]; wbars=[]
As=[]; accs=[]
thetas=np.linspace(-0.8,0.8,9)
ws=[]
for th in thetas:
    fth=(X> th).astype(int)
    Dm = f0!=fth
    if Dm.sum()<100: 
        continue
    w=(fth[Dm]==Y2[Dm]).mean()
    A_=(fth==f0).mean()
    acc=(fth==Y2).mean()
    As.append(A_); accs.append(acc); ws.append(w)
As=np.array(As); accs=np.array(accs); ws=np.array(ws)
# fit acc = a0 + slope*(A-1); slope ~ 1-2 wbar
slope = np.polyfit(As-1, accs-a0, 1)[0]
print("a0:",round(a0,4)," mean win-rate wbar:",round(ws.mean(),4)," fitted slope:",round(slope,4)," 1-2wbar:",round(1-2*ws.mean(),4))
# flip
Y2f=1-Y2
a0f=(f0==Y2f).mean()
accsf=[]; wsf=[]
for th in thetas:
    fth=(X>th).astype(int); Dm=f0!=fth
    if Dm.sum()<100: continue
    wsf.append((fth[Dm]==Y2f[Dm]).mean()); accsf.append((fth==Y2f).mean())
accsf=np.array(accsf); wsf=np.array(wsf)
slopef=np.polyfit(As-1, accsf-a0f, 1)[0]
print("FLIP: wbar':",round(wsf.mean(),4)," (=1-wbar:",round(1-ws.mean(),4),") slope':",round(slopef,4)," -(1-2wbar):",round(-(1-2*ws.mean()),4))
print("w'+w per theta max dev from 1:", round(np.max(np.abs(np.array(wsf)+np.array(ws)-1)),4))

# ============================================================
# (C) gamma identity: gamma = bar_a - 1/2 - M = b_a/2 - M.  (M = E[s]-1/2)
# bar_a = E[eta_a|D]; b_a = 2*bar_a - 1 => bar_a-1/2 = b_a/2. gamma = E[eta_a-s] = (bar_a-1/2)-(E[s]-1/2)=b_a/2 - M. Identity by defn.
# ============================================================
print()
print("=== gamma identity gamma = b_a/2 - M (definitional) ===")
bar_a=0.63; M=0.05
gamma1=bar_a-0.5-M
b_a=2*bar_a-1
gamma2=b_a/2-M
print("gamma(bar_a-1/2-M)=",round(gamma1,5)," gamma(b_a/2-M)=",round(gamma2,5)," equal:",abs(gamma1-gamma2)<1e-12)
