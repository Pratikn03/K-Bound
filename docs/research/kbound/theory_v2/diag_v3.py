import numpy as np
# diagnose radius magnitude at m=8000
def hoeffding_ec(m,K,delta): return np.sqrt(2*np.log(2*K*K/delta)/m)
def b_radius(m,K,delta,cmin,bmin2):
    ec=hoeffding_ec(m,K,delta); C2=(4*cmin+2)/cmin**2; db2=C2*ec
    floor=max(bmin2-db2,1e-6); return db2/(2*np.sqrt(floor)),ec,C2,db2
m=8000;K=3;delta=0.05;cmin=0.18;bmin2=0.20**2
rb,ec,C2,db2=b_radius(m,K,delta,cmin,bmin2)
print(f"m={m}: ec={ec:.4f}, C2={C2:.2f}, db2(bound on b^2 err)={db2:.4f}, rb(b radius)={rb:.4f}")
print(f" gamma radius contribution rb/2 = {rb/2:.4f}")
# EB radius for M with s_noise=0.25, m=8000:
s=np.random.default_rng(0).normal(0.5,0.25,m); s=np.clip(s,0,1)
Vhat=np.var(s,ddof=1)
rM=np.sqrt(2*Vhat*np.log(2/0.05)/m)+7*np.log(2/0.05)/(3*(m-1))
print(f"rM (EB for M)={rM:.4f}")
print(f"TOTAL r_n = rM + rb/2 = {rM+rb/2:.4f}")
print(f"=> need min_flip|gamma| > beta + r_n = 0.05 + {rM+rb/2:.4f} = {0.05+rM+rb/2:.4f} to reject")
print(f"   POWER case has true gamma=0.17 => min_flip ~ 0.17. 0.17 > {0.05+rM+rb/2:.4f}? {0.17>0.05+rM+rb/2}")
print()
print("PROBLEM: C2=(4cmin+2)/cmin^2 =",round((4*cmin+2)/cmin**2,1),"is HUGE for cmin=0.18 -> radius dominated by b.")
print("The product-ratio b radius is the bottleneck. Options: (a) larger m, (b) better b-estimator variance.")
# The Hoeffding-based bound is a worst-case Lipschitz bound; the ACTUAL sd of bhat is much smaller.
# For an HONEST audit we can use the worst-case PROVEN radius (sound but conservative) OR an empirical-Bernstein
# style plug-in. Let's measure ACTUAL sd of bhat_1 at m=8000 to show the gap.
import itertools
def agreements(F):
    K=F.shape[0]; A=np.empty((K,K))
    for i in range(K):
        for j in range(K): A[i,j]=np.mean(F[i]==F[j])
    return A
def recover(c,K):
    b2=np.zeros(K)
    for i in range(K):
        vals=[c[i,k]*c[i,l]/c[k,l] for k in range(K) for l in range(K) if len({i,k,l})==3 and abs(c[k,l])>1e-9]
        b2[i]=np.median(vals)
    mag=np.sqrt(np.clip(b2,0,None)); s=np.ones(K)
    for i in range(1,K): s[i]=np.sign(c[0,i]) or 1.0
    b=s*mag
    if np.sum(b)<0: b=-b
    return b
def sim(pi,q,m,seed):
    rng=np.random.default_rng(seed);K=len(q)
    Y=(rng.random(m)<pi).astype(int);qy=np.stack([q,q],1)[:,Y]
    C=(rng.random((K,m))<qy).astype(int);F=np.where(C==1,Y[None],1-Y[None])
    return F
q=np.array([0.70,0.72,0.71]);bh=[]
for r in range(300):
    F=sim(0.5,q,8000,1000+r);A=agreements(F);c=2*A-1;b=recover(c,3);bh.append(b[1])
print(f"\nACTUAL sd(bhat_1) at m=8000: {np.std(bh):.4f}  (vs worst-case proven rb={rb:.4f})")
print(f"  => worst-case radius is ~{rb/np.std(bh):.0f}x the actual sd. Conservative but SOUND.")
print(f"  For a usable-power audit, use empirical bootstrap radius; for the SOUNDNESS proof keep worst-case.")
