import numpy as np, itertools, json
from scipy.optimize import linprog

K=4
patterns = np.array(list(itertools.product([0,1],repeat=K)))
S = 2*patterns-1
b_fit = np.array([0.5,0.45,0.4,0.35])     # rank-1 target advantages (interior, decent margins)
c_target = np.outer(b_fit,b_fit)
pairs=[(0,1),(0,2),(0,3),(1,2),(1,3),(2,3)]
A_eq=[np.ones(16)]; b_eq=[1.0]
for (i,j) in pairs:
    A_eq.append(S[:,i]*S[:,j]); b_eq.append(c_target[i,j])
A_eq=np.array(A_eq); b_eq=np.array(b_eq)

# Find an INTERIOR feasible law with b_true != b_fit. Strategy:
# maximize the slack t s.t. p_k >= t for all k (Chebyshev-center-ish), with first moment of model0 shifted.
# Add a small push on E[s0] away from b_fit[0] via a secondary constraint, then center.
# Variables: p (16) and t (1). minimize -t. p_k - t >=0; sum p=1; pairwise eqs; plus E[s0] target.
target_b0 = 0.62   # force model-0 true advantage AWAY from b_fit[0]=0.5
A_eq2 = np.vstack([A_eq, S[:,0]])
b_eq2 = np.append(b_eq, target_b0)
nv=16
# inequalities: -(p_k) + t <= 0  => A_ub row: e_k has -1 on p_k, +1 on t
A_ub=[]; b_ub=[]
for k in range(nv):
    row=np.zeros(nv+1); row[k]=-1; row[nv]=1; A_ub.append(row); b_ub.append(0.0)
A_ub=np.array(A_ub); b_ub=np.array(b_ub)
A_eq_full=np.hstack([A_eq2, np.zeros((A_eq2.shape[0],1))])
c_obj=np.zeros(nv+1); c_obj[nv]=-1.0   # maximize t
bounds=[(0,1)]*nv+[(0,1)]
res=linprog(c_obj,A_ub=A_ub,b_ub=b_ub,A_eq=A_eq_full,b_eq=b_eq2,bounds=bounds,method="highs")
print("LP ok:",res.success, "min pattern prob t=",round(res.x[nv],5))
p=res.x[:nv]
b_true=S.T@p
c_real=np.array([[np.sum(p*S[:,i]*S[:,j]) for j in range(K)] for i in range(K)])
od=~np.eye(K,dtype=bool)
print("b_fit :",b_fit)
print("b_true:",b_true.round(4))
print("min pattern prob:",round(p.min(),5)," (interior if >0)")
print("max offdiag |c_real - rank1|:",np.format_float_scientific(np.max(np.abs((c_real-c_target)[od])),precision=2))
prods=[c_real[0,1]*c_real[2,3],c_real[0,2]*c_real[1,3],c_real[0,3]*c_real[1,2]]
print("tau (max-min products):",round(max(prods)-min(prods),7))
print("|b_true - b_fit| max:",round(np.max(np.abs(b_true-b_fit)),4))

# Now SIMULATE m samples from this exact law and run the product-ratio estimator. It must return ~b_fit.
def recover(c,K):
    b2=np.zeros(K)
    for i in range(K):
        vals=[c[i,k]*c[i,l]/c[k,l] for k in range(K) for l in range(K) if len({i,k,l})==3 and abs(c[k,l])>1e-9]
        b2[i]=np.median(vals)
    mag=np.sqrt(np.clip(b2,0,None)); s=np.ones(K)
    for i in range(1,K): s[i]=np.sign(c[0,i]) or 1.0
    b=s*mag
    if np.sign(np.sum(b))<0: b=-b
    return b
rng=np.random.default_rng(5)
m=400000
idx=rng.choice(16,size=m,p=p/ p.sum())
Cs=patterns[idx]                    # correctness samples (CEI false but pairwise rank-1)
b_hat_emp = 2*Cs.mean(0)-1          # naive first-moment 'truth'
Ss=2*Cs-1
c_emp=np.array([[np.mean(Ss[:,i]*Ss[:,j]) for j in range(K)] for i in range(K)])
b_rec=recover(c_emp,K)
print()
print("SIMULATED from stealth law, m=",m)
print("product-ratio recovered b_rec:",b_rec.round(4)," (should track b_fit, the rank-1 fit)")
print("true first-moment advantages :",b_hat_emp.round(4)," (should track b_true)")
print("=> estimator BIASED by:",round(np.max(np.abs(b_rec-b_hat_emp)),4)," while tau~0")

# save canonical construction
out={"b_fit":b_fit.tolist(),"b_true":b_true.round(6).tolist(),"p":p.round(8).tolist(),
     "min_pattern_prob":float(p.min()),"tau":float(max(prods)-min(prods)),
     "max_b_true_minus_b_fit":float(np.max(np.abs(b_true-b_fit))),
     "patterns":patterns.tolist()}
with open("/sessions/peaceful-blissful-ptolemy/mnt/uav/AutoML_Flagship_V8/docs/research/kbound/theory_v2/stealth_law.json","w") as f:
    json.dump(out,f,indent=2)
print("saved stealth_law.json")
