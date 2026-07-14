import numpy as np, itertools, json
from scipy.optimize import linprog
# Push for a DECISION-LEVEL stealth failure: rank-1 fit says sign(b_1-b_0)>0 but truth says <0 (or vice versa),
# all with tau=0 and interior law.
K=4
patterns=np.array(list(itertools.product([0,1],repeat=K))); S=2*patterns-1
# rank-1 fit: make b_fit_1 > b_fit_0 (candidate1 looks better than frozen f0)
b_fit=np.array([0.35,0.55,0.45,0.4])
c_target=np.outer(b_fit,b_fit)
pairs=[(0,1),(0,2),(0,3),(1,2),(1,3),(2,3)]
A_eq=[np.ones(16)]+[S[:,i]*S[:,j] for (i,j) in pairs]
b_eq=[1.0]+[c_target[i,j] for (i,j) in pairs]
# force TRUE ordering to REVERSE on the (0,1) comparison: want b_true_0 > b_true_1
# add constraints E[s0]=t0, E[s1]=t1 with t0>t1
t0,t1=0.6,0.30
A_eq=np.array(A_eq+[S[:,0],S[:,1]]); b_eq=np.array(b_eq+[t0,t1])
# Chebyshev-center for interiority
nv=16; A_ub=[]; b_ub=[]
for k in range(nv):
    r=np.zeros(nv+1); r[k]=-1; r[nv]=1; A_ub.append(r); b_ub.append(0.0)
A_ub=np.array(A_ub); b_ub=np.array(b_ub)
A_eq_f=np.hstack([A_eq,np.zeros((A_eq.shape[0],1))])
c_obj=np.zeros(nv+1); c_obj[nv]=-1
res=linprog(c_obj,A_ub=A_ub,b_ub=b_ub,A_eq=A_eq_f,b_eq=b_eq,bounds=[(0,1)]*nv+[(0,1)],method="highs")
print("LP ok:",res.success," min prob t=",round(res.x[nv],5) if res.success else None)
if res.success and res.x[nv]>1e-6:
    p=res.x[:nv]; b_true=S.T@p
    c_real=np.array([[np.sum(p*S[:,i]*S[:,j]) for j in range(K)] for i in range(K)])
    od=~np.eye(K,dtype=bool)
    prods=[c_real[0,1]*c_real[2,3],c_real[0,2]*c_real[1,3],c_real[0,3]*c_real[1,2]]
    print("b_fit :",b_fit," -> sign(b1-b0)=",int(np.sign(b_fit[1]-b_fit[0])),"(candidate1 looks better)")
    print("b_true:",b_true.round(4)," -> sign(b1-b0)=",int(np.sign(b_true[1]-b_true[0])),"(truth)")
    print("tau:",round(max(prods)-min(prods),7)," min pattern prob:",round(p.min(),5),
          " max|c-rank1|:",np.format_float_scientific(np.max(np.abs((c_real-c_target)[od])),precision=2))
    print("DECISION FLIPPED:", np.sign(b_fit[1]-b_fit[0])!=np.sign(b_true[1]-b_true[0]))
    out={"b_fit":b_fit.tolist(),"b_true":b_true.round(6).tolist(),"p":p.round(8).tolist(),
         "tau":float(max(prods)-min(prods)),"min_prob":float(p.min()),
         "sign_bfit_1m0":int(np.sign(b_fit[1]-b_fit[0])),"sign_btrue_1m0":int(np.sign(b_true[1]-b_true[0]))}
    json.dump(out,open("/sessions/peaceful-blissful-ptolemy/mnt/uav/AutoML_Flagship_V8/docs/research/kbound/theory_v2/stealth_flip.json","w"),indent=2)
    print("saved stealth_flip.json")
else:
    print("infeasible at requested interiority; try relaxed targets")
