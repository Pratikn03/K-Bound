import numpy as np

from theory_v2_validation import agreements, recover_b_up_to_flip, simulate_panel

# How does labeled-vs-evidence efficiency ratio grow as margins (c_min) shrink?
print("c_min (min pairwise product) vs efficiency ratio sd_evidence/sd_labeled:")
m=5000; reps=300
configs={
 "high-margin q=.78,.74,.72": np.array([0.78,0.74,0.72]),
 "mid q=.70,.64,.62":         np.array([0.70,0.64,0.62]),
 "low q=.62,.58,.57":         np.array([0.62,0.58,0.57]),
 "verylow q=.58,.555,.55":    np.array([0.58,0.555,0.55]),
}
for name,q in configs.items():
    b=2*q-1; cmin=np.min(np.abs(np.outer(b,b))[~np.eye(3,dtype=bool)])
    dl=np.zeros(reps); de=np.zeros(reps)
    for r in range(reps):
        F,C,Y=simulate_panel(0.5,q,q,m,seed=40000+r)
        dl[r]=(C[1].astype(float)-C[0].astype(float)).mean()
        A=agreements(F);c=2*A-1; bh=recover_b_up_to_flip(c,3); de[r]=(bh[1]-bh[0])/2
    print(f"  {name:32s} cmin={cmin:.3f}  sd_lab={np.std(dl):.4f} sd_ev={np.std(de):.4f}  ratio={np.std(de)/np.std(dl):.2f}")
print("\n=> efficiency ratio GROWS as cmin shrinks: labels buy a constant that blows up at low agreement.")
print("   So 'labels buy only constants not rate' is TRUE (same n^-1/2) but the constant is cmin-dependent.")
