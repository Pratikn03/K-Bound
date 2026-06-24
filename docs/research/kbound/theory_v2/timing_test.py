import time

import numpy as np

from theory_v2_validation import agreements, recover_b_up_to_flip, simulate_panel

# time one bootstrap trial
t=time.time()
q=np.array([0.7,0.72,0.71])
F,C,Y=simulate_panel(0.5,q,q,6000,1)
A=agreements(F);c=2*A-1
b=recover_b_up_to_flip(c,3)
# one bootstrap of 120
rng=np.random.default_rng(0); m=6000; gboot=np.empty(120); s=np.clip(rng.normal(0.55,0.25,m),0,1)
t0=time.time()
for bb in range(120):
    ridx=rng.integers(0,m,m); Fb=F[:,ridx]; Ab=agreements(Fb); cb=2*Ab-1
    bh=recover_b_up_to_flip(cb,3); gboot[bb]=abs(bh[1])/2-s[ridx].mean()+0.5-0.5
print("120-boot time:",round(time.time()-t0,3),"s per trial")
print("=> 800 trials x 3 scenarios =",round((time.time()-t0)*800*3,1),"s  (need <40 per bash call)")
