import numpy as np, sys
sys.argv=['x']
exec(open('theory_v2_validation.py').read().split('if __name__')[0])
alpha=0.05; beta=0.05; cmin=0.18; bmin2=0.20**2; K=3; delta=alpha
q=np.random.default_rng(303).uniform(0.62,0.85,size=K)
b_a=2*q[1]-1
margin=0.18
M_true2=b_a/2-(beta+margin)   # true gamma = 0.23
print("q:",q.round(3)," b_a:",round(b_a,3)," M_true2:",round(M_true2,3)," true gamma=b_a/2-M:",round(b_a/2-M_true2,3))
m=6000
for seed in [7000,7001,7002]:
    rng=np.random.default_rng(seed)
    F,C,Y=simulate_panel(0.5,q,q,m,seed=seed)
    A=agreements(F);c=2*A-1
    b_hat=recover_b_up_to_flip(c,K,anchor_sign=+1)
    b_a_hat=b_hat[1]
    s=np.clip(rng.normal(M_true2+0.5,0.25,size=m),0,1)
    M_hat=s.mean()-0.5
    gp=abs(b_a_hat)/2-M_hat; gm=-abs(b_a_hat)/2-M_hat; mf=min(abs(gp),abs(gm))
    # bootstrap
    n_boot=80; gboot=np.empty(n_boot)
    for bb in range(n_boot):
        ridx=rng.integers(0,m,m); Fb=F[:,ridx]; Ab=agreements(Fb);cb=2*Ab-1
        bh=recover_b_up_to_flip(cb,K,anchor_sign=+1)
        gboot[bb]=abs(bh[1])/2-s[ridx].mean()+0.5-0.5
    rb_boot=float(np.quantile(np.abs(gboot-gp),1-alpha))
    print(f" seed{seed}: b_a_hat={b_a_hat:.3f} M_hat={M_hat:.3f} gp={gp:.3f} gm={gm:.3f} min_flip={mf:.3f}")
    print(f"          rb_boot={rb_boot:.4f}  threshold beta+rb_boot={beta+rb_boot:.4f}  reject={mf>beta+rb_boot}")
    # The gboot std:
    print(f"          gboot mean={gboot.mean():.3f} std={gboot.std():.4f}")
