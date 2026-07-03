import sympy as sp
# Perturbation of product-ratio b_i^2 = c_ik c_il / c_kl. Each hat c in [-1,1], |c|>=cmin on relevant entries.
# If |hat c_ab - c_ab| <= e for all pairs, bound |hat b_i^2 - b_i^2|.
# Let g(x,y,z)=xy/z. grad = (y/z, x/z, -xy/z^2). With |x|,|y|<=1, |z|>=cmin, |xy/z|<=1/cmin (since |xy|<=1)...
# Actually |b_i^2|<=1 so |xy/z| = |b_i^2|... but we bound Lipschitz: |y/z|<=1/cmin, |x/z|<=1/cmin,
# |xy/z^2|<=1/cmin^2 *|xy| <= ... use |xy|<=1 => <=1/cmin^2. Hmm tighter: |xy| = |b_i^2 c_kl| <= |c_kl| (since b_i^2<=1)
#   => |xy/z^2| <= |z|/z^2 = 1/|z| <= 1/cmin. So all three partials <= 1/cmin.
# First-order: |hat b_i^2 - b_i^2| <~ (1/cmin)(e+e+e)=3e/cmin. Add second order O(e^2/cmin^2) (drop for small e).
# Then |hat b_i - b_i|: b_i = sqrt(b_i^2). d/du sqrt(u)=1/(2 sqrt(u)) <= 1/(2 b_min) where b_min=min|b_i|.
#   => |hat b_i - b_i| <= (1/(2 b_min)) * 3e/cmin = 3e/(2 b_min cmin). Note b_min^2=cmin-ish; if cmin=b_min^2*(min over k,l)... 
# Cleaner: since b_i^2>= some bmin2, and product-ratio gives b_i^2, use |sqrt(u)-sqrt(v)|<=|u-v|/(sqrt(u)+sqrt(v))<=|u-v|/(2 sqrt(min)).
# Let's just verify the 3/cmin Lipschitz of the product-ratio numerically (worst case search) and the sqrt step.
import numpy as np
rng=np.random.default_rng(0)
cmin=0.15
worst=0
for _ in range(200000):
    # random c entries with |.|>=cmin, |.|<=1
    x=rng.uniform(cmin,1)*rng.choice([-1,1]); y=rng.uniform(cmin,1)*rng.choice([-1,1]); z=rng.uniform(cmin,1)*rng.choice([-1,1])
    e=0.001*rng.uniform()
    dx,dy,dz=(rng.uniform(-1,1,3))*e
    # keep perturbed within bounds
    xp,yp,zp=x+dx,y+dy,z+dz
    if abs(zp)<cmin*0.5: continue
    g=x*y/z; gp=xp*yp/zp
    bound=3*e/cmin + 3*e**2/cmin**2   # first+crude second
    if abs(gp-g)>worst*0: pass
    ratio=abs(gp-g)/(e/cmin)
    worst=max(worst,ratio)
print("empirical worst |Δ(xy/z)| / (e/cmin) over random perturbs:",round(worst,3)," (theory bound 3)")
# => constant 3 is a valid Lipschitz multiplier for the product ratio in units e/cmin (confirmed <3).
print("So r_n for b_i: with |hat c-c|<=e_c whp, |hat b_i^2 - b_i^2| <= 3 e_c/cmin (lead order).")
print("Then |hat b_i - b_i| <= 3 e_c/(cmin*(|hat b_i|+|b_i|)) <= 3 e_c/(2 cmin b_min).")
# Hoeffding for hat c_ab = 2*Abar-1, Abar mean of m Bernoulli => |hat c_ab - c_ab|<= 2*sqrt(log(2 P/delta)/(2m))
#   = sqrt(2 log(2P/delta)/m), P=#pairs. So e_c = sqrt(2 log(2 K^2/delta)/m).
print("e_c = sqrt(2 log(2 K^2/delta)/m)  (Hoeffding, union over <=K^2 pairs).")
