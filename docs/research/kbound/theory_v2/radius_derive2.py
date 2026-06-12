import numpy as np
# Honest derivation. g = xy/z, x,y,z in [-1,1], |x|,|y|,|z|>=cmin.
# Exact finite-difference bound (not just first-order). Perturb each by <=e (e<cmin/2 so denom stays >=cmin/2).
# |g_hat - g| = |x'y'/z' - xy/z|. Use the identity:
#   x'y'/z' - xy/z = [ (x'y'-xy) z + xy(z - z') ] / (z' z).
#   |x'y'-xy| <= |x'||y'-y|+|y||x'-x| <= 1*e + 1*e = 2e  (since |x'|,|y|<=1).
#   |xy(z-z')| <= |xy| e <= 1*e = e.
#   |z' z| >= cmin*(cmin/2) ... but better keep z' >= cmin - e.
# So |g_hat-g| <= (2e*|z| + e)/( |z'||z| ) = e(2|z|+1)/(|z'||z|) <= e(2+1/|z|)/|z'| <= e(2+1/cmin)/(cmin-e).
# With e<=cmin/2: <= e(2+1/cmin)/(cmin/2) = 2e(2+1/cmin)/cmin = e(4cmin+2)/cmin^2.
# For small cmin dominant term 2e/cmin^2. Let's VERIFY this exact bound.
rng=np.random.default_rng(1)
for cmin in [0.1,0.15,0.2,0.3]:
    worst=0; worst_over_bound=0
    for _ in range(300000):
        x=rng.uniform(cmin,1)*rng.choice([-1,1]); y=rng.uniform(cmin,1)*rng.choice([-1,1]); z=rng.uniform(cmin,1)*rng.choice([-1,1])
        e=rng.uniform(0,cmin/2)
        d=rng.uniform(-1,1,3); d=d/ (np.abs(d).max()+1e-12) * e   # ||d||_inf = e
        xp,yp,zp=x+d[0],y+d[1],z+d[2]
        if abs(zp)<cmin-e-1e-9: continue
        diff=abs(xp*yp/zp - x*y/z)
        bound=e*(4*cmin+2)/cmin**2
        worst=max(worst,diff)
        worst_over_bound=max(worst_over_bound, diff/ (bound+1e-15))
    print(f"cmin={cmin}: max ratio diff/bound = {worst_over_bound:.3f}  (valid iff <=1)")
print()
print("=> PROVEN bound: |g_hat - g| <= e_c (4 cmin + 2)/cmin^2.  For the b_i^2 estimator with e=e_c.")
print("   Leading term 2 e_c / cmin^2. We use C2 := (4 cmin + 2)/cmin^2 as the product-ratio constant.")
# sqrt step: |bhat - b| with b=sqrt(b2): |sqrt(u)-sqrt(v)| <= |u-v|/(2 sqrt(min(u,v))).
# With b2 >= bmin2 (=min b_i^2) and the perturbation small: |bhat_i - b_i| <= [C2 e_c]/(2 sqrt(bmin2 - C2 e_c)).
# Define r_n(b_i) = C2 e_c / (2 sqrt(bmin2 - C2 e_c)) for e_c small enough.
