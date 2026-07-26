#!/usr/bin/env python3
"""Post-hoc significance check for Protocol F's beats-both (NOT pre-registered; the locked
endpoint was FA<=alpha & commit>=0.3). Replicates analyze_F.run_split routing for the locked
config (ppi_debias + mondrian, DEV=[0,1], TEST=[2,3,4]) to get per-test-record regrets, then
paired bootstrap over test records. Verifies means match the locked numbers first."""
import sys, numpy as np
sys.path.insert(0, "docs/research/kbound/scripts")
import analyze_F as A
# fix-queue item 15 / defect D10: the certificate radius is the shipped one.
import kbound_decide as _kb  # noqa: E402
ALPHA = 0.10
recs, _panel = A.load_records("experiments/kbound/results/camelyon17_richZ_F_v1/result_884129ba.json")
Z, B, a0, aa, sd, comp = A.arrays(recs)
cal = np.isin(sd, [0, 1]); tst = np.isin(sd, [2, 3, 4])
Zc, Bc = Z[cal], B[cal]; Zt, Bt, a0t, aat = Z[tst], B[tst], a0[tst], aa[tst]
compc, compt = comp[cal], comp[tst]
m = A.fit_point(Zc, Bc); Bhat_c = m.predict(Zc); Bhat_t = m.predict(Zt)
Bhat_c, Bhat_t = A.ppi_debias(Bhat_c, Bc, Zc, Zt, Bhat_t)
# D10: exact split-conformal rank radius via the shipped library, not np.quantile.
eps_glob = float(_kb.conformal_radius(np.abs(Bhat_c - Bc), ALPHA))
dec = np.array(["ABSTAIN"] * len(Bhat_t), dtype=object)
groups = set(compc.tolist())
for g in groups:
    mc = compc == g
    epsg = float(_kb.conformal_radius(np.abs(Bhat_c[mc] - Bc[mc]), ALPHA)) if mc.sum() >= 5 else eps_glob
    mt = compt == g
    dec[mt] = A.decide_global(Bhat_t[mt], epsg)
unseen = ~np.isin(compt, list(groups)); dec[unseen] = A.decide_global(Bhat_t[unseen], eps_glob)
adapt = dec == "ADAPT"; kga = np.where(adapt, aat, a0t); oracle = np.maximum(a0t, aat)
rk = oracle - kga; ra = oracle - aat; rf = oracle - a0t
print(f"reproduced means: KGA={rk.mean():.5f} adapt={ra.mean():.5f} freeze={rf.mean():.5f}  "
      f"(locked: 0.00194/0.00449/0.065)  FA={float((Bt[adapt]<0).mean()):.4f} commit={float((dec!='ABSTAIN').mean()):.3f} n={len(rk)}")
n = len(rk); rng = np.random.default_rng(20260616); BB = 10000; dka = []; dkf = []
for _ in range(BB):
    ii = rng.integers(0, n, n)
    dka.append(ra[ii].mean() - rk[ii].mean())   # >0 => KGA lower regret than always-adapt
    dkf.append(rf[ii].mean() - rk[ii].mean())
dka = np.array(dka); dkf = np.array(dkf)
def ci(a): return [round(float(np.quantile(a, .025)), 5), round(float(np.quantile(a, .975)), 5)]
print(f"KGA<adapt  by {ra.mean()-rk.mean():+.5f}  95%CI {ci(dka)}  P={float((dka>0).mean()):.3f}")
print(f"KGA<freeze by {rf.mean()-rk.mean():+.5f}  95%CI {ci(dkf)}  P={float((dkf>0).mean()):.3f}")
print(f"beats_both_significant: {bool((dka>0).mean()>=0.95 and (dkf>0).mean()>=0.95)}")
