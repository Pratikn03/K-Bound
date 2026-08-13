#!/usr/bin/env python3
"""
run_iwildcam_heldout_seed_sweep.py — held-out H v2 seed-robustness, pending iCloud download.

Run from repo root AFTER downloading
  experiments/kbound/results/iwildcam_full_test/result_e40faf29.json
(Finder > Download Now):

    python3 docs/research/kbound/scripts/run_iwildcam_heldout_seed_sweep.py

Replays the locked held-out configuration (tent_episodic, cal stream seed 0 ->
test stream seed 1, gbr/global exact-rank, alpha=0.10), verifies it reproduces
the locked test row in iwildcam_protocol_H_v2/protocol_result.json, then sweeps
the GBR decision seed (random_state 0..15) and reports no-harm stability.
Diagnostic only until NATURAL_MULTISEED_REPLAY_v1 is locked.
"""
import json, sys
import numpy as np

sys.path.insert(0, 'docs/research/kbound/scripts')
import analyze_F as A
from sklearn.ensemble import GradientBoostingRegressor

ALPHA = 0.10
REC = 'experiments/kbound/results/iwildcam_full_test/result_e40faf29.json'
LOCK = 'experiments/kbound/results/iwildcam_protocol_H_v2/protocol_result.json'

def fit_rs(Zc, Bc, rs):
    return GradientBoostingRegressor(n_estimators=250, max_depth=2, learning_rate=0.05,
                                     subsample=0.8, random_state=rs).fit(Zc, Bc)

def run(recs, rs):
    Z, B, a0, aa, sd, _ = A.arrays(recs)
    cal = np.isin(sd, [0]); tst = np.isin(sd, [1])
    Zc, Bc = Z[cal], B[cal]
    loo = np.empty(len(Bc))
    for i in range(len(Bc)):
        tr = np.arange(len(Bc)) != i
        loo[i] = fit_rs(Zc[tr], Bc[tr], rs).predict(Zc[i:i+1])[0]
    eps = A.conformal_rank_radius(np.abs(loo - Bc), ALPHA)
    dec = A.decide_global(fit_rs(Zc, Bc, rs).predict(Z[tst]), eps)
    return A.metrics(dec, B[tst], a0[tst], aa[tst])

try:
    recs, _ = A.load_records(REC, candidate='tent_episodic')
except Exception as e:
    sys.exit(f'Cannot read {REC} — still an iCloud placeholder? ({e})')

lock = json.load(open(LOCK))['test_locked']
m0 = run(recs, 0)
match = all(abs(m0[k] - lock[k]) < 1e-9 for k in
            ('regret_kga', 'regret_adapt', 'regret_freeze', 'false_adapt'))
print(f'replay(rs=0): KGA {m0["regret_kga"]:.5f} adapt {m0["regret_adapt"]:.5f} '
      f'freeze {m0["regret_freeze"]:.5f} FA {m0["false_adapt"]:.3f} | matches locked: {match}')

rows = [run(recs, rs) for rs in range(16)]
kga = np.array([r['regret_kga'] for r in rows])
fa = np.array([r['false_adapt'] for r in rows])
better = np.array([min(r['regret_adapt'], r['regret_freeze']) for r in rows])
noharm = bool(all(fa <= ALPHA) and all(kga <= better + 0.005))
print(f'decision-seed sweep (16): regret_kga [{kga.min():.5f},{kga.max():.5f}]  '
      f'FA max {fa.max():.3f}  no-harm all seeds: {noharm}')
out = {'replay_matches_locked': bool(match), 'sweep_noharm_all': noharm,
       'regret_kga_min_max': [float(kga.min()), float(kga.max())],
       'false_adapt_max': float(fa.max())}
json.dump(out, open('experiments/kbound/results/natural_seed_robustness_v1/iwildcam_heldout_sweep.json', 'w'), indent=1)
print('Saved -> experiments/kbound/results/natural_seed_robustness_v1/iwildcam_heldout_sweep.json')
