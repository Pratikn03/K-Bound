#!/usr/bin/env python3
"""
natural_multiseed_aggregate.py — Part A of NATURAL_MULTISEED_REPLAY_v1 (DRAFT).

Run from the repository root on the Mac:
    python3 docs/research/kbound/scripts/natural_multiseed_aggregate.py \
        --out experiments/kbound/results/natural_seed_robustness_v1

Re-scores ONLY existing serialized records under the locked per-track configs
(analyze_F pipeline, exact rank conformal, alpha=0.10), producing:
  - per-seed no-harm tables for RxRx1 (3 model seeds), Office-Home (stream seeds
    0-1 primary if downloaded, 2-4 replication), iWildCam (dev screen + held-out
    seed split if the test records file is materialized),
  - a decision-seed (GBR random_state) sweep per track,
  - a markdown table in the paper's tab:multiseed format.

No GPU, no new adaptation. Nothing is promoted: output is diagnostic until
NATURAL_MULTISEED_REPLAY_v1 is locked and its criteria are applied.
"""
import argparse, json, os, sys
import numpy as np
from pathlib import Path

sys.path.insert(0, 'docs/research/kbound/scripts')
import analyze_F as A
from sklearn.ensemble import GradientBoostingRegressor

ALPHA = 0.10
N_DECISION_SEEDS = 16

def fit_point_rs(Zc, Bc, rs):
    return GradientBoostingRegressor(n_estimators=250, max_depth=2, learning_rate=0.05,
                                     subsample=0.8, random_state=rs).fit(Zc, Bc)

def split_metrics(cal_recs, tst_recs, cal_seeds, test_seeds, rs=0):
    """analyze_F 'gbr'+'global' path with random_state exposed; transfer design when
    cal_recs is not tst_recs, seed-split design when they are the same list."""
    Zv, Bv, _, _, sv, _ = A.arrays(cal_recs)
    Zt, Bt, a0t, aat, st, _ = A.arrays(tst_recs)
    ci = np.isin(sv, cal_seeds); ti = np.isin(st, test_seeds)
    if ci.sum() < 2 or ti.sum() == 0:
        return None
    Zc, Bc = Zv[ci], Bv[ci]
    model = fit_point_rs(Zc, Bc, rs)
    loo = np.empty(len(Bc))
    for i in range(len(Bc)):
        tr = np.arange(len(Bc)) != i
        loo[i] = fit_point_rs(Zc[tr], Bc[tr], rs).predict(Zc[i:i+1])[0]
    eps = A.conformal_rank_radius(np.abs(loo - Bc), ALPHA)
    dec = A.decide_global(model.predict(Zt[ti]), eps)
    met = A.metrics(dec, Bt[ti], a0t[ti], aat[ti])
    met['eps_global'] = float(eps)
    return met

def verdict(m):
    better = min(m['regret_adapt'], m['regret_freeze'])
    return {'fa_ok': m['false_adapt'] <= ALPHA,
            'matches_better_policy': bool(m['regret_kga'] <= better + 1e-6),
            'noharm_tol': bool(m['regret_kga'] <= better + 0.005 and m['false_adapt'] <= ALPHA),
            'beats_both_point_UNPROMOTED': bool(m['regret_kga'] < m['regret_adapt'] and m['regret_kga'] < m['regret_freeze'])}

def sweep(cal_recs, tst_recs, cal_seeds, test_seeds):
    rows = [dict(split_metrics(cal_recs, tst_recs, cal_seeds, test_seeds, rs=rs), rs=rs)
            for rs in range(N_DECISION_SEEDS)]
    col = lambda k: np.array([r[k] for r in rows], float)
    return {'n': N_DECISION_SEEDS,
            'regret_kga_min_max': [float(col('regret_kga').min()), float(col('regret_kga').max())],
            'false_adapt_max': float(col('false_adapt').max()),
            'noharm_all': bool(all(verdict(r)['noharm_tol'] for r in rows))}

def maybe(path):
    p = Path(path)
    if not p.exists():
        return None
    try:
        with open(p) as fh:
            if fh.read(16) == '':
                return None  # iCloud placeholder reads empty
    except OSError:
        return None
    return str(p)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default='experiments/kbound/results/natural_seed_robustness_v1')
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    R = {'protocol_draft': 'NATURAL_MULTISEED_REPLAY_v1_DRAFT', 'alpha': ALPHA, 'tracks': {}}
    md = ['# Natural-shift multi-seed robustness (Part A re-scores)', '',
          '| track | seed axis | seed | KGA | adapt | freeze | FA_u | no-harm |',
          '|---|---|---|---|---|---|---|---|']

    # RxRx1: 3 model seeds
    rxf = {0: 'experiments/kbound/results/rxrx1_protocol_c_9plus_modelseed0/result_3f579e72.json',
           1: 'experiments/kbound/results/rxrx1_protocol_c_9plus_modelseed1/result_eef46aea.json',
           2: 'experiments/kbound/results/rxrx1_protocol_c_9plus_modelseed2/result_6585f5b7.json'}
    rx = {}
    for ms, f in rxf.items():
        if not maybe(f):
            rx[f'modelseed{ms}'] = 'MISSING-or-placeholder'; continue
        recs, _ = A.load_records(f, candidate='sar_online')
        m = split_metrics(recs, recs, [0,1,2,3,4], [5,6,7,8,9], rs=0)
        rx[f'modelseed{ms}'] = {**m, 'verdict': verdict(m),
                                'decision_seed_sweep': sweep(recs, recs, [0,1,2,3,4], [5,6,7,8,9])}
        md.append(f"| RxRx1 J | model | {ms} | {m['regret_kga']:.4f} | {m['regret_adapt']:.4f} | "
                  f"{m['regret_freeze']:.4f} | {m['false_adapt']:.3f} | {verdict(m)['noharm_tol']} |")
    R['tracks']['rxrx1_J'] = rx

    # Office-Home: primary (0-1) if materialized, replication (2-4)
    oh_primary_val = maybe('experiments/kbound/results/officehome_full_targetval/result_target_val_361a1e8c.json')
    oh_primary_tst = maybe('experiments/kbound/results/officehome_full_targettest/result_target_test_6605675d.json')
    oh = {}
    pairs = []
    if oh_primary_val and oh_primary_tst:
        pairs += [('primary', oh_primary_val, oh_primary_tst, [0,1])]
    else:
        oh['primary'] = 'SKIPPED (record files are iCloud placeholders — Finder > Download Now to include seeds 0-1)'
    pairs += [('replication',
               'experiments/kbound/results/officehome_protocol_m_repl_targetval/result_target_val_eb504dd6.json',
               'experiments/kbound/results/officehome_protocol_m_repl_targettest/result_target_test_f761540b.json',
               [2,3,4])]
    for name, calf, tstf, seedlist in pairs:
        if not (maybe(calf) and maybe(tstf)):
            oh[name] = 'MISSING-or-placeholder'; continue
        cr, _ = A.load_records(calf, candidate='sar_online_aggressive')
        tr, _ = A.load_records(tstf, candidate='sar_online_aggressive')
        block = {}
        for s in seedlist:
            m = split_metrics(cr, tr, [s], [s], rs=0)
            block[f'stream_seed{s}'] = {**m, 'verdict': verdict(m)}
            md.append(f"| Office-Home M v2 ({name}) | stream | {s} | {m['regret_kga']:.4f} | "
                      f"{m['regret_adapt']:.4f} | {m['regret_freeze']:.4f} | {m['false_adapt']:.3f} | {verdict(m)['noharm_tol']} |")
        pooled = split_metrics(cr, tr, seedlist, seedlist, rs=0)
        block['pooled'] = {**pooled, 'verdict': verdict(pooled),
                           'decision_seed_sweep': sweep(cr, tr, seedlist, seedlist)}
        oh[name] = block
    R['tracks']['officehome_M_v2'] = oh

    # iWildCam: dev screen always; held-out if materialized
    iw = {}
    iwd = maybe('experiments/kbound/results/iwildcam_full_idval/result_489da28f.json')
    if iwd:
        recs, _ = A.load_records(iwd, candidate='tent_episodic')
        m = split_metrics(recs, recs, [0], [1], rs=0)
        iw['dev_screen'] = {**m, 'verdict': verdict(m),
                            'decision_seed_sweep': sweep(recs, recs, [0], [1])}
        md.append(f"| iWildCam H v2 (dev screen) | stream | 0→1 | {m['regret_kga']:.4f} | "
                  f"{m['regret_adapt']:.4f} | {m['regret_freeze']:.4f} | {m['false_adapt']:.3f} | {verdict(m)['noharm_tol']} |")
    iwt = maybe('experiments/kbound/results/iwildcam_full_test/result_e40faf29.json')
    if iwt:
        recs, _ = A.load_records(iwt, candidate='tent_episodic')
        m = split_metrics(recs, recs, [0], [1], rs=0)
        iw['heldout_seed_split'] = {**m, 'verdict': verdict(m),
                                    'decision_seed_sweep': sweep(recs, recs, [0], [1])}
        md.append(f"| iWildCam H v2 (HELD-OUT) | stream | 0→1 | {m['regret_kga']:.4f} | "
                  f"{m['regret_adapt']:.4f} | {m['regret_freeze']:.4f} | {m['false_adapt']:.3f} | {verdict(m)['noharm_tol']} |")
    else:
        iw['heldout_seed_split'] = 'SKIPPED (iwildcam_full_test/result_e40faf29.json is an iCloud placeholder — Finder > Download Now, then rerun)'
    R['tracks']['iwildcam_H_v2'] = iw

    with open(os.path.join(args.out, 'natural_seed_robustness_v1.json'), 'w') as f:
        json.dump(R, f, indent=1)
    with open(os.path.join(args.out, 'natural_seed_robustness_v1.md'), 'w') as f:
        f.write('\n'.join(md) + '\n')
    print('\n'.join(md))
    print('\nSaved ->', args.out)

if __name__ == '__main__':
    main()
