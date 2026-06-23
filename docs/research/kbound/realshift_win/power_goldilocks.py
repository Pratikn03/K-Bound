#!/usr/bin/env python3
"""
Power / feasibility for a CI-robust *pure-label-free* real-shift beats-both.

Uses the LOCKED verifier (no target tuning). Sweeps the two levers that decide whether the
win is reachable:
  - source->OOD detector TRANSFER (the constraint that killed iWildCam / Office-Home:
    the label-free harm signal was sharp in-source but did not carry to OOD)
  - n = number of held-out conditions/seeds logged (drives whether the bootstrap CI excludes 0)
at fixed realistic two-sided mixedness (p_harm) and a good in-source detector (det_noise small).

Output = a phase diagram of P(CI-robust beats-both): the minimum data quality you must hit on
the GPU run for the win to be real. Nothing here is a win on real data; it sizes the requirement.
"""
import numpy as np, json
from verify_realshift_win import make_regime, verify

def pwin(transfer, n, p_harm=0.35, det_noise=0.03, nsim=150, nboot=1200, base=0):
    cir = pte = 0.0
    for s in range(nsim):
        Zc, Bc, Zt, a0, aa = make_regime('goldilocks', n=n, seed=base + s,
                                         det_noise=det_noise, transfer=transfer, p_harm=p_harm)
        r = verify(Zc, Bc, Zt, a0, aa, nboot=nboot, seed=10000 + s)
        cir += r['beats_both_CI_robust']; pte += r['beats_both_point']
    return pte / nsim, cir / nsim

if __name__ == '__main__':
    transfers = [0.3, 0.5, 0.7, 0.9, 1.0]
    ns = [60, 120, 240, 480]
    print("P(CI-robust beats-both)  —  rows = source->OOD detector transfer, cols = n conditions")
    print("(two-sided mixedness p_harm=0.35, good in-source detector det_noise=0.03)\n")
    print("transfer \\ n   " + "  ".join(f"{n:>5d}" for n in ns))
    grid = {}
    for t in transfers:
        row = []
        for n in ns:
            _, c = pwin(t, n); row.append(c); grid[f"transfer{t}_n{n}"] = round(c, 3)
        print(f"   {t:>4.1f}        " + "  ".join(f"{100*c:4.0f}%" for c in row))

    print("\nMixedness sensitivity (transfer=0.9, n=240, det_noise=0.03):")
    print("  p_harm:   " + "  ".join(f"{p:>5.2f}" for p in [0.05, 0.15, 0.30, 0.45, 0.60]))
    mrow = []
    for p in [0.05, 0.15, 0.30, 0.45, 0.60]:
        _, c = pwin(0.9, 240, p_harm=p); mrow.append(c); grid[f"pharm{p}_t0.9_n240"] = round(c, 3)
    print("  P(CIrob): " + "  ".join(f"{100*c:4.0f}%" for c in mrow))

    json.dump(grid, open('power_goldilocks.json', 'w'), indent=2)
    print("\nsaved power_goldilocks.json")
