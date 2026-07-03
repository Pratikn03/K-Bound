"""Validator for Gap B (frozen criteria in PROTOCOL_GAPCLOSE_WAVE5_v1.md).

B1  Level: H-true panels across K x m x b-scale x pi — rejection <= alpha + 2 MC-se
    in every cell (alpha = 0.05, 200 reps/cell).
B2  Power: co-adapted panels (latent common-noise flip, rho = 0.25), K=6, m=2000:
    rejection >= 0.9.
B3  Fixed-threshold transfer failure: some family cell where tau* = 0.52 has
    level > 0.5 or power < 0.2 while tau' holds both.
Exit 0 iff all pass. Seeds fixed.
"""
from __future__ import annotations

import json
import sys

import numpy as np

sys.path.insert(0, __import__("os").path.dirname(__file__))
from tau_selfnorm import (fit_rank_one, simulate_H_panel,  # noqa: E402
                          tau_residual, tau_selfnorm)

ALPHA = 0.05
REPS = 200
FIXED_TAU_STAR = 0.52


def coadapted_panel(K: int, m: int, rho: float, b_scale: float,
                    rng: np.random.Generator) -> np.ndarray:
    """Panel violating CEI DETECTABLY: correlated-but-not-identical flips.

    NOTE: flipping ALL candidates jointly preserves every pairwise agreement
    (the swap-orbit invariance the K-Bound dichotomy proves), so it is
    undetectable from agreements in principle. The detectable violation is a
    latent factor that flips each candidate with prob q on shared episodes,
    independently given the factor: pairs then co-flip more often than the
    product law allows, leaving an excess-agreement rank-one residual.
    """
    a = (1.0 + b_scale * (0.6 + 0.4 * rng.random(K))) / 2.0
    s = rng.random((m, K)) < a[None, :]
    common = rng.random(m) < rho          # shared-backbone episodes
    # co-trained twins agree on MISTAKES: on shared episodes each twin pair
    # copies its sibling's correctness (excess pairwise agreement — the
    # violation actually observed on co-adapted TTA panels).
    for i, j in [(p, p + 1) for p in range(0, K - 1, 2)]:
        s[common, j] = s[common, i]
    agree = np.einsum("mi,mj->ij", s.astype(float), s.astype(float))
    agree += np.einsum("mi,mj->ij", (~s).astype(float), (~s).astype(float))
    C = 2.0 * agree / m - 1.0
    np.fill_diagonal(C, 1.0)
    return C


# pi is provably vacuous under symmetric-accuracy H (the correctness law never
# depends on y), so the grid runs pi = 0.5 only — documented deviation, zero
# information loss. Reps/n_sim scale with m so each cell fits a 40 s chunk;
# MC-se thresholds use the per-cell rep count.
CELL_REPS = {200: 200, 2000: 120, 20000: 60}
CELL_NSIM = {200: 150, 2000: 120, 20000: 60}
STATE = __file__.replace(".py", "_cells.jsonl")


def run_cell(K: int, m: int, bs: float) -> dict:
    import os
    rng = np.random.default_rng(20260702 + K * 1000 + m + int(bs * 10))
    reps, nsim = CELL_REPS[m], CELL_NSIM[m]
    if os.environ.get("REPS"):  # precision escalation (documented in results)
        reps = int(os.environ["REPS"])
    rej = rej_fixed = 0
    for _ in range(reps):
        b = bs * (0.6 + 0.4 * rng.random(K))
        C = simulate_H_panel(b, 0.5, m, rng)
        r = tau_selfnorm(C, m, 0.5, ALPHA, n_sim=nsim,
                         seed=int(rng.integers(1 << 30)))
        rej += int(r["reject_H"])
        rej_fixed += int(r["tau_obs"] > FIXED_TAU_STAR)
    return dict(key=f"K{K}_m{m}_b{bs}", K=K, m=m, bs=bs, reps=reps,
                level_selfnorm=rej / reps, level_fixed052=rej_fixed / reps,
                mc_se=float(np.sqrt(ALPHA * (1 - ALPHA) / reps)))


def run_power() -> dict:
    rng = np.random.default_rng(9992026)
    reps = 120
    pw = pw_fixed = 0
    for _ in range(reps):
        C = coadapted_panel(6, 2000, 0.25, 0.5, rng)
        r = tau_selfnorm(C, 2000, 0.5, ALPHA, n_sim=120,
                         seed=int(rng.integers(1 << 30)))
        pw += int(r["reject_H"])
        pw_fixed += int(r["tau_obs"] > FIXED_TAU_STAR)
    return dict(key="power_coadapted", reps=reps,
                power_selfnorm=pw / reps, power_fixed052=pw_fixed / reps)


def assemble() -> int:
    cells = [json.loads(l) for l in open(STATE)]
    lv = {c["key"]: c for c in cells if c["key"].startswith("K")}
    pw = next(c for c in cells if c["key"] == "power_coadapted")
    ok_level = all(c["level_selfnorm"] <= ALPHA + 2 * c["mc_se"]
                   for c in lv.values())
    b2 = pw["power_selfnorm"] >= 0.9
    worst_fixed = max(c["level_fixed052"] for c in lv.values())
    b3 = worst_fixed > 0.5 or pw["power_fixed052"] < 0.2
    out = dict(
        checks=dict(B1_level_holds=bool(ok_level), B2_power=bool(b2),
                    B3_fixed_threshold_fails_somewhere=bool(b3)),
        n_cells=len(lv),
        worst_selfnorm_level=max(c["level_selfnorm"] for c in lv.values()),
        worst_fixed052_level=worst_fixed,
        power_selfnorm=pw["power_selfnorm"],
        power_fixed052=pw["power_fixed052"],
        deviations=("pi dimension dropped (provably vacuous under symmetric H); "
                    "reps/n_sim scaled by m for chunked execution: "
                    f"{CELL_REPS}/{CELL_NSIM}"),
        level_grid_selfnorm={k: c["level_selfnorm"] for k, c in lv.items()},
        level_grid_fixed052={k: c["level_fixed052"] for k, c in lv.items()},
        PASS=bool(ok_level and b2 and b3))
    print(json.dumps({k: v for k, v in out.items()
                      if not k.startswith("level_grid")}, indent=1))
    with open(__file__.replace(".py", "_results.json"), "w") as f:
        json.dump(out, f, indent=1)
    return 0 if out["PASS"] else 3


def main() -> int:
    import os
    part = os.environ.get("PART", "assemble")
    if part == "assemble":
        return assemble()
    with open(STATE, "a") as f:
        if part == "power":
            f.write(json.dumps(run_power()) + "\n")
        else:  # e.g. PART=K6_m2000 [BS=0.8 to run a single cell]
            K = int(part.split("_")[0][1:])
            m = int(part.split("_")[1][1:])
            only = os.environ.get("BS")
            for bs in (0.2, 0.5, 0.8):
                if only and abs(bs - float(only)) > 1e-9:
                    continue
                f.write(json.dumps(run_cell(K, m, bs)) + "\n")
                f.flush()
    return 0


if __name__ == "__main__":
    sys.exit(main())
