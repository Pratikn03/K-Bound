#!/usr/bin/env python3
"""experiments.kbound.nonoracle_population_diagnostic -- Non-oracle population benefit simulation.

Fits an imperfect predictor on development episodes subject to estimation error,
then evaluates conformal calibration and finite-sample concentration:
    * Predictor fitted on development episodes (slope and intercept with estimation error)
    * Compares:
          1. Point rule: sign(delta_hat)
          2. Cell interval: [delta_hat +/- epsilon_cell]
          3. Population interval: [delta_hat +/- (epsilon_cell + b_samp)]
    * Evaluates exact-truth population coverage, cell coverage, directional commitment errors,
      and commitment cost (abstention rate).
"""

from __future__ import annotations

import json
import math
from pathlib import Path
import numpy as np

from kga.policy import Decision
from kga.population_transfer import (
    compose_conditional_population_interval,
    hoeffding_paired_accuracy_radius,
)


def get_analytic_delta(z: int) -> float:
    return 0.6 * float(z)


def get_prob_y1(z: int) -> float:
    if z == -1:
        return 0.2
    elif z == 0:
        return 0.5
    else:
        return 0.8


def sample_episode(rng: np.random.Generator, m: int) -> tuple[int, float, float]:
    z = int(rng.choice([-1, 0, 1]))
    p = get_prob_y1(z)
    y = rng.binomial(1, p, size=m)
    w = 2 * y - 1
    b_meas = float(np.mean(w))
    delta_pop = get_analytic_delta(z)
    return z, b_meas, delta_pop


def run_diagnostic(
    *,
    n_reps: int = 1000,
    cal_counts: tuple[int, ...] = (18, 19, 39, 99),
    sample_sizes: tuple[int, ...] = (32, 128, 512, 2048),
    alpha_cell: float = 0.05,
    delta_sampling: float = 0.05,
    alpha_population: float = 0.10,
    seed: int = 42,
) -> dict:
    rng = np.random.default_rng(seed)
    summary_records = []

    for N in cal_counts:
        k_rank = math.ceil((N + 1) * (1.0 - alpha_cell))
        is_finite_rank = (k_rank <= N)

        for m in sample_sizes:
            b_samp = hoeffding_paired_accuracy_radius(n=m, delta=delta_sampling)

            cell_cov_count = 0
            pop_cov_count = 0

            point_wrong_dir = 0
            cell_wrong_dir = 0
            pop_wrong_dir = 0

            point_abstain = 0
            cell_abstain = 0
            pop_abstain = 0

            for _ in range(n_reps):
                # 1. Fit imperfect predictor on 15 independent development episodes (m=32)
                dev_z, dev_b = [], []
                for _ in range(15):
                    z_d, b_d, _ = sample_episode(rng, 32)
                    dev_z.append(z_d)
                    dev_b.append(b_d)
                A = np.vstack([dev_z, np.ones(len(dev_z))]).T
                slope, ic = np.linalg.lstsq(A, dev_b, rcond=None)[0]

                def predict_delta(z: int) -> float:
                    return float(slope * z + ic)

                # 2. Draw N calibration episodes
                cal_residuals = []
                for _ in range(N):
                    z_c, b_c, _ = sample_episode(rng, m)
                    cal_residuals.append(abs(predict_delta(z_c) - b_c))
                cal_residuals.sort()

                if is_finite_rank:
                    eps_cell = float(cal_residuals[k_rank - 1])
                else:
                    eps_cell = float("inf")

                # 3. Fresh evaluation episode
                z_t, b_t, delta_t = sample_episode(rng, m)
                dhat_t = predict_delta(z_t)

                # Point rule
                if dhat_t > 0:
                    pt_act = Decision.ADAPT
                elif dhat_t < 0:
                    pt_act = Decision.FREEZE
                else:
                    pt_act = Decision.ABSTAIN

                if pt_act is Decision.ABSTAIN:
                    point_abstain += 1
                if (pt_act is Decision.ADAPT and delta_t <= 0) or (pt_act is Decision.FREEZE and delta_t >= 0):
                    point_wrong_dir += 1

                # Cell interval
                if not math.isinf(eps_cell):
                    if dhat_t - eps_cell > 0:
                        c_act = Decision.ADAPT
                    elif dhat_t + eps_cell < 0:
                        c_act = Decision.FREEZE
                    else:
                        c_act = Decision.ABSTAIN
                    if abs(dhat_t - b_t) <= eps_cell:
                        cell_cov_count += 1
                else:
                    c_act = Decision.ABSTAIN
                    cell_cov_count += 1

                if c_act is Decision.ABSTAIN:
                    cell_abstain += 1
                if (c_act is Decision.ADAPT and delta_t <= 0) or (c_act is Decision.FREEZE and delta_t >= 0):
                    cell_wrong_dir += 1

                # Population interval
                pop_int = compose_conditional_population_interval(
                    delta_hat=dhat_t,
                    epsilon=eps_cell,
                    r_samp=b_samp,
                    alpha_cell=alpha_cell,
                    delta_sampling=delta_sampling,
                    alpha_population=alpha_population,
                )
                pop_act = pop_int.action
                if pop_act is Decision.ABSTAIN:
                    pop_abstain += 1
                if (pop_act is Decision.ADAPT and delta_t <= 0) or (pop_act is Decision.FREEZE and delta_t >= 0):
                    pop_wrong_dir += 1
                if abs(dhat_t - delta_t) <= pop_int.population_radius:
                    pop_cov_count += 1

            record = {
                "N_cal": N,
                "m_eval": m,
                "k_rank": k_rank,
                "is_finite_rank": is_finite_rank,
                "b_samp": round(b_samp, 4),
                "cell_coverage": round(cell_cov_count / n_reps, 4),
                "pop_coverage": round(pop_cov_count / n_reps, 4),
                "point_wrong_dir_rate": round(point_wrong_dir / n_reps, 4),
                "cell_wrong_dir_rate": round(cell_wrong_dir / n_reps, 4),
                "pop_wrong_dir_rate": round(pop_wrong_dir / n_reps, 4),
                "point_abstain_rate": round(point_abstain / n_reps, 4),
                "cell_abstain_rate": round(cell_abstain / n_reps, 4),
                "pop_abstain_rate": round(pop_abstain / n_reps, 4),
            }
            summary_records.append(record)
            print(
                f"N={N:2d}, m={m:4d} | CellCov={record['cell_coverage']:.3f}, PopCov={record['pop_coverage']:.3f} | "
                f"WrongDir: Pt={record['point_wrong_dir_rate']:.3f}, Cell={record['cell_wrong_dir_rate']:.3f}, Pop={record['pop_wrong_dir_rate']:.3f} | "
                f"Abstain: Pt={record['point_abstain_rate']:.3f}, Cell={record['cell_abstain_rate']:.3f}, Pop={record['pop_abstain_rate']:.3f}"
            )

    return {
        "diagnostic_id": "NONORACLE-POPULATION-DIAGNOSTIC-V1",
        "alpha_cell": alpha_cell,
        "delta_sampling": delta_sampling,
        "alpha_population": alpha_population,
        "n_reps": n_reps,
        "records": summary_records,
    }


def main() -> None:
    repo = Path(__file__).resolve().parents[2]
    res = run_diagnostic(n_reps=1000)
    out_path = repo / "experiments/kbound/results/synthetic_diagnostic/nonoracle_diagnostic_summary.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(res, indent=2) + "\n", encoding="utf-8")
    print(f"\nWrote non-oracle diagnostic results to {out_path}")


if __name__ == "__main__":
    main()
