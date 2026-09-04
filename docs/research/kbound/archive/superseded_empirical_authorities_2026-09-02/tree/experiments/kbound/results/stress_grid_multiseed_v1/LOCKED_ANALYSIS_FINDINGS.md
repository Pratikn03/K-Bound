# LOCKED Protocol-A Findings — CIFAR-10-C stress grid, 5-seed re-run

Frozen plan `research_lock/STRESS_GRID_MULTISEED_PROTOCOL_A_v1.yaml` (registered 2026-06-11), executed exactly.
432 conditions x 5 seeds (0-4). Regret = oracle_acc - policy_acc; oracle = max(freeze a0, adapt a_adapted);
KGA ABSTAIN/FREEZE -> freeze (verified vs seed0 summary). Regret pooled per condition across seeds; primary =
per-condition PAIRED bootstrap (10^4) on mean diff, Holm over 6 comparisons (alpha=0.05). This runner's KGA is
the split-conformal route (b_hat(Z)+eps); c_ij/tau_hat/n_D are explicit nulls by design.

## VERDICT: STANDS  — tent AND eata each beat BOTH trivials after Holm. The printed "beats both" claim stands.

| candidate | vs | KGA reg | trivial reg | diff | 95% CI | Holm p | beats? |
|---|---|--:|--:|--:|:--:|--:|:--:|
| tent | always-adapt  | 0.00139 | 0.00774 | -0.00635 | [-0.00787, -0.00489] | 6.0e-4 | YES |
| tent | always-freeze | 0.00139 | 0.12391 | -0.12252 | [-0.13751, -0.10806] | 6.0e-4 | YES |
| eata | always-adapt  | 0.00127 | 0.00327 | -0.00200 | [-0.00278, -0.00125] | 6.0e-4 | YES |
| eata | always-freeze | 0.00127 | 0.13138 | -0.13011 | [-0.14510, -0.11526] | 6.0e-4 | YES |
| sar  | always-adapt  | 0.00152 | 0.00031 | +0.00121 | [+0.00094, +0.00148] | 6.0e-4 | NO (tie) |
| sar  | always-freeze | 0.00152 | 0.14048 | -0.13894 | [-0.15449, -0.12410] | 6.0e-4 | YES |

Between-seed regret variance is tiny (per-condition mean ~1e-5 to 4e-5; seed-to-seed SD of grand-mean ~2e-4).

## Secondary
- False-adapt (ADAPT & B<=0) pooled over 5 seeds: 0/2160 every candidate = 0.000, << alpha=0.10. Coverage
  (KGA decisive action == oracle action): tent 0.999, eata/sar 1.000.
- Harmful base rate (B<0) per-seed range: tent [0.317,0.338], eata [0.234,0.269], sar [0.074,0.116].
- eps_conformal across seeds: tent [0.0199,0.0214] CV 2.7%; eata [0.0150,0.0181] CV 6.8%; sar [0.0124,0.0137] CV 3.8% — stable.

## SAR scope & p* law
SAR ties always-adapt (+0.0012; CI off zero in the *worse* direction by ~1e-3) — pre-stated: its harmful rate
(~0.07-0.12) is at/below p*=0.1, so adapting is almost always safe and KGA's abstention cost isn't repaid.
p* law CONFIRMED: per-seed beats-both is monotone in harmful fraction, single-threshold separable — every
non-beating case (SAR) <=0.116, every beating case (eata,tent) >=0.234; threshold band [0.116,0.234] around p*~0.1.
