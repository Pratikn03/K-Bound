# Table 7 Confidence Intervals — K-Bound vs Baselines

**CI Source**: Pareto-bootstrap distribution from `decisive_tta_results.json`.  
Per-condition a0/aa arrays are **not** stored in the JSON; CIs are derived from  
the 11-point mixing-ratio Pareto curve (each point is the mean over 200 bootstrap  
resamples of a 200-condition synthetic stream), resampled N=5000 times.  
This spans the full operating range of harmful fractions — CIs are consequently wide.  
A future re-run that serialises per-condition arrays would allow tighter paired t-tests.

## Summary Table

| Method | r(KBound) | r(adapt) | r(freeze) | Δ vs adapt [95%CI] | p | d | Δ vs freeze [95%CI] | p | d |
|--------|-----------|----------|-----------|---------------------|---|---|----------------------|---|---|
| cifar10c/tent | 0.00160 | 0.00859 | 0.12319 | -0.01863 [-0.02642, -0.01123] | 0.0015 | -1.417 | -0.07981 [-0.10666, -0.05183] | 0.0005 | -1.685 |
| cifar10c/eata | 0.00148 | 0.00367 | 0.13106 | -0.01490 [-0.02094, -0.00911] | 0.0013 | -1.454 | -0.07484 [-0.10010, -0.04866] | 0.0005 | -1.684 |
| cifar10c/sar | 0.00178 | 0.00176 | 0.13715 | -0.03426 [-0.04881, -0.02036] | 0.0017 | -1.392 | -0.07439 [-0.09948, -0.04836] | 0.0005 | -1.687 |

Δ = regret(K-Bound) − regret(baseline).  Negative Δ = K-Bound is better.
Bootstrap N=5000, seed=42, α=0.05 (two-sided).

## Detailed Results

### cifar10c/tent
CI method: pareto-bootstrap: CIs derived from the stored mixing-ratio Pareto curve (11 p-values x bootstrap-averaged regret), resampled with N_boot=5000. CIs span the full operating range of harmful fractions, not a single point.

- Operational regret — K-Bound: 0.001597
- Operational regret — always-adapt: 0.008594
- Operational regret — always-freeze: 0.123192
- Δ(KBound−adapt) operational: -0.006997
- Δ(KBound−freeze) operational: -0.121595

**vs always-adapt** (pareto-range mean):
  mean Δ = -0.018630, 95%CI = [-0.026421, -0.011225]
  t = -4.4825, p = 0.001527, Cohen's d = -1.4175

**vs always-freeze** (pareto-range mean):
  mean Δ = -0.079815, 95%CI = [-0.106665, -0.051834]
  t = -5.3279, p = 0.000476, Cohen's d = -1.6848

### cifar10c/eata
CI method: pareto-bootstrap: CIs derived from the stored mixing-ratio Pareto curve (11 p-values x bootstrap-averaged regret), resampled with N_boot=5000. CIs span the full operating range of harmful fractions, not a single point.

- Operational regret — K-Bound: 0.001484
- Operational regret — always-adapt: 0.003666
- Operational regret — always-freeze: 0.131061
- Δ(KBound−adapt) operational: -0.002182
- Δ(KBound−freeze) operational: -0.129578

**vs always-adapt** (pareto-range mean):
  mean Δ = -0.014896, 95%CI = [-0.020945, -0.009108]
  t = -4.5979, p = 0.001294, Cohen's d = -1.454

**vs always-freeze** (pareto-range mean):
  mean Δ = -0.074840, 95%CI = [-0.100099, -0.048665]
  t = -5.3241, p = 0.000478, Cohen's d = -1.6836

### cifar10c/sar
CI method: pareto-bootstrap: CIs derived from the stored mixing-ratio Pareto curve (11 p-values x bootstrap-averaged regret), resampled with N_boot=5000. CIs span the full operating range of harmful fractions, not a single point.

- Operational regret — K-Bound: 0.001781
- Operational regret — always-adapt: 0.001759
- Operational regret — always-freeze: 0.137154
- Δ(KBound−adapt) operational: 2.2e-05
- Δ(KBound−freeze) operational: -0.135373

**vs always-adapt** (pareto-range mean):
  mean Δ = -0.034265, 95%CI = [-0.048807, -0.020356]
  t = -4.4014, p = 0.001717, Cohen's d = -1.3918

**vs always-freeze** (pareto-range mean):
  mean Δ = -0.074390, 95%CI = [-0.099479, -0.048360]
  t = -5.334, p = 0.000472, Cohen's d = -1.6868

