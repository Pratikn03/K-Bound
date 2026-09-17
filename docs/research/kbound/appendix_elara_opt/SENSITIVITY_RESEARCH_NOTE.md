# Exploratory Research Note: Sensitivity Extension and Break-Even Odds-Ratio Analysis

**Status:** Non-Submission Exploratory Research Note (Preserved per R3).

## 1. Mathematical Distinction
The established deductive population frontier for K-Bound is:
$$\Delta(\mathcal{C}_\beta) = 2d \left[\max\left(-\frac{1}{2}, M - \beta\right),\, \min\left(\frac{1}{2}, M + \beta\right)\right],$$
where $d = \mu_T(D) > 0$ is the disagreement probability mass, $M$ is the observable disagreement margin, and $\beta \in [0, 1/2]$ is the additive residual threshold on $D$.

The additive residual break-even threshold:
$$\beta^*_{\mathrm{add}} = \max\left(0,\, \frac{|\hat\Delta_0| - \varepsilon}{2d}\right)$$
is conceptually and algebraically distinct from an exponential selection odds-ratio curve:
$$\Gamma^*(\Delta) = \exp(2|\Delta|)$$
modeled after Rosenbaum's sensitivity analysis in observational causal inference. The two quantities must not be equated or conflated.

## 2. Plotted Benchmark Quantities vs. Population Benefit
In exploratory plots (previously Figure 3 / `fig_sensitivity_frontier.pdf`), empirical policy gains from finite-sample runs (such as empirical regret difference on CCT-20, $\Delta = 0.1819$, or CIFAR-10-C Tent, $\Delta = 0.0123$) were used as illustrative inputs. However:
1. Empirical policy-comparison regret differences between KGA and baselines on evaluation cells are *not* the nominal population benefit $\Delta = R_T(f_0) - R_T(f_a)$ required by the theoretical frontier.
2. An empirical certificate envelope $[\hat\Delta_0 \pm \varepsilon]$ is not an exact guarantee against benefit reversal under arbitrary non-exchangeable shifts.

## 3. Preservation
Per R3, this operational sensitivity extension and its associated continuous plots (`fig_sensitivity_frontier.pdf`, Corollary 2, and Equation 8) are preserved here in this research note outside the maintained submission manuscript. The maintained manuscript retains the exact, established population frontier $\Delta(\mathcal{C}_\beta)$ and the Le Cam audit floor.
