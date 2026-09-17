# Exploratory Research Note: Active-Entropy Multiclass Calibration

**Status:** Exploratory / Unproved Theoretical Sketch (Separated from Main Manuscript per R4).

## Context
In multiclass problems with $K$ classes, a union bound over class-conditional errors suggests an uncertainty radius scaling with $2(K-1)$, which rapidly widens intervals for large label alphabets.

## Conjecture
When the deployed model makes confident predictions, the conditional distribution over labels concentrates on an effective class cardinality $\exp(H_C(X)) \ll K$, where:
$$H_C(X) = -\sum_{k=1}^K p_k(X) \log p_k(X)$$
is the prediction entropy.

Under an entropy-constrained active support condition, the finite-sample calibration radius was conjectured to satisfy:
$$\varepsilon_{\mathrm{multi}}(n, \alpha, H_C) = \mathcal{O}\left(\sqrt{\frac{\min(K, \exp(H_C))\,\log(K/\alpha)}{n}}\right).$$

When confidence is high ($H_C \to 0$), this contracts toward a binary-scale $\mathcal{O}(\sqrt{\log(K/\alpha)/n})$ rate.

## Audit Finding (R4)
Because explicit validity conditions, proofs, and finite-sample exchangeability definitions for this bound were not established, this formulation is preserved outside the maintained claims of the main manuscript.
