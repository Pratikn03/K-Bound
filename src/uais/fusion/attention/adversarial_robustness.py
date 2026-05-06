"""Adversarial perturbation engine for stress-testing fusion robustness.

Applies structured score perturbations to simulate worst-case deployment
conditions: suppressed signals, amplified signals, calibration drift, and
Gaussian noise. All methods return copies — inputs are never mutated.
"""

from __future__ import annotations

from enum import Enum
from typing import Dict, List, Optional, Tuple

import numpy as np


class AdversarialAttackType(str, Enum):
    ZERO_ATTACK = "zero_attack"             # Suppress domain signal: score → 0.0
    MAX_ATTACK = "max_attack"               # Amplify domain signal: score → 1.0
    GAUSSIAN_NOISE = "gaussian_noise"      # score += N(0, sigma); clipped to [0,1]
    UNIFORM_NOISE = "uniform_noise"        # score += U(-eps, eps); clipped to [0,1]
    MEAN_SUBSTITUTION = "mean_substitution"  # Replace score with training mean


class AdversarialPerturbationEngine:
    """Generates perturbed feature tensors for adversarial robustness evaluation.

    All public methods return a new features array (copy of input) — the
    original is never modified. Masks are returned unchanged since attacks
    corrupt values, not missingness.
    """

    def __init__(
        self,
        domain_order: List[str],
        score_index: int,
        random_seed: int = 42,
    ) -> None:
        self.domain_order = list(domain_order)
        self.score_index = score_index
        self.rng = np.random.default_rng(random_seed)

    def _domain_indices(self, target_domain: Optional[str]) -> List[int]:
        if target_domain is None:
            return list(range(len(self.domain_order)))
        if target_domain not in self.domain_order:
            raise ValueError(f"Unknown domain '{target_domain}'. Known: {self.domain_order}")
        return [self.domain_order.index(target_domain)]

    def apply_attack(
        self,
        features: np.ndarray,
        masks: np.ndarray,
        attack_type: AdversarialAttackType,
        target_domain: Optional[str] = None,
        sigma: float = 0.1,
        eps: float = 0.1,
        reference_mean: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Apply a structured attack to domain scores.

        Parameters
        ----------
        features      : [N, D, F] — not mutated
        masks         : [N, D] bool — returned unchanged
        attack_type   : which attack to apply
        target_domain : domain name to attack, or None for all available domains
        sigma         : std dev for GAUSSIAN_NOISE
        eps           : half-range for UNIFORM_NOISE
        reference_mean: [D] training-set mean scores for MEAN_SUBSTITUTION
        """
        perturbed = features.copy().astype(np.float32)
        domain_idxs = self._domain_indices(target_domain)

        for d in domain_idxs:
            available = ~masks[:, d]
            if not available.any():
                continue

            scores = perturbed[available, d, self.score_index]

            if attack_type == AdversarialAttackType.ZERO_ATTACK:
                scores = np.zeros_like(scores)

            elif attack_type == AdversarialAttackType.MAX_ATTACK:
                scores = np.ones_like(scores)

            elif attack_type == AdversarialAttackType.GAUSSIAN_NOISE:
                noise = self.rng.normal(0.0, sigma, size=scores.shape).astype(np.float32)
                scores = np.clip(scores + noise, 0.0, 1.0)

            elif attack_type == AdversarialAttackType.UNIFORM_NOISE:
                noise = self.rng.uniform(-eps, eps, size=scores.shape).astype(np.float32)
                scores = np.clip(scores + noise, 0.0, 1.0)

            elif attack_type == AdversarialAttackType.MEAN_SUBSTITUTION:
                if reference_mean is None:
                    raise ValueError("reference_mean required for MEAN_SUBSTITUTION attack")
                scores = np.full_like(scores, float(reference_mean[d]))

            perturbed[available, d, self.score_index] = scores

        return perturbed, masks

    def simulate_domain_drift(
        self,
        features: np.ndarray,
        masks: np.ndarray,
        target_domain: str,
        noise_levels: List[float],
        drift_type: str = "gaussian",
        shift_amount: float = 0.1,
        scale_factor: float = 1.2,
    ) -> Dict[float, np.ndarray]:
        """Sweep domain scores across a range of drift intensities.

        Returns a dict mapping each noise level to its perturbed features array.
        Used to generate the performance-vs-drift curves that are the core
        evidence for the CRAF paper (Table 2 / Figure 1).

        drift_type options
        ------------------
        "gaussian" : add N(0, level) noise — simulates random sensor degradation
        "shift"    : add constant `level * shift_amount` bias — simulates score recalibration drift
        "scale"    : multiply by `1 + level * (scale_factor - 1)` — simulates magnitude drift
        """
        d = self.domain_order.index(target_domain)
        result: Dict[float, np.ndarray] = {}

        for level in noise_levels:
            perturbed = features.copy().astype(np.float32)
            available = ~masks[:, d]
            if not available.any():
                result[level] = perturbed
                continue

            scores = perturbed[available, d, self.score_index]

            if drift_type == "gaussian":
                if level == 0.0:
                    pass  # no perturbation at level 0
                else:
                    noise = self.rng.normal(0.0, level, size=scores.shape).astype(np.float32)
                    scores = np.clip(scores + noise, 0.0, 1.0)

            elif drift_type == "shift":
                scores = np.clip(scores + level * shift_amount, 0.0, 1.0)

            elif drift_type == "scale":
                factor = 1.0 + level * (scale_factor - 1.0)
                scores = np.clip(scores * factor, 0.0, 1.0)

            else:
                raise ValueError(f"Unknown drift_type '{drift_type}'. Use 'gaussian', 'shift', or 'scale'.")

            perturbed[available, d, self.score_index] = scores
            result[level] = perturbed

        return result

    def sweep_gaussian_noise(
        self,
        features: np.ndarray,
        masks: np.ndarray,
        sigma_values: List[float],
        target_domain: Optional[str] = None,
    ) -> Dict[float, Tuple[np.ndarray, np.ndarray]]:
        """Convenience wrapper: apply GAUSSIAN_NOISE at multiple sigma levels."""
        return {
            sigma: self.apply_attack(
                features, masks, AdversarialAttackType.GAUSSIAN_NOISE,
                target_domain=target_domain, sigma=sigma,
            )
            for sigma in sigma_values
        }


__all__ = ["AdversarialAttackType", "AdversarialPerturbationEngine"]
