"""Adversarial perturbation engine for stress-testing fusion robustness.

Applies structured score perturbations to simulate worst-case deployment
conditions: suppressed signals, amplified signals, calibration drift, and
Gaussian noise. All methods return copies — inputs are never mutated.

Also includes ``pgd_attack_subset``: a gradient-aligned PGD attack over an
arbitrary subset of domains (closes reviewer gap re: stronger threat model
than zero/max/Gaussian).
"""

from __future__ import annotations

from collections.abc import Sequence
from enum import Enum

import numpy as np

try:
    import torch

    _HAS_TORCH = True
except ImportError:  # torch is an installed dep; tolerate import failure for type-only
    torch = None  # type: ignore
    _HAS_TORCH = False


class AdversarialAttackType(str, Enum):
    ZERO_ATTACK = "zero_attack"  # Suppress domain signal: score → 0.0
    MAX_ATTACK = "max_attack"  # Amplify domain signal: score → 1.0
    GAUSSIAN_NOISE = "gaussian_noise"  # score += N(0, sigma); clipped to [0,1]
    UNIFORM_NOISE = "uniform_noise"  # score += U(-eps, eps); clipped to [0,1]
    MEAN_SUBSTITUTION = "mean_substitution"  # Replace score with training mean


class AdversarialPerturbationEngine:
    """Generates perturbed feature tensors for adversarial robustness evaluation.

    All public methods return a new features array (copy of input) — the
    original is never modified. Masks are returned unchanged since attacks
    corrupt values, not missingness.
    """

    def __init__(
        self,
        domain_order: list[str],
        score_index: int,
        random_seed: int = 42,
    ) -> None:
        self.domain_order = list(domain_order)
        self.score_index = score_index
        self.rng = np.random.default_rng(random_seed)

    def _domain_indices(self, target_domain: str | None) -> list[int]:
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
        target_domain: str | None = None,
        sigma: float = 0.1,
        eps: float = 0.1,
        reference_mean: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
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
        noise_levels: list[float],
        drift_type: str = "gaussian",
        shift_amount: float = 0.1,
        scale_factor: float = 1.2,
    ) -> dict[float, np.ndarray]:
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
        result: dict[float, np.ndarray] = {}

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
        sigma_values: list[float],
        target_domain: str | None = None,
    ) -> dict[float, tuple[np.ndarray, np.ndarray]]:
        """Convenience wrapper: apply GAUSSIAN_NOISE at multiple sigma levels."""
        return {
            sigma: self.apply_attack(
                features,
                masks,
                AdversarialAttackType.GAUSSIAN_NOISE,
                target_domain=target_domain,
                sigma=sigma,
            )
            for sigma in sigma_values
        }

    def pgd_attack_subset(
        self,
        model,
        features: np.ndarray,
        masks: np.ndarray,
        labels: np.ndarray,
        target_domains: Sequence[str],
        epsilon: float = 0.1,
        step_size: float = 0.02,
        n_steps: int = 10,
        device=None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Gradient-aligned PGD attack over a subset of domains.

        Perturbs only the score channel for the named domain subset, bounded
        in an L_inf ball of radius ``epsilon`` around the original scores, by
        following the sign of the loss gradient w.r.t. the score input. This
        is the realistic threat model reviewers ask for: an attacker who has
        compromised a strict subset of domain experts (not all of them).

        Parameters
        ----------
        model : nn.Module with forward(features, key_padding_mask=...) →
                (logits, *, *) and supporting gradient flow through features.
        features : [N, D, F] float numpy
        masks    : [N, D] bool numpy (unchanged)
        labels   : [N] {0, 1} numpy — ground truth (used to compute loss
                   direction; for untargeted attacks we always *increase* loss).
        target_domains : iterable of domain names in ``self.domain_order``.
        epsilon : L_inf perturbation budget on the score channel.
        step_size : PGD step magnitude (in score-channel units).
        n_steps : number of PGD iterations.
        device : torch.device — defaults to model's parameter device.

        Returns
        -------
        (perturbed_features, masks) : same shapes as input.

        Notes
        -----
        Only the score channel (``self.score_index``) of each target domain is
        perturbed; other feature channels (embeddings, confidence) are left
        untouched. Perturbation is masked to available (non-missing) entries.
        """
        if not _HAS_TORCH:
            raise RuntimeError("torch is required for pgd_attack_subset.")
        if device is None:
            device = next(model.parameters()).device
        target_idxs = [self.domain_order.index(d) for d in target_domains]
        if not target_idxs:
            return features.copy().astype(np.float32), masks

        feat_t = torch.tensor(features, dtype=torch.float32, device=device)
        mask_t = torch.tensor(masks, dtype=torch.bool, device=device)
        labels_t = torch.tensor(labels.astype(np.float32), device=device)

        orig_scores = feat_t[:, :, self.score_index].clone()
        # Random init within epsilon-ball (skipped for determinism in test)
        delta = torch.zeros_like(orig_scores, requires_grad=False)
        # Only allow perturbation on target domains × available entries
        edit_mask = torch.zeros_like(orig_scores, dtype=torch.bool)
        for d in target_idxs:
            edit_mask[:, d] = ~mask_t[:, d]

        was_training = model.training
        model.eval()
        try:
            for _ in range(n_steps):
                delta.requires_grad_(True)
                perturbed = feat_t.clone()
                new_scores = orig_scores + delta * edit_mask.float()
                new_scores = new_scores.clamp(0.0, 1.0)
                perturbed[:, :, self.score_index] = new_scores
                out = model(perturbed, key_padding_mask=mask_t)
                logits = out[0] if isinstance(out, tuple) else out
                logits = logits.squeeze(-1)
                # Untargeted: maximize BCE loss.
                loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, labels_t, reduction="mean")
                grad = torch.autograd.grad(loss, delta, retain_graph=False)[0]
                with torch.no_grad():
                    delta = delta + step_size * torch.sign(grad) * edit_mask.float()
                    # Project to L_inf ball
                    delta = delta.clamp(-epsilon, epsilon)
        finally:
            if was_training:
                model.train()

        with torch.no_grad():
            adv_scores = (orig_scores + delta * edit_mask.float()).clamp(0.0, 1.0)
            perturbed = feat_t.clone()
            perturbed[:, :, self.score_index] = adv_scores
        return perturbed.detach().cpu().numpy().astype(np.float32), masks


__all__ = ["AdversarialAttackType", "AdversarialPerturbationEngine"]
