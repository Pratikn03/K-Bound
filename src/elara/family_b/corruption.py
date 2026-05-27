"""Phase 2.2B — k-of-D corruption helpers (validation-fold safe).

Wraps the runner's existing ``AdversarialPerturbationEngine`` primitive
to produce reproducible corrupted feature/mask tensors for an
arbitrary split. The validation-fold variant lets RGA-v2 gate-threshold
selection (G1 ``tau_min`` and G3 ``tau_q``) read **only** validation
fold data — never the test fold — matching the locked Phase-2.B
contract rule ``selection_used_test_metrics=False``.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from itertools import combinations

import numpy as np

# The runner already implements the perturbation engine. We re-use it
# verbatim rather than re-implementing.
from uais.fusion.attention.adversarial_robustness import (  # noqa: E402
    AdversarialAttackType,
    AdversarialPerturbationEngine,
)


@dataclass(frozen=True)
class KOfDCorruptionResult:
    """Result of one (attack, k, subset) corruption injection."""

    condition: str
    attack: str
    failed_domains: tuple[str, ...]
    failed_domain_count: int
    features: np.ndarray
    masks: np.ndarray


def inject_corruption(
    features: np.ndarray,
    masks: np.ndarray,
    *,
    domain_order: list[str],
    score_index: int,
    attack_name: str,
    k_values: Iterable[int],
    sigma: float,
    seed: int,
) -> list[KOfDCorruptionResult]:
    """Produce a list of corruption injections for an arbitrary split.

    The function is identical to the runner's
    ``_k_domain_corruption_conditions`` but returns typed records and
    is split-agnostic — callers pass validation or test features as
    appropriate.

    Args:
      features: [N, D, F] feature tensor for the chosen split.
      masks:    [N, D] bool mask, True = domain missing.
      domain_order: list of domain names matching the feature axis.
      score_index: index into the feature axis at which the score lives.
      attack_name: one of ``zero_attack``, ``max_attack``, ``gaussian_noise``,
                   ``missing_domain_failure`` (subject to engine support).
      k_values: iterable of integer k values to apply (0 = clean).
      sigma: noise scale used by gaussian-noise attacks.
      seed: PRNG seed used by the corruption engine. The caller should
            offset this from the training seed so that validation-fold
            and test-fold corruptions are not identical.

    Returns:
      one ``KOfDCorruptionResult`` per (attack, k, subset) cell.
    """
    try:
        attack_type = AdversarialAttackType(attack_name)
    except ValueError as e:
        raise ValueError(f"unknown attack_name {attack_name!r}") from e

    engine = AdversarialPerturbationEngine(domain_order, score_index, random_seed=int(seed))
    out: list[KOfDCorruptionResult] = []
    for k in sorted({int(v) for v in k_values}):
        if k < 0 or k > len(domain_order):
            raise ValueError(f"k {k} outside [0, {len(domain_order)}]")
        if k == 0:
            out.append(
                KOfDCorruptionResult(
                    condition="clean:k0",
                    attack="none",
                    failed_domains=(),
                    failed_domain_count=0,
                    features=features,
                    masks=masks,
                )
            )
            continue
        for subset in combinations(domain_order, k):
            pert_feat = features.copy()
            pert_mask = masks.copy()
            for domain in subset:
                pert_feat, pert_mask = engine.apply_attack(
                    pert_feat,
                    pert_mask,
                    attack_type,
                    target_domain=domain,
                    sigma=sigma,
                )
            out.append(
                KOfDCorruptionResult(
                    condition=f"{attack_name}:k{k}:{','.join(subset)}",
                    attack=attack_name,
                    failed_domains=subset,
                    failed_domain_count=k,
                    features=pert_feat,
                    masks=pert_mask,
                )
            )
    return out


def validation_fold_corruption_grid(
    val_features: np.ndarray,
    val_masks: np.ndarray,
    *,
    domain_order: list[str],
    score_index: int,
    attacks: Iterable[str] = ("zero_attack", "max_attack"),
    k_values: Iterable[int] = (1, 2, 3, 4),
    sigma: float = 1.0,
    base_seed: int = 0,
) -> dict[tuple[str, int], list[KOfDCorruptionResult]]:
    """Produce a grid of validation-fold corrupted tensors for gate
    threshold selection.

    The grid is keyed by (attack, k); each value is the list of
    KOfDCorruptionResult objects (one per failed-domain subset).

    Critically, this function never reads test-fold data — it accepts
    only validation features / masks. Drivers that consume its output
    must produce a selection_log row with
    ``selection_used_test_metrics=False`` for each selection event.
    """
    grid: dict[tuple[str, int], list[KOfDCorruptionResult]] = {}
    for attack in attacks:
        # Offset PRNG by attack id and a base seed so that the same
        # attack with the same validation features is reproducible while
        # different attacks see different perturbation noise streams.
        attack_seed = int(base_seed) + (hash(attack) & 0xFFFF)
        injections = inject_corruption(
            val_features,
            val_masks,
            domain_order=domain_order,
            score_index=score_index,
            attack_name=attack,
            k_values=list(k_values),
            sigma=float(sigma),
            seed=attack_seed,
        )
        for r in injections:
            grid.setdefault((attack, int(r.failed_domain_count)), []).append(r)
    return grid


__all__ = [
    "KOfDCorruptionResult",
    "inject_corruption",
    "validation_fold_corruption_grid",
]
