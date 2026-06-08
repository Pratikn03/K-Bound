"""Numerical validation of K-Bound Theorem 2 (plug-in regret decomposition).

Statement under test (the EXACT identity):

    R(ghat) - R(g*)  =  E[ |Delta(Z)| * 1{ sign(Deltahat(Z)) != sign(Delta(Z)) } ]

where, for a gate g: Z -> {adapt, freeze} (encoded 1 / 0),

    R(g) = E[ chosen-action loss ]
         = E[ g(Z) * ell(f_a(X), Y) + (1 - g(Z)) * ell(f_0(X), Y) ],

the Bayes-optimal gate is  g*(z) = 1[Delta(z) > 0]  with
    Delta(z) = E[ ell(f_0(X), Y) - ell(f_a(X), Y) | Z = z ],
and the plug-in gate is  ghat(z) = 1[Deltahat(z) > 0]  for a noisy estimate
Deltahat(z) = Delta(z) + noise.

There are two distinct things to confirm, and we report BOTH:

  (A) The identity is EXACT at the level of CONDITIONAL risk. Writing the risk
      with respect to the action-conditional means a_adapt(z) = E[ell(f_a)|Z=z]
      and a_freeze(z) = E[ell(f_0)|Z=z] = a_adapt(z) + Delta(z), the pointwise
      excess of ghat over g* equals |Delta(z)| exactly on sign-mismatch and 0
      otherwise. Averaging the per-z conditional excess (LHS_cond) must equal
      the RHS to MACHINE precision -- there is no Monte-Carlo gap here because
      both sides are deterministic functions of the same sampled (Delta, sign)
      pairs. This is the literal content of the theorem.

  (B) The identity holds for REALIZED losses up to Monte-Carlo error. We draw
      actual per-sample losses ell0_i, ella_i whose conditional means match
      a_freeze(z_i), a_adapt(z_i), form the realized risks of ghat and g* by
      averaging the chosen-action loss, and confirm
      LHS_realized = Rhat(ghat) - Rhat(g*) ~ RHS within a few standard errors.

(A) is the rigorous check (the theorem is an identity, not an inequality);
(B) confirms it survives sampling of the losses themselves, which is how the
quantity would actually be measured.

We sweep several noise levels (including noise = 0, where the gates agree a.e.
and BOTH sides must be ~0, and very large noise, where ghat -> a coin flip and
the regret -> E[|Delta|]/2, matching the minimax corollary).

Run:  python experiments/kbound/theory_validation/val_thm2_regret.py
Pure numpy; prints an LHS-vs-RHS table and a JSON blob, and asserts (A).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict

import numpy as np


# -----------------------------------------------------------------------------
# Core identity helpers
# -----------------------------------------------------------------------------
def bayes_gate(delta: np.ndarray) -> np.ndarray:
    """g*(z) = 1[Delta(z) > 0]  (adapt iff benefit strictly positive)."""
    return (delta > 0.0).astype(np.int64)


def plugin_gate(delta_hat: np.ndarray) -> np.ndarray:
    """ghat(z) = 1[Deltahat(z) > 0]."""
    return (delta_hat > 0.0).astype(np.int64)


def sign_mismatch(delta: np.ndarray, delta_hat: np.ndarray) -> np.ndarray:
    """1{ sign(Deltahat) != sign(Delta) }, comparing the gate DECISIONS.

    The gate thresholds at 0 (adapt iff value > 0), so the decision-relevant
    'sign' is exactly 1[. > 0]. Ties at Delta == 0 contribute |Delta| = 0 to
    the RHS regardless, so any tie convention is immaterial to the identity;
    we use the gate's own >0 convention for both to keep (A) literally exact.
    """
    return (plugin_gate(delta_hat) != bayes_gate(delta)).astype(np.float64)


def rhs_regret(delta: np.ndarray, delta_hat: np.ndarray) -> float:
    """RHS of the theorem: E[ |Delta| * 1{sign mismatch} ] (Monte-Carlo mean)."""
    return float(np.mean(np.abs(delta) * sign_mismatch(delta, delta_hat)))


def conditional_risk(gate: np.ndarray, a_freeze: np.ndarray, a_adapt: np.ndarray) -> float:
    """R(g) = E[ chosen-action conditional loss ].

    gate == 1 -> adapt -> a_adapt(z);  gate == 0 -> freeze -> a_freeze(z).
    """
    chosen = np.where(gate == 1, a_adapt, a_freeze)
    return float(np.mean(chosen))


# -----------------------------------------------------------------------------
# One experiment cell at a fixed noise level
# -----------------------------------------------------------------------------
@dataclass
class CellResult:
    noise_sd: float
    n: int
    # (A) exact conditional-risk check
    lhs_conditional: float          # R(ghat) - R(g*) via conditional action means
    rhs_identity: float             # E[|Delta| 1{mismatch}]
    abs_gap_exact: float            # |lhs_conditional - rhs_identity| (must be ~0)
    # (B) realized-loss Monte-Carlo check
    lhs_realized: float             # Rhat(ghat) - Rhat(g*) from sampled losses
    realized_se: float              # standard error of the realized difference
    realized_z: float               # (lhs_realized - rhs_identity) / realized_se
    # diagnostics
    mismatch_rate: float            # P(sign(Deltahat) != sign(Delta))
    mean_abs_delta: float           # E[|Delta|]
    half_mean_abs_delta: float      # E[|Delta|]/2  (minimax / coin-flip ceiling)
    near_boundary_regret: float     # regret contribution from |Delta| < eps band
    eps_band: float


def run_cell(
    delta: np.ndarray,
    *,
    noise_sd: float,
    a_adapt: np.ndarray,
    rng: np.random.Generator,
    loss_obs_sd: float = 0.30,
    eps_band: float = 0.05,
) -> CellResult:
    """Validate the identity at one noise level on a FIXED Delta sample.

    delta      : Delta(z_i), the true conditional benefit at each sampled z.
    a_adapt    : a_adapt(z_i) = E[ell(f_a)|Z=z_i]; then a_freeze = a_adapt + delta
                 because Delta = E[ell(f_0) - ell(f_a)] = a_freeze - a_adapt.
    noise_sd   : SD of the Gaussian estimator noise in Deltahat = Delta + noise.
    loss_obs_sd: SD of the per-sample loss noise (only affects (B), the realized
                 check; the conditional check (A) is independent of it).
    """
    n = delta.shape[0]
    a_freeze = a_adapt + delta  # by definition of Delta

    # Plug-in estimator: Deltahat = Delta + Gaussian noise.
    delta_hat = delta + noise_sd * rng.standard_normal(n)

    g_star = bayes_gate(delta)
    g_hat = plugin_gate(delta_hat)

    # ---- (A) EXACT identity via conditional action means -------------------
    r_star_cond = conditional_risk(g_star, a_freeze, a_adapt)
    r_hat_cond = conditional_risk(g_hat, a_freeze, a_adapt)
    lhs_conditional = r_hat_cond - r_star_cond
    rhs = rhs_regret(delta, delta_hat)
    abs_gap_exact = abs(lhs_conditional - rhs)

    # ---- (B) REALIZED-loss Monte-Carlo check -------------------------------
    # Draw actual losses with the right conditional means. Any zero-mean noise
    # works; we use a Gaussian. (Losses need not be in [0,1] for the identity
    # -- only the conditional means matter -- so we keep it simple and general.)
    ell_freeze = a_freeze + loss_obs_sd * rng.standard_normal(n)  # ell(f_0(X),Y)
    ell_adapt = a_adapt + loss_obs_sd * rng.standard_normal(n)   # ell(f_a(X),Y)
    realized_loss_star = np.where(g_star == 1, ell_adapt, ell_freeze)
    realized_loss_hat = np.where(g_hat == 1, ell_adapt, ell_freeze)
    per_sample_excess = realized_loss_hat - realized_loss_star
    lhs_realized = float(np.mean(per_sample_excess))
    realized_se = float(np.std(per_sample_excess, ddof=1) / np.sqrt(n))
    realized_z = (lhs_realized - rhs) / realized_se if realized_se > 0 else 0.0

    # ---- diagnostics -------------------------------------------------------
    mism = sign_mismatch(delta, delta_hat)
    mismatch_rate = float(np.mean(mism))
    mean_abs_delta = float(np.mean(np.abs(delta)))
    near = np.abs(delta) < eps_band
    near_boundary_regret = float(np.mean(np.abs(delta) * mism * near))

    return CellResult(
        noise_sd=float(noise_sd),
        n=int(n),
        lhs_conditional=lhs_conditional,
        rhs_identity=rhs,
        abs_gap_exact=float(abs_gap_exact),
        lhs_realized=lhs_realized,
        realized_se=realized_se,
        realized_z=float(realized_z),
        mismatch_rate=mismatch_rate,
        mean_abs_delta=mean_abs_delta,
        half_mean_abs_delta=0.5 * mean_abs_delta,
        near_boundary_regret=near_boundary_regret,
        eps_band=float(eps_band),
    )


# -----------------------------------------------------------------------------
# Corollary checks
# -----------------------------------------------------------------------------
def check_corollary_zero_noise(delta: np.ndarray, a_adapt: np.ndarray) -> dict:
    """Corollary (a): noise=0 => sign(Deltahat)=sign(Delta) a.e. => regret 0."""
    rng = np.random.default_rng(0)
    cell = run_cell(delta, noise_sd=0.0, a_adapt=a_adapt, rng=rng)
    return {
        "lhs_conditional": cell.lhs_conditional,
        "rhs_identity": cell.rhs_identity,
        "regret_is_zero": bool(abs(cell.lhs_conditional) < 1e-12 and cell.rhs_identity < 1e-12),
    }


def check_corollary_near_boundary(delta: np.ndarray, a_adapt: np.ndarray,
                                  eps_band: float = 0.05, noise_sd: float = 0.20) -> dict:
    """Corollary (c): on {|Delta| < eps} the committal regret is <= eps.

    We confirm the per-z regret contribution of EVERY near-boundary point is
    <= eps (it equals |Delta| <= eps when mismatched, 0 otherwise), which is
    exactly the 'abstaining there costs at most eps' justification.
    """
    rng = np.random.default_rng(7)
    delta_hat = delta + noise_sd * rng.standard_normal(delta.shape[0])
    mism = sign_mismatch(delta, delta_hat)
    near = np.abs(delta) < eps_band
    per_z_regret_near = np.abs(delta)[near] * mism[near]
    max_near = float(per_z_regret_near.max()) if per_z_regret_near.size else 0.0
    return {
        "eps_band": eps_band,
        "n_near_boundary": int(near.sum()),
        "max_per_z_regret_in_band": max_near,
        "bounded_by_eps": bool(max_near <= eps_band + 1e-12),
    }


def check_corollary_minimax(delta: np.ndarray, a_adapt: np.ndarray) -> dict:
    """Minimax corollary: a LABEL-FREE estimator that cannot recover sign(Delta)
    (Theorem 1 unknowable regime) errs on the sign with prob >= 1/2 on the
    worst-case two-point pair, so its expected regret >= E[|Delta|]/2.

    Empirical surrogate. In the unknowable regime the estimator's sign is
    independent of the true sign (it has no information), i.e. it is a coin
    flip. We simulate Deltahat as PURE noise (independent of Delta): the
    mismatch probability -> 1/2 and the realized regret -> E[|Delta|]/2.
    We confirm the regret concentrates at E[|Delta|]/2 (the minimax floor).
    """
    rng = np.random.default_rng(11)
    n = delta.shape[0]
    # Estimator carries NO information about sign(Delta): independent noise.
    delta_hat_uninformative = rng.standard_normal(n)
    rhs = rhs_regret(delta, delta_hat_uninformative)
    floor = 0.5 * float(np.mean(np.abs(delta)))
    return {
        "regret_uninformative_estimator": rhs,
        "minimax_floor_half_E_abs_delta": floor,
        "ratio_to_floor": rhs / floor if floor > 0 else float("nan"),
        "mismatch_rate": float(np.mean(sign_mismatch(delta, delta_hat_uninformative))),
    }


# -----------------------------------------------------------------------------
# Driver
# -----------------------------------------------------------------------------
@dataclass
class ValidationReport:
    n_samples: int
    seed: int
    delta_distribution: str
    noise_levels: list
    cells: list = field(default_factory=list)
    corollary_zero_noise: dict = field(default_factory=dict)
    corollary_near_boundary: dict = field(default_factory=dict)
    corollary_minimax: dict = field(default_factory=dict)
    exact_identity_max_gap: float = 0.0
    exact_identity_holds: bool = False
    realized_all_within_4se: bool = False


def validate_thm2(
    n: int = 400_000,
    seed: int = 20260604,
    noise_levels: tuple[float, ...] = (0.0, 0.05, 0.1, 0.25, 0.5, 1.0, 2.0),
    eps_band: float = 0.05,
) -> ValidationReport:
    """Sample z with a known Delta(z), sweep estimator noise, validate identity."""
    rng = np.random.default_rng(seed)

    # ---- Known Delta(z): a continuous distribution straddling 0 so both gate
    # decisions occur and |Delta| varies (small near the boundary, large in the
    # tails). A standard normal is the canonical choice; the identity is
    # distribution-free, so the SHAPE only affects the numbers, not the match.
    delta = rng.standard_normal(n)  # Delta(z) ~ N(0,1)

    # a_adapt(z) = E[ell(f_a)|Z=z]: an arbitrary baseline level (the identity
    # is invariant to it, since it cancels in R(ghat)-R(g*)). Use a non-trivial
    # function of z to show the cancellation is real, not an artifact of a_adapt
    # being constant.
    a_adapt = 0.5 + 0.1 * rng.standard_normal(n)
    a_adapt = np.clip(a_adapt, 0.0, 1.0)

    cells: list[CellResult] = []
    for s in noise_levels:
        # Fresh independent rng per cell for reproducibility & independence.
        cell_rng = np.random.default_rng(seed + int(round(s * 1000)) + 1)
        cells.append(run_cell(delta, noise_sd=s, a_adapt=a_adapt, rng=cell_rng,
                              eps_band=eps_band))

    exact_gap = max(c.abs_gap_exact for c in cells)
    realized_ok = all(abs(c.realized_z) <= 4.0 for c in cells)

    report = ValidationReport(
        n_samples=n,
        seed=seed,
        delta_distribution="Delta(z) ~ N(0,1); a_adapt(z) ~ clip(N(0.5,0.1),0,1)",
        noise_levels=list(noise_levels),
        cells=[asdict(c) for c in cells],
        corollary_zero_noise=check_corollary_zero_noise(delta, a_adapt),
        corollary_near_boundary=check_corollary_near_boundary(delta, a_adapt, eps_band=eps_band),
        corollary_minimax=check_corollary_minimax(delta, a_adapt),
        exact_identity_max_gap=float(exact_gap),
        exact_identity_holds=bool(exact_gap < 1e-9),
        realized_all_within_4se=bool(realized_ok),
    )
    return report


def _fmt(x: float, w: int = 12, p: int = 6) -> str:
    return f"{x:>{w}.{p}f}"


def main() -> None:
    rep = validate_thm2()

    print("=" * 100)
    print("K-Bound Theorem 2  --  plug-in regret decomposition  --  numerical validation")
    print("=" * 100)
    print(f"n per cell        : {rep.n_samples:,}")
    print(f"seed              : {rep.seed}")
    print(f"Delta / a_adapt   : {rep.delta_distribution}")
    print()
    print("Identity:  R(ghat) - R(g*)  ==  E[ |Delta| * 1{sign mismatch} ]")
    print()

    # ---- (A) exact conditional-risk table ----------------------------------
    print("-" * 100)
    print("(A) EXACT check  (conditional action-value risk;  no MC gap expected)")
    print("-" * 100)
    hdr = (f"{'noise_sd':>10} | {'LHS=R(ghat)-R(g*)':>18} | {'RHS=E|D|1{mis}':>16} | "
           f"{'|gap|':>12} | {'mismatch':>9}")
    print(hdr)
    print("-" * len(hdr))
    for c in rep.cells:
        print(f"{c['noise_sd']:>10.3f} | {_fmt(c['lhs_conditional'],18)} | "
              f"{_fmt(c['rhs_identity'],16)} | {c['abs_gap_exact']:>12.2e} | "
              f"{c['mismatch_rate']:>9.4f}")
    print()
    print(f"  max |gap| over all noise levels = {rep.exact_identity_max_gap:.3e}  "
          f"-> EXACT identity holds: {rep.exact_identity_holds}")
    print()

    # ---- (B) realized-loss Monte-Carlo table -------------------------------
    print("-" * 100)
    print("(B) REALIZED-LOSS check  (sampled per-example losses;  match to MC error)")
    print("-" * 100)
    hdr2 = (f"{'noise_sd':>10} | {'LHS_realized':>14} | {'RHS_identity':>14} | "
            f"{'SE':>10} | {'z-score':>9} | {'within 4SE':>11}")
    print(hdr2)
    print("-" * len(hdr2))
    for c in rep.cells:
        print(f"{c['noise_sd']:>10.3f} | {_fmt(c['lhs_realized'],14)} | "
              f"{_fmt(c['rhs_identity'],14)} | {c['realized_se']:>10.5f} | "
              f"{c['realized_z']:>9.3f} | {str(abs(c['realized_z'])<=4.0):>11}")
    print()
    print(f"  all realized LHS within 4 SE of RHS: {rep.realized_all_within_4se}")
    print()

    # ---- corollaries -------------------------------------------------------
    print("-" * 100)
    print("Corollaries")
    print("-" * 100)
    z = rep.corollary_zero_noise
    print(f"(a) zero-noise  : LHS={z['lhs_conditional']:.3e}  RHS={z['rhs_identity']:.3e}  "
          f"-> regret is exactly 0: {z['regret_is_zero']}")
    nb = rep.corollary_near_boundary
    print(f"(c) near-bdry   : {nb['n_near_boundary']:,} pts with |Delta|<{nb['eps_band']}; "
          f"max per-z regret in band = {nb['max_per_z_regret_in_band']:.4f} "
          f"<= eps: {nb['bounded_by_eps']}  (=> abstaining there costs <= eps)")
    mm = rep.corollary_minimax
    print(f"minimax         : uninformative (label-free, no sign info) estimator "
          f"regret = {mm['regret_uninformative_estimator']:.4f}")
    print(f"                  minimax floor E[|Delta|]/2          = "
          f"{mm['minimax_floor_half_E_abs_delta']:.4f}  "
          f"(ratio {mm['ratio_to_floor']:.4f}, mismatch {mm['mismatch_rate']:.4f})")
    print()

    # ---- monotonicity note (corollary b: bigger benefit costs more) --------
    print("-" * 100)
    print("Corollary (b) sanity: regret rises monotonically with estimator noise")
    print("  (more sign errors, and large-|Delta| errors are the costly ones)")
    print("-" * 100)
    prev = None
    mono = True
    for c in rep.cells:
        tag = ""
        if prev is not None and c['rhs_identity'] + 1e-9 < prev:
            mono = False
            tag = "  <-- non-monotone"
        print(f"  noise_sd={c['noise_sd']:>5.2f}  regret={c['rhs_identity']:.5f}"
              f"  mismatch={c['mismatch_rate']:.4f}{tag}")
        prev = c['rhs_identity']
    print(f"  monotone non-decreasing in noise: {mono}")
    print(f"  regret ceiling (estimator -> coin flip) = E[|Delta|]/2 = "
          f"{rep.cells[0]['half_mean_abs_delta']:.4f};")
    print(f"  finite Gaussian noise sd=2.0 reaches {rep.cells[-1]['rhs_identity']:.4f} "
          f"(mismatch {rep.cells[-1]['mismatch_rate']:.3f}), still climbing toward 1/2;")
    print(f"  the ceiling is hit exactly by the fully-uninformative estimator "
          f"(minimax check above: {mm['regret_uninformative_estimator']:.4f}).")
    print()

    # ---- machine-readable blob --------------------------------------------
    print("=" * 100)
    print("JSON")
    print("=" * 100)
    print(json.dumps(asdict(rep), indent=2))

    # ---- hard assertions (this is a VALIDATION, so fail loudly) -----------
    assert rep.exact_identity_holds, (
        f"EXACT identity violated: max gap {rep.exact_identity_max_gap:.3e} >= 1e-9")
    assert rep.realized_all_within_4se, "Realized-loss check exceeded 4 SE somewhere"
    assert rep.corollary_zero_noise["regret_is_zero"], "Zero-noise regret not 0"
    assert rep.corollary_near_boundary["bounded_by_eps"], "Near-boundary regret exceeded eps"
    print("\nALL CHECKS PASSED.")


if __name__ == "__main__":
    main()
