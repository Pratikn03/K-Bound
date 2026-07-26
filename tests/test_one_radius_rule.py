"""ONE radius rule: the tripwire for defect D9.

Fix-queue item 25 removed the ``k = min(n, ceil((n+1)(1-alpha)))`` clamp from
``kga/certificate.py``.  The audit that followed found the rule was still stated
once and implemented *twice*: the driver shim
``docs/research/kbound/scripts/kbound_decide.py`` still defaulted to
``clamp="min_n"``, so the shipped library and the code that produces the tables
disagreed for every calibration pool of ``n <= 8`` at ``alpha = 0.10``.

The rule, in full, is::

    eps = r_(k),   k = ceil((n + 1) * (1 - alpha)),   over sorted |Delta_hat - Delta|
    pool           = leave-one-out-of-pool: cell i's radius excludes cell i
    k > n          => eps = +inf  (UserWarning) or InsufficientCalibrationError
                      -- never r_(n), never an interpolated quantile

This module fails if any of the three defects that produced D9 come back:

* a **clamp** -- ``min(n, k)`` inside a radius function, or a ``clamp`` /
  ``on_infeasible='clamp'`` knob on a decision entry point;
* an **interpolated** ``np.quantile`` / ``np.percentile`` radius;
* **drift between the two implementations** at the infeasible sizes, which is
  exactly where the clamp used to hide.

The guards are both behavioural and source-level.  The source-level ones are AST
based (not regex) and carry an explicit allowlist; the two allowlisted functions
are the *only* places in ``kga`` permitted to write ``min(n, ceil(...))``, and
neither is on a decision path.
"""

from __future__ import annotations

import ast
import math
import sys
import warnings
from pathlib import Path

import numpy as np
import pytest

from kga.certificate import (
    InsufficientCalibrationError,
    conformal_radii_loo,
    conformal_split,
    legacy_clamped_radius,
    min_calibration_size,
    split_conformal_rank_radius,
)
from kga.policy import decide_kga

ALPHA = 0.1
REPO = Path(__file__).resolve().parents[1]
DRIVER = REPO / "docs" / "research" / "kbound" / "scripts" / "kbound_decide.py"
LIBRARY_FILES = (REPO / "kga" / "certificate.py", REPO / "kga" / "policy.py")

#: The only functions allowed to contain ``min(n, ceil(...))``.  Neither is
#: reachable from :func:`kga.policy.decide_kga`:
#: ``conformal_attained_level`` reports the coverage *ceiling* at a pool size and
#: selects no order statistic; ``legacy_clamped_radius`` is the superseded rule,
#: kept under its own name so that the under-coverage regression test has
#: something to measure.  ``fa_ceiling`` is the third and last: it evaluates the
#: arithmetic ceiling ``(n - k)/n`` that FA_u cannot exceed under in-pool rank
#: calibration, and likewise selects no order statistic.
CLAMP_ALLOWLIST = frozenset({"conformal_attained_level", "legacy_clamped_radius", "fa_ceiling"})

#: Functions that compute or apply a certificate radius.  None of them may call
#: an interpolating quantile.
RADIUS_FUNCTIONS = frozenset(
    {
        "split_conformal_rank_radius",
        "conformal_radii_loo",
        "conformal_split",
        "decide_kga",
        "conformal_radius",
        "_rank_radius_local",
        "radii_loo",
        "radii_in_pool",
        "radii_holdout",
        "legacy_clamped_radius",
    }
)


# ---------------------------------------------------------------------------
# source-level guards
# ---------------------------------------------------------------------------
def _functions(path: Path):
    """Yield ``(name, node)`` for every function defined in ``path``."""
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            yield node.name, node


def _is_min_of_ceil(node: ast.AST) -> bool:
    """True for ``min(<anything>, ceil(...))`` in either argument order."""
    if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "min"):
        return False
    for arg in node.args:
        inner = arg
        if isinstance(inner, ast.Call) and isinstance(inner.func, ast.Name) and inner.func.id == "int":
            inner = inner.args[0] if inner.args else inner
        if isinstance(inner, ast.Call):
            fn = inner.func
            name = fn.attr if isinstance(fn, ast.Attribute) else getattr(fn, "id", "")
            if name == "ceil":
                return True
    return False


def _quantile_calls(node: ast.AST) -> list[str]:
    """Names of interpolating-quantile calls inside ``node``."""
    found = []
    for sub in ast.walk(node):
        if isinstance(sub, ast.Call):
            fn = sub.func
            name = fn.attr if isinstance(fn, ast.Attribute) else getattr(fn, "id", "")
            if name in {"quantile", "nanquantile", "percentile", "nanpercentile"}:
                found.append(name)
    return found


@pytest.mark.parametrize("path", [*LIBRARY_FILES, DRIVER], ids=lambda p: p.name)
def test_no_rank_clamp_outside_the_allowlist(path: Path):
    """``min(n, ceil(...))`` may appear only in the two allowlisted functions."""
    if not path.exists():
        pytest.skip(f"{path} is not present in this release")
    offenders = []
    for name, node in _functions(path):
        if name in CLAMP_ALLOWLIST:
            continue
        for sub in ast.walk(node):
            if _is_min_of_ceil(sub):
                offenders.append(f"{path.name}::{name} line {sub.lineno}")
    assert not offenders, (
        "the exact-rank clamp k = min(n, ceil((n+1)(1-alpha))) is back in a radius path: "
        + ", ".join(offenders)
        + ". The clamp attains only n/(n+1) < 1-alpha; the rule is +inf => ABSTAIN. "
        "If you genuinely need the superseded value, call "
        "kga.certificate.legacy_clamped_radius() by name."
    )


@pytest.mark.parametrize("path", [*LIBRARY_FILES, DRIVER], ids=lambda p: p.name)
def test_no_interpolated_quantile_in_a_radius_function(path: Path):
    """No radius function may call ``np.quantile`` / ``np.percentile``."""
    if not path.exists():
        pytest.skip(f"{path} is not present in this release")
    offenders = []
    for name, node in _functions(path):
        if name not in RADIUS_FUNCTIONS:
            continue
        for call in _quantile_calls(node):
            offenders.append(f"{path.name}::{name} calls {call}()")
    assert not offenders, (
        "an interpolated quantile is back in a radius function: "
        + ", ".join(offenders)
        + ". np.quantile's linear interpolation is not an observed order statistic and "
        "does not satisfy the finite-sample rank argument."
    )


@pytest.mark.parametrize("path", [*LIBRARY_FILES, DRIVER], ids=lambda p: p.name)
def test_no_clamp_parameter_on_any_decision_entry_point(path: Path):
    """No public radius/decision function may take a ``clamp``-style argument."""
    if not path.exists():
        pytest.skip(f"{path} is not present in this release")
    offenders = []
    for name, node in _functions(path):
        if name not in RADIUS_FUNCTIONS:
            continue
        args = node.args
        for arg in [*args.posonlyargs, *args.args, *args.kwonlyargs]:
            if arg.arg == "clamp":
                offenders.append(f"{path.name}::{name}({arg.arg}=...)")
    assert not offenders, (
        "a clamp switch is back on a decision entry point: "
        + ", ".join(offenders)
        + ". The radius rule must have no knob that changes it."
    )


def test_decide_kga_has_no_infeasibility_knob():
    """Neither ``decide_kga`` may let a caller re-enable the clamp."""
    import inspect

    params = set(inspect.signature(decide_kga).parameters)
    assert "on_infeasible" not in params and "clamp" not in params, (
        f"kga.policy.decide_kga exposes an infeasibility knob: {sorted(params)}"
    )


def test_clamp_is_rejected_as_an_on_infeasible_mode():
    with pytest.raises(ValueError, match="'inf' or 'raise'"):
        split_conformal_rank_radius(np.arange(1.0, 9.0), ALPHA, on_infeasible="clamp")
    with pytest.raises(ValueError, match="'inf' or 'raise'"):
        conformal_split(0.1, np.arange(1.0, 9.0), alpha=ALPHA, on_infeasible="clamp")


# ---------------------------------------------------------------------------
# behavioural guards
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("n", [1, 2, 3, 5, 8])
def test_infeasible_pools_never_return_a_finite_radius(n):
    r = np.abs(np.random.default_rng(n).standard_normal(n))
    with pytest.warns(UserWarning, match="needs n >="):
        eps = split_conformal_rank_radius(r, ALPHA)
    assert math.isinf(eps)
    # and the clamped value it used to return is strictly smaller, i.e. the old
    # behaviour really was the more permissive one.
    assert legacy_clamped_radius(r, ALPHA) < eps
    with pytest.raises(InsufficientCalibrationError):
        split_conformal_rank_radius(r, ALPHA, on_infeasible="raise")


@pytest.mark.parametrize("n", [9, 10, 12, 27, 60, 135])
def test_radius_is_always_an_observed_order_statistic(n):
    """An interpolated quantile would almost surely not be one of the residuals."""
    r = np.abs(np.random.default_rng(n).standard_normal(n))
    eps = split_conformal_rank_radius(r, ALPHA)
    assert eps in set(r.tolist()), "radius is not an observed residual -- interpolation is back"
    k = int(math.ceil((n + 1) * (1.0 - ALPHA)))
    assert eps == float(np.sort(r)[k - 1])


def test_loo_at_nine_cells_is_infeasible_and_abstains_everywhere():
    """The case the clamp was hiding: n = 9 cells -> LOO pools of 8 -> no radius.

    Camelyon17 Table VIII is exactly this shape (9 cells per seed).  Under the
    declared rule it cannot be certified at alpha = 0.10; the old clamp returned
    the maximum residual instead and produced decisions.
    """
    rng = np.random.default_rng(11)
    b_true = rng.normal(0.0, 0.08, 9)
    b_hat = b_true + rng.normal(0.0, 0.01, 9)
    with pytest.warns(UserWarning, match="needs n >="):
        eps, dec = decide_kga(b_hat, b_true, alpha=ALPHA)
    assert np.all(np.isinf(eps))
    assert set(np.asarray(dec).ravel().tolist()) == {"ABSTAIN"}
    # ten cells is the first LOO-feasible size at alpha = 0.10.
    assert min_calibration_size(ALPHA) == 9
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert np.all(np.isfinite(conformal_radii_loo(np.abs(rng.standard_normal(10)), ALPHA)))


# ---------------------------------------------------------------------------
# the two implementations must not drift apart again
# ---------------------------------------------------------------------------
def _driver():
    if not DRIVER.exists():
        pytest.skip("driver module kbound_decide.py is not present in this release")
    sys.path.insert(0, str(DRIVER.parent))
    try:
        import kbound_decide  # noqa: PLC0415
    except Exception as exc:  # pragma: no cover - reported, never swallowed
        pytest.skip(f"driver module kbound_decide.py is not importable: {exc!r}")
    finally:
        if str(DRIVER.parent) in sys.path:
            sys.path.remove(str(DRIVER.parent))
    return kbound_decide


@pytest.mark.parametrize("n", [1, 3, 5, 8, 9, 10, 27, 60])
def test_driver_and_library_radii_agree_including_the_infeasible_sizes(n):
    """D9's actual failure: the two agreed for n >= 9 and disagreed below it."""
    kb = _driver()
    r = np.abs(np.random.default_rng(1000 + n).standard_normal(n))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        lib = split_conformal_rank_radius(r, ALPHA)
        drv = kb.conformal_radius(r, ALPHA)
    assert (math.isinf(lib) and math.isinf(drv)) or lib == drv, (
        f"driver and library disagree at n={n}: library {lib!r} vs driver {drv!r}"
    )


def test_driver_decisions_match_the_library_at_an_infeasible_pool():
    kb = _driver()
    rng = np.random.default_rng(7)
    b_true = rng.normal(0.0, 0.08, 9)
    b_hat = b_true + rng.normal(0.0, 0.01, 9)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        eps_lib, dec_lib = decide_kga(b_hat, b_true, alpha=ALPHA)
        eps_drv, dec_drv = kb.decide_from_records(b_hat, b_true, alpha=ALPHA, calibration="loo")
    assert np.all(np.isinf(np.asarray(eps_lib))) and np.all(np.isinf(np.asarray(eps_drv, dtype=float)))
    assert [str(d) for d in np.asarray(dec_lib).ravel()] == [str(d) for d in np.asarray(dec_drv).ravel()]


# ---------------------------------------------------------------------------
# defect D10 -- tree-wide census of interpolated radii, with a frozen allowlist
# ---------------------------------------------------------------------------
#: Every place in the tree that still assigns an *interpolated* quantile to a
#: certificate radius, with the reason it is allowed to.  Nothing here is on a
#: promoted-number path (see ``docs/research/kbound/THEORY_TO_CODE_MAP.md``).
#: A new entry means a new second rule: fix the code, do not extend this dict
#: unless you can write a reason as specific as the ones below.
INTERPOLATED_RADIUS_ALLOWLIST: dict[str, str] = {
    # -- superseded exploratory v1 scripts.  Each carries a SUPERSEDED RULE
    #    banner naming itself; no promoted number comes from them, and their
    #    archived JSON outputs were produced under the interpolated rule.
    "docs/research/kbound/scripts/cifar_tent_mps.py": "superseded v1 (banner in file)",
    "docs/research/kbound/scripts/kbound_full_experiments.py": "superseded v1 (banner in file)",
    "docs/research/kbound/scripts/kbound_harmful_regime.py": "superseded v1 (banner in file)",
    "docs/research/kbound/scripts/knowability_experiment.py": "superseded v1 (banner in file)",
    "docs/research/kbound/scripts/mixed_regime_experiment.py": "superseded v1 (banner in file)",
    "docs/research/kbound/scripts/tta_collapse_experiment.py": "superseded v1 (banner in file)",
    "docs/research/kbound/scripts/knowability_frontier_validation.py": "superseded v1 (banner in file)",
    "docs/research/kbound/scripts/theory_extensions_validation.py": "superseded v1 (banner in file)",
    # -- deliberate replays / demonstrations OF the superseded rule.  Converting
    #    these would delete the comparison they exist to make.
    "docs/research/kbound/theory_v2/realdata/eps_recal/_probe2.py":
        "labelled 'the superseded rule'; the probe compares the two rules head to head",
    "docs/research/kbound/theory_v2/realdata/eps_recal/eps_recal_camelyon.py":
        "the eps-recalibration study whose subject IS the archived interpolated radius",
    # -- exploratory sweeps that were never promoted.
    "docs/research/kbound/gapclose_wave5/win_hunt_A_universal_gate.py": "exploratory win-hunt sweep, not promoted",
    "docs/research/kbound/gapclose_wave5/win_hunt_E_universal7.py": "exploratory win-hunt sweep, not promoted",
    "docs/research/kbound/realshift_win/verify_realshift_win.py": "exploratory real-shift probe, not promoted",
    "docs/research/kbound/theory_v2/realdata/deepgrid_audit/deepgrid_audit.py":
        "parametric-bootstrap deviation quantile, not a split-conformal residual radius",
    "experiments/kbound/test_3dadam_bootstrap.py": "3D-ADAM benchmark, not a K-Bound panel track",
    "experiments/kbound/test_3dadam_namedcond.py": "3D-ADAM benchmark, not a K-Bound panel track",
    # -- immutable archived analysis scripts.  These ARE the record of how an
    #    archived number was produced; editing them would falsify the record.
    "experiments/kbound/results/camelyon17_fullscale_B_v1/_locked_B_analysis.py":
        "sealed archived analysis script; it documents how the archived number was made",
    "experiments/kbound/results/camelyon17_fullscale_B_v1/estimator_dryrun/dryrun.py":
        "sealed archived dry-run under the same locked results directory",
    # -- not a certificate radius at all: a baseline router's threshold on a
    #    SOURCE statistic tau, which the K-Bound rule does not govern.
    "experiments/kbound/wilds/analyze_camelyon_kbound.py": "route_b baseline tau threshold, not a K-Bound radius",
    "experiments/kbound/wilds/analyze_iwildcam_kbound.py": "route_b baseline tau threshold, not a K-Bound radius",
}

_RADIUS_NAMES = frozenset(
    {"eps", "epsilon", "eps_glob", "epsg", "eps0", "eps_naive", "eps_bonf", "eps_sidak",
     "eps_inpool", "eps_archived", "eps_out", "eps_c", "radius", "tau_star"}
)


def _interpolated_radius_sites(path: Path) -> list[int]:
    """Line numbers where an interpolating quantile is bound to a radius name."""
    try:
        tree = ast.parse(path.read_text(errors="ignore"), filename=str(path))
    except SyntaxError:
        return []
    out = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            value, targets = node.value, node.targets
        elif isinstance(node, ast.AnnAssign) and node.value is not None:
            value, targets = node.value, [node.target]
        else:
            continue
        names = {sub.id for t in targets for sub in ast.walk(t) if isinstance(sub, ast.Name)}
        if names & _RADIUS_NAMES and _quantile_calls(value):
            out.append(node.lineno)
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and any(
            key in node.name.lower() for key in ("radius", "eps", "conformal")
        ):
            for sub in ast.walk(node):
                if isinstance(sub, ast.Return) and sub.value is not None and _quantile_calls(sub.value):
                    out.append(sub.lineno)
    return sorted(set(out))


def test_no_new_interpolated_certificate_radius_in_the_tree():
    """D10: the census of interpolated radii may shrink, never grow."""
    found: dict[str, list[int]] = {}
    for path in sorted(REPO.rglob("*.py")):
        if "__pycache__" in path.parts:
            continue
        rel = str(path.relative_to(REPO))
        if rel == str(Path(__file__).relative_to(REPO)):
            continue
        sites = _interpolated_radius_sites(path)
        if sites:
            found[rel] = sites
    unexpected = {k: v for k, v in found.items() if k not in INTERPOLATED_RADIUS_ALLOWLIST}
    assert not unexpected, (
        "a new interpolated certificate radius appeared -- that is a second rule "
        f"(defect D10): {unexpected}. Use kga.certificate.split_conformal_rank_radius "
        "(or kbound_decide.conformal_radius), or add an entry to "
        "INTERPOLATED_RADIUS_ALLOWLIST with a reason as specific as the ones there."
    )
    stale = sorted(set(INTERPOLATED_RADIUS_ALLOWLIST) - set(found))
    assert not stale, (
        f"these allowlist entries no longer have an interpolated radius: {stale}. "
        "Delete them so the allowlist keeps documenting only real survivors."
    )


# ---------------------------------------------------------------------------
# defect D10 -- every ``decide_kga`` outside the two implementations delegates
# ---------------------------------------------------------------------------
#: The two files allowed to *implement* ``decide_kga``: the library, and the
#: driver-side shim that item 15 introduced so a bare checkout still runs.
DECIDE_KGA_IMPLEMENTATIONS = frozenset(
    {"kga/policy.py", "docs/research/kbound/scripts/kbound_decide.py"}
)


def _is_pure_delegation(node: ast.FunctionDef) -> bool:
    """True iff the body is (optional docstring) + a single ``return <call>``."""
    body = list(node.body)
    if body and isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant) \
            and isinstance(body[0].value.value, str):
        body = body[1:]
    if len(body) != 1 or not isinstance(body[0], ast.Return):
        return False
    return isinstance(body[0].value, ast.Call)


def test_every_decide_kga_fork_is_a_bodiless_delegation():
    """Item 15's actual invariant, asserted rather than asserted-about.

    Seven copy-pasted ``decide_kga`` forks produced every published number while
    the shipped library produced none.  The fix deleted their bodies.  Nothing
    stopped a body growing back, so this pins it: outside the two files that are
    allowed to implement the rule, ``decide_kga`` must be one call and nothing
    else -- no local radius, no local trichotomy, no second estimator.
    """
    offenders = []
    for path in sorted(REPO.rglob("*.py")):
        if "__pycache__" in path.parts:
            continue
        rel = str(path.relative_to(REPO))
        if rel in DECIDE_KGA_IMPLEMENTATIONS:
            continue
        try:
            tree = ast.parse(path.read_text(errors="ignore"), filename=str(path))
        except SyntaxError:
            continue
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and node.name == "decide_kga":
                if not _is_pure_delegation(node):
                    offenders.append(f"{rel}:{node.lineno}")
    assert not offenders, (
        "a decide_kga fork grew a body back (fix-queue item 15 / defect D10): "
        + ", ".join(offenders)
        + ". Forks must be a single delegating call to kbound_decide.decide_kga "
        "(which calls kga.policy.decide_kga); only "
        + ", ".join(sorted(DECIDE_KGA_IMPLEMENTATIONS))
        + " may implement the rule."
    )
