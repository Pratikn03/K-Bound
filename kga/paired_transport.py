"""Conditional paired-benefit certificates for a finite transport probability table.

No target labels are inputs. Validity requires a fixed prediction pair, independent
calibration/evidence sampling, and an externally declared transport contract. An LP
does not verify these assumptions. See next_phase/paired_transport_theory.md.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from decimal import ROUND_CEILING, ROUND_FLOOR, Decimal, localcontext
from typing import Any

import numpy as np
from scipy import sparse
from scipy.optimize import linprog

from kga.transport_numerics import (
    binomial_interval_certificate,
    exact_primal_certificate,
    fraction_record,
    outward_float,
    probability_array,
    rational,
)


def _array(value: Any, ndim: int, name: str) -> np.ndarray:
    arr = np.array(value, dtype=float, copy=True)
    if arr.ndim != ndim or not arr.size or not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} must be a nonempty finite {ndim}-dimensional array")
    arr.setflags(write=False)
    return arr


def _counts(value: Any, ndim: int) -> np.ndarray:
    raw = np.asarray(value)
    if raw.dtype.kind == "b" or any(isinstance(v, (bool, np.bool_)) for v in np.asarray(value, dtype=object).flat):
        raise TypeError("Boolean counts are not observations")
    arr = _array(value, ndim, "counts")
    if np.any(arr < 0) or np.any(arr != np.floor(arr)) or np.any(arr > 2**52):
        raise ValueError("Counts must be nonnegative exact integers below 2**52")
    return arr.astype(np.int64)


def binomial_interval(successes: int, trials: int, alpha: float) -> tuple[float, float]:
    """Rigorously outward CP bounds; resource limits conservatively return [0,1]."""
    lo, hi, _ = binomial_interval_certificate(successes, trials, alpha)
    return lo, hi


@dataclass(frozen=True)
class ConfidenceBounds:
    source_lower: np.ndarray
    source_upper: np.ndarray
    target_lower: np.ndarray
    target_upper: np.ndarray
    alpha: float
    interval_count: int
    per_interval_alpha: float
    source_class_counts: tuple[int, ...]
    target_count: int
    numerical_certificate: dict[str, Any] = field(default_factory=dict)


def confidence_bounds(source_counts: Any, target_counts: Any, *, alpha: float = 0.05) -> ConfidenceBounds:
    """Simultaneous probability boxes via Bonferroni and binomial intervals.

    Source columns are labeled classes; rows are fixed observable prediction-pair
    bins. Source sampling may be stratified by class. All bins, including observed
    zeros, must be retained. Target counts contain observable bins only.
    """
    source, target = _counts(source_counts, 2), _counts(target_counts, 1)
    family_budget = rational(alpha)
    if source.shape[0] != target.size or not 0 < family_budget < 1:
        raise ValueError("Invalid dimensions or family error budget")
    count = source.size + target.size
    budget = family_budget / count
    sl, su = np.empty(source.shape), np.empty(source.shape)
    totals = [sum(map(int, source[:, y])) for y in range(source.shape[1])]
    certificates = []
    for r, y in np.ndindex(source.shape):
        sl[r, y], su[r, y], proof = binomial_interval_certificate(int(source[r, y]), totals[y], budget)
        certificates.append({"entry": ["source", r, y], "certificate": proof})
    tl, tu = np.empty(target.size), np.empty(target.size)
    total_target = sum(map(int, target))
    for r in range(target.size):
        tl[r], tu[r], proof = binomial_interval_certificate(int(target[r]), total_target, budget)
        certificates.append({"entry": ["target", r], "certificate": proof})
    return ConfidenceBounds(
        sl,
        su,
        tl,
        tu,
        float(alpha),
        count,
        outward_float(budget, upper=False),
        tuple(totals),
        total_target,
        {
            "family_alpha_exact": fraction_record(family_budget),
            "per_interval_alpha_exact": fraction_record(budget),
            "allocation_sum_equals_family_alpha": True,
            "all_endpoints_outward_verified": True,
            "entries": certificates,
        },
    )


def accuracy_contrasts(pairs: Any, classes: int) -> np.ndarray:
    if isinstance(classes, bool) or not isinstance(classes, int) or classes < 2:
        raise ValueError("At least two classes required")
    raw = np.asarray(pairs)
    if raw.ndim != 2 or raw.shape[1] != 2 or raw.dtype.kind not in "iu" or np.any(raw < 0) or np.any(raw >= classes):
        raise ValueError("Prediction pairs must contain valid integer class indices")
    labels = np.arange(classes)
    return np.asarray((raw[:, 1, None] == labels).astype(float) - (raw[:, 0, None] == labels).astype(float))


@dataclass(frozen=True)
class TransportSpec:
    source_lower: Any
    source_upper: Any
    target_lower: Any
    target_upper: Any
    contrast: Any
    rho: float = 0.0
    assumption_contract: str | None = None
    confidence_certificate: dict[str, Any] | None = None
    numerical_input_metadata: dict[str, Any] = field(default_factory=dict, init=False)

    def __post_init__(self):
        enclosures = {}
        for name, ndim in (
            ("source_lower", 2),
            ("source_upper", 2),
            ("target_lower", 1),
            ("target_upper", 1),
        ):
            converted, changes = probability_array(getattr(self, name), ndim, upper=name.endswith("upper"))
            object.__setattr__(self, name, converted)
            enclosures[name] = changes
        raw_contrast = np.asarray(self.contrast, dtype=object)
        converted_contrast = _array(self.contrast, 2, "contrast")
        if any(
            rational(original) != rational(converted)
            for original, converted in zip(raw_contrast.flat, converted_contrast.flat)
        ):
            raise ValueError(
                "Contrast must be exactly representable in binary64; rational objective enclosure is not supplied"
            )
        object.__setattr__(self, "contrast", converted_contrast)
        exact_rho = rational(self.rho)
        object.__setattr__(self, "rho", outward_float(exact_rho, upper=True))
        object.__setattr__(
            self,
            "numerical_input_metadata",
            {
                "LP_coefficients": "exact_binary64_rationals",
                "float_probability_inputs": "unchanged_exact_binary64",
                "rational_probability_inputs": "outward_enclosed_in_binary64_boxes",
                "outward_enclosures": enclosures,
                "rho_input_exact": fraction_record(exact_rho),
                "rho_outer_bound": self.rho,
                "confidence_certificate": self.confidence_certificate,
            },
        )
        shape = self.source_lower.shape
        if (
            self.source_upper.shape != shape
            or self.contrast.shape != shape
            or self.target_lower.shape != (shape[0],)
            or self.target_upper.shape != (shape[0],)
        ):
            raise ValueError("Incompatible probability-table dimensions")
        if shape[1] < 2 or np.any(np.abs(self.contrast) > 1):
            raise ValueError("Invalid class count or bounded loss contrast")
        for lo, hi in ((self.source_lower, self.source_upper), (self.target_lower, self.target_upper)):
            if np.any(lo < 0) or np.any(hi > 1) or np.any(lo > hi):
                raise ValueError("Probability boxes must satisfy 0 <= lower <= upper <= 1")
            lower_columns = lo.reshape(lo.shape[0], -1).T
            upper_columns = hi.reshape(hi.shape[0], -1).T
            if any(sum(map(rational, column)) > 1 for column in lower_columns) or any(
                sum(map(rational, column)) < 1 for column in upper_columns
            ):
                raise ValueError("Probability boxes do not intersect the simplex")
        if isinstance(self.rho, bool) or not np.isfinite(self.rho) or not 0 <= self.rho <= 1:
            raise ValueError("Transport sensitivity rho must lie in [0,1]")
        if self.assumption_contract is not None and (
            not isinstance(self.assumption_contract, str) or not self.assumption_contract.strip()
        ):
            raise ValueError("Assumption contract must be a nonempty identifier or None")

    @classmethod
    def from_confidence(cls, bands: ConfidenceBounds, contrast: Any, **kwargs):
        return cls(
            bands.source_lower,
            bands.source_upper,
            bands.target_lower,
            bands.target_upper,
            contrast,
            confidence_certificate=bands.numerical_certificate,
            **kwargs,
        )


@dataclass(frozen=True)
class TransportResult:
    lower: float | None
    upper: float | None
    action: str
    status: str
    verified: bool
    witnesses: dict[str, Any] = field(default_factory=dict)
    assumption_contract: str | None = None


def _program(spec: TransportSpec, *, budget: bool = True, harmful: bool = False):
    rows, classes = spec.source_lower.shape
    cells = rows * classes
    n = 3 * cells + classes
    # x = (target joint t, transported source joint s, target prior pi, abs slack v).
    s0, p0, v0 = cells, 2 * cells, 2 * cells + classes
    inequalities: list[dict[int, float]] = []
    equalities: list[dict[int, float]] = []
    ib: list[float] = []
    eb: list[float] = []

    def add(entries, bound, equality=False):
        (equalities if equality else inequalities).append(entries)
        (eb if equality else ib).append(float(bound))

    add({p0 + y: 1 for y in range(classes)}, 1, True)
    for y in range(classes):
        for offset in (0, s0):
            add({**{offset + r * classes + y: 1 for r in range(rows)}, p0 + y: -1}, 0, True)
    for r in range(rows):
        add({r * classes + y: 1 for y in range(classes)}, spec.target_upper[r])
        add({r * classes + y: -1 for y in range(classes)}, -spec.target_lower[r])
        for y in range(classes):
            j = r * classes + y
            add({s0 + j: 1, p0 + y: -spec.source_upper[r, y]}, 0)
            add({s0 + j: -1, p0 + y: spec.source_lower[r, y]}, 0)
            add({j: 1, s0 + j: -1, v0 + j: -1}, 0)
            add({j: -1, s0 + j: 1, v0 + j: -1}, 0)
    if budget:
        add({v0 + j: 0.5 for j in range(cells)}, spec.rho)
    if harmful:
        add({j: val for j, val in enumerate(spec.contrast.flat) if val != 0}, 0)

    def matrix(entries):
        rr, cc, vv = [], [], []
        for i, row in enumerate(entries):
            for j, value in row.items():
                if value:
                    rr.append(i)
                    cc.append(j)
                    vv.append(value)
        return sparse.csr_matrix((vv, (rr, cc)), shape=(len(entries), n))

    return matrix(inequalities), np.asarray(ib), matrix(equalities), np.asarray(eb), n, cells, v0


def conservative_dual_bound(c, aub, bub, aeq, beq, inequality_dual, equality_dual) -> float:
    """Verified weak-duality lower bound for min c.x with 0 <= x <= 1.

    Project inequality multipliers to <=0. Correct *all* stationarity residuals
    using the known unit box. Directed Decimal arithmetic bounds the operations
    on the exact binary64 input coefficients; no solver stationarity assertion is
    trusted. This verifies the LP bound, not scientific sampling assumptions.
    """
    c = np.asarray(c, dtype=float)
    matrices = []
    for mat, rhs, dual, clip in ((aub, bub, inequality_dual, True), (aeq, beq, equality_dual, False)):
        rhs, dual = np.asarray(rhs, dtype=float), np.asarray(dual, dtype=float)
        matrix = sparse.csr_matrix(mat, shape=(len(rhs), len(c))) if len(rhs) else sparse.csr_matrix((0, len(c)))
        if dual.shape != rhs.shape or matrix.shape != (len(rhs), len(c)):
            raise ValueError("Dual certificate shape mismatch")
        if not all(np.all(np.isfinite(a)) for a in (c, rhs, dual, matrix.data)):
            raise ValueError("Nonfinite dual certificate")
        matrices.append((matrix, rhs, np.minimum(dual, 0.0) if clip else dual))

    def dec(x):
        return Decimal.from_float(float(x))

    # Upper bound every coordinate of A^T y + E^T z.
    with localcontext() as ctx:
        ctx.prec, ctx.rounding = 60, ROUND_CEILING
        sums = [Decimal(0) for _ in c]
        for matrix, _, dual in matrices:
            for i in range(matrix.shape[0]):
                d = dec(dual[i])
                if not d:
                    continue
                for p in range(matrix.indptr[i], matrix.indptr[i + 1]):
                    j = matrix.indices[p]
                    sums[j] += dec(matrix.data[p]) * d
    with localcontext() as ctx:
        ctx.prec, ctx.rounding = 60, ROUND_FLOOR
        value = Decimal(0)
        for _, rhs, dual in matrices:
            for b, d in zip(rhs, dual):
                value += dec(b) * dec(d)
        for coefficient, total in zip(c, sums):
            value += min(Decimal(0), dec(coefficient) - total)
    value_float = float(value)
    if not np.isfinite(value_float):
        raise ValueError("Overflow in dual certificate")
    return float(np.nextafter(value_float, -np.inf))


def _minimize(program, objective):
    aub, bub, aeq, beq, n, cells, _ = program
    result = linprog(objective, A_ub=aub, b_ub=bub, A_eq=aeq, b_eq=beq, bounds=[(0.0, 1.0)] * n, method="highs")
    if not result.success:
        return None, {
            "status": "solver_reported_infeasible" if result.status == 2 else "solver_failed",
            "solver_status": int(result.status),
            "message": str(result.message),
            "exact_infeasibility_proved": False,
        }
    x = np.asarray(result.x)
    try:
        if x.shape != (n,) or not np.all(np.isfinite(x)):
            raise ValueError("Invalid primal witness")
        residual = max(
            float(np.max(aub @ x - bub, initial=0)),
            float(np.max(np.abs(aeq @ x - beq), initial=0)),
            float(np.max(-x, initial=0)),
            float(np.max(x - 1, initial=0)),
        )
        lower = conservative_dual_bound(objective, aub, bub, aeq, beq, result.ineqlin.marginals, result.eqlin.marginals)
        primal = float(objective @ x)
        gap = primal - lower
        if residual > 1e-7 or gap < -1e-8 or gap > 1e-6:
            raise ValueError("Primal residual or conservative dual gap exceeds verification tolerance")
        exact_solution, exact_proof = exact_primal_certificate(aub, bub, aeq, beq, x)
        if exact_solution is None:
            return None, {"status": "exact_primal_not_certified", "exact_primal": exact_proof}
    except (ValueError, TypeError, AttributeError, OverflowError) as error:
        return None, {"status": "numerical_verification_failed", "message": str(error)}
    return lower, {
        "status": "checked",
        "dual_lower_bound": lower,
        "primal_objective": primal,
        "primal_residual": residual,
        "dual_gap": gap,
        "target_joint": x[:cells].tolist(),
        "solution": x.tolist(),
        "inequality_dual": result.ineqlin.marginals.tolist(),
        "equality_dual": result.eqlin.marginals.tolist(),
        "exact_primal": exact_proof,
    }


def _result(spec, lower, upper, witnesses) -> TransportResult:
    for witness in witnesses.values():
        witness["numerical_input_metadata"] = spec.numerical_input_metadata
    if lower is None or upper is None:
        status = next((v["status"] for v in witnesses.values() if v["status"] != "checked"), "solver_failed")
        return TransportResult(None, None, "ABSTAIN", status, False, witnesses, spec.assumption_contract)
    if lower > upper:
        return TransportResult(
            None, None, "ABSTAIN", "numerical_verification_failed", False, witnesses, spec.assumption_contract
        )
    action = "ADAPT" if lower > 1e-10 else "FREEZE" if upper < -1e-10 else "ABSTAIN"
    status = "conditional_certificate"
    if spec.assumption_contract is None:
        action, status = "ABSTAIN", "assumptions_not_declared"
    return TransportResult(lower, upper, action, status, True, witnesses, spec.assumption_contract)


def solve_benefit(spec: TransportSpec) -> TransportResult:
    program = _program(spec)
    objective = np.zeros(program[4])
    objective[: program[5]] = spec.contrast.ravel()
    lo, lw = _minimize(program, objective)
    neg_hi, uw = _minimize(program, -objective)
    return _result(
        spec,
        max(-1.0, lo) if lo is not None else None,
        min(1.0, -neg_hi) if neg_hi is not None else None,
        {"lower": lw, "upper": uw},
    )


def separate_accuracy_interval(spec: TransportSpec, frozen_accuracy: Any, candidate_accuracy: Any) -> TransportResult:
    """Separate extrema on the SAME polytope, followed by interval subtraction.

    This is a deliberately specified coupling relaxation, not an external method.
    """
    if not np.array_equal(np.asarray(candidate_accuracy) - np.asarray(frozen_accuracy), spec.contrast):
        raise ValueError("Accuracy coefficients do not define the declared paired contrast")
    program = _program(spec)
    bounds, witnesses = {}, {}
    for name, values in (("frozen", frozen_accuracy), ("candidate", candidate_accuracy)):
        arr = _array(values, 2, name)
        if arr.shape != spec.contrast.shape or np.any(arr < 0) or np.any(arr > 1):
            raise ValueError("Accuracy coefficients must match the table and lie in [0,1]")
        objective = np.zeros(program[4])
        objective[: program[5]] = arr.ravel()
        for direction, sign in (("lower", 1), ("upper", -1)):
            bound, witness = _minimize(program, sign * objective)
            witnesses[f"{name}_{direction}"] = witness
            bounds[f"{name}_{direction}"] = sign * bound if bound is not None else None
    if any(v is None for v in bounds.values()):
        return _result(spec, None, None, witnesses)
    finite_bounds = {name: value for name, value in bounds.items() if value is not None}
    lo = np.nextafter(finite_bounds["candidate_lower"] - finite_bounds["frozen_upper"], -np.inf)
    hi = np.nextafter(finite_bounds["candidate_upper"] - finite_bounds["frozen_lower"], np.inf)
    return _result(spec, max(-1.0, float(lo)), min(1.0, float(hi)), witnesses)


def break_even_budget(spec: TransportSpec) -> TransportResult:
    """Lower bound the smallest external TV allowance admitting nonpositive benefit.

    The lower endpoint is a directed weak-duality certificate. No upper endpoint
    is returned in this API. The exact rational feasible witness is separately
    recorded; the historical approximate objective remains diagnostic only.
    """
    program = _program(spec, budget=False, harmful=True)
    objective = np.zeros(program[4])
    objective[program[6] :] = 0.5
    lower, witness = _minimize(program, objective)
    if lower is None:
        return TransportResult(None, None, "ABSTAIN", witness["status"], False, {"minimum": witness})
    return TransportResult(
        max(0.0, lower),
        None,
        "ABSTAIN",
        "sensitivity_lower_bound",
        True,
        {"minimum": witness},
        spec.assumption_contract,
    )
