"""Bounded finite-fiber discovery with rational arithmetic, not theorem authority.

No floating-point LP solver, target dataset, or target-label oracle is used. The
independent checker in certificate_check.py validates the emitted certificates
without importing this solver. Lean remains the theorem-status authority.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from itertools import combinations, islice
from math import comb
from typing import Iterable, Sequence

Q = Fraction
Vector = tuple[Q, ...]
Matrix = tuple[Vector, ...]


class ProtocolFailure(ValueError):
    """Invalid mathematical input or an empty ambiguity fiber."""


class EmptyFiber(ProtocolFailure):
    """The declared equality and simplex constraints have no feasible world."""


class SearchLimit(RuntimeError):
    """Bounded discovery budget exceeded; never an identification result."""


def rational(value: str | int | Q) -> Q:
    if isinstance(value, bool) or isinstance(value, float):
        raise ProtocolFailure("Use exact rational strings or integers, not bool/float")
    if not isinstance(value, (str, int, Q)):
        raise ProtocolFailure("Unsupported rational representation")
    try:
        return Q(value)
    except (ValueError, ZeroDivisionError) as exc:
        raise ProtocolFailure("Invalid rational") from exc


def dot(left: Sequence[Q], right: Sequence[Q]) -> Q:
    if len(left) != len(right):
        raise ProtocolFailure("Dot-product dimension mismatch")
    return sum((x * y for x, y in zip(left, right)), Q(0))


def _rref(rows: Sequence[Sequence[Q]], width: int) -> tuple[list[list[Q]], list[int]]:
    if any(len(row) != width for row in rows):
        raise ProtocolFailure("Ragged matrix")
    result = [[rational(x) for x in row] for row in rows]
    pivots: list[int] = []
    for col in range(width):
        pivot = next((i for i in range(len(pivots), len(result)) if result[i][col]), None)
        if pivot is None:
            continue
        position = len(pivots)
        result[position], result[pivot] = result[pivot], result[position]
        divisor = result[position][col]
        result[position] = [x / divisor for x in result[position]]
        for i, row in enumerate(result):
            if i != position and row[col]:
                scale = row[col]
                result[i] = [x - scale * y for x, y in zip(row, result[position])]
        pivots.append(col)
        if len(pivots) == len(result):
            break
    return result, pivots


def rank(rows: Sequence[Sequence[Q]], width: int) -> int:
    return len(_rref(rows, width)[1])


def unique_solution(rows: Sequence[Sequence[Q]], rhs: Sequence[Q], width: int) -> Vector | None:
    if len(rows) != len(rhs):
        raise ProtocolFailure("Right-hand-side dimension mismatch")
    if any(len(row) != width for row in rows):
        raise ProtocolFailure("Ragged matrix")
    reduced, pivots = _rref([list(row) + [value] for row, value in zip(rows, rhs)], width + 1)
    if width in pivots or len(pivots) != width:
        return None
    answer = [Q(0)] * width
    for row, col in zip(reduced, pivots):
        answer[col] = row[-1]
    return tuple(answer)


def null_basis(rows: Sequence[Sequence[Q]], width: int) -> tuple[Vector, ...]:
    reduced, pivots = _rref(rows, width)
    answer = []
    for free in (i for i in range(width) if i not in pivots):
        vec = [Q(0)] * width
        vec[free] = Q(1)
        for row, pivot in zip(reduced, pivots):
            vec[pivot] = -row[free]
        answer.append(tuple(vec))
    return tuple(answer)


@dataclass(frozen=True)
class Problem:
    name: str
    classes: int
    masses: Vector
    # One frozen cost vector for each stratum; candidates[candidate][stratum][class].
    frozen: Matrix
    candidates: tuple[Matrix, ...]
    restrictions: Matrix = ()
    values: Vector = ()
    justifications: tuple[str, ...] = ()
    ablations: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if isinstance(self.classes, bool) or not isinstance(self.classes, int) or self.classes < 3:
            raise ProtocolFailure("The initial discovery model requires integer K >= 3")
        if not isinstance(self.name, str) or not self.name.strip() or not self.masses or not self.candidates:
            raise ProtocolFailure("Name, strata, and at least one candidate are required")
        object.__setattr__(self, "masses", tuple(rational(x) for x in self.masses))
        object.__setattr__(self, "frozen", tuple(tuple(rational(x) for x in row) for row in self.frozen))
        object.__setattr__(self, "candidates", tuple(tuple(tuple(rational(x) for x in row) for row in cost) for cost in self.candidates))
        object.__setattr__(self, "restrictions", tuple(tuple(rational(x) for x in row) for row in self.restrictions))
        object.__setattr__(self, "values", tuple(rational(x) for x in self.values))
        if any(x < 0 for x in self.masses) or sum(self.masses) != 1:
            raise ProtocolFailure("Stratum masses must be nonnegative and sum exactly to one")
        for cost in (self.frozen, *self.candidates):
            if len(cost) != self.strata or any(len(row) != self.classes for row in cost):
                raise ProtocolFailure("Cost dimensions do not match strata and classes")
            if any(x < 0 or x > 1 for row in cost for x in row):
                raise ProtocolFailure("Costs must be in [0,1]; record any scaling outside this oracle")
        if any(len(row) != self.dimension for row in self.restrictions):
            raise ProtocolFailure("Observable restriction dimension mismatch")
        if not (len(self.restrictions) == len(self.values) == len(self.justifications) == len(self.ablations)):
            raise ProtocolFailure("Every restriction needs a value, justification, and removal ablation")
        if any(not isinstance(x, str) or not x.strip() for x in (*self.justifications, *self.ablations)):
            raise ProtocolFailure("Empty restriction justification or ablation")

    @property
    def strata(self) -> int:
        return len(self.masses)

    @property
    def dimension(self) -> int:
        return self.classes * self.strata

    @property
    def equalities(self) -> Matrix:
        normalization = tuple(tuple(Q(int(i // self.classes == s)) for i in range(self.dimension)) for s in range(self.strata))
        return normalization + self.restrictions

    @property
    def rhs(self) -> Vector:
        return (Q(1),) * self.strata + self.values

    def contrast(self, candidate: int) -> Vector:
        cost = self.candidates[candidate]
        return tuple(self.masses[s] * (self.frozen[s][y] - cost[s][y]) for s in range(self.strata) for y in range(self.classes))

    def payload(self) -> dict:
        return {
            "name": self.name,
            "classes": self.classes,
            "masses": [str(x) for x in self.masses],
            "frozen": [[str(x) for x in row] for row in self.frozen],
            "candidates": [[[str(x) for x in row] for row in cost] for cost in self.candidates],
            "restrictions": [[str(x) for x in row] for row in self.restrictions],
            "values": [str(x) for x in self.values],
            "justifications": list(self.justifications),
            "ablations": list(self.ablations),
            "structural_availability": "mathematical_toy_only_not_established_from_unlabeled_data",
        }


def _check_search_budget(n: int, k: int, limit: int) -> None:
    if type(limit) is not int or limit < 1:
        raise ProtocolFailure("Enumeration budget must be a positive integer")
    if comb(n, k) > limit:
        raise SearchLimit(f"Exact enumeration needs {comb(n, k)} subsets; limit is {limit}")


def vertices(problem: Problem, *, max_subsets: int = 100_000) -> tuple[Vector, ...]:
    """Enumerate vertices of {x>=0: Hx=h}; H includes every simplex affine row."""
    h, b, d = problem.equalities, problem.rhs, problem.dimension
    r = rank(h, d)
    if rank([list(row) + [value] for row, value in zip(h, b)], d + 1) != r:
        raise EmptyFiber("Empty fiber: inconsistent affine restrictions")
    _check_search_budget(d, d - r, max_subsets)
    found: set[Vector] = set()
    for zeros in combinations(range(d), d - r):
        active = tuple(tuple(Q(int(i == zero)) for i in range(d)) for zero in zeros)
        point = unique_solution(h + active, b + (Q(0),) * len(active), d)
        if point is not None and all(x >= 0 for x in point) and all(dot(row, point) == value for row, value in zip(h, b)):
            found.add(point)
    if not found:
        raise EmptyFiber("Empty fiber: no feasible product-simplex vertex")
    return tuple(sorted(found))


def _dual(problem: Problem, objective: Vector, optimum: Q, *, max_subsets: int) -> Vector:
    """Find lambda: H^T lambda <= objective and h.lambda == minimum."""
    basis: list[int] = []
    for i, row in enumerate(problem.equalities):
        if rank([problem.equalities[j] for j in basis] + [row], problem.dimension) > len(basis):
            basis.append(i)
    size = len(basis)
    _check_search_budget(problem.dimension, size, max_subsets)
    for columns in combinations(range(problem.dimension), size):
        equations = tuple(tuple(problem.equalities[j][i] for j in basis) for i in columns)
        dual = unique_solution(equations, tuple(objective[i] for i in columns), size)
        if dual is None:
            continue
        if any(sum((problem.equalities[j][i] * dual[t] for t, j in enumerate(basis)), Q(0)) > objective[i] for i in range(problem.dimension)):
            continue
        if dot(tuple(problem.rhs[j] for j in basis), dual) != optimum:
            continue
        expanded = [Q(0)] * len(problem.equalities)
        for j, value in zip(basis, dual):
            expanded[j] = value
        return tuple(expanded)
    raise RuntimeError("No exact dual witness found; do not emit an uncertified extremum")


def strict_action(lower: Q, upper: Q) -> str:
    if lower > upper:
        raise ProtocolFailure("Reversed interval")
    if lower > 0:
        return "ADAPT"
    if upper < 0:
        return "FREEZE"
    return "ABSTAIN"


def certify(problem: Problem, candidate: int = 0, *, max_subsets: int = 100_000) -> dict:
    if isinstance(candidate, bool) or not isinstance(candidate, int) or not 0 <= candidate < len(problem.candidates):
        raise ProtocolFailure("Unknown candidate")
    points = vertices(problem, max_subsets=max_subsets)
    objective = problem.contrast(candidate)
    minimizing = min(points, key=lambda x: dot(objective, x))
    maximizing = max(points, key=lambda x: dot(objective, x))
    lower, upper = dot(objective, minimizing), dot(objective, maximizing)
    minimum_dual = _dual(problem, objective, lower, max_subsets=max_subsets)
    maximum_dual = tuple(-x for x in _dual(problem, tuple(-x for x in objective), -upper, max_subsets=max_subsets))
    rows = problem.equalities
    all_contrasts = [problem.contrast(j) for j in range(len(problem.candidates))]
    return {
        "schema": "mvc.exact-fiber-certificate.v1",
        "problem": problem.payload(),
        "candidate": candidate,
        "orientation": "frozen_cost_minus_candidate_cost",
        "comparison_scope": "fixed_candidate_vs_frozen_only",
        "deployment_selection": "NOT_IMPLEMENTED",
        "lower": str(lower),
        "upper": str(upper),
        "action": strict_action(lower, upper),
        "point_identified_at_realized_fiber": lower == upper,
        "minimum": {"point": list(map(str, minimizing)), "dual": list(map(str, minimum_dual))},
        "maximum": {"point": list(map(str, maximizing)), "dual": list(map(str, maximum_dual))},
        "discovery_diagnostics_not_checked_by_lp_checker": {
            "vertices": len(points),
            "affine_operator_rank": rank(rows, problem.dimension),
            "unrestricted_linear_rank_increment": rank([*rows, *all_contrasts], problem.dimension) - rank(rows, problem.dimension),
            "admissible_supplement_minimum": "NOT_INFERRED_FROM_RANK",
        },
    }


def minimum_allowed_moments(problem: Problem, allowed_rows: Iterable[Sequence[Q]]) -> int | None:
    """Finite exact search over a declared moment catalog; a discovery result only.

    This is uniform affine identification, not realized-fiber sign identification.
    The caller, not this function, must justify availability of allowed moments.
    """
    allowed = tuple(tuple(rational(x) for x in row) for row in islice(allowed_rows, 17))
    if any(len(row) != problem.dimension for row in allowed):
        raise ProtocolFailure("Supplement dimension mismatch")
    if len(allowed) > 16:
        raise SearchLimit("Admissible catalog search is limited to 16 moments")
    contrasts = [problem.contrast(j) for j in range(len(problem.candidates))]
    for size in range(len(allowed) + 1):
        for selected in combinations(allowed, size):
            augmented = [*problem.equalities, *selected]
            if rank([*augmented, *contrasts], problem.dimension) == rank(augmented, problem.dimension):
                return size
    return None
