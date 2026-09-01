"""Declared finite discovery grid and exact counterexample witnesses.

These are mathematical synthetic cases, not scientifically available target
moments, trained models, natural-shift scores, or prospective experiments.
"""

from __future__ import annotations

from fractions import Fraction as Q

from exact_oracle import Problem


def coordinate(dimension: int, index: int) -> tuple[Q, ...]:
    return tuple(Q(int(i == index)) for i in range(dimension))


def named_cases() -> list[Problem]:
    return [
        Problem(
            "T5_null_direction_without_sign_crossing", 3, (Q(1),),
            ((Q(0), Q(1, 2), Q(1)),), (((Q(1), Q(0), Q(0)),),),
            ((Q(1), Q(0), Q(0)),), (Q(1, 5),),
            ("Toy structural restriction eta_0=1/5; not asserted observable from unlabeled data",),
            ("Remove eta_0=1/5 to recover benefit interval [-1,1]",),
        ),
        Problem(
            "T3_primitive_moments_exceed_rank_increment", 3, (Q(1),),
            ((Q(0), Q(1, 2), Q(1)),), (((Q(0), Q(0), Q(0)),),),
        ),
        Problem(
            "T2_realized_vertex_not_uniform_affine_identification", 3, (Q(1),),
            ((Q(1), Q(0), Q(0)),), (((Q(0), Q(0), Q(0)),),),
            ((Q(1), Q(1), Q(0)),), (Q(0),),
            ("Toy zero event mass forces eta_0=eta_1=0 by nonnegativity",),
            ("Remove this row to allow the whole simplex",),
        ),
        Problem("boundary_lower_zero", 3, (Q(1),), ((Q(0), Q(1), Q(1)),), (((Q(0), Q(0), Q(0)),),)),
        Problem("boundary_upper_zero", 3, (Q(1),), ((Q(0), Q(0), Q(0)),), (((Q(1), Q(0), Q(0)),),)),
        Problem("opposite_worlds_qualified_T5", 3, (Q(1),), ((Q(1), Q(0), Q(0)),), (((Q(0), Q(1), Q(0)),),)),
        Problem(
            "rare_stratum_asymmetric_two_world_family", 3, (Q(15, 16), Q(1, 16)),
            ((Q(1, 4), Q(1, 8), Q(1, 16)), (Q(1), Q(0), Q(0))),
            (((Q(1, 4), Q(1, 8), Q(1, 16)), (Q(0), Q(1, 2), Q(0))),),
        ),
    ]


def _costs(k: int, m: int, pattern: int) -> tuple:
    if pattern == 0:  # zero contrasts and ties
        frozen = tuple((Q(0),) * k for _ in range(m))
        return frozen, (frozen, frozen)
    if pattern == 1:  # symmetric opposing class costs
        frozen = tuple(tuple(Q((y + s) % 2) for y in range(k)) for s in range(m))
        first = tuple(tuple(1 - x for x in row) for row in frozen)
        return frozen, (first, frozen)
    if pattern == 2:  # asymmetric bounded rational costs
        frozen = tuple(tuple(Q(y, k) for y in range(k)) for _ in range(m))
        first = tuple(tuple(Q(k - y, 2 * k) for y in range(k)) for _ in range(m))
        second = tuple(tuple(Q((y + 1) % k, k) for y in range(k)) for _ in range(m))
        return frozen, (first, second)
    if pattern == 3:  # single catastrophic outcome, different candidate conflicts
        frozen = tuple(tuple(Q(int(y == k - 1)) for y in range(k)) for _ in range(m))
        first = tuple(tuple(Q(int(y == 0)) for y in range(k)) for _ in range(m))
        second = tuple(tuple(Q(int(y == 1)) for y in range(k)) for _ in range(m))
        return frozen, (first, second)
    # Strict tiny benefits for one candidate; adverse second candidate.
    frozen = tuple((Q(1, 2),) * k for _ in range(m))
    first = tuple((Q(1, 2) - Q(1, 1024),) * k for _ in range(m))
    second = tuple((Q(3, 4),) * k for _ in range(m))
    return frozen, (first, second)


def bounded_grid() -> list[Problem]:
    cases = []
    for k in (3, 4, 5):
        for masses in ((Q(1),), (Q(1, 2), Q(1, 2)), (Q(7, 8), Q(1, 8)), (Q(1), Q(0))):
            m, d = len(masses), len(masses) * k
            for operator, name in enumerate(("none", "redundant", "face", "full", "inconsistent")):
                frozen, candidates = _costs(k, m, (k + operator) % 5)
                if name == "none":
                    rows, values = (), ()
                elif name == "redundant":
                    rows = (coordinate(d, 0), coordinate(d, 0))
                    values = (Q(1, k), Q(1, k))
                elif name == "face":
                    rows, values = (coordinate(d, 0),), (Q(0),)
                elif name == "full":
                    rows = tuple(coordinate(d, i) for i in range(d))
                    values = (Q(1, k),) * d
                else:
                    rows = (coordinate(d, 0), coordinate(d, 0))
                    values = (Q(0), Q(1))
                cases.append(Problem(
                    f"grid_K{k}_q{'_'.join(str(x).replace('/', '-') for x in masses)}_{name}",
                    k, masses, frozen, candidates, rows, values,
                    tuple(f"Exploratory toy {name} operator row {i}; no natural availability claim" for i in range(len(rows))),
                    tuple(f"Remove row {i}; remove its duplicate too for effective redundant-row ablation" for i in range(len(rows))),
                ))
    return cases
