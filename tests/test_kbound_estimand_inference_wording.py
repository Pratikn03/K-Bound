"""Guard the cell estimand and display-only CCT-20 inference clarification.

The table generator writes artifacts at module import time. Compile only its
two required functions so these tests never execute the full generator, load
datasets, or write into the maintained release.
"""

from __future__ import annotations

import ast
import hashlib
import json
import math
import os
import re
from collections.abc import Callable
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
KBOUND = ROOT / "docs/research/kbound"
TABLE_SCRIPT = KBOUND / "scripts/make_tables.py"
FIGURE_SCRIPT = KBOUND / "scripts/plot_kga_interval_rule.py"
BODY = KBOUND / "kbound_submission_body.tex"
SUPPLEMENT = KBOUND / "kbound_submission_supplement.tex"
ABSTRACT = KBOUND / "kbound_abstract_core.tex"
SEALED_TABLE_PATH = KBOUND / "paper/generated/cct20_primary_table.tex"
RELEASE_PATH = KBOUND / "paper/generated/cct20_release_manifest.json"

OLD_HEADER = (
    r"Comparator & Baseline regret $-$ KGA regret & Bonferroni 97.5\% CI & "
    r"Exact $p$ & Holm $p$ & Reject at .05 \\"
)
LOCKED_ROWS = (
    r"Always adapt & 0.18190227 & [0.08501071, 0.31066374] & 0.00195312 & 0.00390625 & yes \\"
    "\n"
    r"Always freeze & 0 & [0, 0] & 1 & 1 & no \\"
    "\n"
)
SEALED_FIXTURE = (
    "% Historical generator comment, retained only in the sealed source.\n"
    r"\begin{tabular}{@{}lrrrrc@{}}" "\n"
    r"\toprule" "\n"
    + OLD_HEADER
    + "\n"
    + r"\midrule"
    + "\n"
    + LOCKED_ROWS
    + r"\bottomrule"
    + "\n"
    + r"\end{tabular}"
    + "\n"
).encode("ascii")


def _load_renderer() -> Callable[[], None]:
    """Extract function definitions without running any module-level code."""
    tree = ast.parse(TABLE_SCRIPT.read_text(encoding="utf-8"), filename=str(TABLE_SCRIPT))
    names = {"_load_json", "_write_cct20_primary_display_table"}
    functions = [
        node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in names
    ]
    assert {node.name for node in functions} == names
    assert len(functions) == len(names), "Renderer helpers must have unique definitions"
    module = ast.fix_missing_locations(ast.Module(body=functions, type_ignores=[]))
    namespace = {"hashlib": hashlib, "json": json, "os": os}
    exec(compile(module, str(TABLE_SCRIPT), "exec"), namespace)
    return namespace["_write_cct20_primary_display_table"]


def _prepare_renderer(
    tmp_path: Path,
    source: bytes = SEALED_FIXTURE,
    identity: dict | None = None,
) -> tuple[Callable[[], None], Path, Path, Path]:
    source_path = tmp_path / "cct20_primary_table.tex"
    manifest_path = tmp_path / "cct20_release_manifest.json"
    display_path = tmp_path / "cct20_primary_table_display.tex"
    source_path.write_bytes(source)
    if identity is None:
        identity = {"bytes": len(source), "sha256": hashlib.sha256(source).hexdigest()}
    manifest_path.write_text(
        json.dumps({"generated_artifacts": {"cct20_primary_table_tex": identity}}),
        encoding="ascii",
    )
    renderer = _load_renderer()
    renderer.__globals__.update(
        CCT_RELEASE=str(manifest_path),
        CCT_PRIMARY_TABLE=str(source_path),
        CCT_PRIMARY_DISPLAY_OUT=str(display_path),
    )
    return renderer, source_path, manifest_path, display_path


def _result_body(table: bytes) -> bytes:
    assert table.count(b"\\midrule") == 1
    return table.partition(b"\\midrule")[2]


def _normalized_tex(path: Path) -> str:
    # Commented-out caveats must not satisfy the manuscript wording contract.
    source = re.sub(r"(?<!\\)%[^\n]*", "", path.read_text(encoding="utf-8"))
    return " ".join(source.split())


def test_primary_display_changes_only_headings_and_preserves_locked_rows(tmp_path: Path) -> None:
    render, source_path, manifest_path, display_path = _prepare_renderer(tmp_path)
    manifest_before = manifest_path.read_bytes()

    render()
    display = display_path.read_bytes()

    assert _result_body(display) == _result_body(SEALED_FIXTURE)
    assert LOCKED_ROWS.encode("ascii") in display
    assert OLD_HEADER.encode("ascii") not in display
    assert rb"Nominal 97.5\% CI" in display
    assert rb"Sign-flip $p$ & Holm $p$ & Holm flag (.05)" in display
    assert hashlib.sha256(SEALED_FIXTURE).hexdigest().encode("ascii") in display
    assert b"% Historical generator comment" not in display
    assert source_path.read_bytes() == SEALED_FIXTURE
    assert manifest_path.read_bytes() == manifest_before

    render()
    assert display_path.read_bytes() == display, "Display regeneration must be deterministic"


def test_primary_display_preserves_current_release_authorities(tmp_path: Path) -> None:
    """Exercise the actual resident compact authority, never its upstream data."""
    source_before = SEALED_TABLE_PATH.read_bytes()
    release_before = RELEASE_PATH.read_bytes()
    identity = json.loads(release_before)["generated_artifacts"]["cct20_primary_table_tex"]
    render, _, _, display_path = _prepare_renderer(tmp_path, source_before, identity)

    render()

    assert _result_body(display_path.read_bytes()) == _result_body(source_before)
    assert SEALED_TABLE_PATH.read_bytes() == source_before
    assert RELEASE_PATH.read_bytes() == release_before


@pytest.mark.parametrize("mutation", ["same_size_numeric_change", "different_byte_count"])
def test_primary_display_rejects_changed_source_without_overwriting_output(
    tmp_path: Path, mutation: str
) -> None:
    render, source_path, _, display_path = _prepare_renderer(tmp_path)
    if mutation == "same_size_numeric_change":
        changed = SEALED_FIXTURE.replace(b"0.18190227", b"0.28190227")
        assert len(changed) == len(SEALED_FIXTURE)
    else:
        changed = SEALED_FIXTURE + b"\n"
    source_path.write_bytes(changed)
    sentinel = b"previous display must survive validation failure\n"
    display_path.write_bytes(sentinel)

    with pytest.raises(ValueError, match="differs from its sealed manifest"):
        render()

    assert display_path.read_bytes() == sentinel
    assert source_path.read_bytes() == changed


@pytest.mark.parametrize("field", ["bytes", "sha256"])
def test_primary_display_rejects_incorrect_manifest_identity(tmp_path: Path, field: str) -> None:
    identity = {"bytes": len(SEALED_FIXTURE), "sha256": hashlib.sha256(SEALED_FIXTURE).hexdigest()}
    identity[field] = len(SEALED_FIXTURE) + 1 if field == "bytes" else "0" * 64
    render, _, _, display_path = _prepare_renderer(tmp_path, identity=identity)

    with pytest.raises(ValueError, match="differs from its sealed manifest"):
        render()

    assert not display_path.exists()


@pytest.mark.parametrize(
    "source",
    [
        pytest.param(
            SEALED_FIXTURE.replace(b"Exact $p$", b"Different statistic"),
            id="changed-header",
        ),
        pytest.param(
            SEALED_FIXTURE.replace(
                OLD_HEADER.encode("ascii"), (OLD_HEADER + "\n" + OLD_HEADER).encode("ascii")
            ),
            id="duplicate-header",
        ),
        pytest.param(SEALED_FIXTURE.replace(b"\\midrule", b""), id="missing-midrule"),
        pytest.param(
            SEALED_FIXTURE.replace(b"\\midrule", b"\\midrule\n\\midrule"),
            id="duplicate-midrule",
        ),
    ],
)
def test_primary_display_rejects_changed_schema_even_with_matching_hash(
    tmp_path: Path, source: bytes
) -> None:
    # Recompute the fixture seal so this exercises layout validation, not hashing.
    render, source_path, _, display_path = _prepare_renderer(tmp_path, source)
    sentinel = b"previous display must survive schema failure\n"
    display_path.write_bytes(sentinel)

    with pytest.raises(ValueError, match="Unexpected sealed CCT-20 primary table layout"):
        render()

    assert display_path.read_bytes() == sentinel
    assert source_path.read_bytes() == source


def test_primary_display_rejects_result_body_normalization(tmp_path: Path) -> None:
    source = SEALED_FIXTURE.replace(
        b"\\midrule\n", b"\\midrule\n% A comment inside the sealed result body.\n"
    )
    render, _, _, display_path = _prepare_renderer(tmp_path, source)

    with pytest.raises(ValueError, match="changed a result row"):
        render()

    assert not display_path.exists()


@pytest.mark.parametrize(
    "fragment",
    [
        r"\Delta_j^{\mathrm{cell}} =S(f_a;\mathcal E_j)-S(f_0;\mathcal E_j).",
        "Regret is the policy's score shortfall from that oracle",
        r"|\widehat\Delta-\Delta^{\mathrm{cell}}|\le\varepsilon",
        "A population application would instead require coverage of",
        "No such sampling-error bound is established for the reported panels",
    ],
)
def test_shared_body_distinguishes_measured_cell_and_population_targets(fragment: str) -> None:
    # Formula spacing is immaterial; changing its target is not.
    assert "".join(fragment.split()) in "".join(_normalized_tex(BODY).split())


@pytest.mark.parametrize(
    "fragment",
    [
        "Bootstrap levels are nominal throughout",
        "valid marginal coverage of their constituent intervals",
        r"G\overset{d}{=}s\odot G",
        r"\quad\text{for every }",
        r"s\in\{-1,+1\}^{m}",
        "Exchangeability, marginal symmetry, or a single global sign symmetry alone is not sufficient",
        "Enumeration removes Monte Carlo error but does not verify this null model",
        r"Holm adjustment requires valid input $p$-values",
    ],
)
def test_shared_body_requires_nominal_bootstrap_and_coordinatewise_null(fragment: str) -> None:
    assert fragment in _normalized_tex(BODY)


@pytest.mark.parametrize(
    "fragment",
    [
        "observed evaluation-cell benefit",
        "empirical companion, not an implementation of this frontier",
        "labeled historical outcomes",
        "do not establish interval coverage",
        "population-risk protection on unseen natural shifts",
    ],
)
def test_shared_abstract_keeps_estimand_and_inference_caveats(fragment: str) -> None:
    assert fragment in _normalized_tex(ABSTRACT)


def test_cct20_manuscript_uses_display_without_reinterpreting_locked_pass() -> None:
    body = _normalized_tex(BODY) + " " + _normalized_tex(SUPPLEMENT)
    assert r"\input{paper/generated/cct20_primary_table_display.tex}" in body
    assert r"\input{paper/generated/cct20_primary_table.tex}" not in body
    assert r"nominal pointwise 95\% bootstrap lower endpoints" in body
    assert "The unchanged locked rule therefore returns pass on the stored cells" in body
    assert "not a finite-sample population-safety guarantee" in body
    assert r"The recorded verdict remains \CCTVerdict" in body


def test_interval_figure_is_an_explicitly_illustrative_static_rule() -> None:
    """Inspect the plotting contract without importing Matplotlib or rendering."""
    tree = ast.parse(FIGURE_SCRIPT.read_text(encoding="utf-8"), filename=str(FIGURE_SCRIPT))
    example_nodes = [
        node
        for node in tree.body
        if isinstance(node, ast.Assign)
        and any(isinstance(target, ast.Name) and target.id == "EXAMPLES" for target in node.targets)
    ]
    assert len(example_nodes) == 1
    examples = ast.literal_eval(example_nodes[0].value)
    assert [(action, prediction, radius) for action, prediction, radius, _, _ in examples] == [
        ("ADAPT", 0.04, 0.01),
        ("FREEZE", -0.04, 0.01),
        ("ABSTAIN", 0.005, 0.01),
    ]
    for action, prediction, radius, _, _ in examples:
        lower, upper = prediction - radius, prediction + radius
        expected = "ADAPT" if lower > 0 else "FREEZE" if upper < 0 else "ABSTAIN"
        assert action == expected

    labels = {}
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr in {"set_title", "set_xlabel"}
        ):
            assert node.func.attr not in labels
            labels[node.func.attr] = ast.literal_eval(node.args[0])
    assert "measured-cell benefit" in labels["set_xlabel"]
    assert "Illustration only" in labels["set_title"]
    assert "not an experimental result" in labels["set_title"]


def test_operational_fallback_is_not_a_certified_freeze() -> None:
    body = _normalized_tex(BODY) + " " + _normalized_tex(SUPPLEMENT)
    assert r"Missing/invalid feature & schema and finite-value check & retain $f_0$; abstain" in body
    assert "Batch too small & minimum-size contract & wait or abstain" in body
    assert "abstain or freeze" not in body
    assert "wait or freeze" not in body


def test_population_frontier_stays_inside_feasible_margin_range() -> None:
    source = (KBOUND / "scripts/make_submission_figures.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    frontier = next(
        node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "frontier"
    )
    parameters = frontier.body[0]
    assert isinstance(parameters, ast.Assign)
    assert isinstance(parameters.targets[0], ast.Tuple)
    assert [target.id for target in parameters.targets[0].elts] == ["b", "limit"]
    beta, limit = ast.literal_eval(parameters.value)
    assert 0 < beta < limit <= 0.5
    ticks = next(
        node for node in ast.walk(frontier)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "set_xticks"
    )
    assert all(abs(value) <= limit for value in ast.literal_eval(ticks.args[0]))
    assert r"illustration: $\beta=0.1$" in source
    assert 'if args.frontier_only:\n        frontier()\n    else:' in source
    build = (KBOUND / "scripts/build_pdfs.sh").read_text(encoding="utf-8")
    assert '"$PY" scripts/make_submission_figures.py --frontier-only' in build


def _load_verdict_checker() -> dict:
    """Load pure prose checks without importing or walking release authorities."""
    path = ROOT / "src/scripts/validate_manuscript_claims.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    functions = {
        "_strip_tex_comments_for_claims", "_normalize_claim_text", "_claim_clause",
        "_has_unsafe_match", "_is_finite_number", "_validate_cct20_verdict_usage",
    }
    constants = {"CCT20_VERDICT_CLAIMS", "REQUIRED_CCT20_NUMBER_MACROS"}
    selected = [
        node for node in tree.body
        if (isinstance(node, ast.FunctionDef) and node.name in functions)
        or (
            isinstance(node, ast.Assign)
            and any(isinstance(target, ast.Name) and target.id in constants for target in node.targets)
        )
    ]
    assert {node.name for node in selected if isinstance(node, ast.FunctionDef)} == functions
    namespace = {"re": re, "math": math}
    module = ast.fix_missing_locations(ast.Module(body=selected, type_ignores=[]))
    exec(compile(module, str(path), "exec"), namespace)
    return namespace


def test_prose_may_qualify_the_sealed_claim_without_expanding_it() -> None:
    namespace = _load_verdict_checker()
    body = BODY.read_text(encoding="utf-8")
    assert r"\CCTManuscriptClaim" not in body
    # The historical macro is still required in sealed generated metadata.
    assert "CCTManuscriptClaim" in namespace["REQUIRED_CCT20_NUMBER_MACROS"]
    release = json.loads(RELEASE_PATH.read_text(encoding="utf-8"))
    assert namespace["_validate_cct20_verdict_usage"](
        body, namespace["_normalize_claim_text"](body), release,
    ) == []


@pytest.mark.parametrize("context", ["CCT-20 retains its verdict.", "% " + r"\CCTVerdict"])
def test_completed_prose_still_requires_an_active_verdict_macro(context: str) -> None:
    namespace = _load_verdict_checker()
    problems = namespace["_validate_cct20_verdict_usage"](
        context, namespace["_normalize_claim_text"](context),
        {"verdict": {"code": "SAFE_UTILITY_ONLY"}},
    )
    assert "completed CCT-20 manuscript must consume generated " + r"\CCTVerdict" in problems


@pytest.mark.parametrize(
    "overclaim",
    [
        "CCT-20: KGA makes both ADAPT and FREEZE decisions.",
        "CCT-20: KGA beats both fixed policies.",
    ],
)
def test_scoped_prose_change_does_not_disable_exposure_or_comparator_guards(overclaim: str) -> None:
    namespace = _load_verdict_checker()
    context = r"\CCTVerdict. " + overclaim
    release = json.loads(RELEASE_PATH.read_text(encoding="utf-8"))
    problems = namespace["_validate_cct20_verdict_usage"](
        context, namespace["_normalize_claim_text"](context), release,
    )
    assert problems, "Overclaims about the unchanged all-freeze result must still fail"
