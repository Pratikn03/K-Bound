"""Display-only QA using AST-extracted helpers, never make_tables' module body.

The generator has import-time writes. These tests execute only its named
format/validation functions, inject in-memory authorities, and write solely to
pytest temporary directories. No raw outcomes, models, inference or fits run.
"""

from __future__ import annotations

import ast
import copy
import json
import math
import os
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
GENERATOR = ROOT / "docs/research/kbound/scripts/make_tables.py"
GENERATED = ROOT / "docs/research/kbound/paper/generated"
CANONICAL = ROOT / "experiments/kbound/results/reconciled_panels_v1/canonical_panel_results.json"
HELPERS = {
    "f", "_display_score_fields", "_render_metric_table",
    "_write_metric_separated_display_tables", "_write_cct20_safe_utility_display_table",
}


def resident_text(path: Path) -> str:
    if getattr(path.stat(), "st_flags", 0) & 0x40000000:
        raise RuntimeError(f"nonresident test authority; no restore attempted: {path}")
    return path.read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def authorities() -> tuple[dict, dict]:
    return (
        json.loads(resident_text(CANONICAL)),
        json.loads(resident_text(GENERATED / "cct20_release_manifest.json")),
    )


@pytest.fixture
def helpers(tmp_path: Path, authorities) -> dict:
    tree = ast.parse(resident_text(GENERATOR), filename=str(GENERATOR))
    selected = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in HELPERS]
    assert {node.name for node in selected} == HELPERS
    assert len(selected) == len(HELPERS)
    # No original imports, global expressions, assignments, or top-level calls
    # enter this compiled module. All paths/data below are explicitly injected.
    functions_only = ast.Module(body=selected, type_ignores=[])
    output = tmp_path / "paper/generated"
    output.mkdir(parents=True)
    canonical, cct = authorities
    namespace = {
        "math": math,
        "os": os,
        "canonical": copy.deepcopy(canonical),
        "KBOUND": str(tmp_path),
        "CCT_RELEASE": "in-memory-sealed-cct20",
        "_load_json": lambda path: copy.deepcopy(cct),
        "output": output,
    }
    exec(compile(functions_only, str(GENERATOR), "exec"), namespace)
    return namespace


def score() -> dict:
    return {
        "n": 10,
        "regret": {"kga": 0.0123456, "always_adapt": 0.2345678, "always_freeze": 0.3456789},
        "fa_u": 0.0123456,
        "decision_coverage": 0.7,
        "adapt_count": 4,
        "freeze_count": 3,
        "abstain_count": 3,
    }


def rendered_rows(source: str) -> list[str]:
    return [line for line in source.splitlines() if " & " in line and not line.startswith(("Candidate &", "Protocol &", "Comparator &"))]


def expected_row(label: str, recorded: dict, *, primary=False, aggregate=False) -> str:
    n = recorded["n_domain_seed_units" if aggregate else "n"]
    columns = [label, str(n), *(format(recorded["regret"][key], ".4f") for key in ("kga", "always_adapt", "always_freeze"))]
    if primary:
        columns.append("/".join(str(recorded[key]) for key in ("adapt_count", "freeze_count", "abstain_count")))
    columns.extend(format(recorded[key], ".4f") for key in ("fa_u", "decision_coverage"))
    return " & ".join(columns) + r" \\"


def test_actual_outputs_match_helpers_without_importing_generator_body(helpers) -> None:
    helpers["_write_metric_separated_display_tables"]()
    for name in (
        "kbound_primary_accuracy_table.tex",
        "kbound_auxiliary_accuracy_table.tex",
        "kbound_auxiliary_balanced_accuracy_table.tex",
    ):
        assert (helpers["output"] / name).read_text() == resident_text(GENERATED / name)


def test_primary_keeps_all_three_candidates_and_adverse_sar(helpers) -> None:
    helpers["_write_metric_separated_display_tables"]()
    text = (helpers["output"] / "kbound_primary_accuracy_table.tex").read_text()
    recorded = helpers["canonical"]["panels"]["cifar10c"]["panel"]["candidates"]
    assert rendered_rows(text) == [
        expected_row(label, recorded[candidate], primary=True)
        for candidate, label in (("tent", "Tent"), ("eata", "EATA"), ("sar", "SAR"))
    ]
    assert recorded["sar"]["regret"]["kga"] > recorded["sar"]["regret"]["always_adapt"]
    assert "SAR & 2160 & 0.0016 & 0.0003 & 0.1405 & 1446/0/714" in text
    assert "Commitment rate" in text and "Coverage" not in text


def test_auxiliary_metric_membership_and_each_recorded_value_are_preserved(helpers) -> None:
    helpers["_write_metric_separated_display_tables"]()
    panel = helpers["canonical"]["panels"]
    accuracy_expected = [
        ("Office-Home primary", panel["officehome"]["primary"]["exact_rank_transfer_score"], False),
        ("Office-Home stream seeds", panel["officehome"]["test_stream_seed_replication"]["exact_rank_transfer_score"], False),
        *[(f"ImageNet-C {label}", panel["imagenetc"]["panel"]["candidates"][candidate], False)
          for candidate, label in (("tent", "Tent"), ("eata", "EATA"), ("sar", "SAR"))],
        ("PACS (aggregate)", panel["pacs"]["pooled_domain_seed_mean"], True),
        ("CIFAR-10.1", panel["cifar101"]["replay"]["exact_rank_transfer_score"], False),
    ]
    balanced_expected = [
        ("ImageNet-R backbones", panel["imagenet_r"]["panel"]["architecture_panel_aggregate"], False),
        ("Camelyon17 OOD", panel["camelyon17"]["ood"]["replay"]["exact_rank_transfer_score"], False),
        *[(f"Camelyon17 B--v2 {label}", panel["camelyon17"]["b_v2_diagnostic"]["panel"]["candidates"][candidate], False)
          for candidate, label in (("tent", "Tent"), ("eata", "EATA"), ("sar", "SAR"))],
        ("RxRx1 model seed 0", panel["rxrx1"]["primary_model_seed0"]["exact_rank_transfer_score"], False),
    ]
    for filename, expected in (
        ("kbound_auxiliary_accuracy_table.tex", accuracy_expected),
        ("kbound_auxiliary_balanced_accuracy_table.tex", balanced_expected),
    ):
        source = (helpers["output"] / filename).read_text()
        assert rendered_rows(source) == [expected_row(label, recorded, aggregate=aggregate) for label, recorded, aggregate in expected]
    accuracy_labels = {label for label, _, _ in accuracy_expected}
    balanced_labels = {label for label, _, _ in balanced_expected}
    assert accuracy_labels.isdisjoint(balanced_labels)
    assert len(accuracy_labels) == 7 and len(balanced_labels) == 6


def test_pacs_score_unit_never_replaces_its_decision_denominator(helpers) -> None:
    aggregate = {
        "n_domain_seed_units": 12,
        "n": 999,
        "regret": {"kga": 0.043136388063430786, "always_adapt": 0.01763668394199124, "always_freeze": 0.04460611166777434},
        "fa_u": 2 / 216,
        "decision_coverage": 176 / 216,
        "adapt_count": "not replayable",
    }
    original = copy.deepcopy(aggregate)
    n, values = helpers["_display_score_fields"](aggregate, aggregate=True)
    assert n == 12
    assert values[-2:] == [2 / 216, 176 / 216]
    text = helpers["_render_metric_table"]([("PACS (aggregate)", aggregate, True)])
    assert rendered_rows(text) == [r"PACS (aggregate) & 12 & 0.0431 & 0.0176 & 0.0446 & 0.0093 & 0.8148 \\"]
    assert aggregate == original


def test_iwildcam_numerical_block_is_never_accessed_or_displayed(helpers) -> None:
    class Withheld:
        def __getitem__(self, key):
            raise AssertionError("withheld iWildCam numeric authority must not be read")

    helpers["canonical"]["panels"]["iwildcam"] = Withheld()
    helpers["_write_metric_separated_display_tables"]()
    for path in helpers["output"].glob("*.tex"):
        assert "iwildcam" not in path.read_text().lower()


@pytest.mark.parametrize("invalid", [0, -1, 10.0, True, None])
def test_invalid_evaluation_counts_fail(helpers, invalid) -> None:
    record = score()
    record["n"] = invalid
    with pytest.raises(ValueError, match="positive evaluation-unit count"):
        helpers["_display_score_fields"](record)


@pytest.mark.parametrize("invalid", [float("nan"), float("inf"), float("-inf"), -0.1, True, None, "0.1"])
@pytest.mark.parametrize("field", ["regret", "fa_u", "decision_coverage"])
def test_nonfinite_negative_or_malformed_scores_fail(helpers, invalid, field) -> None:
    record = score()
    if field == "regret":
        record["regret"]["kga"] = invalid
    else:
        record[field] = invalid
    with pytest.raises(ValueError, match="invalid recorded score"):
        helpers["_display_score_fields"](record)


@pytest.mark.parametrize("field", ["fa_u", "decision_coverage"])
def test_frequencies_above_one_fail(helpers, field) -> None:
    record = score()
    record[field] = 1.1
    with pytest.raises(ValueError, match="frequencies"):
        helpers["_display_score_fields"](record)


@pytest.mark.parametrize("invalid", [-1, True, None, 4.5])
def test_invalid_action_counts_fail(helpers, invalid) -> None:
    record = score()
    record["adapt_count"] = invalid
    with pytest.raises(ValueError, match="invalid action counts"):
        helpers["_display_score_fields"](record)


def test_action_sum_and_commitment_mismatches_fail(helpers) -> None:
    record = score()
    record["adapt_count"] += 1
    with pytest.raises(ValueError, match="action counts do not match"):
        helpers["_display_score_fields"](record)
    record = score()
    record["decision_coverage"] = 0.6
    with pytest.raises(ValueError, match="commitment rate disagrees"):
        helpers["_display_score_fields"](record)


def test_all_rows_are_validated_before_any_display_output_is_replaced(helpers) -> None:
    first = helpers["output"] / "kbound_primary_accuracy_table.tex"
    first.write_text("preexisting display", encoding="ascii")
    helpers["canonical"]["panels"]["rxrx1"]["primary_model_seed0"]["exact_rank_transfer_score"]["n"] = 0
    with pytest.raises(ValueError):
        helpers["_write_metric_separated_display_tables"]()
    assert first.read_text() == "preexisting display"
    assert list(helpers["output"].iterdir()) == [first]


def test_four_decimal_rendering_preserves_fields_and_all_actions(helpers) -> None:
    recorded = score()
    before = copy.deepcopy(recorded)
    source = helpers["_render_metric_table"]([("Synthetic", recorded, False)], primary=True)
    assert rendered_rows(source) == [r"Synthetic & 10 & 0.0123 & 0.2346 & 0.3457 & 4/3/3 & 0.0123 & 0.7000 \\"]
    assert recorded == before


def test_cct_display_uses_exact_sealed_pointwise_95_endpoint(helpers, authorities) -> None:
    sealed = authorities[1]
    before = copy.deepcopy(sealed)
    helpers["_write_cct20_safe_utility_display_table"]()
    source = (helpers["output"] / "cct20_safe_utility_display.tex").read_text()
    assert source == resident_text(GENERATED / "cct20_safe_utility_display.tex")
    safe = sealed["safe_utility"]
    expected = []
    for name, label, threshold in (("versus_always_adapt", "Always adapt", 0.0), ("versus_always_freeze", "Always freeze", -0.005)):
        item = safe[name]
        lower, upper = item["pointwise_95_ci"]
        expected.append(f"{label} & {item['point_estimate']:.4f} & [{lower:.4f}, {upper:.4f}] & $L>{threshold:.4f}$ " + r"\\")
    assert rendered_rows(source) == expected
    assert "Nominal 95\\% CI" in source
    assert safe["passes"] is True
    assert sealed == before


@pytest.mark.parametrize("contrast,threshold", [("versus_always_adapt", 0.0), ("versus_always_freeze", -0.005)])
def test_safe_utility_is_strict_at_each_threshold_not_rounded_or_nonstrict(helpers, authorities, contrast, threshold) -> None:
    changed = copy.deepcopy(authorities[1])
    changed["safe_utility"][contrast]["pointwise_95_ci"][0] = threshold
    changed["safe_utility"]["passes"] = True
    helpers["_load_json"] = lambda path: changed
    with pytest.raises(ValueError, match="contradicts its strict rule"):
        helpers["_write_cct20_safe_utility_display_table"]()
    changed["safe_utility"]["passes"] = False
    helpers["_write_cct20_safe_utility_display_table"]()
    changed["safe_utility"][contrast]["pointwise_95_ci"][0] = threshold + 1e-8
    changed["safe_utility"]["passes"] = True
    helpers["_write_cct20_safe_utility_display_table"]()


@pytest.mark.parametrize("change", ["sign", "margin", "flag", "point_nan", "bound_inf", "reversed"])
def test_invalid_safe_utility_contract_never_writes_display(helpers, authorities, change) -> None:
    altered = copy.deepcopy(authorities[1])
    safe = altered["safe_utility"]
    if change == "sign":
        safe["contrast_sign"] = "kga_minus_baseline"
    elif change == "margin":
        safe["frozen_noninferiority_margin"] = -0.01
    elif change == "flag":
        safe["passes"] = 1
    elif change == "point_nan":
        safe["versus_always_adapt"]["point_estimate"] = float("nan")
    elif change == "bound_inf":
        safe["versus_always_adapt"]["pointwise_95_ci"][1] = float("inf")
    else:
        safe["versus_always_adapt"]["pointwise_95_ci"] = [0.3, 0.1]
    helpers["_load_json"] = lambda path: altered
    with pytest.raises(ValueError):
        helpers["_write_cct20_safe_utility_display_table"]()
    assert not list(helpers["output"].iterdir())
