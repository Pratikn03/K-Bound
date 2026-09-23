"""The publication guard must distinguish historical LOO from current split calibration."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

VALIDATOR_PATH = Path(__file__).resolve().parents[1] / "src/scripts/validate_manuscript_claims.py"
SPEC = importlib.util.spec_from_file_location("manuscript_calibration_scope_validator", VALIDATOR_PATH)
assert SPEC is not None and SPEC.loader is not None
VALIDATOR = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(VALIDATOR)


def test_current_disjoint_protocol_does_not_require_historical_loo_wording(capsys: pytest.CaptureFixture[str]) -> None:
    assert VALIDATOR.main([]) == 0, capsys.readouterr().out


@pytest.mark.parametrize(
    ("original", "replacement", "expected_problem"),
    [
        (
            "used in historical controlled grids",
            "used in the current primary CIFAR table",
            "Protocol A must remain historical",
        ),
        (
            "Protocol B:Three-way cell-outcome-disjoint cross-fitting",
            "Protocol B: Leave-one-condition-out cross-fitted empirical residual calibration",
            "current Protocol B must specify three-way cell-outcome-disjoint cross-fitting",
        ),
        (
            "strictly disjoint estimator-fit and residual-calibration subsets",
            "overlapping estimator-fit and residual-calibration subsets",
            "current Protocol B must separate estimator-fit and residual-calibration subsets",
        ),
        (
            "scored cell outcomes enter neither predictor training nor conformal calibration",
            "scored cell outcomes enter both predictor training and conformal calibration",
            "current Protocol B must exclude scored outcomes from fitting and calibration",
        ),
        (
            "used in the current primary CIFAR table",
            "used only in historical controlled grids",
            "Protocol B must identify its current primary CIFAR use",
        ),
        (
            "scored cell's outcome can indirectly influence other residuals",
            "scored cell's outcome cannot influence any residual",
            "historical Protocol A must disclose indirect scored-outcome dependence",
        ),
    ],
)
def test_historical_loo_phrase_cannot_mask_a_wrong_current_protocol(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    original: str,
    replacement: str,
    expected_problem: str,
) -> None:
    sources, corpus = VALIDATOR._active_manuscript_corpus(VALIDATOR.ROOT, VALIDATOR.ACTIVE_SOURCES)
    assert original in corpus
    changed = corpus.replace(original, replacement, 1)
    changed += "\nHistorical terminology: leave-one-condition-out cross-fitted empirical residual calibration.\n"
    monkeypatch.setattr(VALIDATOR, "_active_manuscript_corpus", lambda root, paths: (sources, changed))

    result = VALIDATOR.main([])

    assert result == 1
    assert expected_problem in capsys.readouterr().out


@pytest.mark.parametrize("placement", ["same_protocol_paragraph", "later_active_context"])
@pytest.mark.parametrize(
    "false_label",
    ["Leave-one-condition-out calibration", "Leave-one-cell-out calibration", "Pooled LOO calibration"],
)
def test_truthful_protocol_text_cannot_mask_an_additional_current_loo_label(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    placement: str,
    false_label: str,
) -> None:
    sources, corpus = VALIDATOR._active_manuscript_corpus(VALIDATOR.ROOT, VALIDATOR.ACTIVE_SOURCES)
    false_statement = f"Protocol B: {false_label}."
    if placement == "same_protocol_paragraph":
        anchor = "scored cell outcomes enter neither predictor training nor conformal calibration."
        assert anchor in corpus
        changed = corpus.replace(anchor, f"{anchor} {false_statement}", 1)
    else:
        changed = corpus + "\n\\paragraph{Current calibration} " + false_statement
    monkeypatch.setattr(VALIDATOR, "_active_manuscript_corpus", lambda root, paths: (sources, changed))

    result = VALIDATOR.main([])

    assert result == 1
    assert "current Protocol B must not be labeled leave-one-out calibration" in capsys.readouterr().out


@pytest.mark.parametrize("first", ["A", "B"])
def test_protocol_guards_are_independent_of_exposition_order(monkeypatch, capsys, first):
    sources, corpus = VALIDATOR._active_manuscript_corpus(VALIDATOR.ROOT, VALIDATOR.ACTIVE_SOURCES)
    start = corpus.index(r"\emph{Protocol B:")
    middle = corpus.index(r"\emph{Protocol A:", start)
    end = corpus.index(r"\begin{algorithm}", middle)
    b, a = corpus[start:middle], corpus[middle:end]
    changed = corpus[:start] + (a + b if first == "A" else b + a) + corpus[end:]
    monkeypatch.setattr(VALIDATOR, "_active_manuscript_corpus", lambda root, paths: (sources, changed))
    assert VALIDATOR.main([]) == 0, capsys.readouterr().out
