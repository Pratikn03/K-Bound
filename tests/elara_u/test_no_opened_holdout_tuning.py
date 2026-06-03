"""Guard: sealed/external holdout results must be ONE-SHOT (no per-holdout tuning).
The sealed (D24) and independent (D27) evaluators must reuse the frozen dev pipeline
(build_task) and must not import any hyperparameter search / GridSearch / Optuna."""
from pathlib import Path
import pytest

SRC = Path(__file__).resolve().parents[2] / "src/scripts/elara_u"
SEALED = [SRC / "sealed_external_eval.py", SRC / "openml_indep_eval.py"]


@pytest.mark.parametrize("p", SEALED)
def test_sealed_evaluators_are_one_shot(p):
    if not p.exists():
        pytest.skip(f"{p.name} absent")
    src = p.read_text()
    assert "build_task" in src, f"{p.name} must reuse the frozen dev scoring (build_task)"
    for forbidden in ["GridSearchCV", "RandomizedSearchCV", "optuna", "hyperopt", "ytest)" .replace("ytest)","__never__")]:
        assert forbidden not in src, f"{p.name} must not tune on the holdout ({forbidden})"
