"""Static regression checks for device selection and GDE state initialization.

This test intentionally avoids importing the full AETTA stack or loading data.
It can run before an expensive smoke evaluation and catches the source-path
failure that previously appeared only after dataset/model setup.
"""
import ast
from pathlib import Path


SOURCE = Path(__file__).resolve().parents[1] / "learner" / "dnn.py"


def _dnn_init_assignments() -> set[str]:
    tree = ast.parse(SOURCE.read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "DNN")
    init = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "__init__")
    names = set()
    for node in ast.walk(init):
        for target in getattr(node, "targets", []):
            if isinstance(target, ast.Attribute) and isinstance(target.value, ast.Name) and target.value.id == "self":
                names.add(target.attr)
    return names


def test_gde_queue_initialized_for_every_dnn_instance():
    names = _dnn_init_assignments()
    assert "prev_net_state_queue" in names
    assert "sa_ref_net_1" in names


def test_module_device_has_mps_fallback():
    text = SOURCE.read_text()
    assert 'torch.backends.mps.is_available()' in text
    assert 'else ("mps" if' in text
