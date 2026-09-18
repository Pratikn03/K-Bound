"""Install metadata must cover the public package without research workloads."""
from pathlib import Path
import tomllib


def test_public_runtime_dependencies_are_explicit_not_broad_research_requirements():
    config = tomllib.loads((Path(__file__).resolve().parents[1] / "pyproject.toml").read_text())
    dependencies = config["project"].get("dependencies", [])
    for name in ("numpy", "scipy", "scikit-learn", "PyYAML"):
        assert any(value.startswith(name + ">=") for value in dependencies), name
    assert not any("torch" in value or "tensorflow" in value for value in dependencies)
    assert "dependencies" not in config.get("tool", {}).get("setuptools", {}).get("dynamic", {})
