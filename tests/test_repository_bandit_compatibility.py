"""Bandit's no-op dependency warning cannot bypass other release failures."""

import subprocess
import sys
from pathlib import Path

import pytest

from docs.research.kbound.scripts import run_repository_verification as runner

KNOWN_MESSAGE = (
    "The verify_requirements argument is now a no-op and is deprecated for removal. Remove the argument from calls."
)


def bandit_command(monkeypatch: pytest.MonkeyPatch, repo: Path) -> list[str]:
    """Intercept only transport; test the actual command built by the quality gate."""
    commands = []
    with monkeypatch.context() as patch:
        patch.setattr(runner, "_run", lambda command, **kwargs: commands.append(command))
        runner._run_quality_gates(repo=repo, python=sys.executable)
    assert len(commands) == 5
    return commands[-1]


@pytest.mark.parametrize(
    "message,category,module,allowed",
    [
        (KNOWN_MESSAGE, "DeprecationWarning", "stevedore.extension", True),
        ("An unrelated deprecation", "DeprecationWarning", "stevedore.extension", False),
        (KNOWN_MESSAGE + " Additional risk.", "DeprecationWarning", "stevedore.extension", False),
        (KNOWN_MESSAGE + "\n", "DeprecationWarning", "stevedore.extension", False),
        (KNOWN_MESSAGE.lower(), "DeprecationWarning", "stevedore.extension", False),
        (KNOWN_MESSAGE, "UserWarning", "stevedore.extension", False),
        (KNOWN_MESSAGE, "DeprecationWarning", "first_party.module", False),
        (KNOWN_MESSAGE, "DeprecationWarning", "stevedore.extension_extra", False),
    ],
)
def test_bandit_child_allows_only_exact_dependency_warning(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capfd: pytest.CaptureFixture[str],
    message: str,
    category: str,
    module: str,
    allowed: bool,
) -> None:
    # A stand-in entrypoint exercises real Python warning handling/subprocess exit.
    (tmp_path / "bandit.py").write_text(
        f"import warnings\nwarnings.warn_explicit({message!r}, {category}, "
        f'filename="fixture.py", lineno=1, module={module!r})\nprint("scanner executed")\n'
    )
    command = bandit_command(monkeypatch, tmp_path)
    if allowed:
        runner._run(command, repo=tmp_path)
        assert "scanner executed" in capfd.readouterr().out
    else:
        with pytest.raises(subprocess.CalledProcessError):
            runner._run(command, repo=tmp_path)
        assert "scanner executed" not in capfd.readouterr().out


def test_non_bandit_child_retains_strict_warning_policy(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("PYTHONWARNINGS", "ignore")
    code = (
        f"import warnings; warnings.warn_explicit({KNOWN_MESSAGE!r}, "
        'DeprecationWarning, filename="fixture.py", lineno=1, module="stevedore.extension")'
    )
    with pytest.raises(subprocess.CalledProcessError):
        runner._run([sys.executable, "-c", code], repo=tmp_path)


@pytest.mark.parametrize("source,finding", [("ANSWER = 42\n", False), ("eval(input())\n", True)])
def test_actual_bandit_scans_and_still_rejects_security_findings(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capfd: pytest.CaptureFixture[str],
    source: str,
    finding: bool,
) -> None:
    # Preserve the real gate's two scan roots; no repository data is examined.
    (tmp_path / "kga").mkdir()
    (tmp_path / "deploy" / "api").mkdir(parents=True)
    (tmp_path / "kga" / "fixture.py").write_text(source)
    (tmp_path / "pyproject.toml").write_text('[tool.bandit]\nexclude_dirs = ["._*"]\n')
    command = bandit_command(monkeypatch, tmp_path)
    if finding:
        with pytest.raises(subprocess.CalledProcessError) as exc:
            runner._run(command, repo=tmp_path)
        assert exc.value.returncode == 1
        assert "B307" in capfd.readouterr().out
    else:
        runner._run(command, repo=tmp_path)
        assert "DeprecationWarning" not in capfd.readouterr().err
