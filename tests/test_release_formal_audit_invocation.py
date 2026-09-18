from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from docs.research.kbound.formal import formal_audit

ROOT = Path(__file__).resolve().parents[1]


def test_population_capstones_are_in_the_theorem_map() -> None:
    missing = formal_audit.theorem_map_checks()
    assert "populationRadius_nonneg" not in missing
    assert "populationDecision_abstain_of_zero_mem" not in missing


@pytest.fixture
def release_test_phase(tmp_path: Path):
    """Run the actual test dispatch and formal wrapper, with expensive peers stubbed."""
    repo = tmp_path / "synthetic checkout"
    kb = repo / "docs/research/kbound"
    for relative in ("runbooks/release_candidate.sh", "formal/build.sh"):
        destination = kb / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes((ROOT / "docs/research/kbound" / relative).read_bytes())

    commands = tmp_path / "bin"
    commands.mkdir()
    # Source-identity discovery is outside this test; never inspect the real Git index.
    git = commands / "git"
    git.write_text(
        '#!/bin/sh\ncase "$*" in\n'
        '  *--show-toplevel) printf "%s\\n" "$RELEASE_FIXTURE_ROOT" ;;\n'
        '  *--verify\\ HEAD) printf "%040d\\n" 1 ;;\n'
        '  *) exit 91 ;;\nesac\n',
        encoding="utf-8",
    )
    git.chmod(0o755)
    python = commands / "fixture-python"
    python.write_text(
        f"#!{sys.executable}\n"
        "import os, sys\n"
        "from pathlib import Path\n"
        "args = sys.argv[1:]\n"
        "if args == ['-']:\n"
        "    sys.stdin.read()\n"
        "elif args[:2] in (['-m', 'pytest'], ['-m', 'kbound_repro.release_checks']):\n"
        "    pass\n"
        "elif args and Path(args[0]).name == 'run_repository_verification.py':\n"
        "    pass\n"
        "elif args and Path(args[0]).name == 'formal_audit.py':\n"
        "    os.execv(sys.executable, [sys.executable, *args])\n"
        "else:\n"
        "    raise SystemExit('unexpected release command: ' + repr(args))\n",
        encoding="utf-8",
    )
    python.chmod(0o755)
    fallback = commands / "python3"
    fallback.write_text("#!/bin/sh\nexit 94\n", encoding="utf-8")
    fallback.chmod(0o755)
    # The command boundary requires a kernel build and the maintained scoped audit.
    # Removing either flag, adding full-foundations, or losing --json-out fails.
    (kb / "formal/formal_audit.py").write_text(
        "import argparse, json, os\n"
        "from pathlib import Path\n"
        "parser = argparse.ArgumentParser()\n"
        "parser.add_argument('--build', action='store_true', required=True)\n"
        "parser.add_argument('--strict-core', action='store_true', required=True)\n"
        "parser.add_argument('--json-out', type=Path, required=True)\n"
        "args = parser.parse_args()\n"
        "status = int(os.environ.get('RELEASE_FIXTURE_AUDIT_STATUS', '0'))\n"
        "if status:\n"
        "    raise SystemExit(status)\n"
        "args.json_out.write_text(json.dumps({'build_checked': args.build,\n"
        "    'strict_core_checked': args.strict_core}))\n",
        encoding="utf-8",
    )
    receipt = kb / "audits/formal_foundations_2026_08_31.json"
    receipt.parent.mkdir()
    env = {
        **{
            key: value for key, value in os.environ.items()
            if key not in {"KBOUND_PYTHON", "PYTHON", "FORMAL_PYTHON"}
        },
        "PATH": str(commands) + os.pathsep + os.environ["PATH"],
        "KBOUND_VERIFY_TOOLCHAIN": "0",
        "KBOUND_STRICT_SOURCE_SEAL": "1",
        "RELEASE_FIXTURE_ROOT": str(repo),
        "RELEASE_FIXTURE_AUDIT_STATUS": "0",
        "PYTHONPATH": "",
    }

    def run(
        *, audit_status: int = 0, python_variable: str = "KBOUND_PYTHON"
    ) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            ["/bin/bash", str(kb / "runbooks/release_candidate.sh"), "test"],
            cwd=repo,
            env={
                **env,
                python_variable: str(python),
                "RELEASE_FIXTURE_AUDIT_STATUS": str(audit_status),
            },
            capture_output=True,
            text=True,
            timeout=20,
        )

    return run, kb, receipt


@pytest.mark.parametrize("python_variable", ["KBOUND_PYTHON", "PYTHON"])
def test_release_test_runs_strict_audit_with_kernel_build(
    release_test_phase, python_variable: str
) -> None:
    run, _, receipt = release_test_phase
    result = run(python_variable=python_variable)
    assert result.returncode == 0, result.stdout + result.stderr
    assert json.loads(receipt.read_text(encoding="utf-8")) == {
        "build_checked": True,
        "strict_core_checked": True,
    }
    assert "completed MODE=test with no warnings" in result.stdout


def test_release_test_propagates_formal_audit_failure(release_test_phase) -> None:
    run, _, receipt = release_test_phase
    receipt.write_text("prior audit receipt\n", encoding="utf-8")
    result = run(audit_status=23)
    assert result.returncode == 23, result.stdout + result.stderr
    assert "completed MODE=test" not in result.stdout
    assert receipt.read_text(encoding="utf-8") == "prior audit receipt\n"


@pytest.mark.parametrize("missing", ["build.sh", "formal_audit.py"])
def test_release_test_requires_formal_audit_files(release_test_phase, missing: str) -> None:
    run, kb, receipt = release_test_phase
    (kb / "formal" / missing).unlink()
    result = run()
    assert result.returncode != 0, result.stdout + result.stderr
    assert "completed MODE=test" not in result.stdout
    assert not receipt.exists()
