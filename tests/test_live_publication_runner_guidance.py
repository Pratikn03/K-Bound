from __future__ import annotations

import os
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EDGE_RUNNER = ROOT / "docs/research/kbound/edge/scripts/run_edge_publication_pipeline.sh"
DECISIVE_RUNNER = ROOT / "docs/research/kbound/scripts/run_decisive_tta.sh"


def _write_executable(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    path.chmod(0o755)


def test_edge_runner_emits_only_maintained_paper_build_guidance(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    runner = repo / "docs/research/kbound/edge/scripts/run_edge_publication_pipeline.sh"
    runner.parent.mkdir(parents=True)
    runner.write_bytes(EDGE_RUNNER.read_bytes())
    _write_executable(repo / ".venv/bin/python", "#!/bin/sh\nexit 0\n")
    _write_executable(
        repo / "docs/research/kbound/scripts/build_dashboard.sh",
        "#!/bin/sh\nexit 0\n",
    )

    result = subprocess.run(
        ["bash", str(runner)],
        cwd=repo,
        capture_output=True,
        text=True,
        timeout=20,
    )

    assert result.returncode == 0, result.stdout + result.stderr
    assert "kbound_short.tex" not in result.stdout
    assert "pdflatex kbound.tex" not in result.stdout
    assert "kbound_submission.tex" in result.stdout
    assert "kbound_tmlr.tex" in result.stdout
    assert "build_pdfs.sh" in result.stdout


def test_decisive_runner_marks_outputs_diagnostic_and_routes_publication_canonically(
    tmp_path: Path,
) -> None:
    repo = tmp_path / "repo"
    runner = repo / "docs/research/kbound/scripts/run_decisive_tta.sh"
    runner.parent.mkdir(parents=True)
    runner.write_bytes(DECISIVE_RUNNER.read_bytes())
    fake_python = tmp_path / "python3.12"
    _write_executable(
        fake_python,
        "#!/bin/sh\nif [ \"${1:-}\" = -c ]; then printf '3.12\\n'; fi\nexit 0\n",
    )
    venv = tmp_path / "venv"
    _write_executable(
        venv / "bin/python",
        "#!/bin/sh\nif [ \"${1:-}\" = -c ]; then printf '3.12\\n'; fi\nexit 0\n",
    )
    (venv / "bin/activate").write_text("python() { return 0; }\n", encoding="utf-8")
    env = os.environ.copy()
    env.update(
        {
            "PYTHON": str(fake_python),
            "KB_VENV": str(venv),
            "FULL": "0",
            "WITH_C100": "0",
        }
    )

    result = subprocess.run(
        ["bash", str(runner)],
        cwd=repo,
        env=env,
        capture_output=True,
        text=True,
        timeout=20,
    )

    assert result.returncode == 0, result.stdout + result.stderr
    assert "kbound_short.tex" not in result.stdout
    assert "pdflatex kbound.tex" not in result.stdout
    assert "diagnostic" in result.stdout.lower()
    assert "release_candidate.sh all" in result.stdout


def test_no_live_internal_kbound_plans_or_process_ledgers() -> None:
    plan_root = ROOT / "docs/superpowers/plans"
    sdd_root = ROOT / ".superpowers/sdd"

    live_plans = sorted(
        path.relative_to(ROOT).as_posix() for path in plan_root.glob("*.md") if "kbound" in path.name.lower()
    )
    live_process_ledgers = sorted(
        path.relative_to(ROOT).as_posix() for path in sdd_root.glob("*/*.md") if "kbound" in path.parent.name.lower()
    )

    assert live_plans + live_process_ledgers == []
