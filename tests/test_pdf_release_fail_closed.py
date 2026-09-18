"""Exercise the actual PDF release phase with isolated build/render services."""
import subprocess
import sys
from pathlib import Path

import pytest

RUNBOOK = Path(__file__).resolve().parents[1] / "docs/research/kbound/runbooks/release_candidate.sh"


@pytest.mark.parametrize("renderer,success", [(None, False), ("raise SystemExit(17)\n", False), ("raise SystemExit(0)\n", True)])
def test_pdf_phase_requires_successful_page_verification(tmp_path, renderer, success):
    scripts = tmp_path / "scripts"
    scripts.mkdir()
    (scripts / "build_pdfs.sh").write_text("exit 0\n")
    if renderer is not None:
        (scripts / "render_pdf_pages.py").write_text(renderer)
    source = RUNBOOK.read_text()
    phase = source[source.index("step_pdf() {"):source.index("# Optional packaging phases")]
    driver = tmp_path / "phase.sh"
    driver.write_text("set -euo pipefail\nKB=$1\nPY=$2\n"
        "verify_release_toolchain_for_phase() { :; }\nlog() { :; }\nwarn() { echo \"$*\" >&2; }\n"
        + phase + "\nstep_pdf\necho RELEASE_PDF_SUCCESS\n")
    result = subprocess.run(["bash", str(driver), str(tmp_path), sys.executable],
                            capture_output=True, text=True, timeout=10)
    assert (result.returncode == 0) is success
    assert ("RELEASE_PDF_SUCCESS" in result.stdout) is success
