"""Release producer must use the complete inventory and preserve receipts on failure."""
import os
from pathlib import Path
import subprocess
import sys

import pytest
from docs.research.kbound.scripts.verify_release_checksums import REQUIRED_RELEASE_PATHS, verify_checksum_file

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize('missing_pdf', [False, True])
def test_producer_is_complete_and_atomic(tmp_path, missing_pdf):
    kb = tmp_path / 'docs/research/kbound'
    for name in REQUIRED_RELEASE_PATHS:
        p = tmp_path / name
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(b'synthetic release fixture')
    script = kb / 'scripts/verify_release_checksums.py'
    script.parent.mkdir(parents=True, exist_ok=True)
    script.write_bytes((ROOT / 'docs/research/kbound/scripts/verify_release_checksums.py').read_bytes())
    receipt = kb / 'KBOUND_RELEASE_SHA256SUMS.txt'
    receipt.write_bytes(b'preserve prior receipt\n')
    if missing_pdf:
        (kb / 'kbound_short_final_draft.pdf').unlink()
    runbook = (ROOT / 'docs/research/kbound/runbooks/release_candidate.sh').read_text()
    body = runbook.split('emit_checksums() {', 1)[1].split('\n}\n', 1)[0]
    shell = 'set -eu\nlog() { :; }\nhave() { command -v "$1" >/dev/null; }\nwarn() { echo "$*" >&2; }\nemit_checksums() {' + body + '\n}\nemit_checksums\n'
    run = subprocess.run(['/bin/bash', '-c', shell], env={**os.environ, 'KB':str(kb), 'REPO':str(tmp_path), 'PY':sys.executable}, capture_output=True, text=True, timeout=20)
    if missing_pdf:
        assert run.returncode != 0
        assert receipt.read_bytes() == b'preserve prior receipt\n'
    else:
        assert run.returncode == 0, run.stderr
        assert verify_checksum_file(receipt, root=tmp_path, required_paths=REQUIRED_RELEASE_PATHS, exact_paths=True) == len(REQUIRED_RELEASE_PATHS)
