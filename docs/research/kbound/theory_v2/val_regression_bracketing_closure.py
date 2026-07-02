#!/usr/bin/env python3
"""Validator for regression_bracketing_closure.tex — wraps regression_conjecture_validation."""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]  # repo root
REG_VAL = ROOT / "docs" / "research" / "kbound" / "scripts" / "regression_conjecture_validation.py"
VENV_PY = HERE / ".venv" / "bin" / "python"
PY = str(VENV_PY if VENV_PY.is_file() else sys.executable)
JSON_PATH = HERE / "val_regression_bracketing_closure_results.json"


def main() -> int:
    if not REG_VAL.is_file():
        print(f"MISSING {REG_VAL}", file=sys.stderr)
        return 1
    parts: dict[str, int] = {}
    tails: dict[str, list[str]] = {}
    for part in ("reg1", "reg2"):
        proc = subprocess.run(
            [PY, str(REG_VAL), "--part", part],
            cwd=ROOT,
            capture_output=True,
            text=True,
        )
        parts[part] = proc.returncode
        tails[part] = proc.stdout.strip().splitlines()[-8:]
        if proc.returncode != 0:
            tails[f"{part}_stderr"] = proc.stderr.strip().splitlines()[-8:]
    ok = all(code == 0 for code in parts.values())
    out = {
        "wrapped_validator": str(REG_VAL.relative_to(ROOT)),
        "parts": parts,
        "stdout_tail": tails,
        "VERDICT": "PASS" if ok else "FAIL",
    }
    JSON_PATH.write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps(out["VERDICT"]))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
