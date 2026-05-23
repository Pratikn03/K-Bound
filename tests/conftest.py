"""
conftest.py – ensure both the repository root and src/ are on sys.path.

This allows tests to import:
  - `scripts.*`  →  resolved from src/scripts/   (src/ is on path first)
  - top-level modules from the repo root
"""
import sys
import os

# Repository root  (one level above the tests/ directory)
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_SRC_ROOT = os.path.join(_REPO_ROOT, "src")

# Insert src/ first so src/scripts/ is found before the shell-only scripts/ dir
for _path in (_SRC_ROOT, _REPO_ROOT):
    if _path not in sys.path:
        sys.path.insert(0, _path)
