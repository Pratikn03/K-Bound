"""Make ELARA-Opt + the validated kbound_tta/kga packages importable under pytest
regardless of the invoking PYTHONPATH."""
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(os.path.dirname(_HERE))
for _p in (os.path.join(_REPO, "packaging", "kbound-tta", "src"), _REPO):
    if _p not in sys.path:
        sys.path.insert(0, _p)
