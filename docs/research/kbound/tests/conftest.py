import os, sys
PKG = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # docs/research/kbound
for p in ("scripts", "vendored_from_elara"):
    fp = os.path.join(PKG, p)
    if fp not in sys.path:
        sys.path.insert(0, fp)
