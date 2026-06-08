"""K-Bound experiment scripts (importable as ``scripts.kbound.*``).

This package marker makes the hermetic-smoke modules importable both as scripts
(``python src/scripts/kbound/smoke_trichotomy.py``) and as a package
(``from scripts.kbound.make_synth_archive import make_synth_archive``) under the
``tests/conftest.py`` ``sys.path`` setup (which puts ``src/`` first).
"""
