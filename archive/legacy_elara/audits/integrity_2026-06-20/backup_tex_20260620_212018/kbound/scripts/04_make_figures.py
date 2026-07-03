"""Thin wrapper -> figures/source/make_figures.py (figures are emitted by the experiment scripts)."""
import runpy, os
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
runpy.run_path(os.path.join(HERE, "figures", "source", "make_figures.py"), run_name="__main__")
