#!/usr/bin/env python3
"""Pipeline stage: regenerate the paper's figures.

Runs both figure generators so the reproduction pipeline emits every figure:
  - figures/source/make_figures.py     (main experiment figures)
  - scripts/make_submission_figures.py (short-paper natural-shift forest +
                                        benefit-sign frontier schematic)
Kept as a numbered stage so that 99_reproduce_kbound.py runs it in order.
"""
import os, runpy

HERE = os.path.dirname(os.path.abspath(__file__))   # .../docs/research/kbound/scripts
KB = os.path.dirname(HERE)                           # .../docs/research/kbound
runpy.run_path(os.path.join(KB, "figures", "source", "make_figures.py"), run_name="__main__")
runpy.run_path(os.path.join(HERE, "make_submission_figures.py"), run_name="__main__")
