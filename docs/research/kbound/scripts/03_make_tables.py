#!/usr/bin/env python3
"""Pipeline stage: regenerate the paper's result-table macros.

Delegates to scripts/make_tables.py, which reads results_source.json (the single
source of truth, honest out-of-fold numbers) and writes
paper/generated/kbound_numbers.tex. Kept as a numbered stage so that
99_reproduce_kbound.py runs it in order.
"""
import os, runpy

HERE = os.path.dirname(os.path.abspath(__file__))
runpy.run_path(os.path.join(HERE, "make_tables.py"), run_name="__main__")
