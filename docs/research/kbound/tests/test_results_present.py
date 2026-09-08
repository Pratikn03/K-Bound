"""Smoke: every organized result file exists and has a sane top-level structure."""

import glob
import json
import os

PKG = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def test_result_groups_populated():
    expect = {"main": 4, "ablations": 1, "tta": 3, "regression": 1, "witness": 1, "multimodal": 8}
    for grp, n in expect.items():
        files = glob.glob(os.path.join(PKG, "results", grp, "*.json"))
        assert len(files) >= n, f"{grp}: {len(files)} < {n}"
        for f in files:
            with open(f, encoding="utf-8") as stream:
                assert isinstance(json.load(stream), dict)
