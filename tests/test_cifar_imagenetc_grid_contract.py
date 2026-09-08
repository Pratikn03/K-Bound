"""Execute real runner functions without importing tensor/model dependencies.

The complete function ASTs are compiled unchanged. Only neural computation,
data loading and post-run analysis are replaced; grid/CLI/filesystem logic runs.
"""
import argparse
import ast
import os
from pathlib import Path
import sys
import time
from types import SimpleNamespace

import numpy as np
import pytest


SOURCE = Path(__file__).resolve().parents[1] / "docs/research/kbound/scripts/cifar_tent_mps_v2.py"


class TensorDouble:
    def permute(self, *axes):
        return self

    def float(self):
        return self

    def __truediv__(self, divisor):
        return self

    def to(self, device):
        return self


class Finished(Exception):
    def __init__(self, rows):
        self.rows = rows


class ModelBoundary(Exception):
    """Stop after availability validation, before any model construction."""


@pytest.fixture
def runner():
    tree = ast.parse(SOURCE.read_text())
    constants = {
        "SEVERITIES", "BATCH_REGIMES", "COMPOSITIONS", "AGGRESSIVENESS",
        "N_REPEATS", "ALPHA", "SEED", "EVAL_CHUNK", "IMAGENET_C_QUICK",
        "CORRUPTION_TO_TAR",
    }
    functions = {
        "main", "run_cifar101_benchmark", "run_imagenet_benchmark",
        "_ic_available", "_ic_corruption_tar", "label_regime",
    }
    nodes = [node for node in tree.body if
             (isinstance(node, ast.FunctionDef) and node.name in functions) or
             (isinstance(node, ast.Assign) and any(
                 isinstance(t, ast.Name) and t.id in constants for t in node.targets))]
    ns = dict(os=os, sys=sys, argparse=argparse, time=time, np=np,
              SEL_BATCH=None, SEL_AGGR=None, ADAPT_LR=None, HELP_THR=0.02,
              SAR_LR=None, SAR_FREEZE_LAYER4=False,
              torch=SimpleNamespace(tensor=lambda value: TensorDouble()),
              _kb=SimpleNamespace(backend=lambda: "synthetic-test"),
              set_global_seed=lambda seed: None,
              get_cifar_model=lambda *a: object(),
              load_cifar_101=lambda root: (np.zeros((2, 1, 1, 3)), np.array([0, 1])),
              _norm_cifar=lambda x: x, acc_on=lambda *a, **kw: 0.5,
              _log_source_reference=lambda *a: None,
              _log_condition_samples=lambda *a: None, _mps_free=lambda: None,
              build_stream_and_eval=lambda *a, **kw: ([], object(), object()),
              evidence_vector=lambda *a: [0.0],
              TTA_METHODS={m: lambda *a, **kw: (object(), object()) for m in ("tent", "eata")})
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(SOURCE), "exec"), ns)

    class Models:
        def __getattr__(self, name):
            raise ModelBoundary(name)

    ns["tv"] = SimpleNamespace(models=Models())
    return ns


def invoke_cli(runner, monkeypatch, tmp_path, benchmark, extra):
    monkeypatch.setattr(sys, "argv", [str(SOURCE), "--benchmarks", benchmark,
        "--device", "cpu", "--out-results", str(tmp_path / "results"),
        "--out-figs", str(tmp_path / "figures"), *extra])
    runner["main"]()


@pytest.mark.parametrize("cap,expected", [(1, 2), (2, 4), (100, 36), (0, 36)])
def test_cifar101_cli_cap_limits_grid_not_repeats_or_methods(runner, monkeypatch, tmp_path, cap, expected):
    # Ignoring the CLI cap or applying it after repeats breaks the positive cases.
    real_run = runner["run_cifar101_benchmark"]

    def stop_after_run(*args, **kwargs):
        _, rows = real_run(*args, **kwargs)
        raise Finished(rows)

    runner["run_cifar101_benchmark"] = stop_after_run
    with pytest.raises(Finished) as done:
        invoke_cli(runner, monkeypatch, tmp_path, "cifar101",
                   ["--methods", "tent", "eata", "--max-cells", str(cap)])
    assert {m: len(rows) for m, rows in done.value.rows.items()} == {"tent": expected, "eata": expected}
    assert [r["condition"] for r in done.value.rows["tent"][:2]] == [
        "cifar101|large_iid|iid|mild|r0", "cifar101|large_iid|iid|mild|r1"]


def test_cifar101_omitted_cap_keeps_original_full_grid(runner):
    _, rows = runner["run_cifar101_benchmark"]("unused", "cpu", ["tent"])
    assert len(rows["tent"]) == 36
    assert rows["tent"][-1]["condition"] == "cifar101|tiny|single_class|aggressive|r1"


@pytest.mark.parametrize("existing", [True, False])
def test_explicit_missing_imagenetc_corruption_fails_before_model_load(runner, monkeypatch, tmp_path, existing):
    # Dropping the absent directory must not proceed with the surviving subset.
    if existing:
        (tmp_path / "gaussian_noise").mkdir()
    with pytest.raises(SystemExit) as error:
        invoke_cli(runner, monkeypatch, tmp_path, "imagenetc", [
            "--imagenetc-root", str(tmp_path), "--corruptions", "gaussian_noise", "shot_noise"])
    assert "shot_noise" in str(error.value)
    assert str(tmp_path) in str(error.value)
    assert not (tmp_path / "results" / "checkpoint.json").exists()


@pytest.mark.parametrize("layout", ["directories", "tar"])
def test_explicit_available_imagenetc_inputs_pass_availability_check(runner, monkeypatch, tmp_path, layout):
    if layout == "directories":
        (tmp_path / "gaussian_noise").mkdir()
        (tmp_path / "shot_noise").mkdir()
    else:
        # Availability only: no archive or data is opened by this test.
        (tmp_path / "noise.tar").touch()
    with pytest.raises(ModelBoundary):
        invoke_cli(runner, monkeypatch, tmp_path, "imagenetc", [
            "--imagenetc-root", str(tmp_path), "--corruptions", "gaussian_noise", "shot_noise"])


def test_implicit_imagenetc_default_retains_available_subset(runner, monkeypatch, tmp_path):
    (tmp_path / "gaussian_noise").mkdir()
    with pytest.raises(ModelBoundary):
        invoke_cli(runner, monkeypatch, tmp_path, "imagenetc", ["--imagenetc-root", str(tmp_path)])
