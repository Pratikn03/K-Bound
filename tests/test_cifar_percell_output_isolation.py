"""Exercise real CIFAR writer and CLI dispatch with synthetic neural boundaries."""
import argparse
import ast
import json
import os
from pathlib import Path
import sys
import time
from types import SimpleNamespace

import numpy as np
import pytest


SOURCE = Path(__file__).resolve().parents[1] / "docs/research/kbound/scripts/cifar_tent_mps_v2.py"


class Finished(Exception):
    """Stop after actual benchmark serialization, before unrelated analysis."""


@pytest.fixture
def runner():
    tree = ast.parse(SOURCE.read_text())
    constants = {
        "SEVERITIES", "BATCH_REGIMES", "COMPOSITIONS", "AGGRESSIVENESS",
        "N_REPEATS", "ALPHA", "SEED", "EVAL_CHUNK", "CIFAR_C_QUICK", "CIFAR_C_ALL",
    }
    nodes = [node for node in tree.body if
             (isinstance(node, ast.FunctionDef) and node.name in {"main", "run_cifar_benchmark"}) or
             (isinstance(node, ast.Assign) and any(
                 isinstance(t, ast.Name) and t.id in constants for t in node.targets))]
    # Never import torchvision, access a dataset, or construct a model. The
    # synthetic boundaries leave grid iteration and the actual JSON writer intact.
    dataset = lambda *a, **kw: SimpleNamespace(data=[], targets=[])
    ns = dict(os=os, sys=sys, argparse=argparse, time=time, np=np,
              SEL_BATCH=None, SEL_AGGR=None, ADAPT_LR=None,
              SAR_LR=None, SAR_FREEZE_LAYER4=False,
              tv=SimpleNamespace(datasets=SimpleNamespace(CIFAR10=dataset, CIFAR100=dataset)),
              _kb=SimpleNamespace(backend=lambda: "synthetic-test"),
              set_global_seed=lambda seed: None,
              get_cifar_model=lambda *a: object(),
              acc_on=lambda *a, **kw: 0.5,
              _log_source_reference=lambda *a: None,
              _log_condition_samples=lambda *a: None,
              load_cifar_c=lambda *a: ([], []),
              cifar_c_severity=lambda *a: ([], []),
              build_stream_and_eval=lambda *a, **kw: ([], [], []),
              evidence_vector=lambda *a: [0.0], label_regime=lambda delta: "neutral",
              TTA_METHODS={"tent": lambda *a, **kw: (object(), object())})
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(SOURCE), "exec"), ns)
    return ns


@pytest.mark.parametrize("which,benchmark", [("10", "cifar10c"), ("100", "cifar100c")])
@pytest.mark.parametrize("existing_historical", [False, True])
def test_cli_percell_dump_uses_selected_root_without_cwd_write(
    runner, monkeypatch, tmp_path, which, benchmark, existing_historical
):
    cwd = tmp_path / "old checkout"
    cwd.mkdir()
    monkeypatch.chdir(cwd)
    old_output = cwd / f"cifar10c_percell_{which}.json"
    if existing_historical:
        old_output.write_bytes(b"historical evidence sentinel\n")
    selected = tmp_path / "fresh selected output"
    real_run = runner["run_cifar_benchmark"]

    def stop_after_run(*args, **kwargs):
        real_run(*args, **kwargs)
        raise Finished

    runner["run_cifar_benchmark"] = stop_after_run
    monkeypatch.setattr(sys, "argv", [str(SOURCE), "--benchmarks", benchmark,
        "--device", "cpu", "--methods", "tent", "--max-cells", "1",
        "--out-results", str(selected), "--out-figs", str(tmp_path / "figures")])
    with pytest.raises(Finished):
        runner["main"]()
    output = selected / f"cifar10c_percell_{which}.json"
    assert output.is_file(), "per-cell evidence must follow --out-results"
    records = json.loads(output.read_text())
    assert len(records) == 2  # one grid cell, both original repeats
    assert [r["condition"] for r in records] == [
        "gaussian_noise|s1|large_iid|iid|mild|r0",
        "gaussian_noise|s1|large_iid|iid|mild|r1",
    ]
    if existing_historical:
        assert old_output.read_bytes() == b"historical evidence sentinel\n"
    else:
        assert not old_output.exists()


@pytest.mark.parametrize("which", ["10", "100"])
def test_direct_caller_omitting_output_root_keeps_legacy_default(runner, monkeypatch, tmp_path, which):
    monkeypatch.chdir(tmp_path)
    _, rows = runner["run_cifar_benchmark"](which, "unused", "cpu", ["tent"], [], max_cells=1)
    assert json.loads((tmp_path / f"cifar10c_percell_{which}.json").read_text()) == rows["tent"] == []
