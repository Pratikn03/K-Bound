#!/usr/bin/env python3
"""A/B validation of the faithful-SAR fix: OLD (broken) SAR vs FIXED SAR.

Run from a checkout, in an env with torch (fix-queue item 30: no machine-local paths):
    cd <repo root>
    python docs/research/kbound/scripts/test_sar_faithful.py

Part A = deterministic, forced-branch mechanical asserts on the FIXED code.
The synthetic A2-A4 fixtures override the entropy threshold to log(K)+1;
they do not validate empirical performance or default-stream reliability.
Importing this module does not run adaptation. Pytest runs Part A; the optional
--compare-historical CLI switch runs Part B.

Part B = OLD-vs-FIXED on a stream the model is CONFIDENT on (so samples pass SAR's
         entropy<E_0 reliability filter and adaptation actually engages).
"""

import importlib.machinery
import importlib.util
import inspect
import math
import os
from contextlib import contextmanager
from unittest.mock import patch

import torch
import torch.nn as nn
import torch.nn.functional as F


@contextmanager
def _observe_sgd():
    """Observe real SGD effects; scoped wrappers always restore the class methods."""
    observations = {"gradients": [], "movements": [], "resets": []}
    real_step = torch.optim.SGD.step
    real_load = torch.optim.SGD.load_state_dict

    def step(optimizer, *args, **kwargs):
        params = [p for group in optimizer.param_groups for p in group["params"]]
        before = [p.detach().clone() for p in params]
        gradient = sum(p.grad.detach().square().sum().item() for p in params if p.grad is not None)
        observations["gradients"].append(gradient)
        result = real_step(optimizer, *args, **kwargs)
        observations["movements"].append(sum((p.detach() - q).square().sum().item() for p, q in zip(params, before)))
        return result

    def load(optimizer, state_dict):
        momentum_before = sum(
            state["momentum_buffer"].square().sum().item()
            for state in optimizer.state.values()
            if "momentum_buffer" in state
        )
        result = real_load(optimizer, state_dict)
        observations["resets"].append((momentum_before, len(optimizer.state)))
        return result

    with patch.object(torch.optim.SGD, "step", step), patch.object(torch.optim.SGD, "load_state_dict", load):
        yield observations


def _assert_active_steps(observed, count):
    assert len(observed["gradients"]) == count, "SAR must execute every forced-reliable optimizer step"
    assert all(math.isfinite(g) and g > 0 for g in observed["gradients"]), "SAR gradients must be positive and finite"


def _run_mechanical_checks(*, compare_historical=False):
    SCRIPTS = os.path.dirname(os.path.abspath(__file__))
    FIXED = os.path.join(SCRIPTS, "cifar_tent_mps_v2.py")
    baks = (
        sorted(f for f in os.listdir(SCRIPTS) if "bak_presar" in f) if compare_historical else []
    )  # pre-fix (broken) harness
    OLD = os.path.join(SCRIPTS, baks[-1]) if baks else None

    def load(name, path):
        loader = importlib.machinery.SourceFileLoader(name, path)
        spec = importlib.util.spec_from_loader(name, loader)
        mod = importlib.util.module_from_spec(spec)
        loader.exec_module(mod)
        return mod

    H = load("H_fixed", FIXED)
    OLDH = load("H_old", OLD) if OLD else None
    torch.manual_seed(0)

    def tiny_model(K=10):
        return nn.Sequential(
            nn.Conv2d(3, 8, 3, padding=1),
            nn.BatchNorm2d(8),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(8, K),
        )

    def stream(n=6, bs=8):
        return [torch.randn(bs, 3, 8, 8) for _ in range(n)]

    def make_confident(m, S, iters=250, lr=1e-2):  # fit to own argmax -> low-entropy => passes E_0 filter
        opt = torch.optim.Adam(m.parameters(), lr)
        for _ in range(iters):
            for xb in S:
                out = m(xb)
                loss = F.cross_entropy(out, out.detach().argmax(1))
                opt.zero_grad()
                loss.backward()
                opt.step()
        return m

    def mean_entropy(m, x):
        with torch.no_grad():
            p = m(x).softmax(1)
            return float(-(p * (p + 1e-9).log()).sum(1).mean())

    def collapse_frac(m, x, K=10):
        with torch.no_grad():
            return m(x).argmax(1).bincount(minlength=K).max().item() / len(x)

    # ---------- Part A: deterministic forced-branch mechanics (FIXED code) ----------
    sig = inspect.signature(H.sar_adapt)
    d = {k: v.default for k, v in sig.parameters.items()}
    assert abs(d["reset_constant_em"] - 0.2) < 1e-12 and d["margin_e0"] is None, "faithful defaults"
    print("[A1] OK  defaults: reset_constant_em=0.2 (absolute), margin_e0 -> 0.4*ln(K)")
    forced_margin = math.log(10) + 1.0  # Entropy <= log(K): admit all synthetic samples.

    # ---- [A2] SAM restore is exact -- ON THE REAL PARAMETER TENSORS -------------
    # FIX-QUEUE ITEM 30 (F2-16).  The previous A2 was:
    #
    #     res_new = (w.clone() - w).norm().item()
    #     assert res_new < 1e-9 < res_old, "SAM restore must be exact"
    #
    # `‖w − w‖` is 0 by construction on a local scratch tensor.  It touched neither
    # `sar_adapt` nor any other project code, so it validated nothing while reading
    # as the headline gate of this file.  Replaced with an assertion on the actual
    # parameters `sar_adapt` mutates: the SAM ascent perturbation must leave no
    # residue, so with lr = 0 the weights must come back bit-identical after N steps
    # (the ascent is the only thing that moves them when the optimiser step is a
    # no-op).  The analytic drift of the OLD bug -- restoring with the *second*
    # gradient instead of the saved weights -- is kept as the contrast value.
    rho = 0.05
    g1 = torch.tensor([2.0, 1.0, -3.0, 0.4])
    g2 = torch.tensor([1.5, 1.2, -2.0, 0.9])
    res_old = (rho * g1 / g1.norm() - rho * g2 / g2.norm()).norm().item()  # old-bug drift/step

    _base_a2 = tiny_model().eval()
    _init_a2 = [p.detach().clone() for p in _base_a2.parameters()]
    with _observe_sgd() as a2:
        _m_a2, _un_a2 = H.sar_adapt(
            _base_a2,
            stream(),
            steps=5,
            lr=0.0,
            num_classes=10,
            margin_e0=forced_margin,
            reset_constant_em=-1e9,
        )
    _assert_active_steps(a2, 30)
    assert not a2["resets"], "A2 must isolate SAM restoration, without recovery"
    res_new = max((p.detach() - q).abs().max().item() for p, q in zip(_m_a2.parameters(), _init_a2))
    assert res_new == 0.0 and _un_a2 == 0.0 and res_old > 1e-9, (
        f"SAM restore must leave no residue on the real parameters: sar_adapt with lr=0 moved a weight by {res_new:.3e}"
    )
    print(
        f"[A2] OK  forced-branch SAM restore: {len(a2['gradients'])} actual lr=0 SGD steps, "
        f"positive gradients, max drift {res_new:.1e}; old-bug analytic drift {res_old:.4f}/step"
    )

    base = tiny_model().eval()
    init = [p.detach().clone() for p in base.parameters()]
    with _observe_sgd() as a3:
        m, un = H.sar_adapt(
            base,
            stream(),
            steps=3,
            lr=1e-2,
            num_classes=10,
            margin_e0=forced_margin,
            reset_constant_em=1e9,
        )
    _assert_active_steps(a3, 18)
    assert all(math.isfinite(delta) and delta > 0 for delta in a3["movements"]), "weights must move before recovery"
    assert len(a3["resets"]) == 18, "recovery must load optimizer state after every forced collapse"
    assert all(math.isfinite(momentum) and momentum > 0 and remaining == 0 for momentum, remaining in a3["resets"]), (
        "recovery must clear populated, nonzero SGD momentum"
    )
    dev = max((p.detach() - q).abs().max().item() for p, q in zip(m.parameters(), init))
    assert dev == 0.0 and un == 0.0, "recovery must restore weights and reset update norm"
    print(
        f"[A3] OK  forced recovery: {len(a3['resets'])} positive updates followed by momentum resets; "
        f"max weight dev {dev:.1e}, update_norm {un:.1e}"
    )

    with _observe_sgd() as a4:
        m2, un2 = H.sar_adapt(
            tiny_model().eval(),
            stream(),
            steps=10,
            lr=2.5e-3,
            num_classes=10,
            margin_e0=forced_margin,
            reset_constant_em=-1e9,
        )
    _assert_active_steps(a4, 60)
    assert all(math.isfinite(delta) and delta > 0 for delta in a4["movements"]), "SGD updates must not be no-ops"
    assert un2 > 0, "unrecovered adaptation must move parameters"
    assert all(torch.isfinite(p).all() for p in m2.parameters()) and math.isfinite(un2)
    assert not a4["resets"], "A4 must expose unrecovered updates"
    print(f"[A4] OK  forced-branch positive finite updates (update_norm {un2:.4f})")

    # ---------- Part B: OLD (broken) vs FIXED SAR on a stream the model is confident on ----------
    def report(tag, m, un, probe):
        fin = all(torch.isfinite(p).all() for p in m.parameters())
        mx = max(p.detach().abs().max().item() for p in m.parameters())
        print(
            f"   {tag:14s} finite={fin}  update_norm={un:9.3f}  max|param|={mx:9.2f}  "
            f"pred_entropy={mean_entropy(m, probe):.3f}  collapse_frac={collapse_frac(m, probe):.2f}"
        )

    if OLDH is None:
        print("\n[B] NOT RUN: historical comparison is opt-in and requires a local backup.")
    else:
        torch.manual_seed(0)
        S = stream(n=6, bs=8)
        probe = torch.cat(S)
        cm = make_confident(tiny_model(), S).eval()  # confident base -> samples pass the E_0 filter
        print(
            f"\n--- Part B: aggressive lr=0.05, 50 steps; base pred_entropy={mean_entropy(cm, probe):.3f} "
            f"(E_0=0.4*ln10={0.4 * math.log(10):.2f}, so samples are reliable) ---"
        )
        print("  B1: reset DISABLED (isolates the SAM step):")
        mo, uno = OLDH.sar_adapt(cm, S, steps=50, lr=0.05, num_classes=10, e_reset=-1e9)
        mf, unf = H.sar_adapt(cm, S, steps=50, lr=0.05, num_classes=10, reset_constant_em=-1e9)
        report("OLD/broken", mo, uno, probe)
        report("FIX/faithful", mf, unf, probe)
        print("  B2: recovery ENABLED (default thresholds 0.2*ln10 old vs 0.2 fixed):")
        mo2, uno2 = OLDH.sar_adapt(cm, S, steps=50, lr=0.05, num_classes=10)
        mf2, unf2 = H.sar_adapt(cm, S, steps=50, lr=0.05, num_classes=10)
        report("OLD/broken", mo2, uno2, probe)
        report("FIX/faithful", mf2, unf2, probe)
        assert all(torch.isfinite(p).all() for p in mf2.parameters()) and math.isfinite(unf2), "FIXED must stay finite"
        print(f"\n   read: the broken SAM injects ~{res_old:.4f}/step of spurious drift (A2); over 50 aggressive")
        print("   steps that shows up as a larger update_norm / harder prediction collapse for OLD, while the")
        print(f"   faithful SAM+recovery (FIX) stays bounded.  B1 update_norm OLD={uno:.2f} vs FIX={unf:.2f}.")

    print("\nGATE: A1 defaults and A2-A4 forced-branch mechanics PASSED; no empirical/default-stream validation.")


def test_faithful_sar_mechanical_checks():
    """Run A1-A4 on actual SAR code without import-time side effects."""
    with torch.random.fork_rng(devices=[]):
        _run_mechanical_checks()


def main():
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--compare-historical", action="store_true")
    args = parser.parse_args()
    with torch.random.fork_rng(devices=[]):
        _run_mechanical_checks(compare_historical=args.compare_historical)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
