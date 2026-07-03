#!/usr/bin/env python3
"""WIN_HUNT_v3 Arm D — first real-data anytime-valid gate (iWildCam stream).

Pre-registered in research_lock/WIN_HUNT_v3_PROTOCOL.yaml. Consumes the logged
streaming pilot (native-order iWildCam test, window=50). Two one-sided
truncated-aGRAPA betting e-processes on per-window benefit b_t = acc_adapted_t
- acc_frozen_t (the construction validated in val_thm3_evalue): SPEND fires
when E+ >= 1/alpha, FREEZE when E- >= 1/alpha; before either fires the gate
plays the safe default (frozen). Anytime false-adapt counted at every window.

Schema-defensive: exits 3 listing available keys if per-window paired
accuracies are absent. Fabricates nothing.

Run (CPU, seconds):
  python3 docs/research/kbound/gapclose_wave5/win_hunt_D_anytime_stream.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[4]
PILOT = ROOT / ("experiments/kbound/results/iwildcam_streaming_pilot/"
                "pilot_test_native_bs16.json")
ALPHA = 0.10
BET_CAP = 0.5
PRIOR_VAR, PRIOR_W = 0.25, 1.0


def find_windows(d):
    """Locate the per-window list and paired accuracy fields, defensively."""
    for holder in (d, d.get("stream", {}), d.get("results", {})):
        for key in ("windows", "records", "window_log", "per_window"):
            w = holder.get(key) if isinstance(holder, dict) else None
            if isinstance(w, list) and w and isinstance(w[0], dict):
                return w, key
    return None, None


def field(w0, cands):
    for c in cands:
        if c in w0:
            return c
    return None


def eprocess(b, alpha, sign=+1):
    """One-sided truncated-aGRAPA wealth on sign*b (null: mean <= 0)."""
    logw = 0.0
    mu, m2, n = 0.0, PRIOR_VAR * PRIOR_W, PRIOR_W
    fired = None
    wealth_path = np.empty(len(b))
    for t, bt in enumerate(b):
        x = sign * float(bt)
        var = max(m2 / n - (mu / n) ** 2 if n > 0 else PRIOR_VAR, 1e-6)
        lam = float(np.clip((mu / n) / var, 0.0, BET_CAP / max(abs(x), 1e-6)))
        logw += float(np.log1p(lam * x))
        wealth_path[t] = logw
        mu += x; m2 += x * x; n += 1
        if fired is None and logw >= np.log(1.0 / alpha):
            fired = t
    return fired, wealth_path


def main() -> int:
    if not PILOT.exists():
        print(f"SCHEMA ERROR: pilot file missing: {PILOT}", file=sys.stderr)
        return 3
    d = json.load(open(PILOT))
    windows, key = find_windows(d)
    if windows is None:
        print("SCHEMA ERROR: no per-window list found. Top-level keys: "
              f"{sorted(d.keys())}; stream keys: "
              f"{sorted(d.get('stream', {}).keys()) if isinstance(d.get('stream'), dict) else 'n/a'}",
              file=sys.stderr)
        return 3
    w0 = windows[0]
    fa_ad = field(w0, ("tent_window_acc", "acc_adapted", "adapted_acc",
                       "acc_tent", "acc_adapt", "aa"))
    fa_fr = field(w0, ("frozen_window_acc", "acc_frozen", "frozen_acc",
                       "acc_f0", "acc_freeze", "a0"))
    if fa_ad is None or fa_fr is None:
        print("SCHEMA ERROR: need paired per-window accuracies; window keys: "
              f"{sorted(w0.keys())}", file=sys.stderr)
        return 3
    aa = np.array([float(w[fa_ad]) for w in windows])
    a0 = np.array([float(w[fa_fr]) for w in windows])
    b = aa - a0
    T = len(b)
    print(f"stream windows: {T} (list key '{key}', fields {fa_ad}/{fa_fr})")

    t_spend, wp = eprocess(b, ALPHA, +1)
    t_freeze, wm = eprocess(b, ALPHA, -1)

    # gate policy: frozen until a process fires; then commit for the remainder
    dec = np.zeros(T, dtype=int)
    if t_spend is not None and (t_freeze is None or t_spend < t_freeze):
        dec[t_spend:] = 1
    elif t_freeze is not None:
        dec[t_freeze:] = -1
    acc_gate = np.where(dec == 1, aa, a0)

    # anytime false-adapt: at each window with a standing SPEND commitment,
    # count violation if running mean benefit from commitment time is <= 0
    fa_any = 0
    if t_spend is not None and dec[-1] == 1:
        run = np.cumsum(b[t_spend:]) / np.arange(1, T - t_spend + 1)
        fa_any = float(np.mean(run <= 0))
    val = dict(gate=float(acc_gate.mean()), always_adapt=float(aa.mean()),
               always_freeze=float(a0.mean()))
    beats_both = val["gate"] > val["always_adapt"] and val["gate"] > val["always_freeze"]
    anytime_ok = fa_any <= ALPHA
    verdict = ("WIN" if (beats_both and anytime_ok) else
               "DEMO" if anytime_ok else "FAIL")

    out = dict(protocol="WIN_HUNT_v3_ARM_D",
               registered="research_lock/WIN_HUNT_v3_PROTOCOL.yaml",
               pilot=str(PILOT.relative_to(ROOT)), n_windows=T, alpha=ALPHA,
               mean_window_benefit=float(b.mean()),
               frac_windows_harmful=float((b < 0).mean()),
               spend_fired_at=t_spend, freeze_fired_at=t_freeze,
               anytime_false_adapt=fa_any, value=val,
               beats_both=bool(beats_both), VERDICT=verdict)
    print(json.dumps(out, indent=1))
    p = ROOT / "research_lock/WIN_HUNT_v3_ARM_D_result.json"
    p.write_text(json.dumps(out, indent=1))
    print(f"saved {p.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
