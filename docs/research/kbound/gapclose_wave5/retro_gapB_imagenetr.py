"""Gap B retro-run (DIAGNOSTIC ONLY, frozen): self-normalized tau on the
ImageNet-R 10-backbone panel's stored agreement matrices.

Data: experiments/kbound/results/imagenetr_protocol_d_multiseed_v1/
per_condition_imagenet-r_<model>_seed<S>.json — each condition stores c_ij,
n_D, tau_hat, tau_star (frozen 0.52) and the kga decision.

Reports, per condition: stored tau_hat vs frozen 0.52, our tau_obs (Wave-5
definition), local null quantile, tau_prime. No win/loss verdict is claimed —
real data has no CEI ground truth; this characterizes how the operating point
transfers. The decisive use is gated on a future pre-registered GPU protocol.
"""
from __future__ import annotations

import glob
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from tau_selfnorm import tau_selfnorm  # noqa: E402

REPO = os.path.abspath(os.path.join(HERE, "..", "..", "..", ".."))
PAT = os.path.join(REPO, "experiments/kbound/results/"
                         "imagenetr_protocol_d_multiseed_v1/per_condition_*.json")


def main() -> int:
    rows = []
    files = sorted(f for f in glob.glob(PAT)
                   if not os.path.basename(f).startswith("._"))
    sl = os.environ.get("SLICE")  # e.g. "0:10" for chunked execution
    if sl:
        a, b = sl.split(":")
        files = files[int(a):int(b)]
    for path in files:
        base = os.path.basename(path)
        if base.startswith("._"):
            continue
        recs = json.load(open(path))
        if isinstance(recs, dict):
            recs = recs.get("records", recs.get("conditions", []))
        for r in recs:
            C = np.array(r.get("c_ij"), dtype=float)
            if C.ndim != 2 or C.shape[0] < 3:
                continue
            np.fill_diagonal(C, 1.0)
            m = int(r.get("n_D") or 0)
            if m < 20:
                continue
            res = tau_selfnorm(C, m, alpha=0.05, n_sim=300,
                               seed=abs(hash(base + str(r.get("condition")))) % (1 << 30))
            rows.append(dict(file=base, condition=r.get("condition"),
                             seed=r.get("seed"), K=int(C.shape[0]), m=m,
                             tau_hat_stored=r.get("tau_hat"),
                             tau_star_frozen=r.get("tau_star"),
                             stored_reject=bool((r.get("tau_hat") or 0)
                                                > (r.get("tau_star") or 0.52)),
                             tau_obs_w5=res["tau_obs"],
                             tau_star_local=res["tau_star_local"],
                             tau_prime=res["tau_prime"],
                             selfnorm_reject=res["reject_H"]))
    jl = os.path.join(HERE, "retro_gapB_rows.jsonl")
    if sl:  # chunked mode: append rows and exit
        with open(jl, "a") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")
        print(f"slice {sl}: wrote {len(rows)} rows")
        return 0
    if os.path.exists(jl) and not rows:  # merge mode
        rows = [json.loads(l) for l in open(jl)]
    n = len(rows)
    stored_rej = sum(r["stored_reject"] for r in rows)
    self_rej = sum(r["selfnorm_reject"] for r in rows)
    flipped_to_pass = sum(1 for r in rows
                          if r["stored_reject"] and not r["selfnorm_reject"])
    out = dict(n_conditions=n,
               stored_frozen052_rejects=stored_rej,
               selfnorm_rejects=self_rej,
               rejects_that_become_passes=flipped_to_pass,
               note=("DIAGNOSTIC ONLY per PROTOCOL_GAPCLOSE_WAVE5_v1. "
                     "No CEI ground truth on real data; decisive use gated on "
                     "a future pre-registered GPU protocol."),
               rows=rows)
    print(json.dumps({k: v for k, v in out.items() if k != "rows"}, indent=1))
    with open(os.path.join(HERE, "retro_gapB_results.json"), "w") as f:
        json.dump(out, f, indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
