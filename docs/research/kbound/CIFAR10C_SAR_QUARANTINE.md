# CIFAR-10-C SAR quarantine

**Status: withheld.** The archived CIFAR-10-C SAR aggregate is not reproducible from the current seed-0 replay and is excluded from paper evidence. It may be retained only for historical audit.

> **Added 2026-07-26 — the cause is confounded and cannot be attributed from this release.**
> Seed 0 is not only the seed whose aggregate fails to replay. It is also the only CIFAR-10-C seed
> that ran on a **different Python (3.12.13 vs 3.14.3), a different torch (2.5.1 vs 2.12.0) and a
> different commit (`4896181799ad` vs `6a237ed489c3`)**, finished three weeks after seeds 1-3, and
> no run manifest in the project records a scikit-learn version even though `b_hat` comes from
> `GradientBoostingRegressor(subsample=0.8)`. See `SUBMISSION_LEDGER.md §10` and
> `REPRODUCE.md §0a`.
>
> Substantively, seed 0's harmful base rate is 0.53 against roughly 0.10 on seeds 1-4 — a 5x
> difference — and on seeds 1-4 KGA's SAR regret (0.00160) *exceeds* always-adapt's (0.00031). The
> quarantine is correct either way. What cannot be said is *why* seed 0 differs: "non-reproducing
> seed" and "different toolchain" are not separable from the artifacts on disk.
>
> **This adds a gate 0 to the list below: the rebuild must run all five seeds under one pinned
> stack, with `scikit_learn` recorded in every `result_manifest.json`.** A rebuild that reproduces
> the anomaly under a *different* environment than the original settles nothing.

**Clean rebuild opened:** `CIFAR10C_SAR_REBUILD_PROTOCOL_v2.yaml` freezes a new five-seed,
SAR-only run in `experiments/kbound/results/cifar10c_sar_rebuild_v2/`. The rebuild never writes
to the archived tree. Its launcher is `runbooks/rebuild_cifar10c_sar.sh`; completing a run does
not reinstate the claim until the validator and final evidence integration gates pass.

Reinstatement requires all of the following:

1. pin the exact implementation, configuration, checkpoint, package environment, and evaluation order;
2. reproduce seed 0 from clean raw outputs;
3. complete every planned seed under the identical protocol;
4. derive action counts, false-adapt events, intervals, and regret from raw decisions rather than rounded summaries;
5. pass schema, leakage, lineage, and evidence-seal checks; and
6. update the claim ledger and manuscript only after those gates pass.

Until then, no aggregate, table row, or comparative wording from the archived SAR run supports a claim.

Run from a terminal with a working MPS PyTorch environment:

```bash
KBOUND_SAR_PYTHON=/path/to/mps/python \
  bash docs/research/kbound/runbooks/rebuild_cifar10c_sar.sh preflight
KBOUND_SAR_PYTHON=/path/to/mps/python \
  bash docs/research/kbound/runbooks/rebuild_cifar10c_sar.sh smoke
KBOUND_SAR_PYTHON=/path/to/mps/python \
  caffeinate -is bash docs/research/kbound/runbooks/rebuild_cifar10c_sar.sh run
KBOUND_SAR_PYTHON=/path/to/mps/python \
  bash docs/research/kbound/runbooks/rebuild_cifar10c_sar.sh finalize
```
