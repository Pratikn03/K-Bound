# Item-11 POEM/AETTA — prep status (2026-07-17)

## Fixed during prep (verified)
1. **Harness numpy call.** numpy 2.0.2 + sklearn 1.6.1 ARE in env `aetta` (the 03:09
   failure was a transient during env build). Robust invocation for the head-to-head
   is `/opt/anaconda3/envs/aetta/bin/python official_baselines_headtohead.py …`
   (verified `--help` runs). Used by the continuation runner; the live orchestrator's
   own Phase C now also works because numpy is present.
2. **POEM `models.Res` import.** `external/poem/main.py` line 40 guarded with try/except
   (models/ is gitignored upstream; only `--model resnet50_bn_torch` needs it). Default
   path `resnet50_gn_timm` uses timm. `python main.py --help` now exits 0 — real CLI
   captured to `poem_help.txt`.
3. **POEM deps.** env `poem` has all imports (timm/pycm/loguru/cotta/sam/protector/…),
   proven by `--help` fully importing the module. timm `resnet50_gn` pretrained loads
   (25.6M params) — no training needed for POEM.

## Plan correction: POEM is ImageNet-C, NOT cifar10
POEM's `main.py` is ImageNet-only (`--data_corruption`, methods no_adapt/tent/eata/sar/
cotta/poem, models resnet50_gn_timm/vitbase_timm/resnet50_bn_torch). There is no cifar10
path. So we run POEM on **ImageNet-C** and compare to K-Bound's ImageNet-C decisions —
which is protocol-matched, since K-Bound's ImageNet-C panel uses the 3 noise corruptions
(gaussian/shot/impulse) present at `~/imagenetc_local/<corruption>/<1..5>/<1000 classes>`.
Full 15-corruption ImageNet-C is NOT on T9; the 3-noise subset is what both sides share.

## Ready to run (after AETTA training frees the GPU)
`docs/research/kbound/runbooks/run_poem_imagenetc.sh`
- fail-closed preflight (data layout + timm load), then runs method∈{poem,no_adapt}
  × seeds × severities × {gaussian,shot,impulse}; writes raw CSV/JSON per cell.
- knobs: `SEEDS`, `SEVERITIES` (K-Bound panel = 1 3 5), `CORRUPTIONS`, `TEST_BATCH_SIZE`.
- **Declare in paper:** POEM headline is batch-size-1 (online); default here is 64 for
  tractability — either match bs1 or disclose the deviation.

## Still open (needs the GPU run + validation — cannot finish during training)
- Actual POEM runs (GPU; sequential AFTER AETTA to avoid MPS contention).
- `poem_decisions.json` builder: pair per condition → {action:ADAPT, a_adapted:poem.top1,
  a0:no_adapt.top1}, keyed to K-Bound's ImageNet-C condition basis (win_hunt_v5_imagenetc),
  NOT the CIFAR canonical in baseline_decisions_adapter.py. Validate on first real output.
- `--data`/clean-val path + the README `dataset/folder.py` torchvision patch: only matters
  if a run errors on the clean/original branch; corrupted runs use `--data_corruption`.

## Untouched / safe
`run_item11_official.sh` reverted to original bytes (live orchestrator, PID 68282, is mid
Phase-A2 — must not be edited). AETTA source training NOT disturbed.
