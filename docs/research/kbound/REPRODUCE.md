# Reproducing K-Bound

This document reproduces the K-Bound short-paper results from committed artifacts. Every headline
number is either (a) rebuilt from raw per-condition logs by the verifier, or (b) traced to a locked
JSON. Nothing here needs the external T9 drive.

## 0. Environment
```
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.lock.txt          # numpy, scikit-learn, matplotlib (pinned)
# GPU re-runs additionally need torch + the WILDS/torchvision data (see §3).
```
A `Dockerfile` at the repo root pins the full environment for GPU re-runs.

## 1. One-command verification (CPU, seconds)
```
python3 docs/research/kbound/scripts/reproduce_headlines.py
```
Exits 0 iff all checks PASS. It:
- **rebuilds** the CIFAR-10-C Tent and EATA beats-both verdicts from the raw per-condition logs
  (`experiments/kbound/results/per_condition_cifar10c_{tent,eata}_seed0.json`) using the exact-rank
  KGA certificate — the verdict is recomputed, not read back;
- checks the decision-gate certificate has FA_u = 0 (`gate_comparison.json`);
- confirms the ImageNet-C SAR, three-source, and Camelyon17 headline numbers are present in the
  locked artifacts (`results_source.json`, `research_lock/KBOUND_MIXED_STREAM_v2.json`,
  `recon_results.json`, `.../imagenetc_aggr/decisive_tta_results.json`).

## 2. Locked artifacts (CI-confirmed headlines)
| Result | Artifact |
|--------|----------|
| CIFAR-10-C Tent/EATA stress beats-both | `results_source.json` (locked_analysis) + raw `per_condition_cifar10c_*` |
| ImageNet-C SAR beats-both | `experiments/kbound/results/win_hunt_v5/imagenetc_aggr/decisive_tta_results.json` |
| Decision-gate comparison | `gate_comparison.json` |
| Three-source mixture (constructed) | `research_lock/KBOUND_MIXED_STREAM_v2.json` |
| Anytime streaming (safety DEMO) | `research_lock/WIN_HUNT_v3_ARM_D_result.json` |
| Exact-rank ablations | `experiments/kbound/results/ablation_exactrank.json` (input SHAs inside) |
| Controller cost profile | `experiments/kbound/results/cost_profile.json` |

## 3. Re-run from logged data (no raw images, seconds each)
```
python3 docs/research/kbound/scripts/ablation_exactrank.py        # alpha/estimator/dropout/transfer
python3 docs/research/kbound/scripts/cost_profile.py              # controller overhead
python3 docs/research/kbound/gapclose_wave5/win_hunt_D_anytime_stream.py   # anytime e-process
```

## 4. Strengthening harnesses (for camera-ready)
- **Official baselines.** Run the authors' released code on the same conditions —
  POEM <https://github.com/yarinbar/poem> (Bar et al., NeurIPS 2024) and
  AETTA <https://github.com/taeckyung/AETTA> (Lee et al., CVPR 2024; estimator `learner/dnn.py::aetta`) —
  convert their per-condition output to the decisions format, then score:
  ```
  python3 docs/research/kbound/scripts/baseline_decisions_adapter.py --method aetta --input aetta_out.csv --out aetta_decisions.json
  python3 docs/research/kbound/scripts/baseline_decisions_adapter.py --method poem  --input poem_out.json  --out poem_decisions.json
  python3 docs/research/kbound/scripts/official_baselines_headtohead.py \
      --decisions poem=poem_decisions.json aetta=aetta_decisions.json
  ```
  (Without `--decisions` it uses clearly-labelled protocol-matched ports.) Produces the head-to-head
  table with a paired-bootstrap CI on the KGA gap and Holm correction.
- **Multi-seed no-harm.** One command runs seeds 0–4 and aggregates (Camelyon17 fully wired; other
  tracks are templated in the script):
  ```
  bash docs/research/kbound/scripts/run_multiseed.sh camelyon
  ```
  or aggregate existing per-seed runs manually:
  ```
  python3 docs/research/kbound/scripts/multiseed_aggregate.py --track iWildCam \
      --glob "experiments/kbound/results/iwildcam_*seed*/*.json"
  ```
  Emits across-seed mean±std, a seed-level bootstrap CI, a stable-no-harm verdict, and a LaTeX row.
  (`--demo` shows a labelled synthetic example.)

## 5. Full GPU re-run (raw data)
The WILDS/CIFAR/ImageNet runners live in `docs/research/kbound/scripts/` (e.g.
`cifar_tent_mps_v2.py`, `run_wilds_camelyon17.py`, `pacs_vlcs_runner.py`) and regenerate the
per-condition logs consumed above. Protocols are pre-registered in `research_lock/*.yaml`.
