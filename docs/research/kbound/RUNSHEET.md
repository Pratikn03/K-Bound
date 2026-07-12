# Run sheet — official baselines, multi-seed, verify (copy-paste)

Everything below runs on your Mac/GPU. K-Bound harness commands assume repo root
`/Volumes/T9/uav/AutoML_Flagship_V8` (or your internal checkout).

## 1. Official AETTA  (conda; CIFAR-10-C)
```bash
git clone https://github.com/taeckyung/AETTA.git && cd AETTA
conda env create -f aetta.yml -n aetta && conda activate aetta
. download_cifar10c.sh          # datasets
. train_src.sh                  # source (frozen) model for CIFAR-10
. tta.sh                        # edit script: DATASET=cifar10, METHOD=aetta  (also run METHOD=no_adapt for the frozen ref)
# per-condition logs land at:  ./log/cifar10/aetta_outdist/<corruption>/<prefix>_<seed>_<dist>/online_eval.json
python print_est.py --dataset cifar10outdist --target aetta
# collect into aetta_out.csv with columns:  condition, est_acc_frozen, est_acc_adapted
```

## 2. Official POEM  (conda py3.10)
```bash
git clone https://github.com/yarinbar/poem.git && cd poem
conda create -n poem python=3.10 && conda activate poem && pip install -r requirements.txt
python main.py --method poem     --model resnet50_gn_timm --exp_type severity_shift --test_batch_size 1
python main.py --method no_adapt --model resnet50_gn_timm --exp_type severity_shift --test_batch_size 1   # frozen ref
# collect per-condition into poem_out.json:  {condition, action: "adapt"|"protect"}  (or adapted vs frozen accuracy)
```

## 3. Convert their output -> decisions, then score (K-Bound harness)
```bash
cd /Volumes/T9/uav/AutoML_Flagship_V8
python3 docs/research/kbound/scripts/baseline_decisions_adapter.py --method aetta --input aetta_out.csv --out aetta_decisions.json
python3 docs/research/kbound/scripts/baseline_decisions_adapter.py --method poem  --input poem_out.json  --out poem_decisions.json
python3 docs/research/kbound/scripts/official_baselines_headtohead.py --candidate tent \
    --decisions poem=poem_decisions.json aetta=aetta_decisions.json
# -> experiments/kbound/results/official_headtohead.json   (regret, FA_u, KGA gap CI, Holm)
```

## 4. Multi-seed no-harm (GPU + WILDS data)
```bash
cd /Volumes/T9/uav/AutoML_Flagship_V8
WILDS_ROOT=~/datasets/wilds bash docs/research/kbound/scripts/run_multiseed.sh camelyon
# -> experiments/kbound/results/multiseed/multiseed_Camelyon17.json
```

## 5. Verify headlines (CPU, seconds)
```bash
python3 docs/research/kbound/scripts/reproduce_headlines.py     # expect: 6 PASS, exit 0
```

## Honest caveat (the one real integration step)
AETTA/POEM run on **corruption × severity**; the K-Bound stream adds
**batch-composition × aggressiveness × repeat** (432 conditions). So either (a) run their models on
the K-Bound condition grid (wire their forward pass into the logged stream), or (b) compare at the
corruption × severity level and say so. The adapter prints exactly which conditions are unmatched, so
you will see immediately if alignment is needed — it never fabricates a missing decision.

Send me `official_headtohead.json` and/or `multiseed_*.json` and I fold the rows into the paper and recompile.
