# Family D — Execution Commands (NOT RUN this session)

**Status:** PRE-REGISTERED. These commands MUST be run verbatim — no
hidden flags, no off-script reruns — when the Family-D compute window
opens. This file forms part of the freeze commit.

```bash
# ---- 0. Verify the freeze commit hash matches the contract ----
git rev-parse HEAD  # must match FAMILY_D_PARTITION_MANIFEST.json:freeze_commit

# ---- 1. Download datasets and record archive hashes ----
# (the download scripts themselves are out of scope of Phase 2 — they
# must exist and produce the hashes recorded in PARTITION_MANIFEST.json)
PYTHONPATH=src .venv/bin/python src/scripts/download_family_d_dataset.py --dataset mpdd
PYTHONPATH=src .venv/bin/python src/scripts/download_family_d_dataset.py --dataset eyecandies

# ---- 2. Verify dataset SHA256 against the manifest ----
PYTHONPATH=src .venv/bin/python src/scripts/verify_family_d_dataset_hashes.py

# ---- 3. Run the 30-seed pilot — MPDD PatchCore supervised-paired (D-H1, D-H2) ----
PYTHONPATH=src .venv/bin/python src/scripts/run_phase2_powered_audited_pilot.py \
    --experiment-id D-H1 --dataset MPDD --backbone PatchCore --protocol supervised-paired \
    --seeds 30 --seed-start 42

# ---- 4. Run the 30-seed pilot — MPDD PatchCore one-class (D-H3) ----
PYTHONPATH=src .venv/bin/python src/scripts/run_phase2_powered_audited_pilot.py \
    --experiment-id D-H3 --dataset MPDD --backbone PatchCore --protocol one-class \
    --seeds 30 --seed-start 42

# ---- 5. Run the 30-seed pilot — Eyecandies PatchCore supervised-paired (D-H4, D-H5) ----
PYTHONPATH=src .venv/bin/python src/scripts/run_phase2_powered_audited_pilot.py \
    --experiment-id D-H4 --dataset Eyecandies --backbone PatchCore --protocol supervised-paired \
    --seeds 30 --seed-start 42

# ---- 6. Audited inference for the family ----
PYTHONPATH=src .venv/bin/python src/scripts/run_phase2_powered_audited_analysis.py \
    --family D --hypotheses D-H1,D-H2,D-H3,D-H4,D-H5

# ---- 7. Validate prediction-archive integrity ----
PYTHONPATH=src .venv/bin/python src/scripts/validate_phase2_prediction_archives.py

# ---- 8. Verify the contract files were not modified between freeze and execution ----
PYTHONPATH=src .venv/bin/python src/scripts/verify_family_d_contract_integrity.py
```

The two helper scripts in steps 2 and 8 do not yet exist; their
contracts are:

- `verify_family_d_dataset_hashes.py` — fails iff the downloaded
  archive SHA256 does not match `FAMILY_D_PARTITION_MANIFEST.json`.
- `verify_family_d_contract_integrity.py` — fails iff any of the five
  Family-D contract files have been modified between the freeze commit
  and `HEAD`.

Both must be added to the repo as part of the same commit that
populates `FAMILY_D_PARTITION_MANIFEST.json:expected_sha256_of_archive`.

## Stop boundary

Nothing in this file is to be executed in the current session. The
contract is in place so that, **when** the Family-D compute window
opens, the execution is mechanical and the inferential statement is
genuinely confirmatory.
