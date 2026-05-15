# External-Gated Runbooks — Items That Need Data, Accounts, or Wall-Clock

**Purpose:** Three items move the paper's grades meaningfully but can't be
executed inside an editing session because they need either external data
(third paired benchmark), a heavy pretrained model (M3DM-style features),
or external accounts and review cycles (arXiv upload + workshop
submission). This file is the step-by-step runbook for each, ready to
execute when the prerequisite clears.

---

## Runbook 1 — Third paired benchmark (Significance 5.5 → 7)

**Goal:** Replicate the cross-benchmark contrastive finding on a third
naturally paired anomaly-detection dataset so the misfire pattern
becomes a generalizable methodological observation rather than an
MVTec-3D-AD-specific anecdote.

**Dataset candidates (priority order):**

1. **CICIDS-2017 + per-host authentication logs.** Network anomaly +
   user-behavior pair. Both are publicly available; the join key is
   `host_id + timestamp_window`. Most natural fit because the project
   already has cyber + behavior infrastructure.
2. **MIMIC-IV + clinical-notes.** Hospital admission anomaly +
   note-text. Pairing key is `subject_id + admission_id`. Requires
   PhysioNet credentialed access (1–2 day approval).
3. **MVTec LOCO-AD** (logical / structural anomaly detection paired
   with the same MVTec-3D RGB+depth setup). Easiest to set up because
   the existing `prepare_mvtec3d_fusion_benchmark.py` can be cloned
   with minimal changes.

**Execution steps:**

```bash
# 1. Acquire data
#    Option A (CICIDS+auth): download CSE-CIC-IDS2018 from
#    https://www.unb.ca/cic/datasets/ids-2018.html and the matching
#    auth-log subset. Place under data/raw/cicids_auth/ with one
#    directory per host.
#
#    Option B (MIMIC-IV): credentialed download from
#    https://physionet.org/content/mimiciv/2.2/

# 2. Build a prepare_cicids_auth_fusion_benchmark.py script modeled
#    after src/scripts/prepare_mvtec3d_fusion_benchmark.py. The script
#    should:
#    - Discover the natural pairing key (host + timestamp_window)
#    - Emit a long-format fusion CSV with two domain rows per sample
#    - Honor the dataset's predefined train/val/test split if one
#      exists, otherwise use a temporal hold-out
#    - Write a metadata JSON tagged natural_pairing=true

# 3. Add a configs/attention_cicids_auth_fusion.yaml mirroring
#    configs/attention_mvtec3d_fusion.yaml; set split_column to the
#    appropriate column name.

# 4. Run the multi-seed benchmark
PYTHONPATH=src python src/scripts/run_breakthrough_experiment.py \
  --config configs/attention_cicids_auth_fusion.yaml \
  --output experiments/fusion/cicids_auth_results.json

# 5. Render the assets via emit_mvtec3d_assets.py (rename the shim
#    or generalize it to accept a --prefix argument; current
#    implementation hard-codes mvtec3d_)
PYTHONPATH=src python src/scripts/emit_mvtec3d_assets.py \
  --input experiments/fusion/cicids_auth_results.json \
  --metadata experiments/fusion/cicids_auth_metadata.json \
  --figures-dir docs/research/figures \
  --tables-dir docs/research/tables
# (rename mvtec3d_* outputs to cicids_*)

# 6. Add a new §Third Paired Benchmark section to the paper that
#    references the new tables and reports whether the misfire
#    pattern reproduces.
```

**Expected outcomes:**
- If the gate misfires on the third dataset too, the cross-benchmark
  contrast becomes a generalizable methodological observation;
  Significance jumps to 7/10 and the paper has a real shot at a
  mid-tier venue main track.
- If the gate works fine on the third dataset, the paper's framing
  shifts to "the misfire is MVTec-specific" with category-aware drift
  signaling as the natural fix; still publishable, different framing.

**Effort:** 2–4 weeks of focused work depending on dataset acquisition
friction.

---

## Runbook 2 — M3DM-style RGB+3D features (Significance 7 → 8)

**Goal:** Replace the lightweight image-statistic scorer
(`prepare_mvtec3d_fusion_benchmark.py:127-147`) with pretrained
ResNet-50 RGB features + PointNet++ depth features, so the absolute
performance ceiling rises. Random forest's current $0.959$ ROC-AUC
mostly reflects the residual non-linear structure in the eight-dim
hand-crafted features; with stronger features, attention has a fair
chance to be competitive.

**Implementation steps:**

```bash
# 1. Install torchvision + pointnet2_ops (or use a pure-PyTorch
#    PointNet++ implementation)
pip install torchvision
# PointNet++: clone from https://github.com/erikwijmans/Pointnet2_PyTorch
# and pip install -e .

# 2. Replace _image_features() in src/scripts/prepare_mvtec3d_fusion_benchmark.py
#    with a function that:
#    - Loads ResNet-50 pretrained on ImageNet
#    - Strips the final FC layer
#    - Returns the 2048-dim penultimate feature for each RGB image
#    For depth, load the XYZ TIFF into a (3, H*W) point cloud and
#    pass through PointNet++ to get a 1024-dim feature.

# 3. Embedding-dim needs to grow from 8 to ~512 (after PCA projection
#    from 2048+1024 → 512) to fit the attention block.
#    Update configs/attention_mvtec3d_fusion.yaml:
#      data.feature_columns: include embedding_0 ... embedding_511
#      model.embed_dim: 128  (was 32)

# 4. Re-run the prepare script (with --device cuda if available; this
#    is the slow step — ~30 min on M-series GPU for all 8 categories)
PYTHONPATH=src python src/scripts/prepare_mvtec3d_fusion_benchmark.py

# 5. Re-run the breakthrough experiment
PYTHONPATH=src python src/scripts/run_breakthrough_experiment.py \
  --config configs/attention_mvtec3d_fusion.yaml \
  --output experiments/fusion/mvtec3d_results.json

# 6. Regenerate paper assets
./scripts/rebuild_paper.sh
```

**Expected outcomes:**
- If attention matches or exceeds random forest at the higher feature
  quality, the paper's framing changes substantially: the gate
  hurts attention on weak features but not on strong ones. The
  contrast finding becomes more nuanced.
- If random forest still wins, the contrast finding strengthens: even
  at strong feature quality, validation-derived drift gates misfire on
  paired data.

**Effort:** 3–7 days, mostly inference time + integration. The
PointNet++ install is the only friction.

---

## Runbook 3 — arXiv preprint upload (Overall 7.5 → 8.5 on visibility)

**Goal:** Get a citeable preprint up so the work is visible, can be
referenced in cold emails / LinkedIn / workshop submission cover
letters, and starts accumulating citations.

**Prerequisite:** arXiv account + cs.LG endorsement. First-time
submitters need an endorsement from a previously-published author
(1–7 day lead time).

**Execution steps:** All checklist items live in
[ARXIV_SUBMISSION_PACKAGE.md](ARXIV_SUBMISSION_PACKAGE.md). The
package is already complete:

1. Title, abstract, MSC class, ACM class, primary cs.LG + cross-list
   cs.AI + stat.ML, CC BY 4.0 license — all drafted.
2. Source-tarball build steps with the exact `tar -czf` command.
3. 12-item pre-flight checklist (compile warnings, undefined refs,
   TODOs, personal-info leakage, etc.).
4. The 30-second elevator pitch for the work.

**Effort:** 2 hours to upload (mostly form-filling); 1–7 days for
endorsement; 1 day for arXiv moderation.

---

## Runbook 4 — Workshop submission (Overall 8.5 → 9.0 on credit)

**Goal:** Convert the preprint into a peer-reviewed workshop
acceptance.

**Primary target:** NeurIPS 2026 Workshop on Distribution Shifts
(deadline ~mid-September 2026, conference December 2026). The
negative-result framing fits this workshop's scope precisely.

**Backup target:** ICLR 2027 Workshop on Reliable ML (deadline ~late
February 2027, conference April/May 2027).

**Execution steps:** All detail lives in
[PUBLICATION_ROADMAP.md](PUBLICATION_ROADMAP.md) §Target 2. Summary:

1. Watch for the CFP announcement (typically 3–4 months before
   deadline).
2. Trim the paper to workshop length (typically 4–8 pages — the
   current 17-page paper compresses cleanly).
3. Submit via OpenReview.
4. If rejected: 2-week revision pass against reviewer feedback and
   submit to ICLR-W backup.

**Effort:** 1 week of preparation per submission; review cycle is
months out.

---

## Composite roadmap

Realistic order:

| Phase | When | Action | Grade impact |
|---|---|---|---|
| Now | Today | Upload arXiv preprint (Runbook 3) | Overall 7.5 → 8 |
| Weeks 1–2 | After upload | M3DM-style features (Runbook 2) | Significance 7 → 8 |
| Weeks 3–6 | Parallel with M3DM | Third paired benchmark (Runbook 1) | Significance → 8 |
| Month 3 | September 2026 | NeurIPS-W submission (Runbook 4) | Overall → 8.5 |
| Month 6 | November 2026 | NeurIPS-W decision | Overall → 9.0 if accepted |

The arXiv upload is the only step with no prerequisites — everything
in this file is downstream of it.
