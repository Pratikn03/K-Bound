# Physical Edge study — capture & publication pipeline

## Start here (R2 kickoff)

```bash
cd /Volumes/T9/uav/AutoML_Flagship_V8
bash scripts/start_r2_physical_capture.sh preflight   # fast status
bash scripts/start_r2_physical_capture.sh pilot       # 4-clip warm-up (real camera)
bash scripts/start_r2_physical_capture.sh session S01   # full session (interactive)
```

Tracker: `edge/R2_SESSION_TRACKER.md`

## Current blocker (read first)

`artifacts_real/raw/` currently holds **1,216 clips** generated with `01_capture_real_session.py --mock`
(random 224×224 noise). They exercise the **code path** but cannot pass the **source model gate**
(≥ 0.80 val balanced acc / macro-F1 on real generalization) or produce meaningful R2 table numbers.

**For publication:** re-capture with a **real camera** (no `--mock`), labeled cardboard boxes P01–P10.

---

## What to photograph

| Class | Show |
|-------|------|
| `ok` | White shipping label, flat on brown cardboard |
| `missing_label` | Same box, label removed |
| `misaligned_label` | Label rotated ~90° |
| `damaged_label` | Label torn / peeling |

Objects **P01–P06** → training (S01). **P07–P08** → validation (S02). **P09–P10** → held-out / replication only.

---

## Capture commands (from `edge/scripts/`)

Use repo venv: `../../../../.venv/bin/python` or `cd AutoML_Flagship_V8` first.

```bash
cd AutoML_Flagship_V8/docs/research/kbound/edge/scripts
PY=../../../../../.venv/bin/python
CFG=../configs/edge_real_phone_v1.yaml
```

### Optional: pilot (4 clips)

```bash
$PY 01_capture_real_session.py --config $CFG --pilot --phone-id phone_a --camera 1
```

### Full protocol (interactive; one session at a time)

```bash
# Source train — S01 (240 windows worth of clips)
$PY 01_capture_real_session.py --config $CFG --session S01 --phone-id phone_a --camera 1

# Source val — S02
$PY 01_capture_real_session.py --config $CFG --session S02 --phone-id phone_a --camera 1

# Calibration fit — S03, S04
$PY 01_capture_real_session.py --config $CFG --session S03 --phone-id phone_a --camera 1
$PY 01_capture_real_session.py --config $CFG --session S04 --phone-id phone_a --camera 1

# Conformal — S05, S06 (different day recommended)
$PY 01_capture_real_session.py --config $CFG --session S05 --phone-id phone_a --camera 1
$PY 01_capture_real_session.py --config $CFG --session S06 --phone-id phone_a --camera 1

# After sealing dev splits (pipeline step 1 with --seal-through):
$PY 02_validate_real_dataset.py --config $CFG --through calibration_conformal --seal-through calibration_conformal --strict

# Held-out — S07, S08 (only after seal)
$PY 01_capture_real_session.py --config $CFG --session S07 --phone-id phone_a --camera 1
$PY 01_capture_real_session.py --config $CFG --session S08 --phone-id phone_a --camera 1

# Replication phone B — S09, S10
$PY 01_capture_real_session.py --config $CFG --session S09 --phone-id phone_b --camera 1
$PY 01_capture_real_session.py --config $CFG --session S10 --phone-id phone_b --camera 1
```

Press **ENTER** at each prompt after placing the box. Press **Q** to pause.

---

## One-command pipeline (after captures exist)

```bash
cd AutoML_Flagship_V8
bash docs/research/kbound/edge/scripts/run_edge_publication_pipeline.sh
```

Steps: validate → train (no bypass) → calibrate → replay held-out + replication → audit → export TeX → dashboard.

---

## Success criteria

| Gate | Threshold |
|------|-----------|
| S02 val balanced acc | ≥ 0.80 |
| S02 val macro-F1 | ≥ 0.80 |
| Held-out balanced acc | > 0.30 (above 4-class chance) |
| KGA abstain rate | < 95% (some adapt/freeze decisions) |
| Audit | 8/8 checks pass |
| Dashboard | `study_status: verified` |
| Paper | `camera_tables_values.tex` filled; `kbound_short.tex` R2 cells non-empty |

---

## Recompile short paper

```bash
cd AutoML_Flagship_V8/docs/research/kbound
pdflatex kbound_short.tex
```

Section **Real-camera deployment protocol** already `\input`s `edge/kbound_camera_main_tables.tex`, which pulls macros from `experiments/kbound/results/edge_real_phone_v1/camera_tables_values.tex`.
