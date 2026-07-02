# EDGE_COMPLETION_CHECKLIST — finishing the real-camera study

> One tick-list to follow on recording day. The **code is done**; what remains is
> recording real frames and running one command. Generating fake camera numbers is
> not allowed — every number below must come from frames your phone actually saw.

---

## 0. Status — the CODE is finished (verified)

- **7,927 lines** of Python, full pipeline `00 → 12`, 22 modules, 18 test files, 4 configs.
- **79/79 torch-free tests pass**; the torch-dependent tests run on your Mac's venv (not bugs).
- **0 real stubs** (the one `NotImplementedError` is a correct abstract base method).
- Synthetic end-to-end already runs; the **live dashboard renders** adapt/freeze/abstain.
- The only missing ingredient is **real recorded frames**. Everything below is recording + one command.

---

## 1. What your phone must see — the 4 label states

A cardboard box with a shipping label, in one of four states:

| Class | What it is | How to stage it |
|---|---|---|
| `ok` | Label present, flat, straight, intact, correct spot | Stick a clean white label squarely on the box |
| `missing_label` | **No label at all** — bare cardboard where a label should be | Peel the label off / leave the spot empty |
| `misaligned_label` | Label present but **crooked / rotated / off-position** | Stick it rotated ~45–90°, or hanging over an edge |
| `damaged_label` | Label present but **torn / peeling / crumpled / smeared** | Rip a corner, peel half off, or crumple it |

**The distinction you asked about:**
- **missing** = the label is *gone* (nothing there — absence).
- **damaged** = the label is *there but ruined* (torn, peeling, crumpled).

**Practical tips**
- One box + a stack of identical white labels (or white paper rectangles / sticky notes) is enough.
- Use ~10 different boxes (P01–P10) so the model learns "label state," not "this one box."
- Per class, capture many frames: rotate the box, move closer/farther, change the lighting.
- Phone on a stand, box centered and filling most of the frame.

---

## 2. Capture plan (sessions S01–S10)

| Session | Split | Boxes | Notes |
|---|---|---|---|
| S01 | source train | P01–P06 | clean, good light, all 4 classes |
| S02 | source val | P07–P08 | clean — **this is the 0.80 gate** |
| S03, S04 | calibration-fit | mixed | moderate physical shifts |
| S05, S06 | calibration-conformal | mixed | **different day**, disjoint shifts |
| **— SEAL dev splits here —** | | | |
| S07, S08 | held-out test | P09–P10 | hard shifts: glare, blur, new angle |
| S09, S10 | replication | P09–P10, phone B | second phone |

Record **dev first (S01–S06) → seal → then held-out (S07–S10)**. That ordering *is* the
anti-leakage guarantee: you must not see the held-out data while tuning.

**Capture (one session at a time), from `edge/scripts/`:**
```bash
cd AutoML_Flagship_V8/docs/research/kbound/edge/scripts
# Phase 1 — source gate only (recommended):
bash run_edge_source_gate.sh

# Phase 2 — after gate passes (calibration + held-out + publication):
bash run_edge_heldout_capture.sh
```

Or manually per session:
```bash
PY=../../../../../.venv/bin/python
CFG=../configs/edge_real_phone_v1.yaml
$PY 01_capture_real_session.py --config $CFG --session S01 --phone-id phone_a --camera 1
# repeat for S02 … S10 ; press ENTER after placing each box, Q to pause
```

**Seal after S06 (before recording held-out):**
```bash
$PY 02_validate_real_dataset.py --config $CFG --through calibration_conformal --seal-through calibration_conformal --strict
```

---

## 3. Watch the live dashboard while you record (recommended)

```bash
$PY 07_shadow_live.py --camera 0 --view window     # try --camera 0, then --camera 1
```
The frozen model is the official output; the candidate + KGA verdict run **in shadow**
(logged, never emitted). See `edge/artifacts/dashboard_demo_montage.png` for the exact view.

---

## 4. Run the whole study — ONE command (verified order)

After all S01–S10 captures exist:
```bash
cd AutoML_Flagship_V8
bash docs/research/kbound/edge/scripts/run_edge_publication_pipeline.sh
```
Order: data-sanity → validate → **train f0 (≥0.80 gate)** → calibration → fit KGA →
held-out replay → replication → runtime profile → ablations → **strict anti-leakage audit**
→ export LaTeX → report + dashboard. (No `--bypass-gate`: the source model must really pass.)

---

## 5. Success gates (must pass for a real result)

| Gate | Threshold |
|---|---|
| S02 val balanced accuracy | ≥ 0.80 |
| S02 val macro-F1 | ≥ 0.80 |
| Held-out balanced accuracy | > 0.30 (above 4-class chance) |
| KGA abstain rate | < 95% (some adapt/freeze happen) |
| Anti-leakage audit | 8/8 checks pass |
| Dashboard | `study_status: verified` |
| Paper | `camera_tables_values.tex` filled; R2 cells non-empty |

---

## 6. Fill the paper

```bash
cd AutoML_Flagship_V8/docs/research/kbound && pdflatex kbound_short.tex
```
The R2 table (currently em-dash "pending") auto-populates from `camera_tables_values.tex`
once step 4 writes it to `docs/experiments/kbound/results/edge_real_phone_v1/`.

---

## 7. Honest outcome expectations

It's a real experiment — three honest endings, all publishable:
- **Held-out win** (KGA beats both fixed policies, CI-robust) → lifts the paper a tier.
- **No-harm** (KGA matches the better policy at zero false-adapt) → clean, honest result.
- **Abstain-heavy** (KGA declines under hard shift) → the certificate behaving correctly;
  frame as the *unknowable* regime the theory predicts.

Report whatever actually happens. None of these is guaranteed in advance.

---

## Gotchas

- The pipeline calls `AutoML_Flagship_V8/.venv/bin/python`. Confirm that venv has
  **torch, torchvision, opencv-python, scikit-learn, joblib, pyyaml**. (The README mentions
  `.venv_wilds` — point `PY` at whichever venv has torch.)
- **Never use `--mock` for publication** (mock = random noise; it fails the 0.80 gate by design).
- Record on **≥2 different days** wherever the plan says "different day" — that day-to-day
  change is what makes the shift real and the conformal calibration meaningful.
