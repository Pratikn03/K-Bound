# Family-D v2 — Execution Precheck

**Phase:** 2.2E / Stage 0
**Status:** **STAGE 0 PASSES; STAGE 2 IMPLEMENTATION CHECK SURFACES PROTOCOL GAP.**

## 1. Repository state

- **Branch:** `exp/elara-phase2-mechanism-and-replication`
- **HEAD:** `5679790` (independent review sign-off commit)
- **Working tree:** clean (`git status --short` returns no entries other than the precheck report being written here)
- **Freeze commit `09153cc` present in history:** ✅ verified via `git log --oneline 09153cc`
- **Sign-off commit `5679790` present in history:** ✅ verified

## 2. Independent sign-off

- File present: `docs/research/phase2/FAMILY_D_V2_INDEPENDENT_REVIEW_SIGNOFF.md` ✅
- Final decision string verified verbatim: **`FAMILY_D_V2_EXECUTION_AUTHORISED_UNDER_FROZEN_CONTRACT`** ✅
- Authorisation language verified verbatim: *"Authorised for one-time execution of D-EYE-1 and D-EYE-2 under the unchanged frozen contract. D-EYE-3 may be executed only as secondary descriptive evidence. No protocol change is authorised."*

## 3. Hash anchors — all 14 RE-VERIFIED at execution start

Live recomputation against frozen anchors in `FAMILY_D_PARTITION_MANIFEST_v2.json`:

**Internal anchors (4 / 4 MATCH):**

| Field | SHA256 | Status |
|---|---|:---:|
| `protocol_yaml_sha256` | `104d90c6bab38671bb4dba15414a05ccebc890679cd681a5d46e06e7c8be4f15` | MATCH |
| `hypotheses_csv_sha256` | `0361a960217f0b32f9a96eef9c261d47af2877a895cbb5e10a0115e8303ad8e2` | MATCH |
| `selection_policy_sha256` | `65f81a240b41e54fd7dafdbdf045f65d5e2d5c06909f0e05d9a56e286712e60b` | MATCH |
| `operator_spec_sha256` | `e18bc05d12fb717b6b5dac738e41c022480374cefaa5556b5cfb48ce1e667f5d` | MATCH |

**Per-archive anchors (10 / 10 MATCH):**

CandyCane, ChocolateCookie, ChocolatePraline, Confetto, GummyBear, HazelnutTruffle, LicoriceSandwich, Lollipop, Marshmallow, PeppermintCandy — every per-archive SHA256 recomputes to the value recorded in the manifest. No drift.

**`test_evaluation_executed`:** `false` (manifest, protocol YAML invariants, protocol YAML provenance, every hypothesis row).

## 4. No prior result artifacts

- `experiments/phase2/family_d/` contains only `eyecandies_archive_sha256.txt`, `eyecandies_archive_inventory.csv`, `eyecandies_schema_verification.json` (Phase 2.2D hash + schema outputs).
- No `family_d_v2_prediction_archive_index.csv`.
- No `family_d_v2_primary_inference.csv`.
- No `family_d_v2_holm_k2.csv`.
- No per-seed metric file.
- No Family-D model artifact.

## 5. No paper or thesis change

- `git diff HEAD docs/research/PAPER_DRAFT_v1.tex` empty.
- `git diff HEAD docs/research/THESIS_CHAPTER_v1.tex` empty.

## 6. Stage 0 verdict

> **STAGE 0 (integrity gate): PASS.**

Every authorisation, hash, file-state and no-prior-result invariant is intact. The frozen contract is unchanged, the sign-off is committed, and no Family-D outcome has been read.

## 7. Stage 2 implementation check — protocol gap surfaced

Per the Phase 2.2E spec Section 2, before any test access I must verify the Family-D execution driver. Honest inspection:

- **No Family-D execution driver exists** in `src/scripts/`. Phase 2.2D produced only the *hash-only download + schema verification* scripts (`family_d_v2_download_eyecandies.py`, `family_d_v2_schema_verify.py`, `family_d_v2_write_partition_manifest.py`).
- **No Eyecandies fusion-input CSV exists** at `experiments/fusion/eyecandies_inputs.csv` or similar. The base RGA pipeline (the frozen "primary method") consumes a per-(sample, domain) CSV with columns `sample_id, domain, label, score, confidence, embedding_*`. Eyecandies on disk is raw image data (RGB ×6 / depth / normal PNGs), not pre-computed per-modality anomaly scores.
- **No Eyecandies-specific per-modality one-class anomaly scoring pipeline exists** in the codebase. The closest analog is `src/scripts/prepare_mvtec_loco_fusion_benchmark.py`, which uses ResNet-50 features + Sobel-gradient (edge_proxy) on MVTec LOCO-AD; Eyecandies needs a different domain pairing (rgb + depth, not rgb + edge_proxy) and the script doesn't generalise.

**What this means for the frozen "base RGA on Eyecandies RGB+depth" execution:**

Producing the per-modality scores that the frozen protocol consumes requires implementing a new image → anomaly-score pipeline. That pipeline must commit to **research-level choices** that are *not* specified in any frozen artifact and were *not* reviewed by the independent reviewer:

1. **Feature backbone**: ResNet-50? ResNet-18? ViT? Which pretrained weights (ImageNet-1k, ImageNet-21k, …)?
2. **Feature layer**: layer2 / layer3 / penultimate? Average-pool vs adaptive-pool?
3. **One-class scoring method**: PatchCore-style memory bank? Padim? KNN? Coreset subsampling ratio?
4. **RGB view aggregation**: average the 6 views? Use one canonical view? Max-pool?
5. **Score normalisation per modality**: raw distance? z-score against train? min-max?
6. **Depth pre-processing**: range normalisation? log-transform?

Each of these affects the per-modality scores that the frozen "base RGA" then consumes — which means each choice materially affects the held-out ROC-AUC delta the protocol reports.

**The independent reviewer did not review these choices.** Reviewing them now in this execution task — or making them silently — would constitute either (a) a *de facto* protocol change without independent re-review, or (b) a *de facto* protocol gap whose closure occurred under the executor's discretion rather than under independent scrutiny.

## 8. Verdict on this execution task

> **STAGE 2 (implementation check): GAP DETECTED.**
>
> The frozen Family-D v2 protocol is internally consistent but it presumes a per-(category, sample, modality) one-class anomaly-score pipeline as upstream of "base RGA". That pipeline does not exist in this codebase. Implementing one in this execution task introduces research-grade design choices that were not part of the freeze and were not reviewed by the independent reviewer.

Per the Phase 2.2E spec verbatim:

> "If any frozen artifact must be changed to make execution possible: STOP. Do not execute Family-D. Create `docs/research/phase2/FAMILY_D_V2_EXECUTION_BLOCKED_PROTOCOL_CHANGE_REQUIRED.md`."

The frozen artifacts themselves are not being changed; rather, **completing them requires writing the upstream scoring pipeline, which carries equivalent epistemic weight to a protocol change**. The honest, scientifically rigorous response is to invoke the spec's blocked branch and stop here.

The follow-up companion file [FAMILY_D_V2_EXECUTION_BLOCKED_PROTOCOL_CHANGE_REQUIRED.md](./FAMILY_D_V2_EXECUTION_BLOCKED_PROTOCOL_CHANGE_REQUIRED.md) (next file written by this task) details the gap, the required closure work for a Phase 2.2E.0 sub-task (lock the Eyecandies scoring pipeline as a v3 freeze artifact, re-run independent review, only then execute), and what claims remain forbidden in the meantime.

## 9. State at end of precheck

- Family-A K=5 results: **unchanged**.
- Family-B closure: **unchanged**.
- Family-D v1 / v2 frozen artifacts: **unchanged**.
- `test_evaluation_executed` everywhere: **still `false`**.
- No model trained, no metric computed, no anomaly mask inspected.
- Paper / thesis: **unchanged**.
- Phase 3 / ELARA-Universal / ORIUS: **not opened**.
