# Next-phase release checklist

This release extends the source branch `codex/kbound-nextphase-20260921`. The
previous Desktop release is immutable and is not the destination for any build.
Use a new directory such as `output/release/KBound_Next_Phase_20260921` in this
worktree. A verification pass establishes the named software/artifact checks;
it does not establish scientific novelty, exchangeability, transport assumptions,
or production field validation.

## Freeze the actual experiment record

Stop study writers before freezing their evidence. Keep the first failed
calibration attempt, original and corrected reports, all protocol/decision
seals, original executed-source copies, raw diagnostic counts, decisions,
outcomes, witnesses, timing receipts, and superseded engineering runs. Do not
replace original results with corrected report bytes. Record the active result
authorities explicitly:

| Study | Active authority and required interpretation |
| --- | --- |
| Calibration | `calibration_value_v1/FINAL_RESULTS_SUMMARY.json`, plus `group_risk_report_05/GROUP_RISK_CORRECTION_RECEIPT.json` and its group metrics. Original `report_03` CIFAR/mixed results and `attempt_04_imagenet_schema_corrected` ImageNet results remain separate. All are retrospective. |
| Paired transport | `paired_transport_v1/receipt.json`, `summary.json`, counts, decisions, outcomes, witnesses, execution sources, and seal. Assumption-violating negative controls remain visible. |
| Natural shift | `natural_v2/calibration_summary.json` and `gate_screen_v2.json`. If eligibility fails, preserve `STOP_NO_TARGET_ACCESS`; no target scoring authority exists or is required. If eligibility passes and authorized target execution completes, additionally bind `target_inference_v2.json`, target decisions/prediction inventories, and reveal receipts. |
| Deployment lifecycle | `deployment_v4/summary.json`, protocol, journal and executed-source snapshots; retain `deployment_v1`, `deployment_v2`, and `deployment_v3` as superseded, not final timings. |
| Natural execution costs | `natural_cost_v1` records the real neural pipeline on the already opened development city, across all five checkpoints with three repeats. Its final receipt must identify timings, memory, exact inputs and executed source; these are cost observations, not held-out effectiveness results. |
| Docker service | The final-source `docker_v2/summary.json`, protocol and log; retain `docker_v1` as preliminary. The certificate service does not demonstrate neural adaptation execution or field deployment. |

The following command is a template to execute **after** the named directories
are complete. Add the final deployment/Docker directory names actually produced;
do not create fake authorities to satisfy the template. Each selected directory
needs at least one `--authority`, and multiple authorities can be supplied.

```bash
"$KBOUND_PYTHON" docs/research/kbound/scripts/build_next_phase_evidence.py \
  --manifest docs/research/kbound/next_phase/evidence_manifest.json \
  --study-directory output/next_phase/calibration_value_v1 \
  --authority output/next_phase/calibration_value_v1/FINAL_RESULTS_SUMMARY.json \
  --authority output/next_phase/calibration_value_v1/group_risk_report_05/GROUP_RISK_CORRECTION_RECEIPT.json \
  --study-directory output/next_phase/paired_transport_v1 \
  --authority output/next_phase/paired_transport_v1/receipt.json \
  --study-directory output/next_phase/natural_v2 \
  --authority output/next_phase/natural_v2/calibration_summary.json \
  --authority output/next_phase/natural_v2/gate_screen_v2.json
```

The tool recursively inventories only the explicitly named immediate children
of `output/next_phase`. It never discovers datasets or unopened targets outside
those directories. It rejects symlinks, special files, unsafe paths, files over
64 MiB, or total evidence over 512 MiB. It preserves all selected files, including
failures and execution snapshots. The manifest and archive are create-only.
If a subsequent correction is needed, retain the old manifest and create a new
named manifest, update the explicit source-seal entry, and refreeze source.

## Commit all maintained sources before verification

The source seal now binds all `kga` modules, root tests, next-phase protocol and
report JSON/Markdown, the explicit study runners, and the exact six So2Sat v2
configuration/receipt paths. The verification runner includes the new natural
runner in its authored-source inventory. Outputs remain ignored: the committed
`evidence_manifest.json` binds their hashes transitively, and the independent
archive verifier checks every selected byte.

These **nine** source/configuration files are ignored and require explicit
forced staging after review (not a broad `git add -f experiments`):

```bash
git add -f -- \
  experiments/kbound/next_phase/__init__.py \
  experiments/kbound/next_phase/natural_runner.py \
  experiments/kbound/next_phase/README.md \
  experiments/kbound/so2sat/prospective_protocol_v2.json \
  experiments/kbound/so2sat/prospective_protocol_v2.json.receipt.json \
  experiments/kbound/so2sat/precalibration_seal_v2.template.json \
  experiments/kbound/so2sat/precalibration_seal_v2.template.json.receipt.json \
  experiments/kbound/so2sat/tent_citymean5_checkpoint_ridge_v2.json \
  experiments/kbound/so2sat/tent_citymean5_checkpoint_ridge_v2.json.receipt.json
```

Review and stage other intended changes normally. Do not stage `output/`, raw
HDF5 files, checkpoint tensors, `.lake`, or unrelated old Desktop changes. The
source seal fails if mandatory new sources/configs/manifest are absent from the
selected commit. Its small maintained scopes also detect untracked new runtime
or protocol files before release.

## Run fresh source-bound checks

Use the verified release Python 3.12 environment, not the absent old `.venv`.
Run repository scripts with `PYTHONPATH=.` from the worktree root so the repository
namespace imports resolve. Set `KBOUND_PYTHON` to that existing verified interpreter.
Set `KBOUND_DOC_PYTHON` and `KBOUND_DOC_RENDERER` to the bundled Python and documents
skill renderer returned by workspace dependency discovery. Do not install document
packages into the content-pinned research interpreter. Set the following task-local
paths; keep package inputs outside the create-only final delivery directory:

```bash
export PYTHONPATH=.
KBOUND_KB=docs/research/kbound
KBOUND_PACKAGE_INPUTS=output/next_phase_release_inputs/20260921
KBOUND_DELIVERY=output/release/KBound_Next_Phase_20260921
```
Freeze the source commit after all experiment/reporting fixes and document-source
updates. The exact current commit must drive every full verification receipt:

```bash
KBOUND_NEXT_COMMIT=$(git rev-parse --verify HEAD)
"$KBOUND_PYTHON" docs/research/kbound/scripts/build_release_source_seal.py \
  --source-commit "$KBOUND_NEXT_COMMIT" --preflight
"$KBOUND_PYTHON" docs/research/kbound/scripts/run_repository_verification.py \
  --repo "$PWD" --python "$KBOUND_PYTHON" \
  --output docs/research/kbound/audits/repository_test_inventory.json \
  --expected-source-commit "$KBOUND_NEXT_COMMIT" \
  --authorize-protected-so2sat --all-gates
```

Do not replace this receipt with an inventory-only or pytest-only run. In
particular, the compatibility runbook's `all` mode with
`KBOUND_PACKAGE_RELEASE=1` invokes the runner again without `--all-gates` and would
overwrite this canonical receipt. Follow the explicit phases below instead.
Complete any runtime/toolchain audit receipt refresh before the final checksum
phase; those audit files are themselves canonically checksummed.

`--authorize-protected-so2sat` authorizes the declared tests/validator to read
the already opened development authority. It is not authorization to execute
target experiments or relax the calibration eligibility screen. Physical-camera
tests remain explicitly deferred future work. Historical target-analysis writers
and vendored upstream environment tests retain their documented exclusions.
Inventory-only and pytest-only receipts are not complete all-gates passes.

Both formal projects pin Lean 4.29.1. The old KBound `.lake` resolves to a roughly
7.0 GiB cache; the old multiclass cache is roughly 6.7 MiB. The new worktree began
without caches; both have now been APFS clone-copied into independent new `.lake`
directories. Verify committed `lake-manifest.json` identities, then build in the
new worktree. Do not share a writable old-release cache through a symlink or run
`lake update`.

The new worktree contains every preexisting explicit source-seal input and the
retained compact So2Sat v1 development/result authorities. Full raw execution
requires the separately hash-bound source HDF5/checkpoints and opaque official
validation/testing containers on the external data volume. Those large files
do not belong in the reviewer ZIP; preserve their hashes, source identities,
access record and license/download instructions. A hash is not missing producer
execution evidence.

## Build and inspect documents

```bash
PYTHON="$KBOUND_PYTHON" BUILD_LONG_TMLR=1 BUILD_DOCX=0 \
  bash docs/research/kbound/scripts/build_pdfs.sh
"$KBOUND_DOC_PYTHON" docs/research/kbound/scripts/build_docx.py \
  --output docs/research/kbound/kbound_short_final_draft.docx
"$KBOUND_DOC_PYTHON" docs/research/kbound/scripts/derive_main_only_documents.py \
  --full-pdf docs/research/kbound/kbound_short_final_draft.pdf \
  --full-docx docs/research/kbound/kbound_short_final_draft.docx \
  --output-dir "$KBOUND_PACKAGE_INPUTS/main_only"
"$KBOUND_PYTHON" docs/research/kbound/scripts/render_pdf_pages.py \
  --output-root "$KBOUND_PACKAGE_INPUTS/pdf_pages"
"$KBOUND_DOC_PYTHON" "$KBOUND_DOC_RENDERER" \
  docs/research/kbound/kbound_short_final_draft.docx \
  --output_dir "$KBOUND_PACKAGE_INPUTS/full_word_render" --emit_pdf
"$KBOUND_DOC_PYTHON" "$KBOUND_DOC_RENDERER" \
  "$KBOUND_PACKAGE_INPUTS/main_only/KBound_Without_Appendices.docx" \
  --output_dir "$KBOUND_PACKAGE_INPUTS/main_word_render" --emit_pdf
mkdir -p "$KBOUND_PACKAGE_INPUTS/main_pdf_pages"
pdftoppm -png -r 144 \
  "$KBOUND_PACKAGE_INPUTS/main_only/KBound_Without_Appendices.pdf" \
  "$KBOUND_PACKAGE_INPUTS/main_pdf_pages/page"
```

The first build produces the compact full paper and synchronized TMLR companion;
it does not create a main-only paper. The derivation tool checks the appendix
boundary and preserves every retained PDF page's text/raster and the DOCX body
prefix. Its output directory is create-only: use a new versioned inputs directory
for a revised derivation. `render_pdf_pages.py` covers only the two canonical PDFs;
the explicit main-PDF render and both Word renders are additional requirements.

`build_docx.py` also writes the mandatory canonical
`docs/research/kbound/kbound_short_final_draft.figures.json`. This records the exact
DOCX hash, five source-PDF rasterizations, and embedded-media hashes. It must be
present from the final Word build before checksums; an old receipt cannot cover a
new DOCX. Native equation and page-layout inspection remain separate requirements.

Render and inspect **every** page of the actual delivered full PDF, main-paper-only
PDF, and Word rendering. Check the main-paper cutoff, references, new tables,
equations, captions, line overflow, figure placement, and absence of appendices
in the requested main-only versions. Retain page counts, SHA-256 hashes, rendering
results, and an actual visual-QA record. PDF rendering alone is not visual review.
The historical `short_final_draft` basename is not proof that a document is short.

After actual inspection, write `$KBOUND_PACKAGE_INPUTS/document_qa.json` with
`status: "PASS"` and one `documents` row for **each** delivered PDF and DOCX. Each
row must contain its repository-relative `path`, `bytes`, `sha256`, positive
integer `pages`, and `inspected_pages` containing every integer from 1 through
`pages`. For DOCX use the page count of its final rendering and record that render
identity and inspection notes as additional fields. Include the compact full
PDF, TMLR PDF, main-only PDF, full DOCX and main-only DOCX if using the command below.
The outer builder rejects missing or partial page inventories and stale file
hashes; it cannot establish that visual inspection actually happened.

Record any generated variants in the outer delivery manifest; the canonical
checksum tool's existing allowlist does not automatically cover a new arbitrary
filename. A source change after the verification run requires a new source
freeze and relevant/full checks; a byte-only document rebuild still requires
fresh checksums and visual inspection.

## Seal and independently package

The order is source/evidence freeze, source-bound all-gates pass, final documents
and inspection, public-bundle verification, source seal, main checksum list,
anonymous supplement, post-checksum list, and finally the outer reviewer package.
All source changes require a new source freeze and verification; output-only
rebuilds require refreshed document inspections and all downstream hashes.

Verify the required CCT-20 public bundle against its unchanged source manifest
before writing the canonical checksums. If it genuinely requires rebuilding,
use the builder's authenticated-source mode and then reverify; missing descriptors
are a blocker, not permission to downgrade provenance:

```bash
"$KBOUND_PYTHON" docs/research/kbound/scripts/build_cct20_public_bundle.py \
  --release-manifest docs/research/kbound/paper/generated/cct20_release_manifest.json \
  --output docs/research/kbound/release/cct20_public_evidence_bundle.zip --check
```

After generated-output checks, emit the canonical source seal in this worktree,
then create the next-phase evidence archive outside the final delivery directory:

```bash
"$KBOUND_PYTHON" docs/research/kbound/scripts/build_release_source_seal.py \
  --source-commit "$KBOUND_NEXT_COMMIT"
"$KBOUND_PYTHON" docs/research/kbound/scripts/build_release_source_seal.py \
  --source-commit "$KBOUND_NEXT_COMMIT" --check
"$KBOUND_PYTHON" docs/research/kbound/scripts/build_next_phase_evidence.py \
  --manifest docs/research/kbound/next_phase/evidence_manifest.json \
  --archive "$KBOUND_PACKAGE_INPUTS/next_phase_evidence.zip"
"$KBOUND_PYTHON" docs/research/kbound/scripts/build_next_phase_evidence.py \
  --manifest docs/research/kbound/next_phase/evidence_manifest.json \
  --archive "$KBOUND_PACKAGE_INPUTS/next_phase_evidence.zip" --check
"$KBOUND_PYTHON" docs/research/kbound/scripts/verify_release_checksums.py \
  docs/research/kbound/KBOUND_RELEASE_SHA256SUMS.txt --root "$PWD" --write
"$KBOUND_PYTHON" docs/research/kbound/scripts/verify_release_checksums.py \
  docs/research/kbound/KBOUND_RELEASE_SHA256SUMS.txt --root "$PWD"
```

The evidence archive is deterministic, byte-preserving, internally verified and
**for internal review**: original receipts retain provenance paths. It is not
anonymous or automatically published. The existing anonymous supplement builder
packages the maintained paper closure/public CCT-20 bundle and embeds the source
seal; it does not automatically include this next-phase evidence ZIP. Keep these
distinct artifacts and review their intended audiences.

The anonymous supplement embeds the completed source seal and main checksum
list, so it must be generated **after** those files. It belongs only to the separate
post-checksum inventory; adding it to the main checksum list would form a cycle.
`--replace-verified` replaces only an independently verified previous archive in
this new worktree; it does not touch the old Desktop release.

```bash
"$KBOUND_PYTHON" docs/research/kbound/scripts/build_anonymous_supplement.py \
  --root "$PWD" \
  --source-seal docs/research/kbound/audits/release_source_seal_2026_08_29.json \
  --checksums docs/research/kbound/KBOUND_RELEASE_SHA256SUMS.txt \
  --public-bundle docs/research/kbound/release/cct20_public_evidence_bundle.zip \
  --output docs/research/kbound/release/kbound_anonymous_supplement.zip \
  --replace-verified
"$KBOUND_PYTHON" docs/research/kbound/scripts/build_anonymous_supplement.py \
  --source-seal docs/research/kbound/audits/release_source_seal_2026_08_29.json \
  --checksums docs/research/kbound/KBOUND_RELEASE_SHA256SUMS.txt \
  --public-bundle docs/research/kbound/release/cct20_public_evidence_bundle.zip \
  --output docs/research/kbound/release/kbound_anonymous_supplement.zip \
  --portable-check
"$KBOUND_PYTHON" - <<'PY'
from pathlib import Path
from docs.research.kbound.scripts.verify_release_checksums import (
    POST_CHECKSUM_FILE, POST_CHECKSUM_REQUIRED_PATHS, write_checksum_file,
)
write_checksum_file(Path(POST_CHECKSUM_FILE), root=Path.cwd(),
                    required_paths=POST_CHECKSUM_REQUIRED_PATHS)
PY
"$KBOUND_PYTHON" docs/research/kbound/scripts/verify_release_checksums.py \
  docs/research/kbound/KBOUND_POST_CHECKSUM_SHA256SUMS.txt --root "$PWD" --post-checksum
"$KBOUND_PYTHON" docs/research/kbound/scripts/verify_release_checksums.py \
  docs/research/kbound/KBOUND_RELEASE_SHA256SUMS.txt --root "$PWD"
```

Do not use `verify_release_checksums.py --write --post-checksum`: its write mode
always selects the main inventory. Do not run the compatibility runbook's
`post-checksums` phase at this point: it refreshes runtime/toolchain audit receipts
that are already covered by the main checksum list.

The outer reviewer delivery must contain:

1. The inspected full and main-paper-only PDFs and requested Word variant(s).
2. The exact source archive/commit identity and source seal.
3. Fresh all-gates verification receipt and useful gate logs.
4. The committed next-phase evidence manifest and verified evidence ZIP.
5. Calibration/theory/natural/deployment result reports with corrected authorities
   and all remaining scientific limits stated.
6. Canonical release checksums plus an exact outer delivery manifest covering
   every extra PDF/Word variant, ZIP, verification receipt and report.
7. The final outer archive's own SHA-256 recorded outside that archive (avoid a
   self-referential checksum).

Create the outer package only after all inputs above are stable. Every document
and report path must be inside this worktree. The source ZIP is the exact union of
source-seal entries and canonical/post-checksum entries, plus both checksum lists;
it is a scoped release inventory, not complete Git history. The builder verifies
nested ZIP members directly without extraction or scanning arbitrary result trees.

```bash
"$KBOUND_PYTHON" docs/research/kbound/scripts/build_next_phase_review_package.py \
  --repo "$PWD" \
  --seal docs/research/kbound/audits/release_source_seal_2026_08_29.json \
  --checksums docs/research/kbound/KBOUND_RELEASE_SHA256SUMS.txt \
  --post-checksums docs/research/kbound/KBOUND_POST_CHECKSUM_SHA256SUMS.txt \
  --evidence-manifest docs/research/kbound/next_phase/evidence_manifest.json \
  --evidence-archive "$KBOUND_PACKAGE_INPUTS/next_phase_evidence.zip" \
  --verification docs/research/kbound/audits/repository_test_inventory.json \
  --qa "$KBOUND_PACKAGE_INPUTS/document_qa.json" \
  --document KBound_With_Appendices.pdf=docs/research/kbound/kbound_short_final_draft.pdf \
  --document KBound_TMLR_With_Appendices.pdf=docs/research/kbound/kbound_tmlr.pdf \
  --document "KBound_Without_Appendices.pdf=$KBOUND_PACKAGE_INPUTS/main_only/KBound_Without_Appendices.pdf" \
  --document KBound_With_Appendices.docx=docs/research/kbound/kbound_short_final_draft.docx \
  --document "KBound_Without_Appendices.docx=$KBOUND_PACKAGE_INPUTS/main_only/KBound_Without_Appendices.docx" \
  --report "main_only_derivation.json=$KBOUND_PACKAGE_INPUTS/main_only/main_only_derivation.json" \
  --report docx_figures.json=docs/research/kbound/kbound_short_final_draft.figures.json \
  --report calibration_value_results.md=docs/research/kbound/next_phase/calibration_value_results.md \
  --report paired_transport_theory.md=docs/research/kbound/next_phase/paired_transport_theory.md \
  --output-directory "$KBOUND_DELIVERY"
"$KBOUND_PYTHON" docs/research/kbound/scripts/build_next_phase_review_package.py \
  --output-directory "$KBOUND_DELIVERY" --check
```

Add other final result/deployment reports with explicit `--report name=path`.
The builder writes the new directory, sibling `.zip`, and sibling
`.zip.receipt.json`; all three destinations must be absent. It rejects duplicate
or unsafe entries, checks every outer member hash, checks source blobs and evidence
hashes from archived bytes, matches the all-gates receipt's commit/tree to the seal,
checks all document QA bindings, and compares the directory with the ZIP. The
external receipt records the final ZIP SHA-256 without a self-hash cycle.
Do not reuse the old release's checksum,
timestamp, test inventory, or status as proof for this release.

## Open scientific gates remain separate

The retrospective calibration comparison does not become a prospective success
when packaged. Natural calibration may legitimately stop the locked target
study. Paired-transport value depends on an externally justified transport budget;
its harmful negative controls are part of the evidence. Lifecycle and loopback
Docker measurements are engineering checks, not independent production sites.
Camera work, missing historical execution evidence, independent field validation,
and adaptive repeated-use statistical guarantees remain open wherever no new
source-bound evidence actually resolves them.
