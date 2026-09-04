# K-Bound publication release checklist

Scope: local publication freeze and external handoff. External publication actions require the
author's GitHub, Zenodo, venue, and optionally PyPI credentials. This checklist does not authorize
a push, tag, upload, or submission.

Revision verification: 2026-08-31. The theorem/novelty/bibliography and maintained-runtime
follow-ups are recorded in `audits/THEOREM_NOVELTY_BIBLIOGRAPHY_REVIEW.md`, Sections 9--11.
Successful component checks are not a clean-source release PASS. No historical seal
should be re-labeled as covering the current uncommitted source edits.

## 1. Freeze the scientific scope

- [x] Confirm `KBOUND_SHORT_RESULT_AUDIT.md` and `KBOUND_SHORT_CLAIM_MANIFEST.md` match the paper.
- [x] Keep CCT-20 as `SAFE_UTILITY_ONLY`: 44 FREEZE, zero ADAPT, one ABSTAIN; no strong routing win.
  Its bootstrap levels are nominal, and the result is not a population-safety guarantee.
- [x] Keep So2Sat as a negative development-gate stop: no feasible candidate, no gate calibration,
  no target access, and no target score.
- [x] Keep iWildCam withheld, POEM/AETTA as historical protocol-matched non-official ports, and the
  retrospective, non-confirmatory CIFAR Holm adjustment over six prospectively named contrasts
  explicit.
- [x] Select anonymous or named-author submission mode and apply the official venue style: the
  maintained TMLR driver is anonymous and uses the vendored official style.
- [x] State gamma as a calibration residual; retain feasible-margin, class-richness,
  oracle audit-floor, and repeated-deployment limits.
- [x] Correct the six bibliography records and distinguish the specific contribution from
  prior impossibility, monitoring, routing, and interval-risk-control methods.
- [x] Reassess previously uncited references: seven now support explicit statements;
  thirteen remain in `paper/references_kbound_context_archive.tex`, not the printed bibliography.
- [x] Extend Lean beyond the old 65-name finite core: 142 registered declarations
  pass the integrated kernel and transitive-axiom checks. Five measurable probability/
  construction layers are proved under explicit assumptions; the sixth historical
  one-bit/H/ratio-rate extension remains partial, with a verified counterexample
  to orbit-selection sufficiency. Do not claim six-layer closure.
- [x] Reject masked NumPy evidence and nonfinite e-value support at public boundaries;
  bind the numeric-validation helper explicitly in current-policy inference provenance.
- [x] Implement the subsequent narrative review: concise abstract, one research question,
  three scientific contributions, ten main sections, and a separate supplement after references.
- [x] Distinguish KGA's empirical cell-benefit gate from the population frontier; keep
  development/calibration label use and unavailable-ABSTAIN semantics explicit.
- [x] Add the four closest missing comparisons and preserve the adverse SAR result in the
  primary controlled table. Separate ordinary-accuracy and balanced-accuracy diagnostics.
- [x] Report current-policy interval inclusion, full width, commitment and directional errors
  separately. Mark the pooled leave-one-out diagnostic as rank-constrained and retrospective,
  not independent held-out coverage validation; retain Tent's one false FREEZE and its denominator.

## 2. Run the local release gate

Use Python 3.12.13 with the hash-complete `requirements-release-macos-arm64.lock.txt` installed
from a clean checkout. Verify both `python_environment_2026_09_02.json` and the path-free external
tool receipt `release_toolchain_2026_09_02.json` against
`release_toolchain_macos_arm64.json`. First commit every
maintained source/code edit and record that clean commit as the source freeze. Do not mix newly
regenerated publication artifacts into that source-freeze commit. The release driver never launches
training or modifies raw datasets. Its default `all` mode does not resolve or require local
ImageNet-R/PACS roots; `deep-local-preflight` and `all-deep-local` are the explicit modes that
perform those read-only dataset checks. Its default manuscript gate is portable: it validates the
receipt-bound CCT-20 manifest, internal ledgers, adjacent receipt, and repository-generated
artifacts without opening the manifest's author-machine absolute upstream paths. The default
`all` mode reuses and semantically verifies the committed strict-provenance public CCT-20 bundle;
it does not refresh that bundle or the CCT-20 release files from machine-local sources.
The external profile must cover all ten executed publication tools, including
Pandoc, `latexpand`, and the Perl interpreter used to run both `latexpand` and
`latexmk`. Each tool-using phase must import the verifier's exact realpaths;
prepending tool directories to `PATH` is not sufficient.

If pre-release generated outputs are already modified, preserve and review them
in a separately identified baseline-artifact commit before the final source freeze;
do not discard them with a reset or silently mix final regenerated outputs into
the source commit. A source freeze is not a certification of full Git integrity:
that check must be reported separately, including cloud-only object failures.

```bash
test -z "$(git status --porcelain --untracked-files=all)"
source_commit="$(git rev-parse HEAD)"
KBOUND_PYTHON=.venv/bin/python \
KBOUND_SOURCE_COMMIT="$source_commit" \
  bash docs/research/kbound/runbooks/release_candidate.sh all
git status --short
```

Review the exact generated-artifact allowlist, rerun the independent checksum and source-seal
verifiers, and then create a separate artifact-only commit. The seal's `source_commit` must remain
the clean source-freeze commit, normally the artifact commit's parent; do not reseal against the
artifact commit itself.

Optional author-machine provenance verification is a separate read-only mode. It is not required
for a clean clone or anonymous submission and should be run only while every sealed local upstream
is mounted:

```bash
KBOUND_PYTHON=.venv/bin/python \
  bash docs/research/kbound/runbooks/release_candidate.sh deep-local-preflight
KBOUND_PYTHON=.venv/bin/python \
  bash docs/research/kbound/runbooks/release_candidate.sh deep-local-provenance
```

Refreshing the CCT-20 release files is a separate, explicitly mutating local mode. To perform the
entire release with that refresh, deep provenance verification, and a rebuilt public bundle, use
`all-deep-local` instead of `all`. Both ZIP builders use `--replace-verified`: an existing archive
must pass its complete semantic verifier before an atomic replacement is permitted.

The standalone `checksums` mode is intentionally byte/environment-independent. It performs only a
canonical structural source-seal check before hashing the authoritative inventory; it does not
claim checkout/source semantics or reproduce the macOS build. The `all` and `all-deep-local` modes
retain the full checkout-bound source-seal and strict macOS environment checks before and after
checksum emission. Linux CI uses `verify-portable-release` to verify committed checksums, checkout
binding, and both archive formats; that mode is an artifact verification gate, not a release build
or dataset-validation claim. The Linux environment receipt is deliberately labeled as a
runtime-and-installed-version check: hash-locked installation authenticates downloaded wheels,
but CI does not claim the macOS installed-content profile. CI writes its regenerated repository
inventory to runner-temporary storage and requires its source commit to equal the exact checkout
SHA, leaving the sealed checkout unchanged. The dashboard gate starts with an empty temporary npm
cache and downloads only the exact integrity-bound TypeScript tarball in `package-lock.json`.
Portable archive verification likewise uses hash-verified Ubuntu Poppler packages, and Lean uses
the hash-verified Lean release archive while building exact Git revisions without executing the
remote Mathlib cache helper. Package gates disable build isolation, use the hash-locked setuptools
backend declared by `pyproject.toml`, and execute both `python -m kga` and the installed `kga`
console script from the built wheel.

Required outcomes:

- [ ] result reconciliation, claim validation, test suite, and the declared Lean core audit pass;
  the documented full-foundations failures must remain disclosed, not waived as proved;
- [x] establish a fully resident durable clone outside synchronized folders;
  `main`, `origin/main`, connectivity, and the no-missing-object walk agree at the
  recorded baseline. The primary, legacy, and So2Sat indexes contain no staged runtime-metadata
  delta to recover;
- [ ] complete the final post-freeze `git fsck --full`, commit-graph write/verify, ref audit, and
  independent checkout test. The earlier clean connectivity check is not this final proof;
- [x] confirm `edge/tests/test_protocol_inventory_reporting.py` is resident, tracked, and readable;
- [x] retain the Section 9 receipts for 847 selected revision/provenance/metadata
  tests and the earlier canonical/manuscript/authority checks as historical component
  evidence, not as a PASS for the later Section 10 source edits;
- [x] rerun 322 runtime/API cases and 42 source-binding cases in each of two
  independent environments; verify 98 release/cleanup guard cases, 220 installed-package
  cases, and the 142-name Lean kernel/axiom audit. These selected checks are not the
  complete release suite or a production/platform certification;
- [x] refresh the resident paper dashboard bindings and rerun strict canonical validation.
  The paper-only mode preserves the physical-edge payload exactly and explicitly marks it
  not rechecked; a separate full refresh must not substitute missing cloud inputs;
- [ ] complete the official-baseline audit and full physical-edge generation/verification;
- [x] rebuild the current compact and anonymous long PDFs and visually verify every page:
  Section 11 records 34 and 36 pages, respectively, with references before appendices;
- [x] rebuild and visually verify the current compact Word export: Section 11 records all
  31 rendered pages, including source-derived algorithm branches and corrected equation layout;
- [x] verify the Section 11 manuscript/canonical/interval checks and 868 distinct selected
  component cases, run in separate processes; these are not a complete release-suite PASS;
- [x] resolve the former mixed-import OpenMP blocker in the pinned release environment: an
  unmodified child imports scikit-learn 1.8.0 followed by Torch 2.5.1 successfully. The complete
  repository run, rather than this import smoke alone, remains the release authority;
- [ ] `KBOUND_RELEASE_SHA256SUMS.txt` is regenerated after every maintained artifact is final;
- [ ] a separate byte-for-byte checksum verification passes;
- [ ] the anonymous archive passes its full semantic/privacy verifier under the exact pinned
  Poppler paths immediately before its separate post-checksum receipt is generated;
- [ ] the source seal binds maintained checkout bytes to the recorded clean source-freeze commit;
- [ ] generated outputs are reviewed and committed separately from the source freeze;
- [ ] the working tree is clean after the final freeze commit.

The dated July [release manifest](archive/legacy_publication_surfaces_2026-09-02/retired_tree/docs/research/kbound/RELEASE_MANIFEST.json)
and [reproducibility report](archive/legacy_publication_surfaces_2026-09-02/retired_tree/docs/research/kbound/reports/reproducibility_release_report.md)
are archived historical snapshots. They do not satisfy this gate.

The earlier page counts and 847-case component count describe the Section 9 build;
the later scoped checks are identified separately above. Section 11 records the
fresh maintained exports and their working-tree identities. Those identities do
not replace a clean-source seal or final release-checksum verification.
On a cloud-backed checkout, an isolated temporary
`PYTHONPYCACHEPREFIX` can avoid reading stale cloud-backed bytecode without deleting
any repository files. It does not resolve unavailable source or Git objects.

## 3. Build the maintained package; distinguish historical reproduction

Build in an isolated temporary directory and exclude AppleDouble metadata. Do not delete or replace
the repository's source tree. The root `kbound-kga` wheel and `kga` CLI are the maintained
publication-primary implementation. The separate `kbound_pkg/kbound` package is a historical
reproduction snapshot, not a certified deployment API; its heuristic gate and optimizer violate
the current deployment contract. Its optional build below checks archival packaging only.

```bash
root_release_tmp="$(mktemp -d)"
COPYFILE_DISABLE=1 tar cf - --exclude='._*' --exclude='__pycache__' \
    pyproject.toml MANIFEST.in README.md LICENSE CITATION.cff kga | \
    ( cd "$root_release_tmp" && tar xf - )
( cd "$root_release_tmp" && COPYFILE_DISABLE=1 python3.12 -m build )
python3.12 -m twine check "$root_release_tmp"/dist/*
python3.12 -m venv "$root_release_tmp/.venv"
"$root_release_tmp/.venv/bin/python" -m pip install "$root_release_tmp"/dist/kbound_kga-*.whl
"$root_release_tmp/.venv/bin/kga" --help

paper_pkg_tmp="$(mktemp -d)"
( cd docs/research/kbound/kbound_pkg && COPYFILE_DISABLE=1 tar cf - \
    --exclude='._*' --exclude='__pycache__' \
    pyproject.toml README.md LICENSE kbound tests ) | ( cd "$paper_pkg_tmp" && tar xf - )
( cd "$paper_pkg_tmp" && COPYFILE_DISABLE=1 python3.12 -m build )
python3.12 -m twine check "$paper_pkg_tmp"/dist/*
python3.12 -m venv "$paper_pkg_tmp/.venv"
"$paper_pkg_tmp/.venv/bin/python" -m pip install "$paper_pkg_tmp"/dist/kbound-*.whl
"$paper_pkg_tmp/.venv/bin/python" -c \
    "import importlib.metadata, kbound; print(importlib.metadata.version('kbound'))"
```

- [x] Inspect the maintained wheel/sdist for caches, logs, data, absolute paths,
  high-confidence secret patterns, and `._*` files; verify archive membership, RECORD hashes,
  source-archive-to-wheel equivalence, isolated installation, imports, and CLI entry points.
- [ ] Inspect any optional historical distributions separately if they are to be shipped.
- [x] Confirm which package is publication-primary. The root `kbound-kga` package provides the
  `kga` CLI and excludes `docs*`; `docs/research/kbound/kbound_pkg` is reproduction-only.
- [x] Exclude the historical heuristic/optimizer from the maintained distribution and describe
  its limitations explicitly. Compatibility tests are not safety certification.
- [ ] Copy the validated distributions into the release staging area only after inspection.
- [ ] Use TestPyPI before any permanent PyPI upload.

## 4. Stage the role-distinct publication bundles

### Anonymous TMLR submission bundle

- `kbound_tmlr.pdf`
- anonymized `kbound_tmlr.tex` source, its live body/abstract and supplementary inputs,
  required figure assets, the vendored TMLR style/licence, and the bibliography;
- portable derived result indices or checksums only when the venue permits supplementary files.

Exclude the three named PDFs, the nonrelease combined PDF/DOCX, named drivers, `CITATION.cff`, root README,
repository URLs, author metadata, and raw CCT-20 provenance authorities. The sealed CCT manifest,
its receipt, and nested upstream identities intentionally retain canonical author-machine paths;
they are audit authorities, not anonymous-submission files.

### Public reproducibility release after deanonymization

- `kbound_short_main.pdf`, `kbound_short_supplement.pdf`, `kbound_tmlr.pdf`, and
  `kbound_full_report.pdf` from `release/current/`;
- the maintained source, figures, bibliography, code, lock files, licences, README, and
  `CITATION.cff`;
- `KBOUND_RELEASE_SHA256SUMS.txt`, portable derived indices, reviewer instructions, and the sealed
  audit/provenance authorities when their path disclosure is intentional and reviewed.

Do not attach the superseded `kbound.pdf`, `kbound_short.pdf`, or `kbound_submission.pdf`, or any
edited/companion variants, as current deliverables.
Their preserved pre-freeze bytes are indexed in the
[2026-09-02 stale-build archive](archive/stale_publication_builds_2026-09-02/README.md).

## 5. Review external metadata

- [ ] Repository description, README, release notes, and `CITATION.cff` use the same title, authors,
  version, licence, and claim scope.
- [ ] No DOI placeholder is presented as a real DOI.
- [ ] No local `/Users/...` or `/Volumes/...` path appears in the anonymous submission bundle or
  portable package metadata. Raw receipt-bound provenance authorities may retain canonical local
  paths only in the reviewed public reproducibility archive; never include them anonymously.
- [ ] No API token, credential, private review, target label, or restricted raw dataset is included.
- [ ] Dataset licences permit the published contents; distribute acquisition scripts rather than raw
  data when required.

## 6. Author-controlled publication actions

Only after the local release commit and checksums are final:

1. Push the reviewed branch or merge it through the repository's normal review process.
2. Create a new release tag that does not overwrite an existing tag.
3. Attach the maintained PDFs, source bundle, checksums, and validated package artifacts.
4. Create or update the Zenodo record and then insert the issued DOI into `CITATION.cff` and README.
5. Submit the correct anonymous or named bundle to the venue.
6. Optionally upload the exact validated distribution first to TestPyPI, then PyPI.

Record the release tag, commit SHA, DOI, artifact hashes, venue submission identifier, and UTC
timestamps in a final immutable release receipt.
