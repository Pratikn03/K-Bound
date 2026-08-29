# K-Bound publication release checklist

Scope: local publication freeze and external handoff. External publication actions require the
author's GitHub, Zenodo, venue, and optionally PyPI credentials. This checklist does not authorize
a push, tag, upload, or submission.

## 1. Freeze the scientific scope

- [x] Confirm `KBOUND_SHORT_RESULT_AUDIT.md` and `KBOUND_SHORT_CLAIM_MANIFEST.md` match the paper.
- [x] Keep CCT-20 as `SAFE_UTILITY_ONLY`: 44 FREEZE, zero ADAPT, one ABSTAIN; no strong routing win.
- [x] Keep So2Sat as a negative development-gate stop: no feasible candidate, no gate calibration,
  no target access, and no target score.
- [x] Keep iWildCam withheld, POEM/AETTA as historical protocol-matched non-official ports, and the
  retrospective, non-confirmatory CIFAR Holm adjustment over six prospectively named contrasts
  explicit.
- [x] Select anonymous or named-author submission mode and apply the official venue style: the
  maintained TMLR driver is anonymous and uses the vendored official style.

## 2. Run the local release gate

Use Python 3.12 with `requirements-research.txt` installed from a clean checkout. First commit every
maintained source/code edit and record that clean commit as the source freeze. Do not mix newly
regenerated publication artifacts into that source-freeze commit. The release driver never launches
training or modifies raw datasets. Its default manuscript gate is portable: it validates the
receipt-bound CCT-20 manifest, internal ledgers, adjacent receipt, and repository-generated
artifacts without opening the manifest's author-machine absolute upstream paths.

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
  bash docs/research/kbound/runbooks/release_candidate.sh deep-local-provenance
```

Required outcomes:

- [ ] result reconciliation, claim validation, test suite, and Lean audit pass;
- [ ] compact and long PDFs build and every page renders;
- [ ] the required compact Word export is rebuilt and passes its document checks;
- [ ] `KBOUND_RELEASE_SHA256SUMS.txt` is regenerated after every maintained artifact is final;
- [ ] a separate byte-for-byte checksum verification passes;
- [ ] the source seal binds maintained checkout bytes to the recorded clean source-freeze commit;
- [ ] generated outputs are reviewed and committed separately from the source freeze;
- [ ] the working tree is clean after the final freeze commit.

The dated July `RELEASE_MANIFEST.json` and `reports/reproducibility_release_report.md` are historical
snapshots. They do not satisfy this gate.

## 3. Build and inspect both Python package surfaces

Build in an isolated temporary directory and exclude AppleDouble metadata. Do not delete or replace
the repository's source tree.

```bash
root_release_tmp="$(mktemp -d)"
COPYFILE_DISABLE=1 tar cf - --exclude='._*' --exclude='__pycache__' \
    pyproject.toml README.md LICENSE kga | ( cd "$root_release_tmp" && tar xf - )
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

- [ ] Inspect both wheel/sdist pairs for caches, logs, data, absolute paths, secrets, and `._*` files.
- [ ] Confirm which package is publication-primary. The root `kbound-kga` package provides the
  `kga` CLI; `docs/research/kbound/kbound_pkg` is the paper's separate `kbound` API surface.
- [ ] Copy the validated distributions into the release staging area only after inspection.
- [ ] Use TestPyPI before any permanent PyPI upload.

## 4. Stage two distinct publication bundles

### Anonymous TMLR submission bundle

- `kbound_tmlr.pdf`
- anonymized `kbound_tmlr.tex` source, its live body/abstract inputs, required figure assets, the
  vendored TMLR style/licence, and the bibliography;
- portable derived result indices or checksums only when the venue permits supplementary files.

Exclude the named compact PDF, compact DOCX, named compact driver, `CITATION.cff`, root README,
repository URLs, author metadata, and raw CCT-20 provenance authorities. The sealed CCT manifest,
its receipt, and nested upstream identities intentionally retain canonical author-machine paths;
they are audit authorities, not anonymous-submission files.

### Public reproducibility release after deanonymization

- `kbound_short_final_draft.pdf`, `kbound_tmlr.pdf`, and `kbound_short_final_draft.docx`;
- the maintained source, figures, bibliography, code, lock files, licences, README, and
  `CITATION.cff`;
- `KBOUND_RELEASE_SHA256SUMS.txt`, portable derived indices, reviewer instructions, and the sealed
  audit/provenance authorities when their path disclosure is intentional and reviewed.

Do not attach the historical compatibility PDFs (`kbound.pdf`, `kbound_short.pdf`, edited/companion
variants) as current deliverables.

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
