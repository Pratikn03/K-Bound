# TMLR Anonymity Audit

- Audit time (UTC): `2026-09-04T08:51:34+00:00`
- PDF: `docs/research/kbound/release/current/kbound_tmlr.pdf`
- PDF SHA-256: `a33dbbb403eedeb4df532bec1c697a99b660e8ef7c7e1e0f5fff1c303951ea62`
- File size: `817941` bytes
- Overall result: **PASS**

This is an identity-leak audit, not a scientific-content or layout review. It scans visible text, hidden-inclusive text, raw reading order, metadata, raw PDF objects, destinations/bookmarks, annotations/actions, URLs, JavaScript, attachments, and (when available) a decoded pypdf object walk.

## Reproduction command

Run from the repository root:

```bash
python docs/research/kbound/scripts/audit_tmlr_anonymity.py --pdf docs/research/kbound/release/current/kbound_tmlr.pdf --report docs/research/kbound/paper/reports/TMLR_ANONYMITY_AUDIT.md
```

## Low-level commands and exact outcomes

| Command | Exit | Result | Output SHA-256 |
|---|---:|---|---|
| `pdfinfo -isodates docs/research/kbound/release/current/kbound_tmlr.pdf` | 0 | OK; 501 output characters | `d611cc47b34770f7776b3e59e23299d61fa6944b681d4c88535acac7e8accc54` |
| `pdfinfo -custom docs/research/kbound/release/current/kbound_tmlr.pdf` | 0 | OK; 189 output characters | `1ad551375cf6958f6f348fcf4be4f23f7f04d04ecf8a7d01113f778b48ca3e61` |
| `pdfinfo -meta docs/research/kbound/release/current/kbound_tmlr.pdf` | 0 | OK; 0 output characters | `01ba4719c80b6fe911b091a7c05124b64eeece964e09c058ef8f9805daca546b` |
| `pdfinfo -dests docs/research/kbound/release/current/kbound_tmlr.pdf` | 0 | OK; 277 named-destination rows | `d16d33c862bcaacd012fbdef3929e783992c00f50948b2bf78406a2573caef1a` |
| `pdfinfo -url docs/research/kbound/release/current/kbound_tmlr.pdf` | 0 | OK; 4 external URL rows | `c76e5996666810ac522912591516c230749d48a539099ab4896aef56f848073c` |
| `pdfinfo -js docs/research/kbound/release/current/kbound_tmlr.pdf` | 0 | OK; no JavaScript | `01ba4719c80b6fe911b091a7c05124b64eeece964e09c058ef8f9805daca546b` |
| `pdfdetach -list docs/research/kbound/release/current/kbound_tmlr.pdf` | 0 | OK; 0 embedded files | `ba4e6d00c8bf0cae7653e6cb54795848d7f0b6da9dc66afde46623013d86e5f1` |
| `pdftotext -layout docs/research/kbound/release/current/kbound_tmlr.pdf -` | 0 | OK; 172698 extracted characters | `c0666f5608dfb117d24d3d93eaa8b2edab7ec1ceac9eb2e8db1cdac238e41607` |
| `pdftotext -raw docs/research/kbound/release/current/kbound_tmlr.pdf -` | 0 | OK; 149679 extracted characters | `e33fe4e114e52025669de124578afe2cc984d8dff7fce144df21d8720bd682de` |
| `pdftohtml -xml -hidden -i -stdout docs/research/kbound/release/current/kbound_tmlr.pdf` | 0 | OK; 614868 extracted characters | `31921e4b32c7365b4b75bde5a2dec607cc2b25ecbe57d06f230f901a1082b29c` |
| `gs -q -dNOPAUSE -dBATCH -sDEVICE=nullpage docs/research/kbound/release/current/kbound_tmlr.pdf` | 0 | OK; Ghostscript parsed every page without diagnostics | `01ba4719c80b6fe911b091a7c05124b64eeece964e09c058ef8f9805daca546b` |

## Document properties

| Field | Exact value |
|---|---|
| Title | `K-Bound: When Is Label-Free Adaptation Knowable?` |
| Author | `<empty>` |
| Subject | `<empty>` |
| Keywords | `<empty>` |
| Creator | `LaTeX with hyperref` |
| Producer | `pdfTeX-1.40.27` |
| Pages | `43` |
| Encrypted | `no` |
| Form | `none` |
| JavaScript | `no` |

- Metadata stream output: empty (no XMP metadata stream).
- Embedded attachments: `0`.
- JavaScript: `absent`.

## URLs and actions

- `19  Annotation    https://proceedings.mlr.press/v267/chen25ch.html`
- `21  Annotation    https://proceedings.mlr.press/v9/david10a.html`
- `21  Annotation    https://doi.org/10.52202/079017-3820`
- `21  Annotation    https://doi.org/10.52202/079017-3820`

## Object-level inventory

- `pypdf` was not importable. The fail-closed Poppler, raw-object, metadata, URL, destination, hidden-text, JavaScript, and attachment checks still ran. Re-run with pypdf available for the additional decoded object graph inventory.

## Prohibited-identity findings

- None. No named author, institution, author email/domain, personal GitHub identity, forge repository URL, email address, absolute home-directory path, acknowledgment, or author-contribution statement was found on any inspected surface.

## Non-identifying review signals

These signals are reported for human review but do not fail the audit by themselves. Generic repository-relative artifact names and generic references to supplementary material do not reveal an author unless paired with an identity, public repository URL, username, or local path.

### Generic repository/source-code wording

- `...t observe the theorem’s target disagreement- conditional residual directly. The repository retains the reported result text but not the named raw beta-sweep artifact. The...`
- `...(24.2%). Neither census froze a reproducible family definition, file list, and repository state. They are overlapping historical search summaries, not two predeclared fa...`
- `...tatus: historical local-port audit; not a synchronized official comparison. The repository retains one historical head-to-head artifact on the CIFAR-10-C Tent stream. Its...`
- `...nt remains a different protocol. F.5 Multiplicity and Search History The repository contains many exploratory configurations. One superseded census reported 1,387...`

### Displayed source or artifact filenames

- `...riority. The source artifact is experiments/kbound/results/mixed_headtohead_v1/ HEADTOHEAD_RESULTS_cifar10c_tent_primary.json; its stored WIN and kga_beats labels are withdrawn from the current claim set....`
- `...Its b ∆i , εi , and action must remain unchanged. The maintained test tests/test_controlled_grid_crossfit.py checks this for the implementation and also exhibits the old radius leak as a n...`
- `...tuning. All primary outcomes and action/effect counts come from paper/generated/cct20_release_manifest.json. Its sidecar paper/generated/cct20_release_manifest.json.receipt.json binds the...`
- `...e from paper/generated/cct20_release_manifest.json. Its sidecar paper/generated/cct20_release_manifest.json.receipt.json binds the file-level digest; the generated macros instead record canonicalized...`

### Displayed repository-relative paths

- `...s not establish current- policy or official superiority. The source artifact is experiments/kbound/results/mixed_headtohead_v1/ HEADTOHEAD_RESULTS_cifar10c_tent_primary.json; its stored WIN and kga_beats lab...`
- `...fits. Its b ∆i , εi , and action must remain unchanged. The maintained test tests/test_controlled_grid_crossfit.py checks this for the implementation and also exhibits the old radius leak as a n...`
- `...t result-driven tuning. All primary outcomes and action/effect counts come from paper/generated/cct20_release_manifest.json. Its sidecar paper/generated/cct20_release_manifest.json.receipt.json binds the...`
- `...ffect counts come from paper/generated/cct20_release_manifest.json. Its sidecar paper/generated/cct20_release_manifest.json.receipt.json binds the file-level digest; the generated macros instead record canonicalized...`

### Supplemental-material references

- None.

## Failure reasons

- None.

## Conclusion

PASS: the audited PDF contains no detected author-identifying text, metadata, hidden object content, personal URLs, local paths, attachments, JavaScript, acknowledgments, or author-contribution statement. The PDF Author field is empty.
