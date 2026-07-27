# Assumption-contract revision — what changed, and what is still open

Supersedes the `SPLICE.md` draft circulated earlier in this work. **Two recommendations in
that draft were wrong and were not applied** — see "Corrections" below.

## New files

| file | what it is |
|---|---|
| `paper/sections/assumption_contract.tex` | three coverage statements as separate definitions; A1–A6; where the guarantee lives; fallback ladder |
| `paper/sections/deployment_gate.tex` | 7-step gate, diagnostic suite with a "cannot show" column, report schema |
| `paper/sections/scope_box.tex` | boxed scope statement (input twice; carries no `\label`) |
| `kga/assumptions.py` | gate, diagnostics, report emitter. 39 tests |
| `kga/assumption_report.py` | stable import path for the schema |
| `scripts/emit_assumption_reports.py` | emits the reports from `NUMBERS_PACK.json`; `--check` / `--validate` |
| `tests/test_assumptions.py` | 39 tests |
| `tests/test_assumption_reports_current.py` | drift guard + report invariants |
| `research_lock/assumption_reports/*.json` | 4 reports, not gitignored, released with the tree |
| `REVIEWER_RESPONSE_ASSUMPTIONS.md` | response letter, real numbers |

## Edited in place

- `kbound_short_body.tex` — 4 `\input`, 1 forward-pointer sentence in §Assumptions and Validity Scope
- `kbound.tex` — 3 `\input`
- `kga/cli.py` — `assumption-gate` subcommand
- `kga/__init__.py` — 14 new exports

## Gate outcomes

| track | gate | units | theoretical coverage claimed |
|---|---|---|---|
| ImageNet-C SAR | `diagnostic_only` | 27 | false |
| CIFAR-10-C SAR | `reject` | 5 | false |
| PACS | `diagnostic_only` | 12 | false |
| ImageNet-R | `diagnostic_only` | 40 | false |

## Corrections to the earlier draft

**1. Do not attach an interval to the 0.898 figure.** The draft recommended restating it
with a cluster bootstrap. That would be wrong: §`sec:fa-identity` already establishes it is
a deterministic function of $n$ under in-sample rank calibration, and records that Wilson
intervals were *removed* from it because they carry no frequentist meaning. Applying the
"report observed coverage with uncertainty" rule here would reintroduce the error the paper
already fixed. `assumption_contract.tex` now carries `rem:no-interval` stating the
exception explicitly.

**2. The response letter should concede less than the draft did.** The submitted paper
already stated the conditionality in several places. What was missing was enforcement, not
candour. The letter says so.

## Verified

- TMLR build (`kbound_tmlr.tex`): 0 errors, 0 undefined references, 0 float overflow, 90 pp
- No label collisions between the new sections and the existing tree
- Every cross-reference in the new files resolves in **both** the short-body build and the
  `kbound.tex` build (six labels the long manuscript lacks were reworded away)
- `tests/test_assumptions.py`: 39 passed
- Report invariants: no theoretical-coverage claim, fallback matches gate, every value has
  provenance with a method string and artifact paths
- `emit_assumption_reports.py --check`: reports current

## Open

- **`kbound_short.tex` and `kbound.tex` were not compiled.** The Linux VM behind the file
  bridge has no `IEEEtran.cls` or `algorithm2e`. The new sections deliberately declare no
  algorithm float precisely because the two drivers load incompatible algorithm packages,
  but the builds still need running on a machine with the full TeX tree.
- `tests/*` were verified in a Python 3.11 + numpy 2.4 environment on byte-identical
  sources; re-run under the repo venv.
- `REPRO_INVENTORY.json` is script-generated and was not hand-edited; regenerate it so the
  new files are inventoried.
- `docs/research/kbound/.tex_check/` holds two stray build logs from a failed first
  attempt. Gitignored, and the bridge cannot delete files — remove it yourself.
- The repo had concurrent edits during this work (`formal/`, `kbound.tex` at 06:37). Diff
  before committing. Nothing here was committed.
