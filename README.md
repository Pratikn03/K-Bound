# K-Bound: When Is Label-Free Adaptation Knowable?

K-Bound studies a deployment decision that ordinary test-time adaptation leaves
implicit: should a proposed update be committed, rejected, or left undecided when
target labels are unavailable?

The framework separates two layers:

- **K-Bound theory** characterizes when a strict adapt or freeze commitment is
  uniformly supportable over a declared target class.
- **KGA** — **Knowability-Guided Adaptation** — is the practical finite-sample
  wrapper. It estimates adaptation benefit from label-free evidence and commits
  only when a calibrated interval excludes zero.

KGA wraps candidate adapters such as Tent, EATA, and SAR. It is a safety and
validity layer, not a new adaptation objective and not a universal accuracy
booster.

## Scientific Status (revised 2026-07-26)

Target venue: **TMLR**. Manuscript status: **NOT FROZEN**. Canonical, always-current version of
this table with numbers and open items:
[`docs/research/kbound/SUBMISSION_LEDGER.md`](docs/research/kbound/SUBMISSION_LEDGER.md).

| Evidence tier | Current result | Defensible reading |
|---|---|---|
| Core theory | Interior impossibility, closed-band abstention, strict-commitment frontier, marginal interval certificate | Conditional on the declared class and stated coverage assumptions. 8 of 13 compiled theorem-level results currently carry no proof in the compiled build. |
| Controlled mixed shifts — CI-supported | **CIFAR-10-C Tent and EATA** beats-both | Routing can improve on both fixed policies when helpful and harmful cells are detectable. This is the one track with real statistical power, and it survives every robustness check including the 2026-07-26 radius fix (0 of 9 504 decisions change). |
| Controlled mixed shifts — point estimate only | **ImageNet-C SAR** | Beats always-freeze on the point estimate (0.0289 vs 0.0319); the freeze-gap CI at the seed-averaged unit includes zero. **Demoted from beats-both 2026-07-26.** |
| Natural shifts | Office-Home, iWildCam, RxRx1 one-sided no-harm | KGA matches the safer fixed policy; no clean single-dataset natural CI-robust beats-both claim. RxRx1 (0 adapts) and iWildCam (1 adapt) leave the guarantee untested, and three of these four tracks have absent source records. |
| Natural shifts — provenance failure | **Camelyon17 OOD** | **Sealed but not recomputable from release.** The promoted triple lives in one sealed YAML and in no computable artifact; the promoted FA_u = 0 is recorded nowhere. Do not cite it as a reproduced number. |
| Weak / incomplete evidence | CIFAR-10.1 (declared negative), ImageNet-R, PACS | Diagnostic only; a null does not prove structural non-identifiability. ImageNet-R is worse than its mean row suggests — KGA loses to always-adapt on 7 of 10 backbones. |
| Physical camera study | Protocol and implementation ready; fresh S01-S10 sessions pending | No real-camera headline result until the machine-readable publication gate passes. The session checklists on disk are unreadable placeholders. |

The promoted benchmark values live in
[the canonical result manifest](docs/research/kbound/paper/generated/kbound_result_manifest.json).
The dashboard and paper tables are built from that manifest rather than from legacy notes.

**Three release caveats a reader should know before running anything:**

1. **143 committed text artifacts are NUL-filled iCloud placeholders**, including the entire
   Office-Home runner and every ablation JSON. Census, one-command recovery, release-guard spec:
   [`docs/research/kbound/PLACEHOLDER_INVENTORY.md`](docs/research/kbound/PLACEHOLDER_INVENTORY.md).
2. **Datasets**: [`DATA.md`](DATA.md) gives per-dataset version, split, licence and acquisition.
   Two of nine (ImageNet-R, Office-Home) are not obtainable from this release as shipped.
3. **The multi-seed runs span three Python/torch stacks**, and no manifest records a scikit-learn
   version. Their spread is not seed variance —
   [`docs/research/kbound/REPRODUCE.md §0a`](docs/research/kbound/REPRODUCE.md).

## Theory and Certificate

The adaptation benefit convention is Delta = R_T(f_0) - R_T(f_a). Positive
benefit favors adaptation; negative benefit favors the frozen model.

At population level, the theory uses the observable margin M, latent drift
gamma, and declared drift budget beta. A strict action is uniformly supportable
outside the band |M| <= beta under the paper's declared-class assumptions.

Real-data KGA uses a separate empirical rule:

- Delta_hat - epsilon > 0: adapt.
- Delta_hat + epsilon < 0: freeze.
- Otherwise: abstain from committing the update.

KGA does not numerically estimate M, gamma, or beta on the reported real
benchmarks, and epsilon is not an estimate of beta.

## Install

~~~bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
~~~

The lightweight certificate API is also packaged under
[docs/research/kbound/kbound_pkg](docs/research/kbound/kbound_pkg).

## Minimal Use

~~~python
import numpy as np

from kga.certificate import conformal_radius, decide

residuals = np.abs(np.random.default_rng(0).normal(size=200)) * 0.05
epsilon = conformal_radius(residuals, alpha=0.10)

print(decide(Bhat=0.12, eps=epsilon))   # adapt
print(decide(Bhat=-0.12, eps=epsilon))  # freeze
print(decide(Bhat=0.01, eps=epsilon))   # abstain
~~~

**The declared calibration rule, as of 2026-07-26, is one rule everywhere**: exact split-conformal
rank quantile with a **leave-one-out-of-pool** radius — cell *i*'s radius is calibrated on the
other n−1 residuals only. It replaces two earlier variants that both shipped: an interpolated
`np.quantile` rule, and an in-pool variant that included cell *i*'s own residual in cell *i*'s
radius. The in-pool variant made epsilon a function of the test labels that the FA_u guarantee
attaches to; correcting it changes **0 of 9 504** committed CIFAR-10-C decisions and moves
ImageNet-C SAR's FA_u from 0/135 to 1/135. Full accounting:
[`SUBMISSION_LEDGER.md §1a`](docs/research/kbound/SUBMISSION_LEDGER.md) and `§9`.

One structural caveat to state wherever FA_u is reported: under in-pool rank calibration
`FA_u ≤ (N−k)/N` is an arithmetic identity — exactly 0 at n ≤ 9 — so "FA_u ≤ α" is not a
measurement on the small-n tracks. Report FA_u against that ceiling, with the ADAPT count and a
Clopper-Pearson bound on FA_c.

Archived stress-grid artifacts are labeled separately because they used leave-one-condition-out
empirical residual calibration or an earlier empirical quantile implementation.

## Researcher Reproduction

Fast checks that do not require raw datasets:

~~~bash
python -m pytest tests docs/research/kbound/kbound_pkg/tests -q
python docs/research/kbound/scripts/build_dashboard_snapshot.py
bash docs/research/kbound/scripts/reproduce_submission.sh
~~~

Lean verification:

~~~bash
cd docs/research/kbound/formal
bash build.sh
~~~

Dataset refreshes and multiseed training are intentionally separate from the
cached artifact audit. See [DATA.md](DATA.md),
[REPRODUCE.md](docs/research/kbound/REPRODUCE.md), and
[research_lock](research_lock).

## Research Dashboard

Build and serve the local dashboard:

~~~bash
bash docs/research/kbound/scripts/build_dashboard.sh
python3 -m http.server 8765 --directory docs/research/kbound
~~~

Open http://127.0.0.1:8765/kbound_dashboard.html.

The dashboard exposes theorem scope, promoted result lineage, negative evidence,
camera readiness, session progress, and exact reproduction commands. Its browser
camera is a connectivity preview only; it does not create study evidence.

## Physical Camera Validation

The physical package/label study is preregistered as edge_real_phone_v1. Source,
calibration, held-out, and replication sessions are separated by session and
device.

~~~bash
python docs/research/kbound/edge/scripts/preflight_r2.py
bash docs/research/kbound/edge/scripts/run_edge_source_gate.sh
bash docs/research/kbound/edge/scripts/run_edge_publication_pipeline.sh
~~~

The final command fails closed unless all of the following hold:

1. S01-S10 contain exactly the expected physical clips.
2. Every clip has physical-capture provenance and a unique hash.
3. S02 balanced accuracy and macro-F1 are both at least 0.80.
4. Development and conformal splits were sealed before held-out access.
5. Phone A held-out and Phone B replication replays are complete.
6. All eight anti-leakage checks pass.
7. publication_gate.json reports passed: true.

Raw phone video remains local until privacy review. Release artifacts contain
manifests, hashes, policy logs, metrics, and table exports.

## Papers

| Artifact | Role |
|---|---|
| [kbound_short.tex](docs/research/kbound/kbound_short.tex) | Authoritative conference-style source |
| [kbound_short_final_draft.pdf](docs/research/kbound/kbound_short_final_draft.pdf) | Current compiled short draft |
| [kbound_short_final_draft.docx](docs/research/kbound/kbound_short_final_draft.docx) | Editable Word rendering synchronized from the final LaTeX source |
| [kbound.tex](docs/research/kbound/kbound.tex) | Historical extended manuscript; not authoritative for claims |
| [manuscript](docs/research/kbound/manuscript) | Long-form chapter organization |
| [formal](docs/research/kbound/formal) | Lean 4 mechanization inventory |
| [manuscript strategy](docs/research/kbound/KBOUND_MANUSCRIPT_STRATEGY.md) | What stays in the conference core versus supplement |

The short paper is the claim-controlled submission draft. The current long
manuscript predates several claim corrections and must not be submitted or used
as a numerical source. It is a historical source for extended proofs and
background only; material moves into the maintained supplement only after its
assumptions and artifact lineage are re-audited.

## Repository Map

| Path | Responsibility |
|---|---|
| [kga](kga) | Maintained certificate and routing core |
| [kbound_pkg](docs/research/kbound/kbound_pkg) | Installable research package |
| [generated paper data](docs/research/kbound/paper/generated) | Canonical result manifest and LaTeX macros |
| [dashboard](docs/research/kbound/dashboard) | TypeScript research dashboard |
| [edge](docs/research/kbound/edge) | Physical-camera protocol, runtime, tests, and reporting |
| [formal](docs/research/kbound/formal) | Lean development and theorem map |
| [experiments](experiments/kbound) | Experiment drivers and immutable result artifacts |
| [research locks](research_lock) | Protocol locks and decision records |
| [tests](tests) | Package, anti-leakage, and claim-consistency tests |

## Formalization Scope

Lean kernel-checks the indexed algebraic and finite-decision spine, including
certificate soundness conditional on coverage and supporting frontier lemmas.
The full target-law necessity construction, general exchangeable-process lift,
optional-stopping foundations, product KL/TV layer, risk alignment, and
calibration transfer remain external to the current mechanization. The paper
states this boundary explicitly.

## Current Submission Risks

- ImageNet-C SAR is a single-seed operating point.
- PACS and ImageNet-R planned seed panels are incomplete.
- The natural datasets provide no-harm evidence, not a clean natural beats-both win.
- Some controlled results depend on archived aggregates whose raw lineage must
  remain documented.
- The physical study still requires fresh held-out sessions.
- Full foundational probability mechanization is incomplete.

## License and Citation

Released code is under the [MIT License](LICENSE). Citation metadata is in
[CITATION.cff](CITATION.cff).
