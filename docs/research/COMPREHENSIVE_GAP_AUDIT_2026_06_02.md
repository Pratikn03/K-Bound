# Comprehensive Gap Audit - 2026-06-02

## Executive Verdict

Current scientific status is **bounded PhD-level / Level 2.5 evidence**, not
flagship, not SOTA, and not strict scientific production-readiness.

Current engineering status after the production hardening pass is: **core API
operational production path implemented and smoke-tested**, with unsupported
heavy routes failing closed. This does not change Gate E/F.

The repository is stronger than a toy project: it has real datasets, locked
negative results, paired inference, dashboard reporting, and API hardening
tests. The remaining gaps are not only Gate E. They fall into five groups:

1. **Scientific confirmation gaps** - strict transfer and natural-degradation
   gates do not pass.
2. **Evidence-scope gaps** - several datasets are opened, diagnostic, proxy, or
   development-only.
3. **Production/API gaps** - core API hardening is implemented; remaining
   production-certification work is real-artifact availability, signing beyond
   checksums, and SLA/operations evidence.
4. **Reproducibility and repo-hygiene gaps** - dependency and sidecar/cache
   hygiene are fixed in the active environment; the worktree is still large and
   must be split into logical release commits.
5. **Code-quality gaps** - `ruff F,E9` is clean; broader style/type gates are
   outside this pass.

## Verification Snapshot

Commands run during this audit:

```bash
jq '{checklist:.summary, items:[.items[]|select(.done==false)|{id,stage,description,evidence}]}' elara_master_c/audits/checklist_progress.json
jq '{gate_e_strict:.gate_e_m2_transfer_confirmed_strict, bounded_v3:.gate_e_m2_bounded_v3_pass, positive_transfer:{confirmed:.gate_e_positive_transfer_confirmed,status:.gate_e_positive_transfer_status,delta_sar:.gate_e_positive_transfer_delta_vs_sar,ci_sar:.gate_e_positive_transfer_ci95_vs_sar,delta_cw:.gate_e_positive_transfer_delta_vs_cw,ci_cw:.gate_e_positive_transfer_ci95_vs_cw}, natural_degradation:{confirmed:.gate_s_natural_degradation_confirmed,status:.gate_s_natural_degradation_status,stress_delta_cw:.gate_s_stress_delta_vs_cw,stress_ci_cw:.gate_s_stress_ci95_vs_cw,clean_delta_cw:.gate_s_clean_delta_vs_cw,clean_ci_cw:.gate_s_clean_ci95_vs_cw,dataset_status:.gate_s_current_dataset_status}, gate_f:.gate_f_scenario_c_scientific}' elara_master_c/audits/confirmatory_statistics_report.json
PYTHONPATH=src .venv/bin/python -m pytest tests/test_api_production_security.py tests/test_production_release_hygiene.py -q
PYTHONPATH=src .venv/bin/python -m pytest tests/test_confirmatory_validity_guard.py tests/test_scenario_c_checklist.py tests/test_v3_patchcore_and_gated_cw.py tests/test_v3_scripts_smoke.py tests/test_positive_transfer_candidate.py tests/test_positive_transfer_confirmatory.py tests/test_realiad_d3_headroom_audit.py tests/test_data_loader_fallback_policy.py -q
PYTHONPATH=src .venv/bin/python src/scripts/validate_manuscript_claims.py
PYTHONPATH=src .venv/bin/python src/scripts/validate_theorem_stack.py
.venv/bin/python -m pip check
PYTHONPATH=src .venv/bin/python -m ruff check deploy src tests --select F,E9
UAIS_API_KEYS=ci-secret UAIS_CORS_ORIGINS=https://ci.example docker compose config
docker build -t uais-api-prod-smoke .
docker run smoke on localhost:18000 for /health and /ready
```

Results:

- API production/release tests: **19 passed**.
- Research-integrity targeted tests: **36 passed**.
- Manuscript claim validator: **clean**.
- Theorem stack validator: **all_ok=true**.
- `pip check`: **No broken requirements found**.
- `ruff --select F,E9`: **All checks passed**.
- Dashboard C++ snapshot binary: **rebuilt** and snapshot regenerated.
- Docker compose config: **valid with required production env**.
- Docker API image: **built**.
- Docker API smoke: **/health=200, /ready unauth=403, /ready auth=503 until
  model artifacts are mounted, request ID present**.
- AppleDouble sidecars after cleanup: **0**.
- Key JSON scan: **0 NaN/Inf** in the main generated reports.

## P0 Scientific Blockers

### P0-S1: Strict Gate E still does not pass

Status: **not confirmed**.

Current generated report:

- `gate_e_m2_transfer_confirmed_strict=false`
- `gate_f_scenario_c_scientific=false`
- bounded v3 evidence is separate and true, but it does not satisfy strict
  flagship transfer.

Reason:

- Clean transfer versus SAR is not reliably positive.
- T9 states clean Gate E is effectively closed/unpassable on opened clean
  external benchmarks because the confidence-weighted mean is already near the
  oracle ceiling.

Effect:

- The project cannot honestly claim universal clean-transfer superiority.
- Any paper framed as "strict flagship Gate E passed" is rejectable.

### P0-S2: D13 natural positive-transfer track failed one co-endpoint

Status: **official fail**.

Current values:

- Candidate vs CW: delta = **+0.0191**, 95% CI **[+0.0148, +0.0235]**.
- Candidate vs SAR: delta = **-0.0858**, 95% CI **[-0.0928, -0.0787]**.

Reason:

- The method improves over confidence-weighted mean but loses to SAR.
- D13 requires **both** SAR and CW to pass.

Effect:

- `gate_e_positive_transfer_confirmed=false`.
- Real-IAD is now opened evidence; it can guide development but cannot be
  reused as a fresh official pass after tuning.

### P0-S3: D16 Real-IAD D3 natural-degradation track failed primary stress CI

Status: **official fail**.

Current values:

- Stress subset vs CW: delta = **+0.0351**, 95% CI **[-0.0276, +0.0980]**.
- Clean vs CW: delta = **0.0000**, 95% CI **[0.0000, 0.0000]**.
- All-test vs CW: positive and supportive, but not the primary endpoint.
- Dataset status: `OPENED_AFTER_D16_OFFICIAL_ATTEMPT`.

Reason:

- The point estimate is positive, but the primary 95% CI crosses zero.
- Clean no-regression is safe, but the gate required positive stress evidence.

Effect:

- `gate_s_natural_degradation_confirmed=false`.
- Real-IAD D3 is valuable real, non-synthetic evidence, but it does not pass
  the natural-degradation gate.

## P1 Paper / Reviewer-Risk Gaps

### P1-R1: Current strongest claim is bounded, not SOTA

Safe claim:

> Reliability gating helps in stress/degradation regimes and has bounded
> real-dataset support; clean-transfer superiority is not confirmed.

Unsafe claim:

> ELARA is universal, SOTA, production-ready, or flagship-ready.

Why reviewers will object:

- Gate E strict is false.
- D13 and D16 are official fails.
- Several transfer attempts are opened/development-only.
- Some datasets are diagnostic/proxy rather than independent multimodal
  transfer evidence.

### P1-R2: Opened datasets cannot become fresh proof after retuning

Opened/spent for at least one track:

- 3D-ADAM
- Real-IAD
- Real-IAD D3
- MulSen-AD
- Eyecandies

These are still useful for:

- development,
- ablations,
- negative evidence,
- operating-boundary analysis,
- reviewer transparency.

They are not valid for:

- a new official pass after method/threshold selection on their outcomes.

### P1-R3: Synthetic/proxy evidence must remain secondary

Current policy is now correct:

- `primary_evidence_basis=real_dataset_evidence`
- `synthetic_primary_evidence_allowed=false`

Remaining risk:

- Reviewers may still object if Eyecandies, VisA-derived views, LOCO-derived
  views, or ELARA-Bench-LA are used too prominently.

Safe use:

- mechanism diagnostics,
- smoke tests,
- failed/negative evidence,
- supporting intuition only.

## P1 Production / API Gaps

### P1-P1: Active Python environment dependency conflicts are fixed

Status: **fixed for the active/API-aligned runtime**.

Changes:

- `setuptools` is pinned below Torch's `<82` requirement.
- The legacy `eyecandies` CLI and its `pipelime-python` dependency were removed
  from the main runtime because `pipelime-python` requires `pydantic<2` and
  conflicts with FastAPI/Pydantic 2.
- `requirements-eyecandies-legacy.txt` documents that the old Eyecandies CLI
  belongs in a separate legacy acquisition environment only.

Current verification:

- `.venv/bin/python -m pip check` -> **No broken requirements found**.

### P1-P2: Optional model routes now fail closed

Status: **fixed for production v1 core-only route scope**.

Changes:

- Stale `uais_v` lazy imports were removed.
- Vision uses the real `uais.vision.vision_resnet` module path when a trusted
  artifact exists.
- NLP/vision/attention remain optional heavy routes and return structured
  `model_unavailable` 503 responses until their dependencies, artifacts, and
  checksums are productionized.
- `/ready` reports unavailable required core models separately from liveness.

### P1-P3: Production API is hardened, but not production-certified

Fixed compared with the older audit:

- API key auth fails closed.
- CORS uses explicit origins.
- `/metrics`, `/system`, and detailed health require auth.
- input sizes are bounded.
- joblib/torch loading requires trust/checksum controls.
- Dockerfile uses non-root user, pinned base, app command, and healthcheck.
- `/ready` is authenticated and separates readiness from liveness.
- optional NLP/vision/attention routes fail closed with `model_unavailable`
  unless production artifacts and dependencies are present.
- stale `uais_v` lazy imports were removed.
- request logging now emits request IDs, route/status/duration, and no payload
  or credential values.
- default compose production startup is API-only; Streamlit and MLflow are
  research-profile services.
- `docs/production/PRODUCTION_RUNBOOK.md` documents environment, checksum,
  deployment, readiness, monitoring, rollback, incident response, and claim
  boundaries.
- CI includes API security regression tests and a Docker API smoke path.

Still missing for production-ready claim:

- secrets/rotation story;
- model artifact signing policy beyond env checksum;
- fuller structured audit logs beyond request-level correlation logs;
- end-to-end model availability tests;
- SLA/error-budget monitoring.

## P2 Repo Hygiene / Reproducibility Gaps

### P2-H1: AppleDouble sidecars were removed

Current count after cleanup:

- **0** `._*` AppleDouble sidecar files outside `.git`.

Effect:

- audit tools can catalog fake Python modules such as `scripts.._...`;
- generated file catalogs become noisy;
- reviewers/CI may see irrelevant artifacts.

Release rule:

- Keep `.gitignore` rule `._*`.
- Re-run `find . -name '._*' -not -path './.git/*' -delete` immediately
  before Docker packaging on the T9/macOS volume, because filesystem access can
  regenerate sidecars.

### P2-H2: Generated caches and bytecode were cleaned

Observed:

- `deploy`, `src`, and `tests` project-local `__pycache__` directories were
  removed after verification.
- `.pytest_cache` and `.ruff_cache` were removed.
- `.tex_build*`, `.venv`, `tmp/`, and generated research outputs remain ignored
  or release-scope artifacts, not production API image inputs.

Effect:

- Not necessarily tracked, but they make audits noisy.
- The API package should not be distributed with local bytecode.

Release rule:

- Clean ignored caches before release.
- Build release artifacts from a clean checkout.

### P2-H3: Dirty working tree is large

The current worktree has many modified/generated/untracked files, including:

- research docs and generated PDFs,
- dashboard snapshot,
- D15/D16 scripts and tests,
- generated split hashes,
- D16 tables,
- synthetic audit doc,
- dataset matrix.

Effect:

- Hard to distinguish intentional evidence updates from transient artifacts.

Fix path:

- Stage/commit in logical groups:
  1. research integrity + D13/D16 semantics,
  2. Real-IAD D3 natural-degradation code,
  3. dataset/synthetic audit,
  4. dashboard/report regeneration,
  5. cleanup-only changes.

## P2 Code-Quality Gaps

Status: **fixed for the focused production syntax/undefined-name gate**.

Current verification:

- `PYTHONPATH=src .venv/bin/python -m ruff check deploy src tests --select F,E9`
  -> **All checks passed**.

Remaining scope:

- This does not claim full style/type perfection. It clears the production
  syntax/undefined-name/unused-symbol blocker identified in this audit.

## P2 Statistical Artifact Gaps

JSON scan found:

- **0 NaN/Inf** in main generated reports.
- Negative metric-like values are present and expected.
- Exact-zero values are present and mostly expected.

Important negative values:

- D13 vs SAR is fully negative.
- D16 stress CI lower bound is negative.
- D16 all-vs-SAR and clean-vs-SAR are negative.
- M2 external legacy transfer remains negative.

Important zero values:

- D16 clean vs CW is exactly zero because the rule defaults to CW/clean no-change.
- M3 healthcare has degenerate zero-width seed CI and is marked invalid.

Conclusion:

- These are not broken JSON values.
- They are honest negative/degenerate outcomes and must stay visible.

## What Is Actually Strong

The system does have defensible value:

- real-dataset-first evidence policy is now explicit;
- MVTec 3D-AD / 3D-ADAM / Real-IAD / Real-IAD D3 / MulSen are integrated;
- bounded v3 stress evidence passes;
- Real-IAD D3 gives real natural-degradation evidence, even though not a gate pass;
- manuscript claim validator is clean;
- theorem stack validator passes;
- API security and production-release regression tests pass;
- Docker API image builds and smoke-tests successfully.

## Final Blocker List

### Blocks flagship/SOTA paper

1. Strict Gate E false.
2. D13 natural positive-transfer official fail.
3. D16 natural-degradation official fail.
4. No fresh/prelocked unopened final transfer holdout remaining for current method.
5. One-class/leaderboard-comparable SOTA evaluation still absent.

### Blocks production-ready claim

1. Model artifact signing is checksum-only, not a full signing/attestation
   workflow.
2. No end-to-end model availability test with real mounted production artifacts.
3. No SLA/error-budget monitoring evidence.
4. Secrets rotation is documented operationally but not automated.

### Blocks clean repo/release

1. Large dirty diff with generated artifacts mixed with source changes.
2. Release must be cut from a clean checkout after sidecar/cache purge.

## Recommended Next Moves

1. **Paper path:** reframe the paper as bounded real-dataset stress/degradation
   reliability-gating research. Do not chase strict Gate E on opened data.
2. **Evidence path:** use Real-IAD D3 and MulSen for method development only;
   acquire or define a prelocked unopened split/category for any future official
   transfer claim.
3. **Production path:** add real-artifact availability tests and a signing or
   attestation workflow before external deployment.
4. **Repo path:** split the current dirty worktree into logical commits and cut
   releases from a clean checkout.
5. **Quality path:** optionally broaden beyond `F,E9` to full ruff/style/type
   gates after the production release boundary is stable.
