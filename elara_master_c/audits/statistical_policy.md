# Statistical policy (Master Scenario C pointer)

**Authoritative copy:** `research_lock/statistical_policy_v1.md`

## Rules (summary)

- Primary endpoints frozen in `research_lock/primary_endpoints_v1.yaml`
- Threshold selection: validation only
- Multiplicity: Holm–Bonferroni within each confirmatory family
- Report effect size + bootstrap 95% CI
- No test-driven tuning on `final_unseen_audit` sets

## Confirmatory requirement

Transfer claim (P4) requires bootstrap CI excluding zero on **new untouched M2**, not Eyecandies.
