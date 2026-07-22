# Independent replication protocol

This is the frozen path for an external group to test K-Bound. The present paper claims internal reproducibility, not independent replication.

## Frozen confirmatory targets

1. Decisions use no target-test labels.
2. A false-adapt event is an `adapt` decision when the realized benefit satisfies `Delta <= 0`.
3. PACS is a completed three-seed null diagnostic.
4. ImageNet-R is a completed four-seed weak-evidence diagnostic; no candidate has a confidence interval supporting beats-both.
5. Protocol D33 is controlled mechanism confirmation, not a natural multimodal benchmark.
6. CIFAR-10-C SAR is excluded while its archived aggregate is unreconciled.

## Required procedure

- Start from a tagged release and run the release candidate command before evaluation.
- Use the official datasets, published checksums, sealed configurations, and declared runtime. Do not tune on target-test labels.
- Publish every seed, raw per-condition decisions, and integer event counts. Do not reconstruct counts from rounded rates.
- Report regret to oracle, adapt/freeze/abstain proportions, false-adapt intervals, empirical interval coverage, and comparisons with both fixed policies.
- Deposit code, logs, manifests, environment lock, and immutable artifacts in a public repository or DOI-backed archive independent of the authors.

## Acceptance rule

Replication is assessed claim by claim. A null or failed result is reproduced when the decision pattern and uncertainty support the same verdict; it need not become favorable. Any new favorable result obtained after changing a sealed protocol is exploratory and cannot retroactively validate a registered claim.
