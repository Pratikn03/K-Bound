#!/usr/bin/env bash
# RETIRED: the historical workflow reused one checkpoint across stream seeds and
# fitted LOO benefit certificates with target-test labels. That is not a valid
# model-seed replication or a disjoint validation->test evaluation.
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
replacement="$script_dir/../scripts/run_multiseed.sh"

cat >&2 <<EOF
RETIRED WORKFLOW: run_iwildcam_episodic.sh

This script intentionally performs no training, evaluation, extraction, or file writes.
The former workflow used target-test labels inside LOO calibration and treated stream
seeds from one checkpoint as independent models. Its aggregates are scientifically invalid.

Use the independent-checkpoint workflow instead:
  bash "$replacement" iwildcam

That workflow produces a lineage-verified development diagnostic only. A held-out
beats-both claim remains unavailable until a disjoint validation-locked test scorer
is implemented and the test target is unopened at lock time.
EOF
exit 64
