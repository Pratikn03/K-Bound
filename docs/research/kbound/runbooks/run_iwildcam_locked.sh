#!/usr/bin/env bash
# RETIRED: despite its historical name, this workflow did not lock decisions on
# a disjoint validation cohort. It opened target-test labels for LOO calibration
# and treated stream seeds from one checkpoint as independent model seeds.
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
replacement="$script_dir/../scripts/run_multiseed.sh"

cat >&2 <<EOF
RETIRED WORKFLOW: run_iwildcam_locked.sh

This script intentionally performs no training, evaluation, extraction, waiting, or writes.
The former "locked" workflow was not a disjoint validation->test protocol and cannot
support model-level confidence intervals or a held-out beats-both claim.

Use the independent-checkpoint workflow instead:
  bash "$replacement" iwildcam

Its current output is explicitly development-only. Do not promote it as held-out
evidence until a validation-locked test scorer and unopened-target audit exist.
EOF
exit 64
