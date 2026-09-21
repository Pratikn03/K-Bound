#!/usr/bin/env bash
# Historical execution source is byte-preserved in the retired-surface archive.
# This compatibility entrypoint never starts training or rebuilds an old draft.
printf '%s\n' 'ERROR: this empirical-training launcher is retired.' >&2
printf '%s\n' 'Use bash docs/research/kbound/runbooks/release_candidate.sh all for release verification.' >&2
printf '%s\n' 'New training requires its separately accepted study manifest; this command starts no job.' >&2
exit 2
