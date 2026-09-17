#!/usr/bin/env bash
# ONE-SHOT MIGRATION, ALREADY EXECUTED — RETAINED AS A RECORD, NOT AS A TOOL.
#
# This script renamed the checkout directory `AutoML_Flagship_V8` to `K-Bound` and
# rewrote the textual references to it in the active tree (docs, scripts, runbooks,
# src, kga). It ran once, on the author's machine, before this release. The repository
# you are reading is the result.
#
# WHY THE BODY IS GONE (defect D8, 2026-07-26).
# Every executable line of the original body was a literal machine-local absolute path —
# necessarily so, since renaming a directory requires naming it. Those strings are
# exactly what `docs/research/kbound/EXTERNAL_STORAGE_POLICY.md` bans from tracked code,
# and the D8 sweep replaced them with a portable repo-root variable. For every other
# script that substitution is correct; for this one it is destructive, because this
# script's *subject* is the literal paths. Several distinct originals collapsed onto the
# same portable token, so the pre-sweep body cannot be reconstructed from what is on
# disk, and this file will not invent one.
#
# WHAT IT DID, precisely, for the record (reconstructed from the surviving header and
# control flow, not from the paths):
#   1. Refused to run while any of `cifar_tent_mps_v2` / `run_item12` / `run_multiseed`
#      was alive, so the rename could not race a training queue.
#   2. `mv <parent>/AutoML_Flagship_V8 <parent>/K-Bound`, refusing if the source was
#      missing or the destination already existed.
#   3. In-place `sed` over `*.md *.sh *.py *.tex *.yaml *.yml *.json *.toml` under
#      `docs kga scripts src tests README.md DATA.md Makefile`, rewriting the old
#      absolute path (in its quoted, unquoted and `$HOME`-relative spellings) to the new
#      one, and the bare token `AutoML_Flagship_V8` to `K-Bound`.
#   4. Deliberately skipped `.git/`, `archive/`, `logs/`, `experiments/kbound/results/`,
#      `research_lock/`, `*.log` and `*.bak*`. Sealed artifacts keep their historical
#      paths on purpose: rewriting them would falsify the run records they document.
#   5. Printed the residual-reference count and `git status`, then the commit command.
#
# If this ever needs doing again, write it fresh against the paths you actually have.
# Do not resurrect a rename script from a comment block.

set -euo pipefail

cat >&2 <<'MSG'
scripts/migrate_repo_name_to_kbound.sh is a record of a completed one-shot migration.
It is intentionally not runnable: its body consisted entirely of machine-local absolute
paths, which are banned from tracked code (EXTERNAL_STORAGE_POLICY.md), and the rename
it performed has already happened. Read the header for exactly what it did.
MSG
exit 2
