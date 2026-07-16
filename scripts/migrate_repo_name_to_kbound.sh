#!/usr/bin/env bash
# One-shot migration: AutoML_Flagship_V8 -> K-Bound (folder name + in-file references).
# RUN ONLY AFTER the item-12 training queue is COMPLETE (script refuses otherwise).
#
# What it does:
#   1. Refuses to run if any training process is alive.
#   2. Renames ~/Documents/AutoML_Flagship_V8 -> ~/Documents/K-Bound.
#   3. Rewrites textual references in ACTIVE files (docs, scripts, runbooks, src, kga):
#        /Users/pratik_n/Documents/AutoML_Flagship_V8 -> /Users/pratik_n/Documents/K-Bound
#        $HOME/Documents/AutoML_Flagship_V8           -> $HOME/Documents/K-Bound
#        ~/Documents/AutoML_Flagship_V8               -> ~/Documents/K-Bound
#        bare "AutoML_Flagship_V8"                    -> "K-Bound"
#   4. Deliberately DOES NOT touch provenance: .git/, archive/, logs/,
#      experiments/kbound/results/, research_lock/, *.log, *.bak* — sealed artifacts
#      keep their historical paths (rewriting them would falsify run records).
#
# After running: re-select the folder in Cowork / re-open in your editor (path changed).
set -euo pipefail

OLD="$HOME/Documents/AutoML_Flagship_V8"
NEW="$HOME/Documents/K-Bound"

if pgrep -f "cifar_tent_mps_v2|run_item12|run_multiseed" >/dev/null 2>&1; then
  echo "ABORT: a training/queue process is still running. Wait for it to finish."; exit 1
fi
[[ -d "$OLD" ]] || { echo "ABORT: $OLD not found (already migrated?)"; exit 1; }
[[ -e "$NEW" ]] && { echo "ABORT: $NEW already exists."; exit 1; }

echo "==> Renaming folder"
mv "$OLD" "$NEW"
cd "$NEW"

echo "==> Rewriting references in active files (excluding provenance)"
grep -rl "AutoML_Flagship_V8" \
    --include="*.md" --include="*.sh" --include="*.py" --include="*.tex" \
    --include="*.yaml" --include="*.yml" --include="*.json" --include="*.toml" \
    docs kga scripts src tests README.md DATA.md Makefile 2>/dev/null \
  | grep -vE "\.bak|experiments/kbound/results/|research_lock/|archive/|logs/|\.log$" \
  | while IFS= read -r f; do
      sed -i '' \
        -e "s|/Users/pratik_n/Documents/AutoML_Flagship_V8|/Users/pratik_n/Documents/K-Bound|g" \
        -e "s|\$HOME/Documents/AutoML_Flagship_V8|\$HOME/Documents/K-Bound|g" \
        -e "s|~/Documents/AutoML_Flagship_V8|~/Documents/K-Bound|g" \
        -e "s|AutoML_Flagship_V8|K-Bound|g" \
        "$f"
      echo "  rewrote: $f"
    done

echo "==> Residual references in active files (expect 0):"
grep -rl "AutoML_Flagship_V8" docs kga scripts src README.md DATA.md 2>/dev/null \
  | grep -vE "\.bak|experiments/kbound/results/|research_lock/|archive/|logs/" | wc -l

echo "==> Git sanity"
git status --porcelain | head -5
echo "==> Done. Commit with:"
echo "    cd $NEW && git add -A && git commit -m 'repo: rename AutoML_Flagship_V8 -> K-Bound in folder name and active references'"
echo "    git push   # goes to origin (K-Bound), branch per your upstream setup"
echo "NOTE: provenance files (results manifests, research_lock, logs, archive) keep the"
echo "      historical name on purpose — they document what actually ran."
