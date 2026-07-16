#!/usr/bin/env bash
# Shrink .git by purging historical bloat (committed mlruns + superseded duplicate
# PDFs) from ALL history, then force-push.
#
#   *** HISTORY REWRITE ***  This rewrites every commit SHA and force-pushes to
#   origin. Run ONLY when (a) the big data deletion has finished and (b) you are
#   authenticated to push to GitHub. Any other clones of this repo must be
#   re-cloned afterwards. A full mirror backup is made first so it is recoverable.
#
# Current canonical PDFs (kbound.pdf, kbound_short.pdf) are intentionally KEPT.
set -euo pipefail
REPO="/Volumes/T9/uav/AutoML_Flagship_V8"
cd "$REPO"

echo "[1/6] Mirror backup of current .git ..."
BK="${REPO}_gitmirror_backup_$(date +%Y%m%d_%H%M%S).git"
git clone --mirror "$REPO/.git" "$BK"
echo "      backup at: $BK   (delete it once the rewrite is verified good)"

echo "[2/6] Install git-filter-repo ..."
python3 -m pip install --user git-filter-repo

echo "[3/6] Analyze largest blobs (report only) ..."
python3 -m git_filter_repo --analyze || true
echo "      report: .git/filter-repo/analysis/blob-shas-and-paths.txt"

echo "[4/6] Purge historical bloat (current PDFs kept) ..."
python3 -m git_filter_repo --force \
  --path src/mlruns --invert-paths \
  --path-glob 'docs/research/kbound/K-Bound_paper*.pdf' --invert-paths \
  --path-glob 'docs/research/kbound/kbound_results-integrated*.pdf' --invert-paths \
  --path docs/research/kbound/kbound_submission.pdf --invert-paths

echo "[5/6] Re-add origin (filter-repo removes it) + repack ..."
git remote add origin https://github.com/Pratikn03/K-Bound.git 2>/dev/null || true
git reflog expire --expire=now --all
git gc --prune=now --aggressive

echo "[6/6] Force-push ALL branches and tags ..."
git push origin --force --all
git push origin --force --tags
echo "DONE. .git is now: $(du -sh .git | cut -f1). Collaborators must re-clone."
