# Phase F — Release & DOI checklist (your accounts required)

I prepared everything below; the publishing steps must run under **your** GitHub /
Zenodo / PyPI identity, so you execute them. Each is copy-paste.

## 0. Land all the hardening work on `main`
```bash
cd /Volumes/T9/uav/AutoML_Flagship_V8
git checkout main
git merge --no-ff phase-cd-arch-git -m "Merge Phases B+C+D (deps, security, drift-guard, CI)"
# (phase-cd-arch-git already contains the Phase B commit)
git push origin main
```

## 1. Build the package wheel (verifies it installs)
```bash
python -m pip install build twine
# IMPORTANT (macOS): build from a CLEAN copy with COPYFILE_DISABLE=1. Otherwise the
# AppleDouble ._* sidecar files on the drive get swept into the sdist and break the
# build ("No distribution was found"). Verified fix:
rm -rf /tmp/kbpkg && mkdir -p /tmp/kbpkg
( cd docs/research/kbound/kbound_pkg && COPYFILE_DISABLE=1 tar cf - \
    --exclude='._*' --exclude='__pycache__' --exclude='dist' --exclude='build' --exclude='*.egg-info' . \
    | ( cd /tmp/kbpkg && tar xf - ) )
( cd /tmp/kbpkg && COPYFILE_DISABLE=1 python -m build )          # -> /tmp/kbpkg/dist/*.whl + *.tar.gz
mkdir -p docs/research/kbound/kbound_pkg/dist && cp /tmp/kbpkg/dist/* docs/research/kbound/kbound_pkg/dist/
python -m pip install docs/research/kbound/kbound_pkg/dist/kbound-*.whl   # smoke install
python -c "import kbound; print('import OK')"
```

## 2. Tag and create the GitHub Release
```bash
git tag -a v0.1.0 -m "K-Bound v0.1.0 — KGA certificate + theory + verified safety battery"
git push origin v0.1.0
# Option A (CLI, needs gh auth login):
gh release create v0.1.0 \
  docs/research/kbound/kbound.pdf docs/research/kbound/kbound_short.pdf \
  docs/research/kbound/kbound_pkg/dist/* \
  --title "K-Bound v0.1.0" \
  --notes-file docs/research/kbound/RELEASE_NOTES_v0.1.0.md
# Option B: GitHub web UI -> Releases -> Draft a new release -> pick tag v0.1.0,
#           paste RELEASE_NOTES, attach the two PDFs + the wheel.
```

## 3. Mint a Zenodo DOI (one-time setup, then automatic)
1. Sign in at https://zenodo.org with GitHub.
2. Zenodo → **GitHub** settings → find `Pratikn03/K-Bound` → flip the toggle **ON**.
3. Re-create the release in step 2 *after* the toggle is on (Zenodo only captures releases made while enabled) — or cut a `v0.1.1` release to trigger it.
4. Zenodo mints a DOI and a version-independent "concept" DOI. Copy the concept DOI.
5. Add it back here:
   - `CITATION.cff` → uncomment the `doi:` line with your DOI.
   - `README.md` → add the Zenodo badge:
     `[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.XXXXXXX.svg)](https://doi.org/10.5281/zenodo.XXXXXXX)`
   - Commit + push.

## 4. (Optional) Publish `kbound` to PyPI
Only if the name is free (check https://pypi.org/project/kbound/). Requires a PyPI
account + API token.
```bash
python -m twine upload docs/research/kbound/kbound_pkg/dist/*
```
> Note: PyPI uploads are permanent (you cannot re-use a version number). Do a
> TestPyPI dry run first: `twine upload --repository testpypi dist/*`.

## 5. Verify
- README shows the DOI badge and `pip install` instructions.
- `CITATION.cff` validates (GitHub shows a "Cite this repository" button).
- The GitHub Release lists the two PDFs + the wheel.
