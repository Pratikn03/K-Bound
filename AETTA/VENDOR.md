# Vendored AETTA baseline

This tree is the **CVPR 2024 AETTA** code used as an external comparator for
K-Bound mixed head-to-head experiments. It is **not** part of the K-Bound /
KGA package.

Official upstream: <https://github.com/taeckyung/AETTA>. The conversion from
the former nested checkout did not retain an independently verifiable upstream
commit identifier. Therefore this vendored snapshot must remain labelled
``protocol-matched port`` until it is replaced from a pinned clean upstream
commit and the native run/conversion audit passes. The monorepo commit and a
content hash identify this snapshot, but neither substitutes for an upstream
commit.

On 2026-07-15 the broken nested git submodule metadata was removed and the
tree was converted to ordinary vendored files in this monorepo. Update by
manually replacing files from upstream if needed; do not re-introduce a nested
`.git` directory.
