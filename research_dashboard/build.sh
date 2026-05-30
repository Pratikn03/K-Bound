#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUILD_DIR="${ROOT}/build"

mkdir -p "${BUILD_DIR}"

if command -v cmake >/dev/null 2>&1; then
  cd "${BUILD_DIR}"
  cmake .. -DCMAKE_BUILD_TYPE=Release
  cmake --build . --config Release -j"$(nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 2)"
else
  echo "cmake not found — compiling directly with clang++"
  clang++ -std=c++17 -Wall -Wextra -O2 \
    -I"${ROOT}/cpp/third_party" \
    -o "${BUILD_DIR}/elara_research_snapshot" \
    "${ROOT}/cpp/main.cpp"
fi

echo
echo "Built: ${BUILD_DIR}/elara_research_snapshot"
echo "Run from repo root:"
echo "  ${BUILD_DIR}/elara_research_snapshot --repo-root /path/to/AutoML_Flagship_V8"
