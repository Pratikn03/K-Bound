#!/usr/bin/env bash
# One fail-closed entry point for Task 3 and the separate DomainNet track.
#
# Modes:
#   smoke             synthetic input checks only; no data or model is opened
#   preflight-task3   validate real official-baseline inputs; no launch
#   run-task3         preflight, then invoke the existing official runbooks
#   preflight-natural validate the real DomainNet audit/split/checkpoint; no launch
#   run-natural       preflight, then invoke an explicitly supplied runner
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"
K="$ROOT/docs/research/kbound"
TASK3_OUTPUT="${TASK3_OUTPUT:-$ROOT/experiments/kbound/results/task3_natural_preflight_20260910/task3}"
NATURAL_OUTPUT="${NATURAL_OUTPUT:-$ROOT/experiments/kbound/results/task3_natural_preflight_20260910/natural}"

select_python() {
  if [[ -n "${KBOUND_PYTHON:-}" ]]; then
    printf '%s\n' "$KBOUND_PYTHON"
  elif [[ -x "$ROOT/.venv/bin/python" ]]; then
    printf '%s\n' "$ROOT/.venv/bin/python"
  else
    command -v python3
  fi
}

PY="$(select_python)"
PREFLIGHT="$K/scripts/task3_natural_preflight.py"

require_executable() {
  local value="$1" label="$2"
  [[ -x "$value" ]] || { echo "FATAL: $label is not executable: $value" >&2; return 2; }
}

task3_args() {
  [[ -n "${IMAGENET_ROOT:-}" ]] || { echo "FATAL: set IMAGENET_ROOT to clean ImageNet root containing val/" >&2; return 2; }
  [[ -n "${IMAGENETC_ROOT:-}" ]] || { echo "FATAL: set IMAGENETC_ROOT to ImageNet-C root" >&2; return 2; }
  [[ -n "${TTALINE_SOURCE:-}" ]] || { echo "FATAL: set TTALINE_SOURCE to a pinned clean official TTA/ALine checkout" >&2; return 2; }
  [[ -n "${POEM_PYTHON:-}" ]] || { echo "FATAL: set POEM_PYTHON to the pinned official POEM environment" >&2; return 2; }
  [[ -n "${AETTA_PYTHON:-}" ]] || { echo "FATAL: set AETTA_PYTHON to the pinned official AETTA environment" >&2; return 2; }
  local poem_source="${POEM_SOURCE:-$ROOT/external/poem_official}"
  local aetta_source="${AETTA_SOURCE:-$ROOT/external/aetta_official}"
  TASK3_ARGS=( \
    --task3 --repo "$ROOT" \
    --imagenet-root "$IMAGENET_ROOT" \
    --imagenetc-root "$IMAGENETC_ROOT" \
    --poem-source "$poem_source" \
    --aetta-source "$aetta_source" \
    --ttaline-source "$TTALINE_SOURCE" \
    --python-executable "$PY" \
    --poem-python "$POEM_PYTHON" \
    --aetta-python "$AETTA_PYTHON" \
    --expected-classes "${EXPECTED_CLASSES:-1000}" \
    --json-out "$TASK3_OUTPUT.json" \
  )
  if [[ -n "${SAR_SOURCE:-}" ]]; then
    TASK3_ARGS+=( --poem-sar-source "$SAR_SOURCE" )
  fi
}

natural_args() {
  [[ -n "${DOMAINNET_ROOT:-}" ]] || { echo "FATAL: set DOMAINNET_ROOT to the DomainNet root" >&2; return 2; }
  [[ -n "${DOMAINNET_AUDIT:-}" ]] || { echo "FATAL: set DOMAINNET_AUDIT to the audited DomainNet JSON" >&2; return 2; }
  [[ -n "${NATURAL_SPLIT_MANIFEST:-}" ]] || { echo "FATAL: set NATURAL_SPLIT_MANIFEST to a locked split JSON" >&2; return 2; }
  [[ -n "${NATURAL_CHECKPOINT:-}" ]] || { echo "FATAL: set NATURAL_CHECKPOINT to the authenticated source checkpoint" >&2; return 2; }
  NATURAL_ARGS=( \
    --natural --repo "$ROOT" \
    --domainnet-root "$DOMAINNET_ROOT" \
    --audit-path "$DOMAINNET_AUDIT" \
    --split-manifest "$NATURAL_SPLIT_MANIFEST" \
    --checkpoint "$NATURAL_CHECKPOINT" \
    --json-out "$NATURAL_OUTPUT.json" \
  )
}

run_smoke() {
  "$PY" - "$ROOT" <<'PY'
import importlib.util
import json
import tempfile
from pathlib import Path

root = Path(__import__("sys").argv[1])
module_path = root / "docs/research/kbound/scripts/task3_natural_preflight.py"
spec = importlib.util.spec_from_file_location("task3_natural_preflight", module_path)
assert spec and spec.loader
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)

with tempfile.TemporaryDirectory(prefix="kbound_task3_natural_smoke_") as tmp:
    base = Path(tmp)
    clean = base / "imagenet" / "val" / "class_000"
    clean.mkdir(parents=True)
    corrupted = base / "imagenet-c"
    for corruption in module.OFFICIAL_CORRUPTIONS:
        (corrupted / corruption / "5" / "class_000").mkdir(parents=True)
    sources = []
    for name in ("poem", "aetta", "ttaline"):
        path = base / name
        path.mkdir()
        (path / ".git").write_text("gitdir: synthetic\n", encoding="utf-8")
        # The native POEM/AETTA gates also require a harmless --help entrypoint.
        (path / "main.py").write_text(
            "import argparse\nargparse.ArgumentParser().parse_args()\n",
            encoding="utf-8",
        )
        sources.append(path)
    task3 = module.check_task3_inputs(
        repo=root, imagenet_root=base / "imagenet", imagenetc_root=corrupted,
        poem_source=sources[0], aetta_source=sources[1], ttaline_source=sources[2],
        python_executable=__import__("sys").executable, expected_classes=1,
        require_clean_sources=False,
    )
    assert task3["status"] == "READY", task3
    audit = base / "audit.json"
    audit.write_text(json.dumps({"integrity_status": "PASS", "total_images": 1}), encoding="utf-8")
    (base / "domainnet").mkdir()
    manifest = base / "splits.json"
    manifest.write_text(json.dumps({"lock": {"locked": True, "sha256": "synthetic"}}), encoding="utf-8")
    checkpoint = base / "checkpoint.pt"
    checkpoint.write_bytes(b"synthetic")
    natural = module.check_natural_inputs(
        repo=root, domainnet_root=base / "domainnet", audit_path=audit,
        split_manifest=manifest, checkpoint=checkpoint,
    )
    assert natural["status"] == "READY", natural
print("SMOKE_STATUS=PASS")
PY
}

preflight_task3() {
  mkdir -p "$(dirname "$TASK3_OUTPUT.json")"
  if ! task3_args; then
    echo "TASK3_STATUS=OPEN" >&2
    return 2
  fi
  set +e
  "$PY" "$PREFLIGHT" "${TASK3_ARGS[@]}"
  local rc=$?
  set -e
  if (( rc != 0 )); then
    echo "TASK3_STATUS=OPEN" >&2
    return 2
  fi
  echo "TASK3_STATUS=READY"
}

preflight_natural() {
  mkdir -p "$(dirname "$NATURAL_OUTPUT.json")"
  if ! natural_args; then
    echo "NATURAL_STATUS=OPEN" >&2
    return 2
  fi
  set +e
  "$PY" "$PREFLIGHT" "${NATURAL_ARGS[@]}"
  local rc=$?
  set -e
  if (( rc != 0 )); then
    echo "NATURAL_STATUS=OPEN" >&2
    return 2
  fi
  echo "NATURAL_STATUS=READY"
}

run_task3() {
  preflight_task3
  : "${TASK3_COMMON_RUNNER:?set TASK3_COMMON_RUNNER to the reviewed runner that executes AETTA, POEM, and TTA/ALine on the same panel}"
  require_executable "$TASK3_COMMON_RUNNER" TASK3_COMMON_RUNNER
  mkdir -p "$TASK3_OUTPUT"
  exec "$TASK3_COMMON_RUNNER" \
    --repo "$ROOT" \
    --imagenet-root "$IMAGENET_ROOT" \
    --imagenetc-root "$IMAGENETC_ROOT" \
    --output "$TASK3_OUTPUT" \
    --seed "${TASK3_SEED:-0}" \
    --severity "${TASK3_SEVERITY:-5}" \
    --corruptions "${TASK3_CORRUPTIONS:-gaussian_noise,shot_noise,impulse_noise}"
}

run_natural() {
  preflight_natural
  : "${NATURAL_RUNNER:?set NATURAL_RUNNER to the reviewed DomainNet runner}"
  [[ -x "$NATURAL_RUNNER" ]] || { echo "FATAL: NATURAL_RUNNER is not executable: $NATURAL_RUNNER" >&2; return 2; }
  mkdir -p "$NATURAL_OUTPUT"
  exec "$NATURAL_RUNNER" --root "$DOMAINNET_ROOT" --manifest "$NATURAL_SPLIT_MANIFEST" --output "$NATURAL_OUTPUT"
}

case "${1:-}" in
  smoke) run_smoke ;;
  preflight-task3) preflight_task3 ;;
  run-task3) run_task3 ;;
  preflight-natural) preflight_natural ;;
  run-natural) run_natural ;;
  *)
    echo "usage: $0 {smoke|preflight-task3|run-task3|preflight-natural|run-natural}" >&2
    exit 2
    ;;
esac
