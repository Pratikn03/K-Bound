# Pytest Collection Closure

Date: 2026-08-28

## Supported dependency profiles

Repository testing uses two explicit Python 3.12 profiles. Their guarantees
must not be mixed with the broad application environment:

1. `requirements-paper.txt` is the human-maintained paper/core input.
   `requirements-paper.lock.txt` is its universal, hash-pinned installation
   artifact. It intentionally excludes Torch, WILDS, the API, and OpenCV and
   therefore does not promise repository-wide collection.
2. `requirements-research.txt` is the portable research input. The
   `requirements-research-ci.lock.txt` artifact is the authoritative GitHub
   Actions profile for CPython 3.12 on Linux x86_64. It binds every transitive
   package and uses direct, hash-pinned CPU-only Torch and torchvision wheels.

The version in `.python-version` is the clean-clone interpreter contract.
`requirements.txt`, `requirements-optional.txt`, and `requirements.lock.txt`
serve the broader application/development surface; the broad lock is an
unhashed environment snapshot and is not a paper or research replication lock.

## Installation

Paper/core environment on any platform supported by the universal lock:

```bash
python3.12 -m venv .venv-paper
.venv-paper/bin/python -m pip --version
.venv-paper/bin/python -m pip install --only-binary=:all: \
  --require-hashes -r requirements-paper.lock.txt
```

Full Linux x86_64 CPU research environment, matching CI:

```bash
python3.12 -m venv .venv-research
.venv-research/bin/python -m pip --version
.venv-research/bin/python -m pip install --only-binary=:all: \
  --require-hashes -r requirements-research-ci.lock.txt
```

On another platform, install `requirements-research.txt` in a fresh Python
3.12 environment. That path keeps the direct package versions fixed but lets
the package index select the platform's native Torch wheel; it is intentionally
less strict than the Linux CI lock.

## Lock maintenance

The paper lock records its complete `uv pip compile` command in its header. The
Linux research lock is regenerated from the portable input plus the two direct
CPU-wheel overrides:

```bash
uv pip compile requirements-research.txt \
  --overrides requirements-research-ci-overrides.txt \
  --python-version 3.12 \
  --python-platform x86_64-unknown-linux-gnu \
  --generate-hashes \
  --output-file requirements-research-ci.lock.txt
```

Review and test lock diffs before committing them. Do not generate either lock
from `pip freeze`, and do not add an editable clone of this repository to any
lock file.

## Verification contract

The paper/core profile must collect and run the paper package and manuscript
research tests:

```bash
.venv-paper/bin/python -m pip check
.venv-paper/bin/python -m pytest --collect-only -q \
  docs/research/kbound/kbound_pkg/tests \
  docs/research/kbound/tests
.venv-paper/bin/python -m pytest -q \
  docs/research/kbound/kbound_pkg/tests \
  docs/research/kbound/tests
```

The full research profile must pass the CPU-wheel assertion, dependency
validation, unfiltered collection, and the complete suite:

```bash
.venv-research/bin/python - <<'PY'
import torch
import torchvision

assert torch.__version__ == "2.5.1+cpu"
assert torchvision.__version__ == "0.20.1+cpu"
assert torch.version.cuda is None
assert not torch.cuda.is_available()
PY
.venv-research/bin/python -m pip check
.venv-research/bin/python -m pytest --collect-only -q
.venv-research/bin/python -m pytest -q --tb=short
```

Keep local paths, local virtual-environment names, transient test counts, and
dirty-worktree failures out of this clean-clone contract. Record dated runtime
evidence in a separate audit artifact.
