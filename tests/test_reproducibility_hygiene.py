"""Reproducibility-hygiene guards (fix-queue item 30, panel findings F2-8 / F3-17).

Three properties, each of which was violated somewhere in the tree when the
review panel ran:

1. **Per-cell RNG seeding must be process-stable.**  ``cifar10c_suite.py``
   seeded each corruption/severity cell with ``sev * 131 + hash(corr) % 9973``.
   CPython salts ``hash()`` on ``str`` per interpreter process unless
   ``PYTHONHASHSEED`` is set, and nothing in this repo sets it, so three
   interpreters drew three different 800-image subsamples for the same cell.
   The suite now uses a blake2b digest; :class:`TestStableSeedIsProcessStable`
   pins that, and deliberately proves the test is not vacuous by showing that
   the builtin ``hash()`` really does differ across the same two processes.

2. **No tracked executable may contain a machine-local path.**
   ``EXTERNAL_STORAGE_POLICY.md:18`` bans them.  The guard used to be scoped to
   ``kga/`` and ``tests/`` only -- "the two trees this file can speak for" --
   which is why it passed green while **94 tracked ``.py``/``.sh`` files** still
   carried ``AutoML_Flagship_V8`` / ``/Volumes/T9`` / ``/Users/pratik`` paths.
   That was defect D8.  :class:`TestNoMachineLocalPaths` now scans the **whole
   repository**, both ``.py`` and ``.sh``, and also catches the class the
   original guard never looked for: Cowork **session-sandbox** mounts
   (``/sessions/<name>/mnt/...``), which are valid only inside one ephemeral
   container.  The survivors are named one by one in
   :data:`MACHINE_LOCAL_ALLOWLIST`, each with the reason it is allowed.

3. **Tests must not leak process-global state.**  :class:`TestNoRawEnvironMutation`
   forbids raw ``os.environ`` assignment in the suite: a ``set ... assert ...
   pop`` sequence leaves the variable set whenever the assertion in the middle
   fails, which makes the outcome of every later test depend on collection
   order.  ``monkeypatch.setenv`` restores it in teardown unconditionally.

Everything here is static analysis or a subprocess; no torch, no network, no
artifacts.
"""

from __future__ import annotations

import ast
import os
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
TESTS = REPO / "tests"
KGA = REPO / "kga"
CIFAR10C_SUITE = REPO / "src" / "scripts" / "kbound" / "cifar10c_suite.py"

#: Machine-local roots that must never appear in tracked source.
BANNED_PATH_FRAGMENTS = ("AutoML_Flagship_V8", "/Volumes/T9", "/Users/pratik", "/sessions/")

#: The complete set of files still permitted to contain a banned fragment, with
#: the reason.  Every one of them either *detects* the pattern or *documents*
#: it; none of them depends on such a path to run.  Shrink this list; do not
#: grow it.  The portable replacements are ``$KBOUND_REPO_ROOT`` (discovered from
#: the file's own location) and ``$KBOUND_EXTERNAL_ROOT`` (one documented
#: variable for the git-excluded volume, which raises when unset --
#: ``docs/research/kbound/kbound_repro/paths.py``).
MACHINE_LOCAL_ALLOWLIST: dict[str, str] = {
    "tests/test_reproducibility_hygiene.py": "this guard; it names the fragments it bans",
    "tests/test_rxrx1_9plus_launcher.py": "asserts the launcher does NOT hard-code a home-directory ckpt",
    "docs/research/kbound/kbound_repro/storage.py": "the scanner; the fragments are its regexes",
    "docs/research/kbound/kbound_repro/check_repo.py": "the scanner's CLI; documents what it flags",
    "docs/research/kbound/kbound_repro/paths.py": "documents which paths it replaces, in prose",
    "docs/research/kbound/kbound_repro/tests/test_storage.py": "builds a synthetic violating file to test the scanner",
    "docs/research/kbound/scrub_submission.py": "the anonymiser; the fragment is a substitution pattern",
    "docs/research/kbound/scripts/code_audit_uav.py": "one prose line recording the volume's historical name",
    "scripts/migrate_repo_name_to_kbound.sh": "record of the completed rename; the old name is its subject",
}


def _readable(path: Path) -> bool:
    """True iff ``path`` holds real text (not an unmaterialised iCloud stub)."""
    try:
        raw = path.read_bytes()
    except OSError:
        return False
    return len(raw) > 0 and b"\x00" not in raw


def _stable_seed_source() -> str:
    """Extract just the ``stable_seed`` function from the CIFAR-10-C suite.

    The module itself imports torch, torchvision and ``imagecorruptions``, none
    of which are installable in a test environment, so we lift the one function
    under test out of the AST instead of importing the module.
    """
    tree = ast.parse(CIFAR10C_SUITE.read_text())
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == "stable_seed":
            return ast.get_source_segment(CIFAR10C_SUITE.read_text(), node) or ""
    return ""


class TestStableSeedIsProcessStable:
    """F2-8: the per-cell seed must not depend on ``PYTHONHASHSEED``."""

    @staticmethod
    def _run(source: str, hashseed: str) -> str:
        env = dict(os.environ, PYTHONHASHSEED=hashseed)
        proc = subprocess.run(
            [sys.executable, "-c", source],
            env=env,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )
        assert proc.returncode == 0, proc.stderr
        return proc.stdout.strip()

    def test_builtin_hash_really_is_salted(self):
        """Control: without this, the next test could pass vacuously."""
        src = 'print(hash("gaussian_noise"))'
        assert self._run(src, "0") != self._run(src, "12345"), (
            "PYTHONHASHSEED had no effect in this interpreter, so the "
            "process-stability assertion below would prove nothing"
        )

    def test_suite_seed_is_identical_across_hash_seeds(self):
        if not _readable(CIFAR10C_SUITE):
            pytest.skip(f"{CIFAR10C_SUITE} is not materialised in this release")
        fn = _stable_seed_source()
        if not fn:
            pytest.fail(
                f"{CIFAR10C_SUITE} defines no stable_seed(); per-cell seeding must "
                "come from a fixed digest, not from Python's salted hash()"
            )
        src = "import hashlib\n" + textwrap.dedent(fn) + (
            "\nprint([stable_seed('cifar10c_suite', c, s)"
            " for c in ('gaussian_noise', 'snow', 'jpeg_compression')"
            " for s in (1, 3, 5)])\n"
        )
        assert self._run(src, "0") == self._run(src, "12345")

    def test_suite_does_not_seed_from_builtin_hash(self):
        """Static guard: no ``hash(`` inside a ``default_rng``/``seed`` call."""
        if not _readable(CIFAR10C_SUITE):
            pytest.skip(f"{CIFAR10C_SUITE} is not materialised in this release")
        tree = ast.parse(CIFAR10C_SUITE.read_text())
        offenders = []
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            fname = getattr(node.func, "attr", getattr(node.func, "id", ""))
            if fname not in ("default_rng", "seed", "manual_seed", "RandomState"):
                continue
            for sub in ast.walk(node):
                if isinstance(sub, ast.Call) and getattr(sub.func, "id", "") == "hash":
                    offenders.append(node.lineno)
        assert not offenders, (
            f"{CIFAR10C_SUITE.name} seeds an RNG from the salted builtin hash() at "
            f"line(s) {sorted(set(offenders))}; use the blake2b stable_seed() helper"
        )


class TestNoMachineLocalPaths:
    """F3-17 / EXTERNAL_STORAGE_POLICY.md:18 -- defect D8, now tree-wide."""

    @staticmethod
    def _offending_files(root: Path) -> dict[str, list[str]]:
        """Map repo-relative path -> offending ``line:text`` for every tracked
        ``.py``/``.sh`` under ``root`` that mentions a banned fragment.

        Comment lines are **not** exempt: a runbook comment telling a reader to
        ``cd /Volumes/T9/...`` is exactly as unusable as executable code, and
        exempting comments is how 94 files stayed invisible to the old guard.
        """
        bad: dict[str, list[str]] = {}
        for pattern in ("*.py", "*.sh"):
            for path in sorted(root.rglob(pattern)):
                if "__pycache__" in path.parts or not _readable(path):
                    continue
                rel = str(path.relative_to(REPO))
                hits = [
                    f"{i}: {line.strip()}"
                    for i, line in enumerate(path.read_text(errors="ignore").splitlines(), 1)
                    if any(frag in line for frag in BANNED_PATH_FRAGMENTS)
                ]
                if hits:
                    bad[rel] = hits
        return bad

    def test_no_machine_local_paths_anywhere_in_the_tree(self):
        found = self._offending_files(REPO)
        unexpected = {k: v for k, v in found.items() if k not in MACHINE_LOCAL_ALLOWLIST}
        assert not unexpected, (
            "machine-local absolute paths are back in tracked executables "
            f"({len(unexpected)} file(s)): {unexpected}. Use $KBOUND_REPO_ROOT "
            "(discovered from the file's own location) or $KBOUND_EXTERNAL_ROOT "
            "(documented in docs/research/kbound/kbound_repro/paths.py, and an error "
            "when unset). EXTERNAL_STORAGE_POLICY.md:18 bans these strings."
        )

    def test_the_allowlist_has_no_stale_entries(self):
        """An allowlist that outlives its violations stops meaning anything."""
        found = self._offending_files(REPO)
        stale = sorted(set(MACHINE_LOCAL_ALLOWLIST) - set(found))
        assert not stale, (
            f"these allowlist entries no longer contain a banned fragment: {stale}. "
            "Delete them so the list keeps documenting only real survivors."
        )

    def test_shipped_library_has_no_machine_local_paths(self):
        assert self._offending_files(KGA) == {}

    def test_test_suite_has_no_machine_local_paths(self):
        offenders = self._offending_files(TESTS)
        assert set(offenders) <= set(MACHINE_LOCAL_ALLOWLIST), offenders


class TestNoRawEnvironMutation:
    """Env vars must be set through ``monkeypatch``, which restores on failure."""

    def test_no_direct_environ_assignment_in_tests(self):
        offenders = []
        for path in sorted(TESTS.rglob("test_*.py")):
            if "__pycache__" in path.parts or not _readable(path):
                continue
            tree = ast.parse(path.read_text())
            for node in ast.walk(tree):
                targets = []
                if isinstance(node, ast.Assign):
                    targets = node.targets
                elif isinstance(node, ast.AugAssign):
                    targets = [node.target]
                for tgt in targets:
                    if (
                        isinstance(tgt, ast.Subscript)
                        and isinstance(tgt.value, ast.Attribute)
                        and tgt.value.attr == "environ"
                    ):
                        offenders.append(f"{path.relative_to(REPO)}:{node.lineno}")
        assert not offenders, (
            "os.environ was assigned directly at "
            f"{offenders}; use monkeypatch.setenv so a failing assertion cannot "
            "leak the variable into the rest of the session"
        )
