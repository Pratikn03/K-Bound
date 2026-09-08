"""Check effective pip options for every CI CPU-lock consumer, without installs."""
from pathlib import Path
import argparse
import shlex

import yaml


def test_ci_cpu_lock_install_commands_supply_hash_checked_cpu_index():
    root = Path(__file__).resolve().parents[1]
    consumers = 0
    for name in ("ci.yml", "kbound-ci.yml"):
        workflow = yaml.safe_load((root / ".github/workflows" / name).read_text())
        for job in workflow["jobs"].values():
            for step in job.get("steps", []):
                for line in step.get("run", "").replace("\\\n", " ").splitlines():
                    if "-r requirements-ci-py312-linux.lock.txt" not in line:
                        continue
                    args = shlex.split(line)
                    assert args[:4] == ["python", "-m", "pip", "install"]
                    parser = argparse.ArgumentParser()
                    parser.add_argument("--require-hashes", action="store_true")
                    parser.add_argument("--only-binary")
                    parser.add_argument("--extra-index-url", action="append", default=[])
                    parser.add_argument("-r")
                    options = parser.parse_args(args[4:])
                    assert options.require_hashes
                    assert options.only_binary == ":all:"
                    assert "https://download.pytorch.org/whl/cpu" in options.extra_index_url
                    consumers += 1
    assert consumers == 3


def test_ci_executes_safe_lock_and_readme_regressions_not_collection_only():
    root = Path(__file__).resolve().parents[1]
    workflow = yaml.safe_load((root / ".github/workflows/kbound-ci.yml").read_text())
    required = {
        "tests/test_ci_cpu_index_contract.py",
        "tests/test_readme_quickstart_authority.py",
    }
    safe_commands = []
    for job in workflow["jobs"].values():
        for step in job.get("steps", []):
            for line in step.get("run", "").replace("\\\n", " ").splitlines():
                if not line.lstrip().startswith("pytest "):
                    continue
                args = shlex.split(line)
                if not args or args[0] != "pytest" or "--collect-only" in args:
                    continue
                targets = {arg for arg in args[1:] if arg.startswith("tests/")}
                if required <= targets:
                    safe_commands.append(targets)
    # A dedicated command must not accidentally pull in protected artifact tests.
    assert safe_commands == [required]
