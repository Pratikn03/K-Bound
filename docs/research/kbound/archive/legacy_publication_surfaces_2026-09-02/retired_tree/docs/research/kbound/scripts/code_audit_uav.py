#!/usr/bin/env python3
"""Automated code audit for the external volume $KBOUND_EXTERNAL_ROOT.

Scans Python for integrity patterns, duplicates vs kbound_only, and hygiene issues.
Does NOT replace human review of proofs — complements theory_audit_full.py.

Usage:
  python3 docs/research/kbound/scripts/code_audit_uav.py
  python3 docs/research/kbound/scripts/code_audit_uav.py --write-report
"""
from __future__ import annotations
# --- defect D8: portable roots (docs/research/kbound/EXTERNAL_STORAGE_POLICY.md bans
# --- machine-local absolute paths in tracked code). KB_REPO_ROOT is discovered from this
# --- file's own location; override with $KBOUND_REPO_ROOT.
import os as _kb_os
from pathlib import Path as _KbPath


def _kb_repo_root() -> str:
    override = _kb_os.environ.get("KBOUND_REPO_ROOT", "").strip()
    if override:
        return str(_KbPath(override).expanduser().resolve())
    here = _KbPath(__file__).resolve()
    for candidate in here.parents:
        if (candidate / "pyproject.toml").exists():
            return str(candidate)
    raise RuntimeError(f"repository root not found above {here}; set KBOUND_REPO_ROOT")


KB_REPO_ROOT = _kb_repo_root()

# --- external (git-excluded) data volume: ONE documented variable, no default.
def _kb_external_root() -> str:
    value = _kb_os.environ.get("KBOUND_EXTERNAL_ROOT", "").strip()
    if not value:
        raise RuntimeError(
            "KBOUND_EXTERNAL_ROOT is not set. This script needs data that is deliberately "
            "not in the git release (raw datasets, checkpoints, caches). Point "
            "KBOUND_EXTERNAL_ROOT at the volume holding them; the expected layout is "
            "documented in docs/research/kbound/kbound_repro/paths.py (EXTERNAL_LAYOUT) "
            "and acquisition is in DATA.md. There is no default on purpose: this used to "
            "be one author's external SSD, and defaulting to $HOME would write gigabytes "
            "somewhere you did not choose."
        )
    return str(_KbPath(value).expanduser().resolve())


KB_EXTERNAL_ROOT = _kb_external_root()


import argparse
import hashlib
import json
import re
import subprocess
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

# The volume being audited ($KBOUND_EXTERNAL_ROOT) and the repository inside it.
# Historically these were /Volumes/T9/uav and .../AutoML_Flagship_V8; the repo has
# since been renamed and may live anywhere, so it is discovered, not assumed.
UAV = Path(KB_EXTERNAL_ROOT)
AUTO = Path(KB_REPO_ROOT)
KBOUND = AUTO / "docs" / "research" / "kbound"
KONLY = UAV / "kbound_only"

SKIP_DIRS = {
    ".venv", "__pycache__", "node_modules", "torch_cache", "tmp",
    ".git", "data", "models", "DomainBed", "build", "dist",
}

# Headline scorers — must use OOF/LOO conformal (G1 integrity)
HEADLINE_SCORERS = [
    AUTO / "docs/research/kbound/scripts/cifar_tent_mps_v2.py",
    AUTO / "docs/research/kbound/scripts/score_kbound_holdout.py",
    AUTO / "docs/research/kbound/scripts/mixed_stream_kbound.py",
    AUTO / "docs/research/kbound/scripts/pacs_vlcs_runner.py",
    AUTO / "docs/research/kbound/kbound_pkg/kbound/certificate.py",
    AUTO / "experiments/kbound/poem_aetta/run_mixed_headtohead.py",
]

BAD_EPS = re.compile(r"predict\(Zc\)\s*-\s*Bc|abs\(Bhat_c\s*-\s*Bc\)")
OK_EPS = re.compile(r"resid_c|_loo|out-of-fold", re.I)
SECRET = re.compile(
    r"(api[_-]?key\s*=\s*['\"][^'\"]+['\"]|password\s*=\s*['\"][^'\"]+['\"]|"
    r"sk-[a-zA-Z0-9]{20,}|AKIA[0-9A-Z]{16})",
    re.I,
)

MIRROR_PATHS = [
    ("kga", AUTO / "kga", KONLY / "kga"),
    ("src/scripts/kbound", AUTO / "src/scripts/kbound", KONLY / "src/scripts/kbound"),
    ("kbound_pkg", KBOUND / "kbound_pkg", KONLY / "docs/research/kbound/kbound_pkg"),
    ("poem_aetta", AUTO / "experiments/kbound/poem_aetta", KONLY / "experiments/kbound/poem_aetta"),
]


def _iter_py(root: Path) -> list[Path]:
    if not root.exists():
        return []
    out = []
    for p in root.rglob("*.py"):
        if any(s in p.parts for s in SKIP_DIRS):
            continue
        if p.name.startswith("._"):
            continue
        out.append(p)
    return out


def _sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def scan_headline_scorers() -> list[dict]:
    rows = []
    for p in HEADLINE_SCORERS:
        if not p.exists():
            rows.append({"path": str(p.relative_to(AUTO)), "status": "MISSING", "issues": ["file not found"]})
            continue
        text = p.read_text(errors="replace")
        issues = []
        for i, line in enumerate(text.splitlines(), 1):
            if BAD_EPS.search(line) and not OK_EPS.search(line):
                issues.append(f"L{i}: in-sample-eps pattern")
        rows.append({
            "path": str(p.relative_to(AUTO)),
            "status": "OK" if not issues else "FAIL",
            "issues": issues,
        })
    return rows


def scan_secrets(roots: list[Path]) -> list[dict]:
    hits = []
    for root in roots:
        for p in _iter_py(root):
            if "test_" in p.name or "/tests/" in str(p):
                continue
            for i, line in enumerate(p.read_text(errors="replace").splitlines(), 1):
                if SECRET.search(line) and "example" not in line.lower():
                    hits.append({"file": str(p), "line": i, "snippet": line.strip()[:80]})
    return hits[:50]


def compare_mirrors() -> list[dict]:
    rows = []
    for name, a, b in MIRROR_PATHS:
        if not a.exists() or not b.exists():
            rows.append({"mirror": name, "status": "skip", "note": "path missing"})
            continue
        a_files = {p.relative_to(a): _sha(p) for p in _iter_py(a)}
        b_files = {p.relative_to(b): _sha(p) for p in _iter_py(b)}
        only_a = set(a_files) - set(b_files)
        only_b = set(b_files) - set(a_files)
        diff_hash = [str(r) for r in a_files if r in b_files and a_files[r] != b_files[r]]
        rows.append({
            "mirror": name,
            "status": "identical" if not only_a and not only_b and not diff_hash else "diverged",
            "only_in_automl": [str(x) for x in sorted(only_a)[:10]],
            "only_in_kbound_only": [str(x) for x in sorted(only_b)[:10]],
            "hash_mismatches": diff_hash[:10],
            "counts": {"automl": len(a_files), "kbound_only": len(b_files)},
        })
    return rows


def count_loc() -> dict:
    stats = {}
    for label, root in [("repository", AUTO), ("kbound_only", KONLY), ("volume_root_src", UAV / "src")]:
        files = _iter_py(root)
        lines = sum(len(p.read_text(errors="replace").splitlines()) for p in files)
        stats[label] = {"py_files": len(files), "lines": lines}
    return stats


def stale_root() -> list[str]:
    notes = []
    for d in ["src", "tests", "docs", "research_lock"]:
        p = UAV / d
        if p.exists() and not any(p.rglob("*.py")) and not any(p.rglob("*.md")):
            notes.append(f"{UAV}/{d}/ is empty placeholder — use the repository at {AUTO}/")
    if (UAV / "kbound.log").exists():
        notes.append("Root kbound.log/aux are stale LaTeX artifacts (Jun 15); canonical papers in <repo>/docs/research/kbound/")
    return notes


def check_canonical_wrappers() -> list[dict]:
    """src/scripts/kbound headline scripts must delegate to docs/research/kbound/scripts/."""
    wrappers = [
        AUTO / "src/scripts/kbound/cifar_tent_mps_v2.py",
        AUTO / "src/scripts/kbound/cifar_tent_online.py",
    ]
    rows = []
    for p in wrappers:
        if not p.exists():
            rows.append({"path": str(p.relative_to(AUTO)), "status": "MISSING"})
            continue
        text = p.read_text(errors="replace")
        ok = "_canonical" in text and "run_canonical" in text
        rows.append({
            "path": str(p.relative_to(AUTO)),
            "status": "OK" if ok else "FAIL",
            "note": "thin wrapper" if ok else "expected run_canonical delegate",
        })
    return rows


def run_pytest_sample() -> tuple[str, int]:
    py = AUTO / ".venv/bin/python"
    if not py.exists():
        return "skipped (no .venv)", 0
    env = {
        **dict(__import__("os").environ),
        "PYTHONPATH": ":".join([
            str(AUTO),
            str(AUTO / "src"),
            str(AUTO / "docs/research/kbound/kbound_pkg"),
            str(AUTO / "docs/research/kbound/edge/src"),
        ]),
    }
    proc = subprocess.run(
        [
            str(py), "-m", "pytest",
            "docs/research/kbound/kbound_pkg/tests",
            "docs/research/kbound/tests",
            "docs/research/kbound/edge/tests",
            "-q", "--tb=no",
        ],
        cwd=str(AUTO),
        capture_output=True,
        text=True,
        timeout=300,
        env=env,
    )
    tail = (proc.stdout or proc.stderr).strip().splitlines()
    line = tail[-1] if tail else f"exit {proc.returncode}"
    return line, proc.returncode


def monorepo_score(wrappers: list[dict], scorers_ok: bool, pytest_rc: int) -> dict:
    wrappers_ok = all(r["status"] == "OK" for r in wrappers)
    tests_ok = pytest_rc == 0
    pyproject = (AUTO / "pyproject.toml").read_text(errors="replace")
    unified_tests = "docs/research/kbound/tests" in pyproject
    ci = (AUTO / ".github/workflows/kbound-ci.yml").read_text(errors="replace")
    ci_research = "kbound-research-tests" in ci
    health = (AUTO / "scripts/monorepo_health.sh").is_file()
    criteria = {
        "canonical_wrappers": wrappers_ok,
        "headline_scorers_g1": scorers_ok,
        "unified_pytest_paths": unified_tests,
        "ci_kbound_research_tests": ci_research,
        "monorepo_health_script": health,
        "research_tests_pass": tests_ok,
    }
    n = sum(criteria.values())
    grade = round(n / len(criteria) * 10, 1)
    return {"criteria": criteria, "grade_10": grade, "passed": n, "total": len(criteria)}


def build_report() -> dict:
    scorers = scan_headline_scorers()
    wrappers = check_canonical_wrappers()
    pytest_line, pytest_rc = run_pytest_sample()
    monorepo = monorepo_score(wrappers, all(r["status"] == "OK" for r in scorers), pytest_rc)
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "scope": "Automated pass over $KBOUND_EXTERNAL_ROOT and the repository inside it",
        "note": "Full line-by-line human review of ~161k LOC is not feasible in one pass; this audit targets integrity + mirrors + hygiene.",
        "loc": count_loc(),
        "stale_root": stale_root(),
        "headline_scorers": scorers,
        "scorers_ok": all(r["status"] == "OK" for r in scorers),
        "canonical_wrappers": wrappers,
        "wrappers_ok": all(r["status"] == "OK" for r in wrappers),
        "mirror_compare": compare_mirrors(),
        "secret_hits": scan_secrets([AUTO / "docs/research/kbound", AUTO / "experiments/kbound", AUTO / "kga"]),
        "pytest_kbound": pytest_line,
        "monorepo": monorepo,
        "recommended_actions": [],
    }


def write_md(report: dict, path: Path) -> None:
    lines = [
        "# UAV / K-Bound Code Audit (automated)",
        "",
        f"Generated: {report['generated_at']}",
        "",
        report["note"],
        "",
        "## Scale",
        "",
    ]
    for k, v in report["loc"].items():
        lines.append(f"- **{k}**: {v['py_files']} Python files, ~{v['lines']:,} lines")
    lines.extend(["", "## Stale / duplicate layout", ""])
    for n in report["stale_root"]:
        lines.append(f"- {n}")
    for m in report["mirror_compare"]:
        lines.append(f"- **{m['mirror']}**: {m['status']} {m.get('counts', '')}")
    lines.extend(["", "## Headline scorer integrity (G1)", ""])
    for r in report["headline_scorers"]:
        mark = "✓" if r["status"] == "OK" else "✗"
        lines.append(f"- {mark} `{r['path']}`")
        for iss in r.get("issues", []):
            lines.append(f"  - {iss}")
    lines.extend([
        "",
        f"**Scorers OK:** {report['scorers_ok']}",
        "",
        "## Canonical wrappers (src → docs)",
        "",
    ])
    for r in report.get("canonical_wrappers", []):
        mark = "✓" if r["status"] == "OK" else "✗"
        lines.append(f"- {mark} `{r['path']}`")
    lines.extend([
        "",
        f"**Wrappers OK:** {report.get('wrappers_ok', False)}",
        "",
        "## Monorepo engineering",
        "",
    ])
    m = report.get("monorepo", {})
    for k, v in m.get("criteria", {}).items():
        lines.append(f"- {'✓' if v else '✗'} {k}")
    lines.extend([
        "",
        f"**Monorepo grade:** {m.get('grade_10', '?')}/10 ({m.get('passed', '?')}/{m.get('total', '?')} criteria)",
        "",
        "## Tests",
        f"- K-Bound research pytest: {report['pytest_kbound']}",
        "",
        "## Secret pattern scan (sample)",
    ])
    if report["secret_hits"]:
        for h in report["secret_hits"][:10]:
            lines.append(f"- `{h['file']}` L{h['line']}: `{h['snippet']}`")
    else:
        lines.append("- No hardcoded secret patterns in K-Bound paths (sample scan).")
    lines.extend([
        "",
        "## What a full line-by-line audit would still need",
        "",
        "- Human review of `kga/` vs `kbound_pkg/` drift (documented in kbound_pkg/README)",
        "- `kbound_only/` sync or deprecate before public dual-repo confusion",
        "- Audit-only scripts under `audits/` and `theory_v2/realdata/eps_recal/` (in-sample OK for diagnostics)",
        "- ELARA/UAIS code under `src/uais/` (legacy product stack, not paper spine)",
    ])
    path.write_text("\n".join(lines) + "\n")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--write-report", action="store_true")
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()
    report = build_report()
    if args.write_report or not args.json:
        out = KBOUND / "reports" / "CODE_AUDIT_UAV.md"
        write_md(report, out)
        print(f"Wrote {out}")
    if args.json:
        print(json.dumps(report, indent=2))
    else:
        print(f"Headline scorers OK: {report['scorers_ok']}")
        print(f"Wrappers OK: {report.get('wrappers_ok', False)}")
        print(f"Monorepo grade: {report.get('monorepo', {}).get('grade_10', '?')}/10")
        print(f"pytest: {report['pytest_kbound']}")
    issues = [r for r in report["headline_scorers"] if r["status"] != "OK"]
    issues += [r for r in report.get("canonical_wrappers", []) if r["status"] != "OK"]
    return 1 if issues else 0


if __name__ == "__main__":
    raise SystemExit(main())
