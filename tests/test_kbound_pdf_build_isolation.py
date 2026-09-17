"""Run the real PDF shell driver without reading real evidence or cloud files.

The external scientific/LaTeX commands are bounded test doubles.  Publication,
shell redirection, temporary-directory handling, and command ordering stay real.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import signal
import shutil
import subprocess
import sys
from types import SimpleNamespace

import pytest


ROOT = Path(__file__).resolve().parents[1]
BUILD_SCRIPT = ROOT / "docs/research/kbound/scripts/build_pdfs.sh"


def _executable(path: Path, body: str) -> None:
    path.write_text(f"#!{sys.executable}\n" + body, encoding="utf-8")
    path.chmod(0o755)


@pytest.fixture
def build_replica(tmp_path: Path) -> SimpleNamespace:
    repo = tmp_path / "repository with spaces"
    paper = repo / "docs/research/kbound"
    scripts = paper / "scripts"
    scripts.mkdir(parents=True)
    script = scripts / "build_pdfs.sh"
    script.write_bytes(BUILD_SCRIPT.read_bytes())
    shared = paper / "paper/sections/shared.tex"
    shared.parent.mkdir(parents=True)
    shared.write_text("Shared relative input stays available.\n", encoding="utf-8")
    for driver in ("kbound_submission", "kbound_tmlr", "kbound_short"):
        (paper / f"{driver}.tex").write_text("\\input{paper/sections/shared.tex}\n", encoding="utf-8")
    (paper / "kbound_short_original_build.log").write_text("historical do not refresh\n", encoding="utf-8")

    tools = tmp_path / "fake-tools"
    tools.mkdir()
    scratch = tmp_path / "local build scratch"
    scratch.mkdir()
    events = tmp_path / "events.jsonl"
    common = """import json, os, pathlib, subprocess, sys
def event(kind, **values):
    with open(os.environ['BUILD_TEST_EVENTS'], 'a', encoding='utf-8') as stream:
        stream.write(json.dumps({'kind': kind, **values}) + '\\n')
paper = pathlib.Path(os.environ['BUILD_TEST_PAPER'])
scratch = pathlib.Path(os.environ['BUILD_TEST_SCRATCH'])
"""
    _executable(tools / "fake-python", common + """
if len(sys.argv) > 1 and sys.argv[1] == '-':
    os.execv(os.environ['BUILD_TEST_REAL_PYTHON'], [os.environ['BUILD_TEST_REAL_PYTHON'], *sys.argv[1:]])
script = pathlib.Path(sys.argv[1]).name
event('python', script=script, args=sys.argv[2:], cwd=os.getcwd())
if os.environ.get('BUILD_TEST_FAIL_VALIDATOR') == script:
    print('deliberate validation failure', file=sys.stderr)
    raise SystemExit(23)
if script == 'build_docx.py':
    output = pathlib.Path(sys.argv[sys.argv.index('--output') + 1])
    output.write_bytes(b'test successful DOCX')
""")
    _executable(tools / "latexmk", common + """
args = sys.argv[1:]
event('latexmk', args=args, cwd=os.getcwd())
assert pathlib.Path.cwd() == paper, 'relative TeX inputs must keep the paper working directory'
assert pathlib.Path('paper/sections/shared.tex').read_text() == 'Shared relative input stays available.\\n'
outargs = [arg.split('=', 1)[1] for arg in args if arg.startswith('-outdir=')]
assert len(outargs) == 1, 'latexmk must receive one isolated -outdir'
outdir = pathlib.Path(outargs[0])
assert outdir.is_absolute() and outdir.is_dir(), 'output directory must be an existing absolute local directory'
assert scratch in outdir.parents, 'all output must be below the requested temporary root'
auxargs = [arg.split('=', 1)[1] for arg in args if arg.startswith('-auxdir=')]
assert auxargs == [str(outdir)], 'auxiliary I/O must use the same fresh local directory'
driver = pathlib.Path(args[-1])
jobs = [arg.split('=', 1)[1] for arg in args if arg.startswith('-jobname=')]
job = jobs[0] if jobs else driver.stem
event('local-output', job=job, path=str(outdir))
for suffix in ('.aux', '.out', '.fdb_latexmk', '.fls'):
    (outdir / (job + suffix)).write_text('local intermediate\\n', encoding='utf-8')
if os.environ.get('BUILD_TEST_OMIT_PDF') != job:
    (outdir / (job + '.pdf')).write_bytes(b'%PDF-1.7\\nfresh ' + job.encode())
if os.environ.get('BUILD_TEST_OMIT_LOG') != job:
    (outdir / (job + '.log')).write_text('fresh TeX log ' + job + '\\n', encoding='utf-8')
print('fresh driver output ' + job)
if os.environ.get('BUILD_TEST_FAIL_JOB') == job:
    print('deliberate latex failure after a partial PDF', file=sys.stderr)
    raise SystemExit(19)
""")
    _executable(tools / "pdfinfo", common + """
event('pdfinfo', args=sys.argv[1:])
path = pathlib.Path(sys.argv[-1])
assert path.is_absolute() and scratch in path.parents, 'verify the local PDF before publication'
assert path.read_bytes().startswith(b'%PDF-1.7\\nfresh ')
if os.environ.get('BUILD_TEST_INVALID_PDF') == path.stem:
    print('deliberate invalid PDF', file=sys.stderr)
    raise SystemExit(1)
print('Pages: 2')
""")
    _executable(tools / "cp", common + """
source = pathlib.Path(sys.argv[-2])
event('cp', source=str(source), destination=sys.argv[-1])
if source.parent.resolve() == paper and source.suffix in {'.log', '.pdf', '.aux'}:
    print('must not open existing derived/cloud outputs as copy sources', file=sys.stderr)
    raise SystemExit(91)
os.execv('/bin/cp', ['/bin/cp', *sys.argv[1:]])
""")
    _executable(tools / "rm", common + """
event('prohibited-cleanup', args=sys.argv[1:])
raise SystemExit('build must retain diagnostics, not recursively clean old files')
""")
    env = {
        **os.environ,
        "PATH": str(tools) + os.pathsep + os.environ.get("PATH", ""),
        "PYTHON": str(tools / "fake-python"),
        "TMPDIR": str(scratch),
        "BUILD_TEST_EVENTS": str(events),
        "BUILD_TEST_PAPER": str(paper),
        "BUILD_TEST_SCRATCH": str(scratch),
        "BUILD_TEST_REAL_PYTHON": sys.executable,
    }
    for key in ("BUILD_LONG_TMLR", "BUILD_HISTORICAL_TMLR", "BUILD_DOCX", "BUILD_DIAGNOSTIC_IEEE"):
        env.pop(key, None)
    return SimpleNamespace(repo=repo, paper=paper, script=script, scratch=scratch, events=events, env=env)


def _run(replica, **environment):
    command = ["bash", str(replica.script)]
    with subprocess.Popen(
        command,
        cwd=replica.repo,
        env={**replica.env, **environment},
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        start_new_session=True,
    ) as process:
        try:
            stdout, stderr = process.communicate(timeout=15)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            stdout, stderr = process.communicate()
            pytest.fail("isolated build timed out; its test-only process group was stopped\n" + stdout + stderr)
        return subprocess.CompletedProcess(command, process.returncode, stdout, stderr)


def _events(replica):
    if not replica.events.exists():
        return []
    return [json.loads(line) for line in replica.events.read_text().splitlines()]


def _old_output_canaries(replica, names):
    targets = {}
    for name in names:
        target = replica.repo.parent / (name + ".canary")
        target.write_bytes(b"old output must never be opened or changed\n")
        (replica.paper / name).symlink_to(target)
        targets[name] = target
    return targets


def test_short_build_isolates_intermediates_and_atomically_replaces_only_successful_outputs(build_replica):
    replica = build_replica
    final_names = (
        "kbound_short_final_draft.pdf", "kbound_short_final_draft.log",
        "kbound_submission_build_driver.log", "kbound_short_final_build.log",
    )
    auxiliary_names = ("kbound_short_final_draft.aux", "kbound_short_final_draft.out", "kbound_short_final_draft.fdb_latexmk")
    canaries = _old_output_canaries(replica, (*final_names, *auxiliary_names))
    result = _run(replica)
    assert result.returncode == 0, result.stdout + result.stderr
    for name, target in canaries.items():
        assert target.read_bytes() == b"old output must never be opened or changed\n", name
    for name in final_names:
        assert not (replica.paper / name).is_symlink(), name
        assert (replica.paper / name).stat().st_mode & 0o777 == 0o644
    for name in auxiliary_names:
        assert (replica.paper / name).is_symlink(), name
    assert (replica.paper / "kbound_short_final_draft.pdf").read_bytes() == b"%PDF-1.7\nfresh kbound_short_final_draft"
    assert (replica.paper / "kbound_short_final_build.log").read_text() == "fresh TeX log kbound_short_final_draft\n"
    assert "fresh driver output" in (replica.paper / "kbound_submission_build_driver.log").read_text()
    local = [event for event in _events(replica) if event["kind"] == "local-output"]
    assert len(local) == 1
    assert (Path(local[0]["path"]) / "kbound_short_final_draft.aux").read_text() == "local intermediate\n"
    assert str(Path(local[0]["path"]).parent) in result.stdout or local[0]["path"] in result.stdout


@pytest.mark.parametrize("problem", ["BUILD_TEST_FAIL_JOB", "BUILD_TEST_OMIT_PDF", "BUILD_TEST_OMIT_LOG", "BUILD_TEST_INVALID_PDF"])
def test_failed_or_incomplete_build_never_replaces_existing_pdf_or_logs(build_replica, problem):
    replica = build_replica
    names = ("kbound_short_final_draft.pdf", "kbound_submission_build_driver.log", "kbound_short_final_build.log")
    canaries = _old_output_canaries(replica, names)
    result = _run(replica, **{problem: "kbound_short_final_draft"})
    assert result.returncode != 0
    for name, target in canaries.items():
        assert (replica.paper / name).is_symlink(), name
        assert target.read_bytes() == b"old output must never be opened or changed\n", name
    local = [event for event in _events(replica) if event["kind"] == "local-output"]
    assert len(local) == 1
    driver = Path(local[0]["path"]) / "kbound_submission_build_driver.log"
    assert "fresh driver output" in driver.read_text()
    assert not any(event["kind"] == "python" and event["script"] == "build_dashboard_snapshot.py" for event in _events(replica))


def test_all_scientific_validation_precedes_latex_and_metadata_refresh_is_last(build_replica):
    result = _run(build_replica, BUILD_DOCX="1")
    assert result.returncode == 0, result.stdout + result.stderr
    events = _events(build_replica)
    calls = [event for event in events if event["kind"] in {"python", "latexmk"}]
    assert [call.get("script", "latexmk") for call in calls] == [
        "build_so2sat_numbers.py", "validate_canonical_release_data.py",
        "build_current_policy_interval_diagnostics.py", "validate_manuscript_claims.py",
        "make_tables.py", "plot_canonical_decision_frontier.py",
        "plot_conceptual_regime_geometry.py", "make_submission_figures.py",
        "plot_kga_interval_rule.py", "latexmk", "build_docx.py", "build_dashboard_snapshot.py",
    ]
    assert calls[2]["args"] == ["--check"]
    assert calls[-1]["args"] == ["--metadata-only"]
    assert (build_replica.paper / "kbound_short_final_draft.docx").read_bytes() == b"test successful DOCX"


def test_failed_scientific_validation_never_starts_latex_or_publishes(build_replica):
    canaries = _old_output_canaries(build_replica, ("kbound_short_final_draft.pdf",))
    result = _run(build_replica, BUILD_TEST_FAIL_VALIDATOR="validate_manuscript_claims.py")
    assert result.returncode != 0
    assert not any(event["kind"] == "latexmk" for event in _events(build_replica))
    assert (build_replica.paper / "kbound_short_final_draft.pdf").is_symlink()
    assert canaries["kbound_short_final_draft.pdf"].read_bytes() == b"old output must never be opened or changed\n"


@pytest.mark.parametrize("long_option", ["BUILD_LONG_TMLR", "BUILD_HISTORICAL_TMLR"])
def test_long_and_explicit_diagnostic_outputs_keep_existing_contracts(build_replica, long_option):
    result = _run(build_replica, **{long_option: "1", "BUILD_DIAGNOSTIC_IEEE": "1"})
    assert result.returncode == 0, result.stdout + result.stderr
    paper = build_replica.paper
    for name, job in (
        ("kbound_short_final_draft.pdf", "kbound_short_final_draft"),
        ("kbound_tmlr.pdf", "kbound_tmlr"),
        ("kbound_short.pdf", "kbound_short"),
        ("kbound_full_ieee_diagnostic.pdf", "kbound_short"),
    ):
        assert (paper / name).read_bytes() == b"%PDF-1.7\nfresh " + job.encode()
    for name in ("kbound_tmlr.log", "kbound_tmlr_build.log", "kbound_short.log", "kbound_full_ieee_diagnostic_build.log"):
        assert (paper / name).is_file()
    assert len([event for event in _events(build_replica) if event["kind"] == "local-output"]) == 3


def test_historical_log_is_not_read_or_copied_on_a_normal_build(build_replica):
    (build_replica.paper / "kbound_short_original_build.log").unlink()
    _old_output_canaries(build_replica, ("kbound_tmlr.log",))
    result = _run(build_replica)
    assert result.returncode == 0, result.stdout + result.stderr
    assert not (build_replica.paper / "kbound_short_original_build.log").exists()
    assert (build_replica.paper / "kbound_tmlr.log").is_symlink()


def test_failed_long_build_retains_previous_long_outputs_after_successful_short(build_replica):
    names = ("kbound_tmlr.pdf", "kbound_tmlr.log", "kbound_tmlr_build.log")
    canaries = _old_output_canaries(build_replica, names)
    result = _run(build_replica, BUILD_LONG_TMLR="1", BUILD_TEST_FAIL_JOB="kbound_tmlr")
    assert result.returncode != 0
    assert (build_replica.paper / "kbound_short_final_draft.pdf").read_bytes() == b"%PDF-1.7\nfresh kbound_short_final_draft"
    for name, target in canaries.items():
        assert (build_replica.paper / name).is_symlink(), name
        assert target.read_bytes() == b"old output must never be opened or changed\n", name


def test_repeated_builds_use_distinct_fresh_local_directories(build_replica):
    first = _run(build_replica)
    second = _run(build_replica)
    assert first.returncode == second.returncode == 0, first.stderr + second.stderr
    outputs = [event["path"] for event in _events(build_replica) if event["kind"] == "local-output"]
    assert len(outputs) == 2 and outputs[0] != outputs[1]
    assert all(Path(path).is_dir() for path in outputs)


def test_invalid_long_option_fails_before_any_build_activity(build_replica):
    result = _run(build_replica, BUILD_LONG_TMLR="invalid")
    assert result.returncode != 0
    assert _events(build_replica) == []


@pytest.mark.parametrize("bibliography", ["manual", "bibtex"])
def test_resident_latex_toolchain_does_not_reuse_old_auxiliary_files(build_replica, tmp_path, bibliography):
    """Catch TeX falling back to the paper directory despite -outdir."""
    latexmk = shutil.which("latexmk")
    pdfinfo = shutil.which("pdfinfo")
    if latexmk is None or pdfinfo is None:
        pytest.skip("A local TeX/Poppler toolchain is not installed")
    real_tools = tmp_path / "resident-tex-tools"
    real_tools.mkdir()
    # Execute each installed wrapper at its original path: the bundled
    # pdfinfo wrapper locates its native binary relative to its own $0.
    for name, binary in (("latexmk", latexmk), ("pdfinfo", pdfinfo)):
        _executable(real_tools / name, f"import os, sys\nos.execv({binary!r}, [{binary!r}, *sys.argv[1:]])\n")
    references = (
        "\\bibliographystyle{plain}\n\\bibliography{localtest}\n"
        if bibliography == "bibtex" else
        "\\begin{thebibliography}{1}\n\\bibitem{localtest} Local reference.\n\\end{thebibliography}\n"
    )
    (build_replica.paper / "kbound_submission.tex").write_text(
        "\\documentclass{article}\n\\usepackage{hyperref}\n\\begin{document}\n"
        "\\tableofcontents\n\\listoffigures\n\\listoftables\n"
        "\\input{paper/sections/shared.tex}\nSee~\\cite{localtest}.\n"
        + references + "\\end{document}\n",
        encoding="utf-8",
    )
    (build_replica.paper / "localtest.bib").write_text(
        "@article{localtest, author={Local Test}, title={Resident fixture}, journal={Test Journal}, year={2026}}\n",
        encoding="utf-8",
    )
    suffixes = (".aux", ".out", ".toc", ".lof", ".lot", ".bbl")
    for suffix in suffixes:
        (build_replica.paper / ("kbound_short_final_draft" + suffix)).write_text(
            "\\errmessage{The build read an old auxiliary file}\n", encoding="utf-8",
        )
    result = _run(build_replica, PATH=str(real_tools) + os.pathsep + build_replica.env["PATH"])
    assert result.returncode == 0, result.stdout + result.stderr
    assert (build_replica.paper / "kbound_short_final_draft.pdf").read_bytes().startswith(b"%PDF-")
    prefix = "==> Local LaTeX intermediates and diagnostics: "
    local_dirs = [Path(line.removeprefix(prefix)) for line in result.stdout.splitlines() if line.startswith(prefix)]
    assert len(local_dirs) == 1
    assert "\\bibcite{localtest}{1}" in (local_dirs[0] / "kbound_short_final_draft.aux").read_text()
    if bibliography == "bibtex":
        bbl = (local_dirs[0] / "kbound_short_final_draft.bbl").read_text()
        assert "\\bibitem{localtest}" in bbl
        assert "Resident fixture" in bbl
    assert "There were undefined references" not in (build_replica.paper / "kbound_short_final_draft.log").read_text()
    for suffix in suffixes:
        assert (build_replica.paper / ("kbound_short_final_draft" + suffix)).read_text() == "\\errmessage{The build read an old auxiliary file}\n"
