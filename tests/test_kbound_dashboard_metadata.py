"""Presentation metadata must come from built/current inputs, never old literals."""

from __future__ import annotations

import copy
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from docs.research.kbound.scripts import build_dashboard_snapshot as dashboard


@pytest.fixture
def pdf_path(tmp_path: Path) -> Path:
    path = tmp_path / "paper.pdf"
    path.write_bytes(b"%PDF-1.7\nresident test input\n")
    return path


def test_pdfinfo_page_count_is_bounded_and_source_backed(monkeypatch, pdf_path):
    monkeypatch.setattr(dashboard.shutil, "which", lambda name: "/tools/pdfinfo")
    calls = []

    def run(args, **kwargs):
        calls.append((args, kwargs))
        return subprocess.CompletedProcess(args, 0, "Title: Paper\nPages: 34\n", "")

    monkeypatch.setattr(dashboard.subprocess, "run", run)
    assert dashboard.pdf_page_count(pdf_path) == 34
    assert calls[0][0] == ["/tools/pdfinfo", str(pdf_path)]
    assert calls[0][1]["timeout"] == 15
    assert calls[0][1]["check"] is True
    assert calls[0][1]["env"]["LC_ALL"] == "C"


@pytest.mark.parametrize("output", ["", "Pages: 0\n", "Pages: -1\n", "Pages: 3.5\n", "Pages: 34\nPages: 35\n"])
def test_pdfinfo_rejects_missing_or_ambiguous_counts(monkeypatch, pdf_path, output):
    monkeypatch.setattr(dashboard.shutil, "which", lambda name: "/tools/pdfinfo")
    monkeypatch.setattr(
        dashboard.subprocess,
        "run",
        lambda *args, **kwargs: subprocess.CompletedProcess(args, 0, output, ""),
    )
    with pytest.raises(ValueError, match="one positive page count"):
        dashboard.pdf_page_count(pdf_path)


@pytest.mark.parametrize(
    "failure",
    [subprocess.TimeoutExpired("pdfinfo", 15), subprocess.CalledProcessError(1, "pdfinfo"), OSError("unavailable")],
)
def test_pdfinfo_execution_failures_are_not_silenced(monkeypatch, pdf_path, failure):
    monkeypatch.setattr(dashboard.shutil, "which", lambda name: "/tools/pdfinfo")

    def fail(*args, **kwargs):
        raise failure

    monkeypatch.setattr(dashboard.subprocess, "run", fail)
    with pytest.raises(RuntimeError, match="Could not verify"):
        dashboard.pdf_page_count(pdf_path)


def test_pdfinfo_dependency_is_required(monkeypatch, pdf_path):
    monkeypatch.setattr(dashboard.shutil, "which", lambda name: None)
    with pytest.raises(RuntimeError, match="pdfinfo is required"):
        dashboard.pdf_page_count(pdf_path)


def test_missing_or_empty_pdf_is_not_given_a_fallback_count(tmp_path):
    path = tmp_path / "absent.pdf"
    with pytest.raises(FileNotFoundError, match="missing"):
        dashboard.pdf_page_count(path)
    path.write_bytes(b"")
    with pytest.raises(ValueError, match="empty"):
        dashboard.pdf_page_count(path)


def test_cloud_only_input_is_rejected_before_read(monkeypatch, pdf_path):
    class CloudMetadata:
        st_size = 100
        st_blocks = 0

    with monkeypatch.context() as scoped:
        scoped.setattr(Path, "is_file", lambda self: True)
        scoped.setattr(Path, "stat", lambda self: CloudMetadata())
        with pytest.raises(ValueError, match="not locally resident"):
            dashboard.require_resident_file(pdf_path)


def test_theory_counts_ignore_comments_definitions_remarks_and_starred_forms(monkeypatch, tmp_path):
    source = tmp_path / "theory.tex"
    source.write_text(
        "\\begin{theorem}[One]\n"
        "% \\begin{theorem} is commented out\n"
        "\\begin{lemma}\n\\begin{proposition}\n\\begin{corollary}\n"
        "\\begin{definition}\n\\begin{remark}\n\\begin{theorem*}\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(dashboard, "THEORY_SOURCES", (source,))
    assert dashboard.theory_statement_counts() == {
        "theorem": 1,
        "lemma": 1,
        "proposition": 1,
        "corollary": 1,
    }


def test_missing_theory_source_does_not_use_historical_counts(monkeypatch, tmp_path):
    monkeypatch.setattr(dashboard, "THEORY_SOURCES", (tmp_path / "missing.tex",))
    with pytest.raises(FileNotFoundError, match="missing"):
        dashboard.theory_statement_counts()


def test_current_theory_statement_scope_is_explicit():
    assert [path.name for path in dashboard.THEORY_SOURCES] == [
        "kbound_submission_body.tex",
        "theory_core_main.tex",
        "theory_certificate.tex",
    ]
    assert dashboard.theory_statement_counts() == {
        "theorem": 3,
        "lemma": 2,
        "proposition": 3,
        "corollary": 1,
    }


def test_theory_strip_does_not_claim_full_formalization(monkeypatch):
    monkeypatch.setattr(dashboard, "pdf_page_count", lambda path: 34)
    monkeypatch.setattr(
        dashboard,
        "theory_statement_counts",
        lambda: {
            "theorem": 4,
            "lemma": 2,
            "proposition": 1,
            "corollary": 1,
        },
    )
    pages, strip = dashboard.presentation_metadata()
    assert pages == 34
    assert strip == {
        "value": "4 theorems",
        "sub": "8 numbered statements; stated assumptions apply",
    }


def test_metadata_only_mode_preserves_every_other_field_and_never_reads_edge(monkeypatch, tmp_path):
    path = tmp_path / "snapshot.json"
    original = {
        "meta": {"paper": dashboard.rel(dashboard.SHORT_PDF), "paper_pages": 22, "current_policy_sha256": "keep"},
        "evidence_strip": {
            "proven_theorems": {"value": "3 core", "sub": "old bridges"},
            "theorem_validators": {"value": "Lean partial", "sub": "external assumptions disclosed"},
        },
        "edge_validation": {"study_status": "pending", "do_not_read_data": True},
        "evidence_board": {"existing_result": [0.1, 0.2, 0.3]},
    }
    path.write_text(json.dumps(original), encoding="utf-8")
    monkeypatch.setattr(dashboard, "OUT", path)
    monkeypatch.setattr(
        dashboard,
        "presentation_metadata",
        lambda: (
            34,
            {
                "value": "4 theorems",
                "sub": "8 numbered statements; stated assumptions apply",
            },
        ),
    )

    def prohibited(*args, **kwargs):
        raise AssertionError("metadata-only refresh must not inspect data or rebuild evidence")

    monkeypatch.setattr(dashboard, "edge_status", prohibited)
    monkeypatch.setattr(dashboard, "build_snapshot", prohibited)
    assert dashboard.main(["--metadata-only"]) == 0
    actual = json.loads(path.read_text())
    expected = copy.deepcopy(original)
    expected["meta"]["paper_pages"] = 34
    expected["evidence_strip"]["proven_theorems"] = {
        "value": "4 theorems",
        "sub": "8 numbered statements; stated assumptions apply",
    }
    assert actual == expected


def test_failed_metadata_refresh_does_not_overwrite_snapshot(monkeypatch, tmp_path):
    path = tmp_path / "snapshot.json"
    original = json.dumps(
        {
            "meta": {"paper": dashboard.rel(dashboard.SHORT_PDF), "paper_pages": 22},
            "evidence_strip": {"proven_theorems": {"value": "old"}},
        }
    ).encode()
    path.write_bytes(original)
    monkeypatch.setattr(dashboard, "OUT", path)

    def fail():
        raise FileNotFoundError("built PDF missing")

    monkeypatch.setattr(dashboard, "presentation_metadata", fail)
    with pytest.raises(FileNotFoundError, match="built PDF missing"):
        dashboard.main(["--metadata-only"])
    assert path.read_bytes() == original


@pytest.fixture
def dashboard_build_replica(tmp_path):
    """Real shell control flow with synthetic, explicitly pinned external tools."""
    repo = tmp_path / "synthetic repository"
    paper = repo / "docs/research/kbound"
    scripts = paper / "scripts"
    scripts.mkdir(parents=True)
    script = scripts / "build_pdfs.sh"
    script.write_bytes((dashboard.KBOUND / "scripts/build_pdfs.sh").read_bytes())
    scratch = tmp_path / "build scratch"
    scratch.mkdir()
    tools = tmp_path / "tools"
    tools.mkdir()
    events = tmp_path / "events.jsonl"
    tool_source = r"""
import json, os, pathlib, sys
role = pathlib.Path(sys.argv[0]).name
paper = pathlib.Path(os.environ["DASHBOARD_BUILD_PAPER"])
with open(os.environ["DASHBOARD_BUILD_EVENTS"], "a", encoding="utf-8") as log:
    log.write(json.dumps({"role": role, "args": sys.argv[1:]}) + "\n")
if role == "python":
    if pathlib.Path(sys.argv[1]).name == "build_dashboard_snapshot.py":
        assert (paper / "kbound_short_final_draft.pdf").read_bytes() == b"%PDF-1.7\nsynthetic fresh PDF\n"
elif role == "perl":
    assert pathlib.Path(sys.argv[1]).name == "latexmk"
    output = pathlib.Path(next(arg.split("=", 1)[1] for arg in sys.argv if arg.startswith("-outdir=")))
    assert pathlib.Path(os.environ["TMPDIR"]) in output.parents
    job = next(arg.split("=", 1)[1] for arg in sys.argv if arg.startswith("-jobname="))
    (output / (job + ".pdf")).write_bytes(b"%PDF-1.7\nsynthetic fresh PDF\n")
    (output / (job + ".log")).write_text("synthetic TeX log\n")
    print("synthetic compiler diagnostics")
    if os.environ.get("DASHBOARD_BUILD_FAIL") == "compile":
        raise SystemExit(17)
elif role == "pdfinfo":
    source = pathlib.Path(sys.argv[-1])
    assert pathlib.Path(os.environ["TMPDIR"]) in source.parents
    assert source.read_bytes() == b"%PDF-1.7\nsynthetic fresh PDF\n"
    if os.environ.get("DASHBOARD_BUILD_FAIL") == "validate":
        raise SystemExit(19)
    print("Pages: 2")
else:
    raise AssertionError("Unexpected external executable: " + role)
"""
    for name in ("python", "perl", "latexmk", "pdflatex", "pdfinfo"):
        tool = tools / name
        tool.write_text(f"#!{sys.executable}\n" + tool_source, encoding="utf-8")
        tool.chmod(0o755)
    env = {
        key: value
        for key, value in os.environ.items()
        if not key.startswith(("BUILD_", "KBOUND_TOOL_", "DASHBOARD_BUILD_"))
    }
    env.update(
        {
            "PYTHON": str(tools / "python"),
            "TMPDIR": str(scratch),
            "SOURCE_SNAPSHOT_COMMIT": "0" * 12,
            "KBOUND_AUTHORIZE_PROTECTED_SO2SAT": "0",
            "DASHBOARD_BUILD_EVENTS": str(events),
            "DASHBOARD_BUILD_PAPER": str(paper),
            **{f"KBOUND_TOOL_{name.upper()}": str(tools / name) for name in ("perl", "latexmk", "pdflatex", "pdfinfo")},
        }
    )
    return SimpleNamespace(repo=repo, paper=paper, script=script, env=env, events=events)


def _run_dashboard_build(replica, **overrides):
    result = subprocess.run(
        ["/bin/bash", str(replica.script)],
        cwd=replica.repo,
        env={**replica.env, **overrides},
        capture_output=True,
        text=True,
        timeout=15,
    )
    events = [json.loads(line) for line in replica.events.read_text().splitlines()] if replica.events.exists() else []
    return result, events


def test_pdf_build_refreshes_only_presentation_metadata_after_compilation(dashboard_build_replica):
    result, events = _run_dashboard_build(dashboard_build_replica)
    assert result.returncode == 0, result.stdout + result.stderr
    compilation = [i for i, event in enumerate(events) if event["role"] == "perl"]
    validation = [i for i, event in enumerate(events) if event["role"] == "pdfinfo"]
    refresh = [
        i
        for i, event in enumerate(events)
        if event["role"] == "python" and Path(event["args"][0]).name == "build_dashboard_snapshot.py"
    ]
    assert len(compilation) == len(validation) == len(refresh) == 1
    assert compilation[0] < validation[0] < refresh[0] == len(events) - 1
    assert events[refresh[0]]["args"] == ["scripts/build_dashboard_snapshot.py", "--metadata-only"]
    assert not any("so2sat" in str(event).lower() for event in events)


def test_missing_pdfinfo_stops_build_before_scientific_or_publication_activity(dashboard_build_replica):
    replica = dashboard_build_replica
    output = replica.paper / "kbound_short_final_draft.pdf"
    output.write_bytes(b"preserve old synthetic PDF\n")
    result, events = _run_dashboard_build(replica, KBOUND_TOOL_PDFINFO=str(replica.repo / "absent-pdfinfo"))
    assert result.returncode != 0
    assert "KBOUND_TOOL_PDFINFO" in result.stderr
    assert events == []
    assert output.read_bytes() == b"preserve old synthetic PDF\n"


@pytest.mark.parametrize("failure", ["compile", "validate"])
def test_failed_pdf_build_never_refreshes_dashboard_or_replaces_pdf(dashboard_build_replica, failure):
    replica = dashboard_build_replica
    output = replica.paper / "kbound_short_final_draft.pdf"
    output.write_bytes(b"preserve old synthetic PDF\n")
    result, events = _run_dashboard_build(replica, DASHBOARD_BUILD_FAIL=failure)
    assert result.returncode != 0
    assert sum(event["role"] == "perl" for event in events) == 1
    assert sum(event["role"] == "pdfinfo" for event in events) == (failure == "validate")
    assert not any(
        event["role"] == "python" and Path(event["args"][0]).name == "build_dashboard_snapshot.py" for event in events
    )
    assert output.read_bytes() == b"preserve old synthetic PDF\n"


def _saved_edge() -> dict:
    return {
        "study_status": "pending",
        "study_label": "Saved physical-study status",
        # Synthetic IDs fixed independently from the production validator.
        "phases": [
            {"id": phase, "label": label, "status": "pending", "detail": "Saved only", "artifact": f"edge/{phase}.json"}
            for phase, label in (
                ("protocol", "Protocol"),
                ("capture", "Capture"),
                ("source", "Source"),
                ("heldout", "Held-out"),
                ("replication", "Replication"),
            )
        ],
        "session_progress": [
            {"session": session, "expected_clips": 10, "captured_clips": 0, "complete": False}
            for session in ("S01", "S02", "S03", "S04", "S05", "S06", "S07", "S08", "S09", "S10")
        ],
        "development_metrics": None,
        "unblock": {
            "all_pass": False,
            "gate_thresholds": {"balanced_acc": 0.8, "macro_f1": 0.8},
            "current": dict.fromkeys(("sessions_complete", "physical_only", "source_gate", "audit_pass"), False),
            "gaps": [{"check": "Saved gate", "passed": False, "detail": "No fresh inspection"}],
            "commands": {"refresh_dashboard": "full physical-check command"},
        },
        "protocol_hash": None,
        "audit_pass": False,
        "extra_preserved_field": {"old_recorded_at": "2020-01-01", "numbers": [0.0, 17], "missing": None},
    }


def test_complete_saved_pending_fixture_is_valid_without_inspecting_edge(monkeypatch):
    snapshot = {
        "meta": {"paper": dashboard.rel(dashboard.SHORT_PDF)},
        "edge_validation": _saved_edge(),
    }
    original = copy.deepcopy(snapshot)

    def prohibited(*args, **kwargs):
        raise AssertionError("cached-edge validation must not inspect physical inputs")

    monkeypatch.setattr(dashboard, "edge_status", prohibited)
    monkeypatch.setattr(dashboard, "session_progress", prohibited)
    monkeypatch.setattr(dashboard, "load", prohibited)
    assert dashboard.validated_saved_edge(snapshot) == original["edge_validation"]
    assert snapshot == original


@pytest.mark.parametrize("identity_kind", ["phase", "session"])
@pytest.mark.parametrize("problem", ["missing", "duplicate", "unexpected"])
def test_incomplete_or_ambiguous_saved_identity_fails_before_paper_rebuild(monkeypatch, identity_kind, problem):
    snapshot = {
        "meta": {"paper": dashboard.rel(dashboard.SHORT_PDF)},
        "edge_validation": _saved_edge(),
    }
    field, identity = ("phases", "id") if identity_kind == "phase" else ("session_progress", "session")
    rows = snapshot["edge_validation"][field]
    if problem == "missing":
        rows.pop()
    elif problem == "duplicate":
        rows[-1][identity] = rows[0][identity]
    else:
        rows[-1][identity] = "unexpected"
    original = copy.deepcopy(snapshot)

    def prohibited(*args, **kwargs):
        raise AssertionError("invalid saved identities must fail before paper/edge inspection")

    monkeypatch.setattr(dashboard, "build_paper_projection", prohibited)
    monkeypatch.setattr(dashboard, "edge_status", prohibited)
    with pytest.raises(ValueError, match=f"invalid {identity_kind} identities"):
        dashboard.refresh_paper_snapshot(snapshot)
    assert snapshot == original


def _write_json(path: Path, value: dict) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    content = (json.dumps(value, indent=2, allow_nan=False) + "\n").encode()
    path.write_bytes(content)
    return hashlib.sha256(content).hexdigest()


@pytest.fixture
def paper_refresh_inputs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> SimpleNamespace:
    """Resident synthetic authorities only; retain the real paper projection."""
    kb = tmp_path / "docs/research/kbound"
    results = tmp_path / "experiments/kbound/results/reconciled_panels_v1"
    paths = {
        "MANIFEST": kb / "paper/generated/kbound_result_manifest.json",
        "CANONICAL_PANEL": results / "canonical_panel_results.json",
        "CURRENT_POLICY": results / "current_policy_cluster_inference.json",
        "FORMAL_REGISTRY": kb / "formal/formal_audit.py",
        "SHORT_PDF": kb / "kbound_short_final_draft.pdf",
        "OUT": kb / "dashboard/data/snapshot.json",
    }
    monkeypatch.setattr(dashboard, "REPO", tmp_path)
    monkeypatch.setattr(dashboard, "KBOUND", kb)
    monkeypatch.setattr(dashboard, "EDGE", kb / "edge")
    monkeypatch.setattr(dashboard, "EDGE_RESULTS", tmp_path / "experiments/kbound/results/edge_real_phone_v1")
    for name, path in paths.items():
        path.parent.mkdir(parents=True, exist_ok=True)
        monkeypatch.setattr(dashboard, name, path)
    paths["SHORT_PDF"].write_bytes(b"%PDF-1.7\nsynthetic built paper\n")
    theory = kb / "synthetic_theory.tex"
    theory.write_text("\\begin{theorem}\n\\begin{lemma}\n", encoding="utf-8")
    monkeypatch.setattr(dashboard, "THEORY_SOURCES", (theory,))
    registry = (
        f"LEGACY_CORE_THEOREMS = {[f'core_{i}' for i in range(65)]!r}\n"
        f"FOUNDATION_THEOREMS = {{'Synthetic': {[f'foundation_{i}' for i in range(77)]!r}}}\n"
        "FOUNDATION_LAYERS: list[dict[str, str]] = "
        + repr([{"status": "MECHANIZED_WITH_EXPLICIT_ASSUMPTIONS"}] * 5 + [{"status": "PARTIAL_COUNTEREXAMPLE_FOUND"}])
        + "\n"
    )
    paths["FORMAL_REGISTRY"].write_text(registry, encoding="utf-8")
    monkeypatch.setattr(dashboard.shutil, "which", lambda name: "/synthetic/pdfinfo")
    monkeypatch.setattr(
        dashboard.subprocess, "run", lambda *args, **kwargs: subprocess.CompletedProcess(args, 0, "Pages: 9\n", "")
    )

    backbones = {f"backbone_{i}": {} for i in range(10)}
    canonical = {
        "panels": {
            "imagenet_r": {"panel": {"seeds": [0, 1, 2, 3], "candidate_count": 10, "candidates": backbones}},
            "pacs": {
                "seeds": [0, 1, 2],
                "aggregate_matches_seed_files": True,
                "decision_replay_available": False,
                "decision_replay_blocker": "Saved cells lack predictions and calibration residuals.",
            },
        }
    }
    canonical_sha = _write_json(paths["CANONICAL_PANEL"], canonical)
    policy_sha = _write_json(
        paths["CURRENT_POLICY"], {"schema": "synthetic-current-policy-authority", "diagnostic": True}
    )
    track_names = (
        "cifar10c_tent",
        "cifar10c_eata",
        "imagenetc_sar",
        "three_source_oof",
        "officehome_M_v2",
        "rxrx1_J",
        "cifar10_1_K",
        "camelyon17_ood",
    )
    tracks = {
        name: {
            "regret": [0.1, 0.2, 0.3],
            "false_adapt_unconditional": 0.0,
            "point_beats_both": True,
            "ci_robust_beats_both": False,
        }
        for name in track_names
    }
    tracks["imagenetc_sar"]["verdict"] = "Synthetic opened SAR diagnostic; no prospective claim."
    tracks["three_source_oof"].update(
        {
            "claim_id": "KB-CLAIM-024",
            "claim_status": "diagnostic",
            "status": "historical_policy_only",
            "verdict": (
                "Historical researcher-constructed routing aggregate; "
                "rerun required under reconciled per-track decisions."
            ),
            "policy_synchronized": False,
            "current_policy_authority": False,
            "numeric_release_eligible": False,
            "headline_promotion_eligible": False,
            "release_eligible_win": False,
            "point_beats_both": False,
            "ci_robust_beats_both": False,
        }
    )
    tracks["imagenet_r_D"] = {"completed_seeds": [0, 1, 2, 3], "per_backbone": copy.deepcopy(backbones)}
    tracks["pacs"] = {"completed_seeds": 3, "decision_replay_available": False}
    manifest = {
        "regenerated_utc": "2026-08-31",
        "tracks": tracks,
        "current_policy_family_sensitivity": {
            "candidates": {
                "tent": {
                    "comparisons": {
                        "always_adapt": {"p_value_retrospective_holm_six_prospectively_named_contrasts": 0.25},
                        "always_freeze": {"p_value_retrospective_holm_six_prospectively_named_contrasts": 0.5},
                    }
                }
            }
        },
        "reconciliation_source": {
            "canonical_panel": dashboard.rel(paths["CANONICAL_PANEL"]),
            "canonical_panel_sha256": canonical_sha,
            "current_policy_family_sensitivity": {
                "artifact": dashboard.rel(paths["CURRENT_POLICY"]),
                "artifact_sha256": policy_sha,
            },
        },
    }
    _write_json(paths["MANIFEST"], manifest)
    original = {
        "meta": {
            "paper": dashboard.rel(paths["SHORT_PDF"]),
            "paper_pages": 2,
            "generated_at": "2020-01-01T00:00:00Z",
            "canonical_panel_sha256": "stale",
            "current_policy_sha256": "stale",
        },
        "evidence_strip": {"proven_theorems": {"value": "stale", "sub": "stale"}},
        "evidence_board": {"stale_values": True},
        "edge_validation": _saved_edge(),
        "provenance": {"legacy_marker": "not a fresh edge check"},
    }
    _write_json(paths["OUT"], original)
    return SimpleNamespace(
        paths=paths,
        theory=theory,
        manifest=manifest,
        canonical=canonical,
        original=original,
        allowed_reads={*paths.values(), theory},
    )


def test_paper_only_rebuilds_current_paper_authorities_and_preserves_edge_without_reads(
    paper_refresh_inputs,
    monkeypatch,
):
    fixture = paper_refresh_inputs
    read_paths = []
    original_open = Path.open
    original_stat = Path.stat

    def guarded_open(path, *args, **kwargs):
        assert path in fixture.allowed_reads, f"unexpected input read: {path}"
        read_paths.append(path)
        return original_open(path, *args, **kwargs)

    def guarded_stat(path, *args, **kwargs):
        assert path != dashboard.EDGE and dashboard.EDGE not in path.parents
        assert path != dashboard.EDGE_RESULTS and dashboard.EDGE_RESULTS not in path.parents
        return original_stat(path, *args, **kwargs)

    def prohibited(*args, **kwargs):
        raise AssertionError("paper-only must not inspect edge, raw data, or directory trees")

    with monkeypatch.context() as guard:
        guard.setattr(Path, "open", guarded_open)
        guard.setattr(Path, "stat", guarded_stat)
        guard.setattr(Path, "glob", prohibited)
        guard.setattr(Path, "rglob", prohibited)
        guard.setattr(os, "scandir", prohibited)
        guard.setattr(dashboard, "edge_status", prohibited)
        guard.setattr(dashboard, "session_progress", prohibited)
        guard.setattr(dashboard, "load", prohibited)
        assert dashboard.main(["--paper-only"]) == 0
    actual = json.loads(fixture.paths["OUT"].read_text())
    assert actual["edge_validation"] == fixture.original["edge_validation"]
    assert fixture.original["meta"]["canonical_panel_sha256"] == "stale"
    assert actual["meta"]["paper_pages"] == 9
    assert (
        actual["meta"]["canonical_panel_sha256"]
        == hashlib.sha256(fixture.paths["CANONICAL_PANEL"].read_bytes()).hexdigest()
    )
    assert (
        actual["meta"]["current_policy_sha256"]
        == hashlib.sha256(fixture.paths["CURRENT_POLICY"].read_bytes()).hexdigest()
    )
    assert actual["provenance"]["manifest_sha256"] == hashlib.sha256(fixture.paths["MANIFEST"].read_bytes()).hexdigest()
    assert actual["research_status"]["edge_study"] == "pending"
    for key in ("meta", "provenance"):
        assert actual[key]["refresh_mode"] == "paper-only"
    edge_refresh = actual["provenance"]["edge_validation_refresh"]
    assert edge_refresh["checked_this_run"] is False
    assert edge_refresh["mode"] == "preserved_not_rechecked"
    assert edge_refresh["source_snapshot_generated_at"] == "2020-01-01T00:00:00Z"
    assert (
        edge_refresh["preserved_edge_canonical_json_sha256"]
        == hashlib.sha256(
            json.dumps(fixture.original["edge_validation"], sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
    )
    assert fixture.paths["MANIFEST"] in read_paths and fixture.paths["CANONICAL_PANEL"] in read_paths
    assert fixture.paths["CURRENT_POLICY"] in read_paths and fixture.paths["FORMAL_REGISTRY"] in read_paths
    boundary = {row["name"]: row for row in actual["evidence_board"]["boundary_negative"]}
    assert boundary["ImageNet-R Protocol D"]["completed_seed_count"] == 4
    assert boundary["ImageNet-R Protocol D"]["backbone_count"] == 10
    assert boundary["ImageNet-R Protocol D"]["status"] == "diagnostic"
    assert boundary["PACS"]["completed_seed_count"] == 3
    assert boundary["PACS"]["status"] == "diagnostic"
    assert boundary["PACS"]["decision_replay_available"] is False
    assert all(row["beats_both_artifact"] is False for row in actual["headline_controlled"])
    scope = actual["provenance"]["formal_scope"]
    assert (scope["registered_lean_checks"], scope["legacy_core_checks"], scope["foundational_checks"]) == (142, 65, 77)
    assert scope["positive_foundational_layers"] == 5 and scope["counterexample_layers"] == 1
    assert scope["full_foundations_proof"] is False
    assert "142 registered Lean" in actual["evidence_strip"]["theorem_validators"]["value"]
    assert "counterexample" in actual["evidence_strip"]["open_theory"]["sub"]


@pytest.mark.parametrize("binding", ["canonical", "current-policy"])
def test_paper_only_rejects_stale_authority_hash_without_overwriting_snapshot(paper_refresh_inputs, binding):
    fixture = paper_refresh_inputs
    original = fixture.paths["OUT"].read_bytes()
    manifest = copy.deepcopy(fixture.manifest)
    if binding == "canonical":
        manifest["reconciliation_source"]["canonical_panel_sha256"] = "0" * 64
    else:
        manifest["reconciliation_source"]["current_policy_family_sensitivity"]["artifact_sha256"] = "0" * 64
    _write_json(fixture.paths["MANIFEST"], manifest)
    with pytest.raises(ValueError, match="stale"):
        dashboard.main(["--paper-only"])
    assert fixture.paths["OUT"].read_bytes() == original


@pytest.mark.parametrize("binding", ["canonical", "current-policy"])
def test_paper_only_does_not_follow_manifest_paths_to_edge_or_raw_inputs(paper_refresh_inputs, monkeypatch, binding):
    fixture = paper_refresh_inputs
    manifest = copy.deepcopy(fixture.manifest)
    original = fixture.paths["OUT"].read_bytes()
    if binding == "canonical":
        manifest["reconciliation_source"]["canonical_panel"] = dashboard.rel(dashboard.EDGE / "raw/forbidden.json")
    else:
        manifest["reconciliation_source"]["current_policy_family_sensitivity"]["artifact"] = dashboard.rel(
            dashboard.EDGE_RESULTS / "forbidden.json"
        )
    _write_json(fixture.paths["MANIFEST"], manifest)
    with pytest.raises(ValueError, match="unexpected"):
        dashboard.main(["--paper-only"])
    assert fixture.paths["OUT"].read_bytes() == original


@pytest.mark.parametrize(
    "name", ["OUT", "MANIFEST", "CANONICAL_PANEL", "CURRENT_POLICY", "FORMAL_REGISTRY", "SHORT_PDF"]
)
def test_paper_only_rejects_dataless_input_before_content_read(paper_refresh_inputs, monkeypatch, name):
    fixture = paper_refresh_inputs
    target = fixture.paths[name]
    original_snapshot = fixture.paths["OUT"].read_bytes()
    original_stat = Path.stat
    original_open = Path.open

    def cloud_stat(path, *args, **kwargs):
        info = original_stat(path, *args, **kwargs)
        if path == target:
            return SimpleNamespace(st_mode=info.st_mode, st_size=info.st_size, st_blocks=8, st_flags=0x40000000)
        return info

    def guarded_open(path, *args, **kwargs):
        assert path != target, "dataless authority content must never be opened"
        return original_open(path, *args, **kwargs)

    with monkeypatch.context() as guard:
        guard.setattr(Path, "stat", cloud_stat)
        guard.setattr(Path, "open", guarded_open)
        with pytest.raises(ValueError, match="not locally resident"):
            dashboard.main(["--paper-only"])
    assert fixture.paths["OUT"].read_bytes() == original_snapshot


@pytest.mark.parametrize(
    "problem",
    [
        "missing-meta",
        "wrong-paper",
        "missing-edge",
        "empty-edge",
        "status-list",
        "status-dict",
        "phase-shape",
        "progress-shape",
        "gate-shape",
        "contradictory-status",
        "nonfinite",
    ],
)
def test_paper_only_rejects_invalid_saved_identity_or_edge_before_rebuilding(
    paper_refresh_inputs, monkeypatch, problem
):
    snapshot = copy.deepcopy(paper_refresh_inputs.original)
    if problem == "missing-meta":
        snapshot.pop("meta")
    elif problem == "wrong-paper":
        snapshot["meta"]["paper"] = "another.pdf"
    elif problem == "missing-edge":
        snapshot.pop("edge_validation")
    elif problem == "empty-edge":
        snapshot["edge_validation"] = {}
    elif problem == "status-list":
        snapshot["edge_validation"]["study_status"] = ["pending"]
    elif problem == "status-dict":
        snapshot["edge_validation"]["study_status"] = {"value": "pending"}
    elif problem == "phase-shape":
        snapshot["edge_validation"]["phases"] = "unknown"
    elif problem == "progress-shape":
        snapshot["edge_validation"]["session_progress"] = {}
    elif problem == "gate-shape":
        snapshot["edge_validation"]["unblock"]["all_pass"] = "true"
    elif problem == "contradictory-status":
        snapshot["edge_validation"]["study_status"] = "verified"
    elif problem == "nonfinite":
        snapshot["edge_validation"]["development_metrics"] = {"latency": float("nan")}

    def prohibited():
        raise AssertionError("invalid existing snapshot must fail before paper/edge reads")

    monkeypatch.setattr(dashboard, "build_paper_projection", prohibited)
    with pytest.raises(ValueError):
        dashboard.refresh_paper_snapshot(snapshot)


@pytest.mark.parametrize(
    "problem", ["imagenet-seeds", "imagenet-backbones", "pacs-seeds", "pacs-replay", "pacs-aggregate"]
)
def test_paper_only_rejects_manifest_canonical_scope_conflicts(paper_refresh_inputs, problem):
    fixture = paper_refresh_inputs
    original_snapshot = fixture.paths["OUT"].read_bytes()
    manifest = copy.deepcopy(fixture.manifest)
    canonical = copy.deepcopy(fixture.canonical)
    if problem == "imagenet-seeds":
        manifest["tracks"]["imagenet_r_D"]["completed_seeds"] = [0, 1, 2]
    elif problem == "imagenet-backbones":
        canonical["panels"]["imagenet_r"]["panel"]["candidate_count"] = 9
    elif problem == "pacs-seeds":
        manifest["tracks"]["pacs"]["completed_seeds"] = 1
    elif problem == "pacs-replay":
        manifest["tracks"]["pacs"]["decision_replay_available"] = True
    elif problem == "pacs-aggregate":
        canonical["panels"]["pacs"]["aggregate_matches_seed_files"] = False
    manifest["reconciliation_source"]["canonical_panel_sha256"] = _write_json(
        fixture.paths["CANONICAL_PANEL"], canonical
    )
    _write_json(fixture.paths["MANIFEST"], manifest)
    with pytest.raises(ValueError, match="inconsistent"):
        dashboard.main(["--paper-only"])
    assert fixture.paths["OUT"].read_bytes() == original_snapshot


def test_default_full_snapshot_still_reads_edge_authorities_and_sessions(paper_refresh_inputs, monkeypatch):
    # A full refresh now requires an authenticated lock, even while capture is
    # pending. Keep this synthetic fixture independent of validator helpers.
    lock_path = dashboard.EDGE / "artifacts_real/protocol_lock.json"
    sidecar_path = dashboard.EDGE / "artifacts_real/protocol_lock.sha256"
    lock = {
        "protocol": "edge_real_phone_v1",
        "schema_version": "kbound-edge-v1",
        "seed": 0,
        "num_classes": 4,
        "classes": ["ok", "missing_label", "misaligned_label", "damaged_label"],
        "image_size": 224,
        "window_size": 32,
        "alpha": 0.10,
        "objects": {
            "source_train_calib": ["P01", "P02", "P03", "P04", "P05", "P06"],
            "source_val_calib": ["P07", "P08"],
            "held_out_replication": ["P09", "P10"],
        },
        "phones": {"phone_a": {}, "phone_b": {}},
        "shifts": {
            "A_sessions": ["mild_light", "side_shadow", "new_background", "glare"],
            "B_sessions": ["motion_blur", "viewpoint_45", "distance_scale", "batch_composition"],
        },
        "sessions": {
            session: {"split": split, "phone_id": phone, "objects": objects, "windows": windows}
            for session, split, phone, objects, windows in (
                ("S01", "source_train", "phone_a", ["P01", "P02", "P03", "P04", "P05", "P06"], 120),
                ("S02", "source_val", "phone_a", ["P07", "P08"], 40),
                ("S03", "calibration_fit_a", "phone_a", ["P01", "P02", "P03", "P04"], 64),
                ("S04", "calibration_fit_b", "phone_a", ["P01", "P02", "P03", "P04"], 80),
                ("S05", "calibration_conformal_a", "phone_a", ["P01", "P02", "P03", "P04"], 64),
                ("S06", "calibration_conformal_b", "phone_a", ["P01", "P02", "P03", "P04"], 80),
                ("S07", "heldout_a", "phone_a", ["P09", "P10"], 64),
                ("S08", "heldout_b", "phone_a", ["P09", "P10"], 80),
                ("S09", "replication_a", "phone_b", ["P09", "P10"], 64),
                ("S10", "replication_b", "phone_b", ["P09", "P10"], 80),
            )
        },
        "paths": {
            "artifacts": "artifacts_real",
            "protocol_lock": "artifacts_real/protocol_lock.json",
            "protocol_lock_sha": "artifacts_real/protocol_lock.sha256",
        },
    }
    _write_json(lock_path, lock)
    lock_hash = hashlib.sha256(json.dumps(lock, sort_keys=True).encode()).hexdigest()
    sidecar_path.write_text(lock_hash + "\n", encoding="ascii")
    edge_reads = []
    session_calls = []
    lock_reads = []
    original_open = Path.open

    def record_open(path, *args, **kwargs):
        if path in {lock_path, sidecar_path}:
            lock_reads.append(path)
        return original_open(path, *args, **kwargs)

    def edge_load(path):
        assert dashboard.EDGE in path.parents or dashboard.EDGE_RESULTS in path.parents
        edge_reads.append(path)
        return None

    def sessions():
        session_calls.append(True)
        return copy.deepcopy(paper_refresh_inputs.original["edge_validation"]["session_progress"])

    monkeypatch.setattr(dashboard, "load", edge_load)
    monkeypatch.setattr(dashboard, "session_progress", sessions)
    monkeypatch.setattr(Path, "open", record_open)
    assert dashboard.main([]) == 0
    actual = json.loads(paper_refresh_inputs.paths["OUT"].read_text())
    assert lock_reads == [lock_path, sidecar_path]
    assert edge_reads == [
        dashboard.EDGE_RESULTS / name
        for name in (
            "model_card.json",
            "heldout_metrics.json",
            "replication_metrics.json",
            "anti_leakage_audit.json",
            "recording_inventory.json",
            "publication_gate.json",
        )
    ]
    assert len(session_calls) == 1
    assert actual["provenance"]["refresh_mode"] == "full"
    assert actual["provenance"]["edge_validation_refresh"]["checked_this_run"] is True
    assert actual["edge_validation"]["study_status"] == "pending"
    assert actual["edge_validation"]["protocol_hash"] == lock_hash
    assert actual["edge_validation"]["phases"][0]["status"] == "verified"


def test_refresh_modes_are_exclusive_and_do_not_write(paper_refresh_inputs):
    original = paper_refresh_inputs.paths["OUT"].read_bytes()
    with pytest.raises(SystemExit) as raised:
        dashboard.main(["--paper-only", "--metadata-only"])
    assert raised.value.code == 2
    assert paper_refresh_inputs.paths["OUT"].read_bytes() == original


def test_paper_only_requires_an_existing_resident_snapshot(paper_refresh_inputs, monkeypatch, tmp_path):
    output = tmp_path / "absent.json"
    monkeypatch.setattr(dashboard, "OUT", output)
    with pytest.raises(FileNotFoundError, match="missing"):
        dashboard.main(["--paper-only"])
    assert not output.exists()


def test_repeated_paper_refresh_keeps_edge_unchecked_and_identical(paper_refresh_inputs):
    original = copy.deepcopy(paper_refresh_inputs.original)
    once = dashboard.refresh_paper_snapshot(original)
    twice = dashboard.refresh_paper_snapshot(once)
    assert original == paper_refresh_inputs.original
    assert twice["edge_validation"] == once["edge_validation"] == original["edge_validation"]
    assert twice["provenance"]["edge_validation_refresh"]["checked_this_run"] is False
    assert twice["provenance"]["edge_validation_refresh"]["mode"] == "preserved_not_rechecked"


def test_current_registered_formal_scope_is_not_a_full_six_layer_proof():
    scope = dashboard.registered_formal_scope()
    # Eight reviewed audit-floor capstones extend the registered inventory;
    # this remains a source inventory, not proof of the empirical premises.
    assert scope["registered_lean_checks"] == 150
    assert scope["legacy_core_checks"] == 65 and scope["foundational_checks"] == 85
    assert scope["positive_foundational_layers"] == 5 and scope["counterexample_layers"] == 1
    assert scope["full_foundations_proof"] is False
