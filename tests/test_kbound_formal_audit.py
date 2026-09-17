"""The release must distinguish real kernel checks from a static name inventory."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from docs.research.kbound.formal import formal_audit as audit


def test_registered_capstones_are_unique_and_imported() -> None:
    assert len(audit.LEGACY_CORE_THEOREMS) == 65
    assert len(audit.VERIFIED_THEOREMS) == len(set(audit.VERIFIED_THEOREMS))
    assert audit.theorem_map_checks() == []
    assert audit.scan_for_forbidden_tokens() == []
    entry = (audit.ROOT / "KBound.lean").read_text(encoding="utf-8")
    for module in audit.FOUNDATION_THEOREMS:
        assert f"import KBound.Probability.{module}\n" in entry
        assert (audit.ROOT / "KBound" / "Probability" / f"{module}.lean").is_file()


def test_five_foundations_do_not_silently_close_the_false_sixth_extension() -> None:
    assert len(audit.FOUNDATION_LAYERS) == 6
    assert [row["status"] for row in audit.FOUNDATION_LAYERS].count(
        "MECHANIZED_WITH_EXPLICIT_ASSUMPTIONS"
    ) == 5
    assert audit.FOUNDATION_LAYERS[-1]["status"] == "PARTIAL_COUNTEREXAMPLE_FOUND"
    assert audit.OPEN_RESEARCH_FRONTIER
    assert "orbit-selection sufficiency" in audit.OPEN_RESEARCH_FRONTIER[0]["current"]
    assert "bool_decoder_iff_constant_on_fibres" in audit.VERIFIED_THEOREMS


def test_nested_comment_mask_preserves_lines_and_does_not_hide_active_hole() -> None:
    source = '/- outer\n /- sorry -/ admit\n-/\ntheorem bad : True := by sorry\n'
    masked = audit.strip_lean_comments(source)
    assert masked.count("\n") == source.count("\n")
    assert masked.splitlines()[3] == "theorem bad : True := by sorry"
    assert "sorry" not in "\n".join(masked.splitlines()[:3])


def test_string_and_line_comment_are_not_proof_commands() -> None:
    source = 'def s := "sorry \\\" admit /-" -- axiom\ntheorem ok : True := by trivial\n'
    masked = audit.strip_lean_comments(source)
    assert all(token not in masked for token in ("sorry", "admit", "axiom"))
    assert "theorem ok" in masked
    assert source.count("\n") == masked.count("\n")


@pytest.mark.parametrize("literal", ["'\"'", "'\\\"'"])
def test_double_quote_character_cannot_hide_the_next_declaration(literal: str) -> None:
    source = f"def quote : Char := {literal}\ntheorem probe : True := by sorry\n"
    assert "by sorry" in audit.strip_lean_comments(source)


@pytest.mark.parametrize("prefix", ["s!", "m!"])
def test_interpolated_lean_code_is_not_masked_as_literal_text(prefix: str) -> None:
    source = prefix + '"{(by /- harmless comment -/ sorry : Nat)}"'
    masked = audit.strip_lean_comments(source)
    assert "sorry" in masked
    assert "harmless" not in masked


def test_nested_interpolation_and_strings_preserve_active_code() -> None:
    source = 's!"{let s := "not code /-"; s!"{(by sorry : Nat)}"}"'
    assert "sorry" in audit.strip_lean_comments(source)


def test_axiom_audit_accepts_only_the_expected_standard_dependencies() -> None:
    output = (
        "'KBound.a' depends on axioms: [propext,\n Classical.choice, Quot.sound]\n"
        "'KBound.b' does not depend on any axioms\n"
    )
    result = audit.parse_axiom_audit(output, ["a", "b"])
    assert result["ok"]
    assert result["dependencies"]["KBound.b"] == []


@pytest.mark.parametrize("forbidden", ["sorryAx", "myCustomAxiom", "Lean.ofReduceBool"])
def test_axiom_audit_rejects_untrusted_transitive_dependency(forbidden: str) -> None:
    result = audit.parse_axiom_audit(
        f"'KBound.a' depends on axioms: [propext, {forbidden}]\n", ["a"]
    )
    assert not result["ok"]
    assert result["forbidden_dependencies"] == {"KBound.a": [forbidden]}


@pytest.mark.parametrize(
    "output",
    [
        "",
        "'KBound.a_longer' does not depend on any axioms\n",
        "'KBound.a' does not depend on any axioms\n" * 2,
        "'KBound.a' depends on axioms: unexpected format\n",
    ],
)
def test_missing_misnamed_duplicate_or_malformed_axiom_output_fails(output: str) -> None:
    assert not audit.parse_axiom_audit(output, ["a"])["ok"]


def test_static_scan_cannot_be_reported_as_kernel_verified(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    report = tmp_path / "static.json"
    monkeypatch.setattr(sys, "argv", ["formal_audit.py", "--json-out", str(report)])
    assert audit.main() == 0
    payload = json.loads(report.read_text())
    assert payload["status"] == "STATIC_PASS"
    assert payload["verified_theorem_count"] == 0
    assert payload["kernel_axiom_audit"] is None


def test_strict_core_requires_actual_build(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    report = tmp_path / "strict.json"
    monkeypatch.setattr(
        sys, "argv", ["formal_audit.py", "--strict-core", "--json-out", str(report)]
    )
    assert audit.main() == 1
    payload = json.loads(report.read_text())
    assert payload["status"] == "FAIL"
    assert "requires --build" in payload["strict_blockers"][0]


def test_successful_build_without_axiom_evidence_fails_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    report = tmp_path / "missing-axioms.json"
    monkeypatch.setattr(
        sys, "argv", ["formal_audit.py", "--build", "--json-out", str(report)]
    )
    monkeypatch.setattr(
        audit, "run", lambda cmd, **kwargs: subprocess.CompletedProcess(cmd, 0, "")
    )
    assert audit.main() == 1
    payload = json.loads(report.read_text())
    assert payload["build_ok"] is True
    assert payload["verified_theorem_count"] == 0
    assert payload["kernel_axiom_audit"]["missing_declarations"]


def test_failed_build_still_writes_failure_receipt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    report = tmp_path / "failed-build.json"
    monkeypatch.setattr(
        sys, "argv", ["formal_audit.py", "--build", "--json-out", str(report)]
    )
    monkeypatch.setattr(
        audit, "run", lambda cmd, **kwargs: subprocess.CompletedProcess(cmd, 1, "build failed")
    )
    assert audit.main() == 1
    assert json.loads(report.read_text())["build_ok"] is False


@pytest.mark.parametrize("quotes", [("'", "'"), ("`", "`"), ("‘", "’")])
def test_unregistered_lean_proof_hole_warning_fails_even_with_good_capstones(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, quotes: tuple[str, str]
) -> None:
    report = tmp_path / "unregistered-hole.json"
    monkeypatch.setattr(
        sys, "argv", ["formal_audit.py", "--build", "--json-out", str(report)]
    )
    monkeypatch.setattr(
        audit, "run", lambda cmd, **kwargs: subprocess.CompletedProcess(
            cmd, 0, f"warning: Extra.lean:2:0: declaration uses {quotes[0]}sorry{quotes[1]}\n"
        )
    )
    monkeypatch.setattr(audit, "inspect_kernel_axioms", lambda: {"ok": True})
    assert audit.main() == 1
    assert json.loads(report.read_text())["compiler_proof_hole_warnings"]
