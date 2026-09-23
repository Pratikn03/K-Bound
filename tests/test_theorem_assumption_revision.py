"""Correspondence checks must fail on omissions and stale source, not certify proofs."""

import copy
import hashlib

import pytest

from docs.research.kbound.scripts import check_theorem_assumption_revision as audit


def fixture_register(tmp_path):
    paper = tmp_path / "paper"
    paper.mkdir()
    (paper / "main.tex").write_text("% \\input{absent}\n\\input{body}\n")
    body = "\\begin{theorem}[Example]\\label{thm:a}\nA.\n\\end{theorem}\n\\begin{proof}By premise.\\end{proof}\n"
    (paper / "body.tex").write_text(body)
    inventory, _ = audit.inventory(tmp_path, ["paper/main.tex"])
    row = inventory[0]
    row.update(
        {
            "assumptions": ["Declared premise"],
            "formal_correspondence": {"status": "latex_only", "declarations": [], "boundary": "Not mechanized"},
            "manual_review": {"status": "reviewed", "edge_cases": [], "findings": []},
        }
    )
    register = {
        "scope": {
            "drivers": ["paper/main.tex"],
            "source_files": [
                {"path": p, "sha256": hashlib.sha256((tmp_path / p).read_bytes()).hexdigest()}
                for p in ("paper/main.tex", "paper/body.tex")
            ],
        },
        "statements": [row],
    }
    return register


def test_complete_inventory_is_correspondence_only(tmp_path):
    report = audit.check_register(tmp_path, fixture_register(tmp_path))
    assert report["status"] == "PASS"
    assert report["mathematical_correctness_established"] is False
    assert report["empirical_assumptions_established"] is False


def test_new_statement_is_not_silently_omitted(tmp_path):
    register = fixture_register(tmp_path)
    with (tmp_path / "paper/body.tex").open("a") as stream:
        stream.write("\\begin{lemma}\\label{lem:new}B.\\end{lemma}\\begin{proof}Proof.\\end{proof}")
    report = audit.check_register(tmp_path, register)
    assert report["status"] == "FAIL"
    assert any("unregistered" in error for error in report["errors"])


def test_proof_only_edit_invalidates_review_binding(tmp_path):
    register = fixture_register(tmp_path)
    with (tmp_path / "paper/body.tex").open("a") as stream:
        stream.write("\\begin{proof}Changed proof.\\end{proof}")
    report = audit.check_register(tmp_path, register)
    assert any("stale source" in error for error in report["errors"])


def test_duplicate_registration_rejected(tmp_path):
    register = fixture_register(tmp_path)
    register["statements"].append(copy.deepcopy(register["statements"][0]))
    assert any("duplicate" in error for error in audit.check_register(tmp_path, register)["errors"])


@pytest.mark.parametrize(
    "content",
    [
        "\\input{missing}",
        "\\input{\\dynamic}",
        "\\input{main}",
        "\\begin{lemma}No label.\\end{lemma}",
        "\\begin{lemma}\\label{x}No end.",
    ],
)
def test_unresolved_graph_or_unlabeled_statement_fails_closed(tmp_path, content):
    fixture_register(tmp_path)
    (tmp_path / "paper/body.tex").write_text(content)
    with pytest.raises(ValueError):
        audit.inventory(tmp_path, ["paper/main.tex"])


def test_changed_statement_rejected_even_when_file_hash_is_refreshed(tmp_path):
    register = fixture_register(tmp_path)
    path = tmp_path / "paper/body.tex"
    path.write_text(path.read_text().replace("A.", "False."))
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    register["scope"]["source_files"][1]["sha256"] = digest
    register["statements"][0]["source"]["sha256"] = digest
    report = audit.check_register(tmp_path, register)
    assert any("statement bytes" in error for error in report["errors"])


def test_escaped_percent_does_not_hide_input(tmp_path):
    fixture_register(tmp_path)
    (tmp_path / "paper/main.tex").write_text("10\\% \\input{body}\n")
    assert len(audit.inventory(tmp_path, ["paper/main.tex"])[0]) == 1


def test_real_comment_after_escaped_percent_is_masked(tmp_path):
    fixture_register(tmp_path)
    (tmp_path / "paper/main.tex").write_text("10\\% % \\input{missing}\n\\input{body}\n")
    assert len(audit.inventory(tmp_path, ["paper/main.tex"])[0]) == 1


def test_unbound_proof_reference_rejected(tmp_path):
    register = fixture_register(tmp_path)
    (tmp_path / "paper/other.tex").write_text("Proof.")
    register["statements"][0]["proof_authority"]["path"] = "paper/other.tex"
    assert any("unbound proof" in error for error in audit.check_register(tmp_path, register)["errors"])


def test_wrong_proof_hash_is_rejected(tmp_path):
    register = fixture_register(tmp_path)
    register["statements"][0]["proof_authority"]["proof_sha256"] = "0" * 64
    assert any("proof bytes" in error for error in audit.check_register(tmp_path, register)["errors"])


def test_title_and_count_metadata_cannot_drift(tmp_path):
    register = fixture_register(tmp_path)
    register["statements"][0]["title"] = "A different result"
    register["scope"]["count"] = 20
    errors = audit.check_register(tmp_path, register)["errors"]
    assert any("title" in error for error in errors)
    assert any("count" in error for error in errors)


def test_lean_references_require_byte_binding(tmp_path):
    register = fixture_register(tmp_path)
    (tmp_path / "Proof.lean").write_text("theorem example_name : True := by trivial\n")
    register["supporting_claims"] = [{"formal_declarations": [{"path": "Proof.lean", "name": "example_name"}]}]
    errors = audit.check_register(tmp_path, register)["errors"]
    assert any("unbound Lean" in error for error in errors)


def test_wrong_source_line_rejected(tmp_path):
    register = fixture_register(tmp_path)
    register["statements"][0]["proof_authority"]["line"] = 999999
    assert any("proof" in error for error in audit.check_register(tmp_path, register)["errors"])


def test_path_cannot_escape_repository(tmp_path):
    fixture_register(tmp_path)
    (tmp_path / "paper/main.tex").write_text("\\input{../../outside}")
    with pytest.raises(ValueError, match="outside"):
        audit.inventory(tmp_path, ["paper/main.tex"])


def test_assumption_links_and_hashes_are_checked(tmp_path):
    register = fixture_register(tmp_path)
    assumptions = {
        "assumptions": [
            {"id": "A1", "source_refs": [{"path": "paper/body.tex", "sha256": "0" * 64, "anchor": "missing premise"}]}
        ],
        "studies": [{"id": "S1", "assumption_ids": ["undefined"]}],
    }
    report = audit.check_register(tmp_path, register, assumptions)
    assert any("unknown assumption" in error for error in report["errors"])
    assert any("stale assumption reference" in error for error in report["errors"])
    assert any("missing assumption anchor" in error for error in report["errors"])
