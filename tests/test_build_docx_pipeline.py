from __future__ import annotations

import hashlib
import importlib.util
from pathlib import Path

import pytest
from docx import Document

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "docs/research/kbound/scripts/build_docx.py"
SPEC = importlib.util.spec_from_file_location("kbound_build_docx", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def test_generated_sources_follow_live_dependency_graph(tmp_path: Path) -> None:
    source = tmp_path / "paper.tex"
    generated = tmp_path / "paper/generated"
    base = generated / "kbound_numbers.tex"
    cct = generated / "cct20_numbers.tex"
    _write(base, r"\newcommand{\BaseMetric}{0.10}")
    _write(cct, r"\newcommand{\CCTMetric}{0.20}")
    _write(source, r"\input{paper/generated/kbound_numbers}")

    sources = MODULE.discover_generated_macro_sources(source, generated_dir=generated, search_root=tmp_path)
    assert sources == (base.resolve(),)
    assert MODULE.generated_macros(sources) == {"BaseMetric": "0.10"}

    _write(
        source,
        "\n".join(
            (
                r"\input{paper/generated/kbound_numbers}",
                r"\input{paper/generated/cct20_numbers}",
            )
        ),
    )
    sources = MODULE.discover_generated_macro_sources(source, generated_dir=generated, search_root=tmp_path)
    assert sources == (base.resolve(), cct.resolve())
    assert MODULE.generated_macros(sources) == {
        "BaseMetric": "0.10",
        "CCTMetric": "0.20",
    }


def test_generated_macros_parse_nested_values_and_reject_conflicts(tmp_path: Path) -> None:
    first = tmp_path / "base_numbers.tex"
    second = tmp_path / "cct20_numbers.tex"
    _write(first, r"\newcommand{\Verdict}{\textnormal{not defined}}")
    _write(second, r"\newcommand{\Verdict}{complete}")

    assert MODULE.generated_macros((first,)) == {"Verdict": r"\textnormal{not defined}"}
    with pytest.raises(RuntimeError, match="defined by both"):
        MODULE.generated_macros((first, second))
    with pytest.raises(RuntimeError, match="no generated macro sources"):
        MODULE.generated_macros(())


def test_generated_macros_resolve_transitive_values_and_reject_cycles(
    tmp_path: Path,
) -> None:
    source = tmp_path / "numbers.tex"
    _write(
        source,
        "\n".join(
            (
                r"\newcommand{\ReleaseSHA}{\CanonicalSHA}",
                r"\newcommand{\CanonicalSHA}{\ArtifactSHA}",
                rf"\newcommand{{\ArtifactSHA}}{{{'a' * 64}}}",
            )
        ),
    )
    assert MODULE.generated_macros((source,)) == {
        "ReleaseSHA": "a" * 64,
        "CanonicalSHA": "a" * 64,
        "ArtifactSHA": "a" * 64,
    }

    _write(
        source,
        "\n".join(
            (
                r"\newcommand{\First}{\Second}",
                r"\newcommand{\Second}{\First}",
            )
        ),
    )
    with pytest.raises(RuntimeError, match="cyclic generated macro definition"):
        MODULE.generated_macros((source,))


def test_generated_sources_are_discovered_through_nested_inputs(tmp_path: Path) -> None:
    source = tmp_path / "paper.tex"
    body = tmp_path / "sections/body.tex"
    generated = tmp_path / "paper/generated"
    numbers = generated / "cct20_numbers.tex"
    _write(source, r"\input{sections/body}")
    _write(body, r"\input{paper/generated/cct20_numbers}")
    _write(numbers, r"\newcommand{\CCTMetric}{0.20}")

    assert MODULE.discover_generated_macro_sources(
        source,
        generated_dir=generated,
        search_root=tmp_path,
    ) == (numbers.resolve(),)


def test_hash_requirements_are_derived_only_from_used_macros() -> None:
    base_sha = "a" * 64
    cct_sha = "b" * 64
    tex = (
        r"\documentclass{article}"
        "\n"
        r"\begin{document}\BaseSHA, \BaseSHA, \CCTExecutionSHA\end{document}"
    )
    requirements = MODULE.generated_value_requirements(
        tex,
        {
            "BaseSHA": base_sha,
            "CCTExecutionSHA": cct_sha,
            "UnusedPendingSHA": r"\textnormal{pending}",
        },
    )
    assert requirements == {base_sha: 2, cct_sha: 1}

    with pytest.raises(RuntimeError, match="not a complete SHA-256"):
        MODULE.generated_value_requirements(
            tex,
            {"BaseSHA": base_sha, "CCTExecutionSHA": "pending"},
        )


def test_generated_definitions_in_document_body_are_removed_before_expansion() -> None:
    digest = "c" * 64
    tex = "\n".join(
        (
            r"\documentclass{article}",
            r"\begin{document}",
            r"\newcommand{\CCTCheckpointCount}{5}",
            rf"\newcommand{{\CCTInferenceSHA}}{{{digest}}}",
            r"\newcommand{\LocalFormatting}[1]{\textbf{#1}}",
            r"There are \CCTCheckpointCount{} checkpoints (\CCTInferenceSHA).",
            r"\end{document}",
        )
    )
    macros = {"CCTCheckpointCount": "5", "CCTInferenceSHA": digest}

    requirements = MODULE.generated_value_requirements(tex, macros)
    expanded = MODULE.expand_generated_macros(tex, macros)

    assert requirements == {digest: 1}
    assert r"\newcommand{5}{5}" not in expanded
    assert r"\newcommand{\CCTCheckpointCount}{5}" not in expanded
    assert rf"\newcommand{{\CCTInferenceSHA}}{{{digest}}}" not in expanded
    assert r"\newcommand{\LocalFormatting}[1]{\textbf{#1}}" in expanded
    assert f"There are 5{{}} checkpoints ({digest})." in expanded


def test_split_bibliography_accepts_data_derived_55th_cct_reference() -> None:
    entries = "\n".join(rf"\bibitem{{key{index}}} Reference {index}." for index in range(1, 56))
    tex = (
        r"Citations \cite{key1,key55}."
        "\n"
        r"\begin{thebibliography}{99}"
        "\n" + entries + "\n" + r"\end{thebibliography}"
    )

    rendered, bibliography = MODULE.split_bibliography(tex)
    assert len(bibliography) == 55
    assert bibliography[-1] == ("key55", "Reference 55.")
    resolved = MODULE.resolve_citations(rendered, bibliography)
    assert "Citations [1, 55]." in resolved
    assert resolved.count(r"\item ") == 55


def test_split_bibliography_accepts_shared_author_year_wrapper() -> None:
    tex = "\n".join(
        (
            r"See \cite{first,second}.",
            r"\begin{thebibliography}{99}",
            r"\KBbibitem{first}{First Author}{2025} First reference.",
            r"\KBbibitem{second}{Second et al.}{2026} Second reference.",
            r"\end{thebibliography}",
        )
    )

    rendered, bibliography = MODULE.split_bibliography(tex)

    assert bibliography == [
        ("first", "First reference."),
        ("second", "Second reference."),
    ]
    assert MODULE.resolve_citations(rendered, bibliography).startswith("See [1, 2].")


def test_citation_resolution_handles_notes_and_fails_closed() -> None:
    entries = [("first", "First reference."), ("second", "Second reference.")]
    tex = r"See \citep[compare][p.~2]{second,first}."
    assert MODULE.resolve_citations(tex, entries) == "See [1, 2]."

    with pytest.raises(RuntimeError, match="unknown citation key"):
        MODULE.resolve_citations(r"See \cite{missing}.", entries)
    with pytest.raises(RuntimeError, match="unsupported or unresolved"):
        MODULE.resolve_citations(r"See \citeauthor{first}.", entries)


def test_split_bibliography_rejects_multiple_active_lists() -> None:
    bibliography = (
        r"\begin{thebibliography}{1}\bibitem{key} Reference."
        r"\end{thebibliography}"
    )
    with pytest.raises(RuntimeError, match="expected one active compact bibliography"):
        MODULE.split_bibliography(bibliography + bibliography)


def test_live_source_manifest_macro_matches_live_manifest_file() -> None:
    sources = MODULE.discover_generated_macro_sources()
    macros = MODULE.generated_macros(sources)
    manifest = ROOT / "experiments/kbound/results/reconciled_panels_v1/source_manifest.json"
    assert macros["SourceManifestSHA"] == hashlib.sha256(manifest.read_bytes()).hexdigest()


def test_small_tables_keep_rows_together_for_word_pagination() -> None:
    doc = Document()
    table = doc.add_table(rows=3, cols=2)

    MODULE.format_tables(doc)

    assert all(
        paragraph.paragraph_format.keep_with_next is True
        for row in table.rows[:-1]
        for cell in row.cells
        for paragraph in cell.paragraphs
    )
    assert all(
        paragraph.paragraph_format.keep_with_next is not True
        for cell in table.rows[-1].cells
        for paragraph in cell.paragraphs
    )
