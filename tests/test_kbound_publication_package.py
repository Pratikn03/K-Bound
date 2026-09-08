"""Synthetic package boundaries; fixtures do not call the production generator."""
from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import zipfile

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "docs/research/kbound/scripts"
SOURCE_PATHS = (
    "kbound_abstract_core.tex", "kbound_full_report.tex", "kbound_main_with_appendix.tex",
    "kbound_submission_body.tex", "kbound_submission_supplement.tex",
    "paper/full_report/evidence_protocol_atlas.tex", "paper/full_report/formal_reproducibility_atlas.tex",
    "paper/full_report/theory_exposition.tex", "paper/generated/empirical_evidence_numbers.tex",
    "paper/sections/officehome_mechanism_check.tex", "paper/sections/proof_traceability.tex",
    "paper/sections/theory_certificate.tex",
)
ROLES = {
    "main_pdf": ("KBound_Main_22pages.pdf", "short_main", "pdf"),
    "combined_pdf": ("KBound_Main_Appendices_49pages.pdf", "main_with_appendix", "pdf"),
    "full_pdf": ("KBound_Full_Report_105pages.pdf", "full_report", "pdf"),
    "main_docx": ("KBound_Main_Editable.docx", "short_main", "docx"),
    "combined_docx": ("KBound_Main_Appendices_Editable.docx", "main_with_appendix", "docx"),
}
W = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
M = "http://schemas.openxmlformats.org/officeDocument/2006/math"
R = "http://schemas.openxmlformats.org/package/2006/relationships"
REVISION = "KBOUND-REVISION-2026-09-08"


def digest(data):
    return hashlib.sha256(data).hexdigest()


@pytest.fixture
def auditor():
    path = SCRIPTS / "audit_publication_package.py"
    assert path.is_file(), "publication helper is not implemented"
    spec = importlib.util.spec_from_file_location("publication_under_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def write_docx(path, revision=REVISION, extra=None):
    parts = {
        "[Content_Types].xml": '<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types"><Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/><Default Extension="xml" ContentType="application/xml"/><Override PartName="/word/document.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml"/></Types>',
        "_rels/.rels": f'<Relationships xmlns="{R}"><Relationship Id="r1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="word/document.xml"/></Relationships>',
        "word/document.xml": f'<w:document xmlns:w="{W}" xmlns:m="{M}"><w:body><w:p><w:r><w:t>{revision} September 8, 2026 b1899ea20567 Pratik Niroula Named compact main paper Main paper with integrated appendices Named full technical report</w:t></w:r><m:oMath><m:r><m:t>x</m:t></m:r></m:oMath></w:p><w:tbl><w:tr><w:tc><w:p/></w:tc></w:tr></w:tbl></w:body></w:document>',
        "word/media/image1.png": b"\x89PNG\r\n\x1a\nsynthetic image",
    }
    parts.update(extra or {})
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as z:
        for name, data in parts.items():
            z.writestr(name, data)


@pytest.fixture
def package(tmp_path):
    repo = tmp_path / "repo"
    delivery = tmp_path / "delivery"
    delivery.mkdir()
    sources = {}
    for i, rel in enumerate(SOURCE_PATHS):
        name = "docs/research/kbound/" + rel
        p = repo / name
        p.parent.mkdir(parents=True, exist_ok=True)
        data = f"synthetic reviewed source {i}\n".encode()
        p.write_bytes(data)
        sources[name] = digest(data)
    authority = {
        "schema": "kbound_manuscript_revision_v1", "revision_id": REVISION,
        "revision_date": "September 8, 2026", "base_commit": "b1899ea20567",
        "canonical_panel_sha256": "1" * 64, "historical_evidence_manifest_sha256": "2" * 64,
        "status": "MANUSCRIPT_REVISION_NOT_A_NEW_SOURCE_SEAL",
        "documents": {
            "short_main": {"anonymous": False, "output_filename": "kbound_short_main.pdf", "role": "Named compact main paper"},
            "main_with_appendix": {"anonymous": False, "output_filename": "kbound_main_with_appendix.pdf", "role": "Main paper with integrated appendices"},
            "full_report": {"anonymous": False, "output_filename": "kbound_full_report.pdf", "role": "Named full technical report"},
            "short_supplement": {"anonymous": False, "output_filename": "kbound_short_supplement.pdf", "role": "Named standalone supplement"},
            "tmlr": {"anonymous": True, "output_filename": "kbound_tmlr.pdf", "role": "Anonymous integrated TMLR review manuscript"},
        },
    }
    authority_path = repo / "docs/research/kbound/paper/release/manuscript_revision.json"
    authority_path.parent.mkdir(parents=True)
    authority_path.write_text(json.dumps(authority))
    contract = {
        "schema": "kbound_publication_package_v1", "revision_id": REVISION,
        "scope": "READ_ONLY_PRE_RELEASE_PACKAGE_NOT_SOURCE_SEAL_OR_CI",
        "revision_authority": {"path": "docs/research/kbound/paper/release/manuscript_revision.json", "sha256": digest(authority_path.read_bytes())},
        "reviewed_manuscript_inputs": sources, "artifacts": {},
    }
    for role, (filename, document, kind) in ROLES.items():
        p = delivery / filename
        if kind == "pdf":
            p.write_bytes(b"%PDF-1.7\nsynthetic " + role.encode())
            counts = {"pages": 2}
        else:
            write_docx(p)
            counts = {"omml": 1, "tables": 1, "media": 1}
        contract["artifacts"][role] = {"filename": filename, "document": document, "kind": kind, "sha256": digest(p.read_bytes()), "bytes": p.stat().st_size, **counts}
    c = tmp_path / "contract.json"
    c.write_text(json.dumps(contract))
    return repo, delivery, c, contract


def pdf_tool(name, *args):
    if name == "pdftotext":
        return REVISION + " September 8, 2026 b1899ea20567 Pratik Niroula Named compact main paper Main paper with integrated appendices Named full technical report " + "1" * 64 + " " + "2" * 64
    if name == "pdfdetach":
        return "0 embedded files\n"
    if args[0] == "-js" or args[0] == "-meta":
        return ""
    return "Pages: 2\nEncrypted: no\nJavaScript: no\nForm: none\nAuthor: Pratik Niroula\n"


def run(auditor, package, tool=pdf_tool):
    repo, delivery, c, _ = package
    return auditor.audit_package(delivery, repo, c, run_tool=tool)


def test_valid_independent_fixture(auditor, package):
    result = run(auditor, package)
    assert result["status"] == "PASS"
    assert len(result["artifacts"]) == 5
    assert result["reviewed_manuscript_inputs"] == 12
    assert result["artifacts"]["main_docx"]["counts"] == {"omml": 1, "tables": 1, "media": 1}


@pytest.mark.parametrize("change", ["unknown", "duplicate_role", "missing_role", "bool_bytes", "bad_hash", "traversal", "absolute", "unknown_source", "missing_source", "bool_pages", "wrong_document", "unknown_field"])
def test_contract_rejected_before_artifact_reads(auditor, package, change):
    _, _, c, data = package
    row = data["artifacts"]["main_pdf"]
    if change == "unknown":
        data["schema"] = "future"
    elif change == "duplicate_role":
        data["artifacts"]["sixth"] = copy.deepcopy(row)
    elif change == "missing_role":
        del data["artifacts"]["main_docx"]
    elif change == "bool_bytes":
        row["bytes"] = True
    elif change == "bad_hash":
        row["sha256"] = "g" * 64
    elif change == "traversal":
        row["filename"] = "../outside.pdf"
    elif change == "absolute":
        row["filename"] = "/outside.pdf"
    elif change == "unknown_source":
        data["reviewed_manuscript_inputs"]["reserved/outcomes.json"] = "0" * 64
    elif change == "missing_source":
        data["reviewed_manuscript_inputs"].pop(next(iter(data["reviewed_manuscript_inputs"])))
    elif change == "bool_pages":
        row["pages"] = True
    elif change == "wrong_document":
        row["document"] = "tmlr"
    elif change == "unknown_field":
        row["private_path"] = "hidden"
    c.write_text(json.dumps(data))
    def forbidden(*args):
        pytest.fail("document parser reached for invalid contract")
    assert run(auditor, package, forbidden)["status"] == "FAIL"


@pytest.mark.parametrize("payload", ['{"schema":"x","schema":"y"}', '{"x":NaN}', '[]', '{'])
def test_invalid_json(auditor, package, payload):
    package[2].write_text(payload)
    assert run(auditor, package)["status"] == "FAIL"


@pytest.mark.parametrize("location", ["package", "artifact", "contract", "source", "source_parent", "authority"])
def test_symlink_rejected(auditor, package, tmp_path, location):
    repo, delivery, c, _ = package
    if location == "package":
        alias = tmp_path / "alias"
        alias.symlink_to(delivery, target_is_directory=True)
        package = repo, alias, c, package[3]
    else:
        path = {"artifact": delivery / "KBound_Main_22pages.pdf", "contract": c, "source": repo / "docs/research/kbound/kbound_abstract_core.tex", "source_parent": repo / "docs/research/kbound/paper/sections", "authority": repo / "docs/research/kbound/paper/release/manuscript_revision.json"}[location]
        target = tmp_path / ("moved-" + location)
        path.rename(target)
        path.symlink_to(target, target_is_directory=target.is_dir())
    assert run(auditor, package)["status"] == "FAIL"


@pytest.mark.parametrize("change", ["extra", "missing", "corrupt", "source_hash", "authority_hash", "directory", "metadata_symlink"])
def test_inventory_and_bindings(auditor, package, change):
    repo, delivery, _, _ = package
    p = delivery / "KBound_Main_22pages.pdf"
    if change == "extra":
        (delivery / "notes.txt").write_text("extra")
    elif change == "missing":
        p.unlink()
    elif change == "corrupt":
        p.write_bytes(b"bad pdf")
    elif change == "source_hash":
        (repo / "docs/research/kbound/kbound_abstract_core.tex").write_text("changed")
    elif change == "authority_hash":
        (repo / "docs/research/kbound/paper/release/manuscript_revision.json").write_text("{}")
    elif change == "directory":
        p.unlink()
        p.mkdir()
    elif change == "metadata_symlink":
        (delivery / "._evil").symlink_to(p)
    assert run(auditor, package)["status"] == "FAIL"


def test_only_appledouble_metadata_ignored(auditor, package):
    (package[1] / "._KBound_Main_22pages.pdf").write_bytes(bytes.fromhex("0005160700020000") + b"\0" * 18)
    assert run(auditor, package)["status"] == "PASS"
    (package[1] / "._not_metadata").write_text("not AppleDouble")
    assert run(auditor, package)["status"] == "FAIL"


@pytest.mark.parametrize("fault", ["pages", "parse_error", "revision", "private_path", "javascript", "encrypted", "attachment"])
def test_pdf_checks(auditor, package, fault):
    def tool(name, *args):
        text = pdf_tool(name, *args)
        if fault == "parse_error":
            raise RuntimeError("parser failed")
        if name == "pdftotext":
            if fault == "revision":
                return "wrong revision"
            if fault == "private_path":
                return text + " /Users/example/private"
        if name == "pdfinfo":
            if fault == "pages":
                return text.replace("Pages: 2", "Pages: 3")
            if fault == "encrypted":
                return text.replace("Encrypted: no", "Encrypted: yes")
            if fault == "javascript" and args[0] == "-js":
                return "alert(1)"
        if name == "pdfdetach" and fault == "attachment":
            return "1 embedded files"
        return text
    assert run(auditor, package, tool)["status"] == "FAIL"


@pytest.mark.parametrize("fault", ["zip", "xml", "traversal", "duplicate", "comment", "tracked", "macro", "private_path", "external_file", "missing_target", "entity", "counts"])
def test_docx_checks_with_rebound_hash(auditor, package, fault):
    _, delivery, c, data = package
    p = delivery / "KBound_Main_Editable.docx"
    extras = {
        "xml": {"word/bad.xml": "<bad>"},
        "traversal": {"../outside.xml": "<x/>"},
        "comment": {"word/comments.xml": f'<w:comments xmlns:w="{W}"><w:comment w:id="0"/></w:comments>'},
        "tracked": {"word/header1.xml": f'<w:hdr xmlns:w="{W}"><w:ins/></w:hdr>'},
        "macro": {"word/vbaProject.bin": b"macro"},
        "private_path": {"word/drawing.xml": '<x descr="/Volumes/example/private.png"/>'},
        "external_file": {"word/_rels/document.xml.rels": f'<Relationships xmlns="{R}"><Relationship Id="x" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/hyperlink" Target="file:///private/a" TargetMode="External"/></Relationships>'},
        "missing_target": {"word/_rels/document.xml.rels": f'<Relationships xmlns="{R}"><Relationship Id="x" Type="image" Target="media/missing.png"/></Relationships>'},
        "entity": {"word/bad.xml": '<!DOCTYPE x [<!ENTITY a "hi">]><x>&a;</x>'},
    }
    if fault == "zip":
        p.write_bytes(b"not zip")
    else:
        write_docx(p, extra=extras.get(fault))
    if fault == "duplicate":
        with zipfile.ZipFile(p, "a") as z:
            z.writestr("word/document.xml", "<x/>")
    row = data["artifacts"]["main_docx"]
    row["sha256"] = digest(p.read_bytes())
    row["bytes"] = p.stat().st_size
    if fault == "counts":
        row["omml"] = 2
    c.write_text(json.dumps(data))
    assert run(auditor, package)["status"] == "FAIL"


def test_cli_new_mode_rejects_missing_package(tmp_path):
    result = subprocess.run([sys.executable, str(SCRIPTS / "audit_current_kbound_release.py"), "--publication-dir", str(tmp_path / "missing")], capture_output=True, text=True, env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"})
    assert result.returncode == 1
    assert json.loads(result.stdout)["status"] == "FAIL"


def test_cli_legacy_options_remain():
    result = subprocess.run([sys.executable, str(SCRIPTS / "audit_current_kbound_release.py"), "--help"], capture_output=True, text=True)
    assert result.returncode == 0
    assert "--pdf-dir" in result.stdout
    assert "--require-exact-inventory" in result.stdout
    assert "--publication-dir" in result.stdout


@pytest.mark.parametrize("fault", ["encoded_private_path", "raw_binary", "member_symlink", "bad_relationship", "foreign_hyperlink", "duplicate_relationship", "wrong_revision", "bad_xml_root"])
def test_additional_docx_boundaries(auditor, package, fault):
    _, delivery, c, data = package
    p = delivery / "KBound_Main_Editable.docx"
    extras = {
        "encoded_private_path": {"word/drawing.xml": '<x descr="&#47;Users&#47;example&#47;private.png"/>'},
        "raw_binary": {"word/raw-data.bin": b"should not be published"},
        "bad_relationship": {"word/_rels/document.xml.rels": f'<Relationships xmlns="{R}"><Relationship Id="x" Type="image" Target="../../../secret"/></Relationships>'},
        "foreign_hyperlink": {"word/_rels/document.xml.rels": f'<Relationships xmlns="{R}"><Relationship Id="x" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/hyperlink" Target="https://example.invalid/data" TargetMode="External"/></Relationships>'},
        "duplicate_relationship": {"word/_rels/document.xml.rels": f'<Relationships xmlns="{R}"><Relationship Id="x" Type="image" Target="media/image1.png"/><Relationship Id="x" Type="image" Target="media/image1.png"/></Relationships>'},
        "bad_xml_root": {"word/document.xml": "<not_word/>"},
    }
    write_docx(p, revision="WRONG-REVISION" if fault == "wrong_revision" else REVISION, extra=extras.get(fault))
    if fault == "member_symlink":
        with zipfile.ZipFile(p, "a") as z:
            info = zipfile.ZipInfo("word/alias.xml")
            info.create_system = 3
            info.external_attr = 0o120777 << 16
            z.writestr(info, "<x/>")
    row = data["artifacts"]["main_docx"]
    row["sha256"] = digest(p.read_bytes())
    row["bytes"] = p.stat().st_size
    c.write_text(json.dumps(data))
    assert run(auditor, package)["status"] == "FAIL"


@pytest.mark.parametrize("fault", ["duplicate_json", "bool_anonymous", "unknown_field", "wrong_identity", "bad_hash"])
def test_rebound_revision_schema(auditor, package, fault):
    repo, _, c, data = package
    p = repo / "docs/research/kbound/paper/release/manuscript_revision.json"
    rev = json.loads(p.read_text())
    if fault == "bool_anonymous":
        rev["documents"]["short_main"]["anonymous"] = 0
    elif fault == "unknown_field":
        rev["unknown"] = "private"
    elif fault == "wrong_identity":
        rev["revision_id"] = "KBOUND-REVISION-1900-01-01"
    elif fault == "bad_hash":
        rev["canonical_panel_sha256"] = "bad"
    p.write_text(json.dumps(rev))
    if fault == "duplicate_json":
        p.write_text(p.read_text().replace('"schema":', '"schema": "duplicate", "schema":', 1))
    data["revision_authority"]["sha256"] = digest(p.read_bytes())
    c.write_text(json.dumps(data))
    assert run(auditor, package)["status"] == "FAIL"


def test_appledouble_requires_existing_counterpart(auditor, package):
    (package[1] / "._unrelated").write_bytes(bytes.fromhex("0005160700020000") + b"\0" * 18)
    assert run(auditor, package)["status"] == "FAIL"


def test_package_traversal_cli(tmp_path):
    p = tmp_path / "sub"
    p.mkdir()
    result = subprocess.run([sys.executable, str(SCRIPTS / "audit_current_kbound_release.py"), "--publication-dir", str(p / "..")], capture_output=True, text=True)
    assert result.returncode == 1
    assert "traversal" in json.loads(result.stdout)["problems"][0]


def test_tools_fail_closed(auditor, package):
    def missing_tool(*args):
        raise FileNotFoundError("synthetic missing tool")
    assert run(auditor, package, missing_tool)["status"] == "FAIL"


def test_numerical_preflight_never_enters_callback_on_missing_input(auditor, tmp_path):
    def callback(*args):
        pytest.fail("numerical callback reached without approved inputs")
    with pytest.raises(OSError):
        auditor.audit_numerical_authorities(tmp_path, callback)


def test_publication_mode_does_not_call_legacy_claim_scans(auditor, package, monkeypatch, capsys):
    sys.path.insert(0, str(SCRIPTS))
    try:
        import audit_current_kbound_release as legacy
        import audit_publication_package as helper
        monkeypatch.setattr(helper, "audit_package", lambda *args: {"status": "PASS", "problems": []})
        monkeypatch.setattr(helper, "audit_numerical_authorities", lambda *args: [])
        def forbidden(*args, **kwargs):
            pytest.fail("legacy Markdown or PDF mode entered")
        for name in ("run_audit", "audit_release_files", "audit_active_claims", "audit_pdfs"):
            monkeypatch.setattr(legacy, name, forbidden)
        monkeypatch.setattr(sys, "argv", ["audit", "--publication-dir", str(package[1])])
        assert legacy.main() == 0
        assert json.loads(capsys.readouterr().out)["historical_numerical_authorities"] == "PASS"
    finally:
        sys.path.remove(str(SCRIPTS))


@pytest.mark.parametrize("fault", ["wrong_root", "wrong_namespace", "missing_override", "wrong_mime", "duplicate_override", "malformed_override", "duplicate_default", "foreign_child"])
def test_word_content_types_declarations(auditor, package, fault):
    namespace = "http://schemas.openxmlformats.org/package/2006/content-types"
    mime = "application/vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml"
    main = f'<Override PartName="/word/document.xml" ContentType="{mime}"/>'
    declarations = {
        "wrong_root": f'<not_content_types note="{mime}"/>',
        "wrong_namespace": f'<Types xmlns="wrong">{main}</Types>',
        "missing_override": f'<Types xmlns="{namespace}"><Default Extension="xml" ContentType="{mime}"/></Types>',
        "wrong_mime": f'<Types xmlns="{namespace}"><Override PartName="/word/document.xml" ContentType="application/xml"/></Types>',
        "duplicate_override": f'<Types xmlns="{namespace}">{main}{main}</Types>',
        "malformed_override": f'<Types xmlns="{namespace}">{main}<Override ContentType="application/xml"/></Types>',
        "duplicate_default": f'<Types xmlns="{namespace}">{main}<Default Extension="xml" ContentType="application/xml"/><Default Extension="xml" ContentType="application/xml"/></Types>',
        "foreign_child": f'<Types xmlns="{namespace}">{main}<NotADeclaration/></Types>',
    }
    _, delivery, c, data = package
    p = delivery / "KBound_Main_Editable.docx"
    write_docx(p, extra={"[Content_Types].xml": declarations[fault]})
    row = data["artifacts"]["main_docx"]
    row["sha256"] = digest(p.read_bytes())
    row["bytes"] = p.stat().st_size
    c.write_text(json.dumps(data))
    assert run(auditor, package)["status"] == "FAIL"


@pytest.mark.parametrize("exception", ["AssertionError", "IndexError"])
def test_publication_cli_controls_legacy_numeric_shape_errors(exception):
    code = (
        "import sys; "
        f"sys.path.insert(0, {str(SCRIPTS)!r}); "
        "import audit_current_kbound_release as cli; "
        "import audit_publication_package as helper; "
        "helper.audit_package=lambda *args: {'status':'PASS','problems':[]}; "
        f"helper.audit_numerical_authorities=lambda *args: (_ for _ in ()).throw({exception}('synthetic private diagnostic')); "
        "sys.argv=['audit', '--publication-dir', 'synthetic']; "
        "sys.exit(cli.main())"
    )
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                            env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"})
    assert result.returncode == 1
    assert result.stderr == ""
    assert json.loads(result.stdout)["status"] == "FAIL"
    assert "synthetic private diagnostic" not in result.stdout
