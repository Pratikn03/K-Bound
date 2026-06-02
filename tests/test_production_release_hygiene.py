from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]


def test_production_compose_defaults_to_api_only_service():
    compose = yaml.safe_load((ROOT / "docker-compose.yml").read_text(encoding="utf-8"))
    services = compose["services"]

    assert "depends_on" not in services["api"]
    assert "profiles" not in services["api"]
    assert "research" in services["streamlit"]["profiles"]
    assert "research" in services["mlflow"]["profiles"]


def test_dockerignore_excludes_large_and_untrusted_release_inputs():
    dockerignore = (ROOT / ".dockerignore").read_text(encoding="utf-8").splitlines()
    ignored = {line.strip() for line in dockerignore if line.strip() and not line.startswith("#")}

    for pattern in {"data", "models", "experiments", "mlruns", "output", "docs", "tests", "._*", "**/._*"}:
        assert pattern in ignored


def test_production_runbook_documents_required_operations():
    runbook = ROOT / "docs" / "production" / "PRODUCTION_RUNBOOK.md"
    text = runbook.read_text(encoding="utf-8")

    required_sections = [
        "Environment",
        "Artifact Checksum Policy",
        "Deployment",
        "Readiness",
        "Monitoring",
        "Rollback",
        "Incident Response",
        "Research Claim Boundary",
    ]
    for section in required_sections:
        assert f"## {section}" in text

    assert "Gate E remains a scientific gate" in text
    assert "UAIS_API_KEYS" in text
    assert "UAIS_CORS_ORIGINS" in text
