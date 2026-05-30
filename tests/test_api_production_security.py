import asyncio
import hashlib
import importlib
import sys

import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient

SECURITY_ENV = {
    "UAIS_API_KEYS",
    "UAIS_CORS_ORIGINS",
    "UAIS_MAX_FEATURES",
    "UAIS_MAX_SCORES",
    "UAIS_MAX_DOMAINS",
    "UAIS_MAX_EMBEDDINGS",
    "UAIS_MAX_TEXT_CHARS",
    "UAIS_MAX_IMAGE_BASE64_CHARS",
    "UAIS_MAX_IMAGE_PIXELS",
    "UAIS_RATE_LIMIT_REQUESTS",
    "UAIS_RATE_LIMIT_WINDOW_SECONDS",
    "UAIS_TRUSTED_MODEL_ARTIFACTS",
    "UAIS_REQUIRE_MODEL_CHECKSUMS",
    "UAIS_MODEL_SHA256_FRAUD",
    "UAIS_MODEL_SHA256_CYBER",
    "UAIS_MODEL_SHA256_FUSION",
    "UAIS_MODEL_SHA256_NLP",
    "UAIS_MODEL_SHA256_VISION",
    "UAIS_MODEL_SHA256_ATTENTION_FUSION",
}


def _reload_api(monkeypatch: pytest.MonkeyPatch, **env: str):
    for key in SECURITY_ENV:
        monkeypatch.delenv(key, raising=False)
    for key, value in env.items():
        monkeypatch.setenv(key, value)
    for name in list(sys.modules):
        if name == "deploy.api" or name.startswith("deploy.api."):
            del sys.modules[name]
    return importlib.import_module("deploy.api.main")


def _reload_auth(monkeypatch: pytest.MonkeyPatch, **env: str):
    for key in SECURITY_ENV:
        monkeypatch.delenv(key, raising=False)
    for key, value in env.items():
        monkeypatch.setenv(key, value)
    for name in list(sys.modules):
        if name == "deploy.api.auth":
            del sys.modules[name]
    return importlib.import_module("deploy.api.auth")


def test_api_key_auth_fails_closed_when_no_keys_configured(monkeypatch):
    auth = _reload_auth(monkeypatch)

    with pytest.raises(HTTPException) as exc_info:
        asyncio.run(auth.verify_api_key(None))

    assert exc_info.value.status_code == 503
    assert "not configured" in exc_info.value.detail.lower()


def test_password_hashing_backend_is_compatible(monkeypatch):
    auth = _reload_auth(monkeypatch, UAIS_API_KEYS="secret")

    hashed = auth.get_password_hash("correct horse battery staple")

    assert auth.verify_password("correct horse battery staple", hashed)
    assert not auth.verify_password("wrong", hashed)


def test_cors_uses_explicit_allowlist_from_environment(monkeypatch):
    api = _reload_api(
        monkeypatch,
        UAIS_API_KEYS="secret",
        UAIS_CORS_ORIGINS="https://console.example, https://ops.example",
    )

    cors = next(m for m in api.app.user_middleware if m.cls.__name__ == "CORSMiddleware")

    assert cors.kwargs["allow_origins"] == ["https://console.example", "https://ops.example"]
    assert "*" not in cors.kwargs["allow_origins"]
    assert cors.kwargs["allow_credentials"] is True


def test_operational_endpoints_require_api_key(monkeypatch):
    api = _reload_api(monkeypatch, UAIS_API_KEYS="secret")
    client = TestClient(api.app)

    for path in ("/health/detailed", "/metrics", "/system"):
        assert client.get(path).status_code == 403
        assert client.get(path, headers={"X-API-Key": "secret"}).status_code in {200, 501}


def test_vision_payload_is_rejected_before_model_loading_when_too_large(monkeypatch):
    api = _reload_api(
        monkeypatch,
        UAIS_API_KEYS="secret",
        UAIS_MAX_IMAGE_BASE64_CHARS="16",
    )
    client = TestClient(api.app)

    response = client.post(
        "/predict_vision",
        headers={"X-API-Key": "secret"},
        json={"image_base64": "A" * 17},
    )

    assert response.status_code == 422


def test_feature_vectors_have_configurable_upper_bounds(monkeypatch):
    api = _reload_api(
        monkeypatch,
        UAIS_API_KEYS="secret",
        UAIS_MAX_FEATURES="2",
    )
    client = TestClient(api.app)

    response = client.post(
        "/predict_fraud",
        headers={"X-API-Key": "secret"},
        json={"features": [1.0, 2.0, 3.0]},
    )

    assert response.status_code == 422


def test_rate_limiter_rejects_requests_over_configured_window(monkeypatch):
    api = _reload_api(
        monkeypatch,
        UAIS_API_KEYS="secret",
        UAIS_RATE_LIMIT_REQUESTS="2",
        UAIS_RATE_LIMIT_WINDOW_SECONDS="60",
    )
    client = TestClient(api.app)

    assert client.get("/health").status_code == 200
    assert client.get("/health").status_code == 200
    assert client.get("/health").status_code == 429


def test_joblib_artifacts_are_not_loaded_without_trust_flag(monkeypatch, tmp_path):
    api = _reload_api(monkeypatch, UAIS_API_KEYS="secret")
    artifact = tmp_path / "model.pkl"
    artifact.write_bytes(b"not a trusted model")

    assert api._load_joblib_artifact(artifact, "fraud") is None


def test_model_artifacts_require_matching_checksum_when_trusted(monkeypatch, tmp_path):
    api = _reload_api(monkeypatch, UAIS_API_KEYS="secret", UAIS_TRUSTED_MODEL_ARTIFACTS="true")
    artifact = tmp_path / "model.pkl"
    artifact.write_bytes(b"trusted bytes")

    assert api._artifact_is_trusted(artifact, "fraud") is False

    monkeypatch.setenv("UAIS_MODEL_SHA256_FRAUD", "0" * 64)
    assert api._artifact_is_trusted(artifact, "fraud") is False

    expected = hashlib.sha256(b"trusted bytes").hexdigest()
    monkeypatch.setenv("UAIS_MODEL_SHA256_FRAUD", expected)
    assert api._artifact_is_trusted(artifact, "fraud") is True


def test_torch_artifacts_require_weights_only_loader(monkeypatch, tmp_path):
    api = _reload_api(
        monkeypatch,
        UAIS_API_KEYS="secret",
        UAIS_TRUSTED_MODEL_ARTIFACTS="true",
        UAIS_REQUIRE_MODEL_CHECKSUMS="false",
    )
    artifact = tmp_path / "model.pt"
    artifact.write_bytes(b"trusted torch bytes")

    class TorchWithoutWeightsOnly:
        def load(self, *args, **kwargs):
            raise TypeError("weights_only is unsupported")

    assert api._load_torch_artifact(TorchWithoutWeightsOnly(), artifact, "vision") is None


def test_health_checker_accepts_sync_and_async_checks():
    from deploy.api.monitoring import HealthChecker

    async def async_check():
        return True

    checker = HealthChecker()
    checker.register_check("sync_ok", lambda: True)
    checker.register_check("async_ok", async_check)

    result = asyncio.run(checker.run_checks())

    assert result["status"] == "healthy"
    assert result["checks"]["sync_ok"]["status"] == "pass"
    assert result["checks"]["async_ok"]["status"] == "pass"
