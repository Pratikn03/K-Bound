"""Model versioning and rollback (Gate P P11).

Tracks active ``model_version`` per model type from ``models/MANIFEST.json`` and
optional runtime overrides. Rollback switches the active version pointer; artifact
reload requires checksum-pinned paths listed in the manifest.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field

from .auth import authenticate

router = APIRouter(prefix="/models", tags=["model-governance"])

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_MANIFEST_PATH = _PROJECT_ROOT / "models" / "MANIFEST.json"

# Runtime rollback overrides: model_type -> version id from manifest history.
_active_model_version: dict[str, str] = {}


def _load_manifest() -> dict[str, Any]:
    if not _MANIFEST_PATH.is_file():
        return {"schema": "uais-model-manifest-v1", "models": {}, "versions": {}}
    return json.loads(_MANIFEST_PATH.read_text(encoding="utf-8"))


def get_active_model_version(model_type: str) -> str:
    if model_type in _active_model_version:
        return _active_model_version[model_type]
    env_key = f"UAIS_MODEL_VERSION_{model_type.upper()}"
    return os.getenv(env_key) or "default"


def model_version_report() -> dict[str, Any]:
    manifest = _load_manifest()
    models: dict[str, Any] = {}
    for model_type, entries in manifest.get("models", {}).items():
        active = get_active_model_version(model_type)
        history = manifest.get("versions", {}).get(model_type, [])
        models[model_type] = {
            "model_version": active,
            "available_versions": history,
            "artifact": entries.get(active) or entries.get("default"),
        }
    return {
        "manifest_path": str(_MANIFEST_PATH.relative_to(_PROJECT_ROOT)),
        "model_version": {k: v["model_version"] for k, v in models.items()},
        "models": models,
        "rollback_note": "POST /models/rollback switches active model_version; "
        "restart or call reload after updating checksum env vars.",
    }


class RollbackRequest(BaseModel):
    model_type: str = Field(..., min_length=1, max_length=64)
    target_version: str = Field(..., min_length=1, max_length=64)


@router.get("/versions")
async def list_model_versions(authenticated: bool = Depends(authenticate)) -> dict[str, Any]:
    """Return active model_version per model type and manifest history."""
    return model_version_report()


@router.post("/rollback")
async def rollback_model_version(
    req: RollbackRequest,
    authenticated: bool = Depends(authenticate),
) -> dict[str, Any]:
    """Rollback active model_version to a prior entry from the manifest."""
    manifest = _load_manifest()
    history = manifest.get("versions", {}).get(req.model_type, [])
    if req.target_version not in history:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"target_version {req.target_version!r} not in manifest history for {req.model_type}",
        )
    _active_model_version[req.model_type] = req.target_version
    artifact = manifest.get("models", {}).get(req.model_type, {}).get(req.target_version)
    return {
        "status": "rollback_applied",
        "model_type": req.model_type,
        "model_version": req.target_version,
        "artifact": artifact,
        "next_steps": [
            "Set UAIS_MODEL_SHA256_* to the rolled-back artifact checksum if changed",
            "Restart API or reload models to pick up the artifact path",
        ],
    }
