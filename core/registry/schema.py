"""Schemas for model manifests and contracts."""
from __future__ import annotations

from pathlib import Path
from typing import List, Optional

from pydantic import BaseModel, Field, model_validator


class ModelManifest(BaseModel):
    model_id: str = Field(..., description="Unique model identifier")
    version: str
    description: str
    author: str
    entrypoint: str = Field(..., description="Executable command to run the model")
    input_types: List[str] = Field(default_factory=list)
    output_type: str = "signals"
    tags: Optional[List[str]] = None
    path: Optional[Path] = Field(default=None, description="Path to manifest file")

    @model_validator(mode="after")
    def ensure_semver_like(self) -> "ModelManifest":
        if self.version.count(".") < 1:
            raise ValueError("version should look like semantic versioning (e.g., 1.0.0)")
        return self
