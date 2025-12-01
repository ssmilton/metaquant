"""Schemas for model manifests and contracts."""
from __future__ import annotations

from pathlib import Path
from typing import List, Optional

from pydantic import BaseModel, ConfigDict, Field, model_validator

RuntimeType = str
PlatformType = str


class HostDependencies(BaseModel):
    """Optional dependency metadata for host-executed models."""

    model_config = ConfigDict(extra="allow")

    python: Optional[str] = Field(
        default=None, description="Python version constraint for host runtime"
    )
    requirements_file: Optional[str] = Field(
        default=None,
        description="Relative path to requirements.txt for uv-managed installation",
    )


class ModelManifest(BaseModel):
    model_id: str = Field(..., description="Unique model identifier")
    version: str
    description: str
    author: str
    runtime: RuntimeType
    platform: PlatformType
    entrypoint: Optional[str] = Field(
        default=None, description="Executable command to run the model (host runtime)"
    )
    docker_image: Optional[str] = Field(
        default=None, description="Docker image for containerized execution"
    )
    docker_entrypoint: Optional[str] = Field(
        default=None, description="Entrypoint/command to run inside the container"
    )
    dependencies: Optional[HostDependencies] = Field(
        default=None, description="Dependency metadata for host runtime"
    )
    input_types: List[str] = Field(default_factory=list)
    output_type: str = "signals"
    tags: Optional[List[str]] = None
    path: Optional[Path] = Field(default=None, description="Path to manifest file")

    @model_validator(mode="after")
    def ensure_semver_like(self) -> "ModelManifest":
        if self.version.count(".") < 1:
            raise ValueError("version should look like semantic versioning (e.g., 1.0.0)")
        return self

    @model_validator(mode="after")
    def validate_runtime_platform(self) -> "ModelManifest":
        allowed_platforms = {"linux/amd64", "windows/amd64", "any"}
        if self.platform not in allowed_platforms:
            raise ValueError(f"platform must be one of {sorted(allowed_platforms)}")

        if self.runtime == "host":
            if not self.entrypoint:
                raise ValueError("entrypoint is required for host runtime")
        elif self.runtime == "docker":
            if self.platform == "any":
                raise ValueError("docker runtime requires a concrete platform")
            if not self.docker_image or not self.docker_entrypoint:
                raise ValueError("docker_image and docker_entrypoint are required for docker runtime")
        else:
            raise ValueError("runtime must be either 'host' or 'docker'")
        return self


class ExecutionNode(BaseModel):
    """Execution environment describing runtime and platform capabilities."""

    name: str
    platform: PlatformType
    runtimes: List[RuntimeType]
    tags: List[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_platform_runtime(self) -> "ExecutionNode":
        allowed_platforms = {"linux/amd64", "windows/amd64", "any"}
        if self.platform not in allowed_platforms:
            raise ValueError(f"platform must be one of {sorted(allowed_platforms)}")
        for runtime in self.runtimes:
            if runtime not in {"host", "docker"}:
                raise ValueError("runtimes must contain 'host' and/or 'docker'")
        if "docker" in self.runtimes and self.platform == "any":
            raise ValueError("docker-capable nodes must declare a concrete platform")
        return self

    def supports(self, manifest: ModelManifest) -> bool:
        platform_match = manifest.platform == "any" or manifest.platform == self.platform
        runtime_match = manifest.runtime in self.runtimes
        return platform_match and runtime_match
