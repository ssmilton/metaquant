"""Configuration loading and validation utilities."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml
from pydantic import BaseModel, Field

from core.registry.schema import ExecutionNode


class StorageConfig(BaseModel):
    duckdb_path: str = Field(..., description="Path to DuckDB database file")


class LoggingConfig(BaseModel):
    level: str = "INFO"
    format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"


class BacktestDefaults(BaseModel):
    initial_capital: float = 1_000_000
    transaction_cost_bps: float = 5
    slippage_bps: float = 1


class DockerRuntimeConfig(BaseModel):
    host: Optional[str] = Field(
        default=None, description="Docker host/daemon address (e.g., tcp://...)"
    )
    default_timeout_secs: int = Field(
        default=300, description="Default timeout for containerized model execution"
    )
    default_memory: Optional[str] = Field(
        default=None, description="Memory limit passed to docker run (e.g., 2g)"
    )
    default_cpus: Optional[float] = Field(
        default=None, description="CPU quota passed to docker run (e.g., 1.5)"
    )
    default_mounts: List[str] = Field(
        default_factory=list, description="List of bind mounts to attach to docker runs"
    )


class AppConfig(BaseModel):
    storage: StorageConfig
    logging: LoggingConfig = LoggingConfig()
    backtest_defaults: BacktestDefaults = BacktestDefaults()
    nodes: List[ExecutionNode] = Field(
        default_factory=lambda: [
            ExecutionNode(
                name="local-linux-dev",
                platform="linux/amd64",
                runtimes=["host", "docker"],
                tags=["dev", "default"],
            )
        ]
    )
    docker: DockerRuntimeConfig = DockerRuntimeConfig()
    env_root: str = Field(
        default=".envs", description="Root folder for per-model virtual environments"
    )

    @staticmethod
    def from_yaml(path: str | Path) -> "AppConfig":
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        return AppConfig(**data)

    def model_dump_json(self, **kwargs: Any) -> str:  # pragma: no cover - passthrough
        return json.dumps(self.model_dump(), **kwargs)


DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent.parent / "config" / "base.yaml"


def load_app_config(path: str | Path | None = None) -> AppConfig:
    """Load application config from YAML.

    Args:
        path: Optional path to YAML config. If None, defaults to config/base.yaml.
    """
    resolved = Path(path) if path else DEFAULT_CONFIG_PATH
    return AppConfig.from_yaml(resolved)
