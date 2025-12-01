"""Configuration loading and validation utilities."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

import yaml
from pydantic import BaseModel, Field


class StorageConfig(BaseModel):
    duckdb_path: str = Field(..., description="Path to DuckDB database file")


class LoggingConfig(BaseModel):
    level: str = "INFO"
    format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"


class BacktestDefaults(BaseModel):
    initial_capital: float = 1_000_000
    transaction_cost_bps: float = 5
    slippage_bps: float = 1


class AppConfig(BaseModel):
    storage: StorageConfig
    logging: LoggingConfig = LoggingConfig()
    backtest_defaults: BacktestDefaults = BacktestDefaults()

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
