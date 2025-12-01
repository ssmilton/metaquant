"""Subprocess-based model runner."""
from __future__ import annotations

import json
import logging
import shlex
import subprocess
from pathlib import Path
from typing import Dict

from .base_contract import ModelInput, ModelOutput
from core.registry.schema import ModelManifest

logger = logging.getLogger(__name__)


class ModelExecutionError(RuntimeError):
    pass


class ModelRunner:
    def __init__(self, manifest: ModelManifest, working_dir: str | Path) -> None:
        self.manifest = manifest
        self.working_dir = Path(working_dir)

    def run(self, input_payload: Dict) -> Dict:
        logger.info("Running model %s", self.manifest.model_id)
        command = shlex.split(self.manifest.entrypoint)
        process = subprocess.Popen(
            command,
            cwd=self.working_dir,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        serialized = json.dumps(input_payload)
        stdout, stderr = process.communicate(serialized)

        if process.returncode != 0:
            raise ModelExecutionError(
                f"Model {self.manifest.model_id} failed with code {process.returncode}: {stderr}"
            )
        try:
            data = json.loads(stdout)
            ModelOutput(**data)  # validation
            return data
        except Exception as exc:  # pragma: no cover - validation path
            raise ModelExecutionError(f"Failed to parse model output: {exc}. Raw: {stdout}") from exc


__all__ = ["ModelRunner", "ModelExecutionError"]
