"""Runtime-aware model runners (host and docker)."""
from __future__ import annotations

import json
import logging
import os
import shlex
import subprocess
import sys
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, List, Optional

from core.config import DockerRuntimeConfig
from core.registry.schema import ExecutionNode, ModelManifest

from .base_contract import ModelOutput

logger = logging.getLogger(__name__)


class ModelExecutionError(RuntimeError):
    """Raised when a model process fails or returns invalid output."""


class BaseModelRunner(ABC):
    """Common interface for executing external models."""

    def __init__(
        self, manifest: ModelManifest, working_dir: str | Path, node: ExecutionNode
    ) -> None:
        self.manifest = manifest
        self.working_dir = Path(working_dir)
        self.node = node

    @abstractmethod
    def run(self, input_payload: Dict, timeout: Optional[int] = None) -> Dict:
        """Execute the model with the given payload."""


class HostModelRunner(BaseModelRunner):
    """Executes models directly on the host using a per-model virtual environment."""

    def __init__(
        self,
        manifest: ModelManifest,
        working_dir: str | Path,
        node: ExecutionNode,
        env_root: str | Path = ".envs",
    ) -> None:
        super().__init__(manifest, working_dir, node)
        self.env_root = Path(env_root)
        self.env_dir = self.env_root / f"{manifest.model_id}-{manifest.version}"

    def _python_executable(self) -> Path:
        script_dir = "Scripts" if os.name == "nt" else "bin"
        return self.env_dir / script_dir / "python"

    def _bin_dir(self) -> Path:
        return self._python_executable().parent

    def _ensure_environment(self) -> None:
        python_path = self._python_executable()
        if python_path.exists():
            return

        logger.info("Creating venv for %s at %s", self.manifest.model_id, self.env_dir)
        self.env_dir.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run([sys.executable, "-m", "venv", str(self.env_dir)], check=True)

        requirements_file = self.manifest.dependencies.requirements_file if self.manifest.dependencies else None
        if requirements_file:
            req_path = (self.working_dir / requirements_file).resolve()
            if req_path.exists():
                logger.info("Installing requirements for %s from %s", self.manifest.model_id, req_path)
                subprocess.run(
                    [str(python_path), "-m", "pip", "install", "-r", str(req_path)],
                    check=True,
                )
            else:
                logger.warning("requirements_file %s not found for %s", req_path, self.manifest.model_id)

    def _build_command(self) -> List[str]:
        assert self.manifest.entrypoint, "entrypoint should be validated for host runtime"
        command = shlex.split(self.manifest.entrypoint)
        python_path = self._python_executable()
        if command and command[0] == "python":
            command[0] = str(python_path)
        return command

    def run(self, input_payload: Dict, timeout: Optional[int] = None) -> Dict:
        logger.info("Running model %s on host node %s", self.manifest.model_id, self.node.name)
        self._ensure_environment()
        command = self._build_command()

        env = os.environ.copy()
        env["PATH"] = f"{self._bin_dir()}{os.pathsep}{env.get('PATH', '')}"

        process = subprocess.Popen(
            command,
            cwd=self.working_dir,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            env=env,
        )
        serialized = json.dumps(input_payload)
        try:
            stdout, stderr = process.communicate(serialized, timeout=timeout)
        except subprocess.TimeoutExpired:
            process.kill()
            raise ModelExecutionError(
                f"Model {self.manifest.model_id} exceeded timeout of {timeout} seconds"
            )

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


class DockerModelRunner(BaseModelRunner):
    """Executes models inside Docker containers (Linux or Windows)."""

    def __init__(
        self,
        manifest: ModelManifest,
        working_dir: str | Path,
        node: ExecutionNode,
        docker_config: Optional[DockerRuntimeConfig] = None,
    ) -> None:
        super().__init__(manifest, working_dir, node)
        self.docker_config = docker_config or DockerRuntimeConfig()

    def _base_command(self) -> List[str]:
        assert self.manifest.docker_image and self.manifest.docker_entrypoint
        cmd: List[str] = [
            "docker",
            "run",
            "--rm",
            "--platform",
            self.manifest.platform,
            "-i",
        ]
        if self.docker_config.default_memory:
            cmd.extend(["--memory", self.docker_config.default_memory])
        if self.docker_config.default_cpus is not None:
            cmd.extend(["--cpus", str(self.docker_config.default_cpus)])
        for mount in self.docker_config.default_mounts:
            cmd.extend(["-v", mount])
        cmd.append(self.manifest.docker_image)
        cmd.extend(shlex.split(self.manifest.docker_entrypoint))
        return cmd

    def run(self, input_payload: Dict, timeout: Optional[int] = None) -> Dict:
        effective_timeout = timeout or self.docker_config.default_timeout_secs
        env = os.environ.copy()
        if self.docker_config.host:
            env["DOCKER_HOST"] = self.docker_config.host

        command = self._base_command()
        logger.info(
            "Running model %s in docker on node %s with platform %s",
            self.manifest.model_id,
            self.node.name,
            self.manifest.platform,
        )
        process = subprocess.Popen(
            command,
            cwd=self.working_dir,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            env=env,
        )
        serialized = json.dumps(input_payload)
        try:
            stdout, stderr = process.communicate(serialized, timeout=effective_timeout)
        except subprocess.TimeoutExpired:
            process.kill()
            raise ModelExecutionError(
                f"Model {self.manifest.model_id} exceeded timeout of {effective_timeout} seconds"
            )

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


class ModelRunnerFactory:
    """Factory that chooses runner implementations based on manifest and node."""

    def __init__(
        self, docker_config: Optional[DockerRuntimeConfig] = None, env_root: str | Path = ".envs"
    ) -> None:
        self.docker_config = docker_config or DockerRuntimeConfig()
        self.env_root = env_root

    def create_runner(
        self, manifest: ModelManifest, working_dir: str | Path, node: ExecutionNode
    ) -> BaseModelRunner:
        if not node.supports(manifest):
            raise ModelExecutionError(
                f"Node {node.name} cannot run model {manifest.model_id} (runtime={manifest.runtime}, platform={manifest.platform})"
            )

        if manifest.runtime == "host":
            return HostModelRunner(manifest, working_dir, node, env_root=self.env_root)
        if manifest.runtime == "docker":
            return DockerModelRunner(
                manifest, working_dir, node, docker_config=self.docker_config
            )
        raise ModelExecutionError(f"Unsupported runtime {manifest.runtime}")


__all__ = [
    "BaseModelRunner",
    "HostModelRunner",
    "DockerModelRunner",
    "ModelRunnerFactory",
    "ModelExecutionError",
]
