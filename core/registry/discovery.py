"""Discovery utilities for model manifests."""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import List

from .schema import ModelManifest

logger = logging.getLogger(__name__)


class ModelRegistry:
    def __init__(self, root: str | Path = "models") -> None:
        self.root = Path(root)

    def list_manifests(self) -> List[ModelManifest]:
        manifests: List[ModelManifest] = []
        for path in self.root.rglob("model.json"):
            try:
                with open(path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                manifest = ModelManifest(**data, path=path)
                manifests.append(manifest)
            except Exception as exc:  # pragma: no cover - narrow scope
                logger.error("Failed to load manifest %s: %s", path, exc)
        return manifests

    def get_model(self, model_id: str) -> ModelManifest:
        for manifest in self.list_manifests():
            if manifest.model_id == model_id:
                return manifest
        raise ValueError(f"Model {model_id} not found")
