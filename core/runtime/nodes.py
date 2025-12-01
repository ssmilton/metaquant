"""Execution node capabilities and scheduling utilities."""
from __future__ import annotations

import logging
from typing import Iterable, List, Optional

from core.registry.schema import ExecutionNode, ModelManifest

logger = logging.getLogger(__name__)


def select_node(
    manifest: ModelManifest,
    nodes: Iterable[ExecutionNode],
    preferred_tags: Optional[List[str]] = None,
) -> ExecutionNode:
    """Select a node capable of running the manifest.

    Selection prefers nodes matching preferred tags and those labeled as "default".
    """

    candidates = [node for node in nodes if node.supports(manifest)]
    if not candidates:
        raise ValueError(
            f"No nodes available for runtime={manifest.runtime} and platform={manifest.platform}"
        )

    preferred_set = set(preferred_tags or [])

    def score(node: ExecutionNode) -> tuple[int, int, str]:
        tag_overlap = len(preferred_set.intersection(node.tags))
        default_tag = 1 if "default" in node.tags else 0
        return (tag_overlap, default_tag, node.name)

    chosen = sorted(candidates, key=score, reverse=True)[0]
    logger.info(
        "Selected node %s for model %s (runtime=%s, platform=%s)",
        chosen.name,
        manifest.model_id,
        manifest.runtime,
        manifest.platform,
    )
    return chosen


__all__ = ["select_node"]
