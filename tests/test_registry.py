from core.registry.discovery import ModelRegistry


def test_manifest_discovery():
    registry = ModelRegistry()
    manifests = registry.list_manifests()
    assert any(m.model_id == "sector_momentum_v1" for m in manifests)
    assert all(m.runtime in {"host", "docker"} for m in manifests)
