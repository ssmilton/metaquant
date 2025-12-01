import pytest

from core.registry.schema import ExecutionNode, ModelManifest
from core.runtime.nodes import select_node
from core.models.runner import ModelRunnerFactory, ModelExecutionError


def test_manifest_validation_for_docker():
    with pytest.raises(ValueError):
        ModelManifest(
            model_id="bad_docker",
            version="1.0.0",
            description="Missing docker fields",
            author="tester",
            runtime="docker",
            platform="linux/amd64",
        )


def test_select_node_matches_platform_and_runtime():
    manifest = ModelManifest(
        model_id="win_model",
        version="0.1.0",
        description="Windows only",
        author="qa",
        runtime="docker",
        platform="windows/amd64",
        docker_image="example/win:v1",
        docker_entrypoint="C:\\app\\run.exe",
    )
    nodes = [
        ExecutionNode(name="linux", platform="linux/amd64", runtimes=["host", "docker"], tags=["default"]),
        ExecutionNode(name="windows", platform="windows/amd64", runtimes=["docker"], tags=["windows", "default"]),
    ]

    selected = select_node(manifest, nodes)
    assert selected.name == "windows"


def test_runner_factory_rejects_incompatible_node():
    manifest = ModelManifest(
        model_id="host_model",
        version="1.0.0",
        description="host only",
        author="qa",
        runtime="host",
        platform="linux/amd64",
        entrypoint="python run.py",
    )
    node = ExecutionNode(name="win", platform="windows/amd64", runtimes=["docker"], tags=[])
    factory = ModelRunnerFactory()

    with pytest.raises(ModelExecutionError):
        factory.create_runner(manifest, ".", node)
