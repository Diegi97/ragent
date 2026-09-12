import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest


@pytest.fixture
def deletion_script():
    path = Path(__file__).parents[1] / "scripts" / "delete_turbopuffer_namespaces.py"
    spec = importlib.util.spec_from_file_location("namespace_deletion", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_deletion_preview_never_constructs_client(deletion_script, monkeypatch):
    def forbidden():
        pytest.fail("preview constructed a client")

    monkeypatch.setattr(deletion_script, "create_turbopuffer_client", forbidden)
    deletion_script.delete_namespaces(
        ["test.source.chunks"], namespace_prefix="test", confirmed=False
    )


def test_deletion_validates_every_target_before_mutating(deletion_script):
    deleted = []
    client = SimpleNamespace(
        namespace=lambda name: SimpleNamespace(
            exists=lambda: True, delete_all=lambda: deleted.append(name)
        )
    )
    with pytest.raises(ValueError, match="Refusing namespace"):
        deletion_script.delete_namespaces(
            ["test.source.chunks", "production.source.chunks"],
            namespace_prefix="test",
            confirmed=True,
            client=client,
        )
    assert not deleted
    deletion_script.delete_namespaces(
        ["test.source.chunks", "test.source.chunks"],
        namespace_prefix="test",
        confirmed=True,
        client=client,
    )
    assert deleted == ["test.source.chunks"]
