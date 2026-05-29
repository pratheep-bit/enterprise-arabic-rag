from app.api import routes
from app.services.document_registry import DocumentRegistry


class FakeVectorStore:
    def list_collections(self):
        return ["legacy_collection"]

    def get_collection_info(self, collection_name):
        return {"name": collection_name, "count": 7}


def test_health_does_not_require_registered_documents(monkeypatch, tmp_path):
    registry = DocumentRegistry(tmp_path / "registry.json")
    monkeypatch.setattr(routes, "VectorStore", lambda: FakeVectorStore())

    # Mock get_startup_time for health check
    monkeypatch.setattr("app.main.get_startup_time", lambda: 0.0)

    response = __import__("asyncio").run(routes.health_check(registry=registry))

    assert response.status == "healthy"
    assert response.documents_count == 1


def test_list_documents_includes_legacy_collections(monkeypatch, tmp_path):
    registry = DocumentRegistry(tmp_path / "registry.json")
    monkeypatch.setattr(routes, "VectorStore", lambda: FakeVectorStore())

    response = __import__("asyncio").run(routes.list_documents(registry=registry))

    assert response.total_count == 1
    assert response.documents[0].document_id == "legacy_collection"
    assert response.documents[0].chunk_count == 7
