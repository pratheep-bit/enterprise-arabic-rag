from app.services.document_registry import DocumentRegistry


def test_document_registry_persists_metadata(tmp_path):
    registry_path = tmp_path / "registry.json"
    registry = DocumentRegistry(registry_path)

    registry.upsert(
        {
            "document_id": "doc123",
            "filename": "sample.pdf",
            "collection_name": "doc_doc123",
            "upload_time": "2026-05-28T00:00:00+00:00",
        }
    )

    reloaded = DocumentRegistry(registry_path)

    assert reloaded.get("doc123")["filename"] == "sample.pdf"
    assert reloaded.count() == 1
    assert reloaded.list()[0]["document_id"] == "doc123"


def test_document_registry_delete_and_clear(tmp_path):
    registry_path = tmp_path / "registry.json"
    registry = DocumentRegistry(registry_path)

    registry.upsert({"document_id": "doc1", "filename": "doc1.pdf"})
    registry.upsert({"document_id": "doc2", "filename": "doc2.pdf"})
    assert registry.count() == 2

    assert registry.delete("doc1") is True
    assert registry.delete("non_existent") is False
    assert registry.count() == 1
    assert registry.get("doc1") is None
    assert registry.get("doc2") is not None

    # Test persistence of delete
    reloaded = DocumentRegistry(registry_path)
    assert reloaded.count() == 1
    assert reloaded.get("doc1") is None

    # Test clear
    reloaded.clear()
    assert reloaded.count() == 0
    assert DocumentRegistry(registry_path).count() == 0
