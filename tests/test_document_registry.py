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
