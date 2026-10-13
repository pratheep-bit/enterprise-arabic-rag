import asyncio
from unittest.mock import MagicMock
import pytest
from fastapi import HTTPException

from app.api import routes
from app.services.document_registry import DocumentRegistry


def test_delete_document_success(tmp_path):
    registry = DocumentRegistry(tmp_path / "registry.json")
    dummy_file = tmp_path / "test.pdf"
    dummy_file.write_text("dummy")

    registry.upsert({
        "document_id": "test_id_123",
        "filename": "test.pdf",
        "collection_name": "doc_test_id_123",
        "file_path": str(dummy_file),
    })

    fake_vector_store = MagicMock()
    fake_vector_store.delete_collection.return_value = True

    response = asyncio.run(
        routes.delete_document(
            document_id="test_id_123",
            registry=registry,
            vector_store=fake_vector_store,
        )
    )

    assert response.status == "deleted"
    assert response.document_id == "test_id_123"
    assert registry.get("test_id_123") is None
    assert not dummy_file.exists()
    fake_vector_store.delete_collection.assert_called_once_with("doc_test_id_123")


def test_delete_document_not_found(tmp_path):
    registry = DocumentRegistry(tmp_path / "registry.json")
    fake_vector_store = MagicMock()
    fake_vector_store.get_collection_info.return_value = None

    with pytest.raises(HTTPException) as exc_info:
        asyncio.run(
            routes.delete_document(
                document_id="nonexistent_id",
                registry=registry,
                vector_store=fake_vector_store,
            )
        )
    assert exc_info.value.status_code == 404
