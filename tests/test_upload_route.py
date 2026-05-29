import asyncio

from langchain_core.documents import Document

from app.api import routes
from app.services.document_registry import DocumentRegistry
from app.utils import file_utils


class FakeUpload:
    filename = "same-name.pdf"
    content_type = "application/pdf"

    async def read(self):
        return b"%PDF-1.7\nfake content"


class FakeExtractor:
    def extract(self, file_path):
        return {
            "pages": [{"page": 1, "text": "هذا نص عربي للاختبار", "char_count": 21}],
            "page_count": 1,
        }


class FakeChunker:
    def chunk_pages(self, pages, source_filename):
        return [
            Document(
                page_content=pages[0]["text"],
                metadata={"source": source_filename, "page": 1},
            )
        ]


class FakeVectorStore:
    def __init__(self):
        self.calls = []

    def add_documents(self, documents, collection_name):
        self.calls.append((documents, collection_name))
from fastapi import Request

def test_upload_uses_unique_collection_and_persists_registry(monkeypatch, tmp_path):
    vector_store = FakeVectorStore()
    registry = DocumentRegistry(tmp_path / "registry.json")
    
    mock_request = Request(scope={
        "type": "http", 
        "client": ("127.0.0.1", 12345), 
        "headers": [],
        "path": "/upload",
        "method": "POST",
        "query_string": b"",
    })

    monkeypatch.setattr(file_utils, "UPLOAD_DIR", str(tmp_path / "uploads"))
    
    first = asyncio.run(routes.upload_document(
        request=mock_request,
        file=FakeUpload(),
        extractor=FakeExtractor(),
        chunker=FakeChunker(),
        vector_store=vector_store,
        registry=registry,
    ))
    
    second = asyncio.run(routes.upload_document(
        request=mock_request,
        file=FakeUpload(),
        extractor=FakeExtractor(),
        chunker=FakeChunker(),
        vector_store=vector_store,
        registry=registry,
    ))

    assert first.document_id != second.document_id
    assert vector_store.calls[0][1] != vector_store.calls[1][1]
    assert vector_store.calls[0][1].startswith("doc_")
    assert registry.get(first.document_id)["filename"] == "same-name.pdf"
    assert vector_store.calls[0][0][0].metadata["document_id"] == first.document_id
