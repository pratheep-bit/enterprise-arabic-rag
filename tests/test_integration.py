"""Integration test — full upload → ask pipeline with mocked external services."""

import asyncio
import json
from unittest.mock import MagicMock, patch

from langchain_core.documents import Document

from app.api import routes
from app.services.document_registry import DocumentRegistry


# ============================================================
# Fakes
# ============================================================


class FakeUpload:
    """Fake UploadFile for testing."""
    filename = "integration_test.pdf"
    content_type = "application/pdf"

    async def read(self):
        return b"%PDF-1.7\nfake pdf content for integration test"


class FakeExtractor:
    def extract(self, file_path):
        return {
            "pages": [
                {"page": 1, "text": "الهدف الرئيسي للمشروع هو تطوير نظام ذكاء اصطناعي", "char_count": 45},
                {"page": 2, "text": "المرحلة الثانية تشمل التوسع في الأسواق الخليجية", "char_count": 44},
            ],
            "page_count": 2,
        }


class FakeChunker:
    def chunk_pages(self, pages, source_filename):
        return [
            Document(
                page_content=page["text"],
                metadata={"source": source_filename, "page": page["page"]},
            )
            for page in pages
        ]


class FakeVectorStore:
    def __init__(self):
        self.stored = []

    def add_documents(self, documents, collection_name):
        self.stored.extend(documents)
        return collection_name

    def search(self, query, top_k=5, collection_name=None):
        return [
            {
                "content": doc.page_content,
                "metadata": doc.metadata,
                "similarity_score": 0.85,
                "vector_score": 0.80,
            }
            for doc in self.stored[:top_k]
        ]

    def list_collections(self):
        return []

    def get_collection_info(self, name):
        return None


class FakeLLMResponse:
    def __init__(self):
        self.content = json.dumps({
            "answer": "الهدف الرئيسي هو تطوير نظام ذكاء اصطناعي",
            "confidence": 0.9,
            "sources": [
                {"page": 1, "document": "integration_test.pdf", "excerpt": "الهدف الرئيسي للمشروع"}
            ],
        })


# ============================================================
# Integration Test
# ============================================================


class TestUploadThenAsk:
    """Full pipeline: upload a document, then ask a question."""

    def test_upload_then_ask_pipeline(self, monkeypatch, tmp_path):
        """Upload a PDF, then ask a question and verify end-to-end response."""
        from app.utils import file_utils

        vector_store = FakeVectorStore()
        registry = DocumentRegistry(tmp_path / "registry.json")

        # Patch service factories
        monkeypatch.setattr(file_utils, "UPLOAD_DIR", str(tmp_path / "uploads"))
        monkeypatch.setattr(routes, "_document_registry", registry)
        monkeypatch.setattr(routes, "_pdf_extractor", FakeExtractor())
        monkeypatch.setattr(routes, "_chunker", FakeChunker())
        monkeypatch.setattr(routes, "_vector_store", vector_store)

        # --- Step 1: Upload ---
        from fastapi import Request
        fake_request = Request(scope={
            "type": "http", 
            "client": ("127.0.0.1", 12345), 
            "headers": [],
            "path": "/upload",
            "method": "POST",
            "query_string": b"",
        })
        
        upload_result = asyncio.run(
            routes.upload_document(
                request=fake_request,
                file=FakeUpload(),
                extractor=FakeExtractor(),
                chunker=FakeChunker(),
                vector_store=vector_store,
                registry=registry,
            )
        )

        assert upload_result.status == "success"
        assert upload_result.page_count == 2
        assert upload_result.chunk_count == 2
        doc_id = upload_result.document_id

        # Verify persistence
        assert registry.get(doc_id) is not None
        assert registry.get(doc_id)["filename"] == "integration_test.pdf"

        # Verify vector store received documents
        assert len(vector_store.stored) == 2

        # --- Step 2: Ask ---
        # Mock the QAChain to avoid real LLM calls
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = FakeLLMResponse()

        from app.rag.qa_chain import QAChain
        qa_chain = QAChain.__new__(QAChain)
        qa_chain.model_name = "test-model"
        qa_chain.llm = mock_llm

        from app.rag.retriever import Retriever
        qa_chain.retriever = Retriever(
            vector_store=vector_store,
        )

        from app.models.schemas import QuestionRequest
        question = QuestionRequest(question="ما هو الهدف الرئيسي للمشروع؟")

        ask_result = asyncio.run(
            routes.ask_question(
                request=fake_request,
                body=question,
                qa_chain=qa_chain,
                vector_store=vector_store,
                registry=registry,
            )
        )

        assert ask_result.answer is not None
        assert len(ask_result.answer) > 0
        assert ask_result.confidence > 0.0
        assert len(ask_result.sources) >= 1
