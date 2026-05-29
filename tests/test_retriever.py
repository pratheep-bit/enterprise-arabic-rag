"""Tests for Retriever — multi-query retrieval and deduplication."""

from unittest.mock import MagicMock

from app.rag.retriever import Retriever


class FakeVectorStore:
    """Fake VectorStore that returns canned search results."""

    def __init__(self, results: list[dict] = None):
        self.results = results or []
        self.search_calls = []

    def search(self, query: str, top_k: int = 5, collection_name=None):
        self.search_calls.append(query)
        return list(self.results)


class TestRetrieverEmptyQuery:
    """Handle empty or whitespace queries."""

    def test_empty_query_returns_empty(self):
        retriever = Retriever(vector_store=FakeVectorStore())
        result = retriever.retrieve("")
        assert result["chunks"] == []
        assert result["dialect_detected"] is False

    def test_whitespace_query_returns_empty(self):
        retriever = Retriever(vector_store=FakeVectorStore())
        result = retriever.retrieve("   ")
        assert result["chunks"] == []


class TestRetrieverMSAQuery:
    """MSA-only query (no dialect expansion)."""

    def test_msa_query_single_search(self):
        store = FakeVectorStore(results=[
            {"content": "الهدف الأول", "metadata": {"page": 1}, "similarity_score": 0.8},
        ])
        retriever = Retriever(vector_store=store)
        result = retriever.retrieve("الأهداف الرئيسية للمشروع")

        assert len(result["chunks"]) == 1
        assert result["dialect_detected"] is False

    def test_msa_query_preserves_original(self):
        store = FakeVectorStore(results=[])
        retriever = Retriever(vector_store=store)
        result = retriever.retrieve("الأهداف الرئيسية للمشروع")
        assert result["query_original"] == "الأهداف الرئيسية للمشروع"


class TestRetrieverDialectExpansion:
    """Dialect query triggers multi-query search."""

    def test_gulf_dialect_expands_query(self):
        store = FakeVectorStore(results=[
            {"content": "معلومات مهمة", "metadata": {"page": 1}, "similarity_score": 0.7},
        ])
        retriever = Retriever(vector_store=store)
        result = retriever.retrieve("شلون الأمور")

        assert result["dialect_detected"] is True
        # Should have searched with at least 2 variants (original + MSA)
        assert len(store.search_calls) >= 2


class TestRetrieverDeduplication:
    """Deduplication of search results."""

    def test_deduplicates_identical_content(self):
        same_result = {"content": "نص مكرر " * 30, "metadata": {"page": 1}, "similarity_score": 0.8}
        store = FakeVectorStore(results=[same_result, same_result])
        retriever = Retriever(vector_store=store)
        result = retriever.retrieve("سؤال بسيط")

        assert len(result["chunks"]) == 1

    def test_dedup_keeps_highest_score(self):
        low = {"content": "نص مكرر " * 30, "metadata": {"page": 1}, "similarity_score": 0.5}
        high = {"content": "نص مكرر " * 30, "metadata": {"page": 1}, "similarity_score": 0.9}

        results = Retriever._deduplicate_results([low, high])
        assert len(results) == 1
        assert results[0]["similarity_score"] == 0.9


class TestRetrieverTopK:
    """Top-k limiting."""

    def test_limits_to_top_k(self):
        results = [
            {"content": f"نص رقم {i} " * 30, "metadata": {"page": i}, "similarity_score": 0.9 - i * 0.1}
            for i in range(10)
        ]
        store = FakeVectorStore(results=results)
        retriever = Retriever(vector_store=store)
        result = retriever.retrieve("سؤال", top_k=3)

        assert len(result["chunks"]) <= 3
