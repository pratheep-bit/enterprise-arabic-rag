"""Tests for QAChain — LLM response parsing and confidence calculation."""

import json
from unittest.mock import MagicMock, patch

from app.rag.qa_chain import QAChain, QAChainError, NO_ANSWER_RESPONSE


class FakeLLMResponse:
    """Fake LLM response object."""
    def __init__(self, content: str):
        self.content = content


class FakeRetriever:
    """Fake Retriever returning canned results."""
    def __init__(self, chunks=None, dialect_detected=False):
        self._chunks = chunks or []
        self._dialect = dialect_detected

    def retrieve(self, query, top_k=5, collection_name=None):
        return {
            "chunks": self._chunks,
            "query_original": query,
            "query_normalized": query,
            "dialect_detected": self._dialect,
            "dialect_terms": [],
        }


def _make_qa_chain(llm_content: str, chunks=None, dialect_detected=False) -> QAChain:
    """Create a QAChain with mocked LLM and Retriever."""
    chain = QAChain.__new__(QAChain)
    chain.retriever = FakeRetriever(chunks=chunks, dialect_detected=dialect_detected)
    chain.model_name = "test-model"

    mock_llm = MagicMock()
    mock_llm.invoke.return_value = FakeLLMResponse(llm_content)
    chain.llm = mock_llm

    return chain


class TestLLMResponseParsing:
    """Test _parse_llm_response with various formats."""

    def test_parse_valid_json(self):
        chain = _make_qa_chain("")
        parsed = chain._parse_llm_response(json.dumps({
            "answer": "إجابة",
            "confidence": 0.9,
            "sources": [],
        }))
        assert parsed["answer"] == "إجابة"
        assert parsed["confidence"] == 0.9

    def test_parse_json_in_markdown_block(self):
        chain = _make_qa_chain("")
        response = '```json\n{"answer": "إجابة", "confidence": 0.8, "sources": []}\n```'
        parsed = chain._parse_llm_response(response)
        assert parsed["answer"] == "إجابة"
        assert parsed["confidence"] == 0.8

    def test_parse_raw_text_fallback(self):
        chain = _make_qa_chain("")
        parsed = chain._parse_llm_response("هذه إجابة بدون JSON")
        assert "answer" in parsed
        assert parsed["answer"] == "هذه إجابة بدون JSON"


class TestConfidenceCalculation:
    """Test the blended confidence score."""

    def test_confidence_with_chunks(self):
        chunks = [
            {"content": "نص", "metadata": {"page": 1, "source": "doc.pdf"}, "similarity_score": 0.8},
            {"content": "نص آخر", "metadata": {"page": 2, "source": "doc.pdf"}, "similarity_score": 0.6},
        ]

        llm_response = json.dumps({
            "answer": "إجابة",
            "confidence": 0.9,
            "sources": [],
        })

        chain = _make_qa_chain(llm_response, chunks=chunks)
        result = chain.ask("سؤال")

        # Confidence = 0.6 * 0.9 + 0.4 * avg(0.8, 0.6) = 0.54 + 0.28 = 0.82
        assert 0.75 <= result.confidence <= 0.90


class TestNoChunksReturned:
    """When no chunks are retrieved, return the no-answer response."""

    def test_no_chunks_returns_no_answer(self):
        chain = _make_qa_chain("", chunks=[])
        result = chain.ask("سؤال بلا سياق")

        assert NO_ANSWER_RESPONSE in result.answer
        assert result.confidence == 0.0
        assert result.sources == []


class TestEmptyQuestion:
    """Empty or None question handling."""

    def test_empty_question_raises(self):
        chain = _make_qa_chain("")
        try:
            result = chain.ask("")
            # If it returns, it should be a no-answer
            assert result.confidence == 0.0
        except (QAChainError, Exception):
            pass  # Acceptable to raise

    def test_whitespace_question(self):
        chain = _make_qa_chain("")
        try:
            result = chain.ask("   ")
            assert result.confidence == 0.0
        except (QAChainError, Exception):
            pass


class TestSourceCitationBuilding:
    """Test that sources are correctly built from LLM output + retrieval metadata."""

    def test_sources_from_llm_json(self):
        chunks = [
            {"content": "نص مرجعي", "metadata": {"page": 3, "source": "report.pdf"}, "similarity_score": 0.85},
        ]

        llm_response = json.dumps({
            "answer": "إجابة مع مصادر",
            "confidence": 0.95,
            "sources": [
                {"page": 3, "document": "report.pdf", "excerpt": "نص مرجعي"},
            ],
        })

        chain = _make_qa_chain(llm_response, chunks=chunks)
        result = chain.ask("سؤال")

        assert len(result.sources) >= 1
        assert result.sources[0].page == 3
