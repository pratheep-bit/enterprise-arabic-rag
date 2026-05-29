"""Tests for VectorStore — collection sanitization, hybrid scoring, and threshold."""

from app.rag.vector_store import _sanitize_collection_name, VectorStore


class TestCollectionNameSanitization:
    """Test _sanitize_collection_name for ChromaDB compliance."""

    def test_basic_name(self):
        assert _sanitize_collection_name("my_document.pdf") == "my_document"

    def test_removes_extension(self):
        result = _sanitize_collection_name("report.pdf")
        assert ".pdf" not in result

    def test_replaces_spaces(self):
        result = _sanitize_collection_name("my document.pdf")
        assert " " not in result

    def test_minimum_length_padding(self):
        result = _sanitize_collection_name("ab")
        assert len(result) >= 3

    def test_maximum_length_truncation(self):
        long_name = "a" * 100 + ".pdf"
        result = _sanitize_collection_name(long_name)
        assert len(result) <= 63

    def test_starts_with_alphanumeric(self):
        result = _sanitize_collection_name("_hidden.pdf")
        assert result[0].isalnum()

    def test_ends_with_alphanumeric(self):
        result = _sanitize_collection_name("file_.pdf")
        assert result[-1].isalnum()

    def test_arabic_filename(self):
        result = _sanitize_collection_name("تقرير_عربي.pdf")
        assert len(result) >= 3
        assert len(result) <= 63

    def test_unicode_characters_allowed(self):
        result = _sanitize_collection_name("مستند_2024.pdf")
        assert len(result) >= 3


class TestHybridScoring:
    """Test _hybrid_score static method."""

    def test_vector_only_no_arabic(self):
        score = VectorStore._hybrid_score(
            query="test",
            content="some content",
            vector_score=0.7,
        )
        # No Arabic terms → should return just the vector score
        assert score == 0.7

    def test_lexical_bonus_on_match(self):
        query = "أهداف المشروع الرئيسية"
        content = "أهداف المشروع الرئيسية هي تطوير النظام"
        score = VectorStore._hybrid_score(
            query=query,
            content=content,
            vector_score=0.5,
        )
        # Should be higher than 0.5 due to lexical match
        assert score > 0.5

    def test_score_capped_at_one(self):
        query = "أهداف المشروع"
        content = "أهداف المشروع الرئيسية"
        score = VectorStore._hybrid_score(
            query=query,
            content=content,
            vector_score=0.95,
        )
        assert score <= 1.0


class TestArabicTermExtraction:
    """Test _terms static method for prefix stripping."""

    def test_extracts_arabic_terms(self):
        terms = VectorStore._terms("أهداف المشروع الرئيسية")
        assert len(terms) > 0

    def test_strips_common_prefixes(self):
        terms = VectorStore._terms("والمشروع بالتفصيل")
        # "وال" prefix should be stripped from "والمشروع"
        assert all(not t.startswith("وال") for t in terms)

    def test_filters_stopwords(self):
        terms = VectorStore._terms("هذا هو المشروع")
        # "هذا" and "هو" are stopwords
        assert "هذا" not in terms
        assert "هو" not in terms

    def test_empty_input(self):
        terms = VectorStore._terms("")
        assert terms == []
