"""Tests for ArabicChunker — Arabic-aware text chunking."""

from app.rag.chunker import ArabicChunker


chunker = ArabicChunker()


class TestSinglePageChunking:
    """Chunking a single page of text."""

    def test_short_text_single_chunk(self):
        pages = [{"page": 1, "text": "هذا نص قصير.", "char_count": 12}]
        chunks = chunker.chunk_pages(pages, source_filename="test.pdf")
        assert len(chunks) >= 1
        assert chunks[0].metadata["source"] == "test.pdf"
        assert chunks[0].metadata["page"] == 1

    def test_long_text_multiple_chunks(self):
        long_text = "هذا نص طويل جداً. " * 100  # ~1900 chars
        pages = [{"page": 1, "text": long_text, "char_count": len(long_text)}]
        chunks = chunker.chunk_pages(pages, source_filename="test.pdf")
        assert len(chunks) > 1

    def test_arabic_punctuation_as_separator(self):
        """Arabic question mark should be used as a split boundary."""
        text = ("أول جملة طويلة جداً " * 30 + "؟ " +
                "ثاني جملة طويلة جداً " * 30)
        pages = [{"page": 1, "text": text, "char_count": len(text)}]
        chunks = chunker.chunk_pages(pages, source_filename="test.pdf")
        assert len(chunks) > 1


class TestMultiPageChunking:
    """Chunking across multiple pages."""

    def test_multi_page_preserves_page_metadata(self):
        pages = [
            {"page": 1, "text": "محتوى الصفحة الأولى. " * 50, "char_count": 1000},
            {"page": 2, "text": "محتوى الصفحة الثانية. " * 50, "char_count": 1000},
        ]
        chunks = chunker.chunk_pages(pages, source_filename="test.pdf")
        page_numbers = {chunk.metadata["page"] for chunk in chunks}
        assert 1 in page_numbers
        assert 2 in page_numbers


class TestEmptyAndEdgeCases:
    """Edge cases: empty pages, empty list."""

    def test_empty_page_list(self):
        chunks = chunker.chunk_pages([], source_filename="test.pdf")
        assert chunks == []

    def test_page_with_empty_text(self):
        pages = [{"page": 1, "text": "", "char_count": 0}]
        chunks = chunker.chunk_pages(pages, source_filename="test.pdf")
        assert len(chunks) == 0


class TestEstimateChunkCount:
    """Test chunk count estimation."""

    def test_estimate_short_text(self):
        count = chunker.estimate_chunk_count("قصير")
        assert count == 1

    def test_estimate_long_text(self):
        long_text = "نص " * 1000  # ~3000 chars
        count = chunker.estimate_chunk_count(long_text)
        assert count > 1


class TestMetadataPreservation:
    """Ensure chunk metadata is correctly attached."""

    def test_source_filename_in_metadata(self):
        pages = [{"page": 1, "text": "مرحبا بالعالم العربي", "char_count": 20}]
        chunks = chunker.chunk_pages(pages, source_filename="arabic_doc.pdf")
        for chunk in chunks:
            assert chunk.metadata["source"] == "arabic_doc.pdf"
