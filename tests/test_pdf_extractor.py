"""Tests for PDFExtractor — extraction, validation, and error handling."""

import pytest

from app.services.pdf_extractor import PDFExtractor, PDFExtractionError


extractor = PDFExtractor()


class TestFileValidation:
    """File existence and type checks."""

    def test_file_not_found(self):
        with pytest.raises(PDFExtractionError) as exc:
            extractor.extract("/nonexistent/path/to/file.pdf")
        assert exc.value.error_code == "FILE_NOT_FOUND"

    def test_non_pdf_extension(self, tmp_path):
        txt_file = tmp_path / "document.txt"
        txt_file.write_text("hello")
        with pytest.raises(PDFExtractionError) as exc:
            extractor.extract(str(txt_file))
        assert exc.value.error_code == "INVALID_FILE_TYPE"

    def test_corrupt_pdf(self, tmp_path):
        bad_pdf = tmp_path / "corrupt.pdf"
        bad_pdf.write_bytes(b"NOT A PDF FILE AT ALL")
        with pytest.raises(PDFExtractionError):
            extractor.extract(str(bad_pdf))


class TestMetadataExtraction:
    """Test that metadata dict has expected keys."""

    def test_metadata_keys(self, tmp_path):
        """Create a minimal valid PDF and check metadata extraction."""
        import fitz
        doc = fitz.open()
        page = doc.new_page()
        page.insert_text((100, 100), "مرحبا بالعالم العربي — نص اختبار")
        pdf_path = tmp_path / "test_meta.pdf"
        doc.save(str(pdf_path))
        doc.close()

        result = extractor.extract(str(pdf_path))
        assert "metadata" in result
        meta = result["metadata"]
        assert "title" in meta
        assert "author" in meta
        assert "creator" in meta


class TestTextExtraction:
    """Test actual text extraction from a valid PDF."""

    def test_extracts_arabic_text(self, tmp_path):
        import fitz
        doc = fitz.open()
        page = doc.new_page()
        page.insert_text((100, 100), "هذا نص عربي للاختبار يحتوي على كلمات كافية")
        pdf_path = tmp_path / "arabic_test.pdf"
        doc.save(str(pdf_path))
        doc.close()

        result = extractor.extract(str(pdf_path))
        assert result["page_count"] == 1
        assert result["total_chars"] > 0
        assert len(result["pages"]) == 1

    def test_multi_page_extraction(self, tmp_path):
        import fitz
        doc = fitz.open()
        for i in range(3):
            page = doc.new_page()
            page.insert_text((100, 100), f"محتوى الصفحة {i + 1} — نص عربي طويل للاختبار")
        pdf_path = tmp_path / "multipage.pdf"
        doc.save(str(pdf_path))
        doc.close()

        result = extractor.extract(str(pdf_path))
        assert result["page_count"] == 3
        assert len(result["pages"]) == 3

    def test_empty_pdf_raises(self, tmp_path):
        """A PDF with no extractable text should raise."""
        import fitz
        doc = fitz.open()
        doc.new_page()  # Empty page
        pdf_path = tmp_path / "empty.pdf"
        doc.save(str(pdf_path))
        doc.close()

        with pytest.raises(PDFExtractionError) as exc:
            extractor.extract(str(pdf_path))
        assert exc.value.error_code == "NO_TEXT_FOUND"


class TestExtractTextOnly:
    """Test the convenience extract_text_only method."""

    def test_returns_string(self, tmp_path):
        import fitz
        doc = fitz.open()
        page = doc.new_page()
        page.insert_text((100, 100), "نص بسيط واحد يحتوي على أكثر من عشرة أحرف")
        pdf_path = tmp_path / "simple.pdf"
        doc.save(str(pdf_path))
        doc.close()

        text = extractor.extract_text_only(str(pdf_path))
        assert isinstance(text, str)
        assert len(text) > 0
