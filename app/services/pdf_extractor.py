"""
pdf_extractor.py — Arabic PDF Text Extraction Service

Uses PyMuPDF (fitz) to extract text from PDF documents,
applying Arabic text normalization per page.
"""

import logging
from pathlib import Path
from typing import Optional

import fitz  # PyMuPDF

from app.utils.arabic_processor import ArabicProcessor

logger = logging.getLogger(__name__)


class PDFExtractionError(Exception):
    """Raised when PDF extraction fails."""

    def __init__(self, message: str, error_code: str = "PDF_ERROR"):
        self.message = message
        self.error_code = error_code
        super().__init__(self.message)


class PDFExtractor:
    """
    Extracts and normalizes Arabic text from PDF documents using PyMuPDF.

    Features:
    - Page-by-page text extraction
    - Arabic text normalization
    - RTL text handling
    - Metadata extraction (page count, author, title)
    - Error handling for corrupt/empty PDFs
    - Placeholder hook for OCR fallback
    """

    def __init__(
        self,
        arabic_processor: Optional[ArabicProcessor] = None,
        min_text_length: int = 10,
    ):
        """
        Initialize the PDF extractor.

        Args:
            arabic_processor: Custom ArabicProcessor instance (uses default if None).
            min_text_length: Minimum characters per page to consider non-empty.
        """
        self.processor = arabic_processor or ArabicProcessor()
        self.min_text_length = min_text_length

    def extract(self, file_path: str) -> dict:
        """
        Extract text from a PDF file.

        Args:
            file_path: Path to the PDF file.

        Returns:
            Dict with keys:
                - pages: List[dict] with {page: int, text: str, char_count: int}
                - metadata: dict with PDF metadata
                - page_count: int
                - total_chars: int
                - empty_pages: List[int] (pages with no extractable text)

        Raises:
            PDFExtractionError: If extraction fails.
        """
        path = Path(file_path)

        # Validate file exists
        if not path.exists():
            raise PDFExtractionError(
                f"الملف غير موجود: {file_path}",
                error_code="FILE_NOT_FOUND",
            )

        if not path.suffix.lower() == ".pdf":
            raise PDFExtractionError(
                "نوع الملف غير مدعوم. يجب أن يكون PDF.",
                error_code="INVALID_FILE_TYPE",
            )

        try:
            doc = fitz.open(str(path))
        except Exception as e:
            logger.error(f"Failed to open PDF '{path.name}': {str(e)}", exc_info=True)
            raise PDFExtractionError(
                "فشل فتح ملف PDF.",
                error_code="PDF_OPEN_ERROR",
            )

        try:
            pages = []
            empty_pages = []
            total_chars = 0

            for page_num in range(len(doc)):
                page = doc[page_num]

                # Extract text from the page. PyMuPDF's plain text mode often
                # emits Arabic words left-to-right, so rebuild lines from words.
                raw_text = self._extract_page_text(page)

                if raw_text and len(raw_text.strip()) >= self.min_text_length:
                    # Apply Arabic normalization
                    processed_text = self.processor.process(raw_text)

                    if processed_text:
                        char_count = len(processed_text)
                        total_chars += char_count

                        pages.append(
                            {
                                "page": page_num + 1,  # 1-indexed
                                "text": processed_text,
                                "char_count": char_count,
                            }
                        )
                    else:
                        empty_pages.append(page_num + 1)
                else:
                    empty_pages.append(page_num + 1)

                    # OCR fallback hook — for future implementation
                    ocr_text = self._ocr_fallback(page)
                    if ocr_text:
                        processed_text = self.processor.process(ocr_text)
                        char_count = len(processed_text)
                        total_chars += char_count
                        pages.append(
                            {
                                "page": page_num + 1,
                                "text": processed_text,
                                "char_count": char_count,
                            }
                        )
                        empty_pages.pop()  # Remove from empty list

            # Extract document metadata
            metadata = self._extract_metadata(doc)
            page_count = len(doc)
            doc.close()

            if not pages:
                raise PDFExtractionError(
                    "لم يتم العثور على نص قابل للاستخراج في المستند.",
                    error_code="NO_TEXT_FOUND",
                )

            logger.info(
                f"Extracted {len(pages)} pages from {path.name} "
                f"({total_chars} total chars, {len(empty_pages)} empty pages)"
            )

            return {
                "pages": pages,
                "metadata": metadata,
                "page_count": page_count,
                "total_chars": total_chars,
                "empty_pages": empty_pages,
            }

        except PDFExtractionError:
            if 'doc' in locals() and not doc.is_closed:
                doc.close()
            raise
        except Exception as e:
            if 'doc' in locals() and not doc.is_closed:
                doc.close()
            logger.error(f"Unexpected PDF extraction error for '{path.name}': {str(e)}", exc_info=True)
            raise PDFExtractionError(
                "خطأ غير متوقع أثناء استخراج النص.",
                error_code="EXTRACTION_ERROR",
            )

    def extract_text_only(self, file_path: str) -> str:
        """
        Extract all text from a PDF as a single string.
        Convenience method for simple use cases.

        Args:
            file_path: Path to the PDF file.

        Returns:
            Complete extracted text as a single string.
        """
        result = self.extract(file_path)
        return "\n\n".join(page["text"] for page in result["pages"])

    @staticmethod
    def _extract_metadata(doc: fitz.Document) -> dict:
        """
        Extract PDF metadata (title, author, etc.).

        Args:
            doc: Open PyMuPDF Document object.

        Returns:
            Dict of metadata fields.
        """
        meta = doc.metadata or {}
        return {
            "title": meta.get("title", ""),
            "author": meta.get("author", ""),
            "subject": meta.get("subject", ""),
            "creator": meta.get("creator", ""),
            "producer": meta.get("producer", ""),
            "creation_date": meta.get("creationDate", ""),
            "modification_date": meta.get("modDate", ""),
        }

    @staticmethod
    def _extract_page_text(page: fitz.Page) -> str:
        """Extract text while preserving right-to-left Arabic word order."""
        words = page.get_text("words")
        if not words:
            return page.get_text("text", sort=True)

        arabic_chars = sum(
            1
            for word in words
            for char in word[4]
            if "\u0600" <= char <= "\u06FF" or "\u0750" <= char <= "\u077F"
        )
        total_chars = sum(len(word[4]) for word in words)
        rtl_page = total_chars > 0 and arabic_chars / total_chars > 0.25

        lines: dict[tuple[int, int], list[tuple[float, str]]] = {}
        line_positions: dict[tuple[int, int], tuple[float, float]] = {}
        for word in words:
            x0, y0, _x1, _y1, text, block_no, line_no, _word_no = word
            key = (block_no, line_no)
            lines.setdefault(key, []).append((x0, text))
            line_positions.setdefault(key, (y0, x0))

        ordered_lines = sorted(lines, key=lambda key: line_positions[key])
        output_lines = []
        for key in ordered_lines:
            line_words = lines[key]
            has_arabic = any(
                "\u0600" <= char <= "\u06FF" or "\u0750" <= char <= "\u077F"
                for _x, text in line_words
                for char in text
            )
            ordered_words = sorted(
                line_words,
                key=lambda item: item[0],
                reverse=rtl_page and has_arabic,
            )
            output_lines.append(" ".join(text for _x, text in ordered_words))

        return "\n".join(output_lines)

    @staticmethod
    def _ocr_fallback(page: fitz.Page) -> Optional[str]:
        """
        OCR fallback hook for image-only pages.
        
        This is an extension point for Tesseract or other OCR integration.
        Override this method to enable OCR support.

        Args:
            page: PyMuPDF page object.

        Returns:
            Extracted text from OCR, or None if no OCR backend is configured.
        """
        # Optional integration point:
        # import pytesseract
        # from PIL import Image
        # pix = page.get_pixmap(dpi=300)
        # img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        # text = pytesseract.image_to_string(img, lang="ara")
        # return text if text.strip() else None
        return None
