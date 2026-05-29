"""
chunker.py — Arabic-Aware Text Chunking Engine

Chunks Arabic text while respecting sentence boundaries, paragraph structure,
and Arabic punctuation rules. Uses LangChain's RecursiveCharacterTextSplitter
with custom Arabic separators.
"""

import logging
import os
from typing import Optional

from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

load_dotenv()

logger = logging.getLogger(__name__)

# ============================================================
# Configuration from environment
# ============================================================

DEFAULT_CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "800"))
DEFAULT_CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "150"))

# ============================================================
# Arabic-specific separators (priority order)
# ============================================================

ARABIC_SEPARATORS = [
    # 1. Paragraph breaks (highest priority)
    "\n\n",

    # 2. Arabic full stop (period equivalent)
    "。",  # CJK full stop — rarely used, lowest priority among punctuation

    # 3. Arabic sentence-ending punctuation
    ".",    # Latin period (commonly used in Arabic texts)
    "؟",   # Arabic question mark
    "！",  # Fullwidth exclamation
    "!",   # ASCII exclamation
    ":",   # Colon (often used in formal Arabic)
    "؛",   # Arabic semicolon
    "،",   # Arabic comma

    # 4. Newline (within paragraphs)
    "\n",

    # 5. Fallback: space-based splitting
    " ",

    # 6. Ultimate fallback: character-level
    "",
]


class ArabicChunker:
    """
    Arabic-aware text chunker for RAG pipelines.

    Features:
    - Respects Arabic sentence boundaries
    - Preserves paragraph structure
    - Configurable chunk size and overlap
    - Maintains page-number metadata per chunk
    - Uses priority-ordered Arabic separators
    """

    def __init__(
        self,
        chunk_size: int = DEFAULT_CHUNK_SIZE,
        chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
        separators: Optional[list[str]] = None,
    ):
        """
        Initialize the Arabic chunker.

        Args:
            chunk_size: Maximum characters per chunk.
            chunk_overlap: Number of overlapping characters between consecutive chunks.
            separators: Custom list of separators (uses Arabic defaults if None).
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.separators = separators or ARABIC_SEPARATORS

        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            separators=self.separators,
            length_function=len,
            is_separator_regex=False,
            keep_separator=True,
        )

    def chunk_pages(
        self,
        pages: list[dict],
        source_filename: str,
    ) -> list[Document]:
        """
        Chunk extracted pages into LangChain Documents with metadata.

        This method processes page-by-page output from the PDF extractor,
        maintaining page-number attribution through the chunking process.

        Args:
            pages: List of dicts from PDFExtractor: [{"page": int, "text": str}, ...]
            source_filename: Name of the source PDF file.

        Returns:
            List of LangChain Document objects with metadata:
                - source: filename
                - page: page number
                - chunk_index: sequential chunk index
                - chunk_size: character count of this chunk
        """
        all_chunks: list[Document] = []
        global_chunk_index = 0

        for page_data in pages:
            page_num = page_data["page"]
            text = page_data["text"]

            if not text or not text.strip():
                continue

            # Split this page's text into chunks
            page_chunks = self.splitter.create_documents(
                texts=[text],
                metadatas=[
                    {
                        "source": source_filename,
                        "page": page_num,
                    }
                ],
            )

            # Add chunk index and size metadata
            for chunk in page_chunks:
                chunk.metadata["chunk_index"] = global_chunk_index
                chunk.metadata["chunk_size"] = len(chunk.page_content)
                all_chunks.append(chunk)
                global_chunk_index += 1

        logger.info(
            f"Chunked {len(pages)} pages from '{source_filename}' "
            f"into {len(all_chunks)} chunks "
            f"(chunk_size={self.chunk_size}, overlap={self.chunk_overlap})"
        )

        return all_chunks

    def chunk_text(
        self,
        text: str,
        metadata: Optional[dict] = None,
    ) -> list[Document]:
        """
        Chunk a single text string into Documents.

        Args:
            text: Text to chunk.
            metadata: Optional metadata to attach to all chunks.

        Returns:
            List of LangChain Document objects.
        """
        if not text or not text.strip():
            return []

        base_metadata = metadata or {}

        chunks = self.splitter.create_documents(
            texts=[text],
            metadatas=[base_metadata],
        )

        for i, chunk in enumerate(chunks):
            chunk.metadata["chunk_index"] = i
            chunk.metadata["chunk_size"] = len(chunk.page_content)

        return chunks

    def estimate_chunk_count(self, text: str) -> int:
        """
        Estimate the number of chunks that will be generated from text.
        Useful for progress estimation.

        Args:
            text: Text to estimate chunks for.

        Returns:
            Estimated number of chunks.
        """
        if not text:
            return 0

        text_len = len(text)
        if text_len <= self.chunk_size:
            return 1

        # Approximate: (total_length - overlap) / (chunk_size - overlap)
        effective_size = self.chunk_size - self.chunk_overlap
        if effective_size <= 0:
            effective_size = self.chunk_size

        return max(1, (text_len + effective_size - 1) // effective_size)


# ============================================================
# Module-level convenience
# ============================================================

_default_chunker = ArabicChunker()


def chunk_arabic_pages(pages: list[dict], source_filename: str) -> list[Document]:
    """Convenience: chunk pages using default settings."""
    return _default_chunker.chunk_pages(pages, source_filename)
