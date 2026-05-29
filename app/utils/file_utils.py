"""
file_utils.py — File Handling Utilities

Provides secure file validation, sanitization, and management
for PDF uploads in the Arabic RAG system.
"""

import os
import re
import uuid
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv

load_dotenv()

# ============================================================
# Configuration
# ============================================================

UPLOAD_DIR = os.getenv("UPLOAD_DIR", "./uploads")
MAX_FILE_SIZE_MB = int(os.getenv("MAX_FILE_SIZE_MB", "20"))
MAX_FILE_SIZE_BYTES = MAX_FILE_SIZE_MB * 1024 * 1024

ALLOWED_EXTENSIONS = {".pdf"}
ALLOWED_MIME_TYPES = {"application/pdf"}

# PDF magic bytes: %PDF
PDF_MAGIC_BYTES = b"%PDF"


class FileValidationError(Exception):
    """Raised when file validation fails."""

    def __init__(self, message: str, error_code: str = "VALIDATION_ERROR"):
        self.message = message
        self.error_code = error_code
        super().__init__(self.message)


def sanitize_filename(filename: str) -> str:
    """
    Sanitize a filename to prevent path traversal and injection attacks.

    Args:
        filename: Original filename from upload.

    Returns:
        Sanitized filename safe for filesystem use.
    """
    # Extract just the filename (no directory components)
    filename = os.path.basename(filename)

    # Remove null bytes
    filename = filename.replace("\x00", "")

    # Replace problematic characters with underscores
    # Allow Arabic characters, alphanumeric, dots, hyphens, underscores
    filename = re.sub(r"[^\w\u0600-\u06FF\u0750-\u077F.\-]", "_", filename)

    # Prevent hidden files
    filename = filename.lstrip(".")

    # Limit length
    name, ext = os.path.splitext(filename)
    if len(name) > 200:
        name = name[:200]

    # Ensure we have a filename
    if not name:
        name = f"document_{uuid.uuid4().hex[:8]}"

    return f"{name}{ext}"


def validate_pdf_file(
    file_content: bytes,
    filename: str,
    content_type: Optional[str] = None,
    max_size_bytes: Optional[int] = None,
) -> None:
    """
    Validate an uploaded PDF file for security and correctness.

    Args:
        file_content: Raw file bytes.
        filename: Original filename.
        content_type: Optional MIME type reported by the client.
        max_size_bytes: Maximum allowed file size in bytes.

    Raises:
        FileValidationError: If validation fails.
    """
    if max_size_bytes is None:
        max_size_bytes = MAX_FILE_SIZE_BYTES

    # Check file is not empty
    if not file_content:
        raise FileValidationError(
            "الملف فارغ. يرجى تحميل ملف PDF صالح.",  # "File is empty"
            error_code="EMPTY_FILE",
        )

    # Check file size
    if len(file_content) > max_size_bytes:
        raise FileValidationError(
            f"حجم الملف يتجاوز الحد المسموح ({MAX_FILE_SIZE_MB} ميغابايت).",
            error_code="FILE_TOO_LARGE",
        )

    # Check file extension
    _, ext = os.path.splitext(filename)
    if ext.lower() not in ALLOWED_EXTENSIONS:
        raise FileValidationError(
            f"نوع الملف غير مدعوم. الأنواع المسموحة: {', '.join(ALLOWED_EXTENSIONS)}",
            error_code="INVALID_EXTENSION",
        )

    if content_type and content_type not in ALLOWED_MIME_TYPES:
        raise FileValidationError(
            "نوع محتوى الملف غير مدعوم. يجب أن يكون application/pdf.",
            error_code="INVALID_MIME_TYPE",
        )

    # Check magic bytes (PDF header)
    if not file_content[:4].startswith(PDF_MAGIC_BYTES):
        raise FileValidationError(
            "الملف ليس ملف PDF صالح.",  # "File is not a valid PDF"
            error_code="INVALID_PDF",
        )


def save_upload(
    file_content: bytes,
    filename: str,
    upload_dir: Optional[str] = None,
) -> Path:
    """
    Save an uploaded file to the upload directory.

    Args:
        file_content: Raw file bytes.
        filename: Original filename (will be sanitized).
        upload_dir: Directory to save to (defaults to UPLOAD_DIR).

    Returns:
        Path to the saved file.
    """
    if upload_dir is None:
        upload_dir = UPLOAD_DIR

    # Ensure upload directory exists
    os.makedirs(upload_dir, exist_ok=True)

    # Sanitize filename and add UUID prefix for uniqueness
    safe_name = sanitize_filename(filename)
    unique_name = f"{uuid.uuid4().hex[:8]}_{safe_name}"
    file_path = Path(upload_dir) / unique_name

    # Write file
    with open(file_path, "wb") as f:
        f.write(file_content)

    return file_path


def delete_upload(file_path: str) -> bool:
    """
    Delete an uploaded file safely.

    Args:
        file_path: Path to the file to delete.

    Returns:
        True if deleted, False if file not found.
    """
    try:
        path = Path(file_path)
        if path.exists() and path.is_file():
            path.unlink()
            return True
        return False
    except OSError:
        return False


def list_uploaded_files(upload_dir: Optional[str] = None) -> list[dict]:
    """
    List all uploaded PDF files with metadata.

    Args:
        upload_dir: Directory to scan (defaults to UPLOAD_DIR).

    Returns:
        List of dicts with filename, size, and modified time.
    """
    if upload_dir is None:
        upload_dir = UPLOAD_DIR

    upload_path = Path(upload_dir)
    if not upload_path.exists():
        return []

    files = []
    for f in upload_path.glob("*.pdf"):
        stat = f.stat()
        files.append(
            {
                "filename": f.name,
                "size_bytes": stat.st_size,
                "size_mb": round(stat.st_size / (1024 * 1024), 2),
                "modified": stat.st_mtime,
                "path": str(f),
            }
        )

    return sorted(files, key=lambda x: x["modified"], reverse=True)
