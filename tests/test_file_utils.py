import pytest

from app.utils.file_utils import FileValidationError, sanitize_filename, validate_pdf_file


def test_sanitize_filename_removes_paths_and_hidden_prefix():
    assert sanitize_filename("../../.secret.pdf") == "secret.pdf"


def test_validate_pdf_file_rejects_wrong_mime_type():
    with pytest.raises(FileValidationError) as exc:
        validate_pdf_file(
            b"%PDF-1.7\n",
            "document.pdf",
            content_type="text/plain",
        )

    assert exc.value.error_code == "INVALID_MIME_TYPE"


def test_validate_pdf_file_accepts_valid_pdf_signature_and_mime():
    validate_pdf_file(
        b"%PDF-1.7\nbody",
        "document.pdf",
        content_type="application/pdf",
    )
