"""
schemas.py — Pydantic Request/Response Models

Defines all API request and response schemas for the Arabic RAG system.
Ensures type safety, validation, and consistent JSON serialization.
"""

from datetime import datetime
from typing import Optional

from pydantic import BaseModel, Field


# ============================================================
# Request Models
# ============================================================


class QuestionRequest(BaseModel):
    """Request model for asking a question against uploaded documents."""

    question: str = Field(
        ...,
        min_length=2,
        max_length=2000,
        description="The question to ask in Arabic (supports MSA and Gulf dialect).",
        examples=["ما هي الأهداف الرئيسية للمشروع؟"],
    )
    top_k: int = Field(
        default=5,
        ge=1,
        le=20,
        description="Number of relevant chunks to retrieve.",
    )
    document_id: Optional[str] = Field(
        default=None,
        max_length=64,
        description="Optional: restrict search to a specific document by ID.",
    )


# ============================================================
# Response Models
# ============================================================


class SourceCitation(BaseModel):
    """A single source citation from the retrieved context."""

    page: int = Field(
        ...,
        description="Page number in the source document.",
    )
    document: str = Field(
        ...,
        description="Source document filename.",
    )
    excerpt: str = Field(
        ...,
        description="Relevant text excerpt from the source.",
    )
    similarity_score: Optional[float] = Field(
        default=None,
        description="Similarity score for this chunk (0.0 to 1.0).",
    )


class AnswerResponse(BaseModel):
    """Response model for a question-answering request."""

    answer: str = Field(
        ...,
        description="The generated answer in Arabic.",
    )
    confidence: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Confidence score (0.0 = no confidence, 1.0 = fully confident).",
    )
    sources: list[SourceCitation] = Field(
        default_factory=list,
        description="Source citations supporting the answer.",
    )
    query_original: Optional[str] = Field(
        default=None,
        description="The original user query.",
    )
    query_normalized: Optional[str] = Field(
        default=None,
        description="The MSA-normalized query (if dialect was detected).",
    )
    dialect_detected: bool = Field(
        default=False,
        description="Whether Gulf dialect terms were detected in the query.",
    )


class UploadResponse(BaseModel):
    """Response model for a successful PDF upload."""

    document_id: str = Field(
        ...,
        description="Unique identifier for the uploaded document.",
    )
    filename: str = Field(
        ...,
        description="Sanitized filename of the uploaded document.",
    )
    page_count: int = Field(
        ...,
        description="Number of pages extracted from the PDF.",
    )
    chunk_count: int = Field(
        ...,
        description="Number of text chunks generated.",
    )
    status: str = Field(
        default="success",
        description="Upload processing status.",
    )
    message: str = Field(
        default="تم تحميل المستند ومعالجته بنجاح.",
        description="Human-readable status message.",
    )


class DocumentInfo(BaseModel):
    """Information about an uploaded document."""

    document_id: str = Field(
        ...,
        description="Unique identifier for the document.",
    )
    filename: str = Field(
        ...,
        description="Original filename of the document.",
    )
    page_count: int = Field(
        default=0,
        description="Number of pages in the document.",
    )
    chunk_count: int = Field(
        default=0,
        description="Number of text chunks stored.",
    )
    upload_time: Optional[str] = Field(
        default=None,
        description="ISO format timestamp of upload.",
    )
    file_size_mb: Optional[float] = Field(
        default=None,
        description="File size in megabytes.",
    )


class DocumentListResponse(BaseModel):
    """Response model for listing all uploaded documents."""

    documents: list[DocumentInfo] = Field(
        default_factory=list,
        description="List of uploaded documents.",
    )
    total_count: int = Field(
        default=0,
        description="Total number of documents.",
    )


class DocumentDeleteResponse(BaseModel):
    """Response model for document deletion."""

    status: str = Field(
        default="deleted",
        description="Status of the deletion request.",
    )
    document_id: str = Field(
        ...,
        description="ID of the deleted document.",
    )
    message: str = Field(
        default="تم حذف المستند بنجاح.",
        description="Localized status message.",
    )


class HealthResponse(BaseModel):
    """Response model for the health check endpoint."""

    status: str = Field(
        default="healthy",
        description="System health status.",
    )
    version: str = Field(
        default="1.0.0",
        description="API version.",
    )
    documents_count: int = Field(
        default=0,
        description="Number of uploaded documents.",
    )
    embedding_model: str = Field(
        default="",
        description="Name of the embedding model in use.",
    )
    llm_model: str = Field(
        default="",
        description="Name of the LLM model in use.",
    )
    uptime_seconds: Optional[float] = Field(
        default=None,
        description="Server uptime in seconds.",
    )
    embedding_model_loaded: bool = Field(
        default=False,
        description="Whether the embedding model has been loaded into memory.",
    )


class ErrorResponse(BaseModel):
    """Structured error response model."""

    error: str = Field(
        ...,
        description="Error type identifier.",
    )
    detail: str = Field(
        ...,
        description="Human-readable error description (Arabic + English).",
    )
    status_code: int = Field(
        ...,
        description="HTTP status code.",
    )


class TranslateRequest(BaseModel):
    """Request model for translating text to English."""
    text: str = Field(
        ...,
        min_length=1,
        max_length=6000,
        description="Arabic text to translate.",
    )


class TranslateResponse(BaseModel):
    """Response model for a translation request."""
    translation: str = Field(..., description="English translation.")


class TextAnalysisRequest(BaseModel):
    """Request model for analyzing Arabic text."""

    text: str = Field(
        ...,
        min_length=1,
        max_length=20000,
        description="Arabic text to analyze.",
    )


class TextAnalysisResponse(BaseModel):
    """Response model for Arabic text analysis."""

    total_chars: int = Field(..., description="Total character count.")
    arabic_chars: int = Field(..., description="Total Arabic character count.")
    arabic_ratio: float = Field(..., description="Ratio of Arabic characters.")
    word_count: int = Field(..., description="Total Arabic words.")
    unique_words: int = Field(..., description="Unique Arabic words.")
    lexical_diversity: float = Field(..., description="Type-token ratio.")
    tatweel_count: int = Field(..., description="Tatweel elongation count.")
    diacritic_count: int = Field(..., description="Tashkeel diacritic count.")
    diacritic_ratio: float = Field(..., description="Diacritics per Arabic char.")
    sentence_count: int = Field(..., description="Estimated sentence count.")
    is_primarily_arabic: bool = Field(..., description="True if text is primarily Arabic.")
    detected_dialects: list[dict[str, str]] = Field(
        default_factory=list,
        description="List of detected dialect terms with their MSA equivalents.",
    )


class BatchQuestionItem(BaseModel):
    """A single question item in a batch query."""

    id: Optional[str] = Field(
        default=None,
        description="Optional client-provided correlation ID.",
    )
    question: str = Field(
        ...,
        min_length=2,
        max_length=2000,
        description="The question to ask in Arabic.",
    )
    top_k: int = Field(
        default=5,
        ge=1,
        le=20,
        description="Number of chunks to retrieve.",
    )
    document_id: Optional[str] = Field(
        default=None,
        max_length=64,
        description="Optional document ID filter.",
    )


class BatchQuestionRequest(BaseModel):
    """Request model for batch question answering."""

    questions: list[BatchQuestionItem] = Field(
        ...,
        min_length=1,
        max_length=20,
        description="List of questions (maximum 20).",
    )


class BatchAnswerItem(BaseModel):
    """An individual answer result in a batch response."""

    id: Optional[str] = Field(default=None, description="Correlation ID matching the request.")
    response: AnswerResponse = Field(..., description="The answer response.")


class BatchAnswerResponse(BaseModel):
    """Response model for batch question answering."""

    total_processed: int = Field(..., description="Count of processed questions.")
    answers: list[BatchAnswerItem] = Field(..., description="List of generated answers.")
