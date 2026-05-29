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
