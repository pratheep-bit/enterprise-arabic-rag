"""
routes.py — FastAPI API Routes

Defines all REST API endpoints for the Arabic RAG system:
- POST /upload   — Upload and process Arabic PDF
- POST /ask      — Ask a question against uploaded documents
- GET  /health   — System health check
- GET  /documents — List uploaded documents
- POST /translate — Translate Arabic to English
"""

import logging
import os
import time
import uuid
from datetime import datetime, timezone
from typing import Optional

from dotenv import load_dotenv
from fastapi import APIRouter, Depends, File, HTTPException, Request, UploadFile
from slowapi import Limiter
from slowapi.util import get_remote_address

from app.models.schemas import (
    AnswerResponse,
    DocumentInfo,
    DocumentListResponse,
    ErrorResponse,
    HealthResponse,
    QuestionRequest,
    UploadResponse,
    TranslateRequest,
    TranslateResponse,
)
from app.rag.chunker import ArabicChunker
from app.rag.embeddings import get_model_name, _embedding_instance
from app.rag.qa_chain import QAChain, QAChainError
from app.rag.vector_store import VectorStore
from app.services.document_registry import DocumentRegistry
from app.services.pdf_extractor import PDFExtractor, PDFExtractionError
from app.utils.file_utils import (
    FileValidationError,
    save_upload,
    sanitize_filename,
    validate_pdf_file,
)

load_dotenv()

logger = logging.getLogger(__name__)

# ============================================================
# Router
# ============================================================

router = APIRouter()

# ============================================================
# Rate limiter reference (attached to app.state in main.py)
# ============================================================

limiter = Limiter(key_func=get_remote_address)

# ============================================================
# Dependency Injection — Service Factories
# ============================================================

_document_registry = DocumentRegistry()

_pdf_extractor: Optional[PDFExtractor] = None
_chunker: Optional[ArabicChunker] = None
_vector_store: Optional[VectorStore] = None
_qa_chain: Optional[QAChain] = None


def get_pdf_extractor() -> PDFExtractor:
    """FastAPI dependency: lazily initialize PDFExtractor."""
    global _pdf_extractor
    if _pdf_extractor is None:
        _pdf_extractor = PDFExtractor()
    return _pdf_extractor


def get_chunker() -> ArabicChunker:
    """FastAPI dependency: lazily initialize ArabicChunker."""
    global _chunker
    if _chunker is None:
        _chunker = ArabicChunker()
    return _chunker


def get_vector_store() -> VectorStore:
    """FastAPI dependency: lazily initialize VectorStore."""
    global _vector_store
    if _vector_store is None:
        _vector_store = VectorStore()
    return _vector_store


def get_qa_chain() -> QAChain:
    """FastAPI dependency: lazily initialize QAChain."""
    global _qa_chain
    if _qa_chain is None:
        _qa_chain = QAChain()
    return _qa_chain


def get_document_registry() -> DocumentRegistry:
    """FastAPI dependency: return the singleton document registry."""
    return _document_registry


# ============================================================
# POST /upload — Upload and process an Arabic PDF
# ============================================================

@router.post(
    "/upload",
    response_model=UploadResponse,
    responses={
        400: {"model": ErrorResponse, "description": "Invalid file"},
        500: {"model": ErrorResponse, "description": "Processing error"},
    },
    summary="Upload an Arabic PDF document",
    description="Upload a PDF file, extract Arabic text, chunk it, generate embeddings, and store in the vector database.",
)
@limiter.limit(os.getenv("RATE_LIMIT_UPLOAD", "30/minute"))
async def upload_document(
    request: Request,
    file: UploadFile = File(..., description="Arabic PDF file to upload"),
    extractor: PDFExtractor = Depends(get_pdf_extractor),
    chunker: ArabicChunker = Depends(get_chunker),
    vector_store: VectorStore = Depends(get_vector_store),
    registry: DocumentRegistry = Depends(get_document_registry),
):
    """Upload and process an Arabic PDF document for question answering."""
    try:
        # Read file content
        file_content = await file.read()
        filename = sanitize_filename(file.filename or "document.pdf")
        document_id = uuid.uuid4().hex[:12]
        collection_name = f"doc_{document_id}"

        # Validate the PDF
        validate_pdf_file(file_content, filename, content_type=file.content_type)

        # Save to disk
        file_path = save_upload(file_content, filename)
        logger.info(f"Saved upload: {file_path}")

        # Extract text
        extraction_result = extractor.extract(str(file_path))

        pages = extraction_result["pages"]
        page_count = extraction_result["page_count"]

        # Chunk the extracted text
        chunks = chunker.chunk_pages(pages, source_filename=filename)
        for chunk in chunks:
            chunk.metadata["document_id"] = document_id

        if not chunks:
            raise HTTPException(
                status_code=400,
                detail={
                    "error": "NO_CHUNKS",
                    "detail": "لم يتم إنشاء أي أجزاء نصية من المستند.",
                    "status_code": 400,
                },
            )

        # Store in vector database
        stored_collection_name = vector_store.add_documents(
            documents=chunks,
            collection_name=collection_name,
        )

        # Register durable metadata after vector storage succeeds.
        registry.upsert({
            "document_id": document_id,
            "filename": filename,
            "collection_name": stored_collection_name,
            "page_count": page_count,
            "chunk_count": len(chunks),
            "file_path": str(file_path),
            "upload_time": datetime.now(timezone.utc).isoformat(),
            "file_size_mb": round(len(file_content) / (1024 * 1024), 2),
        })

        logger.info(
            f"Document processed: id={document_id}, "
            f"pages={page_count}, chunks={len(chunks)}"
        )

        return UploadResponse(
            document_id=document_id,
            filename=filename,
            page_count=page_count,
            chunk_count=len(chunks),
            status="success",
            message=f"تم تحميل ومعالجة المستند بنجاح. تم إنشاء {len(chunks)} جزء نصي من {page_count} صفحة.",
        )

    except FileValidationError as e:
        logger.warning(f"File validation failed: {e.message}")
        raise HTTPException(
            status_code=400,
            detail={
                "error": e.error_code,
                "detail": e.message,
                "status_code": 400,
            },
        )
    except PDFExtractionError as e:
        logger.error(f"PDF extraction failed: {e.message}")
        raise HTTPException(
            status_code=400,
            detail={
                "error": e.error_code,
                "detail": e.message,
                "status_code": 400,
            },
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Upload processing error: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail={
                "error": "PROCESSING_ERROR",
                "detail": "حدث خطأ أثناء معالجة المستند.",
                "status_code": 500,
            },
        )


# ============================================================
# POST /ask — Ask an Arabic question
# ============================================================

@router.post(
    "/ask",
    response_model=AnswerResponse,
    responses={
        400: {"model": ErrorResponse, "description": "Invalid question"},
        500: {"model": ErrorResponse, "description": "QA error"},
    },
    summary="Ask a question about uploaded documents",
    description="Ask a question in Arabic (MSA or Gulf/Egyptian/Levantine dialect) and receive an answer based on uploaded document context.",
)
@limiter.limit(os.getenv("RATE_LIMIT_ASK", "30/minute"))
async def ask_question(
    request: Request,
    body: QuestionRequest,
    qa_chain: QAChain = Depends(get_qa_chain),
    vector_store: VectorStore = Depends(get_vector_store),
    registry: DocumentRegistry = Depends(get_document_registry),
):
    """Ask an Arabic question against uploaded documents."""
    try:
        # Resolve collection name from document_id if provided
        collection_name = None
        if body.document_id:
            doc_info = registry.get(body.document_id)
            if doc_info:
                collection_name = doc_info.get("collection_name")
            elif vector_store.get_collection_info(body.document_id):
                collection_name = body.document_id
            else:
                raise HTTPException(
                    status_code=404,
                    detail={
                        "error": "DOCUMENT_NOT_FOUND",
                        "detail": f"المستند غير موجود: {body.document_id}",
                        "status_code": 404,
                    },
                )

        # Run the QA chain
        answer = qa_chain.ask(
            question=body.question,
            top_k=body.top_k,
            collection_name=collection_name,
        )

        logger.info(
            f"Question answered: confidence={answer.confidence:.2f}, "
            f"sources={len(answer.sources)}"
        )

        return answer

    except QAChainError as e:
        logger.error(f"QA chain error: {e.message}")
        raise HTTPException(
            status_code=500,
            detail={
                "error": e.error_code,
                "detail": e.message,
                "status_code": 500,
            },
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Question processing error: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail={
                "error": "QA_ERROR",
                "detail": "حدث خطأ أثناء الإجابة على السؤال.",
                "status_code": 500,
            },
        )


# ============================================================
# GET /health — Health check
# ============================================================

@router.get(
    "/health",
    response_model=HealthResponse,
    summary="System health check",
    description="Returns the current system status, model info, document count, uptime, and embedding model status.",
)
async def health_check(
    registry: DocumentRegistry = Depends(get_document_registry),
):
    """Check system health and component status."""
    try:
        from app.main import get_startup_time

        vector_store = VectorStore()
        collections = vector_store.list_collections()

        startup = get_startup_time()
        uptime = round(time.time() - startup, 1) if startup > 0 else None

        return HealthResponse(
            status="healthy",
            version="1.0.0",
            documents_count=registry.count() or len(collections),
            embedding_model=get_model_name(),
            llm_model=os.getenv("LLM_MODEL", "gpt-4o"),
            uptime_seconds=uptime,
            embedding_model_loaded=_embedding_instance is not None,
        )
    except Exception as e:
        logger.error(f"Health check error: {str(e)}")
        return HealthResponse(
            status="degraded",
            version="1.0.0",
            documents_count=0,
            embedding_model=get_model_name(),
            llm_model=os.getenv("LLM_MODEL", "gpt-4o"),
            uptime_seconds=None,
            embedding_model_loaded=_embedding_instance is not None,
        )


# ============================================================
# GET /documents — List uploaded documents
# ============================================================

@router.get(
    "/documents",
    response_model=DocumentListResponse,
    summary="List uploaded documents",
    description="Returns a list of all uploaded and processed documents with their metadata.",
)
async def list_documents(
    registry: DocumentRegistry = Depends(get_document_registry),
):
    """List all uploaded documents with metadata."""
    documents = []
    registered_collection_names = set()

    for info in registry.list():
        registered_collection_names.add(info.get("collection_name", ""))
        documents.append(
            DocumentInfo(
                document_id=info.get("document_id", ""),
                filename=info.get("filename", "unknown"),
                page_count=info.get("page_count", 0),
                chunk_count=info.get("chunk_count", 0),
                upload_time=info.get("upload_time"),
                file_size_mb=info.get("file_size_mb"),
            )
        )

    vector_store = VectorStore()
    for collection_name in vector_store.list_collections():
        if collection_name in registered_collection_names:
            continue
        collection_info = vector_store.get_collection_info(collection_name) or {}
        documents.append(
            DocumentInfo(
                document_id=collection_name,
                filename=collection_name,
                page_count=0,
                chunk_count=collection_info.get("count", 0),
                upload_time=None,
                file_size_mb=None,
            )
        )

    return DocumentListResponse(
        documents=documents,
        total_count=len(documents),
    )


# ============================================================
# POST /translate — Translate Arabic to English
# ============================================================

@router.post(
    "/translate",
    response_model=TranslateResponse,
    summary="Translate Arabic text to English",
)
async def translate_text(
    body: TranslateRequest,
    qa_chain: QAChain = Depends(get_qa_chain),
):
    """Translate Arabic text to English using the configured LLM."""
    try:
        from langchain_core.messages import HumanMessage, SystemMessage

        system_msg = "You are a professional translator. Translate the following Arabic text to English accurately. Output ONLY the English translation without any extra comments."
        messages = [
            SystemMessage(content=system_msg),
            HumanMessage(content=body.text),
        ]

        response = qa_chain.llm.invoke(messages)
        return TranslateResponse(translation=response.content.strip())

    except Exception as e:
        logger.error(f"Translation error: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail={
                "error": "TRANSLATION_ERROR",
                "detail": "Translation failed.",
                "status_code": 500,
            },
        )
