"""
main.py — FastAPI Application Entry Point

Configures and starts the Arabic Document Q&A RAG API server.
Includes: structured logging, request ID middleware, API key auth,
rate limiting, and global exception handlers.
"""

import logging
import os
import time
import uuid
from contextlib import asynccontextmanager

from dotenv import load_dotenv
from fastapi import FastAPI, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from slowapi import Limiter
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address
from pythonjsonlogger import json as json_log

from app.api.routes import router

# Load environment variables
load_dotenv()

# ============================================================
# Structured JSON Logging
# ============================================================

_log_format = os.getenv("LOG_FORMAT", "json")  # "json" or "text"


def _configure_logging():
    """Configure structured JSON logging for production observability."""
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)

    # Remove any existing handlers
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)

    handler = logging.StreamHandler()

    if _log_format == "json":
        formatter = json_log.JsonFormatter(
            fmt="%(asctime)s %(name)s %(levelname)s %(message)s",
            rename_fields={"asctime": "timestamp", "levelname": "level"},
            datefmt="%Y-%m-%dT%H:%M:%S",
        )
    else:
        formatter = logging.Formatter(
            "%(asctime)s | %(name)-30s | %(levelname)-8s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )

    handler.setFormatter(formatter)
    root_logger.addHandler(handler)

    # Silence noisy third-party loggers
    logging.getLogger("chromadb.telemetry.product.posthog").setLevel(logging.CRITICAL)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)


_configure_logging()
logger = logging.getLogger(__name__)


def _csv_env(name: str, default: str) -> list[str]:
    """Read a comma-separated environment variable into a clean list."""
    raw = os.getenv(name, default)
    return [item.strip() for item in raw.split(",") if item.strip()]


# ============================================================
# Rate Limiter
# ============================================================

_rate_default = os.getenv("RATE_LIMIT_DEFAULT", "60/minute")
_rate_ask = os.getenv("RATE_LIMIT_ASK", "30/minute")
_rate_upload = os.getenv("RATE_LIMIT_UPLOAD", "30/minute")

limiter = Limiter(key_func=get_remote_address, default_limits=[_rate_default])


# ============================================================
# Startup Tracking (for /health uptime)
# ============================================================

_startup_time: float = 0.0


# ============================================================
# Application Lifespan (Startup / Shutdown)
# ============================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Manages application startup and shutdown events.

    On startup:
    - Validates environment configuration
    - Records startup timestamp for uptime tracking
    """
    global _startup_time
    _startup_time = time.time()

    logger.info("=" * 60)
    logger.info("Starting Arabic Document Q&A RAG System")
    logger.info("=" * 60)

    # Validate critical environment variables
    openai_key = os.getenv("OPENAI_API_KEY", "")
    openrouter_key = os.getenv("OPENROUTER_API_KEY", "")
    if (
        (not openai_key or openai_key == "your-openai-api-key-here")
        and not openrouter_key
    ):
        logger.warning(
            "LLM API key is not set. "
            "The /ask endpoint will fail. "
            "Set OPENAI_API_KEY or OPENROUTER_API_KEY in your .env file."
        )
    else:
        logger.info("LLM API key is configured")

    # Log configuration
    logger.info(f"LLM Model: {os.getenv('LLM_MODEL', 'gpt-4o')}")
    logger.info(f"Embedding Model: {os.getenv('EMBEDDING_MODEL', 'paraphrase-multilingual-mpnet-base-v2')}")
    logger.info(f"ChromaDB Dir: {os.getenv('CHROMA_PERSIST_DIR', './chroma_db')}")
    logger.info(f"Upload Dir: {os.getenv('UPLOAD_DIR', './uploads')}")
    logger.info(f"Max File Size: {os.getenv('MAX_FILE_SIZE_MB', '20')} MB")
    logger.info(f"Chunk Size: {os.getenv('CHUNK_SIZE', '800')}")
    logger.info(f"Top-K: {os.getenv('TOP_K', '5')}")
    logger.info(f"Rate Limits: default={_rate_default}, ask={_rate_ask}, upload={_rate_upload}")

    auth_key = os.getenv("API_AUTH_KEY", "")
    if auth_key:
        logger.info("API authentication is ENABLED")
    else:
        logger.warning("API authentication is DISABLED — set API_AUTH_KEY to enable")

    # Optionally pre-load the embedding model on startup
    # Uncomment to reduce first-request latency (~30s model download on first run)
    # from app.rag.embeddings import get_embedding_function
    # logger.info("Pre-loading embedding model...")
    # get_embedding_function()
    # logger.info("Embedding model loaded.")

    logger.info("=" * 60)
    logger.info("System ready. API docs at: http://localhost:8000/docs")
    logger.info("=" * 60)

    yield  # Application runs here

    # Shutdown
    logger.info("Shutting down Arabic RAG System...")


# ============================================================
# FastAPI Application
# ============================================================

app = FastAPI(
    title="Arabic Document Q&A RAG System",
    description=(
        "نظام الإجابة على الأسئلة من المستندات العربية باستخدام تقنية RAG\n\n"
        "An Arabic Document Question-Answering system using Retrieval-Augmented Generation (RAG). "
        "Supports Modern Standard Arabic (MSA), Gulf, Egyptian, and Levantine dialects."
    ),
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

# Attach rate limiter
app.state.limiter = limiter


# ============================================================
# Middleware: Request ID
# ============================================================

@app.middleware("http")
async def request_id_middleware(request: Request, call_next):
    """Generate a unique request ID for every request and inject into logs/response."""
    request_id = request.headers.get("X-Request-ID") or uuid.uuid4().hex[:16]
    request.state.request_id = request_id

    # Log request start
    logger.info(
        "Request started",
        extra={
            "request_id": request_id,
            "method": request.method,
            "path": request.url.path,
            "client": request.client.host if request.client else "unknown",
        },
    )

    start = time.time()
    response = await call_next(request)
    duration_ms = round((time.time() - start) * 1000, 1)

    # Attach request ID to response
    response.headers["X-Request-ID"] = request_id

    logger.info(
        "Request completed",
        extra={
            "request_id": request_id,
            "method": request.method,
            "path": request.url.path,
            "status_code": response.status_code,
            "duration_ms": duration_ms,
        },
    )

    return response


# ============================================================
# Middleware: API Key Authentication
# ============================================================

_API_AUTH_KEY = os.getenv("API_AUTH_KEY", "")

# Paths that do not require authentication
_PUBLIC_PATHS = {"/health", "/docs", "/redoc", "/openapi.json"}


@app.middleware("http")
async def api_key_auth_middleware(request: Request, call_next):
    """Enforce API key authentication on all endpoints except public ones."""
    if not _API_AUTH_KEY:
        # Auth disabled — no key configured
        return await call_next(request)

    path = request.url.path.rstrip("/")
    if path in _PUBLIC_PATHS:
        return await call_next(request)

    provided_key = request.headers.get("X-API-Key", "")
    if provided_key != _API_AUTH_KEY:
        return JSONResponse(
            status_code=status.HTTP_401_UNAUTHORIZED,
            content={
                "error": "UNAUTHORIZED",
                "detail": "Invalid or missing API key. Provide a valid X-API-Key header.",
                "status_code": 401,
            },
        )

    return await call_next(request)


# ============================================================
# CORS Middleware
# ============================================================

app.add_middleware(
    CORSMiddleware,
    allow_origins=_csv_env(
        "API_CORS_ORIGINS",
        "http://localhost:8501,http://127.0.0.1:8501",
    ),
    allow_credentials=os.getenv("API_CORS_ALLOW_CREDENTIALS", "false").lower() == "true",
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type", "Authorization", "X-API-Key", "X-Request-ID"],
)


# ============================================================
# Global Exception Handlers
# ============================================================

@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    """Return structured 422 responses for Pydantic validation errors."""
    errors = exc.errors()
    detail_parts = []
    for err in errors:
        loc = " → ".join(str(l) for l in err.get("loc", []))
        msg = err.get("msg", "Invalid value")
        detail_parts.append(f"{loc}: {msg}")

    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content={
            "error": "VALIDATION_ERROR",
            "detail": "; ".join(detail_parts),
            "status_code": 422,
            "fields": errors,
        },
    )


@app.exception_handler(RateLimitExceeded)
async def rate_limit_handler(request: Request, exc: RateLimitExceeded):
    """Return structured 429 responses when rate limit is exceeded."""
    return JSONResponse(
        status_code=status.HTTP_429_TOO_MANY_REQUESTS,
        content={
            "error": "RATE_LIMIT_EXCEEDED",
            "detail": "Too many requests. Please slow down.",
            "status_code": 429,
        },
    )


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Catch-all for unhandled exceptions — returns structured JSON, never raw stack traces."""
    request_id = getattr(request.state, "request_id", "unknown")
    logger.error(
        f"Unhandled exception: {exc}",
        exc_info=True,
        extra={"request_id": request_id},
    )
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "error": "INTERNAL_ERROR",
            "detail": "An unexpected error occurred. Please try again.",
            "status_code": 500,
        },
    )


# ============================================================
# Include API Routes
# ============================================================

app.include_router(router, prefix="", tags=["Arabic RAG"])


# ============================================================
# Expose startup_time for /health uptime calculation
# ============================================================

def get_startup_time() -> float:
    """Return the server startup timestamp."""
    return _startup_time


# ============================================================
# Run with Uvicorn (for direct execution)
# ============================================================

if __name__ == "__main__":
    import uvicorn

    host = os.getenv("API_HOST", "0.0.0.0")
    port = int(os.getenv("API_PORT", "8000"))

    uvicorn.run(
        "app.main:app",
        host=host,
        port=port,
        reload=True,
        log_level="info",
    )
