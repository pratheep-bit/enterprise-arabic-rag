"""
embeddings.py — Multilingual Embedding Model Wrapper

Wraps the sentence-transformers multilingual model for use with
LangChain and ChromaDB. Uses singleton pattern to avoid reloading.
"""

import logging
import os
import threading
from typing import Optional

from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings

load_dotenv()

logger = logging.getLogger(__name__)

# ============================================================
# Configuration
# ============================================================

DEFAULT_MODEL_NAME = os.getenv(
    "EMBEDDING_MODEL",
    "sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
)

# ============================================================
# Singleton Embedding Model (thread-safe)
# ============================================================

_embedding_instance: Optional[HuggingFaceEmbeddings] = None
_embedding_lock = threading.Lock()


def get_embedding_function(
    model_name: Optional[str] = None,
) -> HuggingFaceEmbeddings:
    """
    Get the embedding model instance (thread-safe singleton).

    Loads the model on first call and reuses it on subsequent calls.
    The model (~1GB) supports 50+ languages including Arabic.

    Args:
        model_name: HuggingFace model identifier. Defaults to env var.

    Returns:
        HuggingFaceEmbeddings instance ready for use with LangChain.
    """
    global _embedding_instance

    if _embedding_instance is not None:
        return _embedding_instance

    with _embedding_lock:
        # Double-checked locking
        if _embedding_instance is None:
            model = model_name or DEFAULT_MODEL_NAME
            logger.info(f"Loading embedding model: {model}")

            _embedding_instance = HuggingFaceEmbeddings(
                model_name=model,
                model_kwargs={
                    "device": "cpu",  # Use "cuda" if GPU available
                },
                encode_kwargs={
                    "normalize_embeddings": True,  # L2 normalization for cosine similarity
                    "batch_size": 32,
                },
            )

            logger.info(f"Embedding model loaded: {model}")

    return _embedding_instance


def embed_text(text: str) -> list[float]:
    """
    Generate embedding vector for a single text.

    Args:
        text: Text to embed.

    Returns:
        Embedding vector as list of floats.
    """
    model = get_embedding_function()
    return model.embed_query(text)


def embed_texts(texts: list[str]) -> list[list[float]]:
    """
    Generate embedding vectors for multiple texts (batched).

    Args:
        texts: List of texts to embed.

    Returns:
        List of embedding vectors.
    """
    model = get_embedding_function()
    return model.embed_documents(texts)


def get_model_name() -> str:
    """Return the current embedding model name."""
    return DEFAULT_MODEL_NAME
