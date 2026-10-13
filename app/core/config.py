"""
config.py — Centralized Application Settings

Uses Pydantic Settings for strongly-typed, environment-aware configuration.
Provides validated defaults, type conversions, and thread-safe cached access.
"""

from functools import lru_cache
from typing import Optional

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Central configuration for Arabic RAG System."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # API Settings
    api_host: str = Field(default="0.0.0.0", alias="API_HOST")
    api_port: int = Field(default=8000, alias="API_PORT")
    api_auth_key: str = Field(default="", alias="API_AUTH_KEY")
    api_cors_origins: str = Field(
        default="http://localhost:8501,http://127.0.0.1:8501",
        alias="API_CORS_ORIGINS",
    )
    api_cors_allow_credentials: bool = Field(
        default=False,
        alias="API_CORS_ALLOW_CREDENTIALS",
    )

    # Logging
    log_format: str = Field(default="json", alias="LOG_FORMAT")

    # Rate Limiting
    rate_limit_default: str = Field(default="60/minute", alias="RATE_LIMIT_DEFAULT")
    rate_limit_ask: str = Field(default="30/minute", alias="RATE_LIMIT_ASK")
    rate_limit_upload: str = Field(default="30/minute", alias="RATE_LIMIT_UPLOAD")

    # LLM Settings
    llm_model: str = Field(default="gpt-4o", alias="LLM_MODEL")
    llm_temperature: float = Field(default=0.1, alias="LLM_TEMPERATURE")
    openai_api_key: str = Field(default="", alias="OPENAI_API_KEY")
    openrouter_api_key: str = Field(default="", alias="OPENROUTER_API_KEY")

    # Embedding & Storage
    embedding_model: str = Field(
        default="paraphrase-multilingual-mpnet-base-v2",
        alias="EMBEDDING_MODEL",
    )
    chroma_persist_dir: str = Field(default="./chroma_db", alias="CHROMA_PERSIST_DIR")
    upload_dir: str = Field(default="./uploads", alias="UPLOAD_DIR")
    data_dir: str = Field(default="./data", alias="DATA_DIR")
    max_file_size_mb: int = Field(default=20, alias="MAX_FILE_SIZE_MB")
    chunk_size: int = Field(default=800, alias="CHUNK_SIZE")
    chunk_overlap: int = Field(default=150, alias="CHUNK_OVERLAP")
    top_k: int = Field(default=5, alias="TOP_K")
    min_similarity_threshold: float = Field(default=0.15, alias="MIN_SIMILARITY_THRESHOLD")

    # Caching
    enable_query_cache: bool = Field(default=True, alias="ENABLE_QUERY_CACHE")
    cache_ttl_seconds: int = Field(default=3600, alias="CACHE_TTL_SECONDS")

    def get_cors_origins_list(self) -> list[str]:
        """Parse comma-separated CORS origins into a list."""
        return [o.strip() for o in self.api_cors_origins.split(",") if o.strip()]


@lru_cache()
def get_settings() -> Settings:
    """Return cached application settings singleton."""
    return Settings()
