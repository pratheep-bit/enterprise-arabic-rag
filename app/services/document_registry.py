"""
document_registry.py — Durable document metadata registry.

Stores upload metadata outside process memory so document filters survive
application restarts while ChromaDB keeps the chunk vectors.
"""

import json
import os
import tempfile
import threading
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv

load_dotenv()

DATA_DIR = Path(os.getenv("DATA_DIR", "./data"))
REGISTRY_PATH = Path(
    os.getenv("DOCUMENT_REGISTRY_PATH", str(DATA_DIR / "document_registry.json"))
)


class DocumentRegistry:
    """Thread-safe JSON-backed registry keyed by document_id."""

    def __init__(self, path: Optional[str | Path] = None):
        self.path = Path(path) if path is not None else REGISTRY_PATH
        self._lock = threading.RLock()
        self._documents: dict[str, dict] = {}
        self.reload()

    def reload(self) -> None:
        """Reload registry contents from disk."""
        with self._lock:
            if not self.path.exists():
                self._documents = {}
                return

            try:
                with self.path.open("r", encoding="utf-8") as f:
                    payload = json.load(f)
            except (json.JSONDecodeError, OSError):
                self._documents = {}
                return

            documents = payload.get("documents", {}) if isinstance(payload, dict) else {}
            self._documents = documents if isinstance(documents, dict) else {}

    def upsert(self, document: dict) -> None:
        """Insert or replace a document metadata record."""
        document_id = str(document.get("document_id", "")).strip()
        if not document_id:
            raise ValueError("document_id is required")

        with self._lock:
            self._documents[document_id] = document
            self._persist()

    def get(self, document_id: str) -> Optional[dict]:
        """Return a metadata record by document_id."""
        with self._lock:
            document = self._documents.get(document_id)
            return dict(document) if document else None

    def list(self) -> list[dict]:
        """Return all documents ordered newest first."""
        with self._lock:
            return sorted(
                (dict(doc) for doc in self._documents.values()),
                key=lambda item: item.get("upload_time", ""),
                reverse=True,
            )

    def count(self) -> int:
        """Return number of registered documents."""
        with self._lock:
            return len(self._documents)

    def _persist(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"documents": self._documents}

        fd, tmp_name = tempfile.mkstemp(
            prefix=f".{self.path.name}.",
            suffix=".tmp",
            dir=str(self.path.parent),
        )
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                json.dump(payload, f, ensure_ascii=False, indent=2, sort_keys=True)
                f.write("\n")
            os.replace(tmp_name, self.path)
        finally:
            if os.path.exists(tmp_name):
                os.unlink(tmp_name)
