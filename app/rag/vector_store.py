"""
vector_store.py — ChromaDB Vector Store Integration

Manages persistent vector storage for Arabic document chunks.
Provides add, search, list, and delete operations.
"""

import logging
import os
import re
from typing import Optional

from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_core.documents import Document

from app.rag.embeddings import get_embedding_function
from app.utils.arabic_processor import normalize_for_embedding

load_dotenv()

logger = logging.getLogger(__name__)
logging.getLogger("chromadb.telemetry.product.posthog").setLevel(logging.CRITICAL)

# ============================================================
# Configuration
# ============================================================

CHROMA_PERSIST_DIR = os.getenv("CHROMA_PERSIST_DIR", "./chroma_db")
DEFAULT_COLLECTION = "arabic_documents"
MIN_SIMILARITY_THRESHOLD = float(os.getenv("MIN_SIMILARITY_THRESHOLD", "0.15"))
ARABIC_STOPWORDS = {
    "ما", "هي", "هو", "في", "من", "الى", "على", "عن", "عند", "او",
    "اي", "ان", "انها", "انه", "التي", "الذي", "هذا", "هذه", "ذلك",
    "يجب", "الخاص", "الخاصة", "اتباعها", "اتباع", "اجراءات", "الاجراءات",
}


def _chroma_settings():
    """Return Chroma settings suitable for local app usage."""
    from chromadb.config import Settings

    return Settings(anonymized_telemetry=False)


def _sanitize_collection_name(name: str) -> str:
    """
    Sanitize a string for use as a ChromaDB collection name.

    ChromaDB collection names must:
    - Be 3-63 characters long
    - Start and end with alphanumeric
    - Contain only alphanumeric, underscores, hyphens
    - Not contain consecutive periods

    Args:
        name: Raw name to sanitize.

    Returns:
        Valid ChromaDB collection name.
    """
    # Remove file extension
    name = os.path.splitext(name)[0]

    # Replace non-alphanumeric (except _ and -) with underscore
    name = re.sub(r"[^\w\-]", "_", name, flags=re.UNICODE)

    # Remove consecutive underscores
    name = re.sub(r"_+", "_", name)

    # Strip leading/trailing non-alphanumeric
    name = name.strip("_-")

    # Ensure minimum length
    if len(name) < 3:
        name = f"doc_{name}"

    # Truncate to max length
    if len(name) > 63:
        name = name[:63].rstrip("_-")

    # Ensure starts and ends with alphanumeric
    if not name[0].isalnum():
        name = "d" + name
    if not name[-1].isalnum():
        name = name + "0"

    return name


class VectorStore:
    """
    ChromaDB vector store manager for Arabic RAG.

    Features:
    - Persistent storage on disk
    - Document-level collections
    - Semantic similarity search with scores
    - Document metadata management
    - Collection lifecycle (create, list, delete)
    """

    def __init__(
        self,
        persist_directory: Optional[str] = None,
    ):
        """
        Initialize the vector store manager.

        Args:
            persist_directory: Path for ChromaDB persistent storage.
        """
        self.persist_directory = persist_directory or CHROMA_PERSIST_DIR
        self._embedding_function = None

        # Ensure directory exists
        os.makedirs(self.persist_directory, exist_ok=True)

        logger.info(f"VectorStore initialized at: {self.persist_directory}")

    @property
    def embedding_function(self):
        """Load the embedding model only when a vector operation needs it."""
        if self._embedding_function is None:
            self._embedding_function = get_embedding_function()
        return self._embedding_function

    def add_documents(
        self,
        documents: list[Document],
        collection_name: Optional[str] = None,
    ) -> str:
        """
        Add document chunks to the vector store.

        Args:
            documents: List of LangChain Document objects with metadata.
            collection_name: Name for the collection (auto-sanitized).

        Returns:
            The sanitized collection name used.
        """
        if not documents:
            raise ValueError("No documents to add.")

        # Determine collection name
        if collection_name:
            safe_name = _sanitize_collection_name(collection_name)
        else:
            # Use source filename from first document's metadata
            source = documents[0].metadata.get("source", DEFAULT_COLLECTION)
            safe_name = _sanitize_collection_name(source)

        logger.info(
            f"Adding {len(documents)} chunks to collection '{safe_name}'"
        )

        normalized_documents = [
            Document(
                page_content=normalize_for_embedding(document.page_content),
                metadata=document.metadata,
            )
            for document in documents
        ]

        # Create/open ChromaDB collection and add documents
        vectordb = Chroma(
            collection_name=safe_name,
            embedding_function=self.embedding_function,
            persist_directory=self.persist_directory,
            client_settings=_chroma_settings(),
            collection_metadata={"hnsw:space": "cosine"},
        )

        vectordb.add_documents(documents=normalized_documents)

        logger.info(
            f"Successfully stored {len(documents)} chunks in '{safe_name}'"
        )

        return safe_name

    def search(
        self,
        query: str,
        top_k: int = 5,
        collection_name: Optional[str] = None,
    ) -> list[dict]:
        """
        Perform semantic similarity search across stored documents.

        Args:
            query: Search query (Arabic or mixed).
            top_k: Number of results to return.
            collection_name: Specific collection to search (searches default if None).

        Returns:
            List of dicts with keys:
                - content: chunk text
                - metadata: chunk metadata (page, source, etc.)
                - similarity_score: cosine similarity (0.0 to 1.0)
        """
        if not query.strip():
            return []

        # If collection specified, search only that collection
        if collection_name:
            safe_name = _sanitize_collection_name(collection_name)
            return self._search_collection(query, top_k, safe_name)

        # Otherwise, search across all collections
        all_results = []
        collections = self.list_collections()

        if not collections:
            logger.warning("No collections found in vector store.")
            return []

        for col_name in collections:
            results = self._search_collection(query, top_k, col_name)
            all_results.extend(results)

        # Sort by similarity score (descending) and take top_k
        all_results.sort(key=lambda x: x["similarity_score"], reverse=True)
        return all_results[:top_k]

    def _search_collection(
        self,
        query: str,
        top_k: int,
        collection_name: str,
    ) -> list[dict]:
        """
        Search within a specific collection.

        Args:
            query: Search query.
            top_k: Number of results.
            collection_name: Collection to search.

        Returns:
            List of search result dicts.
        """
        try:
            vectordb = Chroma(
                collection_name=collection_name,
                embedding_function=self.embedding_function,
                persist_directory=self.persist_directory,
                client_settings=_chroma_settings(),
            )

            # similarity_search_with_relevance_scores returns (Document, score) tuples
            candidate_k = max(top_k, min(top_k * 4, 20))
            results = vectordb.similarity_search_with_relevance_scores(
                query=query,
                k=candidate_k,
            )

            formatted_results = []
            for doc, score in results:
                vector_score = round(max(0.0, min(1.0, score)), 4)
                hybrid_score = self._hybrid_score(
                    query=query,
                    content=doc.page_content,
                    vector_score=vector_score,
                )

                # Filter out chunks below the minimum similarity threshold
                if hybrid_score < MIN_SIMILARITY_THRESHOLD:
                    continue

                formatted_results.append(
                    {
                        "content": doc.page_content,
                        "metadata": doc.metadata,
                        "similarity_score": hybrid_score,
                        "vector_score": vector_score,
                    }
                )

            formatted_results.sort(key=lambda item: item["similarity_score"], reverse=True)
            return formatted_results[:top_k]

        except Exception as e:
            logger.error(
                f"Error searching collection '{collection_name}': {str(e)}"
            )
            return []

    @staticmethod
    def _terms(text: str) -> list[str]:
        normalized = normalize_for_embedding(text)
        raw_terms = re.findall(r"[\u0600-\u06FF\u0750-\u077F]{3,}", normalized)
        terms = []
        for term in raw_terms:
            term = re.sub(r"^(?:وال|بال|كال|فال|لل|ال|و|ف|ب|ك|ل)", "", term)
            if len(term) >= 3 and term not in ARABIC_STOPWORDS:
                terms.append(term)
        return terms

    @classmethod
    def _hybrid_score(cls, query: str, content: str, vector_score: float) -> float:
        query_terms = cls._terms(query)
        content_terms = set(cls._terms(content))
        if not query_terms or not content_terms:
            return round(vector_score, 4)

        matched_terms = {term for term in query_terms if term in content_terms}
        lexical_score = len(matched_terms) / min(len(set(query_terms)), 8)

        phrase_bonus = 0.0
        content_term_list = cls._terms(content)
        content_bigrams = set(zip(content_term_list, content_term_list[1:]))
        for first, second in zip(query_terms, query_terms[1:]):
            if (first, second) in content_bigrams:
                phrase_bonus = 0.5
                break

        return round(min(1.0, vector_score + (0.25 * lexical_score) + phrase_bonus), 4)

    def list_collections(self) -> list[str]:
        """
        List all collection names in the vector store.

        Returns:
            List of collection name strings.
        """
        try:
            import chromadb

            client = chromadb.PersistentClient(
                path=self.persist_directory,
                settings=_chroma_settings(),
            )
            collections = client.list_collections()
            
            if not collections:
                return []
                
            # Handle Chroma v0.6.0+ (returns list of strings) vs older versions (returns objects)
            if isinstance(collections[0], str):
                return collections
            return [col.name for col in collections]
        except Exception as e:
            logger.error(f"Error listing collections: {str(e)}")
            return []

    def get_collection_info(self, collection_name: str) -> Optional[dict]:
        """
        Get information about a specific collection.

        Args:
            collection_name: Name of the collection.

        Returns:
            Dict with collection info, or None if not found.
        """
        try:
            import chromadb

            safe_name = _sanitize_collection_name(collection_name)
            client = chromadb.PersistentClient(
                path=self.persist_directory,
                settings=_chroma_settings(),
            )
            collection = client.get_collection(name=safe_name)

            return {
                "name": safe_name,
                "count": collection.count(),
            }
        except Exception:
            return None

    def delete_collection(self, collection_name: str) -> bool:
        """
        Delete a collection from the vector store.

        Args:
            collection_name: Name of the collection to delete.

        Returns:
            True if deleted, False if not found.
        """
        try:
            import chromadb

            safe_name = _sanitize_collection_name(collection_name)
            client = chromadb.PersistentClient(
                path=self.persist_directory,
                settings=_chroma_settings(),
            )
            client.delete_collection(name=safe_name)
            logger.info(f"Deleted collection: {safe_name}")
            return True
        except Exception as e:
            logger.error(f"Error deleting collection '{collection_name}': {str(e)}")
            return False

    def get_total_document_count(self) -> int:
        """
        Get the total number of documents (chunks) across all collections.

        Returns:
            Total document count.
        """
        total = 0
        for col_name in self.list_collections():
            info = self.get_collection_info(col_name)
            if info:
                total += info.get("count", 0)
        return total
