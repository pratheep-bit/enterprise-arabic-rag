"""
retriever.py — Semantic Retrieval Pipeline

Handles query normalization, dialect expansion, embedding,
and semantic search against the ChromaDB vector store.
"""

import os
import logging
from typing import Optional

from app.rag.vector_store import VectorStore
from app.utils.arabic_processor import ArabicProcessor, normalize_for_embedding
from app.utils.dialect_mapper import DialectMapper

logger = logging.getLogger(__name__)

DEFAULT_MIN_THRESHOLD = float(os.getenv("MIN_SIMILARITY_THRESHOLD", "0.15"))


class Retriever:
    """
    Arabic semantic retrieval pipeline.

    Pipeline:
    1. Normalize Arabic text in the query
    2. Detect and expand regional dialect terms
    3. Embed query variants
    4. Search ChromaDB for top-k chunks
    5. Merge, deduplicate, filter by similarity threshold, and rank results
    """

    def __init__(
        self,
        vector_store: Optional[VectorStore] = None,
        dialect_mapper: Optional[DialectMapper] = None,
        arabic_processor: Optional[ArabicProcessor] = None,
        min_similarity_threshold: float = DEFAULT_MIN_THRESHOLD,
    ):
        """
        Initialize the retriever.

        Args:
            vector_store: VectorStore instance (creates default if None).
            dialect_mapper: DialectMapper instance (creates default if None).
            arabic_processor: ArabicProcessor instance (creates default if None).
            min_similarity_threshold: Minimum cosine/hybrid score for retrieved chunks.
        """
        self.vector_store = vector_store or VectorStore()
        self.dialect_mapper = dialect_mapper or DialectMapper()
        self.arabic_processor = arabic_processor or ArabicProcessor()
        self.min_similarity_threshold = min_similarity_threshold

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        collection_name: Optional[str] = None,
        min_threshold: Optional[float] = None,
    ) -> dict:
        """
        Retrieve relevant document chunks for an Arabic query.

        Args:
            query: The user's question in Arabic (MSA or Gulf dialect).
            top_k: Number of top results to return.
            collection_name: Optional collection to restrict search to.

        Returns:
            Dict with keys:
                - chunks: List of retrieved chunk dicts (content, metadata, score)
                - query_original: The original query
                - query_normalized: MSA-normalized query (if dialect detected)
                - dialect_detected: bool
                - dialect_terms: List of detected dialect terms
        """
        if not query or not query.strip():
            return {
                "chunks": [],
                "query_original": query,
                "query_normalized": query,
                "dialect_detected": False,
                "dialect_terms": [],
            }

        # Step 1: Normalize the Arabic text
        normalized_query = self.arabic_processor.process(query)

        # Step 2: Detect and expand Gulf dialect
        dialect_detected = self.dialect_mapper.has_dialect_terms(normalized_query)
        dialect_terms = self.dialect_mapper.get_dialect_terms_found(normalized_query)
        query_variants = self.dialect_mapper.expand_query(normalized_query)

        logger.info(
            f"Query: '{normalized_query}' | "
            f"Dialect detected: {dialect_detected} | "
            f"Variants: {len(query_variants)}"
        )

        # Step 3: Search with each query variant
        all_results = []
        for variant in query_variants:
            # Normalize for embedding similarity
            embedding_query = normalize_for_embedding(variant)

            results = self.vector_store.search(
                query=embedding_query,
                top_k=top_k,
                collection_name=collection_name,
            )
            all_results.extend(results)

        # Step 4: Deduplicate by content hash and merge scores
        deduplicated = self._deduplicate_results(all_results)

        # Step 5: Filter by minimum similarity threshold
        effective_threshold = (
            min_threshold if min_threshold is not None else self.min_similarity_threshold
        )
        filtered = [
            chunk for chunk in deduplicated
            if chunk.get("similarity_score", 0.0) >= effective_threshold
        ]

        # Step 6: Sort by similarity score (descending) and take top_k
        filtered.sort(key=lambda x: x["similarity_score"], reverse=True)
        top_results = filtered[:top_k]

        logger.info(
            f"Retrieved {len(top_results)} chunks "
            f"(filtered from {len(all_results)} total across {len(query_variants)} variants, threshold={effective_threshold})"
        )

        # Determine the normalized query string for the response
        msa_query = None
        if dialect_detected and len(query_variants) > 1:
            msa_query = query_variants[1]  # Second variant is MSA-normalized

        return {
            "chunks": top_results,
            "query_original": query,
            "query_normalized": msa_query or normalized_query,
            "dialect_detected": dialect_detected,
            "dialect_terms": dialect_terms,
        }

    @staticmethod
    def _deduplicate_results(results: list[dict]) -> list[dict]:
        """
        Deduplicate results by content, keeping the highest similarity score.

        Args:
            results: List of search result dicts.

        Returns:
            Deduplicated list.
        """
        seen = {}

        for result in results:
            content_key = result["content"][:200]  # Use first 200 chars as key

            if content_key in seen:
                # Keep the higher score
                if result["similarity_score"] > seen[content_key]["similarity_score"]:
                    seen[content_key] = result
            else:
                seen[content_key] = result

        return list(seen.values())
