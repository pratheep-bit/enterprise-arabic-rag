"""
qa_chain.py — End-to-End Question Answering Chain

Orchestrates retrieval, prompt construction, LLM invocation,
and response parsing for Arabic document QA.
"""

import json
import logging
import os
import re
from typing import Optional

from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage

from app.models.schemas import AnswerResponse, SourceCitation
from app.prompts.arabic_qa_prompt import SYSTEM_PROMPT, build_user_prompt
from app.rag.retriever import Retriever

load_dotenv()

logger = logging.getLogger(__name__)

# ============================================================
# Configuration
# ============================================================

LLM_MODEL = os.getenv("LLM_MODEL", "gpt-4o")
LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.1"))
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")

# Confidence score weights (must sum to 1.0)
CONFIDENCE_WEIGHT_LLM = float(os.getenv("CONFIDENCE_WEIGHT_LLM", "0.6"))
CONFIDENCE_WEIGHT_RETRIEVAL = float(os.getenv("CONFIDENCE_WEIGHT_RETRIEVAL", "0.4"))

# No-answer response (anti-hallucination)
NO_ANSWER_RESPONSE = "المعلومات غير موجودة في المستند المرفوع"


class QAChainError(Exception):
    """Raised when the QA chain encounters an error."""

    def __init__(self, message: str, error_code: str = "QA_ERROR"):
        self.message = message
        self.error_code = error_code
        super().__init__(self.message)


class QAChain:
    """
    End-to-end Arabic Question Answering chain.

    Pipeline:
    1. Retrieve relevant chunks via semantic search
    2. Build prompt with context and anti-hallucination rules
    3. Invoke LLM (OpenAI GPT)
    4. Parse structured JSON response
    5. Compute final confidence score
    6. Return structured AnswerResponse
    """

    def __init__(
        self,
        retriever: Optional[Retriever] = None,
        model_name: Optional[str] = None,
        temperature: Optional[float] = None,
        api_key: Optional[str] = None,
    ):
        """
        Initialize the QA chain.

        Args:
            retriever: Retriever instance (creates default if None).
            model_name: LLM model name (defaults to env var).
            temperature: LLM temperature (defaults to env var).
            api_key: OpenAI API key (defaults to env var).
        """
        self.retriever = retriever or Retriever()
        self.model_name = model_name or LLM_MODEL
        self.temperature = temperature if temperature is not None else LLM_TEMPERATURE

        # Initialize LLM
        key = api_key or OPENROUTER_API_KEY or OPENAI_API_KEY
        if not key:
            logger.warning(
                "API key not set. LLM calls will fail. "
                "Set OPENROUTER_API_KEY or OPENAI_API_KEY in your .env file."
            )

        # Use OpenRouter base URL if using OpenRouter key
        is_openrouter = bool(OPENROUTER_API_KEY) or (api_key and "sk-or" in api_key)
        base_url = "https://openrouter.ai/api/v1" if is_openrouter else None
        
        headers = {
            "HTTP-Referer": "http://localhost:8501",
            "X-OpenRouter-Title": "Arabic RAG System"
        } if is_openrouter else None

        self.llm = ChatOpenAI(
            model=self.model_name,
            temperature=self.temperature,
            api_key=key or "not-set",
            base_url=base_url,
            default_headers=headers,
            max_tokens=2000,
        )

        logger.info(
            f"QAChain initialized: model={self.model_name}, temp={self.temperature}"
        )

    def ask(
        self,
        question: str,
        top_k: int = 5,
        collection_name: Optional[str] = None,
    ) -> AnswerResponse:
        """
        Answer an Arabic question using retrieved document context.

        Args:
            question: The user's question in Arabic.
            top_k: Number of chunks to retrieve.
            collection_name: Optional document collection to restrict search.

        Returns:
            AnswerResponse with answer, confidence, and source citations.

        Raises:
            QAChainError: If the pipeline fails.
        """
        if not question or not question.strip():
            raise QAChainError(
                "السؤال فارغ. يرجى إدخال سؤال.",
                error_code="EMPTY_QUESTION",
            )

        try:
            # Step 1: Retrieve relevant chunks
            retrieval_result = self.retriever.retrieve(
                query=question,
                top_k=top_k,
                collection_name=collection_name,
            )

            chunks = retrieval_result["chunks"]
            dialect_detected = retrieval_result["dialect_detected"]
            query_normalized = retrieval_result["query_normalized"]

            # Step 2: Handle empty retrieval
            if not chunks:
                logger.info("No chunks retrieved — returning no-answer response.")
                return AnswerResponse(
                    answer=NO_ANSWER_RESPONSE,
                    confidence=0.0,
                    sources=[],
                    query_original=question,
                    query_normalized=query_normalized,
                    dialect_detected=dialect_detected,
                )

            # Step 3: Build the prompt
            user_prompt = build_user_prompt(question, chunks)

            # Step 4: Invoke LLM
            messages = [
                SystemMessage(content=SYSTEM_PROMPT),
                HumanMessage(content=user_prompt),
            ]

            logger.info(f"Invoking LLM ({self.model_name}) with {len(chunks)} context chunks")
            llm_response = self._invoke_llm_with_retry(messages)
            raw_response = llm_response.content

            # Step 5: Parse the LLM response
            parsed = self._parse_llm_response(raw_response)

            # Step 6: Compute final confidence
            # Combine LLM self-assessment with retrieval similarity scores
            llm_confidence = parsed.get("confidence", 0.5)
            avg_retrieval_score = (
                sum(c["similarity_score"] for c in chunks) / len(chunks)
                if chunks
                else 0.0
            )
            # Weighted average: configurable via env vars
            final_confidence = round(
                CONFIDENCE_WEIGHT_LLM * llm_confidence
                + CONFIDENCE_WEIGHT_RETRIEVAL * avg_retrieval_score,
                4,
            )

            # Step 7: Build source citations
            sources = self._build_sources(parsed, chunks)

            # Step 8: Build response
            return AnswerResponse(
                answer=parsed.get("answer", NO_ANSWER_RESPONSE),
                confidence=final_confidence,
                sources=sources,
                query_original=question,
                query_normalized=query_normalized,
                dialect_detected=dialect_detected,
            )

        except QAChainError:
            raise
        except Exception as e:
            logger.error(f"QA chain error: {str(e)}", exc_info=True)
            raise QAChainError(
                "حدث خطأ أثناء معالجة السؤال.",
                error_code="CHAIN_ERROR",
            )

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        retry=retry_if_exception_type(Exception),
        before_sleep=lambda retry_state: logger.warning(
            f"LLM call failed (attempt {retry_state.attempt_number}), retrying: "
            f"{retry_state.outcome.exception()}"
        ),
        reraise=True,
    )
    def _invoke_llm_with_retry(self, messages: list):
        """Invoke the LLM with automatic retry on transient failures."""
        return self.llm.invoke(messages)

    def _parse_llm_response(self, raw_response: str) -> dict:
        """
        Parse the LLM's JSON response, with fallback handling.

        Args:
            raw_response: Raw text response from the LLM.

        Returns:
            Parsed dict with answer, confidence, and sources.
        """
        # Try to extract JSON from the response
        # The LLM may wrap JSON in markdown code blocks
        json_match = re.search(
            r"```(?:json)?\s*(\{.*?\})\s*```",
            raw_response,
            re.DOTALL,
        )

        json_str = json_match.group(1) if json_match else raw_response

        # Try direct JSON parsing
        try:
            # Find the outermost JSON object
            start = json_str.find("{")
            end = json_str.rfind("}") + 1
            if start >= 0 and end > start:
                parsed = json.loads(json_str[start:end])
                return parsed
        except json.JSONDecodeError:
            pass

        # Fallback: treat entire response as the answer
        logger.warning(
            "Failed to parse LLM response as JSON. Using raw response as answer."
        )
        return {
            "answer": raw_response.strip(),
            "confidence": 0.5,
            "sources": [],
        }

    def _build_sources(
        self,
        parsed_response: dict,
        retrieved_chunks: list[dict],
    ) -> list[SourceCitation]:
        """
        Build source citations from parsed LLM response and retrieved chunks.

        Prefers LLM-provided sources but falls back to retrieval metadata.

        Args:
            parsed_response: Parsed LLM response dict.
            retrieved_chunks: Original retrieved chunks.

        Returns:
            List of SourceCitation objects.
        """
        sources = []

        # Try to use LLM-provided sources
        llm_sources = parsed_response.get("sources", [])
        if llm_sources and isinstance(llm_sources, list):
            for src in llm_sources:
                if isinstance(src, dict):
                    sources.append(
                        SourceCitation(
                            page=src.get("page", 0),
                            document=src.get("document", "unknown"),
                            excerpt=src.get("excerpt", ""),
                            similarity_score=None,
                        )
                    )

        # If no LLM sources, build from retrieval metadata
        if not sources and retrieved_chunks:
            for chunk in retrieved_chunks[:3]:  # Top 3 chunks as sources
                metadata = chunk.get("metadata", {})
                sources.append(
                    SourceCitation(
                        page=metadata.get("page", 0),
                        document=metadata.get("source", "unknown"),
                        excerpt=chunk["content"][:200] + "..."
                        if len(chunk["content"]) > 200
                        else chunk["content"],
                        similarity_score=chunk.get("similarity_score"),
                    )
                )

        return sources

    def get_model_info(self) -> dict:
        """Return current LLM configuration info."""
        return {
            "model": self.model_name,
            "temperature": self.temperature,
        }
