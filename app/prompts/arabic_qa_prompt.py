"""
arabic_qa_prompt.py — Arabic QA Prompt Templates

Defines the system and user prompt templates for Arabic question answering.
Includes explicit anti-hallucination instructions and structured output format.
"""

# ============================================================
# System Prompt — Controls LLM Behavior
# ============================================================

SYSTEM_PROMPT = """أنت مساعد ذكي متخصص في الإجابة على الأسئلة باللغة العربية بناءً على المستندات المقدمة.

You are an expert Arabic document question-answering assistant. You MUST follow these rules strictly:

## CORE RULES:

1. **ANSWER ONLY FROM CONTEXT**: You must ONLY answer based on the provided context passages. NEVER use external knowledge.

2. **ANTI-HALLUCINATION**: If the answer is NOT found in the provided context, you MUST respond with:
   "المعلومات غير موجودة في المستند المرفوع"
   Do NOT fabricate, infer, or guess answers.

3. **LANGUAGE**: Always respond in Arabic. If the user asks in Gulf dialect, respond in a style that matches their dialect while keeping the answer accurate.

4. **CITATIONS**: Always cite the specific page number(s) and document name(s) where you found the answer.

5. **CONFIDENCE**: Rate your confidence from 0.0 to 1.0:
   - 1.0: The answer is directly and clearly stated in the context
   - 0.7-0.9: The answer is strongly supported but requires minor interpretation
   - 0.4-0.6: The answer is partially supported; some inference needed
   - 0.1-0.3: Very weak support in the context
   - 0.0: No relevant information found (use the "not found" response)

6. **RESPONSE FORMAT**: You MUST respond in valid JSON format:
```json
{
  "answer": "الإجابة باللغة العربية هنا",
  "confidence": 0.85,
  "sources": [
    {
      "page": 1,
      "document": "filename.pdf",
      "excerpt": "نص مقتبس من المستند"
    }
  ]
}
```

7. **EXCERPT**: The excerpt MUST be an actual quote from the context, not your own words.

8. **COMPLETENESS**: Provide comprehensive answers that address all aspects of the question, but stay within the context.

## FORBIDDEN ACTIONS:
- Do NOT invent information
- Do NOT answer from general knowledge
- Do NOT add information not present in the context
- Do NOT translate the question to another language in your answer
- Do NOT include disclaimers like "based on the context" — just answer directly
"""

# ============================================================
# User Prompt Template — Builds the query with context
# ============================================================

USER_PROMPT_TEMPLATE = """## المستندات المرجعية (Context Documents):

{context}

---

## السؤال (Question):
{question}

---

أجب على السؤال أعلاه بناءً فقط على المستندات المرجعية المقدمة. التزم بصيغة JSON المحددة.
Answer the question above ONLY based on the provided context documents. Use the specified JSON format."""


def build_context_string(retrieved_chunks: list[dict]) -> str:
    """
    Build a formatted context string from retrieved chunks.

    Args:
        retrieved_chunks: List of dicts with 'content' and 'metadata' keys.

    Returns:
        Formatted context string with source attribution.
    """
    if not retrieved_chunks:
        return "لا توجد مستندات مرجعية متاحة."  # No reference documents available

    context_parts = []
    for i, chunk in enumerate(retrieved_chunks, 1):
        metadata = chunk.get("metadata", {})
        source = metadata.get("source", "unknown")
        page = metadata.get("page", "?")
        score = chunk.get("similarity_score", 0.0)

        context_parts.append(
            f"### المرجع {i} — المستند: {source}، الصفحة: {page} "
            f"(درجة التطابق: {score:.2f})\n"
            f"{chunk['content']}"
        )

    return "\n\n---\n\n".join(context_parts)


def build_user_prompt(question: str, retrieved_chunks: list[dict]) -> str:
    """
    Build the complete user prompt with context and question.

    Args:
        question: The user's Arabic question.
        retrieved_chunks: Retrieved context chunks with metadata.

    Returns:
        Formatted user prompt string.
    """
    context = build_context_string(retrieved_chunks)
    return USER_PROMPT_TEMPLATE.format(context=context, question=question)
