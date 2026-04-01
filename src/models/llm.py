"""
Large Language Model (LLM) wrapper using Groq (llama-3.3-70b-versatile).

Responsible for generating grounded answers from retrieved EV document
chunks. Uses a strict RAG prompt template to prevent hallucination and
ensure answers are sourced from the retrieved context.
"""

import logging
from typing import List, Dict, Any

from groq import Groq

logger = logging.getLogger(__name__)

_RAG_SYSTEM_PROMPT = """You are a senior automotive engineer and technical expert at TATA Motors,
specialising in Electric Vehicles (EV) — battery systems, powertrain, BMS, charging infrastructure,
homologation, and safety compliance.

Your role: answer questions using ONLY the context retrieved from TATA Motors EV technical documents.

Rules:
1. Base your answer exclusively on the provided context. Do not use prior knowledge.
2. If the context does not contain enough information to answer, say so clearly.
3. Quote or reference specific values, table data, or diagram descriptions when available.
4. Use precise engineering language appropriate for a TATA Motors technical audience.
5. Do NOT fabricate specifications, part numbers, or regulatory citations.
"""

_RAG_PROMPT_TEMPLATE = """Retrieved context from EV technical documents:
{context}

---

Question: {question}

Instructions:
- Answer based solely on the above context.
- If the answer involves numerical data from a table, cite the exact values.
- If an image description is referenced, explain what it shows.
- End your answer with a brief note on which source(s) supported it.

Answer:"""


class LanguageModel:
    """Wraps Groq LLM for retrieval-augmented text generation."""

    def __init__(self, api_key: str, model_name: str = "llama-3.3-70b-versatile") -> None:
        """
        Initialise the LLM.

        Args:
            api_key:    Groq API key.
            model_name: Groq model identifier.
        """
        self._client = Groq(api_key=api_key)
        self._model_name = model_name
        logger.info("LanguageModel initialised with model '%s'.", model_name)

    def generate_answer(
        self,
        question: str,
        retrieved_chunks: List[Dict[str, Any]],
    ) -> str:
        """
        Generate a grounded answer from retrieved document chunks.

        Args:
            question:         The user's natural language question.
            retrieved_chunks: List of chunk dicts with keys:
                              'content', 'type', 'source', 'page'.

        Returns:
            The generated answer as a string.
        """
        if not retrieved_chunks:
            return (
                "No relevant context was found in the indexed documents for your question. "
                "Please ingest relevant EV technical documents first."
            )

        context_parts: List[str] = []
        for i, chunk in enumerate(retrieved_chunks, start=1):
            chunk_type = chunk.get("type", "unknown").upper()
            source = chunk.get("source", "unknown")
            page = chunk.get("page", "?")
            content = chunk.get("content", "")

            context_parts.append(
                f"[Chunk {i} | Type: {chunk_type} | Source: {source} | Page: {page}]\n"
                f"{content}"
            )

        context_str = "\n\n---\n\n".join(context_parts)
        prompt = _RAG_PROMPT_TEMPLATE.format(
            context=context_str,
            question=question,
        )

        try:
            response = self._client.chat.completions.create(
                model=self._model_name,
                messages=[
                    {"role": "system", "content": _RAG_SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
            )
            answer = response.choices[0].message.content.strip()
            logger.debug("LLM generated %d-char answer.", len(answer))
            return answer
        except Exception as exc:  # noqa: BLE001
            logger.error("LLM generation failed: %s", exc)
            raise RuntimeError(f"Answer generation failed: {exc}") from exc

    @property
    def model_name(self) -> str:
        """Return the underlying model identifier."""
        return self._model_name
