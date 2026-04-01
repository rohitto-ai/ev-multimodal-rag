"""
RAG Retriever — orchestrates embedding, retrieval, and answer generation.

The retrieval pipeline:
  1. Embed the user's question using the same embedding model used at ingest time.
  2. Query ChromaDB for the top-K most semantically similar chunks
     (across all chunk types: text, table, image).
  3. Format retrieved chunks into a structured context string.
  4. Pass context + question to the Gemini LLM to generate a grounded answer.
  5. Return the answer with source references for citation.
"""

import logging
from typing import Any, Dict, List

from src.ingestion.embedder import Embedder
from src.models.llm import LanguageModel
from src.retrieval.vector_store import VectorStore

logger = logging.getLogger(__name__)


class Retriever:
    """End-to-end RAG pipeline: query → retrieve → generate → cite."""

    def __init__(
        self,
        vector_store: VectorStore,
        embedder: Embedder,
        llm: LanguageModel,
        top_k: int = 5,
    ) -> None:
        """
        Args:
            vector_store: Initialised ChromaDB wrapper.
            embedder:     Sentence-transformer embedding model.
            llm:          Gemini language model for answer generation.
            top_k:        Number of chunks to retrieve per query.
        """
        self._vector_store = vector_store
        self._embedder = embedder
        self._llm = llm
        self._top_k = top_k

    def query(self, question: str) -> Dict[str, Any]:
        """
        Execute the full RAG pipeline for a user question.

        Args:
            question: Natural language question about EV documents.

        Returns:
            Dict with keys:
              - answer:           Generated answer string.
              - sources:          List of source reference dicts.
              - retrieval_count:  Number of chunks retrieved.
              - question:         Echo of the original question.

        Raises:
            ValueError: If the question is empty.
        """
        if not question or not question.strip():
            raise ValueError("Question must not be empty.")

        question = question.strip()
        logger.info("Processing query: '%s'", question[:80])

        # Step 1: Embed the question
        query_embedding = self._embedder.embed_one(question)

        # Step 2: Retrieve top-K chunks from ChromaDB
        retrieved = self._vector_store.query(
            query_embedding=query_embedding,
            n_results=self._top_k,
        )
        logger.info("Retrieved %d chunks for query.", len(retrieved))

        # Step 3: Generate answer via LLM
        answer = self._llm.generate_answer(
            question=question,
            retrieved_chunks=retrieved,
        )

        # Step 4: Build source reference list
        sources = self._build_sources(retrieved)

        return {
            "question": question,
            "answer": answer,
            "sources": sources,
            "retrieval_count": len(retrieved),
        }

    # ──────────────────────────────────────────────────────────────────────────
    # Helpers
    # ──────────────────────────────────────────────────────────────────────────

    @staticmethod
    def _build_sources(chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Build a deduplicated, human-readable list of source references.

        Args:
            chunks: Retrieved chunks with metadata.

        Returns:
            List of source dicts with filename, page, chunk_type, excerpt, distance.
        """
        sources: List[Dict[str, Any]] = []
        seen_ids: set = set()

        for chunk in chunks:
            chunk_id = chunk.get("chunk_id", "")
            if chunk_id in seen_ids:
                continue
            seen_ids.add(chunk_id)

            content = chunk.get("content", "")
            # Provide a short excerpt (first 200 chars)
            excerpt = content[:200].strip()
            if len(content) > 200:
                excerpt += "…"

            sources.append(
                {
                    "filename": chunk.get("source", "unknown"),
                    "page": chunk.get("page", 0),
                    "chunk_type": chunk.get("type", "unknown"),
                    "excerpt": excerpt,
                    "relevance_score": round(1.0 - chunk.get("distance", 0.0), 4),
                }
            )

        # Sort by relevance (highest first)
        sources.sort(key=lambda s: s["relevance_score"], reverse=True)
        return sources
