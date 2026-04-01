"""
ChromaDB Vector Store wrapper.

Manages a persistent ChromaDB collection that stores document chunks
with their embeddings and metadata. Supports add, query, and delete
operations with full metadata filtering support.

Why ChromaDB over FAISS:
  - Native metadata filtering (filter by source filename, chunk type, page)
  - Persistent on-disk storage without manual serialisation
  - Built-in deduplication via document IDs
  - Simple Python-native API with no external server requirement
"""

import logging
from typing import Any, Dict, List, Optional

import chromadb
from chromadb.config import Settings as ChromaSettings

logger = logging.getLogger(__name__)


class VectorStore:
    """Persistent ChromaDB-backed vector store for EV document chunks."""

    def __init__(self, persist_dir: str, collection_name: str) -> None:
        """
        Initialise (or reconnect to) a persistent ChromaDB collection.

        Args:
            persist_dir:     Directory path where ChromaDB stores its data.
            collection_name: Name of the collection to use/create.
        """
        self._client = chromadb.PersistentClient(
            path=persist_dir,
            settings=ChromaSettings(anonymized_telemetry=False),
        )
        self._collection = self._client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},  # Cosine similarity for normalised embeddings
        )
        logger.info(
            "VectorStore connected. Collection '%s' has %d items.",
            collection_name,
            self._collection.count(),
        )

    # ──────────────────────────────────────────────────────────────────────────
    # Write operations
    # ──────────────────────────────────────────────────────────────────────────

    def add_chunks(
        self,
        chunk_ids: List[str],
        documents: List[str],
        embeddings: List[List[float]],
        metadatas: List[Dict[str, Any]],
    ) -> None:
        """
        Upsert document chunks into the collection.

        Using upsert (not add) ensures re-ingesting the same PDF replaces
        stale chunks rather than duplicating them.

        Args:
            chunk_ids:  Unique identifiers for each chunk.
            documents:  Text content of each chunk.
            embeddings: Pre-computed embedding vectors.
            metadatas:  Metadata dicts (must contain 'type', 'source', 'page').
        """
        if not chunk_ids:
            return

        self._collection.upsert(
            ids=chunk_ids,
            documents=documents,
            embeddings=embeddings,
            metadatas=metadatas,
        )
        logger.debug("Upserted %d chunks into the collection.", len(chunk_ids))

    # ──────────────────────────────────────────────────────────────────────────
    # Read operations
    # ──────────────────────────────────────────────────────────────────────────

    def query(
        self,
        query_embedding: List[float],
        n_results: int = 5,
        where: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Retrieve the top-n most similar chunks for a query embedding.

        Args:
            query_embedding: Embedding of the user's query.
            n_results:       Number of chunks to return.
            where:           Optional ChromaDB metadata filter dict.

        Returns:
            List of chunk dicts with keys: content, type, source, page,
            chunk_id, distance.
        """
        total = self._collection.count()
        if total == 0:
            return []

        effective_n = min(n_results, total)

        kwargs: Dict[str, Any] = {
            "query_embeddings": [query_embedding],
            "n_results": effective_n,
            "include": ["documents", "metadatas", "distances"],
        }
        if where:
            kwargs["where"] = where

        results = self._collection.query(**kwargs)

        chunks: List[Dict[str, Any]] = []
        ids = results.get("ids", [[]])[0]
        docs = results.get("documents", [[]])[0]
        metas = results.get("metadatas", [[]])[0]
        dists = results.get("distances", [[]])[0]

        for chunk_id, doc, meta, dist in zip(ids, docs, metas, dists):
            chunks.append(
                {
                    "chunk_id": chunk_id,
                    "content": doc,
                    "type": meta.get("type", "unknown"),
                    "source": meta.get("source", "unknown"),
                    "page": meta.get("page", 0),
                    "distance": round(float(dist), 4),
                }
            )

        return chunks

    def delete_by_source(self, filename: str) -> int:
        """
        Remove all chunks belonging to a specific source document.

        Args:
            filename: The PDF filename to remove.

        Returns:
            Number of chunks deleted.
        """
        results = self._collection.get(
            where={"source": filename},
            include=["documents"],
        )
        ids = results.get("ids", [])
        if ids:
            self._collection.delete(ids=ids)
            logger.info("Deleted %d chunks for source '%s'.", len(ids), filename)
        return len(ids)

    def list_sources(self) -> List[str]:
        """Return a deduplicated list of all ingested source filenames."""
        if self._collection.count() == 0:
            return []
        results = self._collection.get(include=["metadatas"])
        sources = {m.get("source", "") for m in results.get("metadatas", [])}
        return sorted(s for s in sources if s)

    # ──────────────────────────────────────────────────────────────────────────
    # Stats
    # ──────────────────────────────────────────────────────────────────────────

    def count(self) -> int:
        """Total number of chunks in the collection."""
        return self._collection.count()

    def count_by_source(self, filename: str) -> int:
        """Number of chunks for a specific source document."""
        results = self._collection.get(where={"source": filename}, include=["documents"])
        return len(results.get("ids", []))
