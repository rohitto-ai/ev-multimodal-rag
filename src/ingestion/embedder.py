"""
Text Embedder — generates dense vector representations of document chunks.

Uses sentence-transformers (all-MiniLM-L6-v2) which runs fully locally
with no API calls required. Produces 384-dimensional normalised embeddings
suitable for cosine-similarity search in ChromaDB.

Model download happens on first use (~90 MB). Subsequent runs use the
HuggingFace cache (typically ~/.cache/huggingface/).
"""

import logging
from typing import List

from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)


class Embedder:
    """Wraps a sentence-transformer model for batch text embedding."""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2") -> None:
        """
        Load the embedding model.

        Args:
            model_name: HuggingFace model identifier.
                        Defaults to 'all-MiniLM-L6-v2' (384 dims, ~90 MB).
        """
        logger.info("Loading embedding model '%s' ...", model_name)
        self._model = SentenceTransformer(model_name)
        self._model_name = model_name
        self._dimension = self._model.get_sentence_embedding_dimension()
        logger.info(
            "Embedding model loaded. Dimension: %d.", self._dimension
        )

    # ──────────────────────────────────────────────────────────────────────────
    # Public API
    # ──────────────────────────────────────────────────────────────────────────

    def embed(self, texts: List[str]) -> List[List[float]]:
        """
        Embed a list of strings into dense vectors.

        Args:
            texts: List of text strings to embed.

        Returns:
            List of embedding vectors (each is a list of floats, length = dimension).

        Raises:
            ValueError: If texts list is empty.
        """
        if not texts:
            raise ValueError("Cannot embed an empty list of texts.")

        embeddings = self._model.encode(
            texts,
            normalize_embeddings=True,  # L2-normalised → cosine similarity = dot product
            show_progress_bar=False,
            batch_size=32,
        )
        return embeddings.tolist()

    def embed_one(self, text: str) -> List[float]:
        """
        Embed a single string.

        Args:
            text: Text to embed.

        Returns:
            Embedding vector as a list of floats.
        """
        return self.embed([text])[0]

    @property
    def dimension(self) -> int:
        """Embedding vector dimensionality."""
        return self._dimension

    @property
    def model_name(self) -> str:
        """HuggingFace model identifier."""
        return self._model_name
