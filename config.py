"""
Application configuration using Pydantic Settings.
All environment variables are loaded from a .env file.
"""

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Central configuration for the EV Multimodal RAG system."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # ── Gemini API ─────────────────────────────────────────────────────────────
    GEMINI_API_KEY: str
    GEMINI_MODEL: str = "gemini-2.0-flash"

    # ── Embedding (local, sentence-transformers — no API key required) ─────────
    EMBEDDING_MODEL: str = "all-MiniLM-L6-v2"

    # ── ChromaDB (persistent, local vector store) ──────────────────────────────
    CHROMA_PERSIST_DIR: str = "./chroma_db"
    COLLECTION_NAME: str = "ev_rag_multimodal"

    # ── Chunking ───────────────────────────────────────────────────────────────
    CHUNK_SIZE: int = 1500       # characters per text chunk
    CHUNK_OVERLAP: int = 200     # overlap between consecutive chunks

    # ── Retrieval ──────────────────────────────────────────────────────────────
    TOP_K: int = 5               # number of chunks to retrieve per query

    # ── Ingestion limits ───────────────────────────────────────────────────────
    MAX_FILE_SIZE_MB: int = 50


# Singleton settings object shared across the application
settings = Settings()
