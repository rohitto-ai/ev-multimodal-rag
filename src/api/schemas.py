"""
Pydantic request and response schemas for the EV Multimodal RAG API.

Using Pydantic models ensures:
  - Automatic OpenAPI/Swagger documentation
  - Request validation with meaningful error messages
  - Type-safe response serialisation
  - Clear API contracts for downstream consumers
"""

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


# ── /health ───────────────────────────────────────────────────────────────────

class HealthResponse(BaseModel):
    """System health and readiness status."""

    status: str = Field(..., examples=["healthy"], description="Overall system status.")
    gemini_model: str = Field(..., description="Gemini model in use for LLM and VLM.")
    embedding_model: str = Field(..., description="Sentence-transformer model for embeddings.")
    indexed_documents: int = Field(..., description="Number of unique PDF files indexed.")
    total_chunks: int = Field(..., description="Total number of chunks in the vector index.")
    indexed_filenames: List[str] = Field(default_factory=list, description="List of indexed PDF filenames.")
    uptime_seconds: float = Field(..., description="Seconds since the server started.")


# ── /ingest ───────────────────────────────────────────────────────────────────

class ChunkCounts(BaseModel):
    """Breakdown of ingested chunks by type."""

    text: int = Field(..., description="Number of text chunks.")
    table: int = Field(..., description="Number of table chunks.")
    image: int = Field(..., description="Number of image chunks (with VLM summaries).")
    total: int = Field(..., description="Total chunks ingested from this document.")


class IngestResponse(BaseModel):
    """Response returned after successfully ingesting a PDF."""

    message: str = Field(..., description="Human-readable status message.")
    filename: str = Field(..., description="Name of the ingested PDF file.")
    chunks: ChunkCounts = Field(..., description="Chunk breakdown by modality.")
    processing_time_seconds: float = Field(
        ..., description="Total wall-clock time for parsing, summarising, and indexing."
    )


# ── /query ────────────────────────────────────────────────────────────────────

class QueryRequest(BaseModel):
    """Payload for the /query endpoint."""

    question: str = Field(
        ...,
        min_length=3,
        max_length=1000,
        examples=["What is the battery capacity of the TATA Nexon EV Max?"],
        description="Natural language question about the indexed EV documents.",
    )


class SourceReference(BaseModel):
    """A single source chunk referenced in the generated answer."""

    filename: str = Field(..., description="PDF filename the chunk was extracted from.")
    page: int = Field(..., description="1-indexed page number within the PDF.")
    chunk_type: str = Field(
        ..., examples=["text", "table", "image"], description="Modality of the retrieved chunk."
    )
    excerpt: str = Field(..., description="Short excerpt of the retrieved chunk content.")
    relevance_score: float = Field(
        ..., ge=0.0, le=1.0, description="Cosine similarity score (1.0 = most relevant)."
    )


class QueryResponse(BaseModel):
    """Response returned by the /query endpoint."""

    question: str = Field(..., description="Echo of the original question.")
    answer: str = Field(..., description="LLM-generated, grounded answer.")
    sources: List[SourceReference] = Field(
        default_factory=list,
        description="Retrieved chunks that informed the answer.",
    )
    retrieval_count: int = Field(
        ..., description="Number of chunks retrieved from the vector index."
    )


# ── /documents ────────────────────────────────────────────────────────────────

class DocumentInfo(BaseModel):
    """Metadata for a single indexed document."""

    filename: str = Field(..., description="PDF filename.")
    chunk_count: int = Field(..., description="Number of chunks stored for this document.")


class DocumentListResponse(BaseModel):
    """List of all documents currently in the vector index."""

    total_documents: int = Field(..., description="Number of indexed documents.")
    documents: List[DocumentInfo] = Field(..., description="Per-document metadata.")


# ── /delete ───────────────────────────────────────────────────────────────────

class DeleteResponse(BaseModel):
    """Response after deleting a document from the index."""

    message: str = Field(..., description="Status message.")
    filename: str = Field(..., description="The deleted document's filename.")
    chunks_deleted: int = Field(..., description="Number of chunks removed from the index.")


# ── Generic error ─────────────────────────────────────────────────────────────

class ErrorResponse(BaseModel):
    """Standard error response body."""

    detail: str = Field(..., description="Human-readable error description.")
    error_type: Optional[str] = Field(None, description="Machine-readable error category.")
