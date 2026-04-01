"""
FastAPI route definitions for the EV Multimodal RAG API.

Endpoints:
  GET  /health      — System readiness and index statistics
  POST /ingest      — Upload and index a multimodal PDF
  POST /query       — Ask a question against indexed documents
  GET  /documents   — List all indexed documents
  DELETE /documents/{filename} — Remove a document from the index
"""

import logging
import time
from typing import Annotated

from fastapi import APIRouter, Depends, File, HTTPException, Request, UploadFile, status

from src.api.schemas import (
    DeleteResponse,
    DocumentInfo,
    DocumentListResponse,
    HealthResponse,
    IngestResponse,
    QueryRequest,
    QueryResponse,
)
from src.ingestion.parser import PDFParser

logger = logging.getLogger(__name__)

router = APIRouter()


# ── Dependency helpers ────────────────────────────────────────────────────────

def get_embedder(request: Request):
    return request.app.state.embedder


def get_vector_store(request: Request):
    return request.app.state.vector_store


def get_vlm(request: Request):
    return request.app.state.vlm


def get_retriever(request: Request):
    return request.app.state.retriever


def get_settings(request: Request):
    return request.app.state.settings


def get_start_time(request: Request):
    return request.app.state.start_time


# ── GET /health ───────────────────────────────────────────────────────────────

@router.get(
    "/health",
    response_model=HealthResponse,
    summary="System Health Check",
    tags=["System"],
)
def health_check(
    start_time: float = Depends(get_start_time),
    vector_store=Depends(get_vector_store),
    embedder=Depends(get_embedder),
    settings=Depends(get_settings),
) -> HealthResponse:
    """
    Returns current system health, model readiness, and vector index statistics.

    Use this endpoint to verify the system is running and to inspect how many
    documents and chunks have been indexed.
    """
    indexed_filenames = vector_store.list_sources()
    uptime = round(time.time() - start_time, 2)

    return HealthResponse(
        status="healthy",
        gemini_model=settings.GEMINI_MODEL,
        embedding_model=embedder.model_name,
        indexed_documents=len(indexed_filenames),
        total_chunks=vector_store.count(),
        indexed_filenames=indexed_filenames,
        uptime_seconds=uptime,
    )


# ── POST /ingest ──────────────────────────────────────────────────────────────

@router.post(
    "/ingest",
    response_model=IngestResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Ingest a Multimodal PDF",
    tags=["Ingestion"],
)
async def ingest_document(
    file: Annotated[UploadFile, File(description="A multimodal PDF containing text, tables, and/or images.")],
    embedder=Depends(get_embedder),
    vector_store=Depends(get_vector_store),
    vlm=Depends(get_vlm),
    settings=Depends(get_settings),
) -> IngestResponse:
    """
    Parse, summarise, embed, and index a PDF document.

    **Processing pipeline:**
    1. PDF is parsed to extract text chunks, table chunks, and raw images.
    2. Each image is passed through the Gemini Vision model to generate a
       text description (the description is then embedded, not the raw image).
    3. All chunk types are embedded using sentence-transformers.
    4. Chunks are upserted into ChromaDB with full metadata.

    Re-ingesting the same filename will overwrite previous chunks for that file.
    """
    # ── Validate file type ────────────────────────────────────────────────────
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Only PDF files are accepted. Please upload a file with a .pdf extension.",
        )

    # ── Validate file size ────────────────────────────────────────────────────
    content = await file.read()
    max_bytes = settings.MAX_FILE_SIZE_MB * 1024 * 1024
    if len(content) > max_bytes:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"File exceeds maximum allowed size of {settings.MAX_FILE_SIZE_MB} MB.",
        )

    t_start = time.time()
    filename = file.filename

    # ── Save to temp file for parsing ─────────────────────────────────────────
    import tempfile, os
    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
        tmp.write(content)
        tmp_path = tmp.name

    try:
        # ── Parse PDF ────────────────────────────────────────────────────────
        parser = PDFParser(
            chunk_size=settings.CHUNK_SIZE,
            chunk_overlap=settings.CHUNK_OVERLAP,
        )
        parsed = parser.parse(tmp_path)

        text_chunks = parsed["text"]
        table_chunks = parsed["table"]
        image_chunks_raw = parsed["image"]

        # ── Summarise images via VLM ─────────────────────────────────────────
        image_chunks: list = []
        for img_chunk in image_chunks_raw:
            summary = vlm.summarise_image(
                image_b64=img_chunk["content"],
                image_ext=img_chunk.get("image_ext", "png"),
            )
            if summary and "[Image summary unavailable" not in summary:
                image_chunks.append(
                    {
                        "type": "image",
                        "content": summary,  # Store the text description, not raw bytes
                        "page": img_chunk["page"],
                        "source": filename,
                        "chunk_id": img_chunk["chunk_id"],
                    }
                )

        all_chunks = text_chunks + table_chunks + image_chunks
        if not all_chunks:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail="No extractable content found in the PDF. The file may be image-only or corrupted.",
            )

        # ── Remove stale chunks for this document (re-ingest) ────────────────
        vector_store.delete_by_source(filename)

        # ── Embed all chunks ─────────────────────────────────────────────────
        texts = [c["content"] for c in all_chunks]
        embeddings = embedder.embed(texts)

        # ── Build metadata and IDs ────────────────────────────────────────────
        chunk_ids = [c["chunk_id"] for c in all_chunks]
        metadatas = [
            {
                "type": c["type"],
                "source": c["source"],
                "page": int(c["page"]),
            }
            for c in all_chunks
        ]

        # ── Upsert into ChromaDB ──────────────────────────────────────────────
        vector_store.add_chunks(
            chunk_ids=chunk_ids,
            documents=texts,
            embeddings=embeddings,
            metadatas=metadatas,
        )

        elapsed = round(time.time() - t_start, 2)
        logger.info(
            "Ingested '%s': %d text, %d table, %d image chunks in %.2fs.",
            filename, len(text_chunks), len(table_chunks), len(image_chunks), elapsed,
        )

        return IngestResponse(
            message=f"Document '{filename}' ingested successfully.",
            filename=filename,
            chunks={
                "text": len(text_chunks),
                "table": len(table_chunks),
                "image": len(image_chunks),
                "total": len(all_chunks),
            },
            processing_time_seconds=elapsed,
        )

    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Ingestion failed for '%s': %s", filename, exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Ingestion failed: {exc}",
        ) from exc
    finally:
        os.unlink(tmp_path)


# ── POST /query ───────────────────────────────────────────────────────────────

@router.post(
    "/query",
    response_model=QueryResponse,
    summary="Query Indexed Documents",
    tags=["Retrieval"],
)
def query_documents(
    payload: QueryRequest,
    retriever=Depends(get_retriever),
    vector_store=Depends(get_vector_store),
) -> QueryResponse:
    """
    Ask a natural language question against all indexed EV documents.

    **Retrieval pipeline:**
    1. The question is embedded using sentence-transformers.
    2. Top-K most semantically similar chunks are retrieved from ChromaDB
       (spanning text, table, and image-summary chunk types).
    3. Retrieved chunks are passed to Gemini with a strict RAG prompt.
    4. The generated answer is returned along with source references.
    """
    if vector_store.count() == 0:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="No documents have been indexed yet. Please POST to /ingest first.",
        )

    try:
        result = retriever.query(question=payload.question)
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)
        ) from exc
    except RuntimeError as exc:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"LLM generation failed: {exc}",
        ) from exc
    except Exception as exc:
        logger.exception("Query pipeline error: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Query failed: {exc}",
        ) from exc

    return QueryResponse(**result)


# ── GET /documents ────────────────────────────────────────────────────────────

@router.get(
    "/documents",
    response_model=DocumentListResponse,
    summary="List Indexed Documents",
    tags=["Documents"],
)
def list_documents(vector_store=Depends(get_vector_store)) -> DocumentListResponse:
    """
    Return metadata for all documents currently in the vector index.
    """
    sources = vector_store.list_sources()
    docs = [
        DocumentInfo(filename=src, chunk_count=vector_store.count_by_source(src))
        for src in sources
    ]
    return DocumentListResponse(total_documents=len(docs), documents=docs)


# ── DELETE /documents/{filename} ──────────────────────────────────────────────

@router.delete(
    "/documents/{filename}",
    response_model=DeleteResponse,
    summary="Delete a Document from the Index",
    tags=["Documents"],
)
def delete_document(
    filename: str,
    vector_store=Depends(get_vector_store),
) -> DeleteResponse:
    """
    Remove all chunks for the specified PDF from the vector index.

    The filename must exactly match the name used during ingestion
    (e.g., `tata_nexon_ev_report.pdf`).
    """
    sources = vector_store.list_sources()
    if filename not in sources:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Document '{filename}' not found in the index.",
        )

    deleted = vector_store.delete_by_source(filename)
    return DeleteResponse(
        message=f"Document '{filename}' removed from the index.",
        filename=filename,
        chunks_deleted=deleted,
    )
