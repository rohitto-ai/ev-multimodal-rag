"""
EV Multimodal RAG System — FastAPI Application Entry Point

This system enables engineers and product teams at TATA Motors to query
multimodal EV technical documents (battery datasheets, performance reports,
homologation docs) using natural language.

Startup sequence:
  1. Load configuration from .env
  2. Initialise sentence-transformer embedding model (local, no API)
  3. Connect to (or create) ChromaDB persistent collection
  4. Initialise Gemini LLM and VLM clients
  5. Wire retriever pipeline
  6. Mount FastAPI routes and start serving

Run:
  uvicorn main:app --host 0.0.0.0 --port 8000 --reload
"""

import logging
import time
from contextlib import asynccontextmanager

import google.generativeai as genai
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse

from config import settings
from src.api.routes import router
from src.api.ui import build_dashboard_html
from src.ingestion.embedder import Embedder
from src.models.llm import LanguageModel
from src.models.vlm import VisionModel
from src.retrieval.retriever import Retriever
from src.retrieval.vector_store import VectorStore

# ── Logging configuration ──────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


# ── Application lifespan ───────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Manage application startup and shutdown.

    All heavyweight initialisations (model loading, DB connection) happen
    here so they are performed once at startup, not per-request.
    """
    logger.info("═══ EV Multimodal RAG System — Starting Up ═══")

    # Record startup time for uptime reporting
    app.state.start_time = time.time()

    # Validate Gemini API key when present; keep the app bootable without it so
    # the dashboard and schema can still be explored locally.
    if settings.GEMINI_API_KEY:
        try:
            genai.configure(api_key=settings.GEMINI_API_KEY)
            logger.info("Gemini API key validated.")
        except Exception as exc:
            logger.error("Gemini API key configuration failed: %s", exc)
            raise
    else:
        logger.warning("GEMINI_API_KEY is not set. Ingest and query endpoints will return 503 until configured.")

    # Load embedding model (sentence-transformers, local)
    logger.info("Loading embedding model '%s' ...", settings.EMBEDDING_MODEL)
    embedder = Embedder(model_name=settings.EMBEDDING_MODEL)
    app.state.embedder = embedder

    # Connect to ChromaDB
    logger.info("Connecting to ChromaDB at '%s' ...", settings.CHROMA_PERSIST_DIR)
    vector_store = VectorStore(
        persist_dir=settings.CHROMA_PERSIST_DIR,
        collection_name=settings.COLLECTION_NAME,
    )
    app.state.vector_store = vector_store

    # Initialise VLM (Gemini Vision)
    logger.info("Initialising Vision Language Model ...")
    vlm = None
    if settings.GEMINI_API_KEY:
        vlm = VisionModel(
            api_key=settings.GEMINI_API_KEY,
            model_name=settings.GEMINI_MODEL,
        )
    app.state.vlm = vlm

    # Initialise LLM (Gemini)
    logger.info("Initialising Language Model ...")
    llm = None
    if settings.GEMINI_API_KEY:
        llm = LanguageModel(
            api_key=settings.GEMINI_API_KEY,
            model_name=settings.GEMINI_MODEL,
        )
    app.state.llm = llm

    # Wire retriever pipeline
    retriever = None
    if llm is not None:
        retriever = Retriever(
            vector_store=vector_store,
            embedder=embedder,
            llm=llm,
            top_k=settings.TOP_K,
        )
    app.state.retriever = retriever

    # Expose settings for route dependency injection
    app.state.settings = settings

    logger.info(
        "Startup complete. %d documents already indexed.",
        len(vector_store.list_sources()),
    )
    logger.info("═══ System Ready — Serving requests ═══")

    yield  # Application runs here

    # ── Shutdown ──────────────────────────────────────────────────────────────
    logger.info("═══ EV Multimodal RAG System — Shutting Down ═══")


# ── FastAPI application ────────────────────────────────────────────────────────

app = FastAPI(
    title="EV Multimodal RAG System",
    description=(
        "A Retrieval-Augmented Generation (RAG) API for querying multimodal "
        "Electric Vehicle (EV) technical documents from TATA Motors. "
        "Supports ingestion of PDFs with text, tables, and images, and "
        "generates grounded answers using Google Gemini."
    ),
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
    contact={
        "name": "Rohit Bhardwaj",
        "email": "2024tm05006@wilp.bits-pilani.ac.in",
    },
    license_info={
        "name": "MIT",
    },
)

# ── CORS middleware ────────────────────────────────────────────────────────────
# Allow all origins for local development; tighten in production.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Mount routes ───────────────────────────────────────────────────────────────
app.include_router(router, prefix="")


# ── Root redirect ──────────────────────────────────────────────────────────────
@app.get("/", include_in_schema=False, response_class=HTMLResponse)
def root():
    return build_dashboard_html()


@app.get("/api", include_in_schema=False)
def api_root():
    return {
        "message": "EV Multimodal RAG System is running.",
        "docs": "/docs",
        "health": "/health",
        "ui": "/",
    }


# ── Development entry point ───────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info",
    )
