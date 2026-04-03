---
title: EV Multimodal RAG
emoji: 🚗
colorFrom: blue
colorTo: green
sdk: docker
pinned: false
---

# EV Multimodal RAG System — TATA Motors Electric Vehicle Knowledge Base

[![FastAPI](https://img.shields.io/badge/FastAPI-0.115-009688?logo=fastapi)](https://fastapi.tiangolo.com)
[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?logo=python)](https://python.org)
[![Gemini](https://img.shields.io/badge/LLM-Gemini%201.5%20Flash-4285F4?logo=google)](https://aistudio.google.com)
[![ChromaDB](https://img.shields.io/badge/VectorDB-ChromaDB-FF6B35)](https://www.trychroma.com)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow)](LICENSE)

> **BITS WILP — Multimodal RAG Bootcamp | Individual Assignment**
> **Student:** Rohit Bhardwaj | **Email:** 2024tm05006@wilp.bits-pilani.ac.in

---

## Problem Statement

### Domain: Electric Vehicle Engineering at TATA Motors

As a member of the Electric Vehicle (EV) product engineering team at TATA Motors, I work daily with
a large corpus of heterogeneous technical documents — battery datasheets, homologation reports,
ARAI certification submissions, thermal management white papers, motor controller specifications,
and charging infrastructure compliance records. These documents are dense, multimodal artefacts that
combine structured data (cell chemistry tables, performance matrices, regulatory checklists), visual
information (battery architecture schematics, range-versus-speed curves, charging time bar charts),
and dense engineering prose that ties them together.

### The Problem

Engineers and product managers at TATA Motors currently struggle to extract precise, cross-referenced
insights from this document corpus for two core reasons:

**First, document heterogeneity.** A single question — "What is the maximum DC fast-charging power
supported by the Nexon EV Max, and under what temperature conditions does the BMS throttle charging?" —
may require simultaneously reading a table of charging port specifications on page 5, a paragraph about
BMS thermal control strategy on page 12, and interpreting a performance chart on page 9. No keyword
search tool can reason across these three modalities in a single pass. Traditional enterprise search
returns pages; it does not synthesise answers.

**Second, specialised domain terminology.** Indian EV regulatory documents reference standards like
AIS-038, AIS-156, and IS-17017 that generic language models have not been trained on in depth.
Acronyms like SoH (State of Health), PMSM, CCS2, UN GTR 20, and NMC 622 carry very specific technical
meaning. A general-purpose chatbot without grounded context from the actual documents frequently
halluccinates specifications or confuses similar models.

### Why This Problem Is Unique

Unlike a generic document Q&A system, the EV engineering domain presents a specific challenge set:
tables contain battery electrochemical data with units, tolerance ranges, and standard references that
must be read together; diagrams show BMS architecture, thermal loops, and power electronics topology
that require engineering interpretation, not just caption reading; and performance charts (range vs.
speed curves, charging time plots) encode quantitative insight that text alone cannot convey. A system
that treats all content as plain text loses the structured meaning of a table row or the trend encoded
in a chart. Additionally, homologation documents are updated quarterly as regulatory standards evolve,
making a static fine-tuned model impractical — the knowledge must be updatable by re-ingesting new PDFs.

### Why RAG Is the Right Approach

Fine-tuning a large language model on this corpus would require curating thousands of question-answer
pairs from restricted internal documents, retraining every quarter when standards are updated, and
accepting a model that cannot cite its sources — a hard compliance requirement in automotive
engineering. Keyword search returns documents, not answers. RAG uniquely solves all three problems:
it retrieves only the relevant chunks (text, table, or image description) at query time, grounds the
LLM response in those specific chunks, provides verifiable citations (filename + page number), and
updates the knowledge base simply by re-running POST /ingest with a new PDF — no model retraining required.

### Expected Outcomes

A successful system should enable an engineer to ask: "Which Indian safety standards govern the Nexon
EV battery pack, and what is the cell-level IP rating?", and receive a grounded answer that cites the
exact table row from the homologation document and the page it appears on. It should answer queries
that require combining a table (e.g., charging specs) with an image description (e.g., a charging time
chart) — demonstrating true cross-modal retrieval. It should support the product manager asking "what
is the 0–100 km/h time for the Nexon EV Max?" from a specification sheet without knowing which page it
is on or what document it lives in.

---

## Architecture Overview

### System Architecture Diagram

```mermaid
flowchart LR
  classDef apiCall fill:#8ec5ff,stroke:#4b86c2,stroke-width:1px,color:#0b2740;
  classDef lightBox fill:#ffffff,stroke:#6f6f6f,stroke-width:1px,color:#1f1f1f;
  classDef note fill:#f8f8f8,stroke:#b5b5b5,stroke-dasharray: 4 3,color:#222;

  subgraph Notes[ ]
    Y1[YAML\nParser = docling\nChunking = hybrid chunker\nEmbedding = IBM Granite\nVLM = ...\nLLM = ...\nGuardrail = ...]:::note
    Y2[YAML\nParser = docling\nChunking = hybrid chunker\nEmbedding = IBM Granite\nVLM = ...\nLLM = ...\nGuardrail = ...]:::note
    Y3[Flagship as baseline\nGemini / ChatGPT / Claude]:::apiCall
  end

  subgraph Production[Production Grade RAG]
    O[OBSERVABILITY PLATFORM]:::apiCall

    subgraph Ingest[ ]
      D[Multi-Modal PDF\nText\nTables\nImages]:::lightBox --> P[Docling]:::apiCall
      P --> T[Text]:::lightBox
      P --> I[Images]:::lightBox
      P --> B[Tables]:::lightBox
      T --> C[hybridChunker\nIBM Granite Embedder (30M)]:::apiCall
      I --> C
      B --> C
      C --> V[(Vector Database)]:::lightBox
    end

    A[API Server]:::apiCall
    U[UI - Web based App]:::lightBox

    A -->|Data Ingestion| D
    A -->|Data Fetch| V
    A -->|Query| F[FAISS\nGranite\nText Based\nVision Based]:::apiCall
    V -->|Similarity Search| F
    U <--> A
  end

  O --- Production
```

Blue blocks indicate API calls or external services, matching the highlighted nodes in the PNG reference.

This repository keeps the same overall flow, but the concrete implementation is mapped to the local stack in code:

- `Docling` in the PNG corresponds to [src/ingestion/parser.py](src/ingestion/parser.py), which uses PyMuPDF + pdfplumber.
- `IBM Granite Embedder` in the PNG corresponds to [src/ingestion/embedder.py](src/ingestion/embedder.py), which uses sentence-transformers.
- `Vector Database` in the PNG corresponds to [src/retrieval/vector_store.py](src/retrieval/vector_store.py), which uses ChromaDB.
- `FAISS` in the PNG corresponds to the retrieval path in [src/retrieval/retriever.py](src/retrieval/retriever.py), implemented against ChromaDB.
- `API Server` and `UI - Web based App` in the PNG correspond to the FastAPI app in [main.py](main.py) and the generated `/docs` UI.

### Ingestion Flow

1. Client uploads a PDF via `POST /ingest`
2. `PDFParser` uses **PyMuPDF** to extract text and raw images page-by-page
3. **pdfplumber** detects table regions and converts them to Markdown
4. Each extracted image is sent to **Gemini 1.5 Flash Vision** for a text description
5. All chunk types (text, table, image-description) are embedded via **sentence-transformers**
6. Chunks are upserted into **ChromaDB** with metadata: `{type, source, page}`

### Query Flow

1. Client POSTs a question to `/query`
2. The question is embedded using the same sentence-transformer model
3. ChromaDB performs cosine-similarity search returning top-K chunks across all modalities
4. Chunks are assembled into a structured context string
5. **Gemini 1.5 Flash** generates a grounded answer using a strict RAG prompt template
6. The response includes the answer and a list of source references with page numbers

---

## Technology Choices

| Component | Choice | Justification |
|-----------|--------|---------------|
| **Document Parser (Text)** | PyMuPDF (`fitz`) | Fastest PDF text extractor; handles complex PDFs including scanned text with embedded fonts. Superior to pdfminer for engineering docs with mixed content. |
| **Document Parser (Tables)** | pdfplumber | Specialised table detection via heuristic bbox analysis; returns cell-level content unlike PyMuPDF's raw text stream. Enables clean Markdown table generation. |
| **Embedding Model** | sentence-transformers `all-MiniLM-L6-v2` | Fully local execution (no API quota, no latency), 384-dim embeddings, strong semantic similarity for technical English. Avoids Gemini embedding API rate limits during bulk ingestion. |
| **Vector Store** | ChromaDB | Persistent on-disk storage without a separate server process. Native metadata filtering (`where={"type": "table"}`) allows future filtered retrieval per chunk type. Simpler to deploy than Pinecone for a local-first system. |
| **LLM (Answer Generation)** | Google Gemini 1.5 Flash | Free tier via Google AI Studio (15 RPM, 1M tokens/day). 1M-token context window. Outperforms open-source 7B models on instruction following for structured RAG prompts. No GPU required. |
| **VLM (Image Summarisation)** | Google Gemini 1.5 Flash | Same model handles both text and vision natively, reducing integration complexity. Accurately describes engineering charts, schematics, and data tables from EV PDFs. Free tier sufficient. |
| **API Framework** | FastAPI | Native Pydantic integration for automatic Swagger/OpenAPI docs. Async support for file uploads. Better type safety than Flask for production-grade API contracts. |

---

## Setup Instructions

### Prerequisites

- Python 3.10 or higher
- A free Google Gemini API key from [Google AI Studio](https://aistudio.google.com/app/apikey)
- ~2 GB free disk space (for model weights and ChromaDB)

### Step 1: Clone the Repository

```bash
git clone https://github.com/YOUR_USERNAME/ev-multimodal-rag.git
cd ev-multimodal-rag
```

### Step 2: Create a Virtual Environment

```bash
python -m venv venv

# macOS / Linux
source venv/bin/activate

# Windows
venv\Scripts\activate
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

> **Note:** The first run downloads the `all-MiniLM-L6-v2` sentence-transformer model (~90 MB)
> from HuggingFace. Subsequent runs use the local cache.

### Step 4: Configure Environment Variables

```bash
cp .env.example .env
```

Open `.env` and set your Gemini API key:

```env
GEMINI_API_KEY=your_gemini_api_key_here
```

All other settings have sensible defaults and do not need to be changed for a basic setup.

### Step 5: Start the Server

```bash
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

The server starts at `http://localhost:8000`. The browser dashboard is at `http://localhost:8000/`, and the Swagger UI is at `http://localhost:8000/docs`.

If `GEMINI_API_KEY` is not set, the dashboard and health endpoint still load, but `/ingest` and `/query` will return `503 Service Unavailable` until the key is configured.

### Step 6: Ingest the Sample Document

```bash
curl -X POST "http://localhost:8000/ingest" \
  -F "file=@sample_documents/tata_nexon_ev_technical_report.pdf"
```

### Step 7: Run a Query

```bash
curl -X POST "http://localhost:8000/query" \
  -H "Content-Type: application/json" \
  -d '{"question": "What is the battery capacity and ARAI certified range of the Nexon EV Max?"}'
```

---

## API Documentation

### `GET /health`

Returns system readiness, model information, and vector index statistics.

**Sample Response:**
```json
{
  "status": "healthy",
  "gemini_model": "gemini-1.5-flash",
  "embedding_model": "all-MiniLM-L6-v2",
  "indexed_documents": 1,
  "total_chunks": 47,
  "indexed_filenames": ["tata_nexon_ev_technical_report.pdf"],
  "uptime_seconds": 142.5
}
```

---

### `POST /ingest`

Upload a multimodal PDF to parse, summarise, embed, and index.

**Request:** `multipart/form-data` with field `file` (PDF only, max 50 MB)

**Sample Response (201 Created):**
```json
{
  "message": "Document 'tata_nexon_ev_technical_report.pdf' ingested successfully.",
  "filename": "tata_nexon_ev_technical_report.pdf",
  "chunks": {
    "text": 28,
    "table": 12,
    "image": 7,
    "total": 47
  },
  "processing_time_seconds": 14.3
}
```

**Error Responses:**
- `400` — Non-PDF file uploaded
- `413` — File exceeds 50 MB limit
- `422` — PDF has no extractable content
- `500` — Internal parsing or embedding error

---

### `POST /query`

Ask a natural language question against all indexed documents.

**Request Body:**
```json
{
  "question": "What safety standards govern the Nexon EV battery pack?"
}
```

**Sample Response (200 OK):**
```json
{
  "question": "What safety standards govern the Nexon EV battery pack?",
  "answer": "The TATA Nexon EV battery pack is governed by several Indian and international standards including AIS-038 (Rev 1) for electric powertrain safety, AIS-156 (Phase 1 & 2) for battery pack safety, AIS-049 for high-voltage safety, and IS 17017 for charging systems. The pack has also achieved IP67 certification per IEC 60529 and passed UN GTR 20 for thermal runaway propagation prevention. [Source: tata_nexon_ev_technical_report.pdf, Page 6, Table 4]",
  "sources": [
    {
      "filename": "tata_nexon_ev_technical_report.pdf",
      "page": 6,
      "chunk_type": "table",
      "excerpt": "| AIS-038 (Rev 1) | Electric power train — safety requirements | Certified | ARAI |...",
      "relevance_score": 0.9341
    },
    {
      "filename": "tata_nexon_ev_technical_report.pdf",
      "page": 6,
      "chunk_type": "text",
      "excerpt": "The Nexon EV platform has been validated against all mandatory Indian automotive safety standards applicable to Battery Electric Vehicles...",
      "relevance_score": 0.8876
    }
  ],
  "retrieval_count": 5
}
```

**Error Responses:**
- `400` — Empty or too-short question
- `404` — No documents have been indexed yet
- `502` — Gemini API generation failure

---

### `GET /documents`

List all currently indexed documents with chunk counts.

**Sample Response:**
```json
{
  "total_documents": 2,
  "documents": [
    {"filename": "tata_nexon_ev_technical_report.pdf", "chunk_count": 47},
    {"filename": "nexon_ev_service_manual_2024.pdf",   "chunk_count": 83}
  ]
}
```

---

### `DELETE /documents/{filename}`

Remove a document and all its chunks from the vector index.

**Sample Response:**
```json
{
  "message": "Document 'tata_nexon_ev_technical_report.pdf' removed from the index.",
  "filename": "tata_nexon_ev_technical_report.pdf",
  "chunks_deleted": 47
}
```

---

## Screenshots

Screenshots are captured after running the system end-to-end with the sample PDF.
All screenshots are in the `screenshots/` folder and embedded below.

### 1. Swagger UI — All Endpoints
![Swagger UI](screenshots/01_swagger_ui.png)

### 2. POST /ingest — Successful Ingestion
![Ingest Response](screenshots/02_ingest_response.png)

### 3. POST /query — Text-Based Answer
![Text Query](screenshots/03_query_text.png)

### 4. POST /query — Table-Based Answer
![Table Query](screenshots/04_query_table.png)

### 5. POST /query — Image Summary Answer
![Image Query](screenshots/05_query_image.png)

### 6. GET /health — Index Status
![Health Check](screenshots/06_health_check.png)

> **Note:** Screenshots are added after the system is running locally.
> To reproduce: start the server, ingest the sample PDF, and run the queries shown in Setup Instructions.

---

## Limitations & Future Work

### Current Limitations

**PDF parsing quality:** PyMuPDF and pdfplumber work well for digitally-created PDFs but may produce
poor results on scanned documents without embedded text layers. OCR integration (e.g., Tesseract or
Amazon Textract) would be needed for older TATA Motors homologation documents that exist only as scans.

**Image filtering:** The system uses minimum dimension thresholds to skip decorative images, but may
occasionally include page logos or watermarks and send them to the VLM, adding latency. A more
sophisticated image classifier (e.g., filtering out images with near-uniform colour histograms) would
reduce unnecessary VLM calls.

**Chunk boundaries:** The character-based chunker with sentence-boundary heuristics can split at
sub-optimal points for highly structured text (e.g., numbered lists in regulatory documents). A
semantic chunker using embeddings to detect topic shifts would preserve coherence better.

**Multilingual support:** TATA Motors documents include Hindi text in some regulatory filings. The
`all-MiniLM-L6-v2` embedding model is English-centric and degrades on non-English input.

**Gemini API rate limits:** The free tier allows 15 requests per minute, which can become a bottleneck
when ingesting PDFs with many images (each image = one VLM API call). A job queue with exponential
backoff would make this more robust.

### Future Work

- **Semantic chunking** using embedding-based topic segmentation for higher-quality chunk boundaries
- **OCR pipeline** using Tesseract or Docling's built-in OCR for scanned PDF support
- **Re-ranking** using a cross-encoder model (e.g., `ms-marco-MiniLM-L-6-v2`) to reorder retrieved chunks before LLM generation
- **Multi-document reasoning** with explicit citation tracking per sentence in the generated answer
- **Streaming responses** via FastAPI's `StreamingResponse` for long answers to improve perceived latency
- **Authentication middleware** with API key validation for production deployment
- **Docker Compose** setup for reproducible deployment across TATA Motors engineering workstations

---

## Project Structure

```
ev-multimodal-rag/
├── README.md                        # This file
├── main.py                          # FastAPI app + lifespan management
├── config.py                        # Pydantic settings (all env vars)
├── requirements.txt                 # Pinned Python dependencies
├── .env.example                     # API key template (copy to .env)
├── .gitignore                       # Excludes .env, chroma_db/, cache/
│
├── src/
│   ├── ingestion/
│   │   ├── parser.py                # PDF → text/table/image chunks (PyMuPDF + pdfplumber)
│   │   └── embedder.py              # sentence-transformers wrapper
│   ├── retrieval/
│   │   ├── vector_store.py          # ChromaDB wrapper (upsert, query, delete)
│   │   └── retriever.py             # RAG pipeline orchestrator
│   ├── models/
│   │   ├── vlm.py                   # Gemini Vision (image → text summary)
│   │   └── llm.py                   # Gemini LLM (context → grounded answer)
│   └── api/
│       ├── schemas.py               # Pydantic request/response models
│       └── routes.py                # FastAPI route handlers
│
├── sample_documents/
│   └── tata_nexon_ev_technical_report.pdf   # Multimodal EV sample PDF
│
└── screenshots/
    ├── 01_swagger_ui.png
    ├── 02_ingest_response.png
    ├── 03_query_text.png
    ├── 04_query_table.png
    ├── 05_query_image.png
    └── 06_health_check.png
```

---

## License

MIT License — see [LICENSE](LICENSE) for details.

---

*BITS Pilani • Work Integrated Learning Programmes • Multimodal RAG Bootcamp • 2024*
