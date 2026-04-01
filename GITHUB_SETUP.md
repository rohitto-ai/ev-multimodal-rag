# How to Publish This Project to GitHub

Follow these steps to create a public GitHub repository and push the project.

## Step 1: Create the GitHub Repository

1. Go to https://github.com/new
2. Set:
   - **Repository name:** `ev-multimodal-rag`
   - **Description:** `Multimodal RAG system for TATA Motors EV technical documents — BITS WILP Assignment`
   - **Visibility:** Public ✅ (required for submission)
   - Do NOT initialise with README (we already have one)
3. Click **Create repository**

## Step 2: Open Terminal in This Folder

Navigate to the `ev-multimodal-rag` folder on your computer, then run:

```bash
# Initialise git
git init

# Add all files
git add .

# First commit
git commit -m "feat: initial implementation of EV multimodal RAG system

- FastAPI server with /health, /ingest, /query, /documents endpoints
- Multimodal PDF parser (PyMuPDF + pdfplumber) for text, tables, images
- Gemini 1.5 Flash VLM for image summarisation
- Sentence-transformers (all-MiniLM-L6-v2) for local embeddings
- ChromaDB persistent vector store with cosine similarity
- TATA Nexon EV technical specification sample document
- Comprehensive README with architecture diagram and API docs"

# Add GitHub remote (replace YOUR_USERNAME)
git remote add origin https://github.com/YOUR_USERNAME/ev-multimodal-rag.git

# Push to GitHub
git branch -M main
git push -u origin main
```

## Step 3: Verify the Repository

Go to `https://github.com/YOUR_USERNAME/ev-multimodal-rag` and confirm:
- ✅ README renders correctly with the Mermaid diagram
- ✅ All source files are present
- ✅ `sample_documents/` folder contains the PDF
- ✅ `.env` file is NOT pushed (only `.env.example`)
- ✅ `chroma_db/` is NOT pushed

## Step 4: Submit the URL

Submit `https://github.com/YOUR_USERNAME/ev-multimodal-rag` via the course portal.

## Getting Your Gemini API Key (Free)

1. Visit https://aistudio.google.com/app/apikey
2. Sign in with your Google account
3. Click **Create API key**
4. Copy the key into your `.env` file as `GEMINI_API_KEY=your_key_here`

The free tier gives you **15 requests/minute** and **1 million tokens/day** —
more than enough for this assignment.
