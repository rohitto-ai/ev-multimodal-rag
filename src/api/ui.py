"""
Browser dashboard for the EV Multimodal RAG system.

The dashboard is intentionally self-contained so the app can be tested from a
single webpage without any extra frontend build tooling.
"""


def build_dashboard_html() -> str:
    """Return the HTML used for the landing page dashboard."""
    return """
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>EV Multimodal RAG Dashboard</title>
  <style>
    :root {
      --bg: #07111f;
      --panel: rgba(10, 18, 32, 0.92);
      --panel-strong: #0d1728;
      --panel-soft: #11223d;
      --line: rgba(170, 196, 255, 0.16);
      --text: #eaf1ff;
      --muted: #9db0cf;
      --accent: #8ec5ff;
      --accent-2: #67d7c0;
      --warn: #ffb86b;
      --danger: #ff6e7f;
      --ok: #57e39f;
      --shadow: 0 24px 70px rgba(0, 0, 0, 0.45);
      --radius: 22px;
    }

    * { box-sizing: border-box; }
    html, body { margin: 0; min-height: 100%; }
    body {
      color: var(--text);
      background:
        radial-gradient(circle at top left, rgba(142, 197, 255, 0.18), transparent 30%),
        radial-gradient(circle at 80% 20%, rgba(103, 215, 192, 0.14), transparent 26%),
        radial-gradient(circle at 30% 80%, rgba(255, 184, 107, 0.10), transparent 25%),
        linear-gradient(180deg, #07111f 0%, #091426 100%);
      font-family: "Segoe UI", "Helvetica Neue", Arial, sans-serif;
      line-height: 1.45;
    }

    .wrap {
      max-width: 1320px;
      margin: 0 auto;
      padding: 32px 20px 48px;
    }

    .hero {
      display: grid;
      grid-template-columns: 1.55fr 0.95fr;
      gap: 20px;
      align-items: stretch;
      margin-bottom: 20px;
    }

    .panel {
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: var(--radius);
      box-shadow: var(--shadow);
      backdrop-filter: blur(16px);
    }

    .hero-copy {
      padding: 28px;
      position: relative;
      overflow: hidden;
      min-height: 260px;
    }

    .hero-copy::after {
      content: "";
      position: absolute;
      inset: auto -80px -80px auto;
      width: 240px;
      height: 240px;
      border-radius: 50%;
      background: radial-gradient(circle, rgba(142, 197, 255, 0.22), transparent 70%);
      pointer-events: none;
    }

    .eyebrow {
      display: inline-flex;
      gap: 8px;
      align-items: center;
      padding: 8px 12px;
      border-radius: 999px;
      background: rgba(142, 197, 255, 0.12);
      color: var(--accent);
      font-size: 12px;
      letter-spacing: 0.08em;
      text-transform: uppercase;
      margin-bottom: 16px;
    }

    h1 {
      margin: 0 0 12px;
      font-size: clamp(30px, 4.6vw, 58px);
      line-height: 0.98;
      letter-spacing: -0.04em;
    }

    .lede {
      max-width: 68ch;
      color: var(--muted);
      margin: 0 0 20px;
      font-size: 15px;
    }

    .actions {
      display: flex;
      flex-wrap: wrap;
      gap: 10px;
    }

    .btn {
      appearance: none;
      border: 1px solid var(--line);
      background: linear-gradient(180deg, rgba(20, 34, 56, 0.92), rgba(13, 23, 40, 0.96));
      color: var(--text);
      padding: 12px 16px;
      border-radius: 14px;
      cursor: pointer;
      font-weight: 700;
      transition: transform 0.15s ease, border-color 0.15s ease, background 0.15s ease;
    }

    .btn:hover { transform: translateY(-1px); border-color: rgba(142, 197, 255, 0.42); }
    .btn.primary { background: linear-gradient(135deg, #8ec5ff, #67d7c0); color: #07111f; border-color: transparent; }
    .btn.warn { background: linear-gradient(135deg, rgba(255, 184, 107, 0.18), rgba(255, 184, 107, 0.08)); }
    .btn.danger { background: linear-gradient(135deg, rgba(255, 110, 127, 0.16), rgba(255, 110, 127, 0.08)); }

    .hero-stats {
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 14px;
    }

    .stat {
      padding: 18px;
      background: linear-gradient(180deg, rgba(17, 34, 61, 0.96), rgba(10, 18, 32, 0.96));
      border: 1px solid var(--line);
      border-radius: 18px;
      min-height: 114px;
    }

    .stat .label { color: var(--muted); font-size: 12px; text-transform: uppercase; letter-spacing: 0.08em; }
    .stat .value { font-size: 30px; font-weight: 800; margin-top: 10px; letter-spacing: -0.03em; }
    .stat .sub { color: var(--muted); font-size: 13px; margin-top: 6px; }

    .grid {
      display: grid;
      grid-template-columns: 1.08fr 0.92fr;
      gap: 20px;
      align-items: start;
    }

    .card {
      padding: 22px;
    }

    .card h2 {
      margin: 0 0 12px;
      font-size: 18px;
      letter-spacing: -0.02em;
    }

    .card p.helper {
      margin: 0 0 16px;
      color: var(--muted);
      font-size: 14px;
    }

    .form {
      display: grid;
      gap: 12px;
    }

    .field {
      display: grid;
      gap: 8px;
    }

    .field label {
      font-size: 12px;
      text-transform: uppercase;
      letter-spacing: 0.08em;
      color: var(--muted);
    }

    input[type="text"], textarea, input[type="file"] {
      width: 100%;
      color: var(--text);
      background: rgba(6, 12, 22, 0.9);
      border: 1px solid rgba(173, 198, 255, 0.18);
      border-radius: 14px;
      padding: 14px;
      outline: none;
      font: inherit;
    }

    textarea { min-height: 126px; resize: vertical; }

    input:focus, textarea:focus {
      border-color: rgba(142, 197, 255, 0.55);
      box-shadow: 0 0 0 3px rgba(142, 197, 255, 0.12);
    }

    .inline-actions {
      display: flex;
      flex-wrap: wrap;
      gap: 10px;
    }

    .output {
      min-height: 86px;
      margin-top: 14px;
      padding: 16px;
      border-radius: 18px;
      border: 1px solid rgba(173, 198, 255, 0.16);
      background: rgba(6, 12, 22, 0.86);
      color: var(--text);
      white-space: pre-wrap;
      overflow-wrap: anywhere;
    }

    .chips {
      display: flex;
      flex-wrap: wrap;
      gap: 8px;
      margin-top: 14px;
    }

    .chip {
      padding: 7px 11px;
      border-radius: 999px;
      font-size: 12px;
      background: rgba(142, 197, 255, 0.12);
      color: var(--accent);
      border: 1px solid rgba(142, 197, 255, 0.2);
      cursor: pointer;
    }

    .list {
      display: grid;
      gap: 10px;
      margin-top: 12px;
    }

    .doc-row {
      display: flex;
      justify-content: space-between;
      gap: 16px;
      align-items: center;
      padding: 14px 16px;
      border-radius: 14px;
      background: rgba(6, 12, 22, 0.86);
      border: 1px solid rgba(173, 198, 255, 0.14);
    }

    .doc-row strong { display: block; margin-bottom: 4px; }
    .doc-row span { color: var(--muted); font-size: 13px; }

    .pill {
      display: inline-flex;
      align-items: center;
      gap: 8px;
      padding: 8px 10px;
      border-radius: 999px;
      background: rgba(103, 215, 192, 0.12);
      color: var(--accent-2);
      font-size: 12px;
      border: 1px solid rgba(103, 215, 192, 0.18);
    }

    .foot {
      margin-top: 18px;
      color: var(--muted);
      font-size: 13px;
    }

    @media (max-width: 1100px) {
      .hero, .grid { grid-template-columns: 1fr; }
    }
  </style>
</head>
<body>
  <div class="wrap">
    <section class="hero">
      <div class="panel hero-copy">
        <div class="eyebrow">EV Multimodal RAG Dashboard</div>
        <h1>Query EV PDFs with a single browser page.</h1>
        <p class="lede">
          Upload a multimodal PDF, inspect the indexed documents, and ask questions that span text,
          tables, and image summaries. The blue-highlighted blocks in the architecture diagram map to
          the API calls and external model services behind this interface.
        </p>
        <div class="actions">
          <a class="btn primary" href="/docs" style="text-decoration:none;display:inline-flex;align-items:center;">Open Swagger UI</a>
          <button class="btn" onclick="refreshHealth()">Refresh Health</button>
          <button class="btn warn" onclick="loadDocuments()">Load Documents</button>
        </div>
        <div class="chips">
          <span class="chip" onclick="setQuestion('What is the battery capacity and ARAI certified range of the Nexon EV Max?')">Text query</span>
          <span class="chip" onclick="setQuestion('Which table lists the charging specifications and fast-charging power?')">Table query</span>
          <span class="chip" onclick="setQuestion('What does the charging curve image show?')">Image query</span>
        </div>
      </div>

      <div class="panel card">
        <h2>System Status</h2>
        <p class="helper">Live health and document counts from the running API.</p>
        <div class="hero-stats">
          <div class="stat">
            <div class="label">Status</div>
            <div class="value" id="health-status">-</div>
            <div class="sub" id="health-model">Waiting for API...</div>
          </div>
          <div class="stat">
            <div class="label">Documents</div>
            <div class="value" id="health-docs">-</div>
            <div class="sub" id="health-chunks">Chunks: -</div>
          </div>
          <div class="stat">
            <div class="label">Embedder</div>
            <div class="value" style="font-size:22px;line-height:1.15;" id="health-embedder">-</div>
            <div class="sub" id="health-uptime">Uptime: -</div>
          </div>
          <div class="stat">
            <div class="label">Docs API</div>
            <div class="value" style="font-size:22px;line-height:1.15;">/docs</div>
            <div class="sub">Interactive endpoint explorer</div>
          </div>
        </div>
      </div>
    </section>

    <section class="grid">
      <div class="panel card">
        <h2>Ingest PDF</h2>
        <p class="helper">Upload a multimodal PDF to extract text, tables, and images, then index them into ChromaDB.</p>
        <div class="form">
          <div class="field">
            <label for="pdf-file">PDF file</label>
            <input id="pdf-file" type="file" accept="application/pdf" />
          </div>
          <div class="inline-actions">
            <button class="btn primary" onclick="ingestPdf()">Upload and index</button>
            <button class="btn danger" onclick="clearOutput('ingest-output')">Clear</button>
          </div>
        </div>
        <div class="output" id="ingest-output">No ingestion performed yet.</div>
      </div>

      <div class="panel card">
        <h2>Query Documents</h2>
        <p class="helper">Ask a question and receive a grounded answer with retrieved sources.</p>
        <div class="form">
          <div class="field">
            <label for="question">Question</label>
            <textarea id="question" placeholder="What safety standards govern the Nexon EV battery pack?"></textarea>
          </div>
          <div class="inline-actions">
            <button class="btn primary" onclick="queryRag()">Run query</button>
            <button class="btn danger" onclick="clearOutput('query-output')">Clear</button>
          </div>
        </div>
        <div class="output" id="query-output">No query submitted yet.</div>
      </div>
    </section>

    <section class="grid" style="margin-top:20px;">
      <div class="panel card">
        <h2>Indexed Documents</h2>
        <p class="helper">Documents currently stored in the vector index.</p>
        <div class="inline-actions" style="margin-bottom:10px;">
          <button class="btn" onclick="loadDocuments()">Refresh list</button>
        </div>
        <div class="list" id="documents-list">
          <div class="output">Click refresh to load the current index.</div>
        </div>
      </div>

      <div class="panel card">
        <h2>Quick Tips</h2>
        <p class="helper">Use the browser page to validate the assignment end-to-end.</p>
        <div class="output">
          1. Open /docs to verify the OpenAPI schema.
          2. Upload the sample PDF through the ingest form.
          3. Run at least one text query, one table query, and one image query.
          4. Confirm /health shows the indexed document count.
          5. Capture screenshots for submission after the API returns results.
        </div>
      </div>
    </section>

    <p class="foot">
      This page is served directly by FastAPI. No separate frontend build step is required.
    </p>
  </div>

  <script>
    async function refreshHealth() {
      try {
        const response = await fetch('/health');
        const data = await response.json();
        document.getElementById('health-status').textContent = data.status || '-';
        document.getElementById('health-model').textContent = data.gemini_model ? `Model: ${data.gemini_model}` : 'Model: -';
        document.getElementById('health-docs').textContent = data.indexed_documents ?? '-';
        document.getElementById('health-chunks').textContent = `Chunks: ${data.total_chunks ?? '-'}`;
        document.getElementById('health-embedder').textContent = data.embedding_model || '-';
        document.getElementById('health-uptime').textContent = `Uptime: ${data.uptime_seconds ?? '-'}s`;
      } catch (error) {
        document.getElementById('health-status').textContent = 'offline';
        document.getElementById('health-model').textContent = 'Unable to reach /health';
      }
    }

    async function loadDocuments() {
      const container = document.getElementById('documents-list');
      container.innerHTML = '<div class="output">Loading documents...</div>';
      try {
        const response = await fetch('/documents');
        const data = await response.json();
        if (!data.documents || data.documents.length === 0) {
          container.innerHTML = '<div class="output">No documents indexed yet.</div>';
          return;
        }
        container.innerHTML = data.documents.map(doc => `
          <div class="doc-row">
            <div>
              <strong>${doc.filename}</strong>
              <span>${doc.chunk_count} chunks indexed</span>
            </div>
            <button class="btn danger" onclick="deleteDocument('${doc.filename.replace(/'/g, "\\'")}')">Delete</button>
          </div>
        `).join('');
      } catch (error) {
        container.innerHTML = `<div class="output">Failed to load documents: ${error}</div>`;
      }
    }

    async function ingestPdf() {
      const fileInput = document.getElementById('pdf-file');
      const output = document.getElementById('ingest-output');
      if (!fileInput.files.length) {
        output.textContent = 'Choose a PDF file first.';
        return;
      }

      const formData = new FormData();
      formData.append('file', fileInput.files[0]);
      output.textContent = 'Uploading and indexing...';

      try {
        const response = await fetch('/ingest', { method: 'POST', body: formData });
        const data = await response.json();
        output.textContent = JSON.stringify(data, null, 2);
        await refreshHealth();
        await loadDocuments();
      } catch (error) {
        output.textContent = `Ingestion failed: ${error}`;
      }
    }

    async function queryRag() {
      const question = document.getElementById('question').value.trim();
      const output = document.getElementById('query-output');
      if (!question) {
        output.textContent = 'Enter a question first.';
        return;
      }

      output.textContent = 'Retrieving and generating...';

      try {
        const response = await fetch('/query', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ question }),
        });
        const data = await response.json();
        output.textContent = JSON.stringify(data, null, 2);
      } catch (error) {
        output.textContent = `Query failed: ${error}`;
      }
    }

    async function deleteDocument(filename) {
      if (!confirm(`Delete ${filename}?`)) {
        return;
      }

      const output = document.getElementById('documents-list');
      try {
        const response = await fetch(`/documents/${encodeURIComponent(filename)}`, { method: 'DELETE' });
        const data = await response.json();
        output.innerHTML = `<div class="output">${JSON.stringify(data, null, 2)}</div>`;
        await refreshHealth();
        await loadDocuments();
      } catch (error) {
        output.innerHTML = `<div class="output">Delete failed: ${error}</div>`;
      }
    }

    function setQuestion(text) {
      document.getElementById('question').value = text;
    }

    function clearOutput(id) {
      document.getElementById(id).textContent = 'Cleared.';
    }

    refreshHealth();
    loadDocuments();
  </script>
</body>
</html>
""".strip()