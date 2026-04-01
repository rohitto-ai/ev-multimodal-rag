"""
PDF Parser — extracts text, tables, and images from multimodal PDF documents.

Uses:
  - PyMuPDF (fitz) for page-level text extraction and image extraction
  - pdfplumber  for structured table detection and markdown conversion

Each chunk is returned as a dictionary with keys:
  - type:     "text" | "table" | "image"
  - content:  raw text / markdown table / base64-encoded image
  - page:     1-indexed page number
  - source:   original PDF filename
  - chunk_id: unique identifier for deduplication
"""

import base64
import logging
from pathlib import Path
from typing import Any, Dict, List

import fitz  # PyMuPDF
import pdfplumber

logger = logging.getLogger(__name__)

# Minimum image dimensions to skip tiny icons / artifacts
_MIN_IMAGE_WIDTH = 50
_MIN_IMAGE_HEIGHT = 50


class PDFParser:
    """Parses a PDF and returns typed chunks (text, table, image)."""

    def __init__(self, chunk_size: int = 1500, chunk_overlap: int = 200) -> None:
        """
        Args:
            chunk_size:    Maximum character length of each text chunk.
            chunk_overlap: Overlap between consecutive text chunks (characters).
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    # ──────────────────────────────────────────────────────────────────────────
    # Public API
    # ──────────────────────────────────────────────────────────────────────────

    def parse(self, pdf_path: str) -> Dict[str, List[Dict[str, Any]]]:
        """
        Parse a PDF file and return all extracted chunks by type.

        Args:
            pdf_path: Absolute path to the PDF file.

        Returns:
            Dict with keys "text", "table", "image" mapping to lists of chunks.

        Raises:
            FileNotFoundError: If the PDF does not exist.
            ValueError:        If the file is not a valid PDF.
        """
        path = Path(pdf_path)
        if not path.exists():
            raise FileNotFoundError(f"PDF not found: {pdf_path}")
        if path.suffix.lower() != ".pdf":
            raise ValueError(f"Expected a .pdf file, got: {path.suffix}")

        filename = path.name
        logger.info("Parsing PDF: %s", filename)

        text_chunks: List[Dict[str, Any]] = []
        table_chunks: List[Dict[str, Any]] = []
        image_chunks: List[Dict[str, Any]] = []

        # ── Image extraction (PyMuPDF) ────────────────────────────────────────
        image_chunks = self._extract_images(pdf_path, filename)

        # ── Text + Table extraction (pdfplumber) ─────────────────────────────
        with pdfplumber.open(pdf_path) as pdf:
            for page_num, page in enumerate(pdf.pages, start=1):
                # Tables first
                page_tables = self._extract_tables(page, page_num, filename)
                table_chunks.extend(page_tables)

                # Text (full page — some overlap with table text is acceptable)
                page_text_chunks = self._extract_text(page, page_num, filename)
                text_chunks.extend(page_text_chunks)

        logger.info(
            "Parsed '%s': %d text, %d table, %d image chunks.",
            filename,
            len(text_chunks),
            len(table_chunks),
            len(image_chunks),
        )

        return {
            "text": text_chunks,
            "table": table_chunks,
            "image": image_chunks,
        }

    # ──────────────────────────────────────────────────────────────────────────
    # Private helpers
    # ──────────────────────────────────────────────────────────────────────────

    def _extract_images(
        self, pdf_path: str, filename: str
    ) -> List[Dict[str, Any]]:
        """Extract images from all pages using PyMuPDF."""
        chunks: List[Dict[str, Any]] = []
        img_idx = 0

        try:
            doc = fitz.open(pdf_path)
        except Exception as exc:
            logger.error("PyMuPDF failed to open '%s': %s", pdf_path, exc)
            return chunks

        seen_xrefs: set = set()

        for page_num, page in enumerate(doc, start=1):
            image_list = page.get_images(full=True)
            for img_info in image_list:
                xref = img_info[0]
                if xref in seen_xrefs:
                    continue
                seen_xrefs.add(xref)

                try:
                    base_image = doc.extract_image(xref)
                    w, h = base_image.get("width", 0), base_image.get("height", 0)

                    if w < _MIN_IMAGE_WIDTH or h < _MIN_IMAGE_HEIGHT:
                        logger.debug("Skipping tiny image (%dx%d) on page %d.", w, h, page_num)
                        continue

                    image_bytes = base_image["image"]
                    image_ext = base_image.get("ext", "png")

                    chunks.append(
                        {
                            "type": "image",
                            "content": base64.b64encode(image_bytes).decode("utf-8"),
                            "image_ext": image_ext,
                            "page": page_num,
                            "source": filename,
                            "chunk_id": f"{filename}_image_p{page_num}_{img_idx}",
                        }
                    )
                    img_idx += 1

                except Exception as exc:  # noqa: BLE001
                    logger.warning("Failed to extract image xref %d: %s", xref, exc)

        doc.close()
        return chunks

    def _extract_tables(
        self, page: Any, page_num: int, filename: str
    ) -> List[Dict[str, Any]]:
        """Extract tables from a pdfplumber page and convert to Markdown."""
        chunks: List[Dict[str, Any]] = []

        try:
            tables = page.extract_tables()
        except Exception as exc:
            logger.warning("Table extraction failed on page %d: %s", page_num, exc)
            return chunks

        for t_idx, table in enumerate(tables):
            if not table:
                continue

            # Filter out completely empty rows
            non_empty_rows = [
                row for row in table if any(cell for cell in row if cell)
            ]
            if len(non_empty_rows) < 2:  # Need at least a header + one data row
                continue

            markdown = self._table_to_markdown(non_empty_rows)
            if not markdown.strip():
                continue

            chunks.append(
                {
                    "type": "table",
                    "content": markdown,
                    "page": page_num,
                    "source": filename,
                    "chunk_id": f"{filename}_table_p{page_num}_{t_idx}",
                }
            )

        return chunks

    def _extract_text(
        self, page: Any, page_num: int, filename: str
    ) -> List[Dict[str, Any]]:
        """Extract and chunk plain text from a pdfplumber page."""
        try:
            raw_text = page.extract_text()
        except Exception as exc:
            logger.warning("Text extraction failed on page %d: %s", page_num, exc)
            return []

        if not raw_text or not raw_text.strip():
            return []

        return self._split_text(raw_text, page_num, filename)

    def _split_text(
        self, text: str, page: int, source: str
    ) -> List[Dict[str, Any]]:
        """
        Split text into overlapping character-based chunks.

        Attempts to split at sentence boundaries within the last 20% of
        each chunk window to preserve semantic coherence.
        """
        chunks: List[Dict[str, Any]] = []
        text = text.strip()
        start = 0
        chunk_idx = 0

        while start < len(text):
            end = min(start + self.chunk_size, len(text))

            # Try to break at a sentence boundary
            if end < len(text):
                search_start = start + int(self.chunk_size * 0.7)
                for sep in (". ", ".\n", "! ", "? ", "?\n", "!\n"):
                    pos = text.rfind(sep, search_start, end)
                    if pos != -1:
                        end = pos + len(sep)
                        break

            chunk_text = text[start:end].strip()
            if len(chunk_text) > 30:  # Skip trivially short fragments
                chunks.append(
                    {
                        "type": "text",
                        "content": chunk_text,
                        "page": page,
                        "source": source,
                        "chunk_id": f"{source}_text_p{page}_{chunk_idx}",
                    }
                )
                chunk_idx += 1

            next_start = end - self.chunk_overlap
            if next_start <= start:
                next_start = start + 1  # Prevent infinite loop on very short text
            start = next_start

        return chunks

    @staticmethod
    def _table_to_markdown(table: List[List[Any]]) -> str:
        """Convert a list-of-lists table to GitHub-Flavoured Markdown."""
        if not table:
            return ""

        # Normalise cells
        def clean(cell: Any) -> str:
            return str(cell).replace("\n", " ").strip() if cell is not None else ""

        rows = [[clean(c) for c in row] for row in table]
        col_count = max(len(r) for r in rows)

        # Pad short rows
        rows = [r + [""] * (col_count - len(r)) for r in rows]

        header = rows[0]
        separator = ["---"] * col_count
        lines = [
            "| " + " | ".join(header) + " |",
            "| " + " | ".join(separator) + " |",
        ]
        for row in rows[1:]:
            lines.append("| " + " | ".join(row) + " |")

        return "\n".join(lines)
