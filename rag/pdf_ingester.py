"""
AURA — PDF Ingestion Pipeline.

Extracts text from PDF files, chunks it into overlapping segments,
generates embeddings via sentence-transformers, and stores everything
in PostgreSQL with pgvector for similarity search.
"""

import os
from pathlib import Path

from PyPDF2 import PdfReader
from sentence_transformers import SentenceTransformer

from config.settings import EMBEDDING_MODEL, RESEARCH_PDF_DIR
from finance.database import get_session
from finance.models import ResearchDocument


# ---------------------------------------------------------------------------
# Lazy-loaded embedding model
# ---------------------------------------------------------------------------
_embed_model: SentenceTransformer | None = None


def _get_embed_model() -> SentenceTransformer:
    """Return the sentence-transformer model, loading it on first call."""
    global _embed_model
    if _embed_model is None:
        print(f"[RAG] Loading embedding model: {EMBEDDING_MODEL}...")
        _embed_model = SentenceTransformer(EMBEDDING_MODEL)
        print("[RAG] Embedding model loaded.")
    return _embed_model


# ---------------------------------------------------------------------------
# Text Extraction
# ---------------------------------------------------------------------------

def extract_text_from_pdf(filepath: str | Path) -> str:
    """
    Extract all text from a PDF file.

    Args:
        filepath: Path to the PDF file.

    Returns:
        Concatenated text from all pages.
    """
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"PDF file not found: {filepath}")

    reader = PdfReader(str(filepath))
    text_parts = []

    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            text_parts.append(page_text.strip())

    full_text = "\n\n".join(text_parts)
    print(f"[RAG] Extracted {len(full_text)} characters from {filepath.name} ({len(reader.pages)} pages)")
    return full_text


# ---------------------------------------------------------------------------
# Text Chunking
# ---------------------------------------------------------------------------

def chunk_text(
    text: str,
    chunk_size: int = 500,
    overlap: int = 50,
) -> list[str]:
    """
    Split text into overlapping chunks by approximate word count.

    Args:
        text: The full document text.
        chunk_size: Target number of words per chunk.
        overlap: Number of overlapping words between consecutive chunks.

    Returns:
        List of text chunks.
    """
    words = text.split()
    if len(words) <= chunk_size:
        return [text]

    chunks = []
    start = 0

    while start < len(words):
        end = start + chunk_size
        chunk = " ".join(words[start:end])
        chunks.append(chunk)

        if end >= len(words):
            break
        start = end - overlap

    print(f"[RAG] Created {len(chunks)} chunks (size={chunk_size}, overlap={overlap})")
    return chunks


# ---------------------------------------------------------------------------
# Embedding Generation
# ---------------------------------------------------------------------------

def generate_embeddings(texts: list[str]) -> list[list[float]]:
    """
    Generate vector embeddings for a list of text chunks.

    Args:
        texts: List of text strings to embed.

    Returns:
        List of embedding vectors (each a list of floats).
    """
    model = _get_embed_model()
    embeddings = model.encode(texts, show_progress_bar=True, normalize_embeddings=True)
    return embeddings.tolist()


# ---------------------------------------------------------------------------
# PDF Ingestion (Full Pipeline)
# ---------------------------------------------------------------------------

def ingest_pdf(filepath: str | Path) -> int:
    """
    Full PDF ingestion pipeline: extract → chunk → embed → store.

    Args:
        filepath: Path to the PDF file.

    Returns:
        Number of chunks ingested.
    """
    filepath = Path(filepath)
    filename = filepath.name

    # Check if already ingested
    with get_session() as session:
        existing = (
            session.query(ResearchDocument)
            .filter(ResearchDocument.filename == filename)
            .first()
        )
        if existing:
            print(f"[RAG] '{filename}' already ingested. Skipping. Use re_ingest_pdf() to force.")
            return 0

    # Extract text
    text = extract_text_from_pdf(filepath)
    if not text.strip():
        print(f"[RAG] No text extracted from {filename}. Skipping.")
        return 0

    # Chunk text
    chunks = chunk_text(text)

    # Generate embeddings
    print(f"[RAG] Generating embeddings for {len(chunks)} chunks...")
    embeddings = generate_embeddings(chunks)

    # Store in database
    with get_session() as session:
        for chunk_text_content, embedding in zip(chunks, embeddings):
            doc = ResearchDocument(
                filename=filename,
                content=chunk_text_content,
                embedding=embedding,
            )
            session.add(doc)

    print(f"[RAG] Successfully ingested {len(chunks)} chunks from '{filename}'.")
    return len(chunks)


def re_ingest_pdf(filepath: str | Path) -> int:
    """
    Force re-ingestion of a PDF (deletes existing chunks first).

    Args:
        filepath: Path to the PDF file.

    Returns:
        Number of chunks ingested.
    """
    filepath = Path(filepath)
    filename = filepath.name

    # Delete existing chunks
    with get_session() as session:
        deleted = (
            session.query(ResearchDocument)
            .filter(ResearchDocument.filename == filename)
            .delete()
        )
        if deleted:
            print(f"[RAG] Deleted {deleted} existing chunks for '{filename}'.")

    return ingest_pdf(filepath)


def ingest_all_pdfs(directory: str | Path | None = None) -> dict[str, int]:
    """
    Ingest all PDF files in a directory.

    Args:
        directory: Path to directory containing PDFs.
                   Defaults to RESEARCH_PDF_DIR from config.

    Returns:
        Dict mapping filename to number of chunks ingested.
    """
    pdf_dir = Path(directory) if directory else RESEARCH_PDF_DIR
    pdf_dir.mkdir(parents=True, exist_ok=True)

    results = {}
    pdf_files = list(pdf_dir.glob("*.pdf"))

    if not pdf_files:
        print(f"[RAG] No PDF files found in {pdf_dir}")
        return results

    print(f"[RAG] Found {len(pdf_files)} PDF files to ingest.")

    for pdf_path in pdf_files:
        try:
            count = ingest_pdf(pdf_path)
            results[pdf_path.name] = count
        except Exception as e:
            print(f"[RAG ERROR] Failed to ingest {pdf_path.name}: {e}")
            results[pdf_path.name] = -1

    return results


def get_ingested_documents() -> list[dict]:
    """Return a list of all ingested documents with chunk counts."""
    with get_session() as session:
        from sqlalchemy import func
        docs = (
            session.query(
                ResearchDocument.filename,
                func.count(ResearchDocument.id).label("chunk_count"),
            )
            .group_by(ResearchDocument.filename)
            .all()
        )

    return [
        {"filename": d.filename, "chunks": d.chunk_count}
        for d in docs
    ]
