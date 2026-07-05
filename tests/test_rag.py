"""
Tests for the RAG (Retrieval-Augmented Generation) pipeline.

Tests PDF text extraction, text chunking logic, and the ingestion pipeline.
Uses mocked database and model dependencies where needed.
"""

import pytest
from unittest.mock import patch, MagicMock
from contextlib import contextmanager
from pathlib import Path
import tempfile


def _mock_session_context(mock_session):
    """Create a proper context manager mock for @contextmanager."""
    @contextmanager
    def _ctx():
        yield mock_session
    return _ctx


class TestTextChunking:
    """Test the text chunking algorithm."""

    def test_short_text_single_chunk(self):
        """Text shorter than chunk_size should return a single chunk."""
        from rag.pdf_ingester import chunk_text

        text = "This is a short document with only a few words."
        chunks = chunk_text(text, chunk_size=500, overlap=50)
        assert len(chunks) == 1
        assert chunks[0] == text

    def test_chunking_produces_multiple_chunks(self):
        """Longer text should be split into multiple chunks."""
        from rag.pdf_ingester import chunk_text

        # Create text with 1000 words
        words = [f"word{i}" for i in range(1000)]
        text = " ".join(words)

        chunks = chunk_text(text, chunk_size=200, overlap=20)

        assert len(chunks) > 1
        # Each chunk should have approximately 200 words (except the last)
        for chunk in chunks[:-1]:
            chunk_words = chunk.split()
            assert len(chunk_words) == 200

    def test_chunking_with_overlap(self):
        """Consecutive chunks should share overlapping words."""
        from rag.pdf_ingester import chunk_text

        words = [f"word{i}" for i in range(100)]
        text = " ".join(words)

        chunks = chunk_text(text, chunk_size=30, overlap=10)

        # Check overlap between consecutive chunks
        for i in range(len(chunks) - 1):
            current_words = chunks[i].split()
            next_words = chunks[i + 1].split()

            # Last 10 words of current chunk should be first 10 of next
            if len(current_words) >= 10 and len(next_words) >= 10:
                overlap_current = current_words[-10:]
                overlap_next = next_words[:10]
                assert overlap_current == overlap_next, \
                    f"Chunk {i} and {i+1} don't have proper overlap"

    def test_empty_text(self):
        """Empty text should return single empty chunk."""
        from rag.pdf_ingester import chunk_text

        chunks = chunk_text("", chunk_size=500, overlap=50)
        assert len(chunks) == 1

    def test_chunk_size_zero_words(self):
        """Text with exactly chunk_size words should return one chunk."""
        from rag.pdf_ingester import chunk_text

        words = [f"word{i}" for i in range(500)]
        text = " ".join(words)

        chunks = chunk_text(text, chunk_size=500, overlap=50)
        assert len(chunks) == 1


class TestPDFExtraction:
    """Test PDF text extraction."""

    def test_nonexistent_file_raises(self):
        """Trying to extract from a non-existent file should raise."""
        from rag.pdf_ingester import extract_text_from_pdf

        with pytest.raises(FileNotFoundError):
            extract_text_from_pdf("/nonexistent/file.pdf")


class TestDocumentIngestion:
    """Test the document ingestion pipeline."""

    @patch("rag.pdf_ingester.generate_embeddings")
    @patch("rag.pdf_ingester.extract_text_from_pdf")
    @patch("rag.pdf_ingester.get_session")
    def test_ingest_skips_already_ingested(
        self, mock_get_session, mock_extract, mock_embed
    ):
        """Already-ingested documents should be skipped."""
        from rag.pdf_ingester import ingest_pdf

        # Mock session that finds an existing document
        mock_session = MagicMock()
        mock_session.query.return_value.filter.return_value.first.return_value = (
            MagicMock()  # Existing document
        )
        mock_get_session.side_effect = lambda: _mock_session_context(mock_session)()

        result = ingest_pdf("test_report.pdf")

        assert result == 0
        mock_extract.assert_not_called()
        mock_embed.assert_not_called()


class TestIngestedDocuments:
    """Test the ingested documents listing."""

    @patch("rag.pdf_ingester.get_session")
    def test_get_ingested_documents(self, mock_get_session):
        """Should return a list of documents with chunk counts."""
        from rag.pdf_ingester import get_ingested_documents

        mock_row1 = MagicMock()
        mock_row1.filename = "report1.pdf"
        mock_row1.chunk_count = 15

        mock_row2 = MagicMock()
        mock_row2.filename = "report2.pdf"
        mock_row2.chunk_count = 8

        mock_session = MagicMock()
        mock_session.query.return_value.group_by.return_value.all.return_value = [
            mock_row1, mock_row2
        ]
        mock_get_session.side_effect = lambda: _mock_session_context(mock_session)()

        result = get_ingested_documents()

        assert len(result) == 2
        assert result[0]["filename"] == "report1.pdf"
        assert result[0]["chunks"] == 15
        assert result[1]["filename"] == "report2.pdf"
        assert result[1]["chunks"] == 8
