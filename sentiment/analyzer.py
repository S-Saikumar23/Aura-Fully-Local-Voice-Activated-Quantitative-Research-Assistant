"""
AURA — Sentiment Analysis Module.

Provides local sentiment analysis using a HuggingFace transformer model.
Supports analysis of raw text and ingested research documents.
All processing runs locally — no cloud APIs required.
"""

import re
from pathlib import Path

from transformers import pipeline

from config.settings import SENTIMENT_MODEL
from core.llm import ask_llm


# ---------------------------------------------------------------------------
# Lazy-loaded sentiment pipeline
# ---------------------------------------------------------------------------
_sentiment_pipeline = None


def _get_pipeline():
    """Return the sentiment analysis pipeline, loading on first call."""
    global _sentiment_pipeline
    if _sentiment_pipeline is None:
        print(f"[SENTIMENT] Loading model: {SENTIMENT_MODEL}...")
        _sentiment_pipeline = pipeline(
            "sentiment-analysis",
            model=SENTIMENT_MODEL,
            tokenizer=SENTIMENT_MODEL,
            top_k=None,  # Return all labels with scores
            truncation=True,
            max_length=512,
        )
        print("[SENTIMENT] Model loaded.")
    return _sentiment_pipeline


# ---------------------------------------------------------------------------
# Label Mapping
# ---------------------------------------------------------------------------
# The RoBERTa model uses LABEL_0/1/2 internally
_LABEL_MAP = {
    "LABEL_0": "negative",
    "LABEL_1": "neutral",
    "LABEL_2": "positive",
    # Some models use readable labels directly
    "negative": "negative",
    "neutral": "neutral",
    "positive": "positive",
}


def _normalize_label(label: str) -> str:
    """Map model labels to human-readable sentiment."""
    return _LABEL_MAP.get(label, label.lower())


# ---------------------------------------------------------------------------
# Core Analysis Functions
# ---------------------------------------------------------------------------

def analyze_sentiment(text: str) -> dict:
    """
    Analyze the sentiment of a text passage.

    Args:
        text: Text to analyze (max 512 tokens, longer text is truncated).

    Returns:
        Dict with 'label' (positive/negative/neutral), 'score',
        and 'all_scores' breakdown.
    """
    if not text or not text.strip():
        return {"label": "neutral", "score": 0.0, "all_scores": {}}

    pipe = _get_pipeline()
    results = pipe(text[:2000])  # Truncate very long text

    if not results or not results[0]:
        return {"label": "neutral", "score": 0.0, "all_scores": {}}

    # Results is a list of lists of dicts
    scores = {}
    for item in results[0]:
        label = _normalize_label(item["label"])
        scores[label] = round(item["score"], 4)

    # Determine dominant sentiment
    dominant = max(scores, key=scores.get)  # type: ignore[arg-type]

    return {
        "label": dominant,
        "score": scores[dominant],
        "all_scores": scores,
    }


def analyze_long_text(text: str, chunk_size: int = 400) -> dict:
    """
    Analyze sentiment of a long text by averaging chunk-level sentiments.

    Args:
        text: Long text (e.g., full earnings call transcript).
        chunk_size: Number of words per chunk for analysis.

    Returns:
        Dict with aggregate sentiment, chunk count, and score breakdown.
    """
    words = text.split()
    if len(words) <= chunk_size:
        return analyze_sentiment(text)

    # Split into chunks
    chunks = []
    for i in range(0, len(words), chunk_size):
        chunk = " ".join(words[i : i + chunk_size])
        chunks.append(chunk)

    # Analyze each chunk
    all_results = []
    for chunk in chunks:
        result = analyze_sentiment(chunk)
        all_results.append(result)

    # Aggregate scores
    agg_scores = {"positive": 0.0, "negative": 0.0, "neutral": 0.0}
    for r in all_results:
        for label, score in r["all_scores"].items():
            if label in agg_scores:
                agg_scores[label] += score

    # Average
    num_chunks = len(all_results)
    for label in agg_scores:
        agg_scores[label] = round(agg_scores[label] / num_chunks, 4)

    dominant = max(agg_scores, key=agg_scores.get)  # type: ignore[arg-type]

    return {
        "label": dominant,
        "score": agg_scores[dominant],
        "all_scores": agg_scores,
        "num_chunks_analyzed": num_chunks,
    }


def analyze_document_sentiment(filename: str) -> dict:
    """
    Analyze the aggregate sentiment of an ingested research document.

    Args:
        filename: Name of the ingested PDF file.

    Returns:
        Dict with aggregate sentiment analysis results.
    """
    from finance.database import get_session
    from finance.models import ResearchDocument

    with get_session() as session:
        chunks = (
            session.query(ResearchDocument.content)
            .filter(ResearchDocument.filename == filename)
            .order_by(ResearchDocument.id)
            .all()
        )

    if not chunks:
        return {
            "label": "unknown",
            "score": 0.0,
            "message": f"No document found: '{filename}'.",
        }

    # Analyze each stored chunk
    all_results = []
    for chunk in chunks:
        if chunk.content and chunk.content.strip():
            result = analyze_sentiment(chunk.content)
            all_results.append(result)

    if not all_results:
        return {"label": "neutral", "score": 0.0, "message": "No analyzable content."}

    # Aggregate
    agg_scores = {"positive": 0.0, "negative": 0.0, "neutral": 0.0}
    for r in all_results:
        for label, score in r["all_scores"].items():
            if label in agg_scores:
                agg_scores[label] += score

    num = len(all_results)
    for label in agg_scores:
        agg_scores[label] = round(agg_scores[label] / num, 4)

    dominant = max(agg_scores, key=agg_scores.get)  # type: ignore[arg-type]

    return {
        "filename": filename,
        "label": dominant,
        "score": agg_scores[dominant],
        "all_scores": agg_scores,
        "num_chunks_analyzed": num,
    }


# ---------------------------------------------------------------------------
# User-Facing Handler
# ---------------------------------------------------------------------------

def analyze_and_respond(query: str) -> str:
    """
    Handle a sentiment analysis request from the user.

    Determines what to analyze (document, inline text, or general question)
    and returns a natural-language response.

    Args:
        query: The user's sentiment-related question.

    Returns:
        A natural-language response describing the sentiment analysis results.
    """
    query_lower = query.lower()

    # Check if asking about a specific document
    doc_patterns = [
        r"sentiment\s+(?:of|on|for|in)\s+(?:the\s+)?(?:report|document|pdf|paper)\s+['\"]?(.+?)['\"]?$",
        r"(?:report|document|pdf|paper)\s+['\"]?(.+?)['\"]?\s+sentiment",
        r"analyze\s+(?:the\s+)?sentiment\s+(?:of|for)\s+['\"]?(.+?)['\"]?$",
    ]

    for pattern in doc_patterns:
        match = re.search(pattern, query_lower)
        if match:
            filename = match.group(1).strip()
            if not filename.endswith(".pdf"):
                filename += ".pdf"

            result = analyze_document_sentiment(filename)
            if result.get("message"):
                return result["message"]

            return _format_sentiment_response(result, source=f"document '{filename}'")

    # Check if there's inline text to analyze (e.g., "what's the sentiment of 'the market is booming'")
    text_match = re.search(r"['\"](.+?)['\"]", query)
    if text_match:
        text = text_match.group(1)
        result = analyze_sentiment(text)
        return _format_sentiment_response(result, source="the provided text")

    # Check for ingested documents and analyze them all
    from rag.pdf_ingester import get_ingested_documents
    docs = get_ingested_documents()

    if docs:
        # Analyze the first/most recent document
        latest_doc = docs[0]["filename"]
        result = analyze_document_sentiment(latest_doc)
        return _format_sentiment_response(result, source=f"document '{latest_doc}'")

    # Fallback: use LLM to handle the sentiment question
    return ask_llm(
        query,
        system_prompt=(
            "You are AURA, a finance sentiment analysis assistant. "
            "The user is asking about sentiment but no documents are loaded. "
            "Suggest they upload documents via the Documents tab for analysis, "
            "or provide text in quotes for direct sentiment analysis."
        ),
    )


def _format_sentiment_response(result: dict, source: str = "") -> str:
    """Format sentiment analysis results into a natural-language response."""
    label = result["label"].capitalize()
    score = result["score"]
    all_scores = result.get("all_scores", {})

    emoji_map = {"positive": "📈", "negative": "📉", "neutral": "➡️"}
    emoji = emoji_map.get(result["label"], "")

    response = f"{emoji} The overall sentiment of {source} is **{label}** "
    response += f"with a confidence score of {score:.1%}.\n\n"

    if all_scores:
        response += "Detailed breakdown:\n"
        for lbl, scr in sorted(all_scores.items(), key=lambda x: x[1], reverse=True):
            bar_len = int(scr * 20)
            bar = "█" * bar_len + "░" * (20 - bar_len)
            response += f"  • {lbl.capitalize()}: {bar} {scr:.1%}\n"

    if result.get("num_chunks_analyzed"):
        response += f"\nAnalyzed across {result['num_chunks_analyzed']} text segments."

    return response
