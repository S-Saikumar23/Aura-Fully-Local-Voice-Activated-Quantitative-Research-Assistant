"""
AURA — RAG Question Answering Handler.

Combines semantic retrieval with Ollama LLM to answer questions
grounded in the content of ingested research PDF documents.
"""

from rag.retriever import retrieve, retrieve_formatted
from core.llm import ask_llm_with_context, ask_llm


def answer_research_question(query: str) -> str:
    """
    Answer a question using the RAG pipeline.

    1. Retrieve relevant document chunks via semantic search.
    2. Construct a context-augmented prompt.
    3. Send to Ollama LLM for grounded answer generation.

    Args:
        query: The user's natural-language research question.

    Returns:
        A natural-language answer grounded in the retrieved documents,
        or a fallback message if no relevant documents are found.
    """
    # Step 1: Retrieve relevant context
    context = retrieve_formatted(query, top_k=5)

    if not context:
        # No relevant documents found — fall back to general LLM
        return ask_llm(
            f"The user asked: '{query}'\n\n"
            "I don't have any research documents ingested that are relevant to this question. "
            "Please let the user know that no relevant research documents were found, "
            "and suggest they upload relevant PDFs through the Documents tab.",
            system_prompt=(
                "You are AURA, a quantitative research assistant. "
                "Be helpful and suggest next steps when documents are not available."
            ),
        )

    # Step 2: Generate answer with context
    print(f"[RAG QA] Generating answer with {len(context)} chars of context...")

    system_prompt = (
        "You are AURA, a quantitative research AI assistant. "
        "Answer the user's question based ONLY on the provided research document context. "
        "Be precise and cite the source document filename when possible. "
        "If the context doesn't fully answer the question, say what you can determine "
        "from the available documents and note what information is missing."
    )

    answer = ask_llm_with_context(
        query=query,
        context=context,
        system_prompt=system_prompt,
    )

    return answer


def get_document_summary(filename: str) -> str:
    """
    Generate a brief summary of a specific ingested document.

    Args:
        filename: Name of the ingested PDF file.

    Returns:
        A summary of the document's content.
    """
    # Retrieve all chunks from this specific document
    from finance.database import get_session
    from finance.models import ResearchDocument

    with get_session() as session:
        chunks = (
            session.query(ResearchDocument.content)
            .filter(ResearchDocument.filename == filename)
            .order_by(ResearchDocument.id)
            .limit(10)  # First 10 chunks for summary
            .all()
        )

    if not chunks:
        return f"No document found with filename '{filename}'."

    # Combine first chunks as context
    combined_text = "\n\n".join(c.content for c in chunks if c.content)

    return ask_llm(
        f"Please provide a concise summary (3-5 sentences) of the following "
        f"research document '{filename}':\n\n{combined_text}",
        system_prompt=(
            "You are a precise research assistant. Summarize documents concisely, "
            "focusing on key findings, methodology, and conclusions."
        ),
    )
