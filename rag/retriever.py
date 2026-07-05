"""
AURA — Semantic Retriever for Research Documents.

Performs cosine similarity search over document embeddings stored in
PostgreSQL with pgvector. Returns the most relevant document chunks
for a given query.
"""

from sqlalchemy import text as sql_text

from finance.database import get_session
from finance.models import ResearchDocument
from rag.pdf_ingester import _get_embed_model


def retrieve(
    query: str,
    top_k: int = 5,
    similarity_threshold: float = 0.3,
) -> list[dict]:
    """
    Retrieve the most relevant document chunks for a query.

    Uses cosine similarity via pgvector's `<=>` operator.

    Args:
        query: The user's natural-language question.
        top_k: Maximum number of results to return.
        similarity_threshold: Minimum similarity score (0-1).

    Returns:
        List of dicts with 'filename', 'content', 'similarity' keys,
        ordered by relevance (highest similarity first).
    """
    # Generate query embedding
    model = _get_embed_model()
    query_embedding = model.encode([query], normalize_embeddings=True)[0].tolist()

    # Convert embedding to PostgreSQL vector literal
    embedding_str = "[" + ",".join(str(x) for x in query_embedding) + "]"

    with get_session() as session:
        # pgvector cosine distance: <=> returns distance, so similarity = 1 - distance
        # NOTE: We must escape '::' as '\:\:' in SQLAlchemy text() to avoid
        # it being interpreted as a named parameter.
        results = session.execute(
            sql_text(
                """
                SELECT
                    id,
                    filename,
                    content,
                    1 - (embedding <=> cast(:embedding AS vector)) AS similarity
                FROM research_documents
                WHERE embedding IS NOT NULL
                ORDER BY embedding <=> cast(:embedding AS vector)
                LIMIT :top_k
                """
            ),
            {"embedding": embedding_str, "top_k": top_k},
        ).fetchall()

    # Filter by threshold and format results
    retrieved = []
    for row in results:
        sim = float(row.similarity) if row.similarity else 0.0
        if sim >= similarity_threshold:
            retrieved.append({
                "id": row.id,
                "filename": row.filename,
                "content": row.content,
                "similarity": round(sim, 4),
            })

    print(f"[RAG] Retrieved {len(retrieved)} chunks (query: '{query[:50]}...')")
    return retrieved


def retrieve_formatted(query: str, top_k: int = 5) -> str:
    """
    Retrieve relevant chunks and format them as a single context string.

    Convenience method for direct use in RAG prompts.

    Args:
        query: The user's question.
        top_k: Maximum number of chunks to include.

    Returns:
        Formatted context string with source annotations.
    """
    results = retrieve(query, top_k=top_k)

    if not results:
        return ""

    context_parts = []
    for i, r in enumerate(results, 1):
        context_parts.append(
            f"[Source: {r['filename']}, Relevance: {r['similarity']}]\n"
            f"{r['content']}"
        )

    return "\n\n---\n\n".join(context_parts)
