"""
Retriever module for the RAG pipeline.

Combines the embedder and vector store to perform similarity-based
retrieval of relevant document chunks for a given query.
"""

from src.embedder import GeminiEmbedder
from src.vector_store import ChromaVectorStore, SearchResult
from src.config import TOP_K


class Retriever:
    """Retrieves the most relevant document chunks for a user query.

    Encapsulates the embed → search → rank pipeline for queries.

    Args:
        embedder: GeminiEmbedder instance for query embedding.
        vector_store: ChromaVectorStore instance for similarity search.
        top_k: Default number of results to return.
    """

    def __init__(
        self,
        embedder: GeminiEmbedder,
        vector_store: ChromaVectorStore,
        top_k: int = TOP_K,
    ):
        self.embedder = embedder
        self.vector_store = vector_store
        self.top_k = top_k

    def retrieve(
        self, query: str, top_k: int | None = None
    ) -> list[SearchResult]:
        """Retrieve the most relevant chunks for a query.

        Args:
            query: The user's natural language question.
            top_k: Number of results to return. Uses default if None.

        Returns:
            List of SearchResult objects sorted by relevance,
            each containing the chunk text, metadata, and score.
        """
        k = top_k or self.top_k

        # Step 1: Embed the query
        query_embedding = self.embedder.embed_query(query)

        # Step 2: Search the vector store
        results = self.vector_store.query(
            query_embedding=query_embedding,
            top_k=k,
        )

        return results

    def retrieve_with_context(
        self, query: str, top_k: int | None = None
    ) -> dict:
        """Retrieve chunks and format them as context for the LLM.

        Args:
            query: The user's natural language question.
            top_k: Number of results to return.

        Returns:
            Dictionary with:
                - 'results': List of SearchResult objects
                - 'context': Formatted context string for the LLM
                - 'sources': List of unique source references
        """
        results = self.retrieve(query, top_k)

        # Format context for LLM
        context_parts = []
        sources = []

        for i, result in enumerate(results, 1):
            source_ref = result.source
            if result.page:
                source_ref += f", Page/Section {result.page}"
            if result.section:
                source_ref += f" ({result.section})"

            context_parts.append(
                f"[Source {i}: {source_ref}]\n{result.text}"
            )
            sources.append(
                {
                    "index": i,
                    "source": result.source,
                    "page": result.page,
                    "section": result.section,
                    "score": result.score,
                    "excerpt": result.text[:200] + "..."
                    if len(result.text) > 200
                    else result.text,
                }
            )

        context = "\n\n---\n\n".join(context_parts)

        return {
            "results": results,
            "context": context,
            "sources": sources,
        }
