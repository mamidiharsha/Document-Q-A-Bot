"""
Vector Store module for the RAG pipeline.

Provides a wrapper around ChromaDB with persistent storage,
clear separation between indexing and querying operations.
"""

import os
from dataclasses import dataclass, field
from typing import Optional

import chromadb

from src.config import CHROMA_DB_DIR, COLLECTION_NAME


@dataclass
class SearchResult:
    """Represents a single search result from the vector store."""

    text: str
    metadata: dict = field(default_factory=dict)
    distance: float = 0.0

    @property
    def score(self) -> float:
        """Convert distance to a similarity score (0 to 1)."""
        # ChromaDB uses L2 distance by default; lower is better
        return max(0.0, 1.0 - self.distance / 2.0)

    @property
    def source(self) -> str:
        return self.metadata.get("source", "unknown")

    @property
    def page(self) -> Optional[int]:
        page = self.metadata.get("page")
        return int(page) if page is not None else None

    @property
    def section(self) -> Optional[str]:
        return self.metadata.get("section")


class ChromaVectorStore:
    """Persistent vector store using ChromaDB.

    Stores document chunk embeddings on disk so re-indexing is not
    required on every run. Provides clear separation between
    indexing (adding data) and querying (searching data).

    Args:
        persist_dir: Directory for persistent storage.
                     Defaults to config value.
        collection_name: Name of the ChromaDB collection.
                         Defaults to config value.
    """

    def __init__(
        self,
        persist_dir: str = CHROMA_DB_DIR,
        collection_name: str = COLLECTION_NAME,
    ):
        self.persist_dir = persist_dir
        self.collection_name = collection_name

        # Initialize persistent client
        os.makedirs(persist_dir, exist_ok=True)
        self.client = chromadb.PersistentClient(path=persist_dir)
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    # ------------------------------------------------------------------
    # Indexing Operations
    # ------------------------------------------------------------------

    def index_chunks(
        self,
        chunks: list,
        embeddings: list[list[float]],
        batch_size: int = 500,
        progress_callback=None,
    ) -> int:
        """Add document chunks and their embeddings to the vector store.

        Args:
            chunks: List of Chunk objects.
            embeddings: Corresponding embedding vectors.
            batch_size: Number of items to add per batch.
            progress_callback: Optional callback(batch_num, total_batches).

        Returns:
            Number of chunks successfully indexed.
        """
        if len(chunks) != len(embeddings):
            raise ValueError(
                f"Mismatch: {len(chunks)} chunks but {len(embeddings)} embeddings"
            )

        total_batches = (len(chunks) + batch_size - 1) // batch_size
        indexed_count = 0

        for batch_num in range(total_batches):
            start = batch_num * batch_size
            end = min(start + batch_size, len(chunks))

            if progress_callback:
                progress_callback(batch_num + 1, total_batches)

            batch_ids = []
            batch_embeddings = []
            batch_documents = []
            batch_metadatas = []

            for i in range(start, end):
                chunk = chunks[i]
                batch_ids.append(chunk.chunk_id)
                batch_embeddings.append(embeddings[i])
                batch_documents.append(chunk.text)

                # ChromaDB metadata must be flat (str, int, float, bool)
                flat_meta = {}
                for k, v in chunk.metadata.items():
                    if isinstance(v, (str, int, float, bool)):
                        flat_meta[k] = v
                    else:
                        flat_meta[k] = str(v)
                batch_metadatas.append(flat_meta)

            self.collection.add(
                ids=batch_ids,
                embeddings=batch_embeddings,
                documents=batch_documents,
                metadatas=batch_metadatas,
            )
            indexed_count += len(batch_ids)

        return indexed_count

    # ------------------------------------------------------------------
    # Query Operations
    # ------------------------------------------------------------------

    def query(
        self,
        query_embedding: list[float],
        top_k: int = 5,
    ) -> list[SearchResult]:
        """Search for the most similar chunks to a query embedding.

        Args:
            query_embedding: The embedding vector of the user's query.
            top_k: Number of results to return.

        Returns:
            List of SearchResult objects sorted by relevance.
        """
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=min(top_k, self.collection.count()),
            include=["documents", "metadatas", "distances"],
        )

        search_results = []

        if results and results["documents"]:
            for i in range(len(results["documents"][0])):
                search_results.append(
                    SearchResult(
                        text=results["documents"][0][i],
                        metadata=results["metadatas"][0][i] if results["metadatas"] else {},
                        distance=results["distances"][0][i] if results["distances"] else 0.0,
                    )
                )

        return search_results

    # ------------------------------------------------------------------
    # Utility Methods
    # ------------------------------------------------------------------

    def is_indexed(self) -> bool:
        """Check if the collection has any indexed documents.

        Returns:
            True if documents exist in the collection.
        """
        return self.collection.count() > 0

    def count(self) -> int:
        """Get the number of indexed chunks.

        Returns:
            Number of chunks in the collection.
        """
        return self.collection.count()

    def clear(self) -> None:
        """Delete and recreate the collection (reset all data)."""
        self.client.delete_collection(self.collection_name)
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    def get_sources(self) -> list[str]:
        """Get a list of unique source filenames in the collection.

        Returns:
            Sorted list of unique source filenames.
        """
        if not self.is_indexed():
            return []

        # Get all metadata
        all_data = self.collection.get(include=["metadatas"])
        sources = set()
        for meta in all_data["metadatas"]:
            if "source" in meta:
                sources.add(meta["source"])

        return sorted(sources)
