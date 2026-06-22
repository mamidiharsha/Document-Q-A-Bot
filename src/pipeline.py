"""
Pipeline module — End-to-end RAG orchestration.

Provides a single RAGPipeline class that ties together all components:
document loading, chunking, embedding, vector storage, retrieval, and
answer generation. Handles both the indexing and querying workflows.
"""

from src.config import (
    DATA_DIR,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
    TOP_K,
    validate_config,
)
from src.document_loader import load_documents
from src.chunker import RecursiveCharacterChunker
from src.embedder import GeminiEmbedder
from src.vector_store import ChromaVectorStore
from src.retriever import Retriever
from src.generator import AnswerGenerator, GeneratedAnswer


class RAGPipeline:
    """End-to-end RAG pipeline orchestrator.

    Manages the complete flow from document ingestion through to
    answer generation. Provides clear separation between the
    indexing step (one-time data preparation) and the querying step
    (runtime question answering).

    Usage:
        pipeline = RAGPipeline()

        # One-time: Index documents
        pipeline.index()

        # Runtime: Ask questions
        answer = pipeline.query("What is the boiling point of coffee?")
    """

    def __init__(self):
        """Initialize all pipeline components."""
        validate_config()

        self.chunker = RecursiveCharacterChunker(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
        )
        self.embedder = GeminiEmbedder()
        self.vector_store = ChromaVectorStore()
        self.retriever = Retriever(
            embedder=self.embedder,
            vector_store=self.vector_store,
            top_k=TOP_K,
        )
        self.generator = AnswerGenerator()

    # ------------------------------------------------------------------
    # Indexing Workflow
    # ------------------------------------------------------------------

    def index(
        self,
        data_dir: str = DATA_DIR,
        force: bool = False,
        progress_callback=None,
    ) -> dict:
        """Run the full indexing pipeline: load → chunk → embed → store.

        Args:
            data_dir: Directory containing documents to index.
            force: If True, clear existing index and re-index.
            progress_callback: Optional callback(stage, message) for
                               progress reporting.

        Returns:
            Dictionary with indexing statistics.
        """

        def _report(stage: str, message: str):
            if progress_callback:
                progress_callback(stage, message)

        # Check if already indexed
        if self.vector_store.is_indexed() and not force:
            count = self.vector_store.count()
            sources = self.vector_store.get_sources()
            _report("skip", f"Already indexed {count} chunks from {len(sources)} documents")
            return {
                "status": "skipped",
                "chunks": count,
                "documents": len(sources),
                "sources": sources,
            }

        # Clear existing index if forcing re-index
        if force and self.vector_store.is_indexed():
            _report("clear", "Clearing existing index...")
            self.vector_store.clear()

        # Step 1: Load documents
        _report("load", f"Loading documents from {data_dir}...")
        documents = load_documents(data_dir)
        if not documents:
            raise RuntimeError(f"No documents found in {data_dir}")
        _report("load", f"Loaded {len(documents)} document sections")

        # Step 2: Chunk documents
        _report("chunk", "Splitting documents into chunks...")
        chunks = self.chunker.chunk_documents(documents)
        _report("chunk", f"Created {len(chunks)} chunks")

        # Step 3: Embed chunks (batched)
        _report("embed", "Generating embeddings (batched)...")

        def embed_progress(batch_num, total):
            _report("embed", f"Embedding batch {batch_num}/{total}...")

        embeddings = self.embedder.embed_chunks(
            chunks, progress_callback=embed_progress
        )
        _report("embed", f"Generated {len(embeddings)} embeddings")

        # Step 4: Store in vector database
        _report("store", "Storing in ChromaDB...")

        def store_progress(batch_num, total):
            _report("store", f"Storing batch {batch_num}/{total}...")

        indexed_count = self.vector_store.index_chunks(
            chunks, embeddings, progress_callback=store_progress
        )
        sources = self.vector_store.get_sources()
        _report("store", f"Indexed {indexed_count} chunks from {len(sources)} documents")

        return {
            "status": "completed",
            "chunks": indexed_count,
            "documents": len(sources),
            "sources": sources,
            "document_sections": len(documents),
        }

    # ------------------------------------------------------------------
    # Query Workflow
    # ------------------------------------------------------------------

    def query(self, question: str, top_k: int | None = None) -> GeneratedAnswer:
        """Run the full query pipeline: retrieve → generate.

        Args:
            question: The user's natural language question.
            top_k: Number of chunks to retrieve. Uses default if None.

        Returns:
            GeneratedAnswer with the answer text and source citations.
        """
        if not self.vector_store.is_indexed():
            raise RuntimeError(
                "No documents have been indexed yet. "
                "Run the indexing step first: python index.py"
            )

        # Step 1: Retrieve relevant chunks
        retrieval = self.retriever.retrieve_with_context(question, top_k)

        # Step 2: Generate grounded answer
        answer = self.generator.generate(
            query=question,
            context=retrieval["context"],
            sources=retrieval["sources"],
        )

        return answer

    # ------------------------------------------------------------------
    # Utility Methods
    # ------------------------------------------------------------------

    def get_stats(self) -> dict:
        """Get current pipeline statistics.

        Returns:
            Dictionary with indexed chunks count and source documents.
        """
        return {
            "indexed_chunks": self.vector_store.count(),
            "sources": self.vector_store.get_sources(),
            "is_indexed": self.vector_store.is_indexed(),
        }
