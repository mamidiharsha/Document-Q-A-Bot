"""
Text Chunking module for the RAG pipeline.

Implements a recursive character text splitter that splits documents
into overlapping chunks while preserving metadata.
"""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class Chunk:
    """Represents a text chunk with metadata for embedding and retrieval."""

    text: str
    metadata: dict = field(default_factory=dict)

    @property
    def source(self) -> str:
        return self.metadata.get("source", "unknown")

    @property
    def page(self) -> Optional[int]:
        return self.metadata.get("page")

    @property
    def chunk_index(self) -> int:
        return self.metadata.get("chunk_index", 0)

    @property
    def chunk_id(self) -> str:
        """Generate a unique ID for this chunk."""
        source = self.source.replace(" ", "_").replace(".", "_")
        page = self.page or 0
        idx = self.chunk_index
        return f"{source}_p{page}_c{idx}"


class RecursiveCharacterChunker:
    """Splits text into overlapping chunks using a hierarchy of separators.

    The chunker attempts to split text at natural boundaries (paragraphs,
    then sentences, then words) to produce chunks that are as semantically
    coherent as possible while respecting the size constraints.

    Strategy: Recursive Character Text Splitting
    - Tries separators in order: \\n\\n → \\n → . (sentence) → space
    - Applies chunk_size and chunk_overlap parameters
    - Preserves source metadata on every chunk

    Args:
        chunk_size: Maximum number of characters per chunk (default: 500).
        chunk_overlap: Number of overlapping characters between consecutive
                       chunks (default: 100).
        separators: List of separators to try, in order of preference.
    """

    def __init__(
        self,
        chunk_size: int = 500,
        chunk_overlap: int = 100,
        separators: Optional[list[str]] = None,
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.separators = separators or ["\n\n", "\n", ". ", " "]

    def chunk_documents(self, documents: list) -> list[Chunk]:
        """Split a list of documents into chunks with enriched metadata.

        Args:
            documents: List of Document objects from the document loader.

        Returns:
            List of Chunk objects with metadata including source, page,
            section, and chunk_index.
        """
        all_chunks = []

        for doc in documents:
            text_chunks = self._split_text(doc.text)

            for i, chunk_text in enumerate(text_chunks):
                # Build enriched metadata
                chunk_metadata = {
                    **doc.metadata,
                    "chunk_index": i,
                    "total_chunks": len(text_chunks),
                }

                all_chunks.append(
                    Chunk(text=chunk_text, metadata=chunk_metadata)
                )

        return all_chunks

    def _split_text(self, text: str) -> list[str]:
        """Split text recursively using the separator hierarchy.

        Args:
            text: The text to split.

        Returns:
            List of text chunks respecting size and overlap constraints.
        """
        # If text is already small enough, return it as-is
        if len(text) <= self.chunk_size:
            return [text] if text.strip() else []

        # Find the best separator to use
        chunks = self._recursive_split(text, separator_index=0)

        # Apply overlap
        return self._merge_with_overlap(chunks)

    def _recursive_split(self, text: str, separator_index: int) -> list[str]:
        """Recursively split text using separators in order of preference.

        Args:
            text: Text to split.
            separator_index: Current index in the separators list.

        Returns:
            List of text segments.
        """
        if separator_index >= len(self.separators):
            # No more separators — force split by character count
            return self._force_split(text)

        separator = self.separators[separator_index]
        splits = text.split(separator)

        # If we only got one piece, try the next separator
        if len(splits) <= 1:
            return self._recursive_split(text, separator_index + 1)

        # Re-attach the separator to maintain readability
        # (except for whitespace-only separators)
        good_splits = []
        for i, split in enumerate(splits):
            if separator.strip() and i < len(splits) - 1:
                piece = split + separator
            else:
                piece = split

            if piece.strip():
                good_splits.append(piece.strip())

        # Check if any pieces are still too large
        result = []
        for piece in good_splits:
            if len(piece) <= self.chunk_size:
                result.append(piece)
            else:
                # Recursively split the oversized piece
                sub_splits = self._recursive_split(
                    piece, separator_index + 1
                )
                result.extend(sub_splits)

        return result

    def _force_split(self, text: str) -> list[str]:
        """Force-split text by character count when no separator works.

        Args:
            text: Text to split.

        Returns:
            List of text segments of approximately chunk_size length.
        """
        chunks = []
        start = 0
        while start < len(text):
            end = min(start + self.chunk_size, len(text))

            # Try to find a space near the end to avoid splitting words
            if end < len(text):
                last_space = text.rfind(" ", start, end)
                if last_space > start + self.chunk_size // 2:
                    end = last_space

            chunk = text[start:end].strip()
            if chunk:
                chunks.append(chunk)
            start = end

        return chunks

    def _merge_with_overlap(self, segments: list[str]) -> list[str]:
        """Merge small segments and apply overlap between chunks.

        Args:
            segments: List of text segments from recursive splitting.

        Returns:
            List of final chunks with appropriate overlap.
        """
        if not segments:
            return []

        # First, merge segments that are too small
        merged = []
        current = ""

        for segment in segments:
            if not current:
                current = segment
            elif len(current) + len(segment) + 1 <= self.chunk_size:
                current = current + " " + segment
            else:
                merged.append(current)
                current = segment

        if current:
            merged.append(current)

        # Now apply overlap
        if len(merged) <= 1 or self.chunk_overlap <= 0:
            return merged

        result = [merged[0]]
        for i in range(1, len(merged)):
            prev_chunk = merged[i - 1]
            curr_chunk = merged[i]

            # Take the last `chunk_overlap` characters from the previous chunk
            overlap_text = prev_chunk[-self.chunk_overlap:]

            # Find a clean break point in the overlap
            space_pos = overlap_text.find(" ")
            if space_pos > 0:
                overlap_text = overlap_text[space_pos + 1:]

            # Prepend overlap to current chunk (if it fits)
            combined = overlap_text + " " + curr_chunk
            if len(combined) <= self.chunk_size * 1.2:
                result.append(combined)
            else:
                result.append(curr_chunk)

        return result
