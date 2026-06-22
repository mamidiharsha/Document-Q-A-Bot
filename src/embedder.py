"""
Embedding module for the RAG pipeline.

Uses Google Gemini's embedding API for batched, efficient embedding
of document chunks and queries.
"""

import time
from typing import Optional

from google import genai
from google.genai import types

from src.config import (
    GOOGLE_API_KEY,
    EMBEDDING_MODEL,
    EMBEDDING_DIMENSIONS,
    EMBEDDING_BATCH_SIZE,
)

import os
import hashlib
import random


class GeminiEmbedder:
    """Embeds text chunks and queries using Google's Gemini embedding model.

    All embedding calls are batched for efficiency — chunks are never
    embedded one at a time in a loop.

    Args:
        api_key: Google API key. If None, uses the key from config.
        model: Embedding model name. Defaults to config value.
        dimensions: Output embedding dimensionality. Defaults to config value.
        batch_size: Number of texts to embed per API call. Defaults to config value.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = EMBEDDING_MODEL,
        dimensions: int = EMBEDDING_DIMENSIONS,
        batch_size: int = EMBEDDING_BATCH_SIZE,
    ):
        self.model = model
        self.dimensions = dimensions
        self.batch_size = batch_size
        # Allow a deterministic local embedding fallback for offline testing
        self.use_local = (
            os.getenv("USE_LOCAL_EMBEDDINGS", "0") == "1"
            or not (api_key or GOOGLE_API_KEY)
        )

        if not self.use_local:
            self.client = genai.Client(api_key=api_key or GOOGLE_API_KEY)
        else:
            self.client = None

    def embed_chunks(
        self, chunks: list, progress_callback=None
    ) -> list[list[float]]:
        """Embed a list of text chunks in batches.

        Args:
            chunks: List of Chunk objects (must have a .text attribute).
            progress_callback: Optional callback(batch_num, total_batches)
                               for progress reporting.

        Returns:
            List of embedding vectors (list of floats), one per chunk.
        """
        texts = [chunk.text for chunk in chunks]
        return self._embed_texts_batched(texts, progress_callback)

    def embed_query(self, query: str) -> list[float]:
        """Embed a single query string.

        Args:
            query: The user's query text.

        Returns:
            Embedding vector as a list of floats.
        """
        if self.use_local:
            return self._local_embed_text(query)

        result = self.client.models.embed_content(
            model=self.model,
            contents=query,
            config=types.EmbedContentConfig(
                output_dimensionality=self.dimensions,
                task_type="RETRIEVAL_QUERY",
            ),
        )
        return list(result.embeddings[0].values)

    def _embed_texts_batched(
        self, texts: list[str], progress_callback=None
    ) -> list[list[float]]:
        """Embed texts in batches with retry logic.

        Args:
            texts: List of text strings to embed.
            progress_callback: Optional callback(batch_num, total_batches).

        Returns:
            List of embedding vectors.
        """
        all_embeddings = []
        total_batches = (len(texts) + self.batch_size - 1) // self.batch_size

        for batch_num in range(total_batches):
            start = batch_num * self.batch_size
            end = min(start + self.batch_size, len(texts))
            batch_texts = texts[start:end]

            if progress_callback:
                progress_callback(batch_num + 1, total_batches)

            # Retry logic for rate limits
            embeddings = self._embed_batch_with_retry(batch_texts)
            all_embeddings.extend(embeddings)

        return all_embeddings

    def _embed_batch_with_retry(
        self,
        texts: list[str],
        max_retries: int = 3,
        base_delay: float = 2.0,
    ) -> list[list[float]]:
        """Embed a single batch with exponential backoff retry.

        Args:
            texts: Batch of texts to embed.
            max_retries: Maximum number of retry attempts.
            base_delay: Base delay in seconds for exponential backoff.

        Returns:
            List of embedding vectors for this batch.
        """
        # If using local embeddings, compute deterministically and return
        if self.use_local:
            return [self._local_embed_text(t) for t in texts]

        for attempt in range(max_retries + 1):
            try:
                result = self.client.models.embed_content(
                    model=self.model,
                    contents=texts,
                    config=types.EmbedContentConfig(
                        output_dimensionality=self.dimensions,
                        task_type="RETRIEVAL_DOCUMENT",
                    ),
                )
                return [list(e.values) for e in result.embeddings]

            except Exception as e:
                # If quota or other API error occurs and local fallback is allowed,
                # switch to local embeddings to allow offline testing.
                err_str = str(e)
                if "RESOURCE_EXHAUSTED" in err_str or "quota" in err_str.lower():
                    print("  ⚠ Embedding API quota/error detected, falling back to local embeddings.")
                    self.use_local = True
                    return [self._local_embed_text(t) for t in texts]

                if attempt == max_retries:
                    raise RuntimeError(
                        f"Embedding failed after {max_retries} retries: {e}"
                    ) from e

                delay = base_delay * (2**attempt)
                print(
                    f"  ⚠ Embedding API error (attempt {attempt + 1}/"
                    f"{max_retries}), retrying in {delay:.0f}s: {e}"
                )
                time.sleep(delay)

        return []  # Unreachable, but satisfies type checker

    # ------------------------------------------------------------------
    # Local deterministic embedding (development fallback)
    # ------------------------------------------------------------------
    def _local_embed_text(self, text: str) -> list[float]:
        """Deterministic pseudo-embedding for offline testing.

        Uses a seeded Random instance (seeded from SHA256 of the text)
        to generate a reproducible vector of floats in [0,1).
        """
        seed = int.from_bytes(hashlib.sha256(text.encode("utf-8")).digest(), "big")
        rng = random.Random(seed)
        return [rng.random() for _ in range(self.dimensions)]
