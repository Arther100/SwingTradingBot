"""
SwingAdvisorBot — Module 5: Memory & Personalization
embeddings/embedding_engine.py — SentenceTransformer wrapper with caching

Converts text to 384-dimensional vectors using all-MiniLM-L6-v2.
Model loaded once (lazy singleton) — first call downloads if needed.

Features:
  - Lazy model loading (only when first embed call happens)
  - Single text and batch embedding
  - LRU cache for repeated texts (avoids re-encoding)
  - Graceful fallback: returns zero vector if model fails to load
  - Normalized embeddings (unit length for cosine similarity)

Usage:
    engine = EmbeddingEngine()
    vec = engine.embed("Bought HDFCBANK at 769.55")
    vecs = engine.embed_batch(["text1", "text2", "text3"])
"""

from __future__ import annotations

import hashlib
import logging
from functools import lru_cache
from typing import TYPE_CHECKING

from module5_memory.config import (
    EMBEDDING_BATCH_SIZE,
    EMBEDDING_DIMENSIONS,
    EMBEDDING_MODEL_NAME,
    EMBEDDING_NORMALIZE,
)

if TYPE_CHECKING:
    from sentence_transformers import SentenceTransformer

logger = logging.getLogger("swing_advisor.m5_embeddings")


# ─────────────────────────────────────────────────────────────
# Module-level model singleton
# ─────────────────────────────────────────────────────────────

_model: SentenceTransformer | None = None
_model_failed: bool = False


def _get_model() -> SentenceTransformer | None:
    """Lazy-load the SentenceTransformer model.

    Returns None if sentence-transformers is not installed
    or model fails to load. Sets _model_failed flag to avoid
    repeated attempts.
    """
    global _model, _model_failed

    if _model is not None:
        return _model

    if _model_failed:
        return None

    try:
        from sentence_transformers import SentenceTransformer

        _model = SentenceTransformer(EMBEDDING_MODEL_NAME)
        logger.info(f"[Embeddings] Loaded model: {EMBEDDING_MODEL_NAME}")
        return _model
    except Exception as e:
        _model_failed = True
        logger.warning(f"[Embeddings] Failed to load model: {e}. Using zero vectors.")
        return None


def _zero_vector() -> list[float]:
    """Return a zero vector of correct dimensions (fallback)."""
    return [0.0] * EMBEDDING_DIMENSIONS


# ─────────────────────────────────────────────────────────────
# Cache key helper
# ─────────────────────────────────────────────────────────────


def _cache_key(text: str) -> str:
    """Create a stable hash for cache lookups."""
    return hashlib.md5(text.encode("utf-8")).hexdigest()


# ─────────────────────────────────────────────────────────────
# EmbeddingEngine
# ─────────────────────────────────────────────────────────────


class EmbeddingEngine:
    """Wrapper around SentenceTransformer for text embeddings.

    Usage:
        engine = EmbeddingEngine()
        vec = engine.embed("some text")             # list[float], len=384
        vecs = engine.embed_batch(["t1", "t2"])     # list[list[float]]
    """

    def __init__(self) -> None:
        self._cache: dict[str, list[float]] = {}

    @property
    def dimensions(self) -> int:
        """Embedding vector dimensions."""
        return EMBEDDING_DIMENSIONS

    @property
    def model_name(self) -> str:
        """Model identifier."""
        return EMBEDDING_MODEL_NAME

    @property
    def is_available(self) -> bool:
        """Check if embedding model is available."""
        return _get_model() is not None

    def embed(self, text: str) -> list[float]:
        """Embed a single text string.

        Args:
            text: Input text to embed.

        Returns:
            384-dimensional float vector.
            Returns zero vector if model unavailable.
        """
        if not text or not text.strip():
            return _zero_vector()

        # Check cache
        key = _cache_key(text)
        if key in self._cache:
            return self._cache[key]

        model = _get_model()
        if model is None:
            return _zero_vector()

        try:
            embedding = model.encode(
                text,
                normalize_embeddings=EMBEDDING_NORMALIZE,
                show_progress_bar=False,
            )
            vec = embedding.tolist()
            self._cache[key] = vec
            return vec
        except Exception as e:
            logger.error(f"[Embeddings] Encode failed: {e}")
            return _zero_vector()

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Embed multiple texts in a single batch.

        Checks cache first, only encodes uncached texts.

        Args:
            texts: List of input texts.

        Returns:
            List of 384-dimensional float vectors (same order as input).
        """
        if not texts:
            return []

        results: list[list[float] | None] = [None] * len(texts)
        uncached_indices: list[int] = []
        uncached_texts: list[str] = []

        # Check cache for each text
        for i, text in enumerate(texts):
            if not text or not text.strip():
                results[i] = _zero_vector()
                continue

            key = _cache_key(text)
            if key in self._cache:
                results[i] = self._cache[key]
            else:
                uncached_indices.append(i)
                uncached_texts.append(text)

        # Encode uncached texts
        if uncached_texts:
            model = _get_model()
            if model is None:
                for i in uncached_indices:
                    results[i] = _zero_vector()
            else:
                try:
                    embeddings = model.encode(
                        uncached_texts,
                        normalize_embeddings=EMBEDDING_NORMALIZE,
                        batch_size=EMBEDDING_BATCH_SIZE,
                        show_progress_bar=False,
                    )
                    for idx, emb in zip(uncached_indices, embeddings):
                        vec = emb.tolist()
                        self._cache[_cache_key(texts[idx])] = vec
                        results[idx] = vec
                except Exception as e:
                    logger.error(f"[Embeddings] Batch encode failed: {e}")
                    for i in uncached_indices:
                        results[i] = _zero_vector()

        # Fill any remaining None slots
        return [r if r is not None else _zero_vector() for r in results]

    def clear_cache(self) -> None:
        """Clear the embedding cache."""
        self._cache.clear()

    @property
    def cache_size(self) -> int:
        """Number of cached embeddings."""
        return len(self._cache)


# ─────────────────────────────────────────────────────────────
# Module-level singleton
# ─────────────────────────────────────────────────────────────


@lru_cache(maxsize=1)
def get_embedding_engine() -> EmbeddingEngine:
    """Get the global EmbeddingEngine singleton."""
    return EmbeddingEngine()
