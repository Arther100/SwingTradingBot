"""
SwingAdvisorBot — Module 5: Memory & Personalization
vector_store/pinecone_memory.py — Pinecone upsert and query operations

Manages the Pinecone vector index for semantic memory.
All operations are namespace-scoped (trade_memory, market_patterns, etc.).

Features:
  - Lazy Pinecone client initialization
  - Upsert vectors with metadata
  - Query by embedding vector (cosine similarity)
  - Delete by ID or metadata filter
  - Graceful degradation: returns empty results if Pinecone unavailable

Usage:
    store = PineconeMemory()
    store.upsert(chunks, embeddings)
    results = store.query(query_vector, namespace="trade_memory", top_k=3)
"""

from __future__ import annotations

import logging
from typing import Any

from module5_memory.config import (
    PINECONE_API_KEY,
    PINECONE_INDEX_HOST,
)
from module5_memory.embeddings.chunker import Chunk
from module5_memory.models import RetrievedChunk

logger = logging.getLogger("swing_advisor.m5_pinecone")


# ─────────────────────────────────────────────────────────────
# Pinecone Client Singleton
# ─────────────────────────────────────────────────────────────

_index: Any = None
_init_failed: bool = False


def _get_index() -> Any:
    """Lazy-initialize Pinecone index.

    Returns None if:
      - pinecone-client not installed
      - API key / host not configured
      - Connection fails

    Sets _init_failed to avoid repeated attempts.
    """
    global _index, _init_failed

    if _index is not None:
        return _index

    if _init_failed:
        return None

    if not PINECONE_API_KEY or not PINECONE_INDEX_HOST:
        logger.warning("[Pinecone] API key or index host not configured. Skipping.")
        _init_failed = True
        return None

    try:
        from pinecone import Pinecone

        pc = Pinecone(api_key=PINECONE_API_KEY)
        _index = pc.Index(host=PINECONE_INDEX_HOST)
        logger.info(f"[Pinecone] Connected to index at {PINECONE_INDEX_HOST}")
        return _index
    except Exception as e:
        _init_failed = True
        logger.warning(f"[Pinecone] Failed to connect: {e}. Semantic search disabled.")
        return None


# ─────────────────────────────────────────────────────────────
# PineconeMemory
# ─────────────────────────────────────────────────────────────


class PineconeMemory:
    """Manages Pinecone vector operations for SwingAdvisorBot.

    All operations gracefully degrade to no-ops if Pinecone
    is unavailable (returns empty lists, logs warnings).

    Usage:
        store = PineconeMemory()
        if store.is_available:
            store.upsert_chunks(chunks, embeddings)
            results = store.query(vector, "trade_memory", top_k=3)
    """

    @property
    def is_available(self) -> bool:
        """Check if Pinecone index is connected."""
        return _get_index() is not None

    # ═══════════════════════════════════════════════════════
    # UPSERT
    # ═══════════════════════════════════════════════════════

    def upsert_chunks(
        self,
        chunks: list[Chunk],
        embeddings: list[list[float]],
    ) -> int:
        """Upsert chunks with their embeddings into Pinecone.

        Groups by namespace and upserts in batches.

        Args:
            chunks: List of Chunk objects (text + namespace + metadata).
            embeddings: Corresponding embedding vectors (same order/length).

        Returns:
            Number of vectors upserted (0 if Pinecone unavailable).
        """
        index = _get_index()
        if index is None:
            logger.debug("[Pinecone] Skipping upsert — not available.")
            return 0

        if len(chunks) != len(embeddings):
            logger.error("[Pinecone] chunks/embeddings length mismatch.")
            return 0

        # Group by namespace
        ns_groups: dict[str, list[tuple[str, list[float], dict]]] = {}
        for chunk, emb in zip(chunks, embeddings):
            vec_id = self._make_id(chunk)
            metadata = {**chunk.metadata, "_text": chunk.text}
            ns_groups.setdefault(chunk.namespace, []).append((vec_id, emb, metadata))

        total = 0
        for namespace, vectors in ns_groups.items():
            try:
                # Pinecone upsert expects list of (id, values, metadata) tuples
                batch = [(vid, vals, meta) for vid, vals, meta in vectors]
                index.upsert(vectors=batch, namespace=namespace)
                total += len(batch)
                logger.debug(f"[Pinecone] Upserted {len(batch)} vectors to '{namespace}'")
            except Exception as e:
                logger.error(f"[Pinecone] Upsert failed for '{namespace}': {e}")

        return total

    def upsert_single(
        self,
        vector_id: str,
        embedding: list[float],
        metadata: dict,
        namespace: str,
        text: str = "",
    ) -> bool:
        """Upsert a single vector.

        Args:
            vector_id: Unique vector ID.
            embedding: 384-dim float vector.
            metadata: Metadata dict.
            namespace: Target namespace.
            text: Original text (stored in _text metadata).

        Returns:
            True if successful, False otherwise.
        """
        index = _get_index()
        if index is None:
            return False

        try:
            meta = {**metadata, "_text": text}
            index.upsert(
                vectors=[(vector_id, embedding, meta)],
                namespace=namespace,
            )
            return True
        except Exception as e:
            logger.error(f"[Pinecone] Single upsert failed: {e}")
            return False

    # ═══════════════════════════════════════════════════════
    # QUERY
    # ═══════════════════════════════════════════════════════

    def query(
        self,
        vector: list[float],
        namespace: str,
        top_k: int = 3,
        min_score: float = 0.0,
        filter_dict: dict | None = None,
    ) -> list[RetrievedChunk]:
        """Query Pinecone for similar vectors.

        Args:
            vector: Query embedding vector.
            namespace: Namespace to search in.
            top_k: Max results to return.
            min_score: Minimum cosine similarity score.
            filter_dict: Optional metadata filter.

        Returns:
            List of RetrievedChunk sorted by score descending.
            Empty list if Pinecone unavailable.
        """
        index = _get_index()
        if index is None:
            return []

        try:
            query_params: dict[str, Any] = {
                "vector": vector,
                "top_k": top_k,
                "namespace": namespace,
                "include_metadata": True,
            }
            if filter_dict:
                query_params["filter"] = filter_dict

            response = index.query(**query_params)
            matches = response.get("matches", [])

            results: list[RetrievedChunk] = []
            for match in matches:
                score = match.get("score", 0.0)
                if score < min_score:
                    continue

                metadata = match.get("metadata", {})
                text = metadata.pop("_text", "")

                results.append(
                    RetrievedChunk(
                        content=text,
                        score=score,
                        metadata=metadata,
                        namespace=namespace,
                    )
                )

            return sorted(results, key=lambda c: c.score, reverse=True)

        except Exception as e:
            logger.error(f"[Pinecone] Query failed for '{namespace}': {e}")
            return []

    # ═══════════════════════════════════════════════════════
    # DELETE
    # ═══════════════════════════════════════════════════════

    def delete_by_ids(
        self,
        ids: list[str],
        namespace: str,
    ) -> bool:
        """Delete vectors by their IDs.

        Args:
            ids: List of vector IDs to delete.
            namespace: Namespace to delete from.

        Returns:
            True if successful, False otherwise.
        """
        index = _get_index()
        if index is None:
            return False

        try:
            index.delete(ids=ids, namespace=namespace)
            logger.debug(f"[Pinecone] Deleted {len(ids)} vectors from '{namespace}'")
            return True
        except Exception as e:
            logger.error(f"[Pinecone] Delete failed: {e}")
            return False

    def delete_by_filter(
        self,
        filter_dict: dict,
        namespace: str,
    ) -> bool:
        """Delete vectors matching a metadata filter.

        Args:
            filter_dict: Pinecone metadata filter.
            namespace: Namespace to delete from.

        Returns:
            True if successful, False otherwise.
        """
        index = _get_index()
        if index is None:
            return False

        try:
            index.delete(filter=filter_dict, namespace=namespace)
            logger.debug(f"[Pinecone] Deleted by filter from '{namespace}'")
            return True
        except Exception as e:
            logger.error(f"[Pinecone] Filter delete failed: {e}")
            return False

    # ═══════════════════════════════════════════════════════
    # HELPERS
    # ═══════════════════════════════════════════════════════

    @staticmethod
    def _make_id(chunk: Chunk) -> str:
        """Generate a deterministic vector ID from chunk metadata.

        Uses first available ID field from metadata,
        or falls back to namespace + chunk_index.
        """
        for key in ("trade_id", "progress_id", "pattern_id", "conversation_id"):
            if key in chunk.metadata:
                val = chunk.metadata[key]
                idx = chunk.metadata.get("chunk_index", 0)
                return f"{val}_{idx}" if "chunk_index" in chunk.metadata else val

        # Fallback: namespace + hash of text
        import hashlib

        text_hash = hashlib.md5(chunk.text.encode()).hexdigest()[:12]
        return f"{chunk.namespace}_{text_hash}"
