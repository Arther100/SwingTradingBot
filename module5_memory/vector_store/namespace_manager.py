"""
SwingAdvisorBot — Module 5: Memory & Personalization
vector_store/namespace_manager.py — Namespace routing and lifecycle

Decides which Pinecone namespaces to search based on agent type
and manages the full store-and-index pipeline for different data types.

Responsibilities:
  - Route agent queries to correct namespaces (AGENT_FOCUS_MAP)
  - Store a trade: chunk → embed → upsert to Pinecone
  - Store learning progress: chunk → embed → upsert
  - Store conversation/pattern/knowledge: chunk → embed → upsert
  - Validate namespace names against MemoryNamespace enum

Usage:
    mgr = NamespaceManager()
    mgr.store_trade(trade_record)
    mgr.store_learning(learning_progress)
    namespaces = mgr.get_namespaces_for_agent("TradeSetupAgent")
"""

from __future__ import annotations

import logging

from module5_memory.config import AGENT_FOCUS_MAP, ALL_NAMESPACES
from module5_memory.embeddings.chunker import (
    Chunk,
    chunk_conversation,
    chunk_knowledge,
    chunk_learning,
    chunk_market_pattern,
    chunk_trade,
)
from module5_memory.embeddings.embedding_engine import EmbeddingEngine, get_embedding_engine
from module5_memory.models import (
    LearningProgress,
    MemoryNamespace,
    TradeRecord,
)
from module5_memory.vector_store.pinecone_memory import PineconeMemory

logger = logging.getLogger("swing_advisor.m5_ns_manager")


class NamespaceManager:
    """Manages namespace routing and the chunk→embed→upsert pipeline.

    Orchestrates the flow:
      Data model → Chunker → EmbeddingEngine → PineconeMemory

    All store methods are no-ops if Pinecone is unavailable.
    """

    def __init__(
        self,
        pinecone: PineconeMemory | None = None,
        embedding_engine: EmbeddingEngine | None = None,
    ) -> None:
        self._pinecone = pinecone or PineconeMemory()
        self._engine = embedding_engine or get_embedding_engine()

    @property
    def is_available(self) -> bool:
        """Check if both Pinecone and embedding engine are ready."""
        return self._pinecone.is_available and self._engine.is_available

    # ═══════════════════════════════════════════════════════
    # NAMESPACE ROUTING
    # ═══════════════════════════════════════════════════════

    @staticmethod
    def get_namespaces_for_agent(agent_name: str) -> list[str]:
        """Get the namespaces an agent should search.

        Falls back to trade_memory + market_patterns if agent
        is not in the focus map.

        Args:
            agent_name: CrewAI agent class name.

        Returns:
            List of namespace strings.
        """
        return AGENT_FOCUS_MAP.get(
            agent_name,
            [
                MemoryNamespace.TRADE_MEMORY.value,
                MemoryNamespace.MARKET_PATTERNS.value,
            ],
        )

    @staticmethod
    def get_all_namespaces() -> list[str]:
        """Get all configured namespace names."""
        return list(ALL_NAMESPACES)

    @staticmethod
    def is_valid_namespace(namespace: str) -> bool:
        """Check if a namespace string is valid."""
        return namespace in ALL_NAMESPACES

    # ═══════════════════════════════════════════════════════
    # STORE PIPELINE: chunk → embed → upsert
    # ═══════════════════════════════════════════════════════

    def store_trade(self, trade: TradeRecord) -> bool:
        """Store a trade record in Pinecone.

        Pipeline: TradeRecord → chunk_trade → embed → upsert

        Returns:
            True if stored successfully, False otherwise.
        """
        if not self.is_available:
            logger.debug("[NSManager] Skipping store_trade — services unavailable.")
            return False

        chunk = chunk_trade(trade)
        embedding = self._engine.embed(chunk.text)
        count = self._pinecone.upsert_chunks([chunk], [embedding])
        return count > 0

    def store_learning(self, progress: LearningProgress) -> bool:
        """Store learning progress in Pinecone.

        Pipeline: LearningProgress → chunk_learning → embed → upsert

        Returns:
            True if stored successfully, False otherwise.
        """
        if not self.is_available:
            logger.debug("[NSManager] Skipping store_learning — services unavailable.")
            return False

        chunk = chunk_learning(progress)
        embedding = self._engine.embed(chunk.text)
        count = self._pinecone.upsert_chunks([chunk], [embedding])
        return count > 0

    def store_conversation(
        self,
        text: str,
        conversation_id: str,
    ) -> int:
        """Store a conversation in Pinecone (sliding window chunks).

        Returns:
            Number of chunks stored (0 if unavailable).
        """
        if not self.is_available:
            return 0

        chunks = chunk_conversation(text, conversation_id)
        if not chunks:
            return 0

        texts = [c.text for c in chunks]
        embeddings = self._engine.embed_batch(texts)
        return self._pinecone.upsert_chunks(chunks, embeddings)

    def store_market_pattern(
        self,
        pattern_text: str,
        pattern_id: str,
        ticker: str | None = None,
        date: str | None = None,
    ) -> bool:
        """Store a market pattern observation in Pinecone.

        Returns:
            True if stored successfully, False otherwise.
        """
        if not self.is_available:
            return False

        chunk = chunk_market_pattern(pattern_text, pattern_id, ticker, date)
        embedding = self._engine.embed(chunk.text)
        count = self._pinecone.upsert_chunks([chunk], [embedding])
        return count > 0

    def store_knowledge(
        self,
        text: str,
        source: str,
        topic: str | None = None,
    ) -> int:
        """Store knowledge base content in Pinecone (paragraph chunks).

        Returns:
            Number of chunks stored (0 if unavailable).
        """
        if not self.is_available:
            return 0

        chunks = chunk_knowledge(text, source, topic)
        if not chunks:
            return 0

        texts = [c.text for c in chunks]
        embeddings = self._engine.embed_batch(texts)
        return self._pinecone.upsert_chunks(chunks, embeddings)

    # ═══════════════════════════════════════════════════════
    # DELETE
    # ═══════════════════════════════════════════════════════

    def delete_trade(self, trade_id: str) -> bool:
        """Delete a trade's vector from Pinecone."""
        return self._pinecone.delete_by_ids(
            [trade_id],
            MemoryNamespace.TRADE_MEMORY.value,
        )

    def delete_learning(self, progress_id: str) -> bool:
        """Delete a learning progress vector from Pinecone."""
        return self._pinecone.delete_by_ids(
            [progress_id],
            MemoryNamespace.LESSONS.value,
        )
