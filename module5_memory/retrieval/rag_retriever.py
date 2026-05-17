"""
SwingAdvisorBot — Module 5: Memory & Personalization
retrieval/rag_retriever.py — Similarity search across Pinecone namespaces

Queries Pinecone for chunks relevant to a given text query.
Searches across multiple namespaces (based on agent focus map)
and merges results by score.

Features:
  - Embed query text → search N namespaces → merge by score
  - Agent-aware namespace routing
  - min_score filtering
  - max_chunks_total cap across all namespaces
  - Graceful fallback: empty list if services unavailable

Usage:
    retriever = RAGRetriever()
    chunks = retriever.retrieve("HDFCBANK swing trade setup")
    chunks = retriever.retrieve_for_agent("TradeSetupAgent", "HDFCBANK entry")
"""

from __future__ import annotations

import logging

from module5_memory.config import get_retrieval_config
from module5_memory.embeddings.embedding_engine import EmbeddingEngine, get_embedding_engine
from module5_memory.models import RetrievedChunk
from module5_memory.vector_store.namespace_manager import NamespaceManager
from module5_memory.vector_store.pinecone_memory import PineconeMemory

logger = logging.getLogger("swing_advisor.m5_retriever")


class RAGRetriever:
    """Retrieves semantically similar chunks from Pinecone.

    Orchestrates: query text → embed → search namespaces → merge results.

    Usage:
        retriever = RAGRetriever()
        chunks = retriever.retrieve("HDFCBANK support at 760")
        chunks = retriever.retrieve_for_agent("TradeSetupAgent", "HDFCBANK")
    """

    def __init__(
        self,
        pinecone: PineconeMemory | None = None,
        embedding_engine: EmbeddingEngine | None = None,
        namespace_manager: NamespaceManager | None = None,
    ) -> None:
        self._pinecone = pinecone or PineconeMemory()
        self._engine = embedding_engine or get_embedding_engine()
        self._ns_manager = namespace_manager or NamespaceManager(
            pinecone=self._pinecone,
            embedding_engine=self._engine,
        )
        self._config = get_retrieval_config()

    @property
    def is_available(self) -> bool:
        """Check if retrieval services are ready."""
        return self._pinecone.is_available and self._engine.is_available

    def retrieve(
        self,
        query_text: str,
        namespaces: list[str] | None = None,
        top_k: int | None = None,
        min_score: float | None = None,
        max_total: int | None = None,
    ) -> list[RetrievedChunk]:
        """Retrieve relevant chunks across namespaces.

        Args:
            query_text: Natural language query to search for.
            namespaces: Namespaces to search (default: all).
            top_k: Max results per namespace (default: config).
            min_score: Minimum cosine similarity (default: config).
            max_total: Max total chunks returned (default: config).

        Returns:
            List of RetrievedChunk sorted by score descending.
            Empty list if services unavailable or query empty.
        """
        if not query_text or not query_text.strip():
            return []

        if not self.is_available:
            logger.debug("[Retriever] Services unavailable — returning empty.")
            return []

        # Defaults from config
        top_k = top_k or self._config.top_k
        min_score = min_score or self._config.min_score
        max_total = max_total or self._config.max_chunks_total
        namespaces = namespaces or self._ns_manager.get_all_namespaces()

        # Embed query
        query_vector = self._engine.embed(query_text)

        # Search each namespace
        all_chunks: list[RetrievedChunk] = []
        for ns in namespaces:
            if not self._ns_manager.is_valid_namespace(ns):
                logger.warning(f"[Retriever] Skipping invalid namespace: {ns}")
                continue

            chunks = self._pinecone.query(
                vector=query_vector,
                namespace=ns,
                top_k=top_k,
                min_score=min_score,
            )
            all_chunks.extend(chunks)

        # Sort by score descending and cap total
        all_chunks.sort(key=lambda c: c.score, reverse=True)
        result = all_chunks[:max_total]

        logger.debug(
            f"[Retriever] Query '{query_text[:50]}...' → "
            f"{len(result)} chunks from {len(namespaces)} namespaces"
        )
        return result

    def retrieve_for_agent(
        self,
        agent_name: str,
        query_text: str,
        top_k: int | None = None,
        min_score: float | None = None,
    ) -> list[RetrievedChunk]:
        """Retrieve chunks scoped to an agent's focus namespaces.

        Uses AGENT_FOCUS_MAP to determine which namespaces to search.

        Args:
            agent_name: CrewAI agent class name.
            query_text: Natural language query.
            top_k: Max results per namespace.
            min_score: Minimum cosine similarity.

        Returns:
            List of RetrievedChunk sorted by score descending.
        """
        namespaces = self._ns_manager.get_namespaces_for_agent(agent_name)
        logger.debug(f"[Retriever] Agent '{agent_name}' → namespaces: {namespaces}")
        return self.retrieve(
            query_text=query_text,
            namespaces=namespaces,
            top_k=top_k,
            min_score=min_score,
        )

    def retrieve_by_ticker(
        self,
        ticker: str,
        namespaces: list[str] | None = None,
        top_k: int | None = None,
    ) -> list[RetrievedChunk]:
        """Retrieve chunks related to a specific ticker.

        Combines semantic search with metadata filter.

        Args:
            ticker: NSE ticker symbol (e.g. "HDFCBANK").
            namespaces: Namespaces to search.
            top_k: Max results per namespace.

        Returns:
            List of RetrievedChunk sorted by score descending.
        """
        query_text = f"{ticker} trade entry exit stop loss target"
        return self.retrieve(
            query_text=query_text,
            namespaces=namespaces,
            top_k=top_k,
        )
