"""
SwingAdvisorBot — Module 5: Memory & Personalization
embeddings/chunker.py — Text chunking strategies per data type

Converts structured data (trades, lessons, conversations) into
text chunks suitable for embedding and Pinecone storage.

Each data type has its own chunking strategy:
  TradeRecord       → single chunk per trade (to_embedding_text)
  LearningProgress  → single chunk per concept
  Conversation      → sliding window with overlap
  MarketPattern     → single chunk per pattern
  KnowledgeBase     → paragraph-level chunks

Token estimation: ~4 chars = 1 token (conservative).
Chunks stay under 512 tokens for all-MiniLM-L6-v2 max input.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from module5_memory.models import (
    LearningProgress,
    MemoryNamespace,
    TradeRecord,
)

logger = logging.getLogger("swing_advisor.m5_chunker")

# all-MiniLM-L6-v2 max sequence length
MAX_CHUNK_TOKENS = 512
CHARS_PER_TOKEN = 4  # conservative estimate
MAX_CHUNK_CHARS = MAX_CHUNK_TOKENS * CHARS_PER_TOKEN


# ─────────────────────────────────────────────────────────────
# Chunk dataclass
# ─────────────────────────────────────────────────────────────


@dataclass
class Chunk:
    """A single text chunk ready for embedding.

    Attributes:
        text: The chunk text content.
        namespace: Target Pinecone namespace.
        metadata: Key-value metadata for Pinecone storage.
        token_estimate: Approximate token count.
    """

    text: str
    namespace: str
    metadata: dict
    token_estimate: int

    @staticmethod
    def estimate_tokens(text: str) -> int:
        """Estimate token count from text length."""
        return max(1, len(text) // CHARS_PER_TOKEN)


# ─────────────────────────────────────────────────────────────
# Trade chunking
# ─────────────────────────────────────────────────────────────


def chunk_trade(trade: TradeRecord) -> Chunk:
    """Convert a TradeRecord into a single embeddable chunk.

    Uses TradeRecord.to_embedding_text() for content
    and TradeRecord.to_embedding_metadata() for metadata.
    """
    text = trade.to_embedding_text()
    metadata = trade.to_embedding_metadata()

    return Chunk(
        text=text[:MAX_CHUNK_CHARS],
        namespace=MemoryNamespace.TRADE_MEMORY.value,
        metadata=metadata,
        token_estimate=Chunk.estimate_tokens(text),
    )


# ─────────────────────────────────────────────────────────────
# Learning progress chunking
# ─────────────────────────────────────────────────────────────


def chunk_learning(progress: LearningProgress) -> Chunk:
    """Convert a LearningProgress into a single embeddable chunk."""
    score_text = f" Quiz score: {progress.quiz_score}%." if progress.quiz_score is not None else ""
    text = (
        f"Concept: {progress.concept}. "
        f"Taught {progress.times_taught} time(s). "
        f"Last taught: {progress.last_taught.strftime('%Y-%m-%d')}.{score_text}"
    )

    metadata = {
        "progress_id": progress.progress_id,
        "concept": progress.concept,
        "times_taught": progress.times_taught,
        "last_taught": progress.last_taught.isoformat(),
    }
    if progress.quiz_score is not None:
        metadata["quiz_score"] = progress.quiz_score

    return Chunk(
        text=text[:MAX_CHUNK_CHARS],
        namespace=MemoryNamespace.LESSONS.value,
        metadata=metadata,
        token_estimate=Chunk.estimate_tokens(text),
    )


# ─────────────────────────────────────────────────────────────
# Conversation chunking (sliding window)
# ─────────────────────────────────────────────────────────────


def chunk_conversation(
    text: str,
    conversation_id: str,
    max_tokens: int = 400,
    overlap_tokens: int = 50,
) -> list[Chunk]:
    """Split a conversation into overlapping chunks.

    Args:
        text: Full conversation text.
        conversation_id: Unique ID for metadata.
        max_tokens: Max tokens per chunk.
        overlap_tokens: Token overlap between chunks.

    Returns:
        List of Chunk objects with conversation namespace.
    """
    if not text or not text.strip():
        return []

    max_chars = max_tokens * CHARS_PER_TOKEN
    overlap_chars = overlap_tokens * CHARS_PER_TOKEN
    step = max_chars - overlap_chars

    chunks: list[Chunk] = []
    start = 0
    chunk_idx = 0

    while start < len(text):
        end = min(start + max_chars, len(text))
        segment = text[start:end]

        if segment.strip():
            chunks.append(
                Chunk(
                    text=segment,
                    namespace=MemoryNamespace.CONVERSATIONS.value,
                    metadata={
                        "conversation_id": conversation_id,
                        "chunk_index": chunk_idx,
                    },
                    token_estimate=Chunk.estimate_tokens(segment),
                )
            )
            chunk_idx += 1

        start += step

    return chunks


# ─────────────────────────────────────────────────────────────
# Market pattern chunking
# ─────────────────────────────────────────────────────────────


def chunk_market_pattern(
    pattern_text: str,
    pattern_id: str,
    ticker: str | None = None,
    date: str | None = None,
) -> Chunk:
    """Wrap a market pattern observation into a chunk.

    Args:
        pattern_text: The pattern description.
        pattern_id: Unique identifier.
        ticker: Related ticker (optional).
        date: Date of observation (optional).

    Returns:
        Single Chunk in market_patterns namespace.
    """
    metadata: dict = {"pattern_id": pattern_id}
    if ticker:
        metadata["ticker"] = ticker
    if date:
        metadata["date"] = date

    return Chunk(
        text=pattern_text[:MAX_CHUNK_CHARS],
        namespace=MemoryNamespace.MARKET_PATTERNS.value,
        metadata=metadata,
        token_estimate=Chunk.estimate_tokens(pattern_text),
    )


# ─────────────────────────────────────────────────────────────
# Knowledge base chunking (paragraph-level)
# ─────────────────────────────────────────────────────────────


def chunk_knowledge(
    text: str,
    source: str,
    topic: str | None = None,
) -> list[Chunk]:
    """Split knowledge base content into paragraph-level chunks.

    Splits on double newlines. Each paragraph becomes a chunk.

    Args:
        text: Full knowledge base text.
        source: Source identifier (e.g. "investopedia", "manual").
        topic: Topic tag for metadata (optional).

    Returns:
        List of Chunk objects in knowledge_base namespace.
    """
    if not text or not text.strip():
        return []

    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    chunks: list[Chunk] = []

    for i, para in enumerate(paragraphs):
        # Skip very short paragraphs
        if len(para) < 20:
            continue

        metadata: dict = {
            "source": source,
            "chunk_index": i,
        }
        if topic:
            metadata["topic"] = topic

        chunks.append(
            Chunk(
                text=para[:MAX_CHUNK_CHARS],
                namespace=MemoryNamespace.KNOWLEDGE_BASE.value,
                metadata=metadata,
                token_estimate=Chunk.estimate_tokens(para),
            )
        )

    return chunks
