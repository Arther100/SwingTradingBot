"""
SwingAdvisorBot — Module 5: Memory & Personalization
retrieval/context_builder.py — Build ≤300 token memory context for Claude

Assembles a MemoryContext from:
  1. User profile summary (always first, ~80 tokens)
  2. Trade history chunks (up to 100 tokens)
  3. Pattern chunks (up to 70 tokens)
  4. Lesson chunks (up to 50 tokens)

Priority trimming: If total exceeds 300 tokens, trim from lowest
priority (lessons → patterns → trades). Profile is never trimmed.

The output MemoryContext.text is injected directly into Claude's
system prompt in the memory section reserved by M2.

Usage:
    builder = ContextBuilder(sqlite_manager, rag_retriever)
    ctx = builder.build_context(user_id="XCU700", query="HDFCBANK setup")
    # ctx.text → inject into Claude prompt
    # ctx.within_budget → True (≤300 tokens)
"""

from __future__ import annotations

import logging

from module5_memory.config import get_memory_budget
from module5_memory.database.sqlite_manager import SQLiteManager
from module5_memory.models import MemoryContext, MemoryNamespace, RetrievedChunk, UserProfile
from module5_memory.retrieval.rag_retriever import RAGRetriever

logger = logging.getLogger("swing_advisor.m5_context_builder")


# ─────────────────────────────────────────────────────────────
# Token estimation
# ─────────────────────────────────────────────────────────────

CHARS_PER_TOKEN = 4  # conservative estimate


def _estimate_tokens(text: str) -> int:
    """Estimate token count from text length."""
    return max(1, len(text) // CHARS_PER_TOKEN)


def _trim_to_tokens(text: str, max_tokens: int) -> str:
    """Trim text to fit within max_tokens (approximate).

    Trims at word boundaries when possible.
    """
    max_chars = max_tokens * CHARS_PER_TOKEN
    if len(text) <= max_chars:
        return text

    trimmed = text[:max_chars]
    # Try to trim at last space for cleaner output
    last_space = trimmed.rfind(" ")
    if last_space > max_chars * 0.7:
        trimmed = trimmed[:last_space]

    return trimmed + "..."


# ─────────────────────────────────────────────────────────────
# ContextBuilder
# ─────────────────────────────────────────────────────────────


class ContextBuilder:
    """Builds ≤300 token memory context for Claude prompts.

    Combines structured data (SQLite profile) with semantic
    search results (Pinecone chunks) into a single text block.

    Budget enforcement:
      1. Profile summary: always included (≤80 tokens)
      2. Trade chunks: top chunks up to 100 tokens
      3. Pattern chunks: up to 70 tokens
      4. Lesson chunks: up to 50 tokens
      5. If over budget → trim lowest priority first

    Usage:
        builder = ContextBuilder(sqlite_mgr, retriever)
        ctx = builder.build_context("XCU700", "HDFCBANK swing trade")
    """

    def __init__(
        self,
        sqlite_manager: SQLiteManager,
        retriever: RAGRetriever,
    ) -> None:
        self._sqlite = sqlite_manager
        self._retriever = retriever
        self._budget = get_memory_budget()

    def build_context(
        self,
        user_id: str,
        query: str = "",
        agent_name: str | None = None,
    ) -> MemoryContext:
        """Build the full memory context for a Claude prompt.

        Args:
            user_id: User's Zerodha client ID.
            query: Query text for semantic search (optional).
            agent_name: Agent name for namespace scoping (optional).

        Returns:
            MemoryContext with text ≤ 300 tokens.
            Returns empty context if user not found.
        """
        # 1. Get user profile (always needed)
        profile = self._sqlite.get_user_profile(user_id)
        if not profile:
            logger.warning(f"[ContextBuilder] User {user_id} not found — empty context.")
            return MemoryContext(text="", token_estimate=0, chunks_used=0)

        # 2. Build profile section
        profile_text = profile.to_context_summary()
        profile_tokens = _estimate_tokens(profile_text)

        # Cap profile at budget
        if profile_tokens > self._budget.profile_budget:
            profile_text = _trim_to_tokens(profile_text, self._budget.profile_budget)
            profile_tokens = self._budget.profile_budget

        # 3. Retrieve semantic chunks (if query provided)
        trade_text = ""
        pattern_text = ""
        lesson_text = ""
        chunks_used = 0

        if query and self._retriever.is_available:
            chunks = self._retrieve_chunks(query, agent_name)
            trade_text, pattern_text, lesson_text, chunks_used = (
                self._categorize_and_trim(chunks)
            )

        # 4. Assemble final context
        sections: list[str] = [profile_text]

        if trade_text:
            sections.append(f"Recent trades: {trade_text}")
        if pattern_text:
            sections.append(f"Patterns: {pattern_text}")
        if lesson_text:
            sections.append(f"Lessons: {lesson_text}")

        full_text = " | ".join(sections)
        total_tokens = _estimate_tokens(full_text)

        # 5. Final budget enforcement
        if total_tokens > self._budget.total_budget:
            full_text = _trim_to_tokens(full_text, self._budget.total_budget)
            total_tokens = self._budget.total_budget

        context = MemoryContext(
            text=full_text,
            token_estimate=total_tokens,
            chunks_used=chunks_used,
        )

        logger.debug(
            f"[ContextBuilder] Built context: {total_tokens} tokens, "
            f"{chunks_used} chunks, within_budget={context.within_budget}"
        )
        return context

    def build_context_profile_only(self, user_id: str) -> MemoryContext:
        """Build context with just the profile (no semantic search).

        Useful when no specific query is available (e.g. greeting).
        """
        return self.build_context(user_id=user_id, query="")

    # ═══════════════════════════════════════════════════════
    # Private helpers
    # ═══════════════════════════════════════════════════════

    def _retrieve_chunks(
        self,
        query: str,
        agent_name: str | None,
    ) -> list[RetrievedChunk]:
        """Retrieve chunks using RAGRetriever."""
        if agent_name:
            return self._retriever.retrieve_for_agent(agent_name, query)
        return self._retriever.retrieve(query)

    def _categorize_and_trim(
        self,
        chunks: list[RetrievedChunk],
    ) -> tuple[str, str, str, int]:
        """Categorize chunks by namespace and trim to budget.

        Returns:
            (trade_text, pattern_text, lesson_text, chunks_used)
        """
        trade_chunks: list[str] = []
        pattern_chunks: list[str] = []
        lesson_chunks: list[str] = []

        for chunk in chunks:
            if chunk.namespace == MemoryNamespace.TRADE_MEMORY.value:
                trade_chunks.append(chunk.content)
            elif chunk.namespace == MemoryNamespace.MARKET_PATTERNS.value:
                pattern_chunks.append(chunk.content)
            elif chunk.namespace in (
                MemoryNamespace.LESSONS.value,
                MemoryNamespace.KNOWLEDGE_BASE.value,
            ):
                lesson_chunks.append(chunk.content)
            else:
                # Conversations or unknown → treat as trade context
                trade_chunks.append(chunk.content)

        # Trim each category to its budget
        trade_text = _trim_to_tokens(
            " ".join(trade_chunks),
            self._budget.trade_history_budget,
        ) if trade_chunks else ""

        pattern_text = _trim_to_tokens(
            " ".join(pattern_chunks),
            self._budget.pattern_budget,
        ) if pattern_chunks else ""

        lesson_text = _trim_to_tokens(
            " ".join(lesson_chunks),
            self._budget.lesson_budget,
        ) if lesson_chunks else ""

        chunks_used = len(chunks)
        return trade_text, pattern_text, lesson_text, chunks_used
