"""
SwingAdvisorBot — Module 5: Memory & Personalization
config.py — Configuration for SQLite, Pinecone, embeddings, and verification

All external service config loaded from .env.
Sensible defaults for local development.

Config sections:
  SQLite      → Database path, WAL mode
  Pinecone    → API key, index host, namespace defaults
  Embeddings  → Model name, dimensions, batch size
  Memory      → Token budget, retrieval thresholds
  Verification → Claude prompt for 2-round check
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv
from pydantic import BaseModel, Field

load_dotenv()


# ─────────────────────────────────────────────────────────────
# SQLite Configuration
# ─────────────────────────────────────────────────────────────

# Database stored alongside project root
SQLITE_DB_PATH = os.getenv(
    "SQLITE_DB_PATH",
    str(Path(__file__).parent / "data" / "memory.db"),
)

# WAL mode for concurrent read/write
SQLITE_JOURNAL_MODE = "WAL"


# ─────────────────────────────────────────────────────────────
# Pinecone Configuration
# ─────────────────────────────────────────────────────────────

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY", "")
PINECONE_INDEX_HOST = os.getenv("PINECONE_INDEX_HOST", "")

# Namespace constants (match MemoryNamespace enum)
NAMESPACE_TRADE_MEMORY = "trade_memory"
NAMESPACE_MARKET_PATTERNS = "market_patterns"
NAMESPACE_CONVERSATIONS = "conversations"
NAMESPACE_LESSONS = "lessons"
NAMESPACE_KNOWLEDGE_BASE = "knowledge_base"

ALL_NAMESPACES = [
    NAMESPACE_TRADE_MEMORY,
    NAMESPACE_MARKET_PATTERNS,
    NAMESPACE_CONVERSATIONS,
    NAMESPACE_LESSONS,
    NAMESPACE_KNOWLEDGE_BASE,
]


# ─────────────────────────────────────────────────────────────
# Embedding Model Configuration
# ─────────────────────────────────────────────────────────────

EMBEDDING_MODEL_NAME = os.getenv(
    "EMBEDDING_MODEL", "all-MiniLM-L6-v2"
)
EMBEDDING_DIMENSIONS = 384  # all-MiniLM-L6-v2 output size
EMBEDDING_BATCH_SIZE = 32
EMBEDDING_NORMALIZE = True


# ─────────────────────────────────────────────────────────────
# Memory Context Budget
# ─────────────────────────────────────────────────────────────

class MemoryBudget(BaseModel):
    """Token budget for memory context injected into Claude prompts.

    Total: 300 tokens HARD LIMIT (reserved in M2 since day one).

    Breakdown:
      Profile summary:   80 tokens (always included)
      Trade history:    100 tokens (top 2 chunks)
      Pattern chunks:    70 tokens (top 1 chunk)
      Lesson history:    50 tokens (top 1 chunk)
    """

    total_budget: int = Field(default=300, description="Hard limit — never exceed")
    profile_budget: int = Field(default=80, description="User profile summary")
    trade_history_budget: int = Field(default=100, description="Trade history chunks")
    pattern_budget: int = Field(default=70, description="Market pattern chunks")
    lesson_budget: int = Field(default=50, description="Lesson history chunks")


# ─────────────────────────────────────────────────────────────
# Retrieval Configuration
# ─────────────────────────────────────────────────────────────

class RetrievalConfig(BaseModel):
    """Configuration for RAG retrieval from Pinecone."""

    top_k: int = Field(default=3, description="Max chunks to retrieve per namespace")
    min_score: float = Field(default=0.6, description="Minimum cosine similarity score")
    max_chunks_total: int = Field(default=5, description="Max total chunks after filtering")


# ─────────────────────────────────────────────────────────────
# Verification Configuration
# ─────────────────────────────────────────────────────────────

# Claude model for 2-round verification
VERIFICATION_CLAUDE_MODEL = os.getenv("CLAUDE_MODEL", "claude-opus-4-5")
VERIFICATION_TEMPERATURE = 0.2  # Low temp for factual checking
VERIFICATION_MAX_TOKENS = 400
VERIFICATION_MIN_CONFIDENCE = 0.7  # Below this → flag for user review


VERIFICATION_PROMPT = """You are a senior risk reviewer checking advice given by a junior advisor. Review this advice:

{round1_advice}

Check against these criteria:
1. Is advice consistent with user's risk tolerance?
   User tolerance: {risk_tolerance}
   Capital: ₹{capital}

2. Are all price levels grounded in provided data?
   Available prices: {available_prices}

3. Does advice contradict user's past experience?
   Relevant history: {relevant_history}

4. Any unsupported claims or hallucinations?

5. Is position sizing within safe limits?

Respond in valid JSON only. No text outside JSON.
Start with {{ end with }}. No markdown. No backticks.

Required JSON structure:
{{
  "verified": true,
  "issues_found": [],
  "corrected_advice": "corrected version if needed or null",
  "verification_note": "brief explanation",
  "confidence": 0.95
}}

If verified=true and no issues → return original advice unchanged.
If issues found → return corrected version in corrected_advice."""


# ─────────────────────────────────────────────────────────────
# Agent Focus Map — which namespaces each agent searches
# ─────────────────────────────────────────────────────────────

AGENT_FOCUS_MAP: dict[str, list[str]] = {
    "MarketAnalysisAgent": [
        NAMESPACE_MARKET_PATTERNS,
        NAMESPACE_TRADE_MEMORY,
    ],
    "TradeSetupAgent": [
        NAMESPACE_TRADE_MEMORY,
        NAMESPACE_MARKET_PATTERNS,
    ],
    "ReportAgent": [
        NAMESPACE_TRADE_MEMORY,
        NAMESPACE_CONVERSATIONS,
    ],
    "EducationAgent": [
        NAMESPACE_LESSONS,
        NAMESPACE_KNOWLEDGE_BASE,
    ],
}


# ─────────────────────────────────────────────────────────────
# Singletons
# ─────────────────────────────────────────────────────────────

@lru_cache(maxsize=1)
def get_memory_budget() -> MemoryBudget:
    """Get memory budget singleton."""
    return MemoryBudget()


@lru_cache(maxsize=1)
def get_retrieval_config() -> RetrievalConfig:
    """Get retrieval config singleton."""
    return RetrievalConfig()
