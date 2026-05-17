"""
SwingAdvisorBot — Module 7: Education Layer
mcp_tools.py — FastAPI MCP tool endpoints for education

Exposes M7's education capabilities as MCP tools
that other modules can call via HTTP.

From the MCP Tool Registry:
  | get_lesson             | M7 | GET  /tools/get_lesson             | M6, M8 |
  | submit_quiz_answer     | M7 | POST /tools/submit_quiz_answer     | M8     |
  | get_learning_progress  | M7 | GET  /tools/get_learning_progress  | M8     |

All tools return a ToolResponse envelope consistent with M1-M6.
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Optional

from fastapi import APIRouter
from pydantic import BaseModel, Field
from zoneinfo import ZoneInfo

IST = ZoneInfo("Asia/Kolkata")
logger = logging.getLogger("swing_advisor.mcp_tools_m7")


# ─────────────────────────────────────────────────────────────
# Router
# ─────────────────────────────────────────────────────────────

router = APIRouter(prefix="/tools", tags=["M7 Education"])


# ─────────────────────────────────────────────────────────────
# Request / Response Models
# ─────────────────────────────────────────────────────────────


class ToolResponse(BaseModel):
    """Standard MCP tool response envelope."""

    tool: str = Field(..., description="Tool name that produced this response")
    status: str = Field(..., description="ok | error")
    data: Optional[dict] = Field(default=None, description="Response payload")
    error: Optional[str] = Field(default=None, description="Error message if status=error")
    timestamp: str = Field(
        default_factory=lambda: datetime.now(IST).isoformat(),
        description="ISO timestamp (IST)",
    )


class QuizAnswerRequest(BaseModel):
    """Request body for submit_quiz_answer tool."""

    user_id: str = Field(default="XCU700", description="User ID")
    lesson_id: str = Field(..., description="Lesson ID to answer quiz for")
    answer: str = Field(..., description="Quiz answer: A or B")


# ─────────────────────────────────────────────────────────────
# Tool 1: get_lesson
# ─────────────────────────────────────────────────────────────


@router.get(
    "/get_lesson",
    response_model=ToolResponse,
    summary="Get today's lesson",
    description=(
        "Returns today's lesson for the user. Generates if not yet "
        "created today. Lesson is tied to today's real market events. "
        "Included in morning brief automatically."
    ),
)
async def get_lesson(user_id: str = "XCU700") -> ToolResponse:
    """Generate or retrieve today's lesson."""
    try:
        from module7_education.agents.education_agent import education_agent

        lesson = await education_agent.generate_daily_lesson(use_claude=True)

        return ToolResponse(
            tool="get_lesson",
            status="ok",
            data=lesson.model_dump(exclude_none=True, mode="json"),
        )

    except Exception as exc:
        logger.error(f"[MCP] get_lesson failed: {exc}")
        return ToolResponse(
            tool="get_lesson",
            status="error",
            error=str(exc),
        )


# ─────────────────────────────────────────────────────────────
# Tool 2: submit_quiz_answer
# ─────────────────────────────────────────────────────────────


@router.post(
    "/submit_quiz_answer",
    response_model=ToolResponse,
    summary="Submit quiz answer",
    description=(
        "Submits quiz answer and returns immediate feedback. "
        "Updates learning progress in M5 memory."
    ),
)
async def submit_quiz_answer(req: QuizAnswerRequest) -> ToolResponse:
    """Process quiz answer and return feedback."""
    try:
        if req.answer.upper() not in ("A", "B"):
            return ToolResponse(
                tool="submit_quiz_answer",
                status="error",
                error="Answer must be A or B",
            )

        from module7_education.agents.education_agent import education_agent

        feedback = await education_agent.handle_quiz(
            answer=req.answer,
            lesson_id=req.lesson_id,
        )

        return ToolResponse(
            tool="submit_quiz_answer",
            status="ok",
            data={
                "correct": feedback.correct,
                "feedback": feedback.feedback_html,
                "new_score": feedback.new_score,
                "streak": feedback.streak,
            },
        )

    except Exception as exc:
        logger.error(f"[MCP] submit_quiz_answer failed: {exc}")
        return ToolResponse(
            tool="submit_quiz_answer",
            status="error",
            error=str(exc),
        )


# ─────────────────────────────────────────────────────────────
# Tool 3: get_learning_progress
# ─────────────────────────────────────────────────────────────


@router.get(
    "/get_learning_progress",
    response_model=ToolResponse,
    summary="Get learning progress",
    description=(
        "Returns full learning history. Concepts taught, quiz scores, "
        "current difficulty level, and suggested next topics."
    ),
)
async def get_learning_progress(user_id: str = "XCU700") -> ToolResponse:
    """Return full learning state."""
    try:
        from module7_education.agents.education_agent import education_agent

        progress = education_agent.get_learning_progress()

        return ToolResponse(
            tool="get_learning_progress",
            status="ok",
            data=progress,
        )

    except Exception as exc:
        logger.error(f"[MCP] get_learning_progress failed: {exc}")
        return ToolResponse(
            tool="get_learning_progress",
            status="error",
            error=str(exc),
        )
