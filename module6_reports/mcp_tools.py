"""
SwingAdvisorBot — Module 6: Daily Reports + Alerts
mcp_tools.py — FastAPI MCP tool endpoints for reports

Exposes M6's report and alert capabilities as MCP tools
that other modules can call via HTTP.

From the MCP Tool Registry:
  | send_morning_brief   | M6 | POST /tools/send_morning_brief   | M8     |
  | send_evening_review  | M6 | POST /tools/send_evening_review  | M8     |
  | send_weekly_summary  | M6 | POST /tools/send_weekly_summary  | M8     |
  | send_error_alert     | M6 | POST /tools/send_error_alert     | M1-M5  |
  | send_custom_message  | M6 | POST /tools/send_custom_message  | M8     |
  | get_scheduler_status | M6 | GET  /tools/get_scheduler_status | M8     |
  | get_today_alerts     | M6 | GET  /tools/get_today_alerts     | M8     |

All tools return a ToolResponse envelope consistent with M1-M5.
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Optional

from fastapi import APIRouter
from pydantic import BaseModel, Field
from zoneinfo import ZoneInfo

from module6_reports.config import IST

logger = logging.getLogger("swing_advisor.mcp_tools_m6")


# ─────────────────────────────────────────────────────────────
# Router
# ─────────────────────────────────────────────────────────────

router = APIRouter(prefix="/tools", tags=["M6 Reports & Alerts"])


# ─────────────────────────────────────────────────────────────
# Request / Response Models
# ─────────────────────────────────────────────────────────────


class ToolResponse(BaseModel):
    """Standard MCP tool response envelope."""

    tool: str = Field(..., description="Tool name that produced this response")
    status: str = Field(default="success", description="success or error")
    data: Optional[dict] = Field(default=None, description="Tool output data")
    error: Optional[str] = Field(default=None, description="Error message if failed")
    timestamp: str = Field(
        default_factory=lambda: datetime.now(IST).isoformat(),
        description="IST timestamp of response",
    )


class SendReportRequest(BaseModel):
    """Request body for report generation tools."""

    user_id: str = Field(default="XCU700", description="User ID")
    skip_claude: bool = Field(
        default=False,
        description="Skip Claude API call (use template formatting)",
    )


class SendErrorRequest(BaseModel):
    """Request body for error alert tool."""

    source: str = Field(..., description="Module/step that failed")
    message: str = Field(..., description="Human-readable error description")
    is_critical: bool = Field(
        default=False,
        description="Whether this blocks all operations",
    )


class SendMessageRequest(BaseModel):
    """Request body for custom message tool."""

    message: str = Field(..., description="Message text to send")
    parse_mode: str = Field(
        default="HTML",
        description="Telegram parse mode (HTML or Markdown)",
    )


# ─────────────────────────────────────────────────────────────
# REPORT TOOLS
# ─────────────────────────────────────────────────────────────


@router.post("/send_morning_brief")
async def send_morning_brief(request: SendReportRequest) -> ToolResponse:
    """Generate and send morning brief to Telegram.

    Runs the full M1→M2→M3→M4→M5 pipeline, generates
    Claude telegram_text, and sends to Vijay's Telegram.
    """
    try:
        from module6_reports.agents.report_agent import report_agent

        result = await report_agent.generate_and_send_morning_brief(
            user_id=request.user_id,
            skip_claude=request.skip_claude,
        )

        return ToolResponse(
            tool="send_morning_brief",
            status=result["status"],
            data=result,
        )

    except Exception as e:
        logger.error(f"[MCP] send_morning_brief failed: {e}")
        return ToolResponse(
            tool="send_morning_brief",
            status="error",
            error=str(e),
        )


@router.post("/send_evening_review")
async def send_evening_review(request: SendReportRequest) -> ToolResponse:
    """Generate and send evening review to Telegram."""
    try:
        from module6_reports.agents.report_agent import report_agent

        result = await report_agent.generate_and_send_evening_review(
            user_id=request.user_id,
            skip_claude=request.skip_claude,
        )

        return ToolResponse(
            tool="send_evening_review",
            status=result["status"],
            data=result,
        )

    except Exception as e:
        logger.error(f"[MCP] send_evening_review failed: {e}")
        return ToolResponse(
            tool="send_evening_review",
            status="error",
            error=str(e),
        )


@router.post("/send_weekly_summary")
async def send_weekly_summary(request: SendReportRequest) -> ToolResponse:
    """Generate and send weekly summary to Telegram."""
    try:
        from module6_reports.agents.report_agent import report_agent

        result = await report_agent.generate_and_send_weekly_summary(
            user_id=request.user_id,
            skip_claude=request.skip_claude,
        )

        return ToolResponse(
            tool="send_weekly_summary",
            status=result["status"],
            data=result,
        )

    except Exception as e:
        logger.error(f"[MCP] send_weekly_summary failed: {e}")
        return ToolResponse(
            tool="send_weekly_summary",
            status="error",
            error=str(e),
        )


@router.post("/send_error_alert")
async def send_error_alert(request: SendErrorRequest) -> ToolResponse:
    """Send an error alert to Telegram.

    Called by any module when something fails.
    Never silently fail.
    """
    try:
        from module6_reports.agents.report_agent import report_agent

        msg_id = await report_agent.send_error_alert(
            source=request.source,
            message=request.message,
            is_critical=request.is_critical,
        )

        return ToolResponse(
            tool="send_error_alert",
            status="sent" if msg_id else "failed",
            data={"telegram_message_id": msg_id},
        )

    except Exception as e:
        logger.error(f"[MCP] send_error_alert failed: {e}")
        return ToolResponse(
            tool="send_error_alert",
            status="error",
            error=str(e),
        )


@router.post("/send_custom_message")
async def send_custom_message(request: SendMessageRequest) -> ToolResponse:
    """Send a custom message to Telegram."""
    try:
        from module6_reports.agents.report_agent import report_agent

        msg_id = await report_agent.send_custom_message(
            message=request.message,
            parse_mode=request.parse_mode,
        )

        return ToolResponse(
            tool="send_custom_message",
            status="sent" if msg_id else "failed",
            data={"telegram_message_id": msg_id},
        )

    except Exception as e:
        logger.error(f"[MCP] send_custom_message failed: {e}")
        return ToolResponse(
            tool="send_custom_message",
            status="error",
            error=str(e),
        )


# ─────────────────────────────────────────────────────────────
# STATUS TOOLS
# ─────────────────────────────────────────────────────────────


@router.get("/get_scheduler_status")
async def get_scheduler_status() -> ToolResponse:
    """Get current scheduler status and next run times."""
    try:
        from module6_reports.scheduler.report_scheduler import report_scheduler

        status = report_scheduler.get_status()

        return ToolResponse(
            tool="get_scheduler_status",
            status="success",
            data=status,
        )

    except Exception as e:
        logger.error(f"[MCP] get_scheduler_status failed: {e}")
        return ToolResponse(
            tool="get_scheduler_status",
            status="error",
            error=str(e),
        )


@router.get("/get_today_alerts")
async def get_today_alerts() -> ToolResponse:
    """Get all alerts sent today (for dedup visibility)."""
    try:
        from module6_reports.alerts.alert_tracker import alert_tracker

        alerts = alert_tracker.get_today_alerts()
        count = alert_tracker.get_alert_count()

        return ToolResponse(
            tool="get_today_alerts",
            status="success",
            data={
                "alerts": alerts,
                "count": count,
                "date": datetime.now(IST).strftime("%Y-%m-%d"),
            },
        )

    except Exception as e:
        logger.error(f"[MCP] get_today_alerts failed: {e}")
        return ToolResponse(
            tool="get_today_alerts",
            status="error",
            error=str(e),
        )
