"""
SwingAdvisorBot — Production Entry Point
main.py — Railway / cloud deployment entry point

Launches the unified FastAPI server on 0.0.0.0 using Railway's
PORT environment variable (falls back to 8001 for local dev).

All modules are loaded via module1_data_layer.mcp_server which
already mounts M2–M7 routers and starts the report scheduler.

Usage:
  python main.py                  # direct (local / Railway)
  uvicorn main:app --port 8001    # uvicorn CLI
"""

from __future__ import annotations

import os

import uvicorn

# Import the unified FastAPI app (mounts all M1–M7 routers)
from module1_data_layer.mcp_server import app  # noqa: F401  (re-exported as main:app)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8001))
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=port,
        reload=False,
        log_level="info",
    )
