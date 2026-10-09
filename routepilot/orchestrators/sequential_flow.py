from __future__ import annotations

from typing import Any

from routepilot.orchestrators.langgraph_flow import run_once


def run_once_sequential(request: dict[str, Any]) -> dict[str, Any]:
    return run_once(request)
