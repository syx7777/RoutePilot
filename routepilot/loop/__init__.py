"""RoutePilot 通用优化闭环。"""

from __future__ import annotations

from routepilot.loop.proposer import (
    FileEdit,
    LlmProposer,
    Proposal,
    ProposalContext,
    Proposer,
    ScriptedProposer,
    parse_proposal,
    validate_proposal,
)
from routepilot.loop.report import render_markdown, write_report
from routepilot.loop.trial import OptimizationOutcome, TrialRecord, run_optimization

__all__ = [
    "FileEdit",
    "LlmProposer",
    "OptimizationOutcome",
    "Proposal",
    "ProposalContext",
    "Proposer",
    "ScriptedProposer",
    "TrialRecord",
    "parse_proposal",
    "render_markdown",
    "run_optimization",
    "validate_proposal",
    "write_report",
]
