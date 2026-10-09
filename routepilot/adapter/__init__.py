"""RoutePilot 接入层：声明式 manifest + 通用适配器。"""

from __future__ import annotations

from routepilot.adapter.base import Discovery, ProjectAdapter, RunResult, Snapshot
from routepilot.adapter.command import CommandProjectAdapter, build_adapter
from routepilot.adapter.manifest import (
    ArtifactsSpec,
    BudgetSpec,
    ColumnsSpec,
    GuardMetricSpec,
    MetricSpec,
    MetricsSpec,
    ProjectManifest,
    ProjectSpec,
    RunSpec,
    SplitSpec,
    ValidationReport,
    classify_path,
    is_editable,
    load_manifest,
    matches_glob,
    resolve_project_root,
    validate_manifest,
)

__all__ = [
    "ArtifactsSpec",
    "BudgetSpec",
    "ColumnsSpec",
    "CommandProjectAdapter",
    "Discovery",
    "GuardMetricSpec",
    "MetricSpec",
    "MetricsSpec",
    "ProjectAdapter",
    "ProjectManifest",
    "ProjectSpec",
    "RunResult",
    "RunSpec",
    "Snapshot",
    "SplitSpec",
    "ValidationReport",
    "build_adapter",
    "classify_path",
    "is_editable",
    "load_manifest",
    "matches_glob",
    "resolve_project_root",
    "validate_manifest",
]
