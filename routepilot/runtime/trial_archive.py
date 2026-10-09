from __future__ import annotations

import json
import os
import shutil
from pathlib import Path
from typing import Any


ARCHIVE_GROUPS: dict[str, list[str]] = {
    "inputs": [
        "data",
        "data/input_manifest.json",
        "standardized/prediction.csv",
        "standardized/actual.csv",
        "standardized_prediction.csv",
        "standardized_actual.csv",
    ],
    "outputs": [
        "outputs",
        "outputs/real_outputs",
        "prediction.csv",
        "new_metrics.json",
        "new_metrics_summary.csv",
    ],
    "agent1": [
        "agent1_program.md",
        "agent1_skill_route.yaml",
        "analysis_report.md",
        "artifact_contract.json",
        "badcase_diagnosis.md",
        "feature_hypothesis.yaml",
        "candidate_experiments.yaml",
        "experiment_plan.yaml",
        "problem_context.json",
    ],
    "agent2": [
        "agent2_execution_plan.yaml",
        "source_evaluation_context.json",
        "output_contract.json",
        "code",
        "code/train.py",
        "code/agent2_code_modification.yaml",
        "run_status.json",
        "experiment_review.md",
        "review_result.json",
    ],
    "evaluation": [
        "metrics.json",
        "metrics_summary.csv",
        "metric_comparison.json",
        "scene_metrics.csv",
        "badcases.csv",
        "badcase_summary.json",
        "anomaly_summary.json",
        "column_mapping.json",
    ],
    "reports": [
        "forecast_report.md",
        "optimization_suggestions.md",
        "report_context.json",
        "final_report_context.json",
        "final_report.md",
        "artifact_index.md",
    ],
    "logs": [
        "logs",
        "logs/train.log",
        "logs/eval.log",
        "train.log",
        "eval.log",
    ],
    "audit": [
        "agent_status.json",
        "agent_timeline.jsonl",
        "llm_calls.jsonl",
        "audit/llm_yaml_failures",
        "audit/llm_raw_responses",
        "token_usage.json",
        "trace.jsonl",
        "error_report.md",
        "error_report.json",
        "scan_result.json",
        "artifact_summary.json",
        "artifact_contract.json",
        "artifact_manifest.json",
        "code_analysis.json",
        "log_summary.json",
    ],
}

COMPACT_GROUPS: dict[str, list[str]] = {
    "agent1": [
        "agent1_program.md",
        "agent1_skill_route.yaml",
        "analysis_report.md",
        "artifact_contract.json",
        "badcase_diagnosis.md",
        "feature_hypothesis.yaml",
        "candidate_experiments.yaml",
        "experiment_plan.yaml",
        "problem_context.json",
    ],
    "agent2": [
        "agent2_execution_plan.yaml",
        "source_evaluation_context.json",
        "output_contract.json",
        "run_status.json",
        "experiment_review.md",
        "review_result.json",
    ],
    "evaluation": [
        "metrics.json",
        "metrics_summary.csv",
        "new_metrics.json",
        "new_metrics_summary.csv",
        "metric_comparison.json",
        "scene_metrics.csv",
        "badcases.csv",
        "badcase_summary.json",
        "anomaly_summary.json",
        "column_mapping.json",
    ],
    "reports": [
        "forecast_report.md",
        "optimization_suggestions.md",
        "report_context.json",
        "final_report_context.json",
    ],
    "audit": [
        "agent_status.json",
        "agent_timeline.jsonl",
        "llm_calls.jsonl",
        "audit/llm_raw_responses",
        "token_usage.json",
        "trace.jsonl",
        "scan_result.json",
        "artifact_summary.json",
        "artifact_contract.json",
        "artifact_manifest.json",
        "code_analysis.json",
        "log_summary.json",
    ],
}

ROOT_COPIES: dict[str, str] = {
    "final_report.md": "reports/final_report.md",
    "artifact_index.md": "reports/artifact_index.md",
}


def archive_trial_files(trial_dir: str | Path) -> dict[str, Any]:
    trial = Path(trial_dir)
    compact_manifest = compact_trial_files(trial)
    archive = trial / "archive"
    archive.mkdir(parents=True, exist_ok=True)
    manifest: dict[str, Any] = {"archive_dir": archive.as_posix(), "compact_layout": compact_manifest, "groups": {}}
    for group, rel_paths in ARCHIVE_GROUPS.items():
        group_dir = archive / group
        group_dir.mkdir(parents=True, exist_ok=True)
        archived: list[dict[str, str]] = []
        for rel_path in rel_paths:
            source = _resolve_compact_source(trial, rel_path)
            if not source.exists():
                continue
            destination = group_dir / source.name
            _link_or_copy(source, destination)
            archived.append({"source": source.as_posix(), "archive_path": destination.as_posix()})
        manifest["groups"][group] = archived
    (archive / "archive_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return manifest


def compact_trial_files(trial_dir: str | Path) -> dict[str, Any]:
    trial = Path(trial_dir)
    manifest: dict[str, Any] = {"layout": {}, "root_kept": sorted(ROOT_COPIES)}
    for group, filenames in COMPACT_GROUPS.items():
        group_dir = trial / group
        moved: list[dict[str, str]] = []
        for filename in filenames:
            source = trial / filename
            if not source.exists():
                continue
            destination = _unique_destination(group_dir / source.name)
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(source.as_posix(), destination.as_posix())
            moved.append({"from": source.as_posix(), "to": destination.as_posix()})
        if moved:
            manifest["layout"][group] = moved

    _compact_root_logs(trial, manifest)
    _compact_root_standardized(trial, manifest)
    _copy_root_entry_files(trial, manifest)
    return manifest


def _compact_root_logs(trial: Path, manifest: dict[str, Any]) -> None:
    moved: list[dict[str, str]] = []
    logs = trial / "logs"
    for filename in ("train.log", "eval.log"):
        source = trial / filename
        if not source.exists():
            continue
        destination = _unique_destination(logs / filename)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(source.as_posix(), destination.as_posix())
        moved.append({"from": source.as_posix(), "to": destination.as_posix()})
    if moved:
        manifest["layout"].setdefault("logs", []).extend(moved)


def _compact_root_standardized(trial: Path, manifest: dict[str, Any]) -> None:
    moved: list[dict[str, str]] = []
    standardized = trial / "standardized"
    for filename in ("standardized_prediction.csv", "standardized_actual.csv"):
        source = trial / filename
        if not source.exists():
            continue
        destination = _unique_destination(standardized / filename)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(source.as_posix(), destination.as_posix())
        moved.append({"from": source.as_posix(), "to": destination.as_posix()})
    if moved:
        manifest["layout"].setdefault("standardized", []).extend(moved)


def _copy_root_entry_files(trial: Path, manifest: dict[str, Any]) -> None:
    copied: list[dict[str, str]] = []
    for root_name, grouped_rel in ROOT_COPIES.items():
        root_path = trial / root_name
        grouped_path = trial / grouped_rel
        if root_path.exists() and not grouped_path.exists():
            grouped_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(root_path, grouped_path)
            copied.append({"from": root_path.as_posix(), "to": grouped_path.as_posix()})
    if copied:
        manifest["layout"].setdefault("root_entry_copies", []).extend(copied)


def _resolve_compact_source(trial: Path, rel_path: str) -> Path:
    source = trial / rel_path
    if source.exists():
        return source
    filename = Path(rel_path).name
    for group in [*COMPACT_GROUPS, "logs", "standardized", "reports"]:
        candidate = trial / group / filename
        if candidate.exists():
            return candidate
    return source


def _unique_destination(destination: Path) -> Path:
    if not destination.exists():
        return destination
    stem = destination.stem
    suffix = destination.suffix
    parent = destination.parent
    index = 2
    while True:
        candidate = parent / f"{stem}_{index}{suffix}"
        if not candidate.exists():
            return candidate
        index += 1


def _link_or_copy(source: Path, destination: Path) -> None:
    if destination.exists() or destination.is_symlink():
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        relative_source = os.path.relpath(source, destination.parent)
        destination.symlink_to(relative_source, target_is_directory=source.is_dir())
    except OSError:
        if source.is_dir():
            shutil.copytree(source, destination, dirs_exist_ok=True)
        else:
            shutil.copy2(source, destination)
