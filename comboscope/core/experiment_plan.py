from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

import yaml

from comboscope.core.schemas import ExperimentPlan, is_editable_file_allowed
from comboscope.core.train_policy import update_train_policy


def _load_plan(path: str | Path) -> ExperimentPlan:
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    return ExperimentPlan.model_validate(raw)


def _backup(project_dir: Path, rel_paths: list[str]) -> dict[str, Any]:
    backup_dir = project_dir / ".comboscope_backups"
    backup_dir.mkdir(parents=True, exist_ok=True)
    files: list[dict[str, str]] = []
    for rel in rel_paths:
        target = project_dir / rel
        backup_path = backup_dir / rel.replace("/", "__")
        if target.exists():
            shutil.copy2(target, backup_path)
        else:
            backup_path.write_text("", encoding="utf-8")
        files.append({"target_path": target.as_posix(), "backup_path": backup_path.as_posix()})
    return {"files": files}


def _update_feature_config(path: Path, feature_names: list[str]) -> None:
    data = yaml.safe_load(path.read_text(encoding="utf-8")) if path.exists() else {}
    if not isinstance(data, dict):
        data = {}
    existing = data.get("enabled_features") or []
    enabled = list(dict.fromkeys([*existing, *feature_names]))
    data["enabled_features"] = enabled
    path.write_text(yaml.safe_dump(data, allow_unicode=True, sort_keys=False), encoding="utf-8")


def _update_feature_policy(path: Path, feature_names: list[str]) -> None:
    lines = ["ENABLED_FEATURES = ["]
    for name in feature_names:
        lines.append(f'    "{name}",')
    lines.append("]\n")
    path.write_text("\n".join(lines), encoding="utf-8")


def apply_experiment_plan(plan_path: str | Path, project_dir: str | Path) -> dict[str, Any]:
    project = Path(project_dir)
    plan = _load_plan(plan_path)
    invalid = [path for path in plan.editable_files if not is_editable_file_allowed(path)]
    if invalid:
        raise ValueError(f"editable file not allowed: {invalid}")
    backup_info = _backup(project, plan.editable_files)
    feature_names = [change.feature_name for change in plan.changes]
    if "benchmark/feature_config.yaml" in plan.editable_files:
        _update_feature_config(project / "benchmark" / "feature_config.yaml", feature_names)
    if "benchmark/feature_policy.py" in plan.editable_files:
        _update_feature_policy(project / "benchmark" / "feature_policy.py", feature_names)
    if "benchmark/train.py" in plan.editable_files:
        update_train_policy(project / "benchmark" / "train.py", feature_names)
    return {
        "success": True,
        "trial_id": plan.trial_id,
        "features": feature_names,
        "backup_info": backup_info,
        "plan": plan.model_dump(),
    }


def write_plan(path: str | Path, plan: dict[str, Any]) -> None:
    Path(path).write_text(yaml.safe_dump(plan, allow_unicode=True, sort_keys=False), encoding="utf-8")


def write_review(path: str | Path, review: dict[str, Any]) -> None:
    Path(path).write_text(json.dumps(review, ensure_ascii=False, indent=2), encoding="utf-8")
