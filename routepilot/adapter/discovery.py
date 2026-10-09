"""自动发现：扫描一个已有项目，产出可审阅的 manifest 草稿。

设计原则（对应设计文档 §1.4 的边界）：
- 只做"接入前可判定"的推断，不执行任何训练或 LLM 调用；
- 每个推断都带证据，拿不准就放进 `questions` 交给人工确认；
- 不声称零配置——`init` 的产物必须经人审阅后 `validate` 通过才能 `run`。
"""

from __future__ import annotations

import csv
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

IGNORED_DIRS = {
    ".git",
    ".venv",
    "venv",
    "__pycache__",
    ".pytest_cache",
    ".mypy_cache",
    ".ruff_cache",
    "node_modules",
    "archive",
    "backups",
    ".routepilot_backups",
    ".idea",
    ".vscode",
    "dist",
    "build",
}
ENTRYPOINT_NAMES = {"main.py", "train.py", "run.py", "predict.py", "train_forecast.py", "pipeline.py"}
ENTRYPOINT_HINTS = ("train", "run", "predict", "forecast", "pipeline")
CONFIG_SUFFIXES = {".yaml", ".yml"}
DATA_SUFFIXES = {".csv", ".parquet"}
CONFIG_DIR_NAMES = {"config", "configs", "conf", "settings"}
# RoutePilot 自己产出的文件不能反过来被当作业务配置/数据。
ROUTEPILOT_ARTIFACTS = {"routepilot.yaml", "routepilot.yml", "routepilot.discovery.json"}
MAX_DEPTH = 4
MAX_HEADER_BYTES = 8192

PREDICTION_HINTS = ("prediction", "pred", "yhat", "forecast", "predict_value", "预测")
ACTUAL_HINTS = ("actual", "y_true", "truth", "ground", "label", "target", "sale", "demand", "y", "真实", "标签", "销量")
DATE_HINTS = ("ds", "date", "datetime", "timestamp", "日期", "时间")
ID_HINTS = ("unique_id", "item_id", "series_id", "entity_id", "sku", "store", "id")
FEATURE_HINTS = ("feature", "features")
# 子串匹配的最短长度：避免 "y" 这类单字符提示词误命中 "yhat"、"days" 等列。
MIN_SUBSTRING_HINT = 3


@dataclass
class ManifestDraft:
    manifest: dict[str, Any]
    confidence: float
    questions: list[str] = field(default_factory=list)
    evidence: dict[str, Any] = field(default_factory=dict)


def _iter_files(root: Path, max_depth: int = MAX_DEPTH) -> list[Path]:
    found: list[Path] = []
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        relative = path.relative_to(root)
        if any(part in IGNORED_DIRS for part in relative.parts[:-1]):
            continue
        if len(relative.parts) - 1 > max_depth:
            continue
        if relative.name in ROUTEPILOT_ARTIFACTS:
            continue
        found.append(path)
    return sorted(found)


def _config_rank(path: Path, root: Path) -> tuple[int, int, str]:
    """优先选择 configs/ 之类的专用目录下的配置，其次才是更浅的路径。"""
    relative = path.relative_to(root)
    in_config_dir = (
        0 if len(relative.parts) > 1 and relative.parts[0].lower() in CONFIG_DIR_NAMES else 1
    )
    return (in_config_dir, len(relative.parts), relative.name)


def _read_head(path: Path, limit: int = 60) -> str:
    try:
        raw = path.open("rb").read(MAX_HEADER_BYTES)
    except OSError:
        return ""
    return "\n".join(raw.decode("utf-8", errors="ignore").splitlines()[:limit])


def _read_csv_columns(path: Path) -> list[str]:
    for encoding in ("utf-8-sig", "utf-8", "gb18030", "gbk", "latin1"):
        try:
            with path.open(newline="", encoding=encoding) as handle:
                reader = csv.reader(handle)
                return [str(item).lstrip("\ufeff").strip() for item in next(reader)]
        except (UnicodeDecodeError, StopIteration):
            continue
        except OSError:
            return []
    return []


def _argparse_flags(source_head: str) -> list[str]:
    return sorted(set(re.findall(r'add_argument\(\s*["\'](--[A-Za-z0-9_\-]+)["\']', source_head)))


def _pick_column(columns: list[str], hints: tuple[str, ...]) -> str | None:
    lowered = {column.lower(): column for column in columns}
    for hint in hints:
        if hint in lowered:
            return lowered[hint]
    for hint in hints:
        if len(hint) < MIN_SUBSTRING_HINT:
            continue
        for low, original in lowered.items():
            if hint in low:
                return original
    return None


def _looks_like_entrypoint(path: Path, head: str) -> bool:
    name = path.name.lower()
    if name in ENTRYPOINT_NAMES:
        return True
    if any(hint in name for hint in ENTRYPOINT_HINTS) and ("__main__" in head or "argparse" in head):
        return True
    return "argparse" in head and "__main__" in head


def _classify(columns: list[str], path: Path) -> str | None:
    stem = path.stem.lower()
    has_prediction = _pick_column(columns, PREDICTION_HINTS) is not None
    has_actual = _pick_column(columns, ACTUAL_HINTS) is not None
    name_is_prediction = any(hint in stem for hint in PREDICTION_HINTS)
    name_is_actual = any(hint in stem for hint in ACTUAL_HINTS)
    if has_prediction and (name_is_prediction or not has_actual):
        return "prediction"
    if has_actual and (name_is_actual or not has_prediction):
        return "actual"
    return None


def draft_manifest(
    project_root: str | Path,
    *,
    project_name: str | None = None,
    entrypoint: str | None = None,
) -> ManifestDraft:
    root = Path(project_root).resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"project root does not exist: {root}")

    questions: list[str] = []
    evidence: dict[str, Any] = {
        "scanned_root": root.as_posix(),
        "entrypoint_candidates": [],
        "config_files": [],
        "data_files": [],
        "csv_columns": {},
    }

    python_files: list[Path] = []
    config_files: list[Path] = []
    data_files: list[Path] = []
    for path in _iter_files(root):
        relative = path.relative_to(root).as_posix()
        suffix = path.suffix.lower()
        if suffix == ".py":
            python_files.append(path)
        elif suffix in CONFIG_SUFFIXES:
            config_files.append(path)
        elif suffix in DATA_SUFFIXES:
            data_files.append(path)
            if suffix == ".csv":
                columns = _read_csv_columns(path)
                if columns:
                    evidence["csv_columns"][relative] = columns

    entrypoint_candidates = [
        path for path in python_files if _looks_like_entrypoint(path, _read_head(path))
    ]
    evidence["entrypoint_candidates"] = [
        path.relative_to(root).as_posix() for path in entrypoint_candidates
    ]
    evidence["config_files"] = [path.relative_to(root).as_posix() for path in config_files]
    evidence["data_files"] = [path.relative_to(root).as_posix() for path in data_files]

    if entrypoint is None:
        if entrypoint_candidates:
            chosen = sorted(entrypoint_candidates, key=lambda item: (len(item.parts), item.name))[0]
            entrypoint = chosen.relative_to(root).as_posix()
        else:
            questions.append("未找到训练入口，请手动指定 project.entrypoint 与 run.command")
    if len(entrypoint_candidates) > 1:
        questions.append(
            "发现多个入口候选，请确认 project.entrypoint: "
            + ", ".join(path.relative_to(root).as_posix() for path in entrypoint_candidates)
        )

    prediction_file: Path | None = None
    actual_file: Path | None = None
    for path in data_files:
        columns = evidence["csv_columns"].get(path.relative_to(root).as_posix())
        if not columns:
            continue
        kind = _classify(columns, path)
        if kind == "prediction" and prediction_file is None:
            prediction_file = path
        elif kind == "actual" and actual_file is None:
            actual_file = path

    if prediction_file is None:
        questions.append("未识别到预测产物 CSV，请确认 artifacts.prediction")
    if actual_file is None:
        questions.append("未识别到真实值 CSV，请确认 artifacts.actual")

    prediction_columns = (
        evidence["csv_columns"].get(prediction_file.relative_to(root).as_posix(), []) if prediction_file else []
    )
    actual_columns = (
        evidence["csv_columns"].get(actual_file.relative_to(root).as_posix(), []) if actual_file else []
    )
    date_column = _pick_column(prediction_columns, DATE_HINTS) or _pick_column(actual_columns, DATE_HINTS)
    prediction_column = _pick_column(prediction_columns, PREDICTION_HINTS)
    actual_column = _pick_column(actual_columns, ACTUAL_HINTS)
    id_column = _pick_column(prediction_columns, ID_HINTS)

    for label, value in (
        ("prediction_column", prediction_column),
        ("actual_column", actual_column),
    ):
        if value is None:
            questions.append(f"未能从表头推断 {label}，请确认 artifacts.columns")

    command = _draft_command(root, entrypoint, config_files, questions)
    editable, protected = _draft_editable(root, python_files, config_files, entrypoint, data_files)

    manifest: dict[str, Any] = {
        "project": {
            "name": project_name or root.name,
            "root": ".",
            "entrypoint": entrypoint,
        },
        "run": {"command": command, "cwd": ".", "timeout_sec": 1800, "outputs": str(_output_dir(prediction_file, root))},
        "artifacts": {
            "prediction": prediction_file.relative_to(root).as_posix() if prediction_file else "",
            "actual": actual_file.relative_to(root).as_posix() if actual_file else "",
            "columns": {
                "prediction": prediction_column or "",
                "actual": actual_column or "",
                "date": date_column,
                "id": [id_column] if id_column else [],
            },
        },
        "metrics": {
            "primary": {"name": "wape", "direction": "minimize", "min_delta": 0.005},
            "guards": [{"name": "bias", "max_regression": 0.02}],
        },
        "editable": editable,
        "protected": protected,
        "budget": {"max_trials": 3, "max_usd": 5.0, "latency_p95_ms": 60000},
    }

    resolved = sum(
        1
        for value in (entrypoint, prediction_file, actual_file, prediction_column, actual_column, date_column)
        if value
    )
    confidence = round(resolved / 6, 2)
    if not editable:
        questions.append("未推断出可编辑文件，请手工补充 editable")
    evidence["resolved_fields"] = resolved

    return ManifestDraft(manifest=manifest, confidence=confidence, questions=questions, evidence=evidence)


def _output_dir(prediction_file: Path | None, root: Path) -> str:
    if prediction_file is None:
        return "outputs"
    relative = prediction_file.relative_to(root)
    return relative.parent.as_posix() if len(relative.parts) > 1 else "."


def _draft_command(root: Path, entrypoint: str | None, config_files: list[Path], questions: list[str]) -> list[str]:
    if not entrypoint:
        return ["python", "<entrypoint>"]
    command = ["python", entrypoint]
    source = (root / entrypoint).read_text(encoding="utf-8", errors="ignore")
    flags = _argparse_flags(source)
    if "--config" in flags and config_files:
        preferred = sorted(config_files, key=lambda item: _config_rank(item, root))[0]
        command += ["--config", preferred.relative_to(root).as_posix()]
    elif "--output_dir" in flags or "--output-dir" in flags:
        questions.append("入口支持输出目录参数，请确认 run.command 是否需要显式传入")
    return command


def _draft_editable(
    root: Path,
    python_files: list[Path],
    config_files: list[Path],
    entrypoint: str | None,
    data_files: list[Path],
) -> tuple[list[str], list[str]]:
    editable: list[str] = []
    config_dirs = {path.relative_to(root).parent.as_posix() for path in config_files}
    for directory in sorted(config_dirs):
        editable.append(f"{directory}/*.yaml" if directory != "." else "*.yaml")
    for path in python_files:
        relative = path.relative_to(root)
        if relative.as_posix() == entrypoint:
            continue
        if any(hint in path.stem.lower() for hint in FEATURE_HINTS):
            editable.append(relative.as_posix())

    protected: list[str] = []
    data_dirs = {path.relative_to(root).parent.as_posix() for path in data_files}
    for directory in sorted(data_dirs):
        protected.append(f"{directory}/**" if directory != "." else "**/*.csv")
    if entrypoint:
        protected.append(entrypoint)
    protected.append("scripts/**")
    return editable, sorted(set(protected))
