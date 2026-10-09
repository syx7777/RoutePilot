"""RoutePilot 接入协议：声明式 manifest + 路径护栏。

这一层只做"描述与校验"，不执行任何训练、评测或 LLM 调用。
"""

from __future__ import annotations

import fnmatch
import shutil
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field, ValidationError, field_validator, model_validator


class ProjectSpec(BaseModel):
    name: str
    root: str = "."
    entrypoint: str | None = None

    @field_validator("name")
    @classmethod
    def _name_required(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("project.name is required")
        return value


class RunSpec(BaseModel):
    command: list[str] = Field(min_length=1)
    cwd: str = "."
    timeout_sec: int = Field(default=1800, gt=0)
    env: dict[str, str] = Field(default_factory=dict)
    outputs: str | None = None

    @field_validator("command")
    @classmethod
    def _argv_tokens(cls, value: list[str]) -> list[str]:
        tokens = [str(item) for item in value]
        for token in tokens:
            if not token or token != token.strip():
                raise ValueError("run.command entries must be non-empty argv tokens without padding")
        return tokens


class ColumnsSpec(BaseModel):
    model_config = {"extra": "allow"}

    prediction: str
    actual: str
    date: str | None = None
    id: list[str] = Field(default_factory=list)


class SplitSpec(BaseModel):
    column: str
    value: str


class ArtifactsSpec(BaseModel):
    prediction: str
    actual: str
    columns: ColumnsSpec
    split: SplitSpec | None = None


class MetricSpec(BaseModel):
    name: str
    direction: Literal["minimize", "maximize"] = "minimize"
    min_delta: float = 0.0

    @field_validator("min_delta")
    @classmethod
    def _non_negative(cls, value: float) -> float:
        if value < 0:
            raise ValueError("metrics.primary.min_delta must be >= 0")
        return value


class GuardMetricSpec(BaseModel):
    name: str
    max_regression: float = Field(ge=0)


class MetricsSpec(BaseModel):
    primary: MetricSpec
    guards: list[GuardMetricSpec] = Field(default_factory=list)


class BudgetSpec(BaseModel):
    max_trials: int = Field(default=3, gt=0)
    max_usd: float | None = Field(default=None, gt=0)
    latency_p95_ms: int | None = Field(default=None, gt=0)


class ProjectManifest(BaseModel):
    project: ProjectSpec
    run: RunSpec
    artifacts: ArtifactsSpec
    metrics: MetricsSpec
    editable: list[str] = Field(min_length=1)
    protected: list[str] = Field(default_factory=list)
    budget: BudgetSpec = Field(default_factory=BudgetSpec)

    @model_validator(mode="after")
    def _editable_not_shadowed_by_protected(self) -> "ProjectManifest":
        shadowed = [
            pattern
            for pattern in self.editable
            if any(matches_glob(pattern, guard) for guard in self.protected)
        ]
        if shadowed:
            raise ValueError(
                "editable patterns are shadowed by protected patterns: " f"{shadowed}"
            )
        return self


class ValidationReport(BaseModel):
    ok: bool
    errors: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)


def matches_glob(rel_path: str, pattern: str) -> bool:
    """按路径段匹配 glob，`*` 不跨越 `/`，`**` 匹配任意层。

    与 fnmatch 的差异：本函数遵循文件系统语义，`configs/*.yaml` 不会匹配
    `configs/sub/a.yaml`，避免 protected 规则被误放宽。
    """
    rel = rel_path.replace("\\", "/").strip("/")
    pat = pattern.replace("\\", "/").strip("/")
    if not pat:
        return False
    if "/" not in pat:
        return fnmatch.fnmatch(rel.rsplit("/", 1)[-1], pat)
    return _match_parts(rel.split("/"), pat.split("/"))


def _match_parts(rel_parts: list[str], pat_parts: list[str]) -> bool:
    if not pat_parts:
        return not rel_parts
    head, rest = pat_parts[0], pat_parts[1:]
    if head == "**":
        for index in range(len(rel_parts) + 1):
            if _match_parts(rel_parts[index:], rest):
                return True
        return False
    if not rel_parts:
        return False
    if not fnmatch.fnmatch(rel_parts[0], head):
        return False
    return _match_parts(rel_parts[1:], rest)


def classify_path(
    rel_path: str, editable: list[str], protected: list[str]
) -> Literal["protected", "editable", "unlisted"]:
    """判定一个相对路径是否允许被 Agent 修改。protected 优先于 editable。"""
    if any(matches_glob(rel_path, pattern) for pattern in protected):
        return "protected"
    if any(matches_glob(rel_path, pattern) for pattern in editable):
        return "editable"
    return "unlisted"


def is_editable(rel_path: str, manifest: ProjectManifest) -> bool:
    return classify_path(rel_path, manifest.editable, manifest.protected) == "editable"


def load_manifest(path: str | Path) -> ProjectManifest:
    manifest_path = Path(path)
    try:
        raw = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise ValueError(f"manifest is not valid YAML: {manifest_path}: {exc}") from exc
    if not isinstance(raw, dict):
        raise ValueError(f"manifest must be a YAML mapping: {manifest_path}")
    try:
        return ProjectManifest.model_validate(raw)
    except ValidationError as exc:
        raise ValueError(f"manifest failed schema validation: {manifest_path}\n{exc}") from exc


def resolve_project_root(manifest: ProjectManifest, manifest_path: str | Path) -> Path:
    base = Path(manifest_path).resolve().parent
    return (base / manifest.project.root).resolve()


def validate_manifest(manifest: ProjectManifest, project_root: str | Path) -> ValidationReport:
    """校验 manifest 与实际目录是否一致。

    只做"接入前可判定"的检查：入口是否存在、可编辑面是否为空。产物文件由 run
    生成，因此缺失时只记为 warning，不算 error。
    """
    root = Path(project_root)
    errors: list[str] = []
    warnings: list[str] = []

    if not root.is_dir():
        errors.append(f"project root does not exist or is not a directory: {root}")
        return ValidationReport(ok=False, errors=errors, warnings=warnings)

    run_cwd = (root / manifest.run.cwd).resolve()
    if not run_cwd.is_dir():
        errors.append(f"run.cwd does not exist: {manifest.run.cwd}")

    if manifest.project.entrypoint:
        entrypoint = (root / manifest.project.entrypoint).resolve()
        if not entrypoint.is_file():
            errors.append(f"project.entrypoint does not exist: {manifest.project.entrypoint}")

    for label, rel in (
        ("artifacts.prediction", manifest.artifacts.prediction),
        ("artifacts.actual", manifest.artifacts.actual),
    ):
        if not (root / rel).exists():
            warnings.append(f"{label} not found yet (expected after run): {rel}")

    executable = _resolve_executable(manifest.run.command[0])
    if executable is None:
        warnings.append(
            f"run.command executable not found on PATH: {manifest.run.command[0]}"
        )

    return ValidationReport(ok=not errors, errors=errors, warnings=warnings)


def _resolve_executable(token: str) -> str | None:
    if Path(token).is_file():
        return token
    return shutil.which(token)
