"""ProjectAdapter：把"任意 Python 项目"抽象成 RoutePilot 能驱动的四个动作。"""

from __future__ import annotations

import shutil
import tempfile
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path

from routepilot.adapter.manifest import MetricsSpec, ProjectManifest, is_editable

# 快照/回滚时跳过的重目录，避免把仓库级缓存也纳入可编辑面扫描。
_SCAN_SKIP_DIRS = {
    ".git",
    ".venv",
    "venv",
    "__pycache__",
    ".pytest_cache",
    "node_modules",
    ".mypy_cache",
    ".ruff_cache",
}


@dataclass
class Discovery:
    """接入探测结果：解析出的 manifest + 待人工确认项。"""

    manifest: ProjectManifest
    warnings: list[str] = field(default_factory=list)
    confidence: float = 1.0


@dataclass
class RunResult:
    """一次训练/预测执行的原始结果，不含任何指标语义。"""

    success: bool
    returncode: int
    command: list[str]
    cwd: str
    duration_sec: float
    stdout_path: str | None = None
    stderr_path: str | None = None
    prediction_path: str | None = None
    actual_path: str | None = None
    error: str = ""


@dataclass
class Snapshot:
    """editable 面在改动前的备份，用于 rollback。"""

    root: str
    files: list[str] = field(default_factory=list)


class ProjectAdapter(ABC):
    """接入协议的运行时契约。

    `discover` / `run` / `snapshot` / `rollback` 由具体项目实现；`metrics` 默认
    直接来自 manifest，只有指标需要特殊计算时才覆盖。
    """

    def __init__(self, manifest: ProjectManifest, project_root: str | Path):
        self.manifest = manifest
        self.project_root = Path(project_root).resolve()

    @abstractmethod
    def discover(self) -> Discovery:
        """探测项目结构并返回本次接入所使用的 manifest。"""

    @abstractmethod
    def run(self, *, env: dict[str, str] | None = None) -> RunResult:
        """执行一次训练/预测，返回原始运行结果。"""

    def metrics(self) -> MetricsSpec:
        return self.manifest.metrics

    def editable_files(self) -> list[str]:
        """当前磁盘上真实存在、且允许被 Agent 修改的相对路径。"""
        return [
            relative
            for relative in self._walk_project()
            if is_editable(relative, self.manifest)
        ]

    def snapshot(self) -> Snapshot:
        snapshot_root = Path(tempfile.mkdtemp(prefix="routepilot-snapshot-"))
        files: list[str] = []
        for relative in self.editable_files():
            destination = snapshot_root / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(self.project_root / relative, destination)
            files.append(relative)
        return Snapshot(root=str(snapshot_root), files=files)

    def rollback(self, snapshot: Snapshot) -> None:
        """恢复 editable 面：删除新增文件，还原被改动文件。"""
        keep = set(snapshot.files)
        for relative in self.editable_files():
            if relative not in keep:
                (self.project_root / relative).unlink(missing_ok=True)
        for relative in snapshot.files:
            source = Path(snapshot.root) / relative
            if not source.is_file():
                continue
            destination = self.project_root / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, destination)

    def _walk_project(self) -> list[str]:
        root = self.project_root
        if not root.is_dir():
            return []
        found: list[str] = []
        for path in root.rglob("*"):
            if not path.is_file():
                continue
            relative = path.relative_to(root).as_posix()
            if any(part in _SCAN_SKIP_DIRS for part in path.relative_to(root).parts[:-1]):
                continue
            found.append(relative)
        return found
