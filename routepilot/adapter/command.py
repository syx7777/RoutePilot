"""默认适配器：用 manifest 里的 argv 直接驱动子进程。

覆盖绝大多数"已有 train.py / predict.py 入口"的项目，不需要写代码即可接入。
"""

from __future__ import annotations

import os
import subprocess
import time
from pathlib import Path

from routepilot.adapter.base import Discovery, ProjectAdapter, RunResult
from routepilot.adapter.manifest import ProjectManifest, validate_manifest
from routepilot.runtime.interpreter import project_interpreter, resolve_interpreter_token

_LOG_TAIL_LIMIT = 4000


class CommandProjectAdapter(ProjectAdapter):
    def __init__(
        self,
        manifest: ProjectManifest,
        project_root: str | Path,
        *,
        log_dir: str | Path | None = None,
    ):
        super().__init__(manifest, project_root)
        self.log_dir = Path(log_dir).resolve() if log_dir else self.project_root / "runs" / "routepilot-logs"

    def discover(self) -> Discovery:
        report = validate_manifest(self.manifest, self.project_root)
        warnings = [*report.errors, *report.warnings]
        confidence = 1.0 if report.ok else 0.0
        return Discovery(manifest=self.manifest, warnings=warnings, confidence=confidence)

    def run(self, *, env: dict[str, str] | None = None) -> RunResult:
        run_spec = self.manifest.run
        cwd = (self.project_root / run_spec.cwd).resolve()
        command = self._resolve_command(run_spec.command)
        process_env = {**os.environ, **run_spec.env, **(env or {})}
        self.log_dir.mkdir(parents=True, exist_ok=True)
        stdout_path = self.log_dir / "run.stdout.log"
        stderr_path = self.log_dir / "run.stderr.log"
        started = time.perf_counter()
        try:
            completed = subprocess.run(
                command,
                cwd=str(cwd),
                env=process_env,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=run_spec.timeout_sec,
                check=False,
            )
        except subprocess.TimeoutExpired as exc:
            duration = time.perf_counter() - started
            stdout_path.write_text(exc.stdout or "", encoding="utf-8")
            stderr_path.write_text(exc.stderr or "", encoding="utf-8")
            return RunResult(
                success=False,
                returncode=-1,
                command=command,
                cwd=str(cwd),
                duration_sec=duration,
                stdout_path=str(stdout_path),
                stderr_path=str(stderr_path),
                error=f"run timed out after {run_spec.timeout_sec}s",
            )
        duration = time.perf_counter() - started
        stdout_path.write_text(completed.stdout or "", encoding="utf-8")
        stderr_path.write_text(completed.stderr or "", encoding="utf-8")
        success = completed.returncode == 0
        return RunResult(
            success=success,
            returncode=completed.returncode,
            command=command,
            cwd=str(cwd),
            duration_sec=duration,
            stdout_path=str(stdout_path),
            stderr_path=str(stderr_path),
            prediction_path=self._resolve_artifact(self.manifest.artifacts.prediction),
            actual_path=self._resolve_artifact(self.manifest.artifacts.actual),
            error="" if success else self._tail(completed.stderr),
        )

    def _resolve_command(self, command: list[str]) -> list[str]:
        """把裸的 `python` 记号替换为项目自己的解释器，显式路径则保持原样。"""
        resolved = [str(item) for item in command]
        if not resolved:
            return resolved
        resolved[0] = resolve_interpreter_token(resolved[0], self.project_root)
        return resolved

    def _resolve_artifact(self, relative: str) -> str | None:
        path = (self.project_root / relative).resolve()
        return str(path) if path.is_file() else None

    @staticmethod
    def _tail(text: str | None) -> str:
        value = (text or "").strip()
        if len(value) <= _LOG_TAIL_LIMIT:
            return value
        return "...[truncated]\n" + value[-_LOG_TAIL_LIMIT:]


def build_adapter(manifest: ProjectManifest, project_root: str | Path) -> ProjectAdapter:
    """按 manifest 选择适配器。

    目前只有命令行适配器；后续可在此按 `project.kind` 分派代码式适配器。
    """
    return CommandProjectAdapter(manifest, project_root)
