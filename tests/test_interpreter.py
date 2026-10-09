from __future__ import annotations

import sys
from pathlib import Path

from routepilot.runtime.interpreter import project_interpreter, resolve_interpreter_token


def test_project_interpreter_prefers_windows_layout(tmp_path: Path) -> None:
    scripts = tmp_path / ".venv" / "Scripts"
    scripts.mkdir(parents=True)
    executable = scripts / "python.exe"
    executable.write_text("", encoding="utf-8")

    assert project_interpreter(tmp_path) == executable.as_posix()


def test_project_interpreter_supports_posix_layout(tmp_path: Path) -> None:
    bindir = tmp_path / ".venv" / "bin"
    bindir.mkdir(parents=True)
    executable = bindir / "python"
    executable.write_text("", encoding="utf-8")

    assert project_interpreter(tmp_path) == executable.as_posix()


def test_project_interpreter_falls_back_to_current_interpreter(tmp_path: Path) -> None:
    assert project_interpreter(tmp_path) == sys.executable


def test_resolve_interpreter_token_replaces_bare_python(tmp_path: Path) -> None:
    bindir = tmp_path / ".venv" / "bin"
    bindir.mkdir(parents=True)
    executable = bindir / "python"
    executable.write_text("", encoding="utf-8")

    assert resolve_interpreter_token("python", tmp_path) == executable.as_posix()
    assert resolve_interpreter_token("python3", tmp_path) == executable.as_posix()


def test_resolve_interpreter_token_keeps_explicit_path(tmp_path: Path) -> None:
    explicit = tmp_path / "custom_python.exe"
    explicit.write_text("", encoding="utf-8")

    assert resolve_interpreter_token(explicit.as_posix(), tmp_path) == explicit.as_posix()


def test_resolve_interpreter_token_keeps_unrelated_tokens(tmp_path: Path) -> None:
    assert resolve_interpreter_token("bash", tmp_path) == "bash"
