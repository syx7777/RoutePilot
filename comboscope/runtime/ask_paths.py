from __future__ import annotations

import re
from pathlib import Path


PATH_PATTERN = re.compile(r"(?P<path>(?:/|~)[^\s，。；;、]+)")


def extract_existing_dirs_from_ask(ask: str) -> list[Path]:
    dirs: list[Path] = []
    for match in PATH_PATTERN.finditer(ask):
        raw = match.group("path").strip().rstrip(".,，。)")
        path = Path(raw).expanduser()
        if path.exists() and path.is_dir():
            dirs.append(path.resolve())
    return dirs


def resolve_experiment_dir(cli_experiment: str | Path | None, ask: str) -> Path:
    if cli_experiment:
        return Path(cli_experiment).expanduser().resolve()
    candidates = extract_existing_dirs_from_ask(ask)
    if candidates:
        return candidates[0]
    raise ValueError("experiment directory is required: pass --experiment or include an existing directory path in --ask")
