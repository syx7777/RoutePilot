"""解释器解析：被接入项目应当用自己虚拟环境里的 Python 运行。"""

from __future__ import annotations

import sys
from pathlib import Path

# 需要被替换成项目解释器的裸记号（显式给了路径则保持原样）。
PYTHON_TOKENS = {"python", "python.exe", "python3", "python3.exe", "py", "py.exe"}

_VENV_LAYOUTS = (
    (".venv", "Scripts", "python.exe"),
    (".venv", "bin", "python"),
    ("venv", "Scripts", "python.exe"),
    ("venv", "bin", "python"),
)


def project_interpreter(project_root: str | Path) -> str:
    """返回项目自带解释器的 POSIX 路径；找不到则回退到当前解释器。

    同时覆盖 POSIX（.venv/bin/python）与 Windows（.venv/Scripts/python.exe）。
    """
    root = Path(project_root)
    for parts in _VENV_LAYOUTS:
        candidate = root.joinpath(*parts)
        if candidate.is_file():
            return candidate.as_posix()
    return sys.executable


def resolve_interpreter_token(token: str, project_root: str | Path) -> str:
    """把裸的 `python` 记号替换为项目解释器；显式路径原样返回。"""
    head = Path(token)
    if head.is_file() or head.name.lower() not in PYTHON_TOKENS:
        return token
    return project_interpreter(project_root)
