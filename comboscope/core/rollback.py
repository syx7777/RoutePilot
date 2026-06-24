from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any


def rollback_change(backup_info: dict[str, Any]) -> dict[str, Any]:
    restored: list[str] = []
    for item in backup_info.get("files", []):
        source = Path(item["backup_path"])
        target = Path(item["target_path"])
        if source.exists():
            shutil.copy2(source, target)
            restored.append(target.as_posix())
    return {"success": True, "restored": restored}
