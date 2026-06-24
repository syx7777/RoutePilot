from __future__ import annotations

import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class SkillMetadata:
    name: str
    description: str
    path: Path


@dataclass(frozen=True)
class LoadedSkill:
    name: str
    description: str
    path: Path
    content: str


class SkillsManager:
    def __init__(self, skills_root: str | Path):
        self.skills_root = Path(skills_root).resolve()
        self._metadata_cache: list[SkillMetadata] | None = None
        self._skill_cache: dict[str, LoadedSkill] = {}

    def discover_skills(self) -> list[SkillMetadata]:
        if self._metadata_cache is not None:
            return self._metadata_cache
        if not self.skills_root.exists():
            self._metadata_cache = []
            return []

        metadata: list[SkillMetadata] = []
        for skill_file in sorted(self.skills_root.glob("*/SKILL.md")):
            raw = skill_file.read_text(encoding="utf-8")
            frontmatter = self._parse_frontmatter(raw)
            name = str(frontmatter.get("name") or skill_file.parent.name)
            description = str(frontmatter.get("description") or "")
            metadata.append(SkillMetadata(name=name, description=description, path=skill_file))
        self._metadata_cache = metadata
        return metadata

    def load_skill(self, skill_name: str) -> LoadedSkill:
        if skill_name in self._skill_cache:
            return self._skill_cache[skill_name]

        skill_file = self._skill_path(skill_name) / "SKILL.md"
        if not skill_file.exists():
            raise FileNotFoundError(skill_file)
        content = skill_file.read_text(encoding="utf-8")
        frontmatter = self._parse_frontmatter(content)
        skill = LoadedSkill(
            name=str(frontmatter.get("name") or skill_name),
            description=str(frontmatter.get("description") or ""),
            path=skill_file,
            content=content,
        )
        self._skill_cache[skill_name] = skill
        return skill

    def load_reference(self, skill_name: str, relative_path: str | Path) -> str:
        target = self._resolve_inside_skill(skill_name, relative_path)
        if not target.exists() or not target.is_file():
            raise FileNotFoundError(target)
        return target.read_text(encoding="utf-8")

    def run_script(
        self,
        skill_name: str,
        script_name: str | Path,
        args: list[str | Path] | None = None,
        *,
        check: bool = True,
    ) -> subprocess.CompletedProcess[str]:
        script = self._resolve_inside_skill(skill_name, Path("scripts") / script_name)
        if not script.exists() or not script.is_file():
            raise FileNotFoundError(script)
        return subprocess.run(
            [sys.executable, script.as_posix(), *[str(arg) for arg in args or []]],
            cwd=script.parent.as_posix(),
            text=True,
            capture_output=True,
            check=check,
        )

    def _skill_path(self, skill_name: str) -> Path:
        return self._resolve_root_child(skill_name)

    def _resolve_root_child(self, relative_path: str | Path) -> Path:
        target = (self.skills_root / relative_path).resolve()
        if target != self.skills_root and self.skills_root not in target.parents:
            raise ValueError(f"path is outside skills root: {relative_path}")
        return target

    def _resolve_inside_skill(self, skill_name: str, relative_path: str | Path) -> Path:
        skill_dir = self._skill_path(skill_name)
        target = (skill_dir / relative_path).resolve()
        if target != skill_dir and skill_dir not in target.parents:
            raise ValueError(f"path is outside skill directory: {relative_path}")
        return target

    @staticmethod
    def _parse_frontmatter(content: str) -> dict[str, Any]:
        if not content.startswith("---"):
            return {}
        parts = content.split("---", 2)
        if len(parts) < 3:
            return {}
        parsed = yaml.safe_load(parts[1]) or {}
        return parsed if isinstance(parsed, dict) else {}
