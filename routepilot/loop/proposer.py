"""提案层：把"下一步改什么"抽象成可插拔的 Proposer。

MVP 采用「整文件替换」语义（rollback 天然安全）；路径必须落在 manifest 的
editable 白名单内，否则整条提案作废。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

import yaml

from routepilot.adapter.manifest import ProjectManifest, classify_path
from routepilot.runtime.yaml_utils import strip_code_fence


@dataclass
class FileEdit:
    path: str
    content: str


@dataclass
class Proposal:
    summary: str
    edits: list[FileEdit]
    rationale: str = ""


@dataclass
class ProposalContext:
    goal: str
    trial_index: int
    baseline_metrics: dict[str, float]
    best_metrics: dict[str, float]
    history: list[dict[str, Any]] = field(default_factory=list)
    editable_files: dict[str, str] = field(default_factory=dict)
    diagnosis: str = ""


class Proposer(Protocol):
    def propose(self, context: ProposalContext) -> Proposal | None:
        """返回下一个候选实验；返回 None 表示不再提案。"""


def validate_proposal(
    proposal: Proposal, manifest: ProjectManifest
) -> tuple[bool, str]:
    """校验提案只能落在 editable 面上。"""
    if not proposal.edits:
        return False, "proposal 不含任何文件编辑"
    for edit in proposal.edits:
        verdict = classify_path(edit.path, manifest.editable, manifest.protected)
        if verdict != "editable":
            return False, f"{edit.path} 不在 editable 白名单内（判定为 {verdict}）"
        if not edit.content.strip():
            return False, f"{edit.path} 的目标内容为空"
    return True, ""


def parse_proposal(text: str) -> Proposal:
    payload = yaml.safe_load(strip_code_fence(text or "")) or {}
    if not isinstance(payload, dict):
        raise ValueError("proposal 必须是 YAML mapping")
    files = payload.get("files")
    if not isinstance(files, dict) or not files:
        raise ValueError("proposal 缺少非空的 files 映射")
    edits = [
        FileEdit(path=str(path).replace("\\", "/"), content=str(content))
        for path, content in files.items()
    ]
    summary = str(payload.get("summary") or "").strip() or "untitled proposal"
    return Proposal(summary=summary, edits=edits, rationale=str(payload.get("rationale") or ""))


class ScriptedProposer:
    """按预设候选顺序提案：离线、确定性，用于复现实验与回归测试。"""

    def __init__(self, proposals: list[Proposal]):
        self._proposals = list(proposals)
        self._cursor = 0

    @classmethod
    def from_file(cls, path: str | Path) -> "ScriptedProposer":
        payload = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
        raw = payload.get("proposals") if isinstance(payload, dict) else None
        if not isinstance(raw, list) or not raw:
            raise ValueError(f"proposals 文件缺少非空的 proposals 列表: {path}")
        return cls([_proposal_from_mapping(item) for item in raw])

    def propose(self, context: ProposalContext) -> Proposal | None:
        if self._cursor >= len(self._proposals):
            return None
        proposal = self._proposals[self._cursor]
        self._cursor += 1
        return proposal


def _proposal_from_mapping(item: Any) -> Proposal:
    if not isinstance(item, dict):
        raise ValueError("proposals 列表项必须是 mapping")
    files = item.get("files")
    if not isinstance(files, dict) or not files:
        raise ValueError("proposals 列表项缺少非空的 files 映射")
    return Proposal(
        summary=str(item.get("summary") or "untitled proposal"),
        edits=[
            FileEdit(path=str(path).replace("\\", "/"), content=str(content))
            for path, content in files.items()
        ],
        rationale=str(item.get("rationale") or ""),
    )


LLM_SYSTEM_PROMPT = (
    "You are RoutePilot's experiment proposer. "
    "Return ONE YAML mapping with keys summary, rationale, files. "
    "'files' maps a relative path to its COMPLETE new content. "
    "Only edit files listed in editable_files; never touch protected paths. "
    "Return YAML only, no markdown fences, no commentary."
)


class LlmProposer:
    """用 LLM 生成受控编辑包；整文件替换，路径受 editable 白名单约束。"""

    def __init__(
        self,
        llm_client: Any,
        *,
        max_tokens: int = 2048,
        timeout: tuple[int, int] | int = (30, 600),
        stream: bool = True,
    ):
        self.llm_client = llm_client
        self.max_tokens = max_tokens
        self.timeout = timeout
        self.stream = stream

    def propose(self, context: ProposalContext) -> Proposal | None:
        if not getattr(self.llm_client, "available", lambda: False)():
            return None
        prompt = yaml.safe_dump(
            {
                "task": "提出下一个预测实验，直接给出可执行的整文件替换。",
                "goal": context.goal,
                "trial_index": context.trial_index,
                "baseline_metrics": context.baseline_metrics,
                "best_metrics": context.best_metrics,
                "diagnosis": context.diagnosis,
                "previous_trials": [
                    {
                        "summary": item.get("summary"),
                        "decision": item.get("decision"),
                        "reason": item.get("reason"),
                        "metrics": item.get("metrics"),
                    }
                    for item in context.history
                ],
                "editable_files": context.editable_files,
                "hard_rules": [
                    "只返回一个 YAML mapping，不要 Markdown 代码块。",
                    "files 的键必须是 editable_files 里出现过的相对路径。",
                    "每个值是该文件的完整新内容，不是 diff。",
                    "保持数据读取路径、评估口径与输出列名不变。",
                    "不要修改受保护文件。",
                ],
            },
            allow_unicode=True,
            sort_keys=False,
        )
        result = self.llm_client.complete_with_usage(
            LLM_SYSTEM_PROMPT,
            prompt,
            agent="Proposer",
            step="ProposeExperiment",
            max_tokens=self.max_tokens,
            timeout=self.timeout,
            stream=self.stream,
        )
        if not getattr(result, "success", False) or not result.content:
            return None
        try:
            return parse_proposal(result.content)
        except (ValueError, yaml.YAMLError):
            return None
