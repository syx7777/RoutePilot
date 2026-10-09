"""诊断步骤：把当前指标与历史轮次压成一段证据摘要，供提案步骤消费。

这是闭环里第一个被路由的 LLM 步骤——它偏"读证据、做归纳"，难度低，
动态路由应当把它交给弱模型，而把代码生成类的提案步骤交给强模型。
"""

from __future__ import annotations

from typing import Any

import yaml

from routepilot.loop.proposer import ProposalContext

DIAGNOSE_SYSTEM_PROMPT = (
    "You are RoutePilot's diagnosis step. Based on the provided metrics and trial history, "
    "write a short evidence-only diagnosis in Chinese. "
    "State which metric moved, by how much, and what that suggests the next experiment "
    "should target. Do NOT invent numbers that are not in the input. "
    "Return plain text only, no markdown fences, at most 120 words."
)


class LlmDiagnoser:
    """可调用对象：`diagnoser(context) -> str`，失败时返回空串而不是中断闭环。"""

    def __init__(self, llm_client: Any, *, max_tokens: int = 400, timeout: Any = (20, 300)):
        self.llm_client = llm_client
        self.max_tokens = max_tokens
        self.timeout = timeout

    def __call__(self, context: ProposalContext) -> str:
        if not getattr(self.llm_client, "available", lambda: False)():
            return ""
        prompt = yaml.safe_dump(
            {
                "task": "诊断当前预测实验的指标状态，指出下一轮实验应该针对什么。",
                "goal": context.goal,
                "trial_index": context.trial_index,
                "baseline_metrics": context.baseline_metrics,
                "best_metrics": context.best_metrics,
                "previous_trials": [
                    {
                        "summary": item.get("summary"),
                        "decision": item.get("decision"),
                        "reason": item.get("reason"),
                        "metrics": item.get("metrics"),
                    }
                    for item in context.history
                ],
                "editable_files": sorted(context.editable_files),
                "hard_rules": [
                    "只使用输入里出现的数字，不要编造。",
                    "返回纯文本，不要 Markdown 代码块。",
                    "120 字以内。",
                ],
            },
            allow_unicode=True,
            sort_keys=False,
        )
        result = self.llm_client.complete_with_usage(
            DIAGNOSE_SYSTEM_PROMPT,
            prompt,
            agent="Diagnoser",
            step="Diagnose",
            max_tokens=self.max_tokens,
            timeout=self.timeout,
            editable_files=len(context.editable_files),
        )
        if not getattr(result, "success", False) or not result.content:
            return ""
        return result.content.strip()
