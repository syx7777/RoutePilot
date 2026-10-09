from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from routepilot.runtime.yaml_utils import append_yaml_output_contract, safe_load_yaml_mapping


SELECT_BEST_TRIAL_TIMEOUT = (10, 600)


def select_best_trial(
    *,
    ask: str,
    history: list[dict[str, Any]],
    trial_summaries: list[dict[str, Any]],
    llm_client: Any,
) -> dict[str, Any]:
    fallback = _fallback_best_trial(history)
    prompt = yaml.safe_dump(
        {
            "user_ask": ask,
            "run_history": history,
            "trial_summaries": trial_summaries,
            "required_yaml_fields": ["best_trial_id", "decision_basis", "recommended_action"],
        },
        allow_unicode=True,
        sort_keys=False,
    )
    system_prompt, prompt = append_yaml_output_contract(
        "你是 RoutePilot 多轮实验汇总器。只能根据已完成 trial 的结构化结果选择最佳 trial，不要重新定义指标口径。只返回 YAML。",
        prompt,
    )
    try:
        result = llm_client.complete_with_usage(
            system_prompt,
            prompt,
            agent="Agent2",
            step="SelectBestTrial",
            timeout=SELECT_BEST_TRIAL_TIMEOUT,
        )
    except TypeError:
        result = llm_client.complete_with_usage(
            system_prompt,
            prompt,
            agent="Agent2",
            step="SelectBestTrial",
        )
    if not result.content:
        return fallback
    try:
        parsed = safe_load_yaml_mapping(result.content, agent="Agent2", step="SelectBestTrial")
    except ValueError:
        return fallback
    known_ids = {str(row.get("trial_id")) for row in history}
    best_trial_id = str(parsed.get("best_trial_id", fallback["best_trial_id"]))
    if best_trial_id not in known_ids:
        best_trial_id = fallback["best_trial_id"]
    return {
        "source": "llm" if getattr(result, "success", False) else "fallback",
        "best_trial_id": best_trial_id,
        "decision_basis": str(parsed.get("decision_basis") or fallback["decision_basis"]),
        "recommended_action": str(parsed.get("recommended_action") or fallback["recommended_action"]),
    }


def write_best_trial_review(path: str | Path, review: dict[str, Any]) -> None:
    lines = [
        "# 最佳方案复核",
        "",
        f"- 来源：{review.get('source')}",
        f"- 最佳 trial：{review.get('best_trial_id')}",
        f"- 判断依据：{review.get('decision_basis')}",
        f"- 建议动作：{review.get('recommended_action')}",
    ]
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")


def _fallback_best_trial(history: list[dict[str, Any]]) -> dict[str, Any]:
    keep_rows = [row for row in history if row.get("decision") == "keep"]
    selected = keep_rows[0] if keep_rows else (history[-1] if history else {})
    return {
        "source": "fallback",
        "best_trial_id": selected.get("trial_id", "n/a"),
        "decision_basis": "fallback 选择第一个 keep trial；如果没有 keep，则选择最后一轮。",
        "recommended_action": "人工复核 run_history.csv 和各 trial final_report 后再决定是否保留。",
    }
