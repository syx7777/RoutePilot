from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def _read_json(path: Path) -> dict[str, Any]:
    resolved = _resolve_artifact_path(path)
    if not resolved.exists():
        return {}
    return json.loads(resolved.read_text(encoding="utf-8"))


def _resolve_artifact_path(path: Path) -> Path:
    if path.exists():
        return path
    aliases = {
        "review_result.json": "agent2/review_result.json",
        "metric_comparison.json": "evaluation/metric_comparison.json",
        "run_status.json": "agent2/run_status.json",
        "agent_status.json": "audit/agent_status.json",
        "token_usage.json": "audit/token_usage.json",
        "experiment_plan.yaml": "agent1/experiment_plan.yaml",
        "optimization_suggestions.md": "reports/optimization_suggestions.md",
    }
    rel = aliases.get(path.name)
    if rel:
        return path.parent / rel
    return path


def _value(data: dict[str, Any], key: str) -> str:
    value = data.get(key)
    return "n/a" if value is None else str(value)


def _translate_reason(reason: str) -> str:
    translations = {
        "wape improved enough and bias stayed within threshold": "WAPE 改善达到阈值，且 Bias 未超过允许恶化范围",
        "train or evaluation failed": "训练或评测失败",
        "wape improvement is below threshold": "WAPE 改善幅度未达到 keep 阈值",
        "bias regression exceeds threshold": "Bias 恶化超过允许阈值",
        "llm objective improved": "模型选择的主优化目标已改善",
        "llm objective did not improve": "模型选择的主优化目标未改善",
    }
    return translations.get(reason, reason)


def print_run_summary(output_dir: str | Path) -> None:
    output = Path(output_dir)
    review = _read_json(output / "review_result.json")
    comparison = _read_json(output / "metric_comparison.json")
    status = _read_json(output / "run_status.json")
    agent_status = _read_json(output / "agent_status.json")
    token_usage = _read_json(output / "token_usage.json")

    print("ComboScope run completed")
    print(f"output_dir: {output.as_posix()}")
    print(f"decision: {_value(review, 'decision')}")
    print(f"reason: {_translate_reason(_value(review, 'reason'))}")
    print(f"train_success: {_value(status, 'train_success')}")
    print(f"eval_success: {_value(status, 'eval_success')}")
    print(
        "primary_objective: "
        f"metric={_value(comparison, 'primary_metric_label')} "
        f"old={_value(comparison, 'old_primary')} "
        f"new={_value(comparison, 'new_primary')} "
        f"delta={_value(comparison, 'primary_delta')}"
    )
    print(
        "wape: "
        f"old={_value(comparison, 'old_wape')} "
        f"new={_value(comparison, 'new_wape')} "
        f"delta={_value(comparison, 'wape_delta')}"
    )
    print(
        "bias: "
        f"old={_value(comparison, 'old_bias')} "
        f"new={_value(comparison, 'new_bias')} "
        f"delta={_value(comparison, 'bias_delta')}"
    )
    print(f"final_report: {(output / 'final_report.md').as_posix()}")
    print(f"trace: {(output / 'trace.jsonl').as_posix()}")
    print(f"experiment_plan: {_resolve_artifact_path(output / 'experiment_plan.yaml').as_posix()}")
    print(f"optimization_suggestions: {_resolve_artifact_path(output / 'optimization_suggestions.md').as_posix()}")
    for agent in ("Agent1", "Agent2"):
        info = agent_status.get("agents", {}).get(agent, {})
        print(
            f"{agent}: status={_value(info, 'status')} "
            f"duration={_value(info, 'duration_seconds')} "
            f"last_step={_value(info, 'current_step')}"
        )
    print(f"total_tokens: {_value(token_usage, 'total_tokens')}")
    print(f"artifact_index: {(output / 'artifact_index.md').as_posix()}")
    if (output / "error_report.md").exists():
        print(f"error_report: {(output / 'error_report.md').as_posix()}")


def print_trial_summary(trial_id: str, trial_dir: str | Path) -> None:
    output = Path(trial_dir)
    review = _read_json(output / "review_result.json")
    comparison = _read_json(output / "metric_comparison.json")
    agent_status = _read_json(output / "agent_status.json")
    token_usage = _read_json(output / "token_usage.json")
    agent1 = agent_status.get("agents", {}).get("Agent1", {})
    agent2 = agent_status.get("agents", {}).get("Agent2", {})
    print(
        f"{trial_id}: decision={_value(review, 'decision')} "
        f"primary={_value(comparison, 'primary_metric_label')} "
        f"primary_delta={_value(comparison, 'primary_delta')} "
        f"wape_delta={_value(comparison, 'wape_delta')} "
        f"Agent1={_value(agent1, 'duration_seconds')} "
        f"Agent2={_value(agent2, 'duration_seconds')} "
        f"tokens={_value(token_usage, 'total_tokens')} "
        f"final_report={(output / 'final_report.md').as_posix()}"
    )


def print_loop_summary(output_dir: str | Path, history: list[dict[str, Any]]) -> None:
    output = Path(output_dir)
    decisions = ", ".join(f"{row['trial_id']}={row['decision']}" for row in history) or "n/a"
    best = _read_json(output / "best_trial_review.json")
    print("ComboScope loop completed")
    print(f"trials: {len(history)}")
    print(f"decisions: {decisions}")
    if best:
        print(f"best_trial: {best.get('best_trial_id')} source={best.get('source')}")
    print(f"run_history: {(output / 'run_history.csv').as_posix()}")
    print(f"final_report: {(output / 'final_report.md').as_posix()}")
