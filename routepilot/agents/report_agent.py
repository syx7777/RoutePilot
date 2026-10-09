from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from routepilot.core.reports import build_final_report_context, write_final_report as write_decision_report


def write_experiment_review(
    path: str | Path,
    comparison: dict[str, Any],
    run_status: dict[str, Any],
    agent2_review: dict[str, Any] | None = None,
) -> None:
    reason = _translate_reason(str(comparison["reason"]))
    bad_result_reason = _bad_result_reason(comparison, run_status)
    lines = [
        "# 实验复核",
        "",
        f"- 实验决策：{comparison['decision']}",
        f"- 决策原因：{reason}",
        f"- 训练是否成功：{run_status.get('train_success')}",
        f"- 评测是否成功：{run_status.get('eval_success')}",
        f"- WAPE 变化：{comparison.get('wape_delta')}",
        f"- Bias 绝对值变化：{comparison.get('bias_delta')}",
        f"- trial 训练副本：{run_status.get('generated_train_path')}",
        f"- 训练输出目录：{run_status.get('real_output_dir')}",
        "",
        "## 效果不好原因",
        f"- {bad_result_reason}",
    ]
    if agent2_review:
        lines.extend(
            [
                "",
                "## Agent2 模型复核",
                f"- 来源：{agent2_review.get('source')}",
                f"- 最新方案：{agent2_review.get('best_solution')}",
                f"- 下一步动作：{agent2_review.get('next_action')}",
            ]
        )
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_final_report(
    *,
    forecast_report_path: str | Path,
    experiment_review_path: str | Path,
    review_result_path: str | Path,
    output_path: str | Path,
    agent_status_path: str | Path | None = None,
    token_usage_path: str | Path | None = None,
    artifact_index_path: str | Path | None = None,
) -> None:
    review = json.loads(Path(review_result_path).read_text(encoding="utf-8"))
    run_status = _read_json(Path(review_result_path).with_name("run_status.json"))
    comparison = _read_json(Path(review_result_path).with_name("metric_comparison.json"))
    old_metrics = _read_json(Path(review_result_path).with_name("metrics.json"))
    new_metrics = _read_json(Path(review_result_path).with_name("new_metrics.json"))
    problem_context = _read_json(Path(review_result_path).with_name("problem_context.json"))
    experiment_plan = _read_yaml(Path(review_result_path).with_name("experiment_plan.yaml"))
    context = build_final_report_context(
        review_result=review,
        metric_comparison=comparison,
        old_metrics=old_metrics,
        new_metrics=new_metrics,
        problem_context=problem_context,
        experiment_plan=experiment_plan,
        run_status=run_status,
        artifacts={
            "完整评测报告": Path(forecast_report_path).as_posix(),
            "实验复核": Path(experiment_review_path).as_posix(),
            "产物索引": Path(artifact_index_path).as_posix() if artifact_index_path else "artifact_index.md",
        },
    )
    write_decision_report(context, output_path)


def _translate_reason(reason: str) -> str:
    translations = {
        "wape improved enough and bias stayed within threshold": "WAPE 改善达到阈值，且 Bias 未超过允许恶化范围",
        "train or evaluation failed": "训练或评测失败",
        "wape improvement is below threshold": "WAPE 改善幅度未达到 keep 阈值",
        "bias regression exceeds threshold": "Bias 恶化超过允许阈值",
        "llm objective improved": "模型选择的主优化目标已改善",
        "llm objective did not improve": "模型选择的主优化目标未改善",
        "agent2 code generation failed": "Agent2 代码生成失败，未进入训练",
        "feature change was not applied": "本轮特征改动未被真实训练代码消费",
    }
    return translations.get(reason, reason)


def _bad_result_reason(comparison: dict[str, Any], run_status: dict[str, Any]) -> str:
    if run_status.get("train_returncode") == "agent2_code_generation_failed" or run_status.get("agent2_code_generation_success") is False:
        record = run_status.get("agent2_code_modification_path") or "code/agent2_code_modification.yaml"
        return f"Agent2 代码生成失败：没有生成通过编译和特征应用审计的 trial 代码，已跳过训练。修改记录：{record}"
    if run_status.get("feature_application_success") is False:
        audit = run_status.get("feature_application_audit_path") or "code/agent2_feature_application_audit.yaml"
        return f"Agent2 特征应用审计未通过：本轮特征开关没有被真实训练代码消费，已跳过训练。审计文件：{audit}"
    if not run_status.get("train_success"):
        log = run_status.get("train_log_path") or "train.log"
        return f"训练失败，需要优先查看训练日志：{log}"
    if not run_status.get("eval_success"):
        log = run_status.get("eval_log_path") or run_status.get("error") or "eval.log"
        return f"评测或结果标准化失败，需要优先查看：{log}"
    if comparison.get("decision") == "keep":
        return "本轮实验满足 keep 规则，不属于效果不好的 trial。"
    reason = str(comparison.get("reason", ""))
    return _translate_reason(reason)


def _agent_summary_lines(
    agent_status_path: str | Path | None,
    token_usage_path: str | Path | None,
    artifact_index_path: str | Path | None,
) -> list[str]:
    lines: list[str] = []
    status = _read_json(agent_status_path)
    tokens = _read_json(token_usage_path)
    agents = status.get("agents", {}) if isinstance(status, dict) else {}
    if agents:
        for agent, info in agents.items():
            lines.append(
                f"- {agent}：状态={info.get('status')}, 耗时={round(float(info.get('duration_seconds', 0.0)), 3)}s, 当前/最后步骤={info.get('current_step')}"
            )
    else:
        lines.append("- Agent 状态：暂无记录。")
    if tokens:
        lines.append(
            "- Token 汇总："
            f"prompt={tokens.get('prompt_tokens', 0)}, "
            f"completion={tokens.get('completion_tokens', 0)}, "
            f"total={tokens.get('total_tokens', 0)}, "
            f"estimated_calls={tokens.get('estimated_calls', 0)}"
        )
    else:
        lines.append("- Token 汇总：暂无记录。")
    if artifact_index_path:
        lines.append(f"- 产物索引：{Path(artifact_index_path).as_posix()}")
    return lines


def _read_json(path: str | Path | None) -> dict[str, Any]:
    if not path:
        return {}
    target = Path(path)
    if not target.exists():
        return {}
    return json.loads(target.read_text(encoding="utf-8"))


def _read_yaml(path: str | Path | None) -> dict[str, Any]:
    if not path:
        return {}
    target = Path(path)
    if not target.exists():
        return {}
    import yaml

    value = yaml.safe_load(target.read_text(encoding="utf-8")) or {}
    return value if isinstance(value, dict) else {}
