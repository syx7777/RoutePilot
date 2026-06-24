from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any


def build_final_report_context(
    review_result: dict[str, Any],
    metric_comparison: dict[str, Any] | None,
    old_metrics: dict[str, Any] | None,
    new_metrics: dict[str, Any] | None,
    problem_context: dict[str, Any] | None,
    experiment_plan: dict[str, Any] | None,
    run_status: dict[str, Any] | None,
    artifacts: dict[str, Any] | None,
) -> dict[str, Any]:
    metric_comparison = metric_comparison or {}
    old_metrics = old_metrics or {}
    new_metrics = new_metrics or {}
    problem_context = problem_context or {}
    experiment_plan = experiment_plan or {}
    run_status = run_status or {}
    artifacts = artifacts or {}

    train_success = bool(run_status.get("train_success"))
    eval_success = bool(run_status.get("eval_success"))
    comparison_decision = str(metric_comparison.get("decision") or review_result.get("decision") or "rollback")
    decision = str(review_result.get("decision") or comparison_decision)
    if comparison_decision == "rollback" or not train_success or not eval_success:
        decision = "rollback"
    is_effective = decision == "keep" and train_success and eval_success
    feature_names = _feature_names(experiment_plan)
    primary_metric = (
        metric_comparison.get("primary_metric_label")
        or metric_comparison.get("primary_metric")
        or _plan_metric(experiment_plan)
        or "WAPE"
    )
    old_value = metric_comparison.get("old_primary")
    new_value = metric_comparison.get("new_primary")
    delta = metric_comparison.get("primary_delta")
    if old_value is None:
        old_value = _metric_value(old_metrics, str(primary_metric))
    if new_value is None:
        new_value = _metric_value(new_metrics, str(primary_metric))
    old_wape = metric_comparison.get("old_wape")
    new_wape = metric_comparison.get("new_wape")
    if old_wape is None:
        old_wape = _metric_value(old_metrics, "wape")
    if new_wape is None:
        new_wape = _metric_value(new_metrics, "wape")
    if _metric_key(primary_metric) == "wape":
        if old_value is None:
            old_value = old_wape
        if new_value is None:
            new_value = new_wape
    if delta is None:
        delta = _numeric_delta(old_value, new_value)
    wape_delta = metric_comparison.get("wape_delta")
    if wape_delta is None:
        wape_delta = _numeric_delta(old_wape, new_wape)

    old_bias = metric_comparison.get("old_bias")
    new_bias = metric_comparison.get("new_bias")
    if old_bias is None:
        old_bias = _metric_value(old_metrics, "bias")
    if new_bias is None:
        new_bias = _metric_value(new_metrics, "bias")
    bias_abs_delta = metric_comparison.get("bias_delta")
    if bias_abs_delta is None:
        bias_abs_delta = _numeric_abs_delta(old_bias, new_bias)

    old_sample_count = metric_comparison.get("old_sample_count")
    new_sample_count = metric_comparison.get("new_sample_count")
    if old_sample_count is None:
        old_sample_count = _sample_count(old_metrics)
    if new_sample_count is None:
        new_sample_count = _sample_count(new_metrics)
    sample_count_consistent = metric_comparison.get("sample_count_consistent")
    if sample_count_consistent is None and old_sample_count is not None and new_sample_count is not None:
        sample_count_consistent = _same_value(old_sample_count, new_sample_count)

    text_metric_comparison = {
        **metric_comparison,
        "primary_metric_label": primary_metric,
        "old_primary": old_value,
        "new_primary": new_value,
        "primary_delta": delta,
        "old_wape": old_wape,
        "new_wape": new_wape,
        "wape_delta": wape_delta,
        "old_bias": old_bias,
        "new_bias": new_bias,
        "bias_delta": bias_abs_delta,
        "old_sample_count": old_sample_count,
        "new_sample_count": new_sample_count,
        "sample_count_consistent": sample_count_consistent,
    }

    train_log_path = run_status.get("train_log_path") or artifacts.get("train_log")
    train_failure_summary = _log_failure_summary(train_log_path) if not train_success else ""
    effective_run_status = {
        **run_status,
        "train_failure_summary": run_status.get("train_failure_summary") or train_failure_summary,
    }

    context = {
        "decision": decision,
        "best_solution": "modified" if is_effective else "baseline",
        "is_new_plan_effective": is_effective,
        "primary_reason": _primary_reason(review_result, text_metric_comparison, effective_run_status, is_effective),
        "experiment_summary": {
            "feature_name": ", ".join(feature_names) if feature_names else "本轮特征方案",
            "changed_files": _actual_changed_files(run_status, experiment_plan),
            "changes": experiment_plan.get("changes", []),
            "expected_effect": experiment_plan.get("expected_effect") or "",
            "target_problem": experiment_plan.get("target_problem") or problem_context.get("main_problem") or "",
        },
        "run_status": {
            "train_success": train_success,
            "eval_success": eval_success,
            "train_returncode": run_status.get("train_returncode"),
            "eval_returncode": run_status.get("eval_returncode"),
            "train_log_path": train_log_path,
            "eval_log_path": run_status.get("eval_log_path") or artifacts.get("eval_log"),
            "train_failure_summary": run_status.get("train_failure_summary") or train_failure_summary,
            "feature_application_success": run_status.get("feature_application_success"),
            "feature_application_audit_path": run_status.get("feature_application_audit_path"),
            "agent2_code_modification_path": run_status.get("agent2_code_modification_path"),
            "agent2_modified_files": run_status.get("agent2_modified_files", []),
            "agent2_code_generation_success": run_status.get("agent2_code_generation_success"),
            "agent2_code_generation_failure_reason": run_status.get("agent2_code_generation_failure_reason"),
            "agent2_failure_categories": run_status.get("agent2_failure_categories", []),
            "agent2_normalized_rejections": run_status.get("agent2_normalized_rejections", []),
            "agent2_source_locator_path": run_status.get("agent2_source_locator_path"),
        },
        "metric_comparison": {
            "primary_metric": _display_metric(primary_metric, experiment_plan),
            "old_value": old_value,
            "new_value": new_value,
            "delta": delta,
            "old_wape": old_wape,
            "new_wape": new_wape,
            "wape_delta": wape_delta,
            "old_bias": old_bias,
            "new_bias": new_bias,
            "bias_abs_delta": bias_abs_delta,
            "old_sample_count": old_sample_count,
            "new_sample_count": new_sample_count,
            "sample_count_consistent": sample_count_consistent,
        },
        "baseline_opportunities": _baseline_opportunities(problem_context, old_metrics),
        "failure_analysis": _failure_analysis(text_metric_comparison, run_status),
        "next_actions": _next_actions(decision, effective_run_status, text_metric_comparison, problem_context),
        "artifact_links": _artifact_links(artifacts),
    }
    return context


def write_final_report(report_context: dict[str, Any], output_path: str | Path) -> None:
    if report_context.get("decision") == "keep" and report_context.get("is_new_plan_effective"):
        text = _keep_template(report_context)
    else:
        text = _rollback_template(report_context)
    text = _remove_forbidden_baseline_phrases(_remove_unresolved_placeholders(text))
    Path(output_path).write_text(text, encoding="utf-8")


def write_final_report_context(report_context: dict[str, Any], output_path: str | Path) -> None:
    Path(output_path).write_text(json.dumps(report_context, ensure_ascii=False, indent=2), encoding="utf-8")


def _rollback_template(context: dict[str, Any]) -> str:
    exp = context["experiment_summary"]
    run = context["run_status"]
    metric = context["metric_comparison"]
    feature = exp["feature_name"]
    reason = context["primary_reason"]
    lines = [
        "# ComboScope 实验验证结论报告",
        "",
        "## 1. 结论",
        "",
        f"本轮 `{feature}` 实验无效，当前最佳方案仍然是 baseline。原因是：{reason}",
        "",
        "## 2. 当前最佳方案",
        "",
        "- 当前最佳方案：baseline",
        f"- 本轮实验决策：{context['decision']}",
        f"- 主决策口径：{metric.get('primary_metric')}",
        f"- 训练是否成功：{run.get('train_success')}",
        f"- 评测是否成功：{run.get('eval_success')}",
        f"- 失败原因：{reason}",
        "",
        "## 3. 本轮实验做了什么",
        "",
        f"- 实验方案：{feature}",
        f"- 修改位置：{_join_or_na(exp.get('changed_files'))}",
        f"- 修改内容：{_change_summary(exp.get('changes'))}",
        f"- 预期解决的问题：{exp.get('expected_effect') or exp.get('target_problem') or 'n/a'}",
        "",
        "## 4. 为什么本轮优化失败",
        "",
        *_bullet_lines(context.get("failure_analysis")),
        "",
        "## 5. baseline 当前仍可优化的方向",
        "",
        *_bullet_lines(context.get("baseline_opportunities")),
        "",
        "## 6. 下一轮建议",
        "",
        *_numbered_lines(context.get("next_actions")),
        "",
        "## 附录产物",
        "",
        *_artifact_lines(context.get("artifact_links")),
    ]
    return "\n".join(lines) + "\n"


def _keep_template(context: dict[str, Any]) -> str:
    exp = context["experiment_summary"]
    run = context["run_status"]
    metric = context["metric_comparison"]
    feature = exp["feature_name"]
    reason = context["primary_reason"]
    lines = [
        "# ComboScope 实验验证结论报告",
        "",
        "## 1. 结论",
        "",
        f"本轮 `{feature}` 实验有效，当前最佳方案切换为修改后方案。核心收益是：{reason}",
        "",
        "## 2. 当前最佳方案",
        "",
        "- 当前最佳方案：modified",
        f"- 本轮实验决策：{context['decision']}",
        f"- 主指标变化：{metric.get('primary_metric')} 从 {_fmt(metric.get('old_value'))} 变为 {_fmt(metric.get('new_value'))}，变化 {_fmt(metric.get('delta'))}",
        f"- signed Bias 辅助观察变化：{_fmt(metric.get('bias_abs_delta'))}",
        f"- 训练是否成功：{run.get('train_success')}",
        f"- 评测是否成功：{run.get('eval_success')}",
        "",
        "## 3. 本轮修改了什么",
        "",
        f"- 修改文件：{_join_or_na(exp.get('changed_files'))}",
        f"- 新增/修改特征：{feature}",
        f"- 改动目的：{exp.get('expected_effect') or 'n/a'}",
        f"- 对应原始问题：{exp.get('target_problem') or 'n/a'}",
        "",
        "## 4. 效果改善在哪里",
        "",
        f"- 整体指标改善：主指标 {metric.get('primary_metric')} 从 {_fmt(metric.get('old_value'))} 变为 {_fmt(metric.get('new_value'))}，变化 {_fmt(metric.get('delta'))}；WAPE 作为辅助观察指标变化 {_fmt(metric.get('wape_delta'))}",
        f"- 重点场景改善：请结合附录中的细粒度指标继续复核。",
        f"- badcase 改善：请结合附录中的 badcase 明细确认 top error 是否收敛。",
        f"- 是否存在副作用：signed Bias 辅助观察变化为 {_fmt(metric.get('bias_abs_delta'))}，需继续观察高估/低估方向风险。",
        "",
        "## 5. 仍然存在的问题",
        "",
        *_bullet_lines(context.get("baseline_opportunities"), fallback="本轮已改善主目标，但仍建议继续复核高误差场景和 top badcase。"),
        "",
        "## 6. 下一步优化建议",
        "",
        *_numbered_lines(context.get("next_actions")),
        "",
        "## 附录产物",
        "",
        *_artifact_lines(context.get("artifact_links")),
    ]
    return "\n".join(lines) + "\n"


def _primary_reason(
    review_result: dict[str, Any],
    metric_comparison: dict[str, Any],
    run_status: dict[str, Any],
    is_effective: bool,
) -> str:
    if run_status.get("train_returncode") == "agent2_code_generation_failed" or run_status.get("agent2_code_generation_success") is False:
        record = run_status.get("agent2_code_modification_path") or "code/agent2_code_modification.yaml"
        categories = _join_or_na(run_status.get("agent2_failure_categories"))
        failure_reason = run_status.get("agent2_code_generation_failure_reason") or "generated code did not pass pre-training validation"
        detail = _agent2_rejection_summary(run_status.get("agent2_normalized_rejections"))
        return (
            f"Agent2 代码生成失败：{failure_reason}。失败类别：{categories}。"
            f"{detail}因此未进入训练，也不能评价该特征效果。修改记录：{record}。"
        )
    if run_status.get("feature_application_success") is False:
        audit = run_status.get("feature_application_audit_path") or "code/agent2_feature_application_audit.yaml"
        return (
            "Agent2 特征应用审计未通过：本轮只生成了实验声明或开关，"
            f"没有发现该特征被真实训练代码消费，因此已跳过训练。审计文件：{audit}。"
        )
    if run_status.get("train_returncode") == "invalid_train_command":
        error = run_status.get("error") or "train_command 与 trial train.py 的 argparse 契约不匹配"
        return (
            f"训练命令校验失败：{error}。本轮未启动训练，"
            "未产生有效评测指标，因此不能评价该特征效果；需修复 Agent2 CLI 接入后复跑。"
        )
    if not run_status.get("train_success"):
        code = run_status.get("train_returncode")
        summary = run_status.get("train_failure_summary")
        if summary:
            return f"新实验训练失败：{summary}。训练进程返回码为 {code}，未产生有效评测指标，因此本轮未能验证该特征方案有效性。"
        return f"新实验训练失败，训练进程返回码为 {code}，未产生有效评测指标，因此本轮未能验证该特征方案有效性。"
    if not run_status.get("eval_success"):
        return "新实验评测或结果标准化失败，没有可比的新指标，因此本轮未能验证该方案有效性。"
    if is_effective:
        metric = _display_metric(metric_comparison.get("primary_metric_label") or metric_comparison.get("primary_metric") or "主指标", metric_comparison)
        old_value = metric_comparison.get("old_primary")
        new_value = metric_comparison.get("new_primary")
        delta = metric_comparison.get("primary_delta")
        return f"{metric} 从 {_fmt(old_value)} 变为 {_fmt(new_value)}，变化 {_fmt(delta)}，且训练与评测均成功。"
    reason = _decision_reason_text(str(review_result.get("reason") or metric_comparison.get("reason") or ""))
    metric = _display_metric(metric_comparison.get("primary_metric_label") or metric_comparison.get("primary_metric") or "主指标", metric_comparison)
    old_value = metric_comparison.get("old_primary")
    new_value = metric_comparison.get("new_primary")
    delta = metric_comparison.get("primary_delta")
    if old_value is not None or new_value is not None:
        return (
            f"{metric} 从 {_fmt(old_value)} 变为 {_fmt(new_value)}，{_delta_text(delta)}。"
            f"{reason or '该变化未达到用户 ask 对主目标的改善要求'}，当前不能作为最佳方案。"
        )
    if reason:
        return f"{reason}，当前不能作为最佳方案。"
    return "新方案未带来足够的主指标改善，当前不能证明该方案有效。"


def _actual_changed_files(run_status: dict[str, Any], experiment_plan: dict[str, Any]) -> list[str]:
    files = run_status.get("agent2_modified_files") or []
    if files:
        return [_display_trial_code_file(str(item)) for item in files]
    if run_status.get("train_returncode") == "agent2_code_generation_failed" or run_status.get("agent2_code_generation_success") is False:
        return ["未产生有效 trial 修改"]
    return list(experiment_plan.get("editable_files", []))


def _display_trial_code_file(path: str) -> str:
    name = Path(path).name
    if name:
        return f"trial 版 {name}"
    return path


def _failure_analysis(metric_comparison: dict[str, Any], run_status: dict[str, Any]) -> list[str]:
    if run_status.get("train_returncode") == "agent2_code_generation_failed" or run_status.get("agent2_code_generation_success") is False:
        lines = [
            "本轮失败的直接原因是 Agent2 代码生成失败，而不是特征效果被证伪。",
            f"失败类别：{_join_or_na(run_status.get('agent2_failure_categories'))}。",
            f"生成失败原因：{run_status.get('agent2_code_generation_failure_reason') or 'n/a'}。",
            "系统没有获得通过编译、运行时调用契约、特征应用审计和 smoke 校验的 trial 代码，因此阻断训练。",
            f"代码生成记录：{run_status.get('agent2_code_modification_path') or 'code/agent2_code_modification.yaml'}。",
            f"源码定位记录：{run_status.get('agent2_source_locator_path') or 'code/agent2_source_locator.yaml'}。",
        ]
        for item in (run_status.get("agent2_normalized_rejections") or [])[:6]:
            if isinstance(item, dict):
                lines.append(
                    f"Agent2 rejection：{item.get('category') or 'unknown'} | {item.get('path') or 'n/a'} | {item.get('reason') or 'n/a'}。"
                )
        return lines
    if run_status.get("feature_application_success") is False:
        return [
            "本轮失败的直接原因是特征应用审计未通过，而不是特征效果被证伪。",
            "Agent2 没有在 trial 训练副本或 Python 依赖中找到该特征被真实代码消费的证据。",
            f"审计文件路径：{run_status.get('feature_application_audit_path') or 'code/agent2_feature_application_audit.yaml'}。",
            "下一步应先补齐可执行的特征构造或参数消费逻辑，再复跑该实验。",
        ]
    if not run_status.get("train_success"):
        lines = [
            "本轮失败的直接原因是训练失败，而不是特征效果被证伪。",
            f"训练日志路径：{run_status.get('train_log_path') or 'logs/train.log'}。",
            "下一步应先修复执行问题，再复跑同一实验，避免把执行失败误判为特征失败。",
        ]
        if run_status.get("train_failure_summary"):
            lines.insert(1, f"训练日志关键错误：{run_status.get('train_failure_summary')}。")
        return lines
    if not run_status.get("eval_success"):
        return [
            "训练完成后未产生可标准化的评测结果，因此没有可比指标。",
            f"评测日志路径：{run_status.get('eval_log_path') or 'logs/eval.log'}。",
        ]
    if metric_comparison.get("decision") == "rollback":
        metric = _display_metric(metric_comparison.get("primary_metric_label") or metric_comparison.get("primary_metric") or "主指标", metric_comparison)
        old_bias = metric_comparison.get("old_bias")
        new_bias = metric_comparison.get("new_bias")
        return [
            (
                f"主指标 {metric}：{_fmt(metric_comparison.get('old_primary'))} -> "
                f"{_fmt(metric_comparison.get('new_primary'))}，{_delta_text(metric_comparison.get('primary_delta'))}，"
                f"{_metric_result_text(metric, metric_comparison.get('primary_delta'))}，未满足 keep 规则。"
            ),
            (
                f"修改后 test Bias：{_fmt(new_bias)}（原实验 Bias：{_fmt(old_bias)}；"
                f"绝对 Bias 变化 {_fmt(metric_comparison.get('bias_delta'))}）。"
            ),
            f"Bias 方向变化：{_bias_direction(old_bias)} -> {_bias_direction(new_bias)}。",
            _sample_count_text(metric_comparison),
            "因此本轮不能证明新方案有效，当前最佳方案仍然保留 baseline。",
        ]
    return ["本轮没有明显失败信号，但仍需复核未解决场景和潜在副作用。"]


def _baseline_opportunities(problem_context: dict[str, Any], old_metrics: dict[str, Any]) -> list[str]:
    opportunities: list[str] = []
    if old_metrics:
        if old_metrics.get("wape") is not None:
            opportunities.append(f"当前最佳方案 WAPE 为 {_fmt(old_metrics.get('wape'))}，仍可围绕高误差场景继续拆解。")
        if old_metrics.get("bias") is not None:
            direction = "低估" if _to_float(old_metrics.get("bias")) is not None and float(old_metrics["bias"]) < 0 else "高估"
            opportunities.append(f"当前最佳方案 Bias 为 {_fmt(old_metrics.get('bias'))}，存在整体{direction}倾向。")
    main_problem = problem_context.get("main_problem")
    if main_problem:
        opportunities.append(f"当前主要误差问题：{main_problem}。")
    evidence = problem_context.get("evidence") or problem_context.get("top_evidence") or []
    for item in evidence[:3]:
        opportunities.append(f"证据：{item}")
    if not opportunities:
        opportunities.append("继续围绕原始评测结果中的高误差场景、系统性偏差和 top badcase 设计更小粒度实验。")
    return opportunities


def _next_actions(
    decision: str,
    run_status: dict[str, Any],
    metric_comparison: dict[str, Any],
    problem_context: dict[str, Any],
) -> list[str]:
    if run_status.get("train_returncode") == "agent2_code_generation_failed" or run_status.get("agent2_code_generation_success") is False:
        categories = set(run_status.get("agent2_failure_categories") or [])
        actions = [
            "先查看 agent2_code_modification.yaml 中每次 codegen/repair attempt 的 prompt_chars、timeout 和 normalized_rejections。",
        ]
        if "smoke_validation_failed" in categories:
            actions.append("复跑前确认 staging smoke 使用原实验依赖路径，避免把 lightgbm 等外部依赖缺失误判为 Agent2 代码错误。")
        if "invalid_package" in categories:
            actions.append("强化 repair prompt 与解析校验：RepairCodeEdits 必须返回非空 YAML edits/files 包，空 content 后继续进入下一次 repair。")
        if "runtime_contract_error" in categories:
            actions.append("把 AST 调用契约错误中的文件、行号、函数名和 callee_signature 反馈给 Agent2，要求补齐缺失参数且保持原函数签名。")
        actions.extend(
            [
                "复跑前确认 Agent2 能生成通过 compile、runtime contract、feature audit 和 smoke 的 trial 代码。",
                "不要把本轮结果解释为特征无效；当前只是 Agent2 代码生成链路失败。",
            ]
        )
        return actions
    if run_status.get("feature_application_success") is False:
        return [
            "先修复 Agent2 生成代码逻辑，让特征开关对应到真实特征构造或真实参数消费。",
            "复跑同一方案前，确认 agent2_feature_application_audit.yaml 中该 feature 为 applied=true。",
            "不要把本轮结果解释为特征无效；当前只是特征未落地。",
        ]
    if not run_status.get("train_success"):
        actions = [
            "先修复本轮训练失败问题，确认训练命令、依赖和输入数据副本可完整跑通。",
            "复跑同一方案，避免把执行失败误判为特征失败。",
            "不建议在训练失败未解决前继续扩大特征改动范围。",
        ]
        summary = str(run_status.get("train_failure_summary") or "")
        if "missing 1 required positional argument" in summary or "unexpected keyword argument" in summary:
            actions.insert(1, "优先检查 Agent2 是否改变了被训练入口调用的函数签名；替换函数必须兼容原参数和返回结构。")
        return actions
    if not run_status.get("eval_success"):
        return [
            "先修复结果文件查找或字段映射问题，确保能生成标准化 prediction/actual。",
            "再验证本轮特征方案对主目标的影响。",
            "不建议在缺少可比指标时判断方案好坏。",
        ]
    if decision == "keep":
        return [
            "保留本轮修改后方案作为当前最佳方案。",
            "继续围绕仍然较差的切片和 top badcase 做小步特征实验。",
            "持续观察 Bias 变化，避免主指标改善伴随系统性偏差扩大。",
        ]
    return [
        "复核主目标未改善的原因，优先定位高误差切片和偏差方向。",
        "缩小下一轮改动范围，单次只验证一个明确特征假设。",
        "不建议继续堆叠与当前失败方案相同方向的大范围特征。",
    ]


def _artifact_links(artifacts: dict[str, Any]) -> dict[str, str]:
    defaults = {
        "完整评测报告": "forecast_report.md",
        "badcase 明细": "badcases.csv",
        "细粒度指标": "scene_metrics.csv",
        "训练日志": "logs/train.log",
        "执行 trace": "trace.jsonl",
    }
    for key, value in artifacts.items():
        if value:
            defaults[str(key)] = str(value)
    return defaults


def _log_failure_summary(path: Any) -> str:
    if not path:
        return ""
    log_path = Path(str(path))
    if not log_path.exists():
        return ""
    try:
        lines = log_path.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return ""
    exception_prefixes = (
        "TypeError:",
        "ValueError:",
        "KeyError:",
        "FileNotFoundError:",
        "ModuleNotFoundError:",
        "ImportError:",
        "RuntimeError:",
        "AttributeError:",
        "NameError:",
        "SyntaxError:",
    )
    for line in reversed(lines):
        stripped = line.strip()
        if stripped.startswith(exception_prefixes):
            return stripped
    for line in reversed(lines):
        stripped = line.strip()
        if "Traceback (most recent call last)" in stripped:
            return "Traceback found in train.log; see full log for stack trace"
    return ""


def _feature_names(plan: dict[str, Any]) -> list[str]:
    names: list[str] = []
    for change in plan.get("changes", []) or []:
        name = change.get("feature_name") if isinstance(change, dict) else None
        if name:
            names.append(str(name))
    return names


def _plan_metric(experiment_plan: dict[str, Any]) -> str | None:
    definition = experiment_plan.get("evaluation_metric")
    if isinstance(definition, dict):
        return definition.get("objective_label") or definition.get("decision_metric")
    return None


def _display_metric(metric: Any, experiment_plan: dict[str, Any] | None = None) -> str:
    label = str(metric or "")
    plan = experiment_plan or {}
    definition = plan.get("evaluation_metric") if isinstance(plan.get("evaluation_metric"), dict) else {}
    decision_metric = str(definition.get("decision_metric") or "").lower()
    status = str(definition.get("metric_definition_status") or definition.get("status") or "").lower()
    formula = definition.get("metric_formula") or definition.get("formula")
    has_source = bool(definition.get("metric_definition_source") or definition.get("sources"))
    if (status.startswith("resolved") or has_source or formula) and label and decision_metric and label.lower() != decision_metric:
        suffix = f"，源码公式：{formula}" if formula else ""
        return f"{label}（按 Agent1 源码解析映射到 {decision_metric}{suffix}）"
    if status == "unresolved" and label:
        return f"{label}（指标口径未解析）"
    return label


def _sample_count_text(metric_comparison: dict[str, Any]) -> str:
    if metric_comparison.get("sample_count_consistent") is False:
        return (
            f"口径风险：评测样本数不一致，"
            f"{_fmt(metric_comparison.get('old_sample_count'))} -> {_fmt(metric_comparison.get('new_sample_count'))}。"
        )
    if metric_comparison.get("old_sample_count") is not None or metric_comparison.get("new_sample_count") is not None:
        return (
            f"口径检查：评测样本数一致，"
            f"{_fmt(metric_comparison.get('old_sample_count'))} -> {_fmt(metric_comparison.get('new_sample_count'))}。"
        )
    return "口径检查：未发现评测样本数不一致信号。"


def _metric_value(metrics: dict[str, Any], metric_name: str) -> Any:
    if not metric_name:
        return None
    key = metric_name.lower()
    if key in metrics:
        return metrics[key]
    if metric_name in metrics:
        return metrics[metric_name]
    return None


def _metric_key(metric: Any) -> str:
    label = str(metric or "").strip().lower()
    for separator in ("（", "("):
        label = label.split(separator, 1)[0].strip()
    return label


def _numeric_delta(old_value: Any, new_value: Any) -> float | None:
    old_float = _to_float(old_value)
    new_float = _to_float(new_value)
    if old_float is None or new_float is None:
        return None
    return new_float - old_float


def _numeric_abs_delta(old_value: Any, new_value: Any) -> float | None:
    old_float = _to_float(old_value)
    new_float = _to_float(new_value)
    if old_float is None or new_float is None:
        return None
    return abs(new_float) - abs(old_float)


def _sample_count(metrics: dict[str, Any]) -> Any:
    for key in ("sample_count", "rows"):
        if key in metrics:
            return metrics[key]
    return None


def _same_value(left: Any, right: Any) -> bool:
    left_float = _to_float(left)
    right_float = _to_float(right)
    if left_float is not None and right_float is not None:
        return left_float == right_float
    return str(left) == str(right)


def _delta_text(value: Any) -> str:
    number = _to_float(value)
    if number is None:
        return "变化 n/a"
    if number > 0:
        return f"升高 {_fmt(number)}"
    if number < 0:
        return f"降低 {_fmt(abs(number))}"
    return "持平"


def _metric_result_text(metric: Any, delta: Any) -> str:
    number = _to_float(delta)
    key = _metric_key(metric)
    if number is None:
        return "主指标变化不可判断"
    if key == "wape":
        if number > 0:
            return "越低越好的 WAPE 变差"
        if number < 0:
            return "越低越好的 WAPE 有改善但改善幅度不足"
        return "越低越好的 WAPE 没有变化"
    if number > 0:
        return "主指标升高"
    if number < 0:
        return "主指标降低"
    return "主指标没有变化"


def _bias_direction(value: Any) -> str:
    number = _to_float(value)
    if number is None:
        return "n/a"
    if number < 0:
        return "整体低估"
    if number > 0:
        return "整体高估"
    return "无明显整体偏差"


def _decision_reason_text(reason: str) -> str:
    translations = {
        "wape improved enough and bias stayed within threshold": "WAPE 改善达到 keep 阈值，且 Bias 未超过允许恶化范围",
        "train or evaluation failed": "训练或评测失败",
        "wape improvement is below threshold": "WAPE 改善幅度未达到 keep 阈值",
        "bias regression exceeds threshold": "Bias 恶化超过允许阈值",
        "llm objective improved": "模型选择的主优化目标已改善",
        "llm objective did not improve": "模型选择的主优化目标未改善",
        "agent2 code generation failed": "Agent2 代码生成失败，未进入训练",
        "feature change was not applied": "本轮特征改动未被真实训练代码消费",
    }
    return translations.get(reason, reason)


def _change_summary(changes: Any) -> str:
    if not changes:
        return "未识别到具体特征改动。"
    parts = []
    for change in changes:
        if not isinstance(change, dict):
            continue
        name = change.get("feature_name", "unknown_feature")
        action = change.get("action", "modify_feature")
        cli_args = change.get("cli_args") or []
        suffix = f"（开关：{', '.join(map(str, cli_args))}）" if cli_args else ""
        parts.append(f"{action} `{name}`{suffix}")
    return "；".join(parts) if parts else "未识别到具体特征改动。"


def _join_or_na(values: Any) -> str:
    if not values:
        return "n/a"
    if isinstance(values, list):
        return "、".join(str(value) for value in values)
    return str(values)


def _agent2_rejection_summary(values: Any) -> str:
    items = values if isinstance(values, list) else []
    parts: list[str] = []
    for item in items[:3]:
        if not isinstance(item, dict):
            continue
        path = item.get("path") or "n/a"
        category = item.get("category") or "unknown"
        reason = item.get("reason") or "n/a"
        parts.append(f"{category} {path}: {reason}")
    if not parts:
        return ""
    return "关键 rejection：" + "；".join(parts) + "。"


def _bullet_lines(values: Any, fallback: str = "暂无足够证据。") -> list[str]:
    items = values if isinstance(values, list) else []
    if not items:
        items = [fallback]
    return [f"- {item}" for item in items]


def _numbered_lines(values: Any) -> list[str]:
    items = values if isinstance(values, list) else []
    if not items:
        items = ["复核附录产物后再决定下一轮实验。"]
    return [f"{idx}. {item}" for idx, item in enumerate(items, start=1)]


def _artifact_lines(values: Any) -> list[str]:
    if not isinstance(values, dict) or not values:
        return ["- 完整产物索引：artifact_index.md"]
    return [f"- {name}：{path}" for name, path in values.items()]


def _fmt(value: Any) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value)


def _to_float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _remove_forbidden_baseline_phrases(text: str) -> str:
    replacements = {
        "better than baseline": "证明新方案有效",
        "worse than baseline": "未能证明新方案有效",
        "优于 baseline": "证明新方案有效",
        "低于 baseline": "未能证明新方案有效",
        "baseline_wape": "当前最佳方案 WAPE",
        "compared with baseline": "相对当前最佳方案",
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    return text


def _remove_unresolved_placeholders(text: str) -> str:
    return re.sub(r"\$\{[^}\n]+\}", "n/a", text)
