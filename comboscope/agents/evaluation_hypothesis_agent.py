from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import yaml
from pydantic import ValidationError

from comboscope.core.schemas import ExperimentPlan, FeatureHypothesis
from comboscope.runtime.artifact_adapter import read_csv_records, write_json
from comboscope.runtime.yaml_utils import append_yaml_output_contract, safe_load_yaml_mapping, strip_code_fence


FEATURE_HYPOTHESIS_REQUIRED_KEYS = {"target_problem", "hypothesis", "evidence", "proposed_features"}
AGENT1_PLANNING_TIMEOUT = (30, 600)


def build_problem_context(
    *,
    metrics_path: str | Path,
    scene_metrics_path: str | Path,
    badcase_summary_path: str | Path,
    output_path: str | Path,
) -> dict[str, Any]:
    metrics = json.loads(Path(metrics_path).read_text(encoding="utf-8"))
    scenes = read_csv_records(scene_metrics_path, limit=20)
    badcase_summary = json.loads(Path(badcase_summary_path).read_text(encoding="utf-8"))
    high_target_under = [
        row
        for row in scenes
        if (
            ("high_target" in str(row.get("scene", "")) or "high_sales" in str(row.get("scene", "")))
            and "underestimate" in str(row.get("scene", ""))
        )
    ]
    main_problem = "high_target_underestimate" if high_target_under else "general_forecast_error"
    evidence = [
        f"overall_wape={metrics.get('wape')}",
        f"overall_bias={metrics.get('bias')}",
    ]
    if high_target_under:
        evidence.append(f"high_target_underestimate_scene={high_target_under[0]}")
    if badcase_summary.get("counts", {}).get("top_underestimate"):
        evidence.append("top_underestimate badcases are available")
    context = {
        "main_problem": main_problem,
        "affected_scenes": ["high_target", "underestimate"] if high_target_under else [],
        "error_direction": "underestimate" if high_target_under else "mixed",
        "evidence": evidence,
        "candidate_causes": ["recent target momentum may be underrepresented"] if high_target_under else [],
        "feature_opportunities": ["entity_rolling_7d_mean"] if high_target_under else [],
    }
    return write_json(output_path, context)


def generate_feature_hypothesis(problem_context: dict[str, Any], skill_text: str, llm_client: Any) -> dict[str, Any]:
    prompt = yaml.safe_dump(
        {
            "problem_context": problem_context,
            "rules": skill_text,
            "cli_args_contract": _cli_args_contract_rules(),
        },
        allow_unicode=True,
        sort_keys=False,
    )
    return _complete_feature_hypothesis(
        llm_client,
        "Return only YAML matching FeatureHypothesis.",
        prompt,
        agent="Agent1",
        step="GenerateExperimentPlan",
    )


def generate_experiment_plan(feature_hypothesis: dict[str, Any], trial_id: str) -> dict[str, Any]:
    feature = feature_hypothesis["proposed_features"][0]
    plan = {
        "trial_id": trial_id,
        "target_problem": feature_hypothesis["target_problem"],
        "hypothesis": {
            "description": feature_hypothesis["hypothesis"],
            "evidence": feature_hypothesis["evidence"],
        },
        "editable_files": ["benchmark/feature_config.yaml", "benchmark/train.py"],
        "changes": [feature],
        "expected_effect": "reduce high-value underestimation",
        "risk": "may overfit recent demand spikes",
    }
    return ExperimentPlan.model_validate(plan).model_dump()


def generate_real_feature_hypothesis(problem_context: dict[str, Any], skill_text: str, llm_client: Any) -> dict[str, Any]:
    prompt = yaml.safe_dump(
        {
            "scenario": problem_context.get("scenario", "generic_forecast"),
            "model_family": problem_context.get("model_family", "unknown"),
            "objective": problem_context.get("objective", "unknown"),
            "allowed_experiment_type": "feature add/remove only",
            "priority_contract": [
                "Order proposed_features by execution priority; proposed_features[0] is the single highest-priority recommendation Agent2 will execute.",
                "Write hypothesis as the testable claim for proposed_features[0] only; do not combine lower-priority alternatives into the top-level hypothesis.",
                "If the highest-priority recommendation needs multiple coordinated code changes, keep them together in proposed_features[0] via construction, field_sources, code_locations, cli_args, and validation_metrics.",
                "Do not split mandatory parts of the top recommendation into lower-priority proposed_features, because Agent2 will ignore lower-priority suggestions in a one-trial run.",
                "Describe lower-priority alternatives only in their own proposed_features entries, not in the top-level hypothesis.",
            ],
            "cli_args_contract": _cli_args_contract_rules(),
            "problem_context": problem_context,
            "rules": skill_text,
        },
        allow_unicode=True,
        sort_keys=False,
    )
    return _complete_feature_hypothesis(
        llm_client,
        "Return only YAML matching FeatureHypothesis for a generic forecast experiment.",
        prompt,
        agent="Agent1",
        step="GenerateExperimentPlan",
    )


def generate_real_experiment_plan(
    feature_hypothesis: dict[str, Any],
    trial_id: str,
    generated_train_path: str,
    problem_context: dict[str, Any] | None = None,
) -> dict[str, Any]:
    problem_context = problem_context or {}
    candidates = _candidate_experiments(feature_hypothesis)
    candidates = _normalize_candidate_experiments_for_existing_capabilities(candidates, problem_context)
    selected = _highest_priority_candidate(candidates)
    plan = {
        "trial_id": trial_id,
        "scenario": problem_context.get("scenario", "generic_forecast"),
        "model_family": problem_context.get("model_family", "unknown"),
        "objective": problem_context.get("objective", "unknown"),
        "target_problem": feature_hypothesis["target_problem"],
        "hypothesis": {
            "description": _selected_candidate_hypothesis_description(feature_hypothesis, selected),
            "evidence": _selected_candidate_evidence(feature_hypothesis, selected),
        },
        "editable_files": _real_editable_files(trial_id, selected["feature_actions"], _source_entrypoint(problem_context)),
        "changes": selected["feature_actions"],
        "expected_effect": selected["expected_effect"],
        "risk": selected["risk"],
        "source_entrypoint": _source_entrypoint(problem_context),
        "generated_train_path": generated_train_path,
        "output_contract": problem_context.get("output_contract", "forecast_outputs_to_standard_metrics"),
        "evaluation_metric": _evaluation_metric(problem_context),
        "candidate_experiments": candidates,
    }
    return ExperimentPlan.model_validate(plan).model_dump()


def _real_editable_files(trial_id: str, feature_actions: list[dict[str, Any]], source_entrypoint: str) -> list[str]:
    files = [f"runs/{trial_id}/code/train.py"]
    source_entrypoint_name = Path(source_entrypoint).name
    for action in feature_actions:
        for location in action.get("code_locations", []) or []:
            name = Path(str(location)).name
            if name.endswith(".py") and name not in {"train.py", source_entrypoint_name}:
                files.append(f"runs/{trial_id}/code/{name}")
    return list(dict.fromkeys(files))


def write_agent1_program(path: str | Path, problem_context: dict[str, Any], plan: dict[str, Any]) -> None:
    lines = [
        "# Agent1 Program",
        "",
        "## Scenario",
        f"- scenario: {plan.get('scenario') or problem_context.get('scenario', 'generic_forecast')}",
        f"- model_family: {plan.get('model_family') or problem_context.get('model_family', 'unknown')}",
        f"- objective: {plan.get('objective') or problem_context.get('objective', 'unknown')}",
        "",
        "## Current Problem",
        f"- main_problem: {problem_context.get('main_problem')}",
        f"- error_direction: {problem_context.get('error_direction')}",
        "",
        "## Evidence",
    ]
    lines.extend(f"- {item}" for item in problem_context.get("evidence", []))
    lines.extend(
        [
            "",
            "## Selected Experiment",
            f"- trial_id: {plan.get('trial_id')}",
            f"- expected_effect: {plan.get('expected_effect')}",
            f"- risk: {plan.get('risk')}",
            "",
            "## Boundaries",
            "- Do not change model family, objective, label, data split, or metric definitions.",
            "- Agent2 may modify only copied Python files under the trial code/ directory.",
            "- Original experiment source code must remain unchanged.",
        ]
    )
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_yaml(path: str | Path, value: dict[str, Any]) -> None:
    Path(path).write_text(yaml.safe_dump(value, allow_unicode=True, sort_keys=False), encoding="utf-8")


def write_analysis_report(path: str | Path, context: dict[str, Any], hypothesis: dict[str, Any]) -> None:
    lines = [
        "# Agent1 Analysis Report",
        "",
        f"- main_problem: {context.get('main_problem')}",
        f"- error_direction: {context.get('error_direction')}",
        f"- hypothesis: {hypothesis.get('hypothesis')}",
        "",
        "## Evidence",
    ]
    lines.extend(f"- {item}" for item in context.get("evidence", []))
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")


def _cli_args_contract_rules() -> list[str]:
    return [
        "cli_args must contain only real argv tokens.",
        "Each cli_args item must be a single token without spaces, tabs, newlines, or explanatory parentheses.",
        "Use forms like ['--enable_sparsity_rolling'] or ['--rolling_windows', '7', '14', '30'].",
        "Do not write natural-language descriptions in cli_args; put explanations in construction.",
    ]



def _evaluation_metric(problem_context: dict[str, Any]) -> dict[str, Any]:
    for key in ("metric_definition", "evaluation_metric"):
        value = problem_context.get(key)
        if isinstance(value, dict) and value:
            return value
    return {}


def _complete_yaml_mapping(
    llm_client: Any,
    system_prompt: str,
    prompt: str,
    *,
    agent: str,
    step: str,
    timeout: int | float | tuple[int | float, int | float] | None = None,
) -> dict[str, Any]:
    raw = _complete_llm_text(llm_client, system_prompt, prompt, agent=agent, step=step, timeout=timeout)
    return _parse_yaml_mapping(raw, agent=agent, step=step)


def _complete_llm_text(
    llm_client: Any,
    system_prompt: str,
    prompt: str,
    *,
    agent: str,
    step: str,
    timeout: int | float | tuple[int | float, int | float] | None = None,
) -> str:
    system_prompt, prompt = append_yaml_output_contract(system_prompt, prompt)
    if hasattr(llm_client, "complete_with_usage"):
        try:
            result = llm_client.complete_with_usage(system_prompt, prompt, agent=agent, step=step, timeout=timeout)
        except TypeError:
            result = llm_client.complete_with_usage(system_prompt, prompt, agent=agent, step=step)
        if not getattr(result, "success", False) or not getattr(result, "content", ""):
            raise RuntimeError(f"{agent} {step} LLM call failed or returned empty content")
        return str(result.content)
    else:
        if callable(getattr(llm_client, "available", None)) and not llm_client.available():
            raise RuntimeError(f"{agent} {step} LLM client is unavailable")
        raw = llm_client.complete(system_prompt, prompt)
        if not raw:
            raise RuntimeError(f"{agent} {step} LLM call failed or returned empty content")
        return str(raw)


def _parse_yaml_mapping(raw: str, *, agent: str, step: str) -> dict[str, Any]:
    return safe_load_yaml_mapping(str(raw), agent=agent, step=step)


def _complete_feature_hypothesis(llm_client: Any, system_prompt: str, prompt: str, *, agent: str, step: str) -> dict[str, Any]:
    raw = _complete_llm_text(llm_client, system_prompt, prompt, agent=agent, step=step, timeout=AGENT1_PLANNING_TIMEOUT)
    try:
        data = _parse_yaml_mapping(raw, agent=agent, step=step)
    except ValueError as first_error:
        repaired = _repair_feature_hypothesis_yaml(
            llm_client,
            original_prompt=prompt,
            invalid_data={"raw_response": _truncate_text(raw, 8000)},
            validation_error=first_error,
            system_prompt=system_prompt,
            agent=agent,
            step=step,
            timeout=AGENT1_PLANNING_TIMEOUT,
        )
        return FeatureHypothesis.model_validate(_normalize_feature_hypothesis_mapping(repaired)).model_dump()
    return _validate_feature_hypothesis_with_repair(
        data,
        llm_client=llm_client,
        original_prompt=prompt,
        system_prompt=system_prompt,
        agent=agent,
        step=step,
    )


def _validate_feature_hypothesis_with_repair(
    data: dict[str, Any],
    *,
    llm_client: Any,
    original_prompt: str,
    system_prompt: str,
    agent: str,
    step: str,
) -> dict[str, Any]:
    normalized = _normalize_feature_hypothesis_mapping(data)
    try:
        return FeatureHypothesis.model_validate(normalized).model_dump()
    except ValidationError as first_error:
        repaired = _repair_feature_hypothesis_yaml(
            llm_client,
            original_prompt=original_prompt,
            invalid_data=data,
            validation_error=first_error,
            system_prompt=system_prompt,
            agent=agent,
            step=step,
            timeout=AGENT1_PLANNING_TIMEOUT,
        )
        return FeatureHypothesis.model_validate(_normalize_feature_hypothesis_mapping(repaired)).model_dump()


def _normalize_feature_hypothesis_mapping(data: dict[str, Any]) -> dict[str, Any]:
    if FEATURE_HYPOTHESIS_REQUIRED_KEYS & set(data):
        return _normalize_feature_hypothesis_fields(data)
    wrapped = data.get("feature_hypothesis")
    if len(data) == 1 and isinstance(wrapped, dict):
        return _normalize_feature_hypothesis_fields(wrapped)
    return data


def _normalize_feature_hypothesis_fields(data: dict[str, Any]) -> dict[str, Any]:
    normalized = dict(data)
    if "evidence" in normalized:
        normalized["evidence"] = _normalize_evidence_items(normalized["evidence"])
    proposed_features = normalized.get("proposed_features")
    if isinstance(proposed_features, list):
        normalized_features = []
        for feature in proposed_features:
            if isinstance(feature, dict):
                normalized_feature = dict(feature)
                if "evidence" in normalized_feature:
                    normalized_feature["evidence"] = _normalize_evidence_items(normalized_feature["evidence"])
                normalized_features.append(normalized_feature)
            else:
                normalized_features.append(feature)
        normalized["proposed_features"] = normalized_features
    return normalized


def _normalize_evidence_items(value: Any) -> Any:
    if not isinstance(value, list):
        return value
    normalized = []
    for item in value:
        if isinstance(item, str):
            normalized.append(item)
        elif isinstance(item, dict) and len(item) == 1:
            key, item_value = next(iter(item.items()))
            if isinstance(item_value, str):
                normalized.append(f"{key}: {item_value}")
            else:
                normalized.append(item)
        else:
            normalized.append(item)
    return normalized


def _repair_feature_hypothesis_yaml(
    llm_client: Any,
    *,
    original_prompt: str,
    invalid_data: dict[str, Any],
    validation_error: BaseException,
    system_prompt: str,
    agent: str,
    step: str,
    timeout: int | float | tuple[int | float, int | float] | None = None,
) -> dict[str, Any]:
    repair_prompt = yaml.safe_dump(
        {
            "task": "Repair the previous YAML so it matches FeatureHypothesis exactly. Return only YAML.",
            "strict_schema": {
                "target_problem": "required string",
                "hypothesis": "required string",
                "evidence": ["required non-empty list of evidence strings"],
                "proposed_features": [
                    {
                        "action": "add_feature or remove_feature",
                        "feature_name": "required string",
                        "feature_type": "optional string",
                        "cli_args": ["optional real argv tokens only, e.g. --flag or value"],
                        "field_sources": ["optional source columns"],
                        "construction": "optional construction description",
                        "code_locations": ["optional relative Python files"],
                        "validation_metrics": ["optional metric names"],
                    }
                ],
                "confidence": "optional low, medium, or high",
            },
            "hard_rules": [
                "Do not wrap the result under feature_hypothesis.",
                "The YAML top level must contain target_problem, hypothesis, evidence, and proposed_features.",
                "hypothesis must describe proposed_features[0] only; do not combine independent lower-priority alternatives into it.",
                "If there are lower-priority proposed_features, describe them only in their own construction fields.",
                "Use only the evidence and context from the original prompt.",
                "Do not invent business-specific defaults or fallback feature ideas.",
                "cli_args must be a list of real argv tokens only: no whitespace inside a token, no parentheses, and no explanatory text.",
                "Put any explanation for a CLI flag in construction, not in cli_args.",
                "Quote evidence strings that contain ':' so YAML does not parse them as mappings.",
            ],
            "validation_error": str(validation_error),
            "invalid_yaml_data": invalid_data,
            "original_prompt": original_prompt,
        },
        allow_unicode=True,
        sort_keys=False,
    )
    return _complete_yaml_mapping(
        llm_client,
        system_prompt + " Repair the schema only; do not add a wrapper key.",
        repair_prompt,
        agent=agent,
        step=step,
        timeout=timeout,
    )


def _truncate_text(value: str, limit: int) -> str:
    if len(value) <= limit:
        return value
    return value[: max(limit - 80, 0)] + f"\n# ... truncated {len(value) - limit} chars ..."


def _strip_code_fence(text: str) -> str:
    return strip_code_fence(text)


def _candidate_experiments(feature_hypothesis: dict[str, Any]) -> list[dict[str, Any]]:
    evidence = feature_hypothesis.get("evidence") or ["Agent1 forecast evidence"]
    candidates = []
    for index, action in enumerate(feature_hypothesis.get("proposed_features", []), start=1):
        feature_name = action["feature_name"]
        verb = "Remove" if str(action.get("action", "")).startswith("remove") else "Enable"
        candidates.append(
            {
                "experiment_id": f"exp_{feature_name}",
                "title": f"{verb} {feature_name}",
                "priority": index,
                "feature_actions": [action],
                "evidence": evidence,
                "expected_effect": f"validate whether {feature_name} improves the observed forecast error pattern",
                "risk": f"{feature_name} may overfit local patterns or regress other forecast slices",
            }
        )
    return candidates[:3]


def _selected_candidate_hypothesis_description(
    feature_hypothesis: dict[str, Any],
    selected: dict[str, Any],
) -> str:
    actions = [action for action in selected.get("feature_actions", []) or [] if isinstance(action, dict)]
    if not actions:
        return str(feature_hypothesis.get("hypothesis") or selected.get("title") or "Selected feature experiment")
    action_summaries = [_feature_action_hypothesis_summary(action) for action in actions]
    target_problem = str(feature_hypothesis.get("target_problem") or "").strip()
    suffix = f" for {target_problem}" if target_problem else ""
    return "Selected experiment tests " + "; ".join(action_summaries) + suffix + "."


def _feature_action_hypothesis_summary(action: dict[str, Any]) -> str:
    feature_name = str(action.get("feature_name") or "selected_feature").strip()
    pieces = [feature_name]
    cli_args = [str(item) for item in action.get("cli_args", []) or [] if str(item).strip()]
    if cli_args:
        pieces.append("cli_args: " + " ".join(cli_args))
    construction = str(action.get("construction") or "").strip()
    if construction:
        pieces.append("construction: " + construction)
    return " (".join([pieces[0], "; ".join(pieces[1:]) + ")"]) if len(pieces) > 1 else pieces[0]


def _selected_candidate_evidence(feature_hypothesis: dict[str, Any], selected: dict[str, Any]) -> list[str]:
    evidence: list[str] = []
    for action in selected.get("feature_actions", []) or []:
        if isinstance(action, dict):
            evidence.extend(str(item) for item in action.get("evidence", []) or [] if str(item).strip())
    evidence.extend(str(item) for item in selected.get("evidence", []) or [] if str(item).strip())
    evidence.extend(str(item) for item in feature_hypothesis.get("evidence", []) or [] if str(item).strip())
    return list(dict.fromkeys(evidence)) or ["Agent1 forecast evidence"]


def _highest_priority_candidate(candidates: list[dict[str, Any]]) -> dict[str, Any]:
    return min(candidates, key=lambda item: _candidate_priority(item))


def _candidate_priority(candidate: dict[str, Any]) -> tuple[int, str]:
    raw_priority = candidate.get("priority", 9999)
    try:
        priority = int(raw_priority)
    except Exception:
        priority = 9999
    return priority, str(candidate.get("experiment_id") or candidate.get("title") or "")


def _normalize_candidate_experiments_for_existing_capabilities(
    candidates: list[dict[str, Any]],
    problem_context: dict[str, Any],
) -> list[dict[str, Any]]:
    default_windows = _rolling_windows_default_from_context(problem_context)
    if not default_windows:
        return candidates
    normalized: list[dict[str, Any]] = []
    for candidate in candidates:
        actions = []
        changed = False
        for action in candidate.get("feature_actions", []) or []:
            normalized_action = dict(action)
            windows = _rolling_windows_from_feature_name(str(normalized_action.get("feature_name") or ""))
            if windows and not (set(windows) - set(default_windows)):
                replacement = _expanded_rolling_windows(default_windows)
                normalized_action["feature_name"] = f"extend_rolling_windows_{'_'.join(str(item) for item in replacement)}"
                normalized_action["cli_args"] = ["--rolling_windows", *[str(item) for item in replacement]]
                normalized_action["construction"] = (
                    f"Source already enables rolling windows {default_windows}; vary existing CLI value to {replacement}."
                )
                changed = True
            actions.append(normalized_action)
        if changed:
            feature_name = str(actions[0].get("feature_name"))
            candidate = {
                **candidate,
                "experiment_id": f"exp_{feature_name}",
                "title": f"Enable {feature_name}",
                "feature_actions": actions,
                "expected_effect": f"validate whether {feature_name} improves the observed forecast error pattern",
            }
        normalized.append(candidate)
    return normalized


def _rolling_windows_default_from_context(problem_context: dict[str, Any]) -> list[int]:
    defaults = problem_context.get("argparse_defaults") or problem_context.get("cli_defaults") or {}
    if not isinstance(defaults, dict):
        return []
    for key in ("--rolling_windows", "--rolling-windows", "rolling_windows"):
        value = defaults.get(key)
        if isinstance(value, list) and all(isinstance(item, int) for item in value):
            return list(value)
    return []


def _rolling_windows_from_feature_name(feature_name: str) -> list[int]:
    lowered = feature_name.lower()
    if "rolling" not in lowered:
        return []
    return [int(item) for item in re.findall(r"(\d+)\s*d", lowered)]


def _expanded_rolling_windows(default_windows: list[int]) -> list[int]:
    for candidate in [1, 14, 30]:
        if candidate not in default_windows:
            return sorted(set(default_windows + [candidate]))
    return sorted(set(default_windows + [max(default_windows) * 2]))


def _source_entrypoint(problem_context: dict[str, Any]) -> str:
    explicit = problem_context.get("source_entrypoint")
    if explicit:
        return str(explicit)
    candidates = problem_context.get("entrypoint_candidates") or []
    if candidates:
        return str(candidates[0])
    return "train.py"
