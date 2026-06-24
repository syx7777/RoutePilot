from __future__ import annotations

import json
import csv
import re
from pathlib import Path
from typing import Any, TypedDict

import yaml
from langgraph.graph import END, START, StateGraph

from comboscope.agents.evaluation_hypothesis_agent import (
    generate_experiment_plan,
    generate_feature_hypothesis,
    generate_real_experiment_plan,
    generate_real_feature_hypothesis,
    write_agent1_program,
    write_analysis_report,
    write_yaml,
)
from comboscope.agents.report_agent import write_experiment_review
from comboscope.core.compare import compare_metrics
from comboscope.core.experiment_plan import apply_experiment_plan, write_review
from comboscope.core.real_experiment_runner import run_real_experiment, validate_train_command_argparse_choices
from comboscope.core.experiment_runner import run_experiment
from comboscope.core.reports import build_final_report_context, write_final_report, write_final_report_context
from comboscope.core.rollback import rollback_change
from comboscope.runtime.artifact_adapter import metrics_csv_to_json, read_csv_records, summarize_badcases, write_json
from comboscope.runtime.artifact_contract import standardize_from_contract, write_artifact_contract
from comboscope.runtime.agent_run_recorder import AgentRunRecorder
from comboscope.runtime.llm_client import LLMClient, create_llm_client
from comboscope.runtime.skill_runner import SkillRunner
from comboscope.runtime.trace_writer import TraceWriter
from comboscope.runtime.trial_archive import archive_trial_files
from comboscope.runtime.yaml_utils import append_yaml_output_contract, safe_load_yaml_mapping, strip_code_fence, yaml_control_char_summary


AGENT2_COMMAND_PLACEHOLDERS = {
    "{python}",
    "{train_py}",
    "{trial_id}",
    "{trial_dir}",
    "{real_output_dir}",
    "{output_dir}",
}
AGENT2_PLANNING_TIMEOUT = (30, 600)
AGENT1_REPORT_TIMEOUT = (30, 600)
AGENT1_FINAL_REPORT_TIMEOUT = (30, 600)
AGENT1_REPORT_BADCASE_LIMIT = 20
AGENT1_REPORT_SCENE_LIMIT = 30
AGENT1_REPORT_CANDIDATE_LIMIT = 4
AGENT1_REPORT_TEXT_LIMIT = 4000

AGENT1_PLANNING_SKILLS = [
    "forecast-optimization-advisor",
    "forecast-optimization-case-reference",
]
AGENT1_CASE_REFERENCE_PATH = "references/package-lgbm-optimization-case.md"


class ComboScopeState(TypedDict, total=False):
    experiment_dir: str
    output_dir: str
    ask: str
    trial_id: str
    repo_root: str
    model: str
    llm_provider: str
    original_command: str
    decision: str
    errors: list[str]
    artifact_summary_path: str
    artifact_contract_path: str
    scan_result_path: str
    old_metrics_path: str
    scene_metrics_path: str
    badcases_path: str
    problem_context_path: str
    badcase_diagnosis_path: str
    feature_hypothesis_path: str
    experiment_plan_path: str
    run_status_path: str
    new_metrics_path: str
    metric_comparison_path: str
    review_result_path: str
    forecast_report_path: str
    final_report_path: str
    backup_info: dict[str, Any]
    real_mode: bool
    artifact_manifest_path: str
    column_mapping_path: str
    standardized_prediction_path: str
    standardized_actual_path: str
    agent1_program_path: str
    candidate_experiments_path: str
    agent1_skill_route_path: str
    agent2_execution_plan_path: str
    output_contract_path: str
    source_evaluation_context_path: str
    source_evaluation_context: dict[str, Any]


def _paths(state: ComboScopeState) -> dict[str, Path]:
    output = Path(state["output_dir"])
    output.mkdir(parents=True, exist_ok=True)
    return {
        "output": output,
        "trace": output / "trace.jsonl",
        "scan": output / "scan_result.json",
        "artifact": output / "artifact_summary.json",
        "artifact_contract": output / "artifact_contract.json",
        "manifest": output / "artifact_manifest.json",
        "column_mapping": output / "column_mapping.json",
        "standardized_prediction": output / "standardized_prediction.csv",
        "standardized_actual": output / "standardized_actual.csv",
        "code": output / "code_analysis.json",
        "log": output / "log_summary.json",
        "metrics_csv": output / "metrics_summary.csv",
        "metrics_json": output / "metrics.json",
        "scene": output / "scene_metrics.csv",
        "badcases": output / "badcases.csv",
        "badcase_summary": output / "badcase_summary.json",
        "anomaly": output / "anomaly_summary.json",
        "problem": output / "problem_context.json",
        "badcase_diagnosis": output / "badcase_diagnosis.md",
        "hypothesis": output / "feature_hypothesis.yaml",
        "plan": output / "experiment_plan.yaml",
        "analysis_report": output / "analysis_report.md",
        "agent1_program": output / "agent1_program.md",
        "agent1_skill_route": output / "agent1_skill_route.yaml",
        "candidate_experiments": output / "candidate_experiments.yaml",
        "agent2_execution_plan": output / "agent2_execution_plan.yaml",
        "source_evaluation_context": output / "source_evaluation_context.json",
        "output_contract": output / "output_contract.json",
        "run_status": output / "run_status.json",
        "new_metrics_csv": output / "new_metrics_summary.csv",
        "new_metrics": output / "new_metrics.json",
        "comparison": output / "metric_comparison.json",
        "review": output / "review_result.json",
        "experiment_review": output / "experiment_review.md",
        "report_context": output / "report_context.json",
        "final_report_context": output / "final_report_context.json",
        "forecast_report": output / "forecast_report.md",
        "suggestions": output / "optimization_suggestions.md",
        "final": output / "final_report.md",
    }


def _runner(state: ComboScopeState) -> SkillRunner:
    repo_root = Path(state.get("repo_root") or Path.cwd())
    return SkillRunner(repo_root / "skills")


def _trace(state: ComboScopeState) -> TraceWriter:
    return TraceWriter(_paths(state)["trace"])


def _recorder(state: ComboScopeState) -> AgentRunRecorder:
    return AgentRunRecorder(_paths(state)["output"])


def evaluate_and_diagnose(state: ComboScopeState) -> ComboScopeState:
    paths = _paths(state)
    with _recorder(state).step(
        "Agent1",
        "EvaluateAndDiagnose",
        artifacts={
            "scan_result": paths["scan"],
            "artifact_summary": paths["artifact"],
            "source_evaluation_context": paths["source_evaluation_context"],
            "problem_context": paths["problem"],
            "analysis_report": paths["analysis_report"],
        },
    ):
        return _evaluate_generic_and_diagnose(state)


def _evaluate_generic_and_diagnose(state: ComboScopeState) -> ComboScopeState:
    paths = _paths(state)
    trace = _trace(state)
    runner = _runner(state)
    experiment = Path(state["experiment_dir"])
    recorder = _recorder(state)
    client = create_llm_client(provider=state.get("llm_provider"), model=state.get("model"))
    scan = runner.scan_experiment(experiment)
    code = runner.analyze_code(experiment)
    select_call_start = len(getattr(client, "call_results", []) or [])
    try:
        artifact_contract = _select_artifact_contract_with_llm(
            ask=state["ask"],
            experiment=experiment,
            scan=scan,
            code=code,
            client=client,
        )
    finally:
        _record_llm_calls_since(recorder, client, select_call_start)
    write_artifact_contract(paths["artifact_contract"], artifact_contract)
    standardization = standardize_from_contract(
        artifact_contract,
        paths["standardized_prediction"],
        paths["standardized_actual"],
        base_dirs=[experiment],
    )
    artifact = {
        "ask": state["ask"],
        "mode_ready": True,
        "prediction_path": paths["standardized_prediction"].as_posix(),
        "actual_path": paths["standardized_actual"].as_posix(),
        "raw_prediction_path": standardization["prediction_path"],
        "raw_actual_path": standardization["actual_path"],
        "artifact_contract_path": paths["artifact_contract"].as_posix(),
        "standardization": standardization,
    }
    if artifact_contract.get("train_log_path"):
        log_path = _resolve_contract_path(experiment, str(artifact_contract["train_log_path"]))
        log = runner.parse_log(log_path)
        artifact["train_log_path"] = log_path.as_posix()
    else:
        log = runner.parse_log(None)
    source_evaluation_context = _build_source_evaluation_context(
        experiment=experiment,
        artifact_contract=artifact_contract,
        standardized_prediction=paths["standardized_prediction"],
        standardized_actual=paths["standardized_actual"],
        original_command=state.get("original_command", ""),
    )
    write_json(paths["source_evaluation_context"], source_evaluation_context)
    write_json(paths["scan"], scan)
    write_json(paths["artifact"], artifact)
    write_json(paths["code"], code)
    write_json(paths["log"], log)

    prediction = paths["standardized_prediction"]
    actual = paths["standardized_actual"]
    runner.calculate_metrics(prediction, actual, paths["metrics_csv"])
    old_metrics = metrics_csv_to_json(paths["metrics_csv"], paths["metrics_json"])
    runner.tag_scenes(prediction, actual, paths["scene"])
    runner.mine_badcases(prediction, actual, paths["badcases"])
    summarize_badcases(paths["badcases"], paths["badcase_summary"])
    runner.detect_anomalies(paths["metrics_csv"], paths["scene"], paths["log"], paths["anomaly"])
    diagnose_call_start = len(getattr(client, "call_results", []) or [])
    try:
        context = _generate_problem_context_with_llm(
            ask=state["ask"],
            experiment=experiment,
            artifact_contract=artifact_contract,
            scan=scan,
            code=code,
            metrics=old_metrics,
            scene_metrics=read_csv_records(paths["scene"], limit=20),
            badcase_summary=_read_json(paths["badcase_summary"]),
            client=client,
            output_md=paths["badcase_diagnosis"],
        )
    finally:
        _record_llm_calls_since(recorder, client, diagnose_call_start)
    context.update(
        {
            "experiment_dir": experiment.as_posix(),
            "code_files": scan.get("code_files", []),
            "source_entrypoint": artifact_contract.get("source_entrypoint"),
            "entrypoint_candidates": scan.get("possible_entrypoints", []),
            "model_family": artifact_contract.get("model_family", "unknown"),
            "objective": artifact_contract.get("objective", "unknown"),
            "output_contract": artifact_contract.get("output_contract", "forecast_outputs_to_standard_metrics"),
            "artifact_contract_path": paths["artifact_contract"].as_posix(),
        }
    )
    write_json(paths["problem"], context)
    trace.write("evaluate_and_diagnose", {"metrics": old_metrics, "problem": context.get("main_problem")})
    return {
        **state,
        "real_mode": True,
        "scan_result_path": paths["scan"].as_posix(),
        "artifact_summary_path": paths["artifact"].as_posix(),
        "artifact_contract_path": paths["artifact_contract"].as_posix(),
        "old_metrics_path": paths["metrics_json"].as_posix(),
        "scene_metrics_path": paths["scene"].as_posix(),
        "badcases_path": paths["badcases"].as_posix(),
        "problem_context_path": paths["problem"].as_posix(),
        "badcase_diagnosis_path": paths["badcase_diagnosis"].as_posix(),
        "source_evaluation_context_path": paths["source_evaluation_context"].as_posix(),
        "source_evaluation_context": source_evaluation_context,
    }


def _select_artifact_contract_with_llm(
    *,
    ask: str,
    experiment: Path,
    scan: dict[str, Any],
    code: dict[str, Any],
    client: LLMClient,
) -> dict[str, Any]:
    data_headers = _data_headers_for_llm(experiment, scan)
    required_schema = {
        "prediction_path": "relative or absolute CSV containing predictions",
        "actual_path": "relative or absolute CSV containing actuals; may equal prediction_path",
        "prediction_column": "column name to rename to prediction",
        "actual_column": "column name to rename to actual",
        "date_column": "optional date column",
        "id_columns": ["optional entity key columns"],
        "passthrough_columns": ["optional scene columns available before prediction"],
        "split_column": "optional split column",
        "split_value": "optional split value such as test",
        "train_log_path": "optional train log path",
        "source_entrypoint": "relative Python entrypoint for training",
        "model_family": "model family if code evidence supports it, else unknown",
        "objective": "objective if code evidence supports it, else unknown",
        "output_contract": "short generic output contract description",
    }
    prompt = yaml.safe_dump(
        {
            "task": "Select the artifacts and column contract for a generic forecasting experiment. Return only YAML.",
            "ask": ask,
            "experiment_dir": experiment.as_posix(),
            "required_schema": required_schema,
            "scan_result": scan,
            "code_analysis": {
                "entrypoints": code.get("entrypoints", []),
                "argparse_args": code.get("argparse_args", []),
                "feature_modules": code.get("feature_modules", []),
                "model_modules": code.get("model_modules", []),
            },
            "data_headers": data_headers,
        },
        allow_unicode=True,
        sort_keys=False,
    )
    contract = _complete_llm_yaml_mapping_with_repair(
        client,
        "You are ComboScope Agent1. Select artifacts from evidence only. Return YAML and do not invent paths.",
        prompt,
        agent="Agent1",
        step="SelectArtifacts",
        required_schema=required_schema,
        repair_task="Repair the artifact contract into a valid YAML mapping. Preserve selected paths, columns, and evidence-derived fields; change only YAML syntax, quoting, indentation, and block-scalar formatting.",
    )
    return _validate_or_repair_artifact_contract(
        contract,
        client=client,
        experiment=experiment,
        original_prompt=prompt,
        data_headers=data_headers,
    )


def _validate_or_repair_artifact_contract(
    contract: dict[str, Any],
    *,
    client: LLMClient,
    experiment: Path,
    original_prompt: str,
    data_headers: list[dict[str, Any]],
) -> dict[str, Any]:
    try:
        return _validated_artifact_contract(contract, experiment)
    except Exception as first_error:  # noqa: BLE001 - repair only with LLM, then validate again.
        repair_prompt = yaml.safe_dump(
            {
                "task": "Repair the artifact contract. Return only YAML with the same schema.",
                "hard_rules": [
                    "Use only listed paths and columns.",
                    "prediction_column must exist in prediction_path columns.",
                    "actual_column must exist in actual_path columns.",
                    "Do not infer business-specific file patterns.",
                    "Do not add explanations or markdown.",
                ],
                "validation_error": str(first_error),
                "invalid_contract": contract,
                "data_headers": data_headers,
                "original_prompt": original_prompt,
            },
            allow_unicode=True,
            sort_keys=False,
        )
        repaired = _complete_llm_yaml_mapping(
            client,
            "You are ComboScope Agent1. Repair only the artifact contract schema and columns. Return YAML.",
            repair_prompt,
            agent="Agent1",
            step="SelectArtifacts",
        )
        return _validated_artifact_contract(repaired, experiment)


def _validated_artifact_contract(contract: dict[str, Any], experiment: Path) -> dict[str, Any]:
    for key in ("prediction_path", "actual_path", "prediction_column", "actual_column", "source_entrypoint"):
        if not isinstance(contract.get(key), str) or not str(contract[key]).strip():
            raise ValueError(f"artifact contract missing required field: {key}")
    prediction_path = _resolve_contract_path(experiment, str(contract["prediction_path"]))
    actual_path = _resolve_contract_path(experiment, str(contract["actual_path"]))
    _resolve_contract_path(experiment, str(contract["source_entrypoint"]))
    if contract.get("train_log_path"):
        _resolve_contract_path(experiment, str(contract["train_log_path"]))
    prediction_headers = _read_csv_headers(prediction_path)
    actual_headers = _read_csv_headers(actual_path)
    if str(contract["prediction_column"]) not in prediction_headers:
        raise ValueError(
            f"prediction_column column not found: {contract['prediction_column']}; "
            f"path={prediction_path}; available={prediction_headers}"
        )
    if str(contract["actual_column"]) not in actual_headers:
        raise ValueError(
            f"actual_column column not found: {contract['actual_column']}; "
            f"path={actual_path}; available={actual_headers}"
        )
    split_column = contract.get("split_column")
    if split_column and (str(split_column) not in prediction_headers or str(split_column) not in actual_headers):
        raise ValueError(f"split_column must exist in both prediction and actual files: {split_column}")
    contract.setdefault("id_columns", [])
    contract.setdefault("passthrough_columns", [])
    contract.setdefault("model_family", "unknown")
    contract.setdefault("objective", "unknown")
    contract.setdefault("output_contract", "forecast_outputs_to_standard_metrics")
    return contract


def _generate_problem_context_with_llm(
    *,
    ask: str,
    experiment: Path,
    artifact_contract: dict[str, Any],
    scan: dict[str, Any],
    code: dict[str, Any],
    metrics: dict[str, Any],
    scene_metrics: list[dict[str, Any]],
    badcase_summary: dict[str, Any],
    client: LLMClient,
    output_md: Path,
) -> dict[str, Any]:
    required_schema = {
        "main_problem": "short generic problem id",
        "affected_scenes": ["scene labels or segments"],
        "error_direction": "underestimate, overestimate, mixed, or unknown",
        "evidence": ["specific metric/scene/badcase evidence strings"],
        "candidate_causes": ["hypotheses tied to code/log/metric evidence"],
        "feature_opportunities": ["high-level opportunities only if supported"],
        "available_fields": ["columns from artifact contract useful to Agent2"],
        "available_feature_functions": ["relevant functions/modules from code evidence"],
        "available_cli_args": ["relevant CLI args from code evidence"],
        "available_code_locations": ["relative Python files Agent2 may inspect"],
        "badcase_diagnosis_markdown": "Markdown diagnosis as a YAML block scalar string",
    }
    skill_evidence = {
        "metrics": metrics,
        "scene_metrics_sample": scene_metrics,
        "badcase_summary": badcase_summary,
        "artifact_contract": artifact_contract,
        "entrypoints": scan.get("possible_entrypoints", []),
        "code_functions": code.get("function_defs", []) or code.get("code_identifiers", []),
    }
    prompt = yaml.safe_dump(
        {
            "task": "Diagnose forecast badcases and produce the problem context for Agent1 planning. Skills are evidence only; do not invent facts.",
            "ask": ask,
            "experiment_dir": experiment.as_posix(),
            "required_schema": required_schema,
            "hard_rules": [
                "Return one YAML mapping only; do not wrap the response in Markdown fences.",
                "Do not emit standalone Markdown outside YAML.",
                "The badcase_diagnosis_markdown field must be a YAML block scalar using this exact shape: badcase_diagnosis_markdown: |",
                "Indent every line of Markdown under badcase_diagnosis_markdown by at least two spaces.",
                "Use only evidence from skill_evidence; do not invent model family, fields, files, or root causes.",
            ],
            "skill_evidence": skill_evidence,
        },
        allow_unicode=True,
        sort_keys=False,
    )
    context = _complete_llm_yaml_mapping_with_repair(
        client,
        "You are ComboScope Agent1. Produce a generic forecasting badcase diagnosis and planning context from skill evidence only.",
        prompt,
        agent="Agent1",
        step="DiagnoseBadcases",
        required_schema=required_schema,
        repair_task="Repair the badcase diagnosis response into a valid YAML mapping. Preserve the original evidence and wording; change only YAML syntax, quoting, indentation, and Markdown block-scalar formatting.",
    )
    if not isinstance(context.get("main_problem"), str) or not context["main_problem"].strip():
        raise ValueError("badcase diagnosis missing main_problem")
    if not isinstance(context.get("evidence"), list) or not context["evidence"]:
        raise ValueError("badcase diagnosis must include evidence")
    markdown = context.pop("badcase_diagnosis_markdown", "")
    if not isinstance(markdown, str) or not markdown.strip():
        raise ValueError("badcase diagnosis missing badcase_diagnosis_markdown")
    output_md.write_text(markdown.strip() + "\n", encoding="utf-8")
    context.setdefault("available_fields", _contract_fields(artifact_contract))
    context.setdefault("available_feature_functions", _available_feature_functions(code))
    context.setdefault("available_cli_args", _available_cli_args(code))
    context.setdefault("available_code_locations", _available_code_locations(code, artifact_contract.get("source_entrypoint")))
    return context


def _data_headers_for_llm(experiment: Path, scan: dict[str, Any]) -> list[dict[str, Any]]:
    candidates = []
    for key in ("data_files", "candidate_prediction_files", "actual_files", "domain_artifact_files", "metrics_files"):
        candidates.extend(scan.get(key, []) or [])
    records: list[dict[str, Any]] = []
    for rel_path in list(dict.fromkeys(str(item) for item in candidates))[:30]:
        path = _resolve_contract_path(experiment, rel_path, must_exist=False)
        if not path.exists() or path.suffix.lower() != ".csv":
            continue
        try:
            headers = _read_csv_headers(path)
        except (OSError, StopIteration, UnicodeDecodeError):
            headers = []
        records.append({"path": rel_path, "columns": [str(item).lstrip("\ufeff") for item in headers]})
    return records


def _read_csv_headers(path: Path) -> list[str]:
    last_error: UnicodeDecodeError | None = None
    for encoding in ("utf-8-sig", "utf-8", "gb18030", "gbk", "latin1"):
        try:
            with path.open(newline="", encoding=encoding) as handle:
                reader = csv.reader(handle)
                return [str(item).lstrip("\ufeff") for item in next(reader)]
        except UnicodeDecodeError as exc:
            last_error = exc
            continue
    if last_error is not None:
        raise last_error
    return []


def _build_source_evaluation_context(
    *,
    experiment: Path,
    artifact_contract: dict[str, Any],
    standardized_prediction: Path,
    standardized_actual: Path,
    original_command: str,
) -> dict[str, Any]:
    date_column = str(artifact_contract.get("date_column") or "").strip()
    split_column = str(artifact_contract.get("split_column") or "").strip()
    split_value = str(artifact_contract.get("split_value") or "").strip()
    source_entrypoint = str(artifact_contract.get("source_entrypoint") or "").strip()
    source_text = ""
    if source_entrypoint:
        source_path = _resolve_contract_path(experiment, source_entrypoint, must_exist=False)
        if source_path.exists():
            source_text = source_path.read_text(encoding="utf-8", errors="ignore")

    log_text = ""
    if artifact_contract.get("train_log_path"):
        log_path = _resolve_contract_path(experiment, str(artifact_contract["train_log_path"]), must_exist=False)
        if log_path.exists():
            log_text = log_path.read_text(encoding="utf-8", errors="ignore")

    explicit_args = _extract_preserved_eval_args(" ".join([original_command or "", log_text]))
    log_window = _extract_test_window_from_text(log_text)
    artifact_window = _standardized_window(standardized_prediction, date_column)
    if not artifact_window.get("sample_count"):
        artifact_window = _standardized_window(standardized_actual, date_column)

    test_start = log_window.get("test_start") or artifact_window.get("test_start")
    test_end = log_window.get("test_end") or artifact_window.get("test_end")
    preserved_args = dict(explicit_args)
    if test_end and "--train_eval_end" not in preserved_args and _source_supports_flag(source_text, "--train_eval_end"):
        preserved_args["--train_eval_end"] = _compact_date_arg(test_end)

    sample_count = artifact_window.get("sample_count")
    resolved = bool(sample_count and (not date_column or (test_start and test_end)))
    evidence_sources: list[str] = []
    if explicit_args:
        evidence_sources.append("original_command_or_train_log_args")
    if log_window:
        evidence_sources.append("train_log_test_window")
    if artifact_window:
        evidence_sources.append("standardized_artifact_window")
    if artifact_contract:
        evidence_sources.append("artifact_contract")

    return {
        "context_status": "resolved" if resolved else "unresolved",
        "unresolved_reason": "" if resolved else "source evaluation window or sample count could not be determined",
        "test_start": test_start,
        "test_end": test_end,
        "sample_count": sample_count,
        "date_column": date_column,
        "split_column": split_column,
        "split_value": split_value,
        "preserved_train_args": preserved_args,
        "source_entrypoint": source_entrypoint,
        "train_log_path": log_path.as_posix() if log_text and artifact_contract.get("train_log_path") else "",
        "evidence_priority": "explicit args/log window > standardized artifact window > artifact contract",
        "evidence_sources": evidence_sources,
    }


def _extract_preserved_eval_args(text: str) -> dict[str, str]:
    preserved: dict[str, str] = {}
    protected_flags = [
        "--train_eval_start",
        "--train-eval-start",
        "--train_eval_end",
        "--train-eval-end",
        "--valid_days",
        "--valid-days",
        "--test_days",
        "--test-days",
    ]
    for flag in protected_flags:
        pattern = re.compile(rf"(?<!\S){re.escape(flag)}(?:=|\s+)(?P<value>[0-9]{{8}}|[0-9]{{4}}-[0-9]{{2}}-[0-9]{{2}}|\d+)")
        match = pattern.search(text or "")
        if not match:
            continue
        canonical = "--" + flag[2:].replace("-", "_")
        preserved[canonical] = _compact_date_arg(match.group("value"))

    for key in ("train_eval_start", "train_eval_end", "valid_days", "test_days"):
        pattern = re.compile(rf"\b{key}\s*=\s*(?P<value>[0-9]{{8}}|[0-9]{{4}}-[0-9]{{2}}-[0-9]{{2}}|\d+)")
        match = pattern.search(text or "")
        if match:
            preserved.setdefault("--" + key, _compact_date_arg(match.group("value")))

    range_match = re.search(
        r"\btrain_eval_range\s*=\s*(?P<start>[0-9]{4}-[0-9]{2}-[0-9]{2})~(?P<end>[0-9]{4}-[0-9]{2}-[0-9]{2})",
        text or "",
    )
    if range_match:
        preserved.setdefault("--train_eval_start", _compact_date_arg(range_match.group("start")))
        preserved.setdefault("--train_eval_end", _compact_date_arg(range_match.group("end")))
    return preserved


def _extract_test_window_from_text(text: str) -> dict[str, str]:
    if not text:
        return {}
    patterns = [
        r"\btest_window\s*=\s*(?P<start>[0-9]{4}-[0-9]{2}-[0-9]{2})~(?P<end>[0-9]{4}-[0-9]{2}-[0-9]{2})",
        r"\btest\s*=\s*\d+\s*\[(?P<start>[0-9]{4}-[0-9]{2}-[0-9]{2})~(?P<end>[0-9]{4}-[0-9]{2}-[0-9]{2})\]",
        r"\btest\s*\[(?P<start>[0-9]{4}-[0-9]{2}-[0-9]{2}),\s*(?P<end>[0-9]{4}-[0-9]{2}-[0-9]{2})\]",
    ]
    for pattern in patterns:
        matches = list(re.finditer(pattern, text))
        if matches:
            match = matches[-1]
            return {"test_start": match.group("start"), "test_end": match.group("end")}
    return {}


def _standardized_window(path: Path, date_column: str) -> dict[str, Any]:
    if not path.exists():
        return {}
    count = 0
    dates: list[str] = []
    for encoding in ("utf-8-sig", "utf-8", "gb18030", "gbk", "latin1"):
        try:
            with path.open(newline="", encoding=encoding) as handle:
                reader = csv.DictReader(handle)
                for row in reader:
                    count += 1
                    if date_column and row.get(date_column):
                        dates.append(_normalize_date_text(str(row[date_column])))
            break
        except UnicodeDecodeError:
            count = 0
            dates = []
            continue
    result: dict[str, Any] = {"sample_count": count}
    dates = [item for item in dates if item]
    if dates:
        result["test_start"] = min(dates)
        result["test_end"] = max(dates)
    return result


def _normalize_date_text(value: str) -> str:
    text = value.strip()
    match = re.match(r"^([0-9]{4})-?([0-9]{2})-?([0-9]{2})", text)
    if not match:
        return text
    return f"{match.group(1)}-{match.group(2)}-{match.group(3)}"


def _compact_date_arg(value: str) -> str:
    text = str(value).strip()
    if re.match(r"^[0-9]{4}-[0-9]{2}-[0-9]{2}$", text):
        return text.replace("-", "")
    return text


def _source_supports_flag(source_text: str, flag: str) -> bool:
    variants = {flag, flag.replace("_", "-"), flag.replace("-", "_")}
    return any(variant in source_text for variant in variants)


def _contract_fields(contract: dict[str, Any]) -> list[str]:
    fields = [
        contract.get("date_column"),
        contract.get("prediction_column"),
        contract.get("actual_column"),
        contract.get("split_column"),
        *(contract.get("id_columns") or []),
        *(contract.get("passthrough_columns") or []),
    ]
    return [str(item) for item in dict.fromkeys(fields) if item]


def _resolve_contract_path(root: Path, value: str, *, must_exist: bool = True) -> Path:
    path = Path(value)
    if not path.is_absolute():
        path = root / path
    path = path.resolve()
    if must_exist and not path.exists():
        raise FileNotFoundError(f"LLM-selected path does not exist: {path}")
    return path


def _complete_llm_yaml_mapping(
    client: LLMClient,
    system_prompt: str,
    prompt: str,
    *,
    agent: str,
    step: str,
    max_tokens: int | None = None,
    timeout: int | float | tuple[int | float, int | float] | None = None,
    stream: bool = False,
) -> dict[str, Any]:
    kwargs: dict[str, Any] = {"agent": agent, "step": step}
    if max_tokens is not None:
        kwargs["max_tokens"] = max_tokens
    if timeout is not None:
        kwargs["timeout"] = timeout
    # YAML contracts must be parsed as one complete document. Keep them non-streaming
    # so provider-specific streaming event shapes cannot duplicate or splice text.
    _ = stream
    system_prompt, prompt = append_yaml_output_contract(system_prompt, prompt)
    result = client.complete_with_usage(system_prompt, prompt, **kwargs)
    if not result.success or not result.content:
        raise RuntimeError(f"{agent} {step} LLM call failed or returned empty content")
    try:
        parsed = _parse_llm_yaml_mapping(result.content, agent=agent, step=step)
    except ValueError as exc:
        _mark_yaml_parse_result(result, status="failed", error=str(exc), repair_attempted=False, repair_success=False)
        raise
    _mark_yaml_parse_result(result, status="success", repair_attempted=False, repair_success=False)
    return parsed


def _complete_llm_yaml_mapping_with_repair(
    client: LLMClient,
    system_prompt: str,
    prompt: str,
    *,
    agent: str,
    step: str,
    required_schema: dict[str, Any],
    repair_task: str,
    max_tokens: int | None = None,
    timeout: int | float | tuple[int | float, int | float] | None = None,
    stream: bool = False,
) -> dict[str, Any]:
    kwargs: dict[str, Any] = {"agent": agent, "step": step}
    if max_tokens is not None:
        kwargs["max_tokens"] = max_tokens
    if timeout is not None:
        kwargs["timeout"] = timeout
    # YAML contracts must be parsed as one complete document. Keep them non-streaming
    # so provider-specific streaming event shapes cannot duplicate or splice text.
    _ = stream
    system_prompt, prompt = append_yaml_output_contract(system_prompt, prompt)
    result = client.complete_with_usage(system_prompt, prompt, **kwargs)
    if not result.success or not result.content:
        raise RuntimeError(f"{agent} {step} LLM call failed or returned empty content")
    try:
        parsed = _parse_llm_yaml_mapping(result.content, agent=agent, step=step)
        _mark_yaml_parse_result(result, status="success", repair_attempted=False, repair_success=False)
        return parsed
    except ValueError as first_error:
        _mark_yaml_parse_result(result, status="failed", error=str(first_error), repair_attempted=True, repair_success=False)
        repair_prompt = yaml.safe_dump(
            {
                "task": repair_task,
                "validation_error": str(first_error),
                "required_schema": required_schema,
                "hard_rules": [
                    "Return one YAML mapping only, without Markdown fences or explanations.",
                    "Preserve the original facts, evidence, field names, and Markdown diagnosis content.",
                    "Change only YAML syntax, quoting, indentation, and block-scalar formatting.",
                    "If a value contains Markdown, put it in a YAML block scalar with | and indent its lines.",
                    "Do not add business-specific assumptions or unsupported root causes.",
                ],
                "invalid_response": _truncate_text(result.content, 12000),
            },
            allow_unicode=True,
            sort_keys=False,
        )
        repaired = _complete_llm_yaml_mapping(
            client,
            "You are ComboScope YAML repair. Return only a valid YAML mapping that matches the requested schema.",
            repair_prompt,
            agent=agent,
            step=step,
            max_tokens=max_tokens,
            timeout=timeout,
            stream=stream,
        )
        _mark_yaml_parse_result(result, status="failed", error=str(first_error), repair_attempted=True, repair_success=True)
        return repaired


def _parse_llm_yaml_mapping(content: str, *, agent: str, step: str) -> dict[str, Any]:
    return safe_load_yaml_mapping(content, agent=agent, step=step)


def _mark_yaml_parse_result(
    result: Any,
    *,
    status: str,
    error: str | None = None,
    repair_attempted: bool,
    repair_success: bool,
) -> None:
    setattr(result, "yaml_parse_status", status)
    setattr(result, "yaml_parse_error", error)
    setattr(result, "yaml_repair_attempted", bool(repair_attempted))
    setattr(result, "yaml_repair_success", bool(repair_success))
    setattr(result, "yaml_control_chars", yaml_control_char_summary(getattr(result, "content", "") or ""))


def _truncate_text(value: str, limit: int) -> str:
    if len(value) <= limit:
        return value
    return value[:limit] + "\n...[truncated]"


def _strip_code_fence(text: str) -> str:
    return strip_code_fence(text)


def _available_feature_functions(code_analysis: dict[str, Any]) -> list[str]:
    values: list[str] = []
    for key in ("feature_functions", "function_defs", "code_identifiers", "feature_modules", "model_modules", "entrypoints"):
        raw = code_analysis.get(key) or []
        if isinstance(raw, list):
            values.extend(str(item) for item in raw)
    return sorted(dict.fromkeys(values))


def _available_cli_args(code_analysis: dict[str, Any]) -> list[str]:
    raw = code_analysis.get("argparse_args") or code_analysis.get("cli_args") or []
    return sorted(dict.fromkeys(str(item) for item in raw if str(item).startswith("--"))) if isinstance(raw, list) else []


def _available_code_locations(code_analysis: dict[str, Any], source_entrypoint: str | None = None) -> list[str]:
    values: list[str] = []
    if source_entrypoint:
        values.append(str(source_entrypoint))
    for key in ("entrypoints", "feature_modules", "model_modules"):
        raw = code_analysis.get(key) or []
        if isinstance(raw, list):
            values.extend(str(item) for item in raw if str(item).endswith(".py"))
    return sorted(dict.fromkeys(values))


def _enrich_problem_context_for_real_plan(problem_context: dict[str, Any], paths: dict[str, Path]) -> dict[str, Any]:
    enriched = {**problem_context}
    if paths["code"].exists():
        code = json.loads(paths["code"].read_text(encoding="utf-8"))
        enriched.setdefault("available_feature_functions", _available_feature_functions(code))
        enriched.setdefault("available_cli_args", _available_cli_args(code))
        enriched.setdefault("available_code_locations", _available_code_locations(code, enriched.get("source_entrypoint")))
    return enriched


def _agent1_planning_skill_rules(repo_root: Path) -> str:
    manager = SkillRunner(repo_root / "skills").manager
    sections = []
    for skill_name in AGENT1_PLANNING_SKILLS:
        skill = manager.load_skill(skill_name)
        sections.append(f"# Skill: {skill.name}\n\n{skill.content.strip()}")
    reference = manager.load_reference("forecast-optimization-case-reference", AGENT1_CASE_REFERENCE_PATH)
    sections.append(
        "# Reference: forecast-optimization-case-reference/package-lgbm-optimization-case\n\n"
        "Use this reference only as transferable patterns. Do not treat case-specific business fields, "
        "paths, model names, or windows as current experiment facts.\n\n"
        f"{reference.strip()}"
    )
    return "\n\n---\n\n".join(sections)


def generate_experiment_plan_node(state: ComboScopeState) -> ComboScopeState:
    paths = _paths(state)
    recorder = _recorder(state)
    with recorder.step(
        "Agent1",
        "GenerateExperimentPlan",
        artifacts={
            "agent1_skill_route": paths["agent1_skill_route"],
            "feature_hypothesis": paths["hypothesis"],
            "experiment_plan": paths["plan"],
            "agent1_program": paths["agent1_program"],
            "forecast_report": paths["forecast_report"],
        },
    ):
        trace = _trace(state)
        repo_root = Path(state.get("repo_root") or Path.cwd())
        skill_text = _agent1_planning_skill_rules(repo_root)
        problem_context = json.loads(paths["problem"].read_text(encoding="utf-8"))
        client = create_llm_client(provider=state.get("llm_provider"), model=state.get("model"))
        llm_call_start = len(getattr(client, "call_results", []) or [])
        _write_agent1_skill_route(paths["agent1_skill_route"], problem_context)
        try:
            if state.get("real_mode"):
                problem_context = _enrich_problem_context_for_real_plan(problem_context, paths)
                write_json(paths["problem"], problem_context)
                hypothesis = generate_real_feature_hypothesis(problem_context, skill_text, client)
                plan = generate_real_experiment_plan(
                    hypothesis,
                    state["trial_id"],
                    (paths["output"] / "code" / "train.py").as_posix(),
                    problem_context,
                )
                write_yaml(paths["candidate_experiments"], {"candidate_experiments": plan["candidate_experiments"]})
                write_agent1_program(paths["agent1_program"], problem_context, plan)
            else:
                hypothesis = generate_feature_hypothesis(problem_context, skill_text, client)
                plan = generate_experiment_plan(hypothesis, state["trial_id"])
            write_yaml(paths["hypothesis"], hypothesis)
            write_yaml(paths["plan"], plan)
            write_analysis_report(paths["analysis_report"], problem_context, hypothesis)
            _build_agent1_report_context(state, paths)
            _write_llm_forecast_report(
                context_path=paths["report_context"],
                report_path=paths["forecast_report"],
                suggestions_path=paths["suggestions"],
                repo_root=repo_root,
                client=client,
            )
        finally:
            _record_llm_calls_since(recorder, client, llm_call_start)
        trace.write("generate_experiment_plan", {"trial_id": state["trial_id"], "features": [c["feature_name"] for c in plan["changes"]]})
        return {
            **state,
            "feature_hypothesis_path": paths["hypothesis"].as_posix(),
            "experiment_plan_path": paths["plan"].as_posix(),
            "agent1_program_path": paths["agent1_program"].as_posix() if state.get("real_mode") else "",
            "candidate_experiments_path": paths["candidate_experiments"].as_posix() if state.get("real_mode") else "",
            "agent1_skill_route_path": paths["agent1_skill_route"].as_posix(),
            "forecast_report_path": paths["forecast_report"].as_posix(),
        }


def run_experiment_node(state: ComboScopeState) -> ComboScopeState:
    paths = _paths(state)
    with _recorder(state).step(
        "Agent2",
        "RunExperiment",
        artifacts={
            "agent2_execution_plan": paths["agent2_execution_plan"],
            "source_evaluation_context": paths["source_evaluation_context"],
            "run_status": paths["run_status"],
        },
    ):
        trace = _trace(state)
        plan = yaml.safe_load(paths["plan"].read_text(encoding="utf-8"))
        client = create_llm_client(provider=state.get("llm_provider"), model=state.get("model"))
        llm_call_start = len(getattr(client, "call_results", []) or [])
        try:
            execution_plan = _build_agent2_execution_plan(plan, state, client)
        finally:
            _record_llm_calls_since(_recorder(state), client, llm_call_start)
        write_yaml(paths["agent2_execution_plan"], execution_plan)
        if state.get("real_mode"):
            run_call_start = len(getattr(client, "call_results", []) or [])
            try:
                run_status = run_real_experiment(
                    state["experiment_dir"],
                    paths["output"],
                    plan,
                    llm_client=client,
                    execution_plan=execution_plan,
                )
            finally:
                _record_llm_calls_since(_recorder(state), client, run_call_start)
            trace.write("run_real_experiment", {"run_status": run_status})
            return {
                **state,
                "backup_info": {},
                "run_status_path": paths["run_status"].as_posix(),
                "agent2_execution_plan_path": paths["agent2_execution_plan"].as_posix(),
                "output_contract_path": run_status.get("output_contract_path", ""),
            }

        applied = apply_experiment_plan(paths["plan"], state["experiment_dir"])
        run_status = run_experiment(state["experiment_dir"], paths["output"])
        trace.write("run_experiment", {"applied": applied["success"], "run_status": run_status})
        return {
            **state,
            "backup_info": applied["backup_info"],
            "run_status_path": paths["run_status"].as_posix(),
            "agent2_execution_plan_path": paths["agent2_execution_plan"].as_posix(),
        }


def review_result_node(state: ComboScopeState) -> ComboScopeState:
    paths = _paths(state)
    with _recorder(state).step(
        "Agent2",
        "ReviewResult",
        artifacts={
            "new_metrics": paths["new_metrics"],
            "metric_comparison": paths["comparison"],
            "review_result": paths["review"],
            "experiment_review": paths["experiment_review"],
        },
    ):
        trace = _trace(state)
        runner = _runner(state)
        artifact = json.loads(paths["artifact"].read_text(encoding="utf-8"))
        run_status = json.loads(paths["run_status"].read_text(encoding="utf-8"))
        old_metrics = json.loads(paths["metrics_json"].read_text(encoding="utf-8"))
        if state.get("real_mode"):
            run_status = _apply_evaluation_context_consistency_check(
                run_status,
                source_context=_read_json(paths["source_evaluation_context"]),
                old_prediction_path=paths["standardized_prediction"],
            )
            write_json(paths["run_status"], run_status)
        if run_status.get("train_success") and run_status.get("eval_success"):
            actual_path = run_status.get("actual_path") or artifact["actual_path"]
            runner.calculate_metrics(run_status["prediction_path"], actual_path, paths["new_metrics_csv"])
            new_metrics = metrics_csv_to_json(paths["new_metrics_csv"], paths["new_metrics"])
        else:
            new_metrics = write_json(paths["new_metrics"], old_metrics)
        comparison = compare_metrics(old_metrics, new_metrics, run_status)
        write_json(paths["comparison"], comparison)
        review = {
            "decision": comparison["decision"],
            "reason": comparison["reason"],
            "wape_delta": comparison["wape_delta"],
            "bias_delta": comparison["bias_delta"],
        }
        write_review(paths["review"], review)
        if comparison["decision"] == "rollback" and not state.get("real_mode"):
            rollback_change(state.get("backup_info", {}))
        write_experiment_review(paths["experiment_review"], comparison, run_status)
        trace.write("review_result", review)
        return {
            **state,
            "decision": comparison["decision"],
            "new_metrics_path": paths["new_metrics"].as_posix(),
            "metric_comparison_path": paths["comparison"].as_posix(),
            "review_result_path": paths["review"].as_posix(),
        }


def _apply_evaluation_context_consistency_check(
    run_status: dict[str, Any],
    *,
    source_context: dict[str, Any],
    old_prediction_path: Path,
) -> dict[str, Any]:
    if not run_status.get("train_success") or not run_status.get("eval_success"):
        return run_status
    new_prediction = Path(str(run_status.get("prediction_path") or ""))
    if not old_prediction_path.exists() or not new_prediction.exists():
        return run_status
    date_column = str(source_context.get("date_column") or "")
    old_context = _standardized_window(old_prediction_path, date_column)
    new_context = _standardized_window(new_prediction, date_column)
    expected_count = source_context.get("sample_count") or old_context.get("sample_count")
    expected_start = source_context.get("test_start") or old_context.get("test_start")
    expected_end = source_context.get("test_end") or old_context.get("test_end")
    errors: list[str] = []
    if expected_count is not None and new_context.get("sample_count") != expected_count:
        errors.append(f"sample_count drift: source={expected_count}, trial={new_context.get('sample_count')}")
    if date_column and expected_start and new_context.get("test_start") != expected_start:
        errors.append(f"test_start drift: source={expected_start}, trial={new_context.get('test_start')}")
    if date_column and expected_end and new_context.get("test_end") != expected_end:
        errors.append(f"test_end drift: source={expected_end}, trial={new_context.get('test_end')}")
    if not errors:
        return {
            **run_status,
            "evaluation_context_consistent": True,
            "evaluation_context_check": {
                "source": {"sample_count": expected_count, "test_start": expected_start, "test_end": expected_end},
                "trial": new_context,
            },
        }
    return {
        **run_status,
        "eval_success": False,
        "eval_returncode": "evaluation_context_drift",
        "failure_stage": "evaluation_context_drift",
        "error": "评测口径漂移，不评价本轮特征效果: " + "; ".join(errors),
        "evaluation_context_consistent": False,
        "evaluation_context_check": {
            "source": {"sample_count": expected_count, "test_start": expected_start, "test_end": expected_end},
            "baseline_standardized": old_context,
            "trial": new_context,
            "errors": errors,
        },
    }


def write_final_report_node(state: ComboScopeState) -> ComboScopeState:
    paths = _paths(state)
    with _recorder(state).step(
        "Agent1",
        "WriteFinalReport",
        artifacts={
            "forecast_report": paths["forecast_report"],
            "final_report": paths["final"],
            "artifact_index": paths["output"] / "artifact_index.md",
        },
    ):
        trace = _trace(state)
        if not paths["forecast_report"].exists():
            raise FileNotFoundError(
                f"Agent1 forecast_report.md is required before final report: {paths['forecast_report'].as_posix()}"
            )
        _enrich_report_context(paths)
        report_client = create_llm_client(provider=state.get("llm_provider"), model=state.get("model"))
        final_context = _build_final_report_context_from_paths(paths)
        write_final_report_context(final_context, paths["final_report_context"])
        try:
            _write_llm_final_report(
                final_context_path=paths["final_report_context"],
                final_report_path=paths["final"],
                forecast_report_path=paths["forecast_report"],
                client=report_client,
            )
        finally:
            _recorder(state).record_llm_call(report_client.last_call)
        trace.write("write_final_report", {"final_report": paths["final"].as_posix()})
        result = {**state, "final_report_path": paths["final"].as_posix()}
    _recorder(state).write_artifact_index()
    archive_trial_files(paths["output"])
    return result


def _build_agent1_report_context(state: ComboScopeState, paths: dict[str, Path]) -> None:
    runner = _runner(state)
    runner.build_report_context(
        ask=state["ask"],
        mode="full",
        scan_result=paths["scan"],
        artifact_summary=paths["artifact"],
        code_analysis=paths["code"],
        log_summary=paths["log"],
        metrics=paths["metrics_csv"],
        scene_metrics=paths["scene"],
        badcases=paths["badcases"],
        anomaly_summary=paths["anomaly"],
        output=paths["report_context"],
    )
    _enrich_report_context(paths)


def _enrich_report_context(paths: dict[str, Path]) -> None:
    context = json.loads(paths["report_context"].read_text(encoding="utf-8"))
    optional_json = {
        "experiment_plan": paths["plan"],
        "feature_hypothesis": paths["hypothesis"],
        "run_status": paths["run_status"],
        "metric_comparison": paths["comparison"],
        "new_metrics": paths["new_metrics"],
        "review_result": paths["review"],
    }
    for key, path in optional_json.items():
        if path.exists():
            if path.suffix in {".yaml", ".yml"}:
                context[key] = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
            else:
                context[key] = json.loads(path.read_text(encoding="utf-8"))
    feature_audit = paths["output"] / "code" / "agent2_feature_application_audit.yaml"
    if feature_audit.exists():
        context["feature_application_audit"] = yaml.safe_load(feature_audit.read_text(encoding="utf-8")) or {}
    if paths["badcase_diagnosis"].exists():
        context["badcase_diagnosis_md"] = paths["badcase_diagnosis"].read_text(encoding="utf-8", errors="ignore")
    write_json(paths["report_context"], context)


def _write_llm_forecast_report(
    *,
    context_path: Path,
    report_path: Path,
    suggestions_path: Path,
    repo_root: Path,
    client: LLMClient,
) -> None:
    context = json.loads(context_path.read_text(encoding="utf-8"))
    manager = SkillRunner(repo_root / "skills").manager
    report_skill = manager.load_skill("forecast-report-writer").content
    case_rules = _agent1_report_planning_rules(repo_root)
    skill_text = "\n\n---\n\n".join(
        [
            f"# Skill: forecast-report-writer\n\n{report_skill.strip()}",
            "# Agent1 optimization and case-reference rules",
            case_rules,
        ]
    )
    prompt_context = _compact_forecast_report_prompt_context(context)
    prompt = yaml.safe_dump(
        {
            "task": "Write forecast_report.md freely in Chinese from report_context evidence, following the skill rules. Return Markdown only.",
            "skill_rules": skill_text,
            "report_context": prompt_context,
        },
        allow_unicode=True,
        sort_keys=False,
    )
    result = client.complete_with_usage(
        "You are ComboScope Agent1 report writer. Use only provided evidence. Return Markdown, no code fence.",
        prompt,
        agent="Agent1",
        step="WriteForecastReport",
        timeout=AGENT1_REPORT_TIMEOUT,
        stream=True,
    )
    if not result.success or not result.content:
        raise RuntimeError("Agent1 WriteForecastReport LLM call failed or returned empty content")
    report = _strip_code_fence(result.content).strip()
    report = _normalize_report_markdown(report, step="WriteForecastReport")
    if not report:
        raise ValueError("Agent1 WriteForecastReport returned blank Markdown")
    report_path.write_text(report + "\n", encoding="utf-8")
    suggestions_path.write_text(report + "\n", encoding="utf-8")


def _agent1_report_planning_rules(repo_root: Path) -> str:
    manager = SkillRunner(repo_root / "skills").manager
    sections = []
    for skill_name in AGENT1_PLANNING_SKILLS:
        skill = manager.load_skill(skill_name)
        sections.append(f"# Skill: {skill.name}\n\n{_compact_text(skill.content.strip(), 1600)}")
    reference = manager.load_reference("forecast-optimization-case-reference", AGENT1_CASE_REFERENCE_PATH)
    sections.append(
        "# Reference: forecast-optimization-case-reference/package-lgbm-optimization-case\n\n"
        "Use this reference only as transferable patterns. Do not treat case-specific business fields, "
        "paths, model names, or windows as current experiment facts.\n\n"
        f"{_compact_text(reference.strip(), 1000)}"
    )
    return "\n\n---\n\n".join(sections)


def _compact_forecast_report_prompt_context(context: dict[str, Any]) -> dict[str, Any]:
    compact = json.loads(json.dumps(context, ensure_ascii=False))
    if isinstance(compact.get("badcases"), list):
        compact["badcases"] = [_compact_badcase_record_for_report(record) for record in compact["badcases"]]
    _limit_list_field(compact, "badcases", AGENT1_REPORT_BADCASE_LIMIT)
    _limit_list_field(compact, "scene_metrics", AGENT1_REPORT_SCENE_LIMIT)
    if "badcase_diagnosis_md" in compact:
        compact["badcase_diagnosis_md"] = _compact_text(str(compact["badcase_diagnosis_md"]), AGENT1_REPORT_TEXT_LIMIT)
    if isinstance(compact.get("experiment_plan"), dict):
        compact["experiment_plan"] = _compact_experiment_plan_for_report(compact["experiment_plan"])
    if isinstance(compact.get("feature_hypothesis"), dict):
        compact["feature_hypothesis"] = _compact_feature_hypothesis_for_report(compact["feature_hypothesis"])
    return compact


def _compact_experiment_plan_for_report(plan: dict[str, Any]) -> dict[str, Any]:
    keep_keys = [
        "trial_id",
        "target_problem",
        "scenario",
        "model_family",
        "objective",
        "source_entrypoint",
        "generated_train_path",
        "output_contract",
        "evaluation_metric",
        "editable_files",
        "expected_effect",
        "risk",
    ]
    compact = {key: plan[key] for key in keep_keys if key in plan}
    if isinstance(plan.get("hypothesis"), dict):
        compact["hypothesis"] = _compact_feature_hypothesis_for_report(plan["hypothesis"])
    if isinstance(plan.get("changes"), list):
        compact["changes"] = [_compact_feature_action_for_report(action) for action in plan["changes"]]
    if isinstance(plan.get("candidate_experiments"), list):
        candidates = plan["candidate_experiments"]
        compact["candidate_experiments"] = [
            _compact_candidate_experiment_for_report(candidate)
            for candidate in candidates[:AGENT1_REPORT_CANDIDATE_LIMIT]
        ]
        if len(candidates) > AGENT1_REPORT_CANDIDATE_LIMIT:
            compact["candidate_experiments_truncated"] = {
                "original_count": len(candidates),
                "kept_count": AGENT1_REPORT_CANDIDATE_LIMIT,
            }
    return compact


def _compact_feature_hypothesis_for_report(hypothesis: dict[str, Any]) -> dict[str, Any]:
    compact = {
        key: hypothesis[key]
        for key in ["description", "main_problem", "target_problem", "expected_effect", "risk"]
        if key in hypothesis
    }
    if isinstance(hypothesis.get("evidence"), list):
        compact["evidence"] = hypothesis["evidence"][:8]
        if len(hypothesis["evidence"]) > 8:
            compact["evidence_truncated"] = {"original_count": len(hypothesis["evidence"]), "kept_count": 8}
    return compact


def _compact_candidate_experiment_for_report(candidate: dict[str, Any]) -> dict[str, Any]:
    compact = {
        key: candidate[key]
        for key in ["experiment_id", "title", "priority", "expected_effect", "risk", "evidence"]
        if key in candidate
    }
    if isinstance(compact.get("evidence"), list):
        compact["evidence"] = compact["evidence"][:4]
    if isinstance(candidate.get("feature_actions"), list):
        compact["feature_actions"] = [
            _compact_feature_action_for_report(action)
            for action in candidate["feature_actions"][:3]
        ]
        if len(candidate["feature_actions"]) > 3:
            compact["feature_actions_truncated"] = {
                "original_count": len(candidate["feature_actions"]),
                "kept_count": 3,
            }
    return compact


def _compact_feature_action_for_report(action: dict[str, Any]) -> dict[str, Any]:
    compact = {
        key: action[key]
        for key in [
            "action",
            "feature_name",
            "feature_type",
            "group_by",
            "window",
            "cli_flag",
            "cli_args",
            "enabled",
            "field_sources",
            "code_locations",
            "validation_metrics",
            "expected_effect",
            "risk",
        ]
        if key in action
    }
    if "construction" in action:
        compact["construction"] = _compact_text(str(action["construction"]), 800)
    if isinstance(action.get("evidence"), list):
        compact["evidence"] = action["evidence"][:4]
    return compact


def _compact_badcase_record_for_report(record: Any) -> Any:
    if not isinstance(record, dict):
        return record
    compact = {}
    for key, value in record.items():
        normalized = str(key).lower()
        if (
            normalized in {"ds", "date", "actual", "prediction", "abs_error", "ape", "rank", "badcase_type", "scene"}
            or any(marker in normalized for marker in ["id", "code", "name", "actual", "pred", "error", "ape", "scene", "target", "rank"])
        ):
            compact[key] = _compact_value_for_report(value)
    if compact:
        return compact
    return {key: _compact_value_for_report(value) for key, value in list(record.items())[:12]}


def _compact_value_for_report(value: Any) -> Any:
    if isinstance(value, str):
        return _compact_text(value, 500)
    return value


def _limit_list_field(context: dict[str, Any], key: str, limit: int) -> None:
    value = context.get(key)
    if not isinstance(value, list) or len(value) <= limit:
        return
    context[key] = value[:limit]
    context[f"{key}_truncated"] = {"original_count": len(value), "kept_count": limit}


def _compact_text(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    omitted = len(text) - limit
    return f"{text[:limit].rstrip()}\n\n[truncated {omitted} chars]"


def _normalize_report_markdown(report: str, *, step: str) -> str:
    if not report:
        return report
    repaired = _repair_utf8_mojibake(report)
    if _looks_like_mojibake(repaired):
        raise ValueError(f"Agent1 {step} returned garbled Markdown; likely UTF-8 mojibake")
    return repaired


def _repair_utf8_mojibake(text: str) -> str:
    if not _looks_like_mojibake(text):
        return text
    try:
        candidate = text.encode("latin1").decode("utf-8")
    except UnicodeError:
        return text
    if _mojibake_score(candidate) < _mojibake_score(text) and _cjk_count(candidate) >= _cjk_count(text):
        return candidate
    return text


def _looks_like_mojibake(text: str) -> bool:
    c1_controls = sum(1 for char in text if 0x80 <= ord(char) <= 0x9F)
    return c1_controls >= 2 or _mojibake_score(text) >= 20


def _mojibake_score(text: str) -> int:
    markers = ("Ã", "Â", "â", "ã", "ä", "å", "æ", "ç", "è", "é", "ï", "�")
    marker_count = sum(text.count(marker) for marker in markers)
    c1_controls = sum(1 for char in text if 0x80 <= ord(char) <= 0x9F)
    return marker_count + c1_controls


def _cjk_count(text: str) -> int:
    return sum(1 for char in text if "\u4e00" <= char <= "\u9fff")


def _write_llm_final_report(
    *,
    final_context_path: Path,
    final_report_path: Path,
    forecast_report_path: Path,
    client: LLMClient,
) -> None:
    final_context = json.loads(final_context_path.read_text(encoding="utf-8"))
    forecast_excerpt = _read_text_excerpt(forecast_report_path, limit=6000)
    prompt = yaml.safe_dump(
        {
            "task": "Write final_report.md in Chinese as the final experiment verification report. Return Markdown only.",
            "hard_rules": [
                "Use final_report_context as the source of truth for decision, metrics, run status, and artifact paths.",
                "Do not change the deterministic keep/rollback decision.",
                "Do not paste the full forecast_report; reference it as Agent1 evidence.",
                "Focus on whether Agent2's modified experiment was valid, what changed, metrics comparison, failure/success reason, and next action.",
                "Return Markdown only, no code fence.",
            ],
            "required_sections": [
                "1. 结论",
                "2. Agent1 诊断与建议来源",
                "3. Agent2 执行与代码修改",
                "4. 指标对比与决策",
                "5. 失败或提升原因",
                "6. 下一步动作",
                "附录产物",
            ],
            "final_report_context": final_context,
            "agent1_forecast_report_excerpt": forecast_excerpt,
        },
        allow_unicode=True,
        sort_keys=False,
    )
    result = client.complete_with_usage(
        "You are ComboScope final report writer. Use only provided evidence and preserve deterministic decisions.",
        prompt,
        agent="Agent1",
        step="WriteFinalReport",
        timeout=AGENT1_FINAL_REPORT_TIMEOUT,
    )
    if not result.success or not result.content:
        raise RuntimeError("Agent1 WriteFinalReport LLM call failed or returned empty content")
    report = _strip_code_fence(result.content).strip()
    if not report:
        raise ValueError("Agent1 WriteFinalReport returned blank Markdown")
    if re.search(r"\$\{[^}\n]+\}", report):
        write_final_report(final_context, final_report_path)
        return
    final_report_path.write_text(report + "\n", encoding="utf-8")


def _build_final_report_context_from_paths(paths: dict[str, Path]) -> dict[str, Any]:
    return build_final_report_context(
        review_result=_read_json(paths["review"]),
        metric_comparison=_read_json(paths["comparison"]),
        old_metrics=_read_json(paths["metrics_json"]),
        new_metrics=_read_json(paths["new_metrics"]),
        problem_context=_read_json(paths["problem"]),
        experiment_plan=_read_yaml(paths["plan"]),
        run_status=_read_json(paths["run_status"]),
        artifacts={
            "完整评测报告": paths["forecast_report"].as_posix(),
            "badcase 明细": paths["badcases"].as_posix(),
            "细粒度指标": paths["scene"].as_posix(),
            "训练日志": (paths["output"] / "logs" / "train.log").as_posix()
            if (paths["output"] / "logs" / "train.log").exists()
            else (paths["output"] / "train.log").as_posix(),
            "特征应用审计": (paths["output"] / "code" / "agent2_feature_application_audit.yaml").as_posix(),
            "执行 trace": paths["trace"].as_posix(),
            "产物索引": (paths["output"] / "artifact_index.md").as_posix(),
        },
    )


def _read_text_excerpt(path: Path, *, limit: int) -> str:
    if not path.exists():
        return ""
    text = path.read_text(encoding="utf-8", errors="ignore")
    return text[:limit]


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    value = json.loads(path.read_text(encoding="utf-8"))
    return value if isinstance(value, dict) else {}


def _read_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    value = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    return value if isinstance(value, dict) else {}


def _write_agent1_skill_route(path: Path, problem_context: dict[str, Any]) -> None:
    route = {
        "agent": "Agent1",
        "mode": "llm_led_with_skill_evidence",
        "selected_skills": [
            "forecast-experiment-scanner",
            "forecast-code-log-analyzer",
            "forecast-evaluation-analyzer",
            "forecast-badcase-locator",
            "forecast-optimization-advisor",
            "forecast-optimization-case-reference",
            "forecast-report-writer",
        ],
        "main_problem": problem_context.get("main_problem"),
        "notes": [
            "LLM selects artifacts, diagnoses badcases, drafts hypotheses, and writes reports from skill evidence.",
            "Python validates schemas and computes metrics, but does not choose business-specific experiment ideas.",
            "Skill outputs are passed as artifact paths, not large DataFrames in state.",
            "Case-reference skill is used only as transferable patterns grounded by current metrics, badcases, logs, and code evidence.",
        ],
    }
    write_yaml(path, route)


def _record_llm_calls_since(recorder: AgentRunRecorder, client: LLMClient, start_index: int) -> None:
    results = getattr(client, "call_results", None)
    if isinstance(results, list):
        for result in results[start_index:]:
            recorder.record_llm_call(result)
        return
    recorder.record_llm_call(client.last_call)


def _build_agent2_execution_plan(plan: dict[str, Any], state: ComboScopeState, client: LLMClient) -> dict[str, Any]:
    experiment = Path(state["experiment_dir"])
    paths = _paths(state)
    evidence = _agent2_execution_evidence(state, experiment)
    required_schema = {
        "agent": "Agent2",
        "trial_id": plan.get("trial_id", state.get("trial_id")),
        "source_entrypoint": "relative Python source entrypoint to copy into trial code/train.py",
        "python_dependencies": ["relative Python dependency files to copy into trial code/"],
        "required_data_files": [
            {
                "path": "relative or absolute input data file",
                "cli_arg": "optional CLI flag that should receive the trial copy path",
            }
        ],
        "train_command": ["{python}", "{train_py}", "--output_dir", "{real_output_dir}"],
        "output_contract": {
            "prediction_path": "relative to trial dir or real_output_dir after training",
            "actual_path": "relative to trial dir or real_output_dir after training",
            "prediction_column": "column name",
            "actual_column": "column name",
            "date_column": "optional date column",
            "id_columns": ["optional key columns"],
            "passthrough_columns": ["optional scene columns"],
            "split_column": "optional split column",
            "split_value": "optional split value",
        },
        "source_evaluation_context": {
            "test_start": "source baseline test start date",
            "test_end": "source baseline test end date",
            "sample_count": "source baseline standardized prediction row count",
            "date_column": "date column to preserve",
            "split_column": "split column to preserve",
            "split_value": "split value to preserve",
            "preserved_train_args": {"--train_eval_end": "source runtime value, if supported"},
        },
        "guardrails": [
            "copy files only into runs/<trial>/code or runs/<trial>/data",
            "do not modify original experiment source files",
            "do not change metric definitions, labels, or data split",
        ],
    }
    prompt = yaml.safe_dump(
        {
            "task": "Build the concrete execution plan for this generic forecasting trial. Return only YAML.",
            "ask": state.get("ask", ""),
            "experiment_dir": experiment.as_posix(),
            "original_command": state.get("original_command", ""),
            "required_schema": required_schema,
            "hard_rules": [
                "Use source_entrypoint and python_dependencies only from existing Python files.",
                "Use required_data_files only from existing data/config files listed in evidence, or [] if none are needed.",
                "train_command is only the process entrypoint for copied train.py; dependencies such as util.py must be imported/called by train.py, not executed separately.",
                "train_command may use only these placeholders: "
                + ", ".join(sorted(AGENT2_COMMAND_PLACEHOLDERS)),
                "train_command must be an argv token list; list-valued argparse parameters must be separate tokens such as ['--rolling_windows', '7', '14', '30'], not comma strings or quoted shell fragments.",
                "Do not invent placeholders such as {data_path}; use copied data paths through code edits or concrete trial-relative paths.",
                "Do not pass {trial_id} to argparse choice flags such as --experiment; choice values must be one of the source code choices.",
                "Prefer making the copied train.py self-contained with trial defaults instead of piling on CLI arguments.",
                "If the source already exposes an output CLI flag in argparse_args, prefer that exact flag.",
                "train_command may include feature CLI args only for experiment_plan.changes, the selected experiment Agent2 will execute; ignore lower-priority candidate_experiments.",
                "train_command must inherit all source_evaluation_context.preserved_train_args supported by the source entrypoint.",
                "Do not change evaluation windows, split_data, train_eval_end, valid_days/test_days, label, metrics, output filtering, standardization, or output_contract semantics.",
                "If source_evaluation_context.context_status is unresolved, mark the plan unresolved and do not invent evaluation parameters.",
                "For every required_data_files item with cli_arg, the runner will copy it to trial/data and pass that copied path to the CLI flag.",
                "Do not use cwd-sensitive paths like data/<file> in train_command; use {trial_dir}/data/<file> or rely on the runner's declared resource override.",
                "output_contract paths must be relative to real_output_dir or trial dir and must not contain placeholders.",
                "Return YAML only, without markdown.",
            ],
            "experiment_plan": plan,
            "evidence": evidence,
            "agent1_forecast_report_path": paths["forecast_report"].as_posix(),
        },
        allow_unicode=True,
        sort_keys=False,
    )
    parsed = _complete_llm_yaml_mapping_with_repair(
        client,
        "You are ComboScope Agent2. Return a complete YAML execution plan from evidence only.",
        prompt,
        agent="Agent2",
        step="BuildExecutionPlan",
        required_schema=required_schema,
        repair_task="Repair the Agent2 execution plan into a valid YAML mapping. Preserve all facts, paths, commands, metrics, evaluation windows, and field names; change only YAML syntax, quoting, indentation, and block-scalar formatting.",
        timeout=AGENT2_PLANNING_TIMEOUT,
        stream=True,
    )
    return _validate_or_repair_agent2_execution_plan(
        parsed,
        client=client,
        experiment=experiment,
        plan=plan,
        state=state,
        original_prompt=prompt,
        evidence=evidence,
    )


def _agent2_execution_evidence(state: ComboScopeState, experiment: Path) -> dict[str, Any]:
    paths = _paths(state)
    scan = _read_json(paths["scan"])
    code = _read_json(paths["code"])
    artifact_contract = _read_json(paths["artifact_contract"])
    data_headers = _data_headers_for_llm(experiment, scan)
    existing_python_files = sorted(
        dict.fromkeys(
            str(item)
            for item in [
                *(scan.get("code_files", []) or []),
                *(scan.get("possible_entrypoints", []) or []),
                *(code.get("entrypoints", []) or []),
                *(code.get("evaluation_entrypoints", []) or []),
            ]
            if str(item).endswith(".py")
        )
    )
    existing_data_files = sorted(
        dict.fromkeys(
            str(item)
            for key in ("data_files", "config_files", "candidate_prediction_files", "actual_files", "domain_artifact_files")
            for item in (scan.get(key, []) or [])
        )
    )
    return {
        "existing_python_files": existing_python_files,
        "entrypoints": code.get("entrypoints", []) or scan.get("possible_entrypoints", []),
        "argparse_args": code.get("argparse_args", []),
        "existing_data_files": existing_data_files,
        "data_headers": data_headers,
        "artifact_contract": artifact_contract,
        "source_evaluation_context": _read_json(paths["source_evaluation_context"]),
        "code_functions": code.get("function_defs", []),
        "original_command": state.get("original_command", ""),
        "agent1_forecast_report_path": paths["forecast_report"].as_posix() if paths["forecast_report"].exists() else "",
        "agent1_forecast_report_excerpt": _read_text_excerpt(paths["forecast_report"], limit=6000),
    }


def _validate_or_repair_agent2_execution_plan(
    execution_plan: dict[str, Any],
    *,
    client: LLMClient,
    experiment: Path,
    plan: dict[str, Any],
    state: ComboScopeState,
    original_prompt: str,
    evidence: dict[str, Any],
) -> dict[str, Any]:
    try:
        return _validated_agent2_execution_plan(execution_plan, experiment, plan, state)
    except Exception as first_error:  # noqa: BLE001 - repair only with LLM, then validate again.
        repair_prompt = yaml.safe_dump(
            {
                "task": "Repair the Agent2 execution plan so it validates. Return only YAML with the same schema.",
                "validation_error": str(first_error),
                "hard_rules": [
                    "Do not invent paths. source_entrypoint, python_dependencies, and required_data_files must exist.",
                    "Use only these train_command placeholders: " + ", ".join(sorted(AGENT2_COMMAND_PLACEHOLDERS)),
                    "train_command must be an argv token list; list-valued argparse parameters must be separate tokens such as ['--rolling_windows', '7', '14', '30'], not comma strings or quoted shell fragments.",
                    "Remove unsupported placeholders such as {data_path}.",
                    "Do not pass {trial_id} to argparse choice flags such as --experiment; use an allowed source choice or make train.py defaults self-contained.",
                    "Do not add a data CLI argument unless the entrypoint supports it or your code edits will explicitly add it.",
                    "Preserve source_evaluation_context and inherit supported preserved_train_args in train_command.",
                    "Do not modify evaluation windows, split, label, metric, output filtering, standardization, or output contract.",
                    "If copied actual data is needed in output_contract, use data/<copied filename>.",
                    "output_contract paths must not contain placeholders.",
                    "Return YAML only, without markdown.",
                ],
                "invalid_execution_plan": execution_plan,
                "experiment_plan": plan,
                "evidence": evidence,
                "original_prompt": original_prompt,
            },
            allow_unicode=True,
            sort_keys=False,
        )
        repaired = _complete_llm_yaml_mapping(
            client,
            "You are ComboScope Agent2. Repair only invalid paths, placeholders, and schema fields. Return YAML.",
            repair_prompt,
            agent="Agent2",
            step="BuildExecutionPlan",
            timeout=AGENT2_PLANNING_TIMEOUT,
            stream=True,
        )
        return _validated_agent2_execution_plan(repaired, experiment, plan, state)


def _validated_agent2_execution_plan(
    execution_plan: dict[str, Any],
    experiment: Path,
    plan: dict[str, Any],
    state: ComboScopeState,
) -> dict[str, Any]:
    for key in ("source_entrypoint", "train_command", "output_contract"):
        if key not in execution_plan:
            raise ValueError(f"Agent2 execution plan missing required field: {key}")
    source_entrypoint = str(execution_plan.get("source_entrypoint") or "").strip()
    if not source_entrypoint:
        raise ValueError("Agent2 execution plan source_entrypoint must be non-empty")
    source_path = _resolve_contract_path(experiment, source_entrypoint)
    if source_path.suffix != ".py" or not source_path.is_file():
        raise ValueError(f"source_entrypoint must be an existing Python file: {source_entrypoint}")
    dependencies = execution_plan.get("python_dependencies") or execution_plan.get("dependency_files") or []
    if not isinstance(dependencies, list):
        raise ValueError("Agent2 execution plan python_dependencies must be a list")
    for raw_dependency in dependencies:
        dependency = str(raw_dependency)
        dependency_path = _resolve_contract_path(experiment, dependency)
        if dependency_path.suffix != ".py" or not dependency_path.is_file():
            raise ValueError(f"python dependency must be an existing Python file: {dependency}")
    required_data_files = execution_plan.get("required_data_files") or []
    if not isinstance(required_data_files, list):
        raise ValueError("Agent2 execution plan required_data_files must be a list")
    for item in required_data_files:
        if isinstance(item, str):
            data_path = item
        elif isinstance(item, dict):
            data_path = str(item.get("path") or "")
            cli_arg = item.get("cli_arg")
            if cli_arg is not None and (not isinstance(cli_arg, str) or not cli_arg.startswith("--")):
                raise ValueError(f"required_data_files cli_arg must be a CLI flag: {cli_arg}")
        else:
            raise ValueError("required_data_files entries must be strings or mappings")
        if not data_path:
            raise ValueError("required_data_files entries must include a non-empty path")
        resolved_data = _resolve_contract_path(experiment, data_path)
        if not resolved_data.is_file():
            raise ValueError(f"required data file must be an existing file: {data_path}")
    train_command = execution_plan.get("train_command")
    if not isinstance(train_command, list) or not train_command:
        raise ValueError("Agent2 execution plan train_command must be a non-empty list")
    source_context = _source_evaluation_context_for_plan(execution_plan, state)
    train_command = _merge_preserved_train_args_for_plan(train_command, source_path.read_text(encoding="utf-8", errors="ignore"), source_context)
    execution_plan["train_command"] = train_command
    _validate_agent2_train_command(train_command)
    validate_train_command_argparse_choices(train_command, source_path.read_text(encoding="utf-8", errors="ignore"))
    output_contract = execution_plan.get("output_contract")
    if not isinstance(output_contract, dict):
        raise ValueError("Agent2 execution plan output_contract must be a mapping")
    for key in ("prediction_path", "actual_path", "prediction_column", "actual_column"):
        if not isinstance(output_contract.get(key), str) or not str(output_contract[key]).strip():
            raise ValueError(f"Agent2 output_contract missing required field: {key}")
    for key in ("prediction_path", "actual_path"):
        value = str(output_contract.get(key) or "")
        unknown = _unsupported_agent2_placeholders(value)
        if unknown:
            raise ValueError(f"output_contract {key} contains unsupported placeholders: {sorted(unknown)}")
    execution_plan.setdefault("agent", "Agent2")
    execution_plan.setdefault("trial_id", plan.get("trial_id", state.get("trial_id")))
    execution_plan.setdefault("python_dependencies", dependencies)
    execution_plan.setdefault("required_data_files", required_data_files)
    execution_plan.setdefault("feature_changes", plan.get("changes", []))
    execution_plan.setdefault("agent1_forecast_report_path", _paths(state)["forecast_report"].as_posix())
    execution_plan["source_evaluation_context"] = source_context
    return execution_plan


def _source_evaluation_context_for_plan(execution_plan: dict[str, Any], state: ComboScopeState) -> dict[str, Any]:
    context = execution_plan.get("source_evaluation_context")
    if not isinstance(context, dict) or not context:
        context = state.get("source_evaluation_context") or _read_json(_paths(state)["source_evaluation_context"])
    context = context if isinstance(context, dict) else {}
    if not context:
        return {"context_status": "unresolved", "unresolved_reason": "source_evaluation_context was not available"}
    return context


def _merge_preserved_train_args_for_plan(
    train_command: list[Any],
    source_text: str,
    source_context: dict[str, Any],
) -> list[str]:
    preserved = source_context.get("preserved_train_args") if isinstance(source_context, dict) else {}
    if not isinstance(preserved, dict) or not preserved:
        return [str(item) for item in train_command]
    command = [str(item) for item in train_command]
    for raw_flag, raw_value in preserved.items():
        flag = str(raw_flag)
        value = str(raw_value)
        supported = _source_supported_arg_spelling(source_text, flag)
        if supported is None:
            continue
        command = _replace_or_append_flag_value(command, supported, value, source_text)
    return command


def _replace_or_append_flag_value(command: list[str], flag: str, value: str, source_text: str) -> list[str]:
    variants = {flag, flag.replace("_", "-"), flag.replace("-", "_")}
    merged: list[str] = []
    index = 0
    replaced = False
    while index < len(command):
        item = command[index]
        supported_item = _source_supported_arg_spelling(source_text, item) if item.startswith("--") else None
        if item in variants or supported_item == flag:
            merged.extend([item, value])
            replaced = True
            index += 2 if index + 1 < len(command) and not str(command[index + 1]).startswith("--") else 1
            continue
        merged.append(item)
        index += 1
    if not replaced:
        merged.extend([flag, value])
    return merged


def _source_supported_arg_spelling(source_text: str, flag: str) -> str | None:
    for variant in [flag, flag.replace("_", "-"), flag.replace("-", "_")]:
        quoted = {f'"{variant}"', f"'{variant}'"}
        if any(item in source_text for item in quoted):
            return variant
    return None


def _validate_agent2_train_command(train_command: list[Any]) -> None:
    unsupported: set[str] = set()
    for item in train_command:
        unsupported.update(_unsupported_agent2_placeholders(str(item)))
    if unsupported:
        raise ValueError(f"train_command contains unsupported placeholders: {sorted(unsupported)}")
    command_text = " ".join(str(item) for item in train_command)
    if not any(token in command_text.split() for token in ("{train_py}", "train.py", "./train.py")):
        raise ValueError("train_command must execute the trial train.py via {train_py}")
    for item in train_command[2:]:
        if str(item).endswith(".py"):
            raise ValueError("train_command must not pass dependency Python files as extra positional scripts")


def _unsupported_agent2_placeholders(value: str) -> set[str]:
    tokens = set(re.findall(r"\{[A-Za-z_][A-Za-z0-9_]*\}", value))
    return tokens - AGENT2_COMMAND_PLACEHOLDERS


def build_forecast_evaluation_graph():
    graph = StateGraph(ComboScopeState)
    graph.add_node("EvaluateAndDiagnose", evaluate_and_diagnose)
    graph.add_edge(START, "EvaluateAndDiagnose")
    graph.add_edge("EvaluateAndDiagnose", END)
    return graph.compile()


def run_forecast_evaluation(state: ComboScopeState) -> ComboScopeState:
    return build_forecast_evaluation_graph().invoke(state)


def build_graph():
    graph = StateGraph(ComboScopeState)
    graph.add_node("ForecastEvaluationSubgraph", build_forecast_evaluation_graph())
    graph.add_node("GenerateExperimentPlan", generate_experiment_plan_node)
    graph.add_node("RunExperiment", run_experiment_node)
    graph.add_node("ReviewResult", review_result_node)
    graph.add_node("WriteFinalReport", write_final_report_node)
    graph.add_edge(START, "ForecastEvaluationSubgraph")
    graph.add_edge("ForecastEvaluationSubgraph", "GenerateExperimentPlan")
    graph.add_edge("GenerateExperimentPlan", "RunExperiment")
    graph.add_edge("RunExperiment", "ReviewResult")
    graph.add_edge("ReviewResult", "WriteFinalReport")
    graph.add_edge("WriteFinalReport", END)
    return graph.compile()


def run_once(request: dict[str, Any]) -> dict[str, Any]:
    output_dir = Path(request["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    initial: ComboScopeState = {
        "experiment_dir": Path(request["experiment_dir"]).resolve().as_posix(),
        "output_dir": output_dir.resolve().as_posix(),
        "ask": request["ask"],
        "trial_id": request.get("trial_id", output_dir.name),
        "repo_root": Path(request.get("repo_root") or Path.cwd()).resolve().as_posix(),
        "model": request.get("model"),
        "llm_provider": request.get("llm_provider"),
        "original_command": request.get("original_command", ""),
        "errors": [],
    }
    result = build_graph().invoke(initial)
    _refresh_completed_final_report(result)
    return result


def _refresh_completed_final_report(state: ComboScopeState) -> None:
    paths = _paths(state)
    if not paths["final"].exists() or not paths["experiment_review"].exists() or not paths["review"].exists():
        return
    final_context = _build_final_report_context_from_paths(paths)
    write_final_report_context(final_context, paths["final_report_context"])
    write_final_report(final_context, paths["final"])
    archive_trial_files(paths["output"])
