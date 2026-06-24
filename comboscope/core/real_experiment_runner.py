from __future__ import annotations

import ast
import builtins
import csv
import json
import re
import shlex
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import yaml

from comboscope.runtime.artifact_contract import standardize_from_contract, write_artifact_contract
from comboscope.runtime.yaml_utils import append_yaml_output_contract, clean_llm_yaml_text, safe_load_yaml_mapping, strip_code_fence


AGENT2_CODEGEN_TIMEOUT = (30, 600)
AGENT2_LOGGED_EVAL_TIME_TIMEOUT = (10, 600)
AGENT2_CODEGEN_MAX_ATTEMPTS = 2
AGENT2_REPAIR_MAX_ATTEMPTS = 2
AGENT2_BACKTEST_REPAIR_MAX_ATTEMPTS = 2
AGENT2_CONTEXT_CHAR_LIMIT = 200_000
AGENT2_COMPACT_CONTEXT_CHAR_LIMIT = 200_000
AGENT2_PROMPT_TOKEN_LIMIT = 64_000
AGENT2_SOURCE_SLICE_CONTEXT_LINES = 50
AGENT2_MAX_SOURCE_SLICES = 6
AGENT2_ACCEPTED_EDIT_TYPES = {
    "append_module_code",
    "replace_function",
    "insert_after_line",
    "insert_before_return",
    "replace_lines",
    "add_feature_column_in_function",
}
AGENT2_FIRST_PASS_EDIT_TYPES = [
    "replace_function",
    "add_feature_column_in_function",
    "insert_before_return",
    "append_module_code",
]
AGENT2_NARROW_EDIT_TYPES = [
    "replace_function",
    "add_feature_column_in_function",
    "insert_before_return",
    "insert_after_line",
    "replace_lines",
]


def generate_trial_train_wrapper(
    plan: dict[str, Any],
    experiment_dir: str | Path,
    trial_dir: str | Path,
    llm_client: Any | None = None,
    execution_plan: dict[str, Any] | None = None,
) -> Path:
    experiment = Path(experiment_dir).resolve()
    trial = Path(trial_dir).resolve()
    dirs = _trial_dirs(trial)
    source_entrypoint = execution_plan.get("source_entrypoint") if execution_plan else plan.get("source_entrypoint")
    if not source_entrypoint:
        raise ValueError("source_entrypoint is required from Agent1/Agent2 LLM output")
    source_path = experiment / source_entrypoint
    if not source_path.exists():
        raise FileNotFoundError(source_path)
    _copy_declared_python_dependencies(experiment, dirs["code"], source_path, execution_plan or {})

    wrapper = dirs["code"] / "train.py"
    source_text = source_path.read_text(encoding="utf-8")
    _normalize_experiment_plan_for_trial(plan, source_text, trial.name, source_entrypoint)
    arg_overrides = _feature_arg_overrides(plan)
    resource_context = _prepare_trial_data_resources(experiment, dirs["data"], source_text, plan, source_path, execution_plan or {})
    fallback_source = _patch_trial_train_source(source_text, arg_overrides, plan, experiment, resource_context)
    wrapper.write_text(fallback_source, encoding="utf-8")
    modification = _modify_trial_code_with_agent2(
        code_dir=dirs["code"],
        wrapper=wrapper,
        fallback_source=fallback_source,
        plan=plan,
        execution_plan=execution_plan or {},
        resource_context=resource_context,
        llm_client=llm_client,
    )
    _write_code_modification_record(dirs["code"], llm_client, resource_context, modification)
    return wrapper


def run_real_experiment(
    experiment_dir: str | Path,
    trial_dir: str | Path,
    plan: dict[str, Any],
    llm_client: Any | None = None,
    execution_plan: dict[str, Any] | None = None,
) -> dict[str, Any]:
    trial = Path(trial_dir).resolve()
    dirs = _trial_dirs(trial)
    execution_plan = execution_plan or {}
    source_evaluation_context = _source_evaluation_context(execution_plan)
    if source_evaluation_context.get("context_status") == "unresolved":
        status = _unresolved_evaluation_context_status(trial, dirs, source_evaluation_context, execution_plan)
        (trial / "run_status.json").write_text(json.dumps(status, ensure_ascii=False, indent=2), encoding="utf-8")
        return status
    wrapper = generate_trial_train_wrapper(plan, experiment_dir, trial, llm_client=llm_client, execution_plan=execution_plan)
    real_outputs = dirs["real_outputs"]
    output_contract = _output_contract_from_execution_plan(execution_plan)
    output_contract_path = trial / "output_contract.json"
    write_artifact_contract(output_contract_path, output_contract)
    manifest = _read_input_manifest(dirs["data"] / "input_manifest.json")
    train_command_normalizations: list[dict[str, Any]] = []
    train_command = _trial_command(
        plan,
        experiment_dir,
        trial,
        wrapper,
        execution_plan,
        normalizations_out=train_command_normalizations,
    )
    feature_audit = _audit_feature_application(plan, dirs["code"], wrapper, train_command=train_command)
    _update_code_modification_runtime_fields(dirs["code"], train_command, feature_audit, train_command_normalizations)
    modification_status = _code_modification_status_fields(dirs["code"])
    command_validation_error = _train_command_validation_error(train_command, wrapper, manifest)
    if command_validation_error:
        repair = _repair_trial_code_after_train_command_failure(
            code_dir=dirs["code"],
            wrapper=wrapper,
            plan=plan,
            execution_plan=execution_plan,
            resource_context=manifest,
            llm_client=llm_client,
            train_command=train_command,
            train_command_normalizations=train_command_normalizations,
            validation_error=command_validation_error,
            repair_attempt_index=1,
        )
        _merge_train_command_repair_record(dirs["code"], repair)
        repaired_raw_command = repair.get("repaired_train_command")
        if repair.get("code_generation_success") is True:
            if isinstance(repaired_raw_command, list) and repaired_raw_command:
                execution_plan["train_command"] = [str(item) for item in repaired_raw_command]
                _apply_repaired_plan_cli_args(plan, [str(item) for item in repaired_raw_command], wrapper)
            train_command_normalizations = []
            train_command = _trial_command(
                plan,
                experiment_dir,
                trial,
                wrapper,
                execution_plan,
                normalizations_out=train_command_normalizations,
            )
            train_command, logged_eval_time_normalizations = _apply_llm_logged_eval_time_overrides(
                train_command,
                wrapper,
                execution_plan,
                source_evaluation_context,
                llm_client,
            )
            train_command_normalizations.extend(logged_eval_time_normalizations)
            feature_audit = _audit_feature_application(plan, dirs["code"], wrapper, train_command=train_command)
            _update_code_modification_runtime_fields(dirs["code"], train_command, feature_audit, train_command_normalizations)
            modification_status = _code_modification_status_fields(dirs["code"])
            command_validation_error = _train_command_validation_error(train_command, wrapper, manifest)
    if command_validation_error:
        status = {
            "train_success": False,
            "eval_success": False,
            "train_returncode": "invalid_train_command",
            "eval_returncode": None,
            "generated_train_path": wrapper.as_posix(),
            "real_output_dir": real_outputs.as_posix(),
            "data_manifest_path": (dirs["data"] / "input_manifest.json").as_posix(),
            "train_log_path": (dirs["logs"] / "train.log").as_posix(),
            "eval_log_path": (dirs["logs"] / "eval.log").as_posix(),
            "train_command": train_command,
            "train_command_normalizations": train_command_normalizations,
            "output_contract_path": output_contract_path.as_posix(),
            **_source_evaluation_status_fields(source_evaluation_context),
            "input_files": manifest.get("copied_files", {}),
            "missing_input_files": manifest.get("missing_input_files", []),
            "failure_stage": "train_command_validation",
            "error": command_validation_error,
            "feature_application_success": feature_audit.get("success", False),
            "feature_application_audit_success": feature_audit.get("success", False),
            "feature_application_audit_path": feature_audit.get("audit_path"),
            "agent2_backtest_success": False,
            "agent2_backtest_attempts": [],
            "agent2_backtest_repair_used": False,
            **modification_status,
        }
        (dirs["logs"] / "train.log").write_text(
            _train_log_header(train_command, manifest, real_outputs)
            + "train skipped because generated train_command is invalid\n"
            + command_validation_error
            + "\n",
            encoding="utf-8",
        )
        (dirs["logs"] / "eval.log").write_text(
            "train skipped because generated train_command is invalid\n" + command_validation_error + "\n",
            encoding="utf-8",
        )
        shutil.copy2(dirs["logs"] / "train.log", trial / "train.log")
        shutil.copy2(dirs["logs"] / "eval.log", trial / "eval.log")
        (trial / "run_status.json").write_text(json.dumps(status, ensure_ascii=False, indent=2), encoding="utf-8")
        return status
    if modification_status.get("agent2_code_generation_success") is not True:
        feature_audit["success"] = False
        return _agent2_codegen_failure_status(
            trial=trial,
            dirs=dirs,
            wrapper=wrapper,
            real_outputs=real_outputs,
            manifest=manifest,
            train_command=train_command,
            train_command_normalizations=train_command_normalizations,
            output_contract_path=output_contract_path,
            source_evaluation_context=source_evaluation_context,
            modification_status=modification_status,
            feature_audit=feature_audit,
        )
    train_command, logged_eval_time_normalizations = _apply_llm_logged_eval_time_overrides(
        train_command,
        wrapper,
        execution_plan,
        source_evaluation_context,
        llm_client,
    )
    train_command_normalizations.extend(logged_eval_time_normalizations)
    if logged_eval_time_normalizations:
        feature_audit = _audit_feature_application(plan, dirs["code"], wrapper, train_command=train_command)
        _update_code_modification_runtime_fields(dirs["code"], train_command, feature_audit, train_command_normalizations)
        modification_status = _code_modification_status_fields(dirs["code"])
        command_validation_error = _train_command_validation_error(train_command, wrapper, manifest)
        if command_validation_error:
            status = {
                "train_success": False,
                "eval_success": False,
                "train_returncode": "invalid_train_command",
                "eval_returncode": None,
                "generated_train_path": wrapper.as_posix(),
                "real_output_dir": real_outputs.as_posix(),
                "data_manifest_path": (dirs["data"] / "input_manifest.json").as_posix(),
                "train_log_path": (dirs["logs"] / "train.log").as_posix(),
                "eval_log_path": (dirs["logs"] / "eval.log").as_posix(),
                "train_command": train_command,
                "train_command_normalizations": train_command_normalizations,
                "output_contract_path": output_contract_path.as_posix(),
                **_source_evaluation_status_fields(source_evaluation_context),
                "input_files": manifest.get("copied_files", {}),
                "missing_input_files": manifest.get("missing_input_files", []),
                "failure_stage": "train_command_validation",
                "error": command_validation_error,
                "feature_application_success": feature_audit.get("success", False),
                "feature_application_audit_success": feature_audit.get("success", False),
                "feature_application_audit_path": feature_audit.get("audit_path"),
                "agent2_backtest_success": False,
                "agent2_backtest_attempts": [],
                "agent2_backtest_repair_used": False,
                **modification_status,
            }
            (dirs["logs"] / "train.log").write_text(
                _train_log_header(train_command, manifest, real_outputs)
                + "train skipped because generated train_command is invalid\n"
                + command_validation_error
                + "\n",
                encoding="utf-8",
            )
            (dirs["logs"] / "eval.log").write_text(
                "train skipped because generated train_command is invalid\n" + command_validation_error + "\n",
                encoding="utf-8",
            )
            shutil.copy2(dirs["logs"] / "train.log", trial / "train.log")
            shutil.copy2(dirs["logs"] / "eval.log", trial / "eval.log")
            (trial / "run_status.json").write_text(json.dumps(status, ensure_ascii=False, indent=2), encoding="utf-8")
            return status
    missing_input_files = manifest.get("missing_input_files", [])
    if missing_input_files:
        status = {
            "train_success": False,
            "eval_success": False,
            "train_returncode": "missing_input_files",
            "eval_returncode": None,
            "generated_train_path": wrapper.as_posix(),
            "real_output_dir": real_outputs.as_posix(),
            "data_manifest_path": (dirs["data"] / "input_manifest.json").as_posix(),
            "train_log_path": (dirs["logs"] / "train.log").as_posix(),
            "train_command": train_command,
            "train_command_normalizations": train_command_normalizations,
            "output_contract_path": output_contract_path.as_posix(),
            **_source_evaluation_status_fields(source_evaluation_context),
            "input_files": manifest.get("copied_files", {}),
            "missing_input_files": missing_input_files,
            "failure_stage": "prepare_inputs",
            "error": "missing input files for trial training",
            "feature_application_success": feature_audit.get("success", False),
            "feature_application_audit_success": feature_audit.get("success", False),
            "feature_application_audit_path": feature_audit.get("audit_path"),
            "agent2_backtest_success": False,
            "agent2_backtest_attempts": [],
            "agent2_backtest_repair_used": False,
            **modification_status,
        }
        (dirs["logs"] / "train.log").write_text(
            _train_log_header(train_command, manifest, real_outputs)
            + "missing input files:\n"
            + "\n".join(str(item) for item in missing_input_files)
            + "\n",
            encoding="utf-8",
        )
        (dirs["logs"] / "eval.log").write_text("train skipped because required input files are missing\n", encoding="utf-8")
        (trial / "run_status.json").write_text(json.dumps(status, ensure_ascii=False, indent=2), encoding="utf-8")
        return status
    backtest_attempts: list[dict[str, Any]] = []
    backtest_repair_used = False
    for attempt_index in range(1, AGENT2_BACKTEST_REPAIR_MAX_ATTEMPTS + 2):
        if attempt_index > 1:
            _clean_backtest_outputs(dirs)
            train_command_normalizations = []
            train_command = _trial_command(
                plan,
                experiment_dir,
                trial,
                wrapper,
                execution_plan,
                normalizations_out=train_command_normalizations,
            )
            feature_audit = _audit_feature_application(plan, dirs["code"], wrapper, train_command=train_command)
            _update_code_modification_runtime_fields(dirs["code"], train_command, feature_audit, train_command_normalizations)
            modification_status = _code_modification_status_fields(dirs["code"])
        status = _run_agent2_backtest_once(
            trial=trial,
            dirs=dirs,
            wrapper=wrapper,
            real_outputs=real_outputs,
            manifest=manifest,
            output_contract=output_contract,
            output_contract_path=output_contract_path,
            train_command=train_command,
            train_command_normalizations=train_command_normalizations,
            feature_audit=feature_audit,
            modification_status=modification_status,
            source_evaluation_context=source_evaluation_context,
            attempt_index=attempt_index,
            repair_used=backtest_repair_used,
            previous_attempts=backtest_attempts,
        )
        backtest_attempts = list(status["agent2_backtest_attempts"])
        _persist_backtest_status(trial, dirs, status)
        if status.get("agent2_backtest_success"):
            return status
        if attempt_index > AGENT2_BACKTEST_REPAIR_MAX_ATTEMPTS:
            return status
        repair = _repair_trial_code_after_backtest_failure(
            code_dir=dirs["code"],
            wrapper=wrapper,
            plan=plan,
            execution_plan=execution_plan,
            resource_context=manifest,
            llm_client=llm_client,
            failed_status=status,
            repair_attempt_index=attempt_index,
        )
        if repair.get("code_generation_success") is not True:
            feature_audit = _audit_feature_application(plan, dirs["code"], wrapper, train_command=train_command)
            _update_code_modification_runtime_fields(dirs["code"], train_command, feature_audit, train_command_normalizations)
            modification_status = _code_modification_status_fields(dirs["code"])
            status.update(
                {
                    "agent2_backtest_success": False,
                    "agent2_backtest_repair_used": backtest_repair_used,
                    "agent2_backtest_repair_error": repair.get("code_generation_failure_reason") or "Agent2 backtest repair failed",
                    "feature_application_success": feature_audit.get("success", False),
                    "feature_application_audit_success": feature_audit.get("success", False),
                    "feature_application_audit_path": feature_audit.get("audit_path"),
                    **modification_status,
                }
            )
            _persist_backtest_status(trial, dirs, status)
            return status
        _write_code_modification_record(dirs["code"], llm_client, manifest, repair)
        backtest_repair_used = True


def _run_agent2_backtest_once(
    *,
    trial: Path,
    dirs: dict[str, Path],
    wrapper: Path,
    real_outputs: Path,
    manifest: dict[str, Any],
    output_contract: dict[str, Any],
    output_contract_path: Path,
    train_command: list[str],
    train_command_normalizations: list[dict[str, Any]],
    feature_audit: dict[str, Any],
    modification_status: dict[str, Any],
    source_evaluation_context: dict[str, Any],
    attempt_index: int,
    repair_used: bool,
    previous_attempts: list[dict[str, Any]],
) -> dict[str, Any]:
    train_log_path = dirs["logs"] / "train.log"
    with train_log_path.open("w", encoding="utf-8") as log_handle:
        log_handle.write(_train_log_header(train_command, manifest, real_outputs))
        log_handle.write(f"backtest_attempt: {attempt_index}\n")
        log_handle.flush()
        train = subprocess.run(
            train_command,
            cwd=dirs["code"].as_posix(),
            text=True,
            stdout=log_handle,
            stderr=subprocess.STDOUT,
        )

    status: dict[str, Any] = {
        "train_success": train.returncode == 0,
        "eval_success": False,
        "train_returncode": train.returncode,
        "eval_returncode": None,
        "generated_train_path": wrapper.as_posix(),
        "real_output_dir": real_outputs.as_posix(),
        "data_manifest_path": (dirs["data"] / "input_manifest.json").as_posix(),
        "train_log_path": train_log_path.as_posix(),
        "eval_log_path": (dirs["logs"] / "eval.log").as_posix(),
        "train_command": train_command,
        "train_command_normalizations": train_command_normalizations,
        "output_contract_path": output_contract_path.as_posix(),
        **_source_evaluation_status_fields(source_evaluation_context),
        "input_files": manifest.get("copied_files", {}),
        "missing_input_files": [],
        "failure_stage": None if train.returncode == 0 else "train_process",
        "feature_application_success": feature_audit.get("success", False),
        "feature_application_audit_success": feature_audit.get("success", False),
        "feature_application_audit_path": feature_audit.get("audit_path"),
        "agent2_backtest_success": False,
        "agent2_backtest_repair_used": repair_used,
        **modification_status,
    }
    eval_log_lines: list[str] = []
    if train.returncode == 0:
        try:
            standardization, contract_resolution = _standardize_with_contract_resolution(
                output_contract,
                dirs["standardized"] / "prediction.csv",
                dirs["standardized"] / "actual.csv",
                base_dirs=[real_outputs, trial, dirs["code"]],
                real_output_dir=real_outputs,
            )
            (trial / "column_mapping.json").write_text(json.dumps(standardization, ensure_ascii=False, indent=2), encoding="utf-8")
            status.update(
                {
                    "eval_success": True,
                    "eval_returncode": 0,
                    "failure_stage": None,
                    "raw_prediction_output_path": standardization["prediction_path"],
                    "raw_actual_output_path": standardization["actual_path"],
                    "prediction_path": (dirs["standardized"] / "prediction.csv").as_posix(),
                    "actual_path": (dirs["standardized"] / "actual.csv").as_posix(),
                    "standardization_status": standardization,
                    **contract_resolution,
                }
            )
            eval_log_lines.append(f"standardized output contract: {output_contract_path.as_posix()}")
            if contract_resolution.get("contract_resolution") == "auto_discovered":
                eval_log_lines.append(
                    "auto-discovered contract output: "
                    + str(contract_resolution.get("auto_discovered_output_path", ""))
                )
        except Exception as exc:  # noqa: BLE001 - write readable status instead of crashing the graph.
            status["eval_returncode"] = 1
            status["failure_stage"] = "standardize_outputs" if "column" in str(exc).lower() else "find_outputs"
            status["error"] = str(exc)
            eval_log_lines.append(str(exc))
    else:
        eval_log_lines.append("train wrapper failed; skip standardization")
    (dirs["logs"] / "eval.log").write_text("\n".join(eval_log_lines) + "\n", encoding="utf-8")
    status["agent2_backtest_success"] = bool(status.get("train_success") and status.get("eval_success"))
    status["agent2_backtest_attempts"] = [
        *previous_attempts,
        _backtest_attempt_summary(status, attempt_index),
    ]
    return status


def _source_evaluation_context(execution_plan: dict[str, Any]) -> dict[str, Any]:
    context = execution_plan.get("source_evaluation_context") if isinstance(execution_plan, dict) else {}
    return context if isinstance(context, dict) else {}


def _source_evaluation_status_fields(source_evaluation_context: dict[str, Any]) -> dict[str, Any]:
    return {
        "source_evaluation_context": source_evaluation_context,
        "preserved_train_args": source_evaluation_context.get("preserved_train_args", {})
        if isinstance(source_evaluation_context, dict)
        else {},
    }


def _agent2_codegen_failure_status(
    *,
    trial: Path,
    dirs: dict[str, Path],
    wrapper: Path,
    real_outputs: Path,
    manifest: dict[str, Any],
    train_command: list[str],
    train_command_normalizations: list[dict[str, Any]],
    output_contract_path: Path,
    source_evaluation_context: dict[str, Any],
    modification_status: dict[str, Any],
    feature_audit: dict[str, Any],
) -> dict[str, Any]:
    status = {
        "train_success": False,
        "eval_success": False,
        "train_returncode": "agent2_code_generation_failed",
        "eval_returncode": None,
        "generated_train_path": wrapper.as_posix(),
        "real_output_dir": real_outputs.as_posix(),
        "data_manifest_path": (dirs["data"] / "input_manifest.json").as_posix(),
        "train_log_path": (dirs["logs"] / "train.log").as_posix(),
        "eval_log_path": (dirs["logs"] / "eval.log").as_posix(),
        "train_command": train_command,
        "train_command_normalizations": train_command_normalizations,
        "output_contract_path": output_contract_path.as_posix(),
        **_source_evaluation_status_fields(source_evaluation_context),
        "input_files": manifest.get("copied_files", {}),
        "missing_input_files": manifest.get("missing_input_files", []),
        "failure_stage": "agent2_code_generation",
        "error": "agent2 code generation failed",
        "feature_application_success": False,
        "feature_application_audit_success": False,
        "feature_application_audit_path": feature_audit.get("audit_path"),
        "agent2_backtest_success": False,
        "agent2_backtest_attempts": [],
        "agent2_backtest_repair_used": False,
        **modification_status,
    }
    message = _agent2_code_generation_failure_message(modification_status, feature_audit)
    skip_line = "train skipped because Agent2 did not produce verified executable feature code"
    (dirs["logs"] / "train.log").write_text(
        _train_log_header(train_command, manifest, real_outputs) + message + f"\n{skip_line}\n",
        encoding="utf-8",
    )
    (dirs["logs"] / "eval.log").write_text(
        "train skipped because Agent2 code generation failed\n" + message,
        encoding="utf-8",
    )
    shutil.copy2(dirs["logs"] / "train.log", trial / "train.log")
    shutil.copy2(dirs["logs"] / "eval.log", trial / "eval.log")
    (trial / "run_status.json").write_text(json.dumps(status, ensure_ascii=False, indent=2), encoding="utf-8")
    return status


def _unresolved_evaluation_context_status(
    trial: Path,
    dirs: dict[str, Path],
    source_evaluation_context: dict[str, Any],
    execution_plan: dict[str, Any],
) -> dict[str, Any]:
    output_contract_path = trial / "output_contract.json"
    output_contract = _output_contract_from_execution_plan(execution_plan)
    write_artifact_contract(output_contract_path, output_contract)
    reason = source_evaluation_context.get("unresolved_reason") or "source evaluation context is unresolved"
    status = {
        "train_success": False,
        "eval_success": False,
        "train_returncode": "evaluation_context_unresolved",
        "eval_returncode": None,
        "generated_train_path": "",
        "real_output_dir": dirs["real_outputs"].as_posix(),
        "data_manifest_path": (dirs["data"] / "input_manifest.json").as_posix(),
        "train_log_path": (dirs["logs"] / "train.log").as_posix(),
        "eval_log_path": (dirs["logs"] / "eval.log").as_posix(),
        "train_command": [],
        "output_contract_path": output_contract_path.as_posix(),
        "input_files": {},
        "missing_input_files": [],
        "failure_stage": "evaluation_context_unresolved",
        "error": reason,
        "feature_application_success": False,
        "feature_application_audit_success": False,
        "feature_application_audit_path": "",
        "agent2_backtest_success": False,
        "agent2_backtest_attempts": [],
        "agent2_backtest_repair_used": False,
        "agent2_code_generation_success": False,
        "agent2_code_generation_failure_reason": reason,
        "agent2_failure_categories": ["evaluation_context_unresolved"],
        "agent2_normalized_rejections": [{"path": "", "reason": reason, "category": "evaluation_context_unresolved"}],
        **_source_evaluation_status_fields(source_evaluation_context),
    }
    (dirs["logs"] / "train.log").write_text(
        "train skipped because source evaluation context is unresolved\n" + reason + "\n",
        encoding="utf-8",
    )
    (dirs["logs"] / "eval.log").write_text(
        "evaluation skipped because source evaluation context is unresolved\n" + reason + "\n",
        encoding="utf-8",
    )
    shutil.copy2(dirs["logs"] / "train.log", trial / "train.log")
    shutil.copy2(dirs["logs"] / "eval.log", trial / "eval.log")
    return status


def _backtest_attempt_summary(status: dict[str, Any], attempt_index: int) -> dict[str, Any]:
    return {
        "attempt_index": attempt_index,
        "train_success": bool(status.get("train_success")),
        "eval_success": bool(status.get("eval_success")),
        "train_returncode": status.get("train_returncode"),
        "eval_returncode": status.get("eval_returncode"),
        "failure_stage": status.get("failure_stage"),
        "error": status.get("error"),
        "train_log_path": status.get("train_log_path"),
        "eval_log_path": status.get("eval_log_path"),
    }


def _persist_backtest_status(trial: Path, dirs: dict[str, Path], status: dict[str, Any]) -> None:
    if (dirs["logs"] / "train.log").exists():
        shutil.copy2(dirs["logs"] / "train.log", trial / "train.log")
    if (dirs["logs"] / "eval.log").exists():
        shutil.copy2(dirs["logs"] / "eval.log", trial / "eval.log")
    if status.get("prediction_path"):
        shutil.copy2(status["prediction_path"], trial / "standardized_prediction.csv")
    if status.get("actual_path"):
        shutil.copy2(status["actual_path"], trial / "standardized_actual.csv")
    (trial / "run_status.json").write_text(json.dumps(status, ensure_ascii=False, indent=2), encoding="utf-8")


def _clean_backtest_outputs(dirs: dict[str, Path]) -> None:
    for path in (dirs["real_outputs"], dirs["standardized"]):
        if path.exists():
            shutil.rmtree(path)
        path.mkdir(parents=True, exist_ok=True)
    for path in (dirs["logs"] / "train.log", dirs["logs"] / "eval.log"):
        if path.exists():
            path.unlink()


def _deterministic_existing_feature_train_command_repair(
    *,
    code_dir: Path,
    wrapper: Path,
    plan: dict[str, Any],
    resource_context: dict[str, Any],
    train_command: list[str],
    validation_error: str,
) -> dict[str, Any] | None:
    if "unsupported argument" not in validation_error:
        return None
    resolution = _existing_feature_resolution(code_dir, wrapper, plan)
    if not resolution or resolution.get("feature_resolution") != "existing_always_on_feature":
        return None
    repaired = _remove_feature_cli_args_from_command(train_command, _feature_cli_args(plan))
    if repaired == train_command:
        return None
    command_error = _train_command_validation_error(repaired, wrapper, resource_context)
    if command_error:
        return {
            "source": "deterministic_train_command_repair",
            "notes": [command_error],
            "accepted_schema": "train_command",
            "code_generation_success": False,
            "code_generation_failure_reason": command_error,
            "failure_categories": ["train_command_contract_error"],
            "normalized_rejections": [{"path": "train_command", "reason": command_error, "category": "train_command_contract_error"}],
            "repaired_train_command": repaired,
            "feature_resolution": resolution.get("feature_resolution"),
            "command_contract_status": "invalid",
        }
    return {
        "source": "deterministic_train_command_repair",
        "notes": [
            "Removed unsupported feature CLI args because the requested feature already has reachable executable evidence in copied code."
        ],
        "accepted_schema": "train_command",
        "code_generation_success": True,
        "code_generation_failure_reason": "",
        "repaired_train_command": repaired,
        "train_py_changed_reason": "",
        "train_py_unchanged_reason": resolution.get("train_py_unchanged_reason", ""),
        "entrypoint_import_chain_checked": True,
        "feature_resolution": resolution.get("feature_resolution"),
        "command_contract_status": "repaired",
    }


def _deterministic_missing_value_train_command_repair(
    *,
    wrapper: Path,
    plan: dict[str, Any],
    resource_context: dict[str, Any],
    train_command: list[str],
    validation_error: str,
) -> dict[str, Any] | None:
    missing_flag = _missing_value_flag_from_validation_error(validation_error)
    if not missing_flag or not wrapper.exists():
        return None

    source_text = wrapper.read_text(encoding="utf-8", errors="ignore")
    argparse_spec = _argparse_command_spec(source_text)
    flags = argparse_spec.get("flags", {}) or {}
    supported_flag = _argparse_supported_flag(flags, missing_flag) or _supported_arg_spelling(source_text, missing_flag)
    if not supported_flag:
        return None
    arg_spec = flags.get(supported_flag)
    if not arg_spec or not _argparse_spec_accepts_values(arg_spec) or _argparse_spec_accepts_multiple_values(arg_spec):
        return None
    if not _flag_matches_plan_feature_context(plan, missing_flag, supported_flag, arg_spec):
        return None

    repaired_value = _single_safe_enabled_choice(arg_spec)
    if repaired_value is None:
        return None
    repaired = _insert_missing_argparse_value(train_command, missing_flag, supported_flag, repaired_value, source_text)
    if repaired == train_command:
        return None

    command_error = _train_command_validation_error(repaired, wrapper, resource_context)
    if command_error:
        return {
            "source": "deterministic_train_command_repair",
            "notes": [command_error],
            "accepted_schema": "train_command",
            "code_generation_success": False,
            "code_generation_failure_reason": command_error,
            "failure_categories": ["train_command_contract_error"],
            "normalized_rejections": [
                {"path": "train_command", "reason": command_error, "category": "train_command_contract_error"}
            ],
            "repaired_train_command": repaired,
            "command_contract_status": "invalid",
        }
    return {
        "source": "deterministic_train_command_repair",
        "notes": [f"Added missing value {repaired_value!r} for value-taking feature flag {supported_flag}."],
        "accepted_schema": "train_command",
        "code_generation_success": True,
        "code_generation_failure_reason": "",
        "failure_categories": [],
        "normalized_rejections": [],
        "repaired_train_command": repaired,
        "train_py_changed_reason": "",
        "train_py_unchanged_reason": (
            f"train.py already defines and consumes {supported_flag}; the train_command was missing its value token."
        ),
        "entrypoint_import_chain_checked": True,
        "command_contract_status": "repaired",
    }


def _missing_value_flag_from_validation_error(validation_error: str) -> str:
    match = re.search(r"train_command missing value for (?P<flag>--[A-Za-z0-9][A-Za-z0-9_-]*)", validation_error)
    return match.group("flag") if match else ""


def _flag_matches_plan_feature_context(
    plan: dict[str, Any],
    missing_flag: str,
    supported_flag: str,
    arg_spec: dict[str, Any],
) -> bool:
    plan_flags = {
        variant
        for raw_arg in _feature_cli_args(plan)
        if str(raw_arg).startswith("--")
        for variant in _flag_variants(str(raw_arg).split("=", 1)[0])
    }
    if any(flag in plan_flags for flag in _flag_variants(missing_flag) + _flag_variants(supported_flag)):
        return True

    dest = str(arg_spec.get("dest") or "")
    feature_text = _plan_feature_context_text(plan)
    candidates = {
        missing_flag[2:],
        supported_flag[2:],
        dest,
        missing_flag[2:].replace("-", "_"),
        missing_flag[2:].replace("_", "-"),
        supported_flag[2:].replace("-", "_"),
        supported_flag[2:].replace("_", "-"),
    }
    normalized_text = re.sub(r"[^a-z0-9_ -]+", " ", feature_text.lower())
    for candidate in candidates:
        normalized = candidate.strip().lower()
        if normalized and normalized in normalized_text:
            return True
    return False


def _plan_feature_context_text(plan: dict[str, Any]) -> str:
    parts: list[str] = []
    for change in plan.get("changes", []) or []:
        if not isinstance(change, dict):
            continue
        for key in ("feature_name", "feature_type", "construction", "cli_flag"):
            if change.get(key):
                parts.append(str(change.get(key)))
        parts.extend(str(item) for item in change.get("cli_args", []) or [])
        parts.extend(str(item) for item in change.get("audit_tokens", []) or [])
        parts.extend(str(item) for item in change.get("field_sources", []) or [])
        parts.extend(str(item) for item in change.get("validation_metrics", []) or [])
    return " ".join(parts)


def _single_safe_enabled_choice(arg_spec: dict[str, Any]) -> str | None:
    choices = [str(item) for item in sorted(arg_spec.get("choices", set()) or set())]
    if not choices:
        return None
    disabled_values = {"", "0", "false", "no", "none", "off", "disable", "disabled", "null", "nil"}
    active = [item for item in choices if item.strip().lower() not in disabled_values]
    inactive = [item for item in choices if item.strip().lower() in disabled_values]
    if len(active) == 1 and inactive:
        return active[0]
    return None


def _insert_missing_argparse_value(
    command: list[str],
    missing_flag: str,
    supported_flag: str,
    value: str,
    source_text: str,
) -> list[str]:
    variants = set(_flag_variants(missing_flag) + _flag_variants(supported_flag))
    repaired: list[str] = []
    inserted = False
    index = 0
    while index < len(command):
        item = str(command[index])
        flag, inline_value = _split_inline_flag_value(item) if item.startswith("--") else (item, None)
        supported_item = _supported_arg_spelling(source_text, flag) if flag.startswith("--") else None
        repaired.append(item)
        if not inserted and inline_value is None and (flag in variants or supported_item == supported_flag):
            if index + 1 >= len(command) or str(command[index + 1]).startswith("--"):
                repaired.append(value)
                inserted = True
        index += 1
    return repaired


def _remove_feature_cli_args_from_command(command: list[str], feature_args: list[str]) -> list[str]:
    removable_flags = {
        flag
        for raw in feature_args
        for flag in _flag_variants(str(raw).split("=", 1)[0])
        if str(raw).startswith("--")
    }
    if not removable_flags:
        return list(command)
    feature_tokens = [str(item) for item in feature_args]
    repaired: list[str] = []
    index = 0
    while index < len(command):
        item = str(command[index])
        if item.startswith("--"):
            flag, _inline = _split_inline_flag_value(item)
            if flag in removable_flags:
                index += 1
                while index < len(command) and not str(command[index]).startswith("--"):
                    candidate = str(command[index])
                    if candidate in feature_tokens or not repaired:
                        index += 1
                        continue
                    break
                continue
        repaired.append(item)
        index += 1
    return repaired


def _repair_trial_code_after_train_command_failure(
    *,
    code_dir: Path,
    wrapper: Path,
    plan: dict[str, Any],
    execution_plan: dict[str, Any],
    resource_context: dict[str, Any],
    llm_client: Any | None,
    train_command: list[str],
    train_command_normalizations: list[dict[str, Any]],
    validation_error: str,
    repair_attempt_index: int,
) -> dict[str, Any]:
    editable_files = _editable_code_files(code_dir, wrapper)
    raw_output_dir = code_dir.parent / "audit" / "agent2_llm_raw"
    base_record = {
        "source": "llm_train_command_repair",
        "modified_files": [],
        "rejected_files": [],
        "fallback_used": False,
        "editable_files": sorted(editable_files),
        "notes": [],
        "attempts": [],
        "accepted_schema": None,
        "code_generation_success": False,
        "code_generation_failure_reason": "",
        "failure_categories": [],
        "normalized_rejections": [],
        "repair_guidance_history": [],
        "train_py_changed_reason": "",
        "train_py_unchanged_reason": "",
        "entrypoint_import_chain_checked": False,
        "train_command_contract_error_before_repair": validation_error,
        "repaired_train_command": [],
    }
    deterministic = _deterministic_existing_feature_train_command_repair(
        code_dir=code_dir,
        wrapper=wrapper,
        plan=plan,
        resource_context=resource_context,
        train_command=train_command,
        validation_error=validation_error,
    )
    if deterministic is not None:
        return {**base_record, **deterministic}
    deterministic = _deterministic_missing_value_train_command_repair(
        wrapper=wrapper,
        plan=plan,
        resource_context=resource_context,
        train_command=train_command,
        validation_error=validation_error,
    )
    if deterministic is not None:
        return {**base_record, **deterministic}
    if llm_client is None:
        return {**base_record, "code_generation_failure_reason": "Agent2 LLM client is required for train_command repair"}
    if callable(getattr(llm_client, "available", None)) and not llm_client.available():
        return {**base_record, "code_generation_failure_reason": "Agent2 LLM client is unavailable for train_command repair"}

    locator = _build_agent2_source_locator(code_dir, plan, execution_plan, wrapper)
    (code_dir / "agent2_source_locator.yaml").write_text(
        yaml.safe_dump(locator, allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )
    context_pack = _build_agent2_context_pack(code_dir, locator, plan, compact=False)
    if context_pack.get("context_too_large"):
        reason = str(context_pack.get("failure_reason") or "context_too_large")
        return {
            **base_record,
            "notes": [reason],
            "code_generation_failure_reason": reason,
            "failure_categories": ["context_too_large"],
            "normalized_rejections": [{"path": "", "reason": reason, "category": "context_too_large"}],
        }

    result_info = _call_agent2_llm(
        llm_client,
        "你是 ComboScope Agent2。训练命令与 copied train.py 的 argparse 契约不匹配。请只返回修复后的 YAML。",
        _agent2_train_command_repair_prompt(
            plan=plan,
            execution_plan=execution_plan,
            locator=locator,
            context_pack=context_pack,
            train_command=train_command,
            train_command_normalizations=train_command_normalizations,
            validation_error=validation_error,
        ),
        step="RepairTrainCommandContract",
        attempt_index=repair_attempt_index,
        stream=True,
        raw_output_dir=raw_output_dir,
    )
    package = _parse_agent2_code_package(result_info.get("content", ""))
    if not package:
        reason = result_info.get("error") or "LLM did not return a valid YAML package with files/edits or train_command."
        failure = _attach_agent2_failure_context(
            {
                "success": False,
                "failure_reason": reason,
                "rejected_files": [{"path": "", "reason": reason}],
            }
        )
        return {
            **base_record,
            "attempts": [result_info],
            "notes": [reason],
            "code_generation_failure_reason": reason,
            "failure_categories": failure.get("failure_categories", []),
            "normalized_rejections": failure.get("normalized_rejections", []),
        }

    repaired_train_command = _package_train_command(package)
    modified_files: list[str] = []
    modified_functions: list[str] = []
    rejected_files: list[dict[str, str]] = []
    if isinstance(package.get("files"), list) or isinstance(package.get("edits"), list):
        stage_result = _stage_agent2_code_package(
            code_dir=code_dir,
            wrapper=wrapper,
            plan=plan,
            execution_plan=execution_plan,
            resource_context=resource_context,
            editable_files=editable_files,
            package=package,
        )
        if not stage_result.get("success"):
            _cleanup_staging_dir(stage_result.get("staging_dir", code_dir / "_agent2_staging"))
            return {
                **base_record,
                "attempts": [result_info],
                "rejected_files": stage_result.get("rejected_files", []),
                "notes": _package_notes(package) or [stage_result.get("failure_reason", "train_command repair code package failed static validation")],
                "code_generation_failure_reason": stage_result.get("failure_reason", "train_command repair code package failed static validation"),
                "failure_categories": stage_result.get("failure_categories", []),
                "normalized_rejections": stage_result.get("normalized_rejections", []),
                "required_corrections": stage_result.get("required_corrections", []),
                "repaired_train_command": repaired_train_command,
            }
        _copy_staged_modified_files(stage_result["staging_dir"], code_dir, stage_result["modified_files"])
        _cleanup_staging_dir(stage_result["staging_dir"])
        modified_files = sorted(stage_result["modified_files"])
        modified_functions = sorted(stage_result.get("modified_functions", []))
        rejected_files = stage_result.get("rejected_files", [])
    elif repaired_train_command:
        command_error = _train_command_validation_error(repaired_train_command, wrapper, resource_context)
        if command_error:
            return {
                **base_record,
                "attempts": [result_info],
                "notes": _package_notes(package) or [command_error],
                "code_generation_failure_reason": command_error,
                "failure_categories": ["train_command_contract_error"],
                "normalized_rejections": [{"path": "train_command", "reason": command_error, "category": "train_command_contract_error"}],
                "repaired_train_command": repaired_train_command,
            }
    else:
        return {
            **base_record,
            "attempts": [result_info],
            "notes": ["train_command repair package contained no files/edits and no train_command"],
            "code_generation_failure_reason": "train_command repair package contained no files/edits and no train_command",
            "failure_categories": ["invalid_package"],
            "normalized_rejections": [
                {
                    "path": "",
                    "reason": "train_command repair package contained no files/edits and no train_command",
                    "category": "invalid_package",
                }
            ],
        }

    schema = "edits" if package.get("edits") else "files" if package.get("files") else "train_command"
    return {
        **base_record,
        "modified_files": modified_files,
        "modified_functions": modified_functions,
        "rejected_files": rejected_files,
        "notes": _package_notes(package),
        "attempts": [result_info],
        "accepted_schema": schema,
        "code_generation_success": True,
        "code_generation_failure_reason": "",
        "repaired_train_command": repaired_train_command,
        "command_contract_status": "repaired",
        **_agent2_train_py_reason_fields(package, modified_files),
    }


def _agent2_train_command_repair_prompt(
    *,
    plan: dict[str, Any],
    execution_plan: dict[str, Any],
    locator: dict[str, Any],
    context_pack: dict[str, Any],
    train_command: list[str],
    train_command_normalizations: list[dict[str, Any]],
    validation_error: str,
) -> str:
    primary_changes = _agent2_primary_feature_changes(plan)
    payload = {
        "task": "修复 copied trial code 与 train_command 的 argparse 契约。可以修改 trial/code 文件，也可以返回修正后的 train_command；如新增 CLI 参数，必须同步保证 train.py 解析并消费该参数。",
        "preferred_yaml_schema": {
            "edits": [{"path": "train.py", "type": "replace_function", "function": "main", "content": "<replacement function only>"}],
            "train_command": ["{python}", "{train_py}", "--output_dir", "{real_output_dir}"],
            "change_summary": "what train.py/command contract was fixed",
            "expected_training_path": "why repaired train_command now matches argparse and reaches selected features",
            "train_py_changed_reason": "required if train.py is edited",
            "train_py_unchanged_reason": "required if train.py is not edited",
            "entrypoint_import_chain_checked": True,
            "notes": ["short rationale"],
        },
        "accepted_edit_types": _agent2_edit_types_for_prompt(),
        "forbidden_default_schema": "files",
        "hard_rules": [
            "Only modify copied files under trial code/; path must be basename.",
            "Preserve train.py as the single process entrypoint.",
            "Do not modify labels, metrics, data split, evaluation windows, model family, loss/objective, or original experiment source.",
            "If train_command passes a flag, copied train.py must define that argparse flag unless parse_known_args is intentionally used by the source.",
            "Only feature CLI flags that remain in the repaired train_command are hard argparse contracts; declared-only feature cli_args may be satisfied by reachable always-on code evidence.",
            "If the right fix is command-only, return train_command and explain why no code edit is needed.",
            "For value-taking flags such as choices/string/float/int args, train_command must include the value as a separate argv token.",
            "train_command and cli_args must contain only real argv tokens; remove explanatory words like '(optional boolean flag ...)'.",
            "Return YAML only, without Markdown.",
        ],
        "selected_feature_actions": primary_changes,
        "primary_feature_change": primary_changes[0] if primary_changes else {},
        "experiment_plan": _compact_experiment_plan(plan),
        "agent2_execution_plan": execution_plan,
        "source_locator": _compact_agent2_source_locator(locator),
        "validation_failure": {
            "error": validation_error,
            "train_command": train_command,
            "train_command_normalizations": train_command_normalizations,
        },
        "context_pack": context_pack,
    }
    return _budgeted_agent2_yaml_prompt(payload)


def _package_train_command(package: dict[str, Any]) -> list[str]:
    train_command = package.get("train_command")
    if not isinstance(train_command, list) or not train_command:
        return []
    return [str(item) for item in train_command]


def _apply_repaired_plan_cli_args(plan: dict[str, Any], repaired_train_command: list[str], wrapper: Path) -> None:
    source_text = wrapper.read_text(encoding="utf-8", errors="ignore") if wrapper.exists() else ""
    repaired_args = _feature_cli_tokens_from_command(repaired_train_command, source_text, _feature_cli_args(plan))
    for change in plan.get("changes", []) or []:
        if not isinstance(change, dict):
            continue
        raw_flags = [str(item) for item in _change_cli_args(change) if str(item).startswith("--")]
        if not raw_flags:
            continue
        cleaned: list[str] = []
        for raw_flag in raw_flags:
            supported = _supported_arg_spelling(source_text, raw_flag) or raw_flag
            tokens = repaired_args.get(supported)
            if tokens is None:
                for variant in _flag_variants(supported) + _flag_variants(raw_flag):
                    tokens = repaired_args.get(variant)
                    if tokens is not None:
                        break
            if tokens:
                cleaned.extend(tokens)
        change["cli_args"] = cleaned


def _feature_cli_tokens_from_command(command: list[str], source_text: str, feature_args: list[str]) -> dict[str, list[str]]:
    argparse_spec = _argparse_command_spec(source_text)
    flags = argparse_spec.get("flags", {})
    target_by_variant: dict[str, str] = {}
    for raw_arg in feature_args:
        raw_flag = str(raw_arg)
        if not raw_flag.startswith("--"):
            continue
        supported = _argparse_supported_flag(flags, raw_flag) or _supported_arg_spelling(source_text, raw_flag) or raw_flag
        for variant in _flag_variants(raw_flag) + _flag_variants(supported):
            target_by_variant[variant] = supported

    found: dict[str, list[str]] = {}
    index = 0
    while index < len(command):
        item = str(command[index])
        if not item.startswith("--"):
            index += 1
            continue
        flag, inline_value = _split_inline_flag_value(item)
        supported = _argparse_supported_flag(flags, flag) or _supported_arg_spelling(source_text, flag) or flag
        target = target_by_variant.get(flag) or target_by_variant.get(supported)
        arg_spec = flags.get(supported, {})
        if target is None:
            index += 1
            if inline_value is None and _argparse_spec_accepts_values(arg_spec):
                _, index, _ = _collect_argparse_values(command, index - 1, None, supported, arg_spec)
            continue
        if not _argparse_spec_accepts_values(arg_spec):
            found[target] = [supported]
            index += 1
            continue
        values, next_index, _ = _collect_argparse_values(command, index, inline_value, supported, arg_spec)
        found[target] = [supported, *values]
        index = next_index
    return found


def _feature_cli_reconciliation(
    declared_cli_args: list[str],
    source_text: str,
    train_command: list[str] | None,
) -> dict[str, Any]:
    if train_command is None:
        flags = [str(item) for item in declared_cli_args if str(item).startswith("--")]
        return {
            "declared_cli_args": declared_cli_args,
            "effective_cli_args": declared_cli_args,
            "ignored_intent_cli_args": [],
            "notes": ["no train_command supplied; declared feature CLI args are audited as the executable contract"],
            "effective_by_flag": {flag: [flag] for flag in flags},
        }

    command_tokens = _feature_cli_tokens_from_command([str(item) for item in train_command], source_text, declared_cli_args)
    effective_cli_args: list[str] = []
    ignored_flags: list[str] = []
    notes: list[str] = []
    for raw_arg in declared_cli_args:
        raw_flag = str(raw_arg)
        if not raw_flag.startswith("--"):
            continue
        supported = _supported_arg_spelling(source_text, raw_flag) or raw_flag
        tokens = command_tokens.get(supported)
        if tokens is None:
            for variant in _flag_variants(supported) + _flag_variants(raw_flag):
                tokens = command_tokens.get(variant)
                if tokens is not None:
                    break
        if tokens:
            effective_cli_args.extend(tokens)
        else:
            ignored_flags.append(raw_flag)

    ignored_flags = list(dict.fromkeys(ignored_flags))
    if ignored_flags:
        notes.append(
            "declared feature CLI args absent from effective train_command are treated as feature intent, not hard argparse contracts"
        )
    if effective_cli_args:
        notes.append("feature CLI audit is scoped to args present in the effective train_command")
    elif declared_cli_args:
        notes.append("no declared feature CLI args are present in the effective train_command")
    return {
        "declared_cli_args": declared_cli_args,
        "effective_cli_args": effective_cli_args,
        "ignored_intent_cli_args": ignored_flags,
        "notes": notes,
        "effective_by_flag": command_tokens,
    }


def _merge_train_command_repair_record(code_dir: Path, repair: dict[str, Any]) -> None:
    path = code_dir / "agent2_code_modification.yaml"
    record = _read_agent2_code_modification(code_dir)
    if not record:
        record = {"agent": "Agent2", "step": "ModifyTrialTrainPy"}
    attempts = [_sanitize_attempt_record(item) for item in repair.get("attempts", [])]
    record["train_command_repair_attempts"] = [
        *(record.get("train_command_repair_attempts", []) or []),
        *attempts,
    ]
    record["train_command_contract_error_before_repair"] = repair.get("train_command_contract_error_before_repair", "")
    record["train_command_repair_success"] = repair.get("code_generation_success") is True
    record["train_command_repair_error"] = repair.get("code_generation_failure_reason", "")
    if repair.get("repaired_train_command"):
        record["repaired_train_command"] = repair.get("repaired_train_command")
    if repair.get("feature_resolution"):
        record["feature_resolution"] = repair.get("feature_resolution")
    if repair.get("command_contract_status"):
        record["command_contract_status"] = repair.get("command_contract_status")
    modified = sorted({*(str(item) for item in record.get("modified_files", []) or []), *(str(item) for item in repair.get("modified_files", []) or [])})
    record["modified_files"] = modified
    modified_functions = sorted(
        {
            *(str(item) for item in record.get("modified_functions", []) or []),
            *(str(item) for item in repair.get("modified_functions", []) or []),
        }
    )
    if modified_functions:
        record["modified_functions"] = modified_functions
    if repair.get("notes"):
        record["notes"] = [*(record.get("notes", []) or []), *repair.get("notes", [])]
    if repair.get("code_generation_success") is True:
        record["source"] = str(record.get("source") or "llm_required") + "+llm_train_command_repair"
        existing_feature_schema = (
            record.get("feature_resolution") == "existing_always_on_feature"
            or record.get("accepted_schema") == "existing_code_feature"
        )
        if existing_feature_schema:
            record["code_generation_success"] = True
            record["feature_code_success"] = True
            record["code_generation_failure_reason"] = ""
            record["failure_categories"] = []
            record["normalized_rejections"] = []
            record["required_corrections"] = []
        elif record.get("code_generation_success") is True:
            record["code_generation_failure_reason"] = ""
        else:
            record["code_generation_success"] = False
            record["code_generation_failure_reason"] = (
                record.get("code_generation_failure_reason")
                or "feature code generation was not verified before train_command repair"
            )
        record["command_contract_status"] = repair.get("command_contract_status") or "repaired"
        if repair.get("accepted_schema"):
            record["train_command_repair_schema"] = repair.get("accepted_schema")
        for key in ("train_py_changed_reason", "train_py_unchanged_reason", "entrypoint_import_chain_checked"):
            value = repair.get(key)
            if value:
                record[key] = value
    elif repair.get("code_generation_failure_reason"):
        record["command_contract_status"] = "invalid"
    path.write_text(yaml.safe_dump(record, allow_unicode=True, sort_keys=False), encoding="utf-8")


def _repair_trial_code_after_backtest_failure(
    *,
    code_dir: Path,
    wrapper: Path,
    plan: dict[str, Any],
    execution_plan: dict[str, Any],
    resource_context: dict[str, Any],
    llm_client: Any | None,
    failed_status: dict[str, Any],
    repair_attempt_index: int,
) -> dict[str, Any]:
    editable_files = _editable_code_files(code_dir, wrapper)
    raw_output_dir = code_dir.parent / "audit" / "agent2_llm_raw"
    base_record = {
        "source": "llm_backtest_repair",
        "modified_files": [],
        "rejected_files": [],
        "fallback_used": False,
        "editable_files": sorted(editable_files),
        "notes": [],
        "attempts": [],
        "accepted_schema": None,
        "code_generation_success": False,
        "code_generation_failure_reason": "",
        "failure_categories": [],
        "normalized_rejections": [],
        "repair_guidance_history": [],
        "train_py_changed_reason": "",
        "train_py_unchanged_reason": "",
        "entrypoint_import_chain_checked": False,
    }
    if llm_client is None:
        return {**base_record, "code_generation_failure_reason": "Agent2 LLM client is required for backtest repair"}
    if callable(getattr(llm_client, "available", None)) and not llm_client.available():
        return {**base_record, "code_generation_failure_reason": "Agent2 LLM client is unavailable for backtest repair"}
    locator = _build_agent2_source_locator(code_dir, plan, execution_plan, wrapper)
    (code_dir / "agent2_source_locator.yaml").write_text(
        yaml.safe_dump(locator, allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )
    context_pack = _build_agent2_context_pack(code_dir, locator, plan, compact=False)
    if context_pack.get("context_too_large"):
        reason = str(context_pack.get("failure_reason") or "context_too_large")
        return {
            **base_record,
            "notes": [reason],
            "code_generation_failure_reason": reason,
            "failure_categories": ["context_too_large"],
            "normalized_rejections": [{"path": "", "reason": reason, "category": "context_too_large"}],
        }
    prompt = _agent2_backtest_repair_prompt(
        plan=plan,
        execution_plan=execution_plan,
        locator=locator,
        context_pack=context_pack,
        failed_status=failed_status,
        raw_output_dir=raw_output_dir,
    )
    result_info = _call_agent2_llm(
        llm_client,
        "你是 ComboScope Agent2。上一次代码已通过静态校验但回测失败。请只返回修复后的完整 YAML files 包。",
        prompt,
        step="RepairBacktestFailure",
        attempt_index=repair_attempt_index,
        stream=True,
        raw_output_dir=raw_output_dir,
    )
    package = _parse_agent2_code_package(result_info.get("content", ""))
    if not package:
        reason = result_info.get("error") or "LLM did not return a valid YAML edits/files package."
        failure = _attach_agent2_failure_context(
            {
                "success": False,
                "failure_reason": reason,
                "rejected_files": [{"path": "", "reason": reason}],
            }
        )
        return {
            **base_record,
            "attempts": [result_info],
            "notes": [reason],
            "code_generation_failure_reason": reason,
            "failure_categories": failure.get("failure_categories", []),
            "normalized_rejections": failure.get("normalized_rejections", []),
        }
    stage_result = _stage_agent2_code_package(
        code_dir=code_dir,
        wrapper=wrapper,
        plan=plan,
        execution_plan=execution_plan,
        resource_context=resource_context,
        editable_files=editable_files,
        package=package,
    )
    if not stage_result.get("success"):
        _cleanup_staging_dir(stage_result.get("staging_dir", code_dir / "_agent2_staging"))
        return {
            **base_record,
            "attempts": [result_info],
            "rejected_files": stage_result.get("rejected_files", []),
            "notes": _package_notes(package) or [stage_result.get("failure_reason", "backtest repair code package failed static validation")],
            "code_generation_failure_reason": stage_result.get("failure_reason", "backtest repair code package failed static validation"),
            "failure_categories": stage_result.get("failure_categories", []),
            "normalized_rejections": stage_result.get("normalized_rejections", []),
            "required_corrections": stage_result.get("required_corrections", []),
        }
    _copy_staged_modified_files(stage_result["staging_dir"], code_dir, stage_result["modified_files"])
    _cleanup_staging_dir(stage_result["staging_dir"])
    return {
        **base_record,
        "modified_files": sorted(stage_result["modified_files"]),
        "modified_functions": sorted(stage_result.get("modified_functions", [])),
        "rejected_files": stage_result.get("rejected_files", []),
        "notes": _package_notes(package),
        "attempts": [result_info],
        "accepted_schema": "edits" if package.get("edits") else "files",
        "code_generation_success": True,
        "code_generation_failure_reason": "",
        **_agent2_train_py_reason_fields(package, stage_result["modified_files"]),
    }


def _agent2_backtest_repair_prompt(
    *,
    plan: dict[str, Any],
    execution_plan: dict[str, Any],
    locator: dict[str, Any],
    context_pack: dict[str, Any],
    failed_status: dict[str, Any],
    raw_output_dir: Path,
) -> str:
    primary_changes = _agent2_primary_feature_changes(plan)
    payload = {
        "task": "上一版 trial/code 已通过路径、compile、runtime call contract 静态校验，但最小回测训练或输出标准化失败。请修复 selected_feature_actions 对应运行问题，并返回窄 YAML edits。",
        "preferred_yaml_schema": {
            "edits": [{"path": "train.py", "type": "replace_function", "function": "main", "content": "<replacement function only>"}],
            "change_summary": "what runtime/standardization failure was fixed",
            "expected_training_path": "why current train_command should now run and produce output_contract files",
            "entrypoint_import_chain_checked": True,
        },
        "accepted_edit_types": _agent2_edit_types_for_prompt(),
        "forbidden_default_schema": "files",
        "hard_rules": [
            "Only modify copied files under trial code/; path must be basename.",
            "Do not modify labels, data split, metrics, model family, loss/objective, or original experiment source.",
            "Preserve train.py as the single process entrypoint.",
            "Preserve every action in selected_feature_actions; do not drop a coordinated feature action while repairing runtime errors.",
            "Fix the concrete runtime or output-contract error from train_log/eval_log; do not regenerate an unrelated pipeline.",
            "Return YAML only, without Markdown.",
        ],
        "selected_feature_actions": primary_changes,
        "primary_feature_change": primary_changes[0] if primary_changes else {},
        "experiment_plan": _compact_experiment_plan(plan),
        "agent2_execution_plan": execution_plan,
        "source_locator": _compact_agent2_source_locator(locator),
        "failed_status": _backtest_failure_status_for_prompt(failed_status),
        "train_log": _read_text_for_prompt(Path(str(failed_status.get("train_log_path") or ""))),
        "eval_log": _read_text_for_prompt(Path(str(failed_status.get("eval_log_path") or ""))),
        "last_agent2_code_package": _latest_agent2_package_content(raw_output_dir),
        "context_pack": context_pack,
    }
    return _budgeted_agent2_yaml_prompt(payload)


def _backtest_failure_status_for_prompt(status: dict[str, Any]) -> dict[str, Any]:
    keys = [
        "train_success",
        "eval_success",
        "train_returncode",
        "eval_returncode",
        "failure_stage",
        "error",
        "train_command",
        "output_contract_path",
        "real_output_dir",
        "agent2_backtest_attempts",
    ]
    return {key: status.get(key) for key in keys if key in status}


def _read_text_for_prompt(path: Path) -> str:
    if not path.exists() or not path.is_file():
        return ""
    return path.read_text(encoding="utf-8", errors="ignore")


def _latest_agent2_package_content(raw_output_dir: Path) -> str:
    if not raw_output_dir.exists():
        return ""
    for path in sorted(raw_output_dir.glob("*.yaml"), reverse=True):
        try:
            data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        except Exception:
            continue
        if data.get("step") in {"GenerateCodeEdits", "RepairCodeEdits", "RepairBacktestFailure"}:
            content = str(data.get("content") or "")
            if content.strip():
                return content
    return ""


def _train_log_header(command: list[str], manifest: dict[str, Any], real_outputs: Path) -> str:
    return "\n".join(
        [
            "ComboScope Agent2 train context",
            f"train_command: {command}",
            f"input_files: {manifest.get('copied_files', {})}",
            f"missing_input_files: {manifest.get('missing_input_files', [])}",
            f"real_output_dir: {real_outputs.as_posix()}",
            "",
        ]
    )


def _standardize_with_contract_resolution(
    output_contract: dict[str, Any],
    prediction_output: Path,
    actual_output: Path,
    *,
    base_dirs: list[Path],
    real_output_dir: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    try:
        standardization = standardize_from_contract(
            output_contract,
            prediction_output,
            actual_output,
            base_dirs=base_dirs,
        )
        return standardization, {"contract_resolution": "declared"}
    except Exception as first_error:  # noqa: BLE001 - enrich path resolution failures with real-output discovery.
        if "does not resolve to an existing file" not in str(first_error):
            raise
        discovered = _discover_single_contract_output(output_contract, real_output_dir, first_error)
        discovered_contract = dict(output_contract)
        discovered_contract["prediction_path"] = discovered.as_posix()
        discovered_contract["actual_path"] = discovered.as_posix()
        standardization = standardize_from_contract(
            discovered_contract,
            prediction_output,
            actual_output,
            base_dirs=base_dirs,
        )
        return standardization, {
            "contract_resolution": "auto_discovered",
            "auto_discovered_output_path": discovered.as_posix(),
        }


def _discover_single_contract_output(contract: dict[str, Any], real_output_dir: Path, first_error: Exception) -> Path:
    prediction_column = str(contract.get("prediction_column") or "").strip()
    actual_column = str(contract.get("actual_column") or "").strip()
    if not prediction_column or not actual_column:
        raise first_error
    candidates: list[Path] = []
    for candidate in sorted(real_output_dir.rglob("*.csv")):
        headers = _csv_headers(candidate)
        if prediction_column in headers and actual_column in headers:
            candidates.append(candidate.resolve())
    if len(candidates) == 1:
        return candidates[0]
    candidate_text = ", ".join(path.as_posix() for path in candidates) if candidates else "none"
    if not candidates:
        raise FileNotFoundError(
            f"{first_error}; no CSV in real_output_dir contains required output fields "
            f"{prediction_column!r} and {actual_column!r}"
        ) from first_error
    raise FileNotFoundError(
        f"{first_error}; multiple CSVs in real_output_dir contain required output fields "
        f"{prediction_column!r} and {actual_column!r}: {candidate_text}"
    ) from first_error


def _csv_headers(path: Path) -> set[str]:
    for encoding in ("utf-8-sig", "utf-8", "gb18030", "gbk", "latin1"):
        try:
            with path.open(newline="", encoding=encoding) as handle:
                row = next(csv.reader(handle), [])
            return {str(item).lstrip("\ufeff") for item in row}
        except UnicodeDecodeError:
            continue
        except Exception:
            return set()
    return set()


def _normalize_experiment_plan_for_trial(plan: dict[str, Any], source_text: str, trial_id: str, source_entrypoint: str) -> dict[str, Any]:
    validation: list[dict[str, Any]] = []
    changes = _agent2_primary_feature_changes(plan)
    normalized_changes: list[dict[str, Any]] = []
    for change in changes:
        normalized = dict(change)
        _ensure_feature_audit_tokens(normalized)
        note = _normalize_duplicate_rolling_change(normalized, source_text)
        if note:
            validation.append(note)
            _ensure_feature_audit_tokens(normalized)
        normalized_changes.append(normalized)
    if normalized_changes:
        plan["changes"] = normalized_changes
    plan["editable_files"] = _normalize_plan_editable_files(plan.get("editable_files", []), trial_id, source_entrypoint, normalized_changes)
    if validation:
        plan["plan_validation"] = validation
    return plan


def _agent2_primary_feature_changes(plan: dict[str, Any]) -> list[dict[str, Any]]:
    """Agent2 runs the complete selected highest-priority feature experiment."""
    changes = [change for change in plan.get("changes", []) if isinstance(change, dict)]
    if changes:
        return [dict(change) for change in changes]

    candidates = [item for item in plan.get("candidate_experiments", []) or [] if isinstance(item, dict)]
    candidates_with_actions = [item for item in candidates if isinstance(item.get("feature_actions"), list) and item["feature_actions"]]
    if candidates_with_actions:
        top = min(candidates_with_actions, key=lambda item: _candidate_priority(item))
        return [dict(action) for action in top.get("feature_actions", []) if isinstance(action, dict)]

    return []


def _candidate_priority(candidate: dict[str, Any]) -> tuple[int, str]:
    raw_priority = candidate.get("priority", 9999)
    try:
        priority = int(raw_priority)
    except Exception:
        priority = 9999
    return priority, str(candidate.get("experiment_id") or candidate.get("title") or "")


def _ensure_feature_audit_tokens(change: dict[str, Any]) -> None:
    feature_name = str(change.get("feature_name") or "").strip()
    feature_type = str(change.get("feature_type") or "").strip()
    construction = str(change.get("construction") or "")
    lowered_text = " ".join([feature_name, feature_type, construction]).lower()
    slug = str(change.get("feature_slug") or "").strip() or _default_feature_slug(feature_name, feature_type, construction)
    if slug:
        change["feature_slug"] = slug

    tokens: list[str] = []
    for raw in change.get("audit_tokens", []) or []:
        token = str(raw).strip()
        if token:
            tokens.append(token)
    for token in [slug, feature_type]:
        if token:
            tokens.append(token)
    if "rolling" in lowered_text:
        tokens.append("rolling_stat")
        if "sum" in lowered_text or "总和" in lowered_text:
            tokens.append("rolling_sum")
        if "mean" in lowered_text or "均值" in lowered_text or "avg" in lowered_text:
            tokens.append("rolling_mean")
    if "lifecycle" in lowered_text or "package_age" in lowered_text or "days_to_end" in lowered_text or "生命周期" in lowered_text:
        tokens.extend(["lifecycle_interaction", "package_age_days", "days_to_end"])
        change["feature_type"] = feature_type or "lifecycle_interaction"
    if "holiday" in lowered_text or "节假日" in lowered_text:
        tokens.extend(["holiday_position", "holiday_span", "holiday_eve"])
        change["feature_type"] = feature_type or "holiday_position"
    if "calibration" in lowered_text or "校准" in lowered_text:
        tokens.extend(["group_calibration", "calibration"])
        change["feature_type"] = feature_type or "group_calibration"
    if "sample_weight" in lowered_text or "train_weight" in lowered_text or "训练权重" in lowered_text:
        tokens.extend(["train_weight", "sample_weight"])
        change["feature_type"] = feature_type or "train_weight"
    if "zero" in lowered_text or "零销量" in lowered_text or "guardrail" in lowered_text:
        tokens.extend(["zero_demand_guardrail", "guardrail", "zero_demand"])
        change["feature_type"] = feature_type or "zero_demand_guardrail"
    if tokens:
        change["audit_tokens"] = list(dict.fromkeys(tokens))
    if not change.get("field_sources"):
        change["field_sources"] = _default_field_sources_for_change(change)
    if not change.get("construction"):
        change["construction"] = _default_construction_for_change(change)
    change.setdefault("code_locations", [])


def _default_feature_slug(feature_name: str, feature_type: str, construction: str) -> str:
    text = " ".join([feature_name, feature_type, construction]).lower()
    if "rolling" in text and ("sum" in text or "mean" in text or "avg" in text):
        return "rolling_recent_sales_sum_mean"
    if "lifecycle" in text or "package_age" in text or "days_to_end" in text or "生命周期" in text:
        return "lifecycle_interaction"
    if "holiday" in text or "节假日" in text:
        return "holiday_position"
    if "calibration" in text or "校准" in text:
        return "group_calibration"
    if "sample_weight" in text or "train_weight" in text or "训练权重" in text:
        return "train_weight"
    if "zero" in text or "零销量" in text or "guardrail" in text:
        return "zero_demand_guardrail"
    return _slugify_feature_token(feature_name or feature_type or "feature")


def _default_field_sources_for_change(change: dict[str, Any]) -> list[str]:
    feature_type = str(change.get("feature_type") or "").lower()
    if feature_type == "rolling_stat":
        return ["ds", "label_col", "package_id_cols"]
    if feature_type == "lifecycle_interaction":
        return ["package_age_days", "days_to_end"]
    if feature_type == "holiday_position":
        return ["is_holiday", "holiday_span_day_idx", "holiday_span_days"]
    if feature_type == "group_calibration":
        return ["validation_prediction", "validation_actual", "group_fields"]
    if feature_type == "train_weight":
        return ["date_col", "sample_weight", "model.fit"]
    if feature_type == "zero_demand_guardrail":
        return ["prediction", "zero_demand_signal", "guardrail"]
    return []


def _default_construction_for_change(change: dict[str, Any]) -> str:
    feature_type = str(change.get("feature_type") or "").lower()
    if feature_type == "rolling_stat":
        return (
            "In build_features, use the existing rolling feature path and include sum/mean statistics "
            "with existing id columns, label_col, rolling_windows, and history_gap_days."
        )
    if feature_type == "lifecycle_interaction":
        return "In build_features, derive lifecycle interaction columns from existing package_age_days and days_to_end before feature_cols is returned."
    if feature_type == "holiday_position":
        return "In build_features or an existing holiday helper, derive holiday first/middle/last/eve position columns from existing holiday span fields."
    if feature_type == "group_calibration":
        return "Use an existing group calibration CLI/helper if present; otherwise make only minimal wiring changes and preserve metrics/split logic."
    if feature_type == "train_weight":
        return "Use an existing train weight CLI/helper if present and ensure sample_weight reaches the model fit call."
    if feature_type == "zero_demand_guardrail":
        return "Use an existing post-prediction guardrail path if present and ensure the adjusted prediction is consumed by outputs."
    return "Implement the feature through an existing reachable train or feature-building path with minimal code changes."


def _slugify_feature_token(text: str) -> str:
    slug = re.sub(r"[^0-9A-Za-z_]+", "_", text).strip("_").lower()
    return re.sub(r"_+", "_", slug)


def _normalize_duplicate_rolling_change(change: dict[str, Any], source_text: str) -> dict[str, Any] | None:
    feature_name = str(change.get("feature_name") or "")
    windows = _rolling_windows_from_feature_name(feature_name)
    default_windows = _argparse_default_int_list(source_text, "--rolling_windows")
    if not windows or not default_windows or set(windows) - set(default_windows):
        return None
    replacement = _expanded_rolling_windows(default_windows)
    change["feature_name"] = f"extend_rolling_windows_{'_'.join(str(item) for item in replacement)}"
    change["feature_type"] = change.get("feature_type") or "rolling_stat"
    change["cli_args"] = ["--rolling_windows", *[str(item) for item in replacement]]
    change["construction"] = (
        f"Source already enables rolling windows {default_windows}; vary the existing CLI parameter to {replacement}."
    )
    change.setdefault("code_locations", [])
    return {
        "category": "duplicate_existing_cli_default",
        "original_feature_name": feature_name,
        "replacement_feature_name": change["feature_name"],
        "cli_arg": "--rolling_windows",
        "source_default": default_windows,
        "replacement_value": replacement,
    }


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


def _argparse_default_int_list(source_text: str, flag: str) -> list[int]:
    if _supported_arg_spelling(source_text, flag) is None:
        return []
    pattern = re.compile(r"add_argument\((?P<args>.*?)\)", re.DOTALL)
    for match in pattern.finditer(source_text):
        args = match.group("args")
        if flag not in args and flag.replace("_", "-") not in args:
            continue
        default_match = re.search(r"default\s*=\s*\[(?P<values>[^\]]*)\]", args, re.DOTALL)
        if not default_match:
            continue
        values = [int(item) for item in re.findall(r"-?\d+", default_match.group("values"))]
        if values:
            return values
    return []


def _normalize_plan_editable_files(
    editable_files: list[Any],
    trial_id: str,
    source_entrypoint: str,
    changes: list[dict[str, Any]],
) -> list[str]:
    files = [f"runs/{trial_id}/code/train.py"]
    source_entrypoint_name = Path(source_entrypoint).name
    for raw in editable_files:
        name = Path(str(raw)).name
        if name.endswith(".py"):
            files.append(f"runs/{trial_id}/code/{name}")
    for change in changes:
        for location in change.get("code_locations", []) or []:
            name = Path(str(location)).name
            if name.endswith(".py") and name not in {source_entrypoint_name, "train.py"}:
                files.append(f"runs/{trial_id}/code/{name}")
    return list(dict.fromkeys(files))


def _feature_cli_args(plan: dict[str, Any]) -> list[str]:
    args: list[str] = []
    for change in plan.get("changes", []):
        args.extend(_change_cli_args(change))
    return args


def _feature_arg_overrides(plan: dict[str, Any]) -> dict[str, bool]:
    overrides: dict[str, bool] = {}
    for arg in _feature_cli_args(plan):
        if not arg.startswith("--"):
            continue
        if not (arg.startswith("--enable-") or arg.startswith("--no-enable-")):
            continue
        enabled = not arg.startswith("--no-")
        name = arg[5:] if arg.startswith("--no-") else arg[2:]
        name = name.replace("-", "_")
        overrides[name] = enabled
    return overrides


def _trial_command(
    plan: dict[str, Any],
    experiment_dir: str | Path,
    trial: Path,
    wrapper: Path,
    execution_plan: dict[str, Any],
    resource_context: dict[str, Any] | None = None,
    normalizations_out: list[dict[str, Any]] | None = None,
) -> list[str]:
    experiment = Path(experiment_dir).resolve()
    python_bin = experiment / ".venv" / "bin" / "python"
    python_executable = python_bin.as_posix() if python_bin.exists() else sys.executable
    raw_command = execution_plan.get("train_command")
    if not isinstance(raw_command, list) or not raw_command:
        raise ValueError("Agent2 execution_plan.train_command must be a non-empty list")
    placeholders = {
        "{python}": python_executable,
        "{train_py}": wrapper.as_posix(),
        "{trial_id}": str(plan["trial_id"]),
        "{trial_dir}": trial.as_posix(),
        "{real_output_dir}": _trial_dirs(trial)["real_outputs"].as_posix(),
        "{output_dir}": _trial_dirs(trial)["real_outputs"].as_posix(),
    }
    command = []
    for item in raw_command:
        text = str(item)
        for key, value in placeholders.items():
            text = text.replace(key, value)
        if text in {"python", "python3"}:
            text = python_executable
        elif text in {"train.py", "./train.py"}:
            text = wrapper.as_posix()
        command.append(text)
    if Path(command[0]).name.startswith("python") and len(command) > 1 and command[1] in {"train.py", "./train.py"}:
        command[1] = wrapper.as_posix()
    source_text = wrapper.read_text(encoding="utf-8", errors="ignore") if wrapper.exists() else ""
    command = _apply_preserved_train_args(command, source_text, execution_plan)
    command = _merge_plan_cli_args(command, plan, source_text)
    resource_context = resource_context or _read_input_manifest(_trial_dirs(trial)["data"] / "input_manifest.json")
    command = _apply_declared_resource_cli_args(command, source_text, resource_context)
    command, normalizations = _normalize_train_command_argv(command, source_text)
    if normalizations_out is not None:
        normalizations_out.extend(normalizations)
    return command


def _audit_train_command_for_code(
    *,
    plan: dict[str, Any],
    wrapper: Path,
    execution_plan: dict[str, Any] | None,
    resource_context: dict[str, Any] | None,
    package_train_command: list[str] | None = None,
) -> list[str] | None:
    raw_command = package_train_command
    if not raw_command and isinstance(execution_plan, dict):
        raw_plan_command = execution_plan.get("train_command")
        if isinstance(raw_plan_command, list) and raw_plan_command:
            raw_command = [str(item) for item in raw_plan_command]
    if not raw_command:
        return None

    trial = wrapper.parent.parent
    placeholders = {
        "{python}": sys.executable,
        "{train_py}": wrapper.as_posix(),
        "{trial_id}": str(plan.get("trial_id") or trial.name),
        "{trial_dir}": trial.as_posix(),
        "{real_output_dir}": _trial_dirs(trial)["real_outputs"].as_posix(),
        "{output_dir}": _trial_dirs(trial)["real_outputs"].as_posix(),
    }
    command: list[str] = []
    for item in raw_command:
        text = str(item)
        for key, value in placeholders.items():
            text = text.replace(key, value)
        if text in {"python", "python3"}:
            text = sys.executable
        elif text in {"train.py", "./train.py"}:
            text = wrapper.as_posix()
        command.append(text)
    if Path(command[0]).name.startswith("python") and len(command) > 1 and command[1] in {"train.py", "./train.py"}:
        command[1] = wrapper.as_posix()

    source_text = wrapper.read_text(encoding="utf-8", errors="ignore") if wrapper.exists() else ""
    command = _apply_preserved_train_args(command, source_text, execution_plan or {})
    command = _merge_plan_cli_args(command, plan, source_text)
    command = _apply_declared_resource_cli_args(command, source_text, resource_context)
    command, _ = _normalize_train_command_argv(command, source_text)
    return command


def _normalize_train_command_argv(command: list[str], source_text: str) -> tuple[list[str], list[dict[str, Any]]]:
    tokens, shell_records = _split_cli_shell_tokens([str(item) for item in command])
    tokens, argparse_records = _normalize_argparse_tokens(tokens, source_text)
    return tokens, shell_records + argparse_records


def _apply_llm_logged_eval_time_overrides(
    train_command: list[str],
    wrapper: Path,
    execution_plan: dict[str, Any],
    source_evaluation_context: dict[str, Any],
    llm_client: Any | None,
) -> tuple[list[str], list[dict[str, Any]]]:
    log_text, log_path = _source_eval_log_text(execution_plan, source_evaluation_context)
    if not log_text.strip():
        return train_command, []
    if llm_client is None:
        return train_command, []
    if callable(getattr(llm_client, "available", None)) and not llm_client.available():
        return train_command, []

    source_text = wrapper.read_text(encoding="utf-8", errors="ignore") if wrapper.exists() else ""
    result = _call_agent2_llm(
        llm_client,
        "你是 ComboScope 的日志评测时间判定器。只从日志中判定真实评测/回测窗口，返回 YAML。",
        _logged_eval_time_prompt(
            log_text=log_text,
            log_path=log_path,
            source_evaluation_context=source_evaluation_context,
            train_command=train_command,
            argparse_args=_argparse_args_index(source_text),
        ),
        step="DetermineLoggedEvalTime",
        attempt_index=1,
        stream=False,
        timeout=AGENT2_LOGGED_EVAL_TIME_TIMEOUT,
    )
    parsed = _parse_logged_eval_time_response(str(result.get("content") or ""))
    if not parsed.get("available") or str(parsed.get("confidence") or "").lower() != "high":
        return train_command, []

    updated = list(train_command)
    normalizations: list[dict[str, Any]] = []
    reason = str(parsed.get("reason") or "").strip()
    overrides: list[tuple[str, str, str]] = []
    train_eval_start = _compact_eval_date(parsed.get("train_eval_start"))
    train_eval_end = _compact_eval_date(parsed.get("train_eval_end"))
    test_start = _compact_eval_date(parsed.get("test_start"))
    test_end = _compact_eval_date(parsed.get("test_end"))

    if train_eval_start and train_eval_end:
        overrides.extend(
            [
                ("--train_eval_start", train_eval_start, "train_eval_start"),
                ("--train_eval_end", train_eval_end, "train_eval_end"),
            ]
        )
    elif test_end:
        overrides.append(("--train_eval_end", test_end, "test_end"))

    if test_start and test_end:
        overrides.extend(
            [
                ("--fixed_test_start", test_start, "test_start"),
                ("--fixed_test_end", test_end, "test_end"),
            ]
        )

    for raw_flag, value, value_source in overrides:
        supported = _supported_arg_spelling(source_text, raw_flag)
        if supported is None:
            continue
        updated = _replace_or_append_command_flag(updated, supported, value, source_text)
        normalizations.append(
            {
                "source": "llm_logged_eval_time",
                "flag": supported,
                "value": value,
                "value_source": value_source,
                "log_path": log_path,
                "reason": reason,
            }
        )
    return updated, normalizations


def _source_eval_log_text(execution_plan: dict[str, Any], source_evaluation_context: dict[str, Any]) -> tuple[str, str]:
    for path in _source_eval_log_candidates(execution_plan, source_evaluation_context):
        if not path.exists() or not path.is_file():
            continue
        return path.read_text(encoding="utf-8", errors="ignore"), path.as_posix()
    return "", ""


def _source_eval_log_candidates(execution_plan: dict[str, Any], source_evaluation_context: dict[str, Any]) -> list[Path]:
    candidates: list[Path] = []
    contexts = [
        source_evaluation_context,
        execution_plan.get("source_evaluation_context") if isinstance(execution_plan, dict) else {},
        execution_plan,
    ]
    keys = (
        "train_log_path",
        "source_train_log_path",
        "source_run_log_path",
        "source_log_path",
        "log_path",
    )
    for context in contexts:
        if not isinstance(context, dict):
            continue
        for key in keys:
            raw = context.get(key)
            if isinstance(raw, str) and raw.strip():
                candidates.append(Path(raw).expanduser())
        raw_paths = context.get("log_paths")
        if isinstance(raw_paths, list):
            candidates.extend(Path(str(item)).expanduser() for item in raw_paths if str(item).strip())
    unique: list[Path] = []
    seen: set[str] = set()
    for path in candidates:
        key = path.as_posix()
        if key in seen:
            continue
        seen.add(key)
        unique.append(path)
    return unique


def _logged_eval_time_prompt(
    *,
    log_text: str,
    log_path: str,
    source_evaluation_context: dict[str, Any],
    train_command: list[str],
    argparse_args: list[dict[str, Any]],
) -> str:
    payload = {
        "task": "从训练/回测日志中判定真实评测时间，并只返回 YAML。不要根据日志时间戳、文件名日期、数据加载日期或普通业务日期作判断。",
        "return_schema": {
            "available": "true if a real evaluation/backtest window is explicitly identifiable",
            "confidence": "high|medium|low",
            "train_eval_start": "YYYYMMDD or empty",
            "train_eval_end": "YYYYMMDD or empty",
            "test_start": "YYYYMMDD or empty",
            "test_end": "YYYYMMDD or empty",
            "reason": "short evidence phrase from the log",
        },
        "selection_rules": [
            "Prefer dates explicitly attached to test_window, Fixed split, train_eval_range, evaluation summary, or backtest summary.",
            "Ignore timestamps at the start of log lines.",
            "Ignore Loaded data times, file-name dates, ordinary data/date columns, and unrelated business dates.",
            "Use confidence=high only when the log clearly marks the chosen dates as evaluation or backtest windows.",
            "If ambiguous, set available=false or confidence=low.",
        ],
        "source_evaluation_context": source_evaluation_context,
        "current_train_command": train_command,
        "argparse_args": argparse_args[:80],
        "log_path": log_path,
        "log_text": _prompt_log_excerpt(log_text),
    }
    return yaml.safe_dump(payload, allow_unicode=True, sort_keys=False)


def _prompt_log_excerpt(text: str, limit: int = 20000) -> str:
    if len(text) <= limit:
        return text
    half = max(limit // 2 - 40, 0)
    return text[:half] + "\n...[log truncated]...\n" + text[-half:]


def _parse_logged_eval_time_response(content: str) -> dict[str, Any]:
    text = clean_llm_yaml_text(content)
    if not text:
        return {}
    try:
        parsed = yaml.safe_load(text) or {}
    except Exception:
        try:
            parsed = json.loads(text)
        except Exception:
            return {}
    if not isinstance(parsed, dict):
        return {}
    return parsed


def _compact_eval_date(value: Any) -> str:
    text = str(value or "").strip()
    if not text:
        return ""
    match = re.fullmatch(r"([0-9]{4})-([0-9]{2})-([0-9]{2})", text)
    if match:
        return "".join(match.groups())
    if re.fullmatch(r"[0-9]{8}", text):
        return text
    return ""


def _normalize_cli_args_for_source(source_text: str, args: list[str]) -> list[str]:
    tokens, _ = _split_cli_shell_tokens([str(item) for item in args])
    tokens, _ = _normalize_argparse_tokens(tokens, source_text)
    return tokens


def _change_cli_args(change: dict[str, Any]) -> list[str]:
    args: list[str] = []
    if change.get("cli_args"):
        args.extend(_split_cli_shell_tokens([str(item) for item in change.get("cli_args", [])])[0])
    elif change.get("cli_flag"):
        flag = str(change["cli_flag"])
        enabled = change.get("enabled", not str(change.get("action", "")).startswith("remove"))
        if enabled:
            args.append(flag)
        elif flag.startswith("--enable-"):
            args.append("--no-enable-" + flag[len("--enable-") :])
        else:
            args.append(flag)
    return args


def _split_cli_shell_tokens(args: list[str]) -> tuple[list[str], list[dict[str, Any]]]:
    tokens: list[str] = []
    records: list[dict[str, Any]] = []
    for raw in args:
        text = str(raw)
        if text.lstrip().startswith("--") and any(ch.isspace() for ch in text):
            try:
                parts = shlex.split(text)
            except ValueError:
                parts = [text]
            if len(parts) > 1 and parts[0].startswith("--"):
                tokens.extend(parts)
                records.append({"kind": "split_shell_cli_token", "original": text, "replacement": parts})
                continue
        tokens.append(text)
    return tokens, records


def _normalize_argparse_tokens(tokens: list[str], source_text: str) -> tuple[list[str], list[dict[str, Any]]]:
    spec = _argparse_command_spec(source_text)
    flags = spec.get("flags", {})
    if not flags:
        return tokens, []
    normalized: list[str] = []
    records: list[dict[str, Any]] = []
    index = 0
    while index < len(tokens):
        item = str(tokens[index])
        if not item.startswith("--") or item == "--":
            normalized.append(item)
            index += 1
            continue
        flag, inline_value = _split_inline_flag_value(item)
        supported = _argparse_supported_flag(flags, flag)
        arg_spec = flags.get(supported or "")
        output_flag = supported or flag
        if supported and supported != flag:
            records.append({"kind": "normalize_flag_spelling", "original": flag, "replacement": supported})
        if inline_value is not None:
            values = _normalize_argparse_value_tokens(output_flag, inline_value, arg_spec, records)
            normalized.extend([output_flag, *values])
            index += 1
            continue
        normalized.append(output_flag)
        index += 1
        if not arg_spec or not _argparse_spec_accepts_values(arg_spec):
            continue
        if not _argparse_spec_accepts_multiple_values(arg_spec):
            continue
        while index < len(tokens) and not str(tokens[index]).startswith("--"):
            value = str(tokens[index])
            normalized.extend(_normalize_argparse_value_tokens(output_flag, value, arg_spec, records))
            index += 1
    return normalized, records


def _normalize_argparse_value_tokens(
    flag: str,
    value: str,
    arg_spec: dict[str, Any] | None,
    records: list[dict[str, Any]],
) -> list[str]:
    if not arg_spec:
        return [value]
    value_type = str(arg_spec.get("type") or "")
    if value_type not in {"int", "builtins.int", "float", "builtins.float"}:
        return [value]
    if not _argparse_spec_accepts_multiple_values(arg_spec):
        return [value]
    clean = _strip_cli_value_quotes(value)
    if "," not in clean:
        return [clean]
    parts = [part.strip() for part in clean.split(",")]
    if not parts or any(part == "" for part in parts):
        return [clean]
    records.append({"kind": "split_csv_numeric_list", "flag": flag, "original": value, "replacement": parts})
    return parts


def _split_inline_flag_value(item: str) -> tuple[str, str | None]:
    if item.startswith("--") and "=" in item:
        flag, value = item.split("=", 1)
        return flag, value
    return item, None


def _strip_cli_value_quotes(value: str) -> str:
    text = str(value).strip()
    if len(text) >= 2 and text[0] == text[-1] and text[0] in {"'", '"'}:
        return text[1:-1]
    return text


def _argparse_supported_flag(flags: dict[str, dict[str, Any]], flag: str) -> str | None:
    if flag in flags:
        return flag
    if flag.startswith("--no-"):
        for variant in _flag_variants("--" + flag[len("--no-") :]):
            negative = "--no-" + variant[2:]
            if negative in flags:
                return negative
    for variant in _flag_variants(flag):
        if variant in flags:
            return variant
    return None


def _argparse_spec_accepts_values(arg_spec: dict[str, Any]) -> bool:
    action = str(arg_spec.get("action") or "store")
    return action.split(".")[-1] not in {
        "store_true",
        "store_false",
        "store_const",
        "append_const",
        "count",
        "help",
        "version",
        "BooleanOptionalAction",
    }


def _argparse_spec_accepts_multiple_values(arg_spec: dict[str, Any]) -> bool:
    nargs = arg_spec.get("nargs")
    if isinstance(nargs, int):
        return nargs > 1
    return str(nargs) in {"*", "+"}


def _apply_preserved_train_args(command: list[str], source_text: str, execution_plan: dict[str, Any]) -> list[str]:
    source_context = _source_evaluation_context(execution_plan)
    preserved = source_context.get("preserved_train_args") if isinstance(source_context, dict) else {}
    if not isinstance(preserved, dict) or not preserved:
        return command
    merged = list(command)
    for raw_flag, raw_value in preserved.items():
        flag = str(raw_flag)
        value = str(raw_value)
        supported = _supported_arg_spelling(source_text, flag)
        if supported is None:
            continue
        merged = _replace_or_append_command_flag(merged, supported, value, source_text)
    return merged


def _replace_or_append_command_flag(command: list[str], flag: str, value: str, source_text: str) -> list[str]:
    merged: list[str] = []
    index = 0
    replaced = False
    variants = set(_flag_variants(flag))
    while index < len(command):
        item = str(command[index])
        supported_item = _supported_arg_spelling(source_text, item) if item.startswith("--") else None
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


def _apply_declared_resource_cli_args(
    command: list[str],
    source_text: str,
    resource_context: dict[str, Any] | None,
) -> list[str]:
    cli_resource_args = _resource_cli_arg_paths(resource_context)
    if not command or not cli_resource_args:
        return command
    resource_by_flag: dict[str, str] = {}
    supported_flags: dict[str, str] = {}
    for raw_flag, target_path in cli_resource_args.items():
        supported = _supported_arg_spelling(source_text, raw_flag) or raw_flag
        supported_flags[raw_flag] = supported
        for variant in _flag_variants(raw_flag) + _flag_variants(supported):
            resource_by_flag[variant] = target_path

    merged: list[str] = []
    seen_resource_flags: set[str] = set()
    index = 0
    while index < len(command):
        item = str(command[index])
        supported_item = _supported_arg_spelling(source_text, item) if item.startswith("--") else None
        resource_path = resource_by_flag.get(item) or resource_by_flag.get(supported_item or "")
        if resource_path is None:
            merged.append(item)
            index += 1
            continue
        merged.append(item)
        merged.append(resource_path)
        seen_resource_flags.update(_flag_variants(item))
        if supported_item:
            seen_resource_flags.update(_flag_variants(supported_item))
        index += 2 if index + 1 < len(command) and not str(command[index + 1]).startswith("--") else 1

    for raw_flag, target_path in cli_resource_args.items():
        supported = supported_flags.get(raw_flag) or _supported_arg_spelling(source_text, raw_flag)
        if not supported:
            continue
        if any(variant in seen_resource_flags for variant in _flag_variants(raw_flag) + _flag_variants(supported)):
            continue
        merged.extend([supported, target_path])
    return merged


def _resource_cli_arg_paths(resource_context: dict[str, Any] | None) -> dict[str, str]:
    if not isinstance(resource_context, dict):
        return {}
    cli_resource_args = resource_context.get("cli_resource_args", {})
    if not isinstance(cli_resource_args, dict):
        return {}
    resources: dict[str, str] = {}
    for raw_flag, raw_path in cli_resource_args.items():
        flag = str(raw_flag or "").strip()
        path = str(raw_path or "").strip()
        if flag.startswith("--") and path:
            resources[flag] = path
    return resources


def _merge_plan_cli_args(command: list[str], plan: dict[str, Any], source_text: str) -> list[str]:
    plan_args = _filter_supported_args(source_text, _feature_cli_args(plan))
    if not plan_args:
        return command
    plan_flags = {item for item in plan_args if item.startswith("--")}
    if not plan_flags:
        return command
    merged: list[str] = []
    index = 0
    while index < len(command):
        item = command[index]
        supported = _supported_arg_spelling(source_text, item) if item.startswith("--") else None
        if supported in plan_flags:
            index += 1
            while index < len(command) and not str(command[index]).startswith("--"):
                index += 1
            continue
        merged.append(item)
        index += 1
    merged.extend(plan_args)
    return merged


def validate_train_command_argparse_choices(train_command: list[Any], source_text: str) -> None:
    error = _train_command_argparse_choice_error([str(item) for item in train_command], source_text)
    if error:
        raise ValueError(error)


def _train_command_validation_error(command: list[str], wrapper: Path, resource_context: dict[str, Any] | None = None) -> str:
    if not command:
        return "train_command is empty"
    if len(command) < 2:
        return "train_command must execute the trial train.py entrypoint"
    wrapper_path = wrapper.resolve()
    entrypoint = Path(command[1]).resolve()
    if entrypoint != wrapper_path and command[1] not in {"{train_py}", "train.py", "./train.py"}:
        return "train_command must execute the trial train.py as its only Python entrypoint"
    for item in command[2:]:
        if str(item).endswith(".py"):
            return "train_command must not pass dependency Python files as extra positional scripts; train.py imports them"
    source_text = wrapper.read_text(encoding="utf-8", errors="ignore") if wrapper.exists() else ""
    argparse_error = _train_command_argparse_contract_error(command, source_text)
    if argparse_error:
        return argparse_error
    return _train_command_resource_path_error(command, wrapper, source_text, resource_context)


def _train_command_resource_path_error(
    command: list[str],
    wrapper: Path,
    source_text: str,
    resource_context: dict[str, Any] | None,
) -> str:
    cli_resource_args = _resource_cli_arg_paths(resource_context)
    if not cli_resource_args:
        return ""
    command_values = _command_flag_values(command, source_text)
    for raw_flag, expected in cli_resource_args.items():
        supported = _supported_arg_spelling(source_text, raw_flag) or raw_flag
        value = ""
        for variant in _flag_variants(raw_flag) + _flag_variants(supported):
            if variant in command_values:
                value = command_values[variant]
                break
        if not value:
            return f"train_command missing declared resource argument {supported}; expected copied trial data path {expected}"
        actual_path = Path(value)
        actual_resolved = actual_path if actual_path.is_absolute() else (wrapper.parent / actual_path)
        expected_resolved = Path(expected)
        if not actual_resolved.exists():
            return (
                f"train_command declared resource argument {supported} points to missing path {value!r}; "
                f"expected copied trial data path {expected}"
            )
        if expected_resolved.exists() and actual_resolved.resolve() != expected_resolved.resolve():
            return (
                f"train_command declared resource argument {supported} should use copied trial data path "
                f"{expected}; got {value!r}"
            )
    return ""


def _command_flag_values(command: list[str], source_text: str) -> dict[str, str]:
    values: dict[str, str] = {}
    index = 0
    while index < len(command):
        item = str(command[index])
        if not item.startswith("--"):
            index += 1
            continue
        value_index = index + 1
        if value_index >= len(command) or str(command[value_index]).startswith("--"):
            index += 1
            continue
        value = str(command[value_index])
        values[item] = value
        supported = _supported_arg_spelling(source_text, item)
        if supported:
            values[supported] = value
            for variant in _flag_variants(supported):
                values[variant] = value
        index = value_index + 1
    return values


def _train_command_argparse_contract_error(command: list[str], source_text: str) -> str:
    argparse_spec = _argparse_command_spec(source_text)
    flags = argparse_spec.get("flags", {})
    positionals = argparse_spec.get("positionals", [])
    if not flags and not positionals:
        return ""
    allow_unknown = bool(argparse_spec.get("allow_unknown_args"))
    index = 2 if len(command) >= 2 and not str(command[0]).startswith("--") else 0
    positional_index = 0
    while index < len(command):
        item = str(command[index])
        if item == "--":
            break
        if not item.startswith("--"):
            if allow_unknown:
                index += 1
                continue
            if positional_index >= len(positionals):
                return f"train_command passes unexpected positional argument {item!r}; generated train.py does not define positional arguments"
            arg_spec = positionals[positional_index]
            values, next_index, arity_error = _collect_positional_values(command, index, arg_spec)
            if arity_error:
                return arity_error
            for value in values:
                value_error = _argparse_value_contract_error(str(arg_spec.get("dest") or "positional"), value, arg_spec)
                if value_error:
                    return value_error
            positional_index += 1
            index = next_index
            continue
        flag, inline_value = _split_inline_flag_value(item)
        supported = _argparse_supported_flag(flags, flag)
        if supported is None:
            if allow_unknown:
                index += 1
                if inline_value is None and index < len(command) and not str(command[index]).startswith("--"):
                    index += 1
                continue
            return f"train_command passes unsupported argument {flag}; generated train.py does not define it"
        arg_spec = flags[supported]
        if not _argparse_spec_accepts_values(arg_spec):
            if inline_value is not None:
                return f"train_command passes unexpected value {inline_value!r} to flag argument {supported}"
            index += 1
            continue
        values, next_index, arity_error = _collect_argparse_values(command, index, inline_value, supported, arg_spec)
        if arity_error:
            return arity_error
        for value in values:
            value_error = _argparse_value_contract_error(supported, value, arg_spec)
            if value_error:
                return value_error
        index = next_index
    return ""


def _collect_positional_values(
    command: list[str],
    index: int,
    arg_spec: dict[str, Any],
) -> tuple[list[str], int, str]:
    name = str(arg_spec.get("dest") or "positional")
    nargs = arg_spec.get("nargs")
    if nargs is None:
        return [str(command[index])], index + 1, ""
    if nargs == "?":
        if index < len(command) and not str(command[index]).startswith("--"):
            return [str(command[index])], index + 1, ""
        return [], index, ""
    if nargs in {"*", "+"}:
        values: list[str] = []
        value_index = index
        while value_index < len(command) and not str(command[value_index]).startswith("--"):
            values.append(str(command[value_index]))
            value_index += 1
        if nargs == "+" and not values:
            return [], value_index, f"train_command missing one or more values for positional argument {name}"
        return values, value_index, ""
    if isinstance(nargs, int):
        values = []
        value_index = index
        while value_index < len(command) and len(values) < nargs and not str(command[value_index]).startswith("--"):
            values.append(str(command[value_index]))
            value_index += 1
        if len(values) != nargs:
            return values, value_index, f"train_command expected {nargs} value(s) for positional argument {name}; got {len(values)}"
        return values, value_index, ""
    return [str(command[index])], index + 1, ""


def _collect_argparse_values(
    command: list[str],
    index: int,
    inline_value: str | None,
    flag: str,
    arg_spec: dict[str, Any],
) -> tuple[list[str], int, str]:
    if inline_value is not None:
        return [inline_value], index + 1, ""
    nargs = arg_spec.get("nargs")
    value_index = index + 1
    if nargs is None:
        if value_index >= len(command) or str(command[value_index]).startswith("--"):
            return [], value_index, f"train_command missing value for {flag}"
        return [str(command[value_index])], value_index + 1, ""
    if nargs == "?":
        if value_index < len(command) and not str(command[value_index]).startswith("--"):
            return [str(command[value_index])], value_index + 1, ""
        return [], value_index, ""
    if nargs in {"*", "+"}:
        values: list[str] = []
        while value_index < len(command) and not str(command[value_index]).startswith("--"):
            values.append(str(command[value_index]))
            value_index += 1
        if nargs == "+" and not values:
            return [], value_index, f"train_command missing one or more values for {flag}"
        return values, value_index, ""
    if isinstance(nargs, int):
        values = []
        while value_index < len(command) and len(values) < nargs and not str(command[value_index]).startswith("--"):
            values.append(str(command[value_index]))
            value_index += 1
        if len(values) != nargs:
            return values, value_index, f"train_command expected {nargs} value(s) for {flag}; got {len(values)}"
        return values, value_index, ""
    if value_index >= len(command) or str(command[value_index]).startswith("--"):
        return [], value_index, f"train_command missing value for {flag}"
    return [str(command[value_index])], value_index + 1, ""


def _argparse_value_contract_error(flag: str, value: str, arg_spec: dict[str, Any]) -> str:
    clean = _strip_cli_value_quotes(value)
    value_type = str(arg_spec.get("type") or "")
    converted: Any = clean
    if value_type in {"int", "builtins.int"}:
        try:
            converted = int(clean)
        except ValueError:
            return f"train_command passes invalid int value {value!r} to {flag}"
    elif value_type in {"float", "builtins.float"}:
        try:
            converted = float(clean)
        except ValueError:
            return f"train_command passes invalid float value {value!r} to {flag}"
    choices = {str(item) for item in arg_spec.get("choices", set()) or set()}
    if choices and clean not in choices and str(converted) not in choices:
        return f"train_command passes invalid value {value!r} to {flag}; allowed choices are {sorted(choices)}"
    return ""


def _train_command_argparse_choice_error(command: list[str], source_text: str) -> str:
    choices = _argparse_choice_flags(source_text)
    if not choices:
        return ""
    index = 0
    while index < len(command):
        item = str(command[index])
        if item not in choices:
            index += 1
            continue
        allowed = choices[item]
        value_index = index + 1
        if value_index >= len(command) or str(command[value_index]).startswith("--"):
            index += 1
            continue
        value = str(command[value_index])
        if value not in allowed:
            return (
                f"train_command passes invalid value {value!r} to {item}; "
                f"allowed choices are {sorted(allowed)}"
            )
        index = value_index + 1
    return ""


def _argparse_command_spec(source_text: str) -> dict[str, Any]:
    try:
        tree = ast.parse(source_text)
    except SyntaxError:
        return {"flags": {}, "positionals": [], "allow_unknown_args": False}
    flags: dict[str, dict[str, Any]] = {}
    positionals: list[dict[str, Any]] = []
    uses_parse_args = False
    uses_parse_known_args = False
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        call_name = _call_name(node.func).split(".")[-1]
        if call_name == "parse_args":
            uses_parse_args = True
        elif call_name == "parse_known_args":
            uses_parse_known_args = True
        if call_name != "add_argument":
            continue
        arg_strings = [
            str(arg.value)
            for arg in node.args
            if isinstance(arg, ast.Constant) and isinstance(arg.value, str)
        ]
        arg_flags = [item for item in arg_strings if item.startswith("--")]
        positional_args = [item for item in arg_strings if item and not item.startswith("-")]
        if not arg_flags:
            if positional_args:
                positionals.append(
                    {
                        "flags": [],
                        "dest": _argparse_positional_dest_from_call(node, positional_args[0]),
                        "type": _argparse_type_from_call(node),
                        "nargs": _argparse_nargs_from_call(node),
                        "action": _argparse_action_from_call(node),
                        "choices": _literal_choices_from_call(node),
                        "default": _argparse_default_from_call(node),
                        "line": getattr(node, "lineno", 0),
                    }
                )
            continue
        action = _argparse_action_from_call(node)
        spec = {
            "flags": arg_flags,
            "dest": _argparse_dest_from_call(node, arg_flags),
            "type": _argparse_type_from_call(node),
            "nargs": _argparse_nargs_from_call(node),
            "action": action,
            "choices": _literal_choices_from_call(node),
            "default": _argparse_default_from_call(node),
            "line": getattr(node, "lineno", 0),
        }
        for flag in _argparse_aliases_for_action(arg_flags, action):
            flags[flag] = spec
    return {
        "flags": flags,
        "positionals": positionals,
        "allow_unknown_args": uses_parse_known_args and not uses_parse_args,
    }


def _argparse_aliases_for_action(arg_flags: list[str], action: str) -> list[str]:
    aliases = list(arg_flags)
    if action.split(".")[-1] == "BooleanOptionalAction":
        for flag in arg_flags:
            if flag.startswith("--") and not flag.startswith("--no-"):
                aliases.append("--no-" + flag[2:])
    return list(dict.fromkeys(aliases))


def _argparse_dest_from_call(node: ast.Call, arg_flags: list[str]) -> str:
    dest_node = _argparse_keyword_node(node, "dest")
    if isinstance(dest_node, ast.Constant) and isinstance(dest_node.value, str):
        return dest_node.value
    flag = next((item for item in arg_flags if not item.startswith("--no-")), arg_flags[0])
    body = flag[5:] if flag.startswith("--no-") else flag[2:]
    return body.replace("-", "_")


def _argparse_positional_dest_from_call(node: ast.Call, fallback: str) -> str:
    dest_node = _argparse_keyword_node(node, "dest")
    if isinstance(dest_node, ast.Constant) and isinstance(dest_node.value, str):
        return dest_node.value
    return fallback.replace("-", "_")


def _argparse_type_from_call(node: ast.Call) -> str:
    value = _argparse_keyword_node(node, "type")
    if value is None:
        return ""
    if isinstance(value, ast.Name):
        return value.id
    if isinstance(value, ast.Attribute):
        return _call_name(value)
    if isinstance(value, ast.Constant) and value.value is not None:
        return str(value.value)
    if isinstance(value, ast.Call):
        return _call_name(value.func)
    return _node_snippet(value)


def _argparse_action_from_call(node: ast.Call) -> str:
    value = _argparse_keyword_node(node, "action")
    if value is None:
        return "store"
    if isinstance(value, ast.Constant) and value.value is not None:
        return str(value.value)
    if isinstance(value, (ast.Name, ast.Attribute)):
        return _call_name(value)
    return _node_snippet(value)


def _argparse_nargs_from_call(node: ast.Call) -> str | int | None:
    value = _argparse_keyword_node(node, "nargs")
    if value is None:
        return None
    try:
        parsed = ast.literal_eval(value)
    except Exception:
        return _node_snippet(value)
    if isinstance(parsed, (str, int)):
        return parsed
    return str(parsed)


def _argparse_default_from_call(node: ast.Call) -> Any:
    value = _argparse_keyword_node(node, "default")
    if value is None:
        return None
    try:
        return ast.literal_eval(value)
    except Exception:
        return _node_snippet(value)


def _argparse_keyword_node(node: ast.Call, name: str) -> ast.AST | None:
    for keyword in node.keywords:
        if keyword.arg == name:
            return keyword.value
    return None


def _argparse_choice_flags(source_text: str) -> dict[str, set[str]]:
    try:
        tree = ast.parse(source_text)
    except SyntaxError:
        return {}
    flags: dict[str, set[str]] = {}
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        if _call_name(node.func) not in {"add_argument", "parser.add_argument", "argparse.ArgumentParser.add_argument"}:
            continue
        arg_flags = [
            str(arg.value)
            for arg in node.args
            if isinstance(arg, ast.Constant) and isinstance(arg.value, str) and str(arg.value).startswith("--")
        ]
        if not arg_flags:
            continue
        choices = _literal_choices_from_call(node)
        if not choices:
            continue
        for flag in arg_flags:
            flags[flag] = choices
    return flags


def _literal_choices_from_call(node: ast.Call) -> set[str]:
    for keyword in node.keywords:
        if keyword.arg != "choices":
            continue
        value = keyword.value
        if isinstance(value, (ast.List, ast.Tuple, ast.Set)):
            choices: set[str] = set()
            for item in value.elts:
                if isinstance(item, ast.Constant) and item.value is not None:
                    choices.add(str(item.value))
            return choices
    return set()



def _supported_flag_values(source_text: str, pairs: list[tuple[str, str]], *, required: bool = False) -> list[str]:
    args: list[str] = []
    for flag, value in pairs:
        supported_flag = _supported_arg_spelling(source_text, flag)
        if required or supported_flag is not None:
            args.extend([supported_flag or flag, value])
    return args


def _filter_supported_args(source_text: str, args: list[str]) -> list[str]:
    args = _normalize_cli_args_for_source(source_text, args)
    argparse_spec = _argparse_command_spec(source_text)
    flags = argparse_spec.get("flags", {})
    filtered: list[str] = []
    index = 0
    while index < len(args):
        item = args[index]
        if not item.startswith("--"):
            index += 1
            continue
        supported_item = _supported_arg_spelling(source_text, item)
        if supported_item is None:
            index += 2 if index + 1 < len(args) and not args[index + 1].startswith("--") else 1
            continue
        flag, inline_value = _split_inline_flag_value(item)
        supported_flag = _argparse_supported_flag(flags, flag) or supported_item
        arg_spec = flags.get(supported_flag, {})
        filtered.append(supported_item)
        index += 1
        if inline_value is not None:
            filtered.append(inline_value)
            continue
        if not _argparse_spec_accepts_values(arg_spec):
            while index < len(args) and not args[index].startswith("--"):
                filtered.append(args[index])
                index += 1
            continue
        values, next_index, _ = _collect_argparse_values(args, index - 1, inline_value, supported_item, arg_spec)
        filtered.extend(values)
        index = next_index
        while index < len(args) and not args[index].startswith("--"):
            filtered.append(args[index])
            index += 1
    return filtered


def _arg_supported(source_text: str, flag: str) -> bool:
    return _supported_arg_spelling(source_text, flag) is not None


def _supported_arg_spelling(source_text: str, flag: str) -> str | None:
    if flag.startswith("--no-"):
        for variant in _flag_variants("--" + flag[len("--no-") :]):
            if _add_argument_supports(source_text, variant, require_boolean_optional=True):
                return "--no-" + variant[2:]
        return None
    for variant in _flag_variants(flag):
        if _add_argument_supports(source_text, variant):
            return variant
    return None


def _flag_variants(flag: str) -> list[str]:
    if not flag.startswith("--"):
        return [flag]
    body = flag[2:]
    variants = [
        flag,
        "--" + body.replace("-", "_"),
        "--" + body.replace("_", "-"),
    ]
    return list(dict.fromkeys(variants))


def _add_argument_supports(source_text: str, flag: str, *, require_boolean_optional: bool = False) -> bool:
    pattern = re.compile(r"add_argument\((?P<body>.*?)\)", re.DOTALL)
    quoted = {f'"{flag}"', f"'{flag}'"}
    for match in pattern.finditer(_strip_comment_lines(source_text)):
        body = match.group("body")
        if not any(item in body for item in quoted):
            continue
        if require_boolean_optional and "BooleanOptionalAction" not in body:
            continue
        return True
    return False


def _read_input_manifest(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _copy_data_file(source: Path, data_dir: Path) -> Path:
    target = data_dir / source.name
    if source.exists() and source.resolve() != target.resolve():
        shutil.copy2(source, target)
    return target


def _prepare_trial_data_resources(
    experiment: Path,
    data_dir: Path,
    source_text: str,
    plan: dict[str, Any],
    source_path: Path,
    execution_plan: dict[str, Any],
) -> dict[str, Any]:
    copied_files: dict[str, str] = {}
    cli_resource_args: dict[str, str] = {}
    path_rewrites: dict[str, str] = {}
    missing_input_files: list[str] = []
    for item in execution_plan.get("required_data_files", []) or []:
        if isinstance(item, str):
            raw_path = item
            cli_arg = None
        elif isinstance(item, dict):
            raw_path = str(item.get("path") or "")
            cli_arg = item.get("cli_arg")
        else:
            continue
        if not raw_path:
            continue
        source = Path(raw_path)
        if not source.is_absolute():
            source = experiment / source
        if not source.exists() or not source.is_file():
            missing_input_files.append(f"{cli_arg or 'data'}: {source.as_posix()}")
            continue
        target = _copy_data_file(source, data_dir)
        copied_files[source.as_posix()] = target.as_posix()
        if cli_arg:
            cli_resource_args[str(cli_arg)] = target.as_posix()
        path_rewrites[source.as_posix()] = target.as_posix()
    for old, new in (execution_plan.get("path_rewrites") or {}).items():
        if old and new:
            path_rewrites[str(old)] = str(new)
    manifest = {
        "original_experiment_dir": experiment.as_posix(),
        "source_entrypoint": source_path.as_posix(),
        "trial_data_dir": data_dir.as_posix(),
        "copied_files": copied_files,
        "cli_resource_args": cli_resource_args,
        "path_rewrites": path_rewrites,
        "missing_input_files": missing_input_files,
        "feature_changes": plan.get("changes", []),
        "note": "Agent2 LLM declares training resources copied to trial/data; original experiment directory is read-only.",
    }
    (data_dir / "input_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return manifest


def _strip_comment_lines(source_text: str) -> str:
    return "\n".join(line for line in source_text.splitlines() if not line.lstrip().startswith("#"))


def _copy_declared_python_dependencies(experiment: Path, trial: Path, source_path: Path, execution_plan: dict[str, Any]) -> None:
    dependency_files = execution_plan.get("python_dependencies") or execution_plan.get("dependency_files") or []
    for raw in dependency_files:
        dep = Path(str(raw))
        if not dep.is_absolute():
            dep = experiment / dep
        dep = dep.resolve()
        if dep == source_path.resolve():
            continue
        if dep.suffix != ".py" or not dep.exists() or not dep.is_file():
            raise FileNotFoundError(f"declared Python dependency not found: {dep}")
        shutil.copy2(dep, trial / dep.name)


def _trial_dirs(trial: Path) -> dict[str, Path]:
    dirs = {
        "trial": trial,
        "code": trial / "code",
        "data": trial / "data",
        "outputs": trial / "outputs",
        "real_outputs": trial / "outputs" / "real_outputs",
        "logs": trial / "logs",
        "standardized": trial / "standardized",
    }
    for path in dirs.values():
        path.mkdir(parents=True, exist_ok=True)
    return dirs


def _audit_feature_application(
    plan: dict[str, Any],
    code_dir: Path,
    wrapper: Path,
    train_command: list[str] | None = None,
    modified_functions: list[str] | None = None,
) -> dict[str, Any]:
    audit_path = code_dir / "agent2_feature_application_audit.yaml"
    changes = [change for change in plan.get("changes", []) if isinstance(change, dict)]
    code_files = sorted(path for path in code_dir.glob("*.py") if path.is_file())
    modification = _read_agent2_code_modification(code_dir)
    if wrapper.exists() and wrapper not in code_files:
        code_files.insert(0, wrapper)
    if not changes:
        audit = {
            "success": True,
            "reason": "no feature changes requested",
            "features": [],
            "modified_files": modification.get("modified_files", []),
            "feature_evidence_files": [],
            "entrypoint_import_chain_checked": False,
            "audit_path": audit_path.as_posix(),
        }
        audit_path.write_text(yaml.safe_dump(audit, allow_unicode=True, sort_keys=False), encoding="utf-8")
        return audit

    effective_modified_functions = (
        [str(item) for item in modified_functions]
        if modified_functions is not None
        else [str(item) for item in modification.get("modified_functions", []) or []]
    )
    features = []
    for change in changes:
        feature = _audit_single_feature_application(change, code_files, wrapper, train_command=train_command)
        _attach_entrypoint_reachability(feature, wrapper, code_files)
        _attach_feature_activation_checks(change, feature, wrapper, code_files, train_command)
        _attach_implementation_assertions(change, feature, code_files, effective_modified_functions)
        features.append(feature)
    success = all(bool(item.get("applied")) for item in features)
    modified_files = modification.get("modified_files", []) or []
    cli_only_variation = modification.get("accepted_schema") in {"train_command_args", "existing_code_feature"} or modification.get("source") == "plan_cli_args"
    if not modified_files and not cli_only_variation:
        success = False
        for feature in features:
            if feature.get("applied"):
                feature["stale_evidence_warning"] = (
                    "evidence exists in copied code, but Agent2 produced no valid modified trial files in this run"
                )
            feature["applied"] = False
            feature["failure_reason"] = "Agent2 produced no valid modified trial files for this requested feature"
    evidence_files = sorted(
        {
            str(evidence.get("path"))
            for feature in features
            for evidence in feature.get("evidence", [])
            if evidence.get("path")
        }
    )
    audit = {
        "success": success,
        "reason": "all requested feature changes have executable evidence"
        if success
        else (
            "Agent2 produced no valid modified trial files for requested feature changes"
            if not modified_files and not cli_only_variation
            else "one or more requested feature changes have no executable evidence"
        ),
        "features": features,
        "modified_files": modified_files,
        "modified_functions": effective_modified_functions,
        "feature_evidence_files": evidence_files,
        "entrypoint_import_chain_checked": True,
        "audit_path": audit_path.as_posix(),
    }
    audit_path.write_text(yaml.safe_dump(audit, allow_unicode=True, sort_keys=False), encoding="utf-8")
    return audit


def _read_agent2_code_modification(code_dir: Path) -> dict[str, Any]:
    path = code_dir / "agent2_code_modification.yaml"
    if not path.exists():
        return {}
    try:
        value = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except Exception:
        return {}
    return value if isinstance(value, dict) else {}


def _code_modification_status_fields(code_dir: Path) -> dict[str, Any]:
    modification = _read_agent2_code_modification(code_dir)
    return {
        "agent2_code_modification_path": (code_dir / "agent2_code_modification.yaml").as_posix(),
        "agent2_modified_files": modification.get("modified_files", []),
        "agent2_modified_functions": modification.get("modified_functions", []),
        "agent2_rejected_files": modification.get("rejected_files", []),
        "agent2_fallback_used": modification.get("fallback_used"),
        "agent2_code_generation_success": modification.get("code_generation_success"),
        "agent2_feature_code_success": modification.get("feature_code_success"),
        "agent2_code_generation_failure_reason": modification.get("code_generation_failure_reason"),
        "agent2_failure_categories": modification.get("failure_categories", []),
        "agent2_normalized_rejections": modification.get("normalized_rejections", []),
        "agent2_source_locator_path": (code_dir / "agent2_source_locator.yaml").as_posix(),
        "agent2_train_command_repair_success": modification.get("train_command_repair_success"),
        "agent2_train_command_repair_error": modification.get("train_command_repair_error", ""),
        "agent2_train_command_contract_error_before_repair": modification.get("train_command_contract_error_before_repair", ""),
        "agent2_repaired_train_command": modification.get("repaired_train_command", []),
        "agent2_feature_resolution": modification.get("feature_resolution", ""),
        "agent2_command_contract_status": modification.get("command_contract_status", ""),
    }


def _audit_single_feature_application(
    change: dict[str, Any],
    code_files: list[Path],
    wrapper: Path,
    train_command: list[str] | None = None,
) -> dict[str, Any]:
    feature_name = str(change.get("feature_name") or "").strip()
    wrapper_text = wrapper.read_text(encoding="utf-8", errors="ignore") if wrapper.exists() else ""
    declared_cli_args = _normalize_cli_args_for_source(wrapper_text, _change_cli_args(change))
    if change.get("cli_flag"):
        flag = str(change["cli_flag"])
        if flag not in declared_cli_args:
            declared_cli_args.append(flag)
    reconciliation = _feature_cli_reconciliation(declared_cli_args, wrapper_text, train_command)
    cli_args = [str(item) for item in reconciliation["effective_cli_args"]]
    cli_arg_flags = [str(item) for item in cli_args if str(item).startswith("--")]
    cli_arg_flags = list(dict.fromkeys(cli_arg_flags))
    cli_dests = _feature_cli_dest_names(cli_arg_flags)
    parsed_flags = [flag for flag in (_supported_arg_spelling(wrapper_text, arg) for arg in cli_arg_flags) if flag]
    unparsed_cli_args = [arg for arg in cli_arg_flags if _supported_arg_spelling(wrapper_text, arg) is None]
    evidence: list[dict[str, Any]] = []
    consumed_dests: set[str] = set()

    if parsed_flags and cli_dests:
        for path in code_files:
            for item in _arg_usage_evidence(path, cli_dests):
                item["kind"] = "cli_arg_consumed"
                item["parsed_flags"] = parsed_flags
                evidence.append(item)
                snippet = str(item.get("snippet") or "")
                for dest in cli_dests:
                    if dest in snippet:
                        consumed_dests.add(dest)

    tokens = _change_feature_tokens(change)
    for path in code_files:
        evidence.extend(_feature_token_evidence(path, tokens))
        evidence.extend(_declared_feature_column_evidence(path, _declared_feature_columns(change)))
        evidence.extend(_structured_feature_evidence(change, path))

    evidence = _dedupe_evidence(evidence)
    unconsumed_cli_dests = [dest for dest in cli_dests if dest not in consumed_dests]
    cli_contract_success = not cli_arg_flags or (not unparsed_cli_args and not unconsumed_cli_dests)
    non_cli_evidence = [
        item
        for item in evidence
        if isinstance(item, dict) and item.get("kind") != "cli_arg_consumed"
    ]
    ignored_intent_cli_args = [str(item) for item in reconciliation["ignored_intent_cli_args"]]
    has_required_evidence = bool(non_cli_evidence) if ignored_intent_cli_args else bool(evidence)
    applied = has_required_evidence and cli_contract_success
    failure_reason = ""
    if not applied:
        if unparsed_cli_args:
            failure_reason = "feature CLI args are not parsed by the generated train.py: " + ", ".join(unparsed_cli_args)
        elif unconsumed_cli_dests:
            failure_reason = "feature CLI args are parsed but not consumed outside argument parsing: " + ", ".join(unconsumed_cli_dests)
        elif ignored_intent_cli_args and not non_cli_evidence:
            failure_reason = (
                "declared feature CLI args are absent from train_command and no non-CLI executable feature evidence was found"
            )
        else:
            failure_reason = "no non-metadata code path constructs or consumes this feature"

    return {
        "feature_name": feature_name,
        "feature_slug": change.get("feature_slug"),
        "audit_tokens": change.get("audit_tokens", []),
        "action": change.get("action"),
        "cli_args": cli_args,
        "declared_cli_args": declared_cli_args,
        "effective_cli_args": cli_args,
        "ignored_intent_cli_args": ignored_intent_cli_args,
        "cli_reconciliation_notes": reconciliation["notes"],
        "cli_dest_names": cli_dests,
        "parsed_flags": parsed_flags,
        "cli_contract_success": cli_contract_success,
        "unparsed_cli_args": unparsed_cli_args,
        "unconsumed_cli_dests": unconsumed_cli_dests,
        "declared_feature_columns": _declared_feature_columns(change),
        "applied": applied,
        "evidence": evidence,
        "failure_reason": failure_reason,
    }


def _attach_entrypoint_reachability(feature: dict[str, Any], wrapper: Path, code_files: list[Path]) -> None:
    evidence = [item for item in feature.get("evidence", []) or [] if isinstance(item, dict)]
    reachability = _entrypoint_reachability_for_evidence(evidence, wrapper, code_files)
    feature["entrypoint_import_chain"] = reachability
    if not evidence:
        feature["applied"] = False
        return
    cli_evidence = [item for item in evidence if item.get("kind") == "cli_arg_consumed"]
    if feature.get("cli_args"):
        cli_reachability = _entrypoint_reachability_for_evidence(cli_evidence, wrapper, code_files)
        feature["cli_consumption_reachability"] = cli_reachability
        if not feature.get("cli_contract_success", False):
            feature["applied"] = False
            return
        if not cli_reachability.get("reachable", False):
            feature["applied"] = False
            feature["failure_reason"] = "feature CLI args are parsed but not consumed in a reachable train.py path"
            return
    if not reachability.get("reachable", False):
        feature["applied"] = False
        feature["failure_reason"] = "feature evidence exists but is not reachable from train.py entrypoint import/call chain"


def _attach_feature_activation_checks(
    change: dict[str, Any],
    feature: dict[str, Any],
    wrapper: Path,
    code_files: list[Path],
    train_command: list[str] | None,
) -> None:
    declared_columns = _declared_feature_column_application(change, feature, wrapper, code_files, train_command)
    if declared_columns:
        feature["declared_feature_column_application"] = declared_columns
        if not declared_columns.get("success", False):
            feature["applied"] = False
            feature["failure_category"] = (
                "inactive_feature_gate"
                if _declared_feature_columns_blocked_by_inactive_gate(declared_columns)
                else "declared_feature_columns_missing"
            )
            feature["failure_reason"] = (
                "declared feature columns are missing, unreachable, disabled, or not included in model feature columns"
            )
            return

    if not feature.get("applied"):
        return

    conditional_wiring = _conditional_cli_dependency_wiring(feature, wrapper, code_files, train_command)
    if conditional_wiring:
        feature["conditional_cli_wiring"] = conditional_wiring
        if not conditional_wiring.get("success", False):
            feature["applied"] = False
            feature["failure_category"] = "conditional_cli_not_wired"
            feature["failure_reason"] = (
                "feature CLI args are consumed in dependency code but not passed through train.py call sites"
            )
            return

    if not _change_requires_guardrail_activation(change, feature):
        return
    cli_guardrail_activation = _feature_guardrail_cli_activation_status(feature)
    if cli_guardrail_activation.get("active"):
        feature["guardrail_activation"] = cli_guardrail_activation
        return
    guardrail_activation = _guardrail_activation_status(wrapper, train_command)
    feature["guardrail_activation"] = guardrail_activation
    if not guardrail_activation.get("active", False):
        feature["applied"] = False
        feature["failure_category"] = "guardrail_not_activated"
        feature["failure_reason"] = (
            "guardrail feature evidence exists but trend_guardrail is not activated by train_command or argparse default"
        )


def _attach_implementation_assertions(
    change: dict[str, Any],
    feature: dict[str, Any],
    code_files: list[Path],
    modified_functions: list[str],
) -> None:
    required_calls = _required_plan_calls(change)
    if not required_calls:
        return
    modified_set = {str(item) for item in modified_functions if str(item)}
    assertions: list[dict[str, Any]] = []
    for call_name in required_calls:
        evidence = _call_evidence_for_name(code_files, call_name)
        modified_evidence = [
            item
            for item in evidence
            if f"{Path(str(item.get('path') or '')).name}::{item.get('function')}" in modified_set
        ]
        assertions.append(
            {
                "kind": "required_plan_call",
                "call": call_name,
                "evidence": evidence,
                "modified_function_evidence": modified_evidence,
                "success": bool(modified_evidence) if modified_set else bool(evidence),
            }
        )

    feature["implementation_assertions"] = {
        "success": all(item.get("success", False) for item in assertions),
        "modified_functions": sorted(modified_set),
        "assertions": assertions,
    }
    if not feature["implementation_assertions"]["success"]:
        feature["applied"] = False
        feature["failure_category"] = "implementation_assertion_failed"
        feature["failure_reason"] = (
            "required plan calls were not implemented in the functions modified by this Agent2 code package"
        )


def _required_plan_calls(change: dict[str, Any]) -> list[str]:
    construction = str(change.get("construction") or "")
    calls: list[str] = []
    patterns = [
        r"\bcall\s+([A-Za-z_][A-Za-z0-9_]*)\b",
        r"\b调用\s*([A-Za-z_][A-Za-z0-9_]*)\b",
    ]
    for pattern in patterns:
        calls.extend(match.group(1) for match in re.finditer(pattern, construction, flags=re.IGNORECASE))
    return list(dict.fromkeys(calls))


def _call_evidence_for_name(code_files: list[Path], call_name: str) -> list[dict[str, Any]]:
    evidence: list[dict[str, Any]] = []
    for path in code_files:
        if not path.exists() or not path.is_file():
            continue
        try:
            tree = ast.parse(path.read_text(encoding="utf-8", errors="ignore"))
        except SyntaxError:
            continue

        class Visitor(ast.NodeVisitor):
            def __init__(self) -> None:
                self.function_stack: list[str] = []

            def visit_FunctionDef(self, node: ast.FunctionDef) -> None:  # noqa: N802
                self.function_stack.append(node.name)
                self.generic_visit(node)
                self.function_stack.pop()

            def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:  # noqa: N802
                self.function_stack.append(node.name)
                self.generic_visit(node)
                self.function_stack.pop()

            def visit_Call(self, node: ast.Call) -> None:  # noqa: N802
                if _call_name(node.func).split(".")[-1] == call_name:
                    item = _evidence(path, int(getattr(node, "lineno", 0) or 0), call_name)
                    item["function"] = self.function_stack[-1] if self.function_stack else ""
                    evidence.append(item)
                self.generic_visit(node)

        Visitor().visit(tree)
    return _dedupe_evidence(evidence)


def _declared_feature_column_application(
    change: dict[str, Any],
    feature: dict[str, Any],
    wrapper: Path,
    code_files: list[Path],
    train_command: list[str] | None,
) -> dict[str, Any] | None:
    columns = [str(item) for item in feature.get("declared_feature_columns", []) or [] if str(item)]
    if not columns:
        return None
    require_feature_cols = _change_requires_feature_cols_integration(change)
    statuses: list[dict[str, Any]] = []
    for column in columns:
        evidence = [
            item
            for item in feature.get("evidence", []) or []
            if isinstance(item, dict)
            and item.get("kind") == "declared_feature_column_constructed"
            and item.get("declared_feature_column") == column
        ]
        reachability = (
            _entrypoint_reachability_for_evidence(evidence, wrapper, code_files)
            if evidence
            else {"reachable": False, "mode": "missing_declared_feature_column"}
        )
        activation = _declared_feature_column_activation_status(wrapper, code_files, evidence, train_command)
        feature_cols = (
            _declared_feature_column_model_input_status(code_files, column)
            if require_feature_cols
            else {"active": True, "source": "not_required"}
        )
        statuses.append(
            {
                "column": column,
                "constructed": bool(evidence),
                "evidence": evidence,
                "reachable": bool(reachability.get("reachable", False)),
                "reachability": reachability,
                "activation": activation,
                "feature_cols": feature_cols,
                "success": bool(evidence)
                and bool(reachability.get("reachable", False))
                and bool(activation.get("active", False))
                and bool(feature_cols.get("active", False)),
            }
        )
    return {
        "success": all(item.get("success", False) for item in statuses),
        "require_feature_cols": require_feature_cols,
        "columns": statuses,
    }


def _declared_feature_columns_blocked_by_inactive_gate(declared_columns: dict[str, Any]) -> bool:
    statuses = [item for item in declared_columns.get("columns", []) or [] if isinstance(item, dict)]
    if not statuses:
        return False
    inactive = [
        item
        for item in statuses
        if item.get("constructed")
        and item.get("reachable")
        and (item.get("activation") or {}).get("active") is False
    ]
    return bool(inactive) and len(inactive) == len(statuses)


def _change_requires_feature_cols_integration(change: dict[str, Any]) -> bool:
    text = str(change.get("construction") or "").lower()
    return "feature_cols" in text or "model input" in text or "model-input" in text


def _declared_feature_column_activation_status(
    wrapper: Path,
    code_files: list[Path],
    evidence: list[dict[str, Any]],
    train_command: list[str] | None,
) -> dict[str, Any]:
    wrapper_resolved = wrapper.resolve()
    code_by_resolved = {path.resolve(): path for path in code_files if path.exists()}
    inactive_call_sites: list[dict[str, Any]] = []
    checked: list[dict[str, Any]] = []
    for item in evidence:
        path = Path(str(item.get("path") or "")).resolve()
        function_name = str(item.get("function") or "")
        if not function_name:
            continue

        if path == wrapper_resolved:
            if function_name in {"main", "run_online_pipeline", "train_backtest_and_refit_model", "build_features"}:
                continue
            call_sites = _wrapper_function_activation_call_sites(wrapper, function_name, train_command)
            checked.append({"path": wrapper.name, "function": function_name, "call_sites": call_sites})
        else:
            dependency = code_by_resolved.get(path)
            if dependency is None:
                continue
            call_sites = _wrapper_dependency_function_call_sites(wrapper, dependency, function_name, train_command)
            checked.append({"path": dependency.name, "function": function_name, "call_sites": call_sites})

        if call_sites and any(site.get("active") for site in call_sites):
            return {"active": True, "source": "active_wrapper_call_site", "checked": checked}
        inactive_call_sites.extend(site for site in call_sites if site.get("active") is False)
    if inactive_call_sites:
        return {"active": False, "source": "inactive_wrapper_call_site", "checked": checked}
    return {"active": True, "source": "no_inactive_gate_detected", "checked": checked}


def _wrapper_function_activation_call_sites(
    wrapper: Path,
    function_name: str,
    train_command: list[str] | None,
) -> list[dict[str, Any]]:
    try:
        source_text = wrapper.read_text(encoding="utf-8", errors="ignore")
        tree = ast.parse(source_text)
    except SyntaxError:
        return []
    call_sites: list[dict[str, Any]] = []

    class Visitor(ast.NodeVisitor):
        def __init__(self) -> None:
            self.condition_stack: list[ast.AST] = []
            self.function_stack: list[str] = []

        def visit_FunctionDef(self, node: ast.FunctionDef) -> None:  # noqa: N802
            self.function_stack.append(node.name)
            self.generic_visit(node)
            self.function_stack.pop()

        def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:  # noqa: N802
            self.function_stack.append(node.name)
            self.generic_visit(node)
            self.function_stack.pop()

        def visit_If(self, node: ast.If) -> None:  # noqa: N802
            self.condition_stack.append(node.test)
            for child in node.body:
                self.visit(child)
            self.condition_stack.pop()
            if node.orelse:
                self.condition_stack.append(ast.UnaryOp(op=ast.Not(), operand=node.test))
                for child in node.orelse:
                    self.visit(child)
                self.condition_stack.pop()

        def visit_Call(self, node: ast.Call) -> None:  # noqa: N802
            if _call_name(node.func).split(".")[-1] == function_name:
                conditions = [
                    _condition_activation_status(source_text, train_command or [], condition)
                    for condition in self.condition_stack
                ]
                call_sites.append(
                    {
                        "path": wrapper.name,
                        "line": int(getattr(node, "lineno", 0) or 0),
                        "function": self.function_stack[-1] if self.function_stack else "",
                        "conditions": conditions,
                        "active": all(item.get("active", True) for item in conditions),
                    }
                )
            self.generic_visit(node)

    Visitor().visit(tree)
    return call_sites


def _condition_activation_status(source_text: str, command: list[str], node: ast.AST) -> dict[str, Any]:
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.Not):
        nested = _condition_activation_status(source_text, command, node.operand)
        if nested.get("active") is None:
            return nested
        return {**nested, "active": not bool(nested.get("active")), "negated": not bool(nested.get("negated", False))}
    if isinstance(node, ast.BoolOp):
        parts = [_condition_activation_status(source_text, command, item) for item in node.values]
        if isinstance(node.op, ast.And):
            return {"active": all(item.get("active", True) for item in parts), "source": "and", "parts": parts}
        if isinstance(node.op, ast.Or):
            return {"active": any(item.get("active", False) for item in parts), "source": "or", "parts": parts}
    dest = _condition_args_dest(node)
    if dest:
        value = _effective_arg_value_for_command(source_text, command, dest)
        return {"active": _truthy_arg_value(value), "source": "args_condition", "dest": dest, "value": value}
    return {"active": True, "source": "unresolved_condition", "snippet": _node_snippet(node)}


def _condition_args_dest(node: ast.AST) -> str:
    if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name) and node.value.id == "args":
        return node.attr
    if (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "getattr"
        and len(node.args) >= 2
        and isinstance(node.args[0], ast.Name)
        and node.args[0].id == "args"
        and isinstance(node.args[1], ast.Constant)
        and isinstance(node.args[1].value, str)
    ):
        return node.args[1].value
    return ""


def _effective_arg_value_for_command(source_text: str, command: list[str], dest: str) -> Any:
    command_value = _command_value_for_arg_dest(command, source_text, dest)
    if command_value is not None:
        return command_value
    preset_value = _preset_arg_value_for_command(source_text, command, dest)
    if preset_value is not None:
        return preset_value
    return _argparse_effective_default_for_dest(source_text, dest)


def _preset_arg_value_for_command(source_text: str, command: list[str], dest: str) -> Any | None:
    experiment_value = _command_value_for_arg_dest(command, source_text, "experiment")
    if experiment_value is None:
        return None
    try:
        tree = ast.parse(source_text)
    except SyntaxError:
        return None
    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        if "preset" not in node.name:
            continue
        value = _preset_branch_assignment(node.body, str(experiment_value), dest)
        if value is not None:
            return value
    return None


def _preset_branch_assignment(statements: list[ast.stmt], experiment_value: str, dest: str) -> Any | None:
    found: Any | None = None
    for statement in statements:
        if isinstance(statement, ast.Assign):
            assigned = _args_assignment_value(statement, dest)
            if assigned is not None:
                found = assigned
        elif isinstance(statement, ast.If):
            branch_value = _experiment_condition_value(statement.test, experiment_value)
            if branch_value is True:
                assigned = _preset_branch_assignment(statement.body, experiment_value, dest)
                if assigned is not None:
                    found = assigned
                if _statements_contain_return(statement.body):
                    return found
            elif branch_value is False:
                assigned = _preset_branch_assignment(statement.orelse, experiment_value, dest)
                if assigned is not None:
                    found = assigned
                if _statements_contain_return(statement.orelse):
                    return found
    return found


def _statements_contain_return(statements: list[ast.stmt]) -> bool:
    return any(isinstance(node, ast.Return) for statement in statements for node in ast.walk(statement))


def _args_assignment_value(statement: ast.Assign, dest: str) -> Any | None:
    for target in statement.targets:
        if isinstance(target, ast.Attribute) and isinstance(target.value, ast.Name) and target.value.id == "args" and target.attr == dest:
            try:
                return ast.literal_eval(statement.value)
            except Exception:
                return _node_snippet(statement.value)
    return None


def _experiment_condition_value(node: ast.AST, experiment_value: str) -> bool | None:
    if not isinstance(node, ast.Compare) or len(node.ops) != 1 or len(node.comparators) != 1:
        return None
    left = node.left
    right = node.comparators[0]
    if not (
        isinstance(left, ast.Attribute)
        and isinstance(left.value, ast.Name)
        and left.value.id == "args"
        and left.attr == "experiment"
        and isinstance(right, ast.Constant)
    ):
        return None
    expected = str(right.value)
    if isinstance(node.ops[0], ast.Eq):
        return experiment_value == expected
    if isinstance(node.ops[0], ast.NotEq):
        return experiment_value != expected
    return None


def _argparse_effective_default_for_dest(source_text: str, dest: str) -> Any | None:
    flags = _argparse_command_spec(source_text).get("flags", {}) or {}
    for spec in flags.values():
        if spec.get("dest") != dest:
            continue
        default = spec.get("default")
        if default is not None:
            return default
        action = str(spec.get("action") or "")
        if action == "store_true":
            return False
        if action == "store_false":
            return True
    return None


def _truthy_arg_value(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() not in {"", "0", "false", "none", "no", "off"}


def _declared_feature_column_model_input_status(code_files: list[Path], column: str) -> dict[str, Any]:
    for path in code_files:
        status = _feature_cols_includes_column(path, column)
        if status.get("active"):
            return status
    return {"active": False, "source": "feature_cols_not_found", "column": column}


def _feature_cols_includes_column(path: Path, column: str) -> dict[str, Any]:
    try:
        tree = ast.parse(path.read_text(encoding="utf-8", errors="ignore"))
    except SyntaxError:
        return {"active": False, "source": "syntax_error", "path": path.name, "column": column}
    lowered = column.lower()
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and any(isinstance(target, ast.Name) and target.id == "feature_cols" for target in node.targets):
            if _expr_collection_contains_string(node.value, lowered):
                return {"active": True, "source": "explicit_feature_cols", "path": path.name, "column": column}
            if _expr_uses_dataframe_columns(node.value):
                return {"active": True, "source": "dataframe_columns_feature_cols", "path": path.name, "column": column}
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            if not isinstance(node.func.value, ast.Name) or node.func.value.id != "feature_cols":
                continue
            if node.func.attr == "append" and node.args and _expr_collection_contains_string(node.args[0], lowered):
                return {"active": True, "source": "feature_cols_append", "path": path.name, "column": column}
            if node.func.attr == "extend" and node.args and _expr_collection_contains_string(node.args[0], lowered):
                return {"active": True, "source": "feature_cols_extend", "path": path.name, "column": column}
    return {"active": False, "source": "feature_cols_not_found", "path": path.name, "column": column}


def _expr_collection_contains_string(node: ast.AST, lowered: str) -> bool:
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value.lower() == lowered
    if isinstance(node, (ast.List, ast.Tuple, ast.Set)):
        return any(_expr_collection_contains_string(item, lowered) for item in node.elts)
    return False


def _expr_uses_dataframe_columns(node: ast.AST) -> bool:
    if isinstance(node, ast.ListComp):
        return any(_expr_uses_dataframe_columns(generator.iter) for generator in node.generators)
    if isinstance(node, ast.Call) and node.args:
        return any(_expr_uses_dataframe_columns(arg) for arg in node.args)
    if isinstance(node, ast.Attribute) and node.attr == "columns":
        return True
    return any(_expr_uses_dataframe_columns(child) for child in ast.iter_child_nodes(node))


def _conditional_cli_dependency_wiring(
    feature: dict[str, Any],
    wrapper: Path,
    code_files: list[Path],
    train_command: list[str] | None = None,
) -> dict[str, Any] | None:
    if not feature.get("cli_args"):
        return None
    cli_evidence = [
        item
        for item in feature.get("evidence", []) or []
        if isinstance(item, dict) and item.get("kind") == "cli_arg_consumed"
    ]
    if not cli_evidence:
        return None
    wrapper_resolved = wrapper.resolve()
    code_by_resolved = {path.resolve(): path for path in code_files if path.exists()}
    checked: list[dict[str, Any]] = []
    missing: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    for item in cli_evidence:
        evidence_path = Path(str(item.get("path") or "")).resolve()
        if evidence_path == wrapper_resolved:
            continue
        dependency = code_by_resolved.get(evidence_path)
        if dependency is None:
            continue
        function_name = str(item.get("function") or "") or _function_name_at_line(dependency, int(item.get("line", 0) or 0))
        if not function_name:
            continue
        key = (dependency.resolve().as_posix(), function_name)
        if key in seen:
            continue
        seen.add(key)
        call_sites = _wrapper_dependency_function_call_sites(wrapper, dependency, function_name, train_command)
        checked.append({"path": dependency.name, "function": function_name, "call_sites": call_sites})
        if not call_sites:
            missing.append(
                {
                    "path": dependency.name,
                    "function": function_name,
                    "reason": "no direct train.py call site found to verify args pass-through",
                }
            )
            continue
        missing.extend(call for call in call_sites if call.get("active") is not False and not call.get("passes_args"))
    if not checked:
        return None
    return {"success": not missing, "checked": checked, "missing_args_call_sites": missing}


def _wrapper_dependency_function_call_sites(
    wrapper: Path,
    dependency: Path,
    function_name: str,
    train_command: list[str] | None = None,
) -> list[dict[str, Any]]:
    try:
        source_text = wrapper.read_text(encoding="utf-8", errors="ignore")
        wrapper_tree = ast.parse(source_text)
        dependency_tree = ast.parse(dependency.read_text(encoding="utf-8", errors="ignore"))
    except SyntaxError:
        return []
    dependency_module = dependency.stem
    dependency_functions = {
        node.name
        for node in ast.walk(dependency_tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    imported_symbols: dict[str, str] = {}
    module_aliases: set[str] = set()
    wildcard_import = False
    for node in ast.walk(wrapper_tree):
        if isinstance(node, ast.ImportFrom) and (node.module or "").split(".")[-1] == dependency_module:
            for alias in node.names:
                if alias.name == "*":
                    wildcard_import = True
                    continue
                imported_symbols[alias.asname or alias.name] = alias.name
        elif isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.split(".")[-1] == dependency_module:
                    module_aliases.add(alias.asname or alias.name.split(".")[-1])

    arg_index = _function_param_index(dependency_tree, function_name, "args")
    call_sites: list[dict[str, Any]] = []

    class Visitor(ast.NodeVisitor):
        def __init__(self) -> None:
            self.function_stack: list[str] = []
            self.condition_stack: list[ast.AST] = []

        def visit_FunctionDef(self, node: ast.FunctionDef) -> None:  # noqa: N802
            self.function_stack.append(node.name)
            self.generic_visit(node)
            self.function_stack.pop()

        def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:  # noqa: N802
            self.function_stack.append(node.name)
            self.generic_visit(node)
            self.function_stack.pop()

        def visit_If(self, node: ast.If) -> None:  # noqa: N802
            self.condition_stack.append(node.test)
            for child in node.body:
                self.visit(child)
            self.condition_stack.pop()
            if node.orelse:
                self.condition_stack.append(ast.UnaryOp(op=ast.Not(), operand=node.test))
                for child in node.orelse:
                    self.visit(child)
                self.condition_stack.pop()

        def visit_Call(self, node: ast.Call) -> None:  # noqa: N802
            if _wrapper_call_targets_dependency_function(
                node,
                function_name,
                dependency_functions,
                imported_symbols,
                module_aliases,
                wildcard_import,
            ):
                conditions = [
                    _condition_activation_status(source_text, train_command or [], condition)
                    for condition in self.condition_stack
                ]
                call_sites.append(
                    {
                        "path": wrapper.name,
                        "line": int(getattr(node, "lineno", 0) or 0),
                        "function": self.function_stack[-1] if self.function_stack else "",
                        "snippet": _node_snippet(node.func),
                        "passes_args": _call_passes_args_object(node, arg_index),
                        "conditions": conditions,
                        "active": all(item.get("active", True) for item in conditions),
                    }
                )
            self.generic_visit(node)

    Visitor().visit(wrapper_tree)
    return call_sites


def _wrapper_call_targets_dependency_function(
    node: ast.Call,
    function_name: str,
    dependency_functions: set[str],
    imported_symbols: dict[str, str],
    module_aliases: set[str],
    wildcard_import: bool,
) -> bool:
    func = node.func
    if isinstance(func, ast.Name):
        if imported_symbols.get(func.id) == function_name:
            return True
        return wildcard_import and func.id == function_name and func.id in dependency_functions
    if isinstance(func, ast.Attribute) and isinstance(func.value, ast.Name) and func.value.id in module_aliases:
        return func.attr == function_name
    return False


def _function_param_index(tree: ast.AST, function_name: str, param_name: str) -> int | None:
    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) or node.name != function_name:
            continue
        params = [arg.arg for arg in node.args.args]
        if param_name in params:
            return params.index(param_name)
    return None


def _call_passes_args_object(node: ast.Call, arg_index: int | None) -> bool:
    for keyword in node.keywords:
        if keyword.arg == "args":
            return True
    if arg_index is None:
        return False
    return len(node.args) > arg_index and isinstance(node.args[arg_index], ast.Name) and node.args[arg_index].id == "args"


def _change_requires_guardrail_activation(change: dict[str, Any], feature: dict[str, Any]) -> bool:
    text = " ".join(
        str(change.get(key) or "")
        for key in ("feature_name", "feature_type", "construction")
    ).lower()
    if "guardrail" not in text and "zero_demand_guardrail" not in text:
        return False
    guardrail_evidence = [
        item
        for item in feature.get("evidence", []) or []
        if isinstance(item, dict)
        and (
            item.get("kind") == "zero_demand_guardrail_consumed"
            or "guardrail" in str(item.get("snippet") or "").lower()
        )
    ]
    return bool(guardrail_evidence)


def _guardrail_activation_status(wrapper: Path, train_command: list[str] | None) -> dict[str, Any]:
    source_text = wrapper.read_text(encoding="utf-8", errors="ignore") if wrapper.exists() else ""
    command_value = _command_value_for_arg_dest(train_command or [], source_text, "trend_guardrail")
    if command_value is not None:
        return {
            "active": _guardrail_value_active(command_value),
            "source": "train_command",
            "value": command_value,
        }
    default_value = _argparse_default_for_dest(source_text, "trend_guardrail")
    if default_value is not None:
        return {
            "active": _guardrail_value_active(str(default_value)),
            "source": "argparse_default",
            "value": str(default_value),
        }
    return {"active": False, "source": "missing", "value": ""}


def _feature_guardrail_cli_activation_status(feature: dict[str, Any]) -> dict[str, Any]:
    flags = [
        str(item)
        for item in feature.get("cli_args", []) or []
        if str(item).startswith("--") and "guardrail" in str(item).lower()
    ]
    enabled_flags = [flag for flag in flags if not flag.startswith("--no-")]
    reachability = feature.get("cli_consumption_reachability") or {}
    if enabled_flags and feature.get("cli_contract_success") and reachability.get("reachable", False):
        return {"active": True, "source": "feature_cli_args", "value": " ".join(enabled_flags)}
    return {"active": False, "source": "feature_cli_args", "value": " ".join(flags)}


def _command_value_for_arg_dest(command: list[str], source_text: str, dest: str) -> str | None:
    if not command:
        return None
    flags_by_dest = {
        flag
        for flag, spec in (_argparse_command_spec(source_text).get("flags", {}) or {}).items()
        if spec.get("dest") == dest
    }
    if not flags_by_dest:
        flags_by_dest = {"--" + dest, "--" + dest.replace("_", "-")}
    index = 0
    while index < len(command):
        token = str(command[index])
        flag = token.split("=", 1)[0] if token.startswith("--") else token
        if flag not in flags_by_dest:
            index += 1
            continue
        if "=" in token:
            return token.split("=", 1)[1]
        if index + 1 < len(command) and not str(command[index + 1]).startswith("--"):
            return str(command[index + 1])
        return "true"
    return None


def _argparse_default_for_dest(source_text: str, dest: str) -> Any | None:
    flags = _argparse_command_spec(source_text).get("flags", {}) or {}
    for spec in flags.values():
        if spec.get("dest") == dest:
            return spec.get("default")
    return None


def _guardrail_value_active(value: str) -> bool:
    return str(value).strip().lower() not in {"", "none", "false", "0", "off", "no"}


def _entrypoint_reachability_for_evidence(evidence: list[dict[str, Any]], wrapper: Path, code_files: list[Path]) -> dict[str, Any]:
    wrapper_resolved = wrapper.resolve()
    for item in evidence:
        path = Path(str(item.get("path") or "")).resolve()
        if path == wrapper_resolved:
            return {"reachable": True, "mode": "direct_entrypoint_evidence", "path": wrapper.as_posix()}

    code_by_resolved = {path.resolve(): path for path in code_files if path.exists()}
    checked: list[dict[str, Any]] = []
    for item in evidence:
        evidence_path = Path(str(item.get("path") or "")).resolve()
        dependency = code_by_resolved.get(evidence_path)
        if dependency is None or dependency == wrapper_resolved:
            continue
        function_name = str(item.get("function") or "") or _function_name_at_line(dependency, int(item.get("line", 0) or 0))
        reachable = _reachable_dependency_functions_from_wrapper(wrapper, dependency)
        checked.append(
            {
                "path": dependency.name,
                "function": function_name,
                "reachable_functions": sorted(reachable),
            }
        )
        if function_name and function_name in reachable:
            return {
                "reachable": True,
                "mode": "imported_dependency_function",
                "path": dependency.as_posix(),
                "function": function_name,
            }
    return {
        "reachable": False,
        "mode": "unreachable_dependency_evidence",
        "checked": checked,
    }


def _function_name_at_line(path: Path, line: int) -> str:
    if line <= 0:
        return ""
    try:
        tree = ast.parse(path.read_text(encoding="utf-8", errors="ignore"))
    except SyntaxError:
        return ""
    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        start = int(getattr(node, "lineno", 0))
        end = int(getattr(node, "end_lineno", start))
        if start <= line <= end:
            return node.name
    return ""


def _function_names_in_file(path: Path) -> list[str]:
    try:
        tree = ast.parse(path.read_text(encoding="utf-8", errors="ignore"))
    except SyntaxError:
        return []
    return [
        node.name
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    ]


def _reachable_dependency_functions_from_wrapper(wrapper: Path, dependency: Path) -> set[str]:
    try:
        wrapper_tree = ast.parse(wrapper.read_text(encoding="utf-8", errors="ignore"))
        dependency_tree = ast.parse(dependency.read_text(encoding="utf-8", errors="ignore"))
    except SyntaxError:
        return set()
    dependency_module = dependency.stem
    dependency_functions = {
        node.name
        for node in ast.walk(dependency_tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    imported_symbols: dict[str, str] = {}
    module_aliases: set[str] = set()
    wildcard_import = False
    for node in ast.walk(wrapper_tree):
        if isinstance(node, ast.ImportFrom) and (node.module or "").split(".")[-1] == dependency_module:
            for alias in node.names:
                if alias.name == "*":
                    wildcard_import = True
                    continue
                imported_symbols[alias.asname or alias.name] = alias.name
        elif isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.split(".")[-1] == dependency_module:
                    module_aliases.add(alias.asname or alias.name.split(".")[-1])

    seeds: set[str] = set()
    for node in ast.walk(wrapper_tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if isinstance(func, ast.Name):
            if func.id in imported_symbols:
                seeds.add(imported_symbols[func.id])
            elif wildcard_import and func.id in dependency_functions:
                seeds.add(func.id)
        elif isinstance(func, ast.Attribute) and isinstance(func.value, ast.Name) and func.value.id in module_aliases:
            if func.attr in dependency_functions:
                seeds.add(func.attr)

    call_graph = _dependency_call_graph(dependency_tree, dependency_functions)
    reachable = set(seeds)
    queue = list(seeds)
    while queue:
        current = queue.pop()
        for called in call_graph.get(current, set()):
            if called in reachable:
                continue
            reachable.add(called)
            queue.append(called)
    return reachable


def _dependency_call_graph(tree: ast.AST, dependency_functions: set[str]) -> dict[str, set[str]]:
    graph: dict[str, set[str]] = {}
    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        calls: set[str] = set()
        for child in ast.walk(node):
            if not isinstance(child, ast.Call):
                continue
            call_name = _call_name(child.func).split(".")[-1]
            if call_name in dependency_functions:
                calls.add(call_name)
        graph[node.name] = calls
    return graph


def _feature_cli_dest_names(cli_args: list[str]) -> list[str]:
    dests: list[str] = []
    for arg in cli_args:
        if not arg.startswith("--"):
            continue
        body = arg[5:] if arg.startswith("--no-") else arg[2:]
        if not body:
            continue
        dests.append(body.replace("-", "_"))
    return list(dict.fromkeys(dests))


_DECLARED_FEATURE_COLUMN_STOPWORDS = {
    "args",
    "data",
    "date",
    "date_col",
    "df",
    "feature_col",
    "feature_cols",
    "group_cols",
    "id_cols",
    "label_col",
    "model",
    "output_dir",
    "prediction",
    "row",
    "rows",
    "split",
    "target_col",
    "train_command",
}


def _declared_feature_columns(change: dict[str, Any]) -> list[str]:
    construction = str(change.get("construction") or "")
    if not construction:
        return []
    source_fields = {str(item).strip().lower() for item in change.get("field_sources", []) or [] if str(item).strip()}
    candidates: list[str] = []
    patterns = [
        r"(?m)^\s*[-*]\s*([A-Za-z_][A-Za-z0-9_]{2,})\s*:",
        r"\b(?:set|create|compute|derive|add|include)\s+([A-Za-z_][A-Za-z0-9_]{2,})\b",
        r"[`'\"]([A-Za-z_][A-Za-z0-9_]{2,})[`'\"]",
    ]
    for pattern in patterns:
        for match in re.finditer(pattern, construction):
            candidates.append(match.group(1))
    declared: list[str] = []
    for candidate in candidates:
        token = candidate.strip()
        lowered = token.lower()
        if not _is_declared_feature_column_candidate(lowered, source_fields):
            continue
        declared.append(token)
    return list(dict.fromkeys(declared))


def _is_declared_feature_column_candidate(token: str, source_fields: set[str]) -> bool:
    if "_" not in token:
        return False
    if token in source_fields or token in _DECLARED_FEATURE_COLUMN_STOPWORDS:
        return False
    if token.endswith("_col") or token.endswith("_cols"):
        return False
    helper_prefixes = (
        "add_",
        "apply_",
        "build_",
        "create_",
        "encode_",
        "fetch_",
        "generate_",
        "infer_",
        "load_",
        "parse_",
        "run_",
        "train_",
    )
    return not token.startswith(helper_prefixes)


def _change_feature_tokens(change: dict[str, Any]) -> list[str]:
    tokens = _feature_code_tokens(str(change.get("feature_name") or ""))
    tokens.extend(_declared_feature_columns(change))
    for key in ("feature_slug", "feature_type"):
        token = str(change.get(key) or "").strip()
        if token:
            tokens.extend(_feature_code_tokens(token))
    for raw in change.get("audit_tokens", []) or []:
        token = str(raw).strip()
        if token:
            tokens.extend(_feature_code_tokens(token))
    return list(dict.fromkeys(tokens))


def _feature_code_tokens(feature_name: str) -> list[str]:
    if not feature_name:
        return []
    snake = re.sub(r"[^0-9A-Za-z_]+", "_", feature_name).strip("_")
    hyphen = snake.replace("_", "-")
    return [token for token in dict.fromkeys([feature_name, snake, hyphen]) if token]


def _arg_usage_evidence(path: Path, cli_dests: list[str]) -> list[dict[str, Any]]:
    try:
        tree = ast.parse(path.read_text(encoding="utf-8", errors="ignore"))
    except SyntaxError:
        return []

    class Visitor(ast.NodeVisitor):
        def __init__(self) -> None:
            self.function_stack: list[str] = []
            self.evidence: list[dict[str, Any]] = []

        def visit_FunctionDef(self, node: ast.FunctionDef) -> None:  # noqa: N802 - ast visitor API
            self.function_stack.append(node.name)
            self.generic_visit(node)
            self.function_stack.pop()

        def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:  # noqa: N802 - ast visitor API
            self.function_stack.append(node.name)
            self.generic_visit(node)
            self.function_stack.pop()

        def visit_Attribute(self, node: ast.Attribute) -> None:  # noqa: N802 - ast visitor API
            if (
                isinstance(node.value, ast.Name)
                and node.value.id == "args"
                and node.attr in cli_dests
                and not self._inside_parse_args()
            ):
                item = _evidence(path, node.lineno, f"args.{node.attr}")
                item["function"] = self.function_stack[-1] if self.function_stack else ""
                self.evidence.append(item)
            self.generic_visit(node)

        def visit_Call(self, node: ast.Call) -> None:  # noqa: N802 - ast visitor API
            if (
                isinstance(node.func, ast.Name)
                and node.func.id == "getattr"
                and len(node.args) >= 2
                and isinstance(node.args[0], ast.Name)
                and node.args[0].id == "args"
                and isinstance(node.args[1], ast.Constant)
                and node.args[1].value in cli_dests
                and not self._inside_parse_args()
            ):
                item = _evidence(path, node.lineno, f"getattr(args, {node.args[1].value!r})")
                item["function"] = self.function_stack[-1] if self.function_stack else ""
                self.evidence.append(item)
            self.generic_visit(node)

        def _inside_parse_args(self) -> bool:
            return "parse_args" in self.function_stack

    visitor = Visitor()
    visitor.visit(tree)
    return visitor.evidence


def _feature_token_evidence(path: Path, tokens: list[str]) -> list[dict[str, Any]]:
    if not tokens:
        return []
    try:
        tree = ast.parse(path.read_text(encoding="utf-8", errors="ignore"))
    except SyntaxError:
        return []
    lowered_tokens = [token.lower() for token in tokens]

    class Visitor(ast.NodeVisitor):
        def __init__(self) -> None:
            self.function_stack: list[str] = []
            self.feature_column_vars_stack: list[set[str]] = [set()]
            self.string_bindings_stack: list[dict[str, str]] = [{}]
            self.feature_function_names: set[str] = {
                node.name
                for node in ast.walk(tree)
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
                and _contains_any_token(node.name, lowered_tokens)
            }
            self.evidence: list[dict[str, Any]] = []

        def visit_FunctionDef(self, node: ast.FunctionDef) -> None:  # noqa: N802 - ast visitor API
            self.function_stack.append(node.name)
            self.feature_column_vars_stack.append(set())
            self.string_bindings_stack.append({})
            self.generic_visit(node)
            self.string_bindings_stack.pop()
            self.feature_column_vars_stack.pop()
            self.function_stack.pop()

        def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:  # noqa: N802 - ast visitor API
            self.function_stack.append(node.name)
            self.feature_column_vars_stack.append(set())
            self.string_bindings_stack.append({})
            self.generic_visit(node)
            self.string_bindings_stack.pop()
            self.feature_column_vars_stack.pop()
            self.function_stack.pop()

        def visit_Assign(self, node: ast.Assign) -> None:  # noqa: N802 - ast visitor API
            for target in node.targets:
                self._update_feature_column_binding(target, node.value)
            if self._inside_feature_build_path():
                for target in node.targets:
                    if _node_constructs_feature_column(
                        target,
                        lowered_tokens,
                        self._feature_column_vars(),
                        self._string_bindings(),
                    ):
                        item = _evidence(path, node.lineno, _node_snippet(target))
                        item["kind"] = "feature_column_constructed"
                        item["function"] = self.function_stack[-1] if self.function_stack else ""
                        self.evidence.append(item)
            self.generic_visit(node)

        def visit_AnnAssign(self, node: ast.AnnAssign) -> None:  # noqa: N802 - ast visitor API
            if node.value is not None:
                self._update_feature_column_binding(node.target, node.value)
            if self._inside_feature_build_path() and _node_constructs_feature_column(
                node.target,
                lowered_tokens,
                self._feature_column_vars(),
                self._string_bindings(),
            ):
                item = _evidence(path, node.lineno, _node_snippet(node.target))
                item["kind"] = "feature_column_constructed"
                item["function"] = self.function_stack[-1] if self.function_stack else ""
                self.evidence.append(item)
            self.generic_visit(node)

        def visit_AugAssign(self, node: ast.AugAssign) -> None:  # noqa: N802 - ast visitor API
            if self._inside_feature_build_path() and _node_constructs_feature_column(
                node.target,
                lowered_tokens,
                self._feature_column_vars(),
                self._string_bindings(),
            ):
                item = _evidence(path, node.lineno, _node_snippet(node.target))
                item["kind"] = "feature_column_constructed"
                item["function"] = self.function_stack[-1] if self.function_stack else ""
                self.evidence.append(item)
            self.generic_visit(node)

        def visit_Call(self, node: ast.Call) -> None:  # noqa: N802 - ast visitor API
            call_name = _call_name(node.func)
            if call_name and (call_name in self.feature_function_names or _contains_any_token(call_name, lowered_tokens)):
                item = _evidence(path, node.lineno, call_name)
                item["kind"] = "feature_helper_called"
                item["function"] = self.function_stack[-1] if self.function_stack else ""
                self.evidence.append(item)
            self.generic_visit(node)

        def _inside_feature_build_path(self) -> bool:
            if not self.function_stack:
                return True
            current = self.function_stack[-1]
            return current in {
                "build_features",
                "run_online_pipeline",
                "train_backtest_and_refit_model",
                "main",
            }

        def _feature_column_vars(self) -> set[str]:
            return self.feature_column_vars_stack[-1]

        def _string_bindings(self) -> dict[str, str]:
            return self.string_bindings_stack[-1]

        def _update_feature_column_binding(self, target: ast.AST, value: ast.AST) -> None:
            if not isinstance(target, ast.Name):
                return
            bindings = self._string_bindings()
            feature_vars = self._feature_column_vars()
            pattern = _string_pattern_from_expr(value, bindings)
            if pattern is not None:
                bindings[target.id] = pattern
            else:
                bindings.pop(target.id, None)
            if _expr_mentions_feature_token(value, lowered_tokens, feature_vars, bindings):
                feature_vars.add(target.id)
            else:
                feature_vars.discard(target.id)

    visitor = Visitor()
    visitor.visit(tree)
    return visitor.evidence


def _declared_feature_column_evidence(path: Path, columns: list[str]) -> list[dict[str, Any]]:
    if not columns:
        return []
    try:
        tree = ast.parse(path.read_text(encoding="utf-8", errors="ignore"))
    except SyntaxError:
        return []
    evidence: list[dict[str, Any]] = []
    lowered_columns = {column.lower(): column for column in columns}

    class Visitor(ast.NodeVisitor):
        def __init__(self) -> None:
            self.function_stack: list[str] = []
            self.feature_column_vars_stack: list[set[str]] = [set()]
            self.string_bindings_stack: list[dict[str, str]] = [{}]

        def visit_FunctionDef(self, node: ast.FunctionDef) -> None:  # noqa: N802
            self.function_stack.append(node.name)
            self.feature_column_vars_stack.append(set())
            self.string_bindings_stack.append({})
            self.generic_visit(node)
            self.string_bindings_stack.pop()
            self.feature_column_vars_stack.pop()
            self.function_stack.pop()

        def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:  # noqa: N802
            self.function_stack.append(node.name)
            self.feature_column_vars_stack.append(set())
            self.string_bindings_stack.append({})
            self.generic_visit(node)
            self.string_bindings_stack.pop()
            self.feature_column_vars_stack.pop()
            self.function_stack.pop()

        def visit_Assign(self, node: ast.Assign) -> None:  # noqa: N802
            for target in node.targets:
                self._update_feature_column_binding(target, node.value)
            for target in node.targets:
                self._add_evidence_for_target(target, node.lineno)
            self.generic_visit(node)

        def visit_AnnAssign(self, node: ast.AnnAssign) -> None:  # noqa: N802
            if node.value is not None:
                self._update_feature_column_binding(node.target, node.value)
            self._add_evidence_for_target(node.target, node.lineno)
            self.generic_visit(node)

        def visit_AugAssign(self, node: ast.AugAssign) -> None:  # noqa: N802
            self._add_evidence_for_target(node.target, node.lineno)
            self.generic_visit(node)

        def _feature_column_vars(self) -> set[str]:
            return self.feature_column_vars_stack[-1]

        def _string_bindings(self) -> dict[str, str]:
            return self.string_bindings_stack[-1]

        def _update_feature_column_binding(self, target: ast.AST, value: ast.AST) -> None:
            if not isinstance(target, ast.Name):
                return
            bindings = self._string_bindings()
            feature_vars = self._feature_column_vars()
            pattern = _string_pattern_from_expr(value, bindings)
            if pattern is not None:
                bindings[target.id] = pattern
            else:
                bindings.pop(target.id, None)
            if any(_expr_mentions_feature_token(value, [column], feature_vars, bindings) for column in lowered_columns):
                feature_vars.add(target.id)
            else:
                feature_vars.discard(target.id)

        def _add_evidence_for_target(self, target: ast.AST, line: int) -> None:
            for lowered, column in lowered_columns.items():
                if not _node_constructs_feature_column(
                    target,
                    [lowered],
                    self._feature_column_vars(),
                    self._string_bindings(),
                ):
                    continue
                item = _evidence(path, line, _node_snippet(target))
                item["kind"] = "declared_feature_column_constructed"
                item["function"] = self.function_stack[-1] if self.function_stack else ""
                item["declared_feature_column"] = column
                evidence.append(item)

    Visitor().visit(tree)
    return evidence


def _structured_feature_evidence(change: dict[str, Any], path: Path) -> list[dict[str, Any]]:
    try:
        tree = ast.parse(path.read_text(encoding="utf-8", errors="ignore"))
    except SyntaxError:
        return []
    checks = _structured_evidence_checks(change)
    if not checks:
        return []

    class Visitor(ast.NodeVisitor):
        def __init__(self) -> None:
            self.function_stack: list[str] = []
            self.evidence: list[dict[str, Any]] = []

        def visit_FunctionDef(self, node: ast.FunctionDef) -> None:  # noqa: N802 - ast visitor API
            self.function_stack.append(node.name)
            self.generic_visit(node)
            self.function_stack.pop()

        def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:  # noqa: N802 - ast visitor API
            self.function_stack.append(node.name)
            self.generic_visit(node)
            self.function_stack.pop()

        def visit_Call(self, node: ast.Call) -> None:  # noqa: N802 - ast visitor API
            call_name = _call_name(node.func)
            if "rolling_sum" in checks and self._inside_feature_build_path() and call_name.split(".")[-1] == "generate_rolling_features":
                if _call_functions_include_sum(node):
                    item = _evidence(path, node.lineno, "generate_rolling_features(..., functions=..., sum)")
                    item["kind"] = "rolling_sum_constructed"
                    item["function"] = self.function_stack[-1] if self.function_stack else ""
                    self.evidence.append(item)
            if "sample_weight" in checks and _call_consumes_sample_weight(node):
                item = _evidence(path, node.lineno, call_name or "model.fit(..., sample_weight=...)")
                item["kind"] = "sample_weight_consumed"
                item["function"] = self.function_stack[-1] if self.function_stack else ""
                self.evidence.append(item)
            if "calibration" in checks and _call_name_matches_any(call_name, ["calibration", "calibrator"]):
                item = _evidence(path, node.lineno, call_name)
                item["kind"] = "calibration_fit_apply"
                item["function"] = self.function_stack[-1] if self.function_stack else ""
                self.evidence.append(item)
            if "zero_demand_guardrail" in checks and _call_name_matches_any(call_name, ["guardrail", "zero_demand", "zero"]):
                item = _evidence(path, node.lineno, call_name)
                item["kind"] = "zero_demand_guardrail_consumed"
                item["function"] = self.function_stack[-1] if self.function_stack else ""
                self.evidence.append(item)
            self.generic_visit(node)

        def visit_Assign(self, node: ast.Assign) -> None:  # noqa: N802 - ast visitor API
            if self._inside_feature_build_path():
                for target in node.targets:
                    if "rolling_sum" in checks and _node_constructs_rolling_sum_column(target):
                        item = _evidence(path, node.lineno, _node_snippet(target))
                        item["kind"] = "rolling_sum_constructed"
                        item["function"] = self.function_stack[-1] if self.function_stack else ""
                        self.evidence.append(item)
                    if "lifecycle" in checks and _node_constructs_named_feature(target, ["lifecycle", "package_age", "days_to_end"]):
                        item = _evidence(path, node.lineno, _node_snippet(target))
                        item["kind"] = "lifecycle_feature_constructed"
                        item["function"] = self.function_stack[-1] if self.function_stack else ""
                        self.evidence.append(item)
                    if "holiday" in checks and _node_constructs_named_feature(target, ["holiday", "eve", "span"]):
                        item = _evidence(path, node.lineno, _node_snippet(target))
                        item["kind"] = "holiday_position_constructed"
                        item["function"] = self.function_stack[-1] if self.function_stack else ""
                        self.evidence.append(item)
                    if "zero_demand_guardrail" in checks and _node_constructs_named_feature(target, ["zero_demand", "guardrail"]):
                        item = _evidence(path, node.lineno, _node_snippet(target))
                        item["kind"] = "zero_demand_guardrail_consumed"
                        item["function"] = self.function_stack[-1] if self.function_stack else ""
                        self.evidence.append(item)
            self.generic_visit(node)

        def _inside_feature_build_path(self) -> bool:
            if not self.function_stack:
                return True
            return self.function_stack[-1] in {
                "build_features",
                "run_online_pipeline",
                "train_backtest_and_refit_model",
                "main",
            }

    visitor = Visitor()
    visitor.visit(tree)
    return visitor.evidence


def _structured_evidence_checks(change: dict[str, Any]) -> set[str]:
    checks: set[str] = set()
    if _is_rolling_sum_change(change):
        checks.add("rolling_sum")
    if _change_matches_type(change, "lifecycle_interaction"):
        checks.add("lifecycle")
    if _change_matches_type(change, "holiday_position"):
        checks.add("holiday")
    if _change_matches_type(change, "train_weight"):
        checks.add("sample_weight")
    if _change_matches_type(change, "group_calibration"):
        checks.add("calibration")
    if _change_matches_type(change, "zero_demand_guardrail"):
        checks.add("zero_demand_guardrail")
    return checks


def _is_rolling_sum_change(change: dict[str, Any]) -> bool:
    text = " ".join(
        str(change.get(key) or "")
        for key in ("feature_name", "feature_slug", "feature_type", "construction")
    ).lower()
    tokens = " ".join(str(item).lower() for item in change.get("audit_tokens", []) or [])
    combined = f"{text} {tokens}"
    return "rolling" in combined and ("sum" in combined or "总和" in combined)


def _is_sample_weight_wiring_change(change: dict[str, Any]) -> bool:
    text = " ".join(
        str(change.get(key) or "")
        for key in ("feature_name", "feature_slug", "feature_type", "construction")
    ).lower()
    tokens = " ".join(str(item).lower() for item in change.get("audit_tokens", []) or [])
    combined = f"{text} {tokens}"
    return "sample_weight" in combined or "train_weight" in combined or "训练权重" in combined


def _is_zero_history_flag_change(change: dict[str, Any]) -> bool:
    text = " ".join(
        str(change.get(key) or "")
        for key in ("feature_name", "feature_slug", "feature_type", "construction")
    ).lower()
    tokens = " ".join(str(item).lower() for item in change.get("audit_tokens", []) or [])
    combined = f"{text} {tokens}"
    if "zero_history_flag" in combined:
        return True
    zero_terms = ("true_pos_cnt == 0", "true_pos_cnt==0", "label_col == 0", "target_col == 0")
    history_terms = ("past 14", "过去 14", "14 天", "14天", "rolling")
    threshold_terms = (">=10", ">= 10", "至少10", "至少 10")
    return ("zero" in combined or "为0" in combined or "== 0" in combined) and any(
        term in combined for term in history_terms
    ) and any(term in combined for term in threshold_terms + zero_terms)


def _call_functions_include_sum(node: ast.Call) -> bool:
    for keyword in node.keywords:
        if keyword.arg != "functions":
            continue
        return _literal_collection_contains(keyword.value, "sum")
    return False


def _literal_collection_contains(node: ast.AST, expected: str) -> bool:
    if isinstance(node, ast.Constant):
        return str(node.value) == expected
    if isinstance(node, (ast.Tuple, ast.List, ast.Set)):
        return any(_literal_collection_contains(item, expected) for item in node.elts)
    return False


def _node_constructs_rolling_sum_column(node: ast.AST) -> bool:
    if isinstance(node, ast.Subscript):
        pattern = _string_pattern_from_expr(node.slice, {})
        normalized = (pattern or "").lower()
        return "rolling" in normalized and "sum" in normalized
    if isinstance(node, (ast.Tuple, ast.List)):
        return any(_node_constructs_rolling_sum_column(item) for item in node.elts)
    return False


def _node_constructs_named_feature(node: ast.AST, tokens: list[str]) -> bool:
    if isinstance(node, ast.Subscript):
        pattern = _string_pattern_from_expr(node.slice, {})
        lowered = (pattern or "").lower()
        return any(token in lowered for token in tokens)
    if isinstance(node, (ast.Tuple, ast.List)):
        return any(_node_constructs_named_feature(item, tokens) for item in node.elts)
    return False


def _call_consumes_sample_weight(node: ast.Call) -> bool:
    return any(keyword.arg == "sample_weight" for keyword in node.keywords)


def _call_name_matches_any(call_name: str, tokens: list[str]) -> bool:
    lowered = call_name.lower()
    return bool(lowered) and any(token in lowered for token in tokens)


def _contains_any_token(value: str, lowered_tokens: list[str]) -> bool:
    lowered = value.lower()
    return any(token and token in lowered for token in lowered_tokens)


def _node_constructs_feature_column(
    node: ast.AST,
    lowered_tokens: list[str],
    feature_column_vars: set[str] | None = None,
    string_bindings: dict[str, str] | None = None,
) -> bool:
    if isinstance(node, ast.Subscript):
        return _subscript_uses_feature_name(node, lowered_tokens, feature_column_vars or set(), string_bindings or {})
    if isinstance(node, (ast.Tuple, ast.List)):
        return any(_node_constructs_feature_column(item, lowered_tokens, feature_column_vars, string_bindings) for item in node.elts)
    return False


def _subscript_uses_feature_name(
    node: ast.Subscript,
    lowered_tokens: list[str],
    feature_column_vars: set[str],
    string_bindings: dict[str, str],
) -> bool:
    return _expr_mentions_feature_token(node.slice, lowered_tokens, feature_column_vars, string_bindings)


def _expr_mentions_feature_token(
    node: ast.AST,
    lowered_tokens: list[str],
    feature_column_vars: set[str],
    string_bindings: dict[str, str],
) -> bool:
    pattern = _string_pattern_from_expr(node, string_bindings)
    if pattern is not None and _contains_any_token(pattern, lowered_tokens):
        return True
    if isinstance(node, ast.Name):
        return node.id in feature_column_vars
    return any(_expr_mentions_feature_token(child, lowered_tokens, feature_column_vars, string_bindings) for child in ast.iter_child_nodes(node))


def _string_pattern_from_expr(node: ast.AST, string_bindings: dict[str, str]) -> str | None:
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    if isinstance(node, ast.Name):
        return string_bindings.get(node.id)
    if isinstance(node, ast.JoinedStr):
        parts: list[str] = []
        for value in node.values:
            if isinstance(value, ast.Constant) and isinstance(value.value, str):
                parts.append(value.value)
            elif isinstance(value, ast.FormattedValue):
                parts.append(_string_pattern_from_expr(value.value, string_bindings) or "{}")
        return "".join(parts)
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
        left = _string_pattern_from_expr(node.left, string_bindings)
        right = _string_pattern_from_expr(node.right, string_bindings)
        if left is not None and right is not None:
            return left + right
    return None


def _node_snippet(node: ast.AST) -> str:
    try:
        return ast.unparse(node)
    except Exception:
        return node.__class__.__name__


def _call_name(node: ast.AST) -> str:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        parent = _call_name(node.value)
        return f"{parent}.{node.attr}" if parent else node.attr
    return ""


def _metadata_or_argparse_line(line: str) -> bool:
    if not line or line.startswith("#"):
        return True
    metadata_markers = [
        "COMBOSCOPE_ARG_OVERRIDES",
        "COMBOSCOPE_FEATURE_CHANGES",
        "COMBOSCOPE_ORIGINAL_EXPERIMENT_DIR",
        "COMBOSCOPE_ORIGINAL_PYTHON_PACKAGE_DIR",
        "_comboscope",
        "setattr(args, key, value)",
        "args.",
        "add_argument(",
    ]
    return any(marker in line for marker in metadata_markers)


def _evidence(path: Path, lineno: int, snippet: str) -> dict[str, Any]:
    return {"path": path.as_posix(), "line": int(lineno), "snippet": snippet}


def _dedupe_evidence(evidence: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen: set[tuple[str, int, str, str]] = set()
    deduped: list[dict[str, Any]] = []
    for item in evidence:
        key = (
            str(item.get("kind", "")),
            int(item.get("line", 0)),
            str(item.get("path", "")),
            str(item.get("snippet", "")),
        )
        if key in seen:
            continue
        seen.add(key)
        deduped.append(item)
    return deduped


def _feature_application_failure_message(audit: dict[str, Any]) -> str:
    lines = [
        "feature application audit failed",
        f"audit_path: {audit.get('audit_path')}",
        f"reason: {audit.get('reason')}",
    ]
    for feature in audit.get("features", []):
        if feature.get("applied"):
            continue
        lines.append(
            "- "
            + str(feature.get("feature_name"))
            + ": "
            + str(feature.get("failure_reason") or "no executable evidence")
        )
    return "\n".join(lines) + "\n"


def _agent2_code_generation_failure_message(modification_status: dict[str, Any], audit: dict[str, Any]) -> str:
    lines = [
        "Agent2 code generation failed",
        f"reason: {modification_status.get('agent2_code_generation_failure_reason') or 'no verified code edits were produced'}",
        f"failure_categories: {modification_status.get('agent2_failure_categories', [])}",
        f"code_modification_path: {modification_status.get('agent2_code_modification_path')}",
        f"source_locator_path: {modification_status.get('agent2_source_locator_path')}",
        f"audit_path: {audit.get('audit_path')}",
    ]
    for rejection in modification_status.get("agent2_normalized_rejections", [])[:5]:
        lines.append(
            "- rejected "
            + str(rejection.get("category") or "unknown")
            + ": "
            + str(rejection.get("reason") or "")
        )
    for feature in audit.get("features", []):
        if feature.get("applied"):
            continue
        lines.append(
            "- "
            + str(feature.get("feature_name"))
            + ": "
            + str(feature.get("failure_reason") or "no executable evidence")
        )
    return "\n".join(lines) + "\n"


def _existing_feature_resolution(code_dir: Path, wrapper: Path, plan: dict[str, Any]) -> dict[str, Any] | None:
    changes = [change for change in plan.get("changes", []) if isinstance(change, dict)]
    if not changes or _entrypoint_unresolved_global_calls(wrapper):
        return None
    code_files = sorted(path for path in code_dir.glob("*.py") if path.is_file())
    if wrapper.exists() and wrapper not in code_files:
        code_files.insert(0, wrapper)

    cli_features: list[dict[str, Any]] = []
    always_on_features: list[dict[str, Any]] = []
    for change in changes:
        requested_cli_args = _change_cli_args(change)
        feature = _audit_single_feature_application(change, code_files, wrapper)
        _attach_entrypoint_reachability(feature, wrapper, code_files)
        _attach_feature_activation_checks(change, feature, wrapper, code_files, train_command=None)
        if feature.get("applied") and _feature_has_reachable_cli_consumption(feature):
            cli_features.append(feature)
            continue
        if feature.get("failure_category") in {"conditional_cli_not_wired", "guardrail_not_activated"}:
            return None

        if not requested_cli_args:
            return None
        existing_change = dict(change)
        existing_change.pop("cli_args", None)
        existing_change.pop("cli_flag", None)
        existing_feature = _audit_single_feature_application(existing_change, code_files, wrapper)
        _attach_entrypoint_reachability(existing_feature, wrapper, code_files)
        _attach_feature_activation_checks(existing_change, existing_feature, wrapper, code_files, train_command=None)
        if existing_feature.get("applied"):
            existing_feature["ignored_requested_cli_args"] = requested_cli_args
            always_on_features.append(existing_feature)
            continue
        return None

    feature_names = [
        str(feature.get("feature_slug") or feature.get("feature_name") or "feature")
        for feature in [*cli_features, *always_on_features]
    ]
    if always_on_features:
        return {
            "source": "existing_code_feature",
            "modified_files": [],
            "rejected_files": [],
            "fallback_used": False,
            "notes": [
                "Requested feature already has reachable executable evidence in copied code; no code edit is needed: "
                + ", ".join(list(dict.fromkeys(feature_names)))
            ],
            "accepted_schema": "existing_code_feature",
            "code_generation_success": True,
            "feature_code_success": True,
            "code_generation_failure_reason": "",
            "failure_categories": [],
            "normalized_rejections": [],
            "repair_guidance_history": [],
            "train_py_changed_reason": "",
            "train_py_unchanged_reason": (
                "train.py already reaches the requested feature path; unsupported feature CLI args can be removed from train_command."
            ),
            "entrypoint_import_chain_checked": True,
            "feature_resolution": "existing_always_on_feature",
            "command_contract_status": "needs_repair",
            "existing_feature_audit": {"success": True, "features": always_on_features},
        }
    if cli_features:
        return {
            "source": "existing_cli_fast_path",
            "modified_files": [],
            "rejected_files": [],
            "fallback_used": False,
            "notes": [
                "Enabled already parsed and consumed CLI feature path without code edits: "
                + ", ".join(list(dict.fromkeys(feature_names)))
            ],
            "accepted_schema": "train_command_args",
            "code_generation_success": True,
            "feature_code_success": True,
            "code_generation_failure_reason": "",
            "failure_categories": [],
            "normalized_rejections": [],
            "repair_guidance_history": [],
            "train_py_changed_reason": "",
            "train_py_unchanged_reason": "train.py already exposes, consumes, and reaches the requested feature template through existing CLI wiring.",
            "entrypoint_import_chain_checked": True,
            "feature_resolution": "existing_cli_feature",
            "command_contract_status": "valid",
            "existing_feature_audit": {"success": True, "features": cli_features},
        }
    return None


def _feature_has_reachable_cli_consumption(feature: dict[str, Any]) -> bool:
    if not feature.get("cli_args"):
        return False
    if not feature.get("cli_contract_success"):
        return False
    reachability = feature.get("cli_consumption_reachability") or {}
    return bool(reachability.get("reachable", False))


def _plan_can_run_as_cli_only_variation(plan: dict[str, Any], code_dir: Path, wrapper: Path) -> bool:
    validation_categories = {
        str(item.get("category") or "")
        for item in plan.get("plan_validation", [])
        if isinstance(item, dict)
    }
    if "duplicate_existing_cli_default" not in validation_categories:
        return False
    cli_args = [item for item in _feature_cli_args(plan) if item.startswith("--")]
    if not cli_args:
        return False
    wrapper_text = wrapper.read_text(encoding="utf-8", errors="ignore") if wrapper.exists() else ""
    parsed_flags = [_supported_arg_spelling(wrapper_text, arg) for arg in cli_args]
    if not parsed_flags or any(flag is None for flag in parsed_flags):
        return False
    cli_dests = _feature_cli_dest_names(cli_args)
    if not cli_dests:
        return False
    consumed: set[str] = set()
    for path in sorted(code_dir.glob("*.py")):
        for item in _arg_usage_evidence(path, cli_dests):
            snippet = str(item.get("snippet") or "")
            for dest in cli_dests:
                if dest in snippet:
                    consumed.add(dest)
    if not set(cli_dests).issubset(consumed):
        return False
    code_files = sorted(path for path in code_dir.glob("*.py") if path.is_file())
    if wrapper.exists() and wrapper not in code_files:
        code_files.insert(0, wrapper)
    for change in [change for change in plan.get("changes", []) if isinstance(change, dict)]:
        feature = _audit_single_feature_application(change, code_files, wrapper)
        _attach_entrypoint_reachability(feature, wrapper, code_files)
        _attach_feature_activation_checks(change, feature, wrapper, code_files, train_command=None)
        if not feature.get("applied"):
            return False
    return True


def _modify_trial_code_with_agent2(
    *,
    code_dir: Path,
    wrapper: Path,
    fallback_source: str,
    plan: dict[str, Any],
    execution_plan: dict[str, Any],
    resource_context: dict[str, Any],
    llm_client: Any | None,
) -> dict[str, Any]:
    editable_files = _editable_code_files(code_dir, wrapper)
    raw_output_dir = code_dir.parent / "audit" / "agent2_llm_raw"
    locator = _build_agent2_source_locator(code_dir, plan, execution_plan, wrapper)
    (code_dir / "agent2_source_locator.yaml").write_text(
        yaml.safe_dump(locator, allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )
    code_index = _build_agent2_code_index(code_dir, locator, plan, execution_plan, wrapper)
    (code_dir / "agent2_code_index.yaml").write_text(
        yaml.safe_dump(code_index, allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )
    base_record = {
        "source": "llm_required",
        "modified_files": [],
        "rejected_files": [],
        "fallback_used": False,
        "editable_files": sorted(editable_files),
        "notes": [],
        "source_locator_path": (code_dir / "agent2_source_locator.yaml").as_posix(),
        "code_index_path": (code_dir / "agent2_code_index.yaml").as_posix(),
        "attempts": [],
        "code_generation_success": False,
        "feature_code_success": False,
        "code_generation_failure_reason": "",
        "feature_resolution": "needs_code_patch",
        "command_contract_status": "unknown",
    }
    existing_resolution = _existing_feature_resolution(code_dir, wrapper, plan)
    if existing_resolution is not None:
        return {**base_record, **existing_resolution}
    if _plan_can_run_as_cli_only_variation(plan, code_dir, wrapper):
        return {
            **base_record,
            "source": "plan_cli_args",
            "notes": ["Source already supports the requested feature variation through existing CLI arguments."],
            "code_generation_success": True,
            "feature_code_success": True,
            "code_generation_failure_reason": "",
            "accepted_schema": "train_command_args",
            "train_py_changed_reason": "",
            "train_py_unchanged_reason": "train.py already exposes and consumes the requested feature variation through existing CLI arguments.",
            "entrypoint_import_chain_checked": True,
            "feature_resolution": "existing_cli_feature",
            "command_contract_status": "valid",
        }
    cli_fast_path = _existing_cli_fast_path(code_dir, wrapper, plan)
    if cli_fast_path is not None:
        return {**base_record, **cli_fast_path}
    if llm_client is None:
        return {**base_record, "code_generation_failure_reason": "Agent2 LLM client is required"}
    if callable(getattr(llm_client, "available", None)) and not llm_client.available():
        return {**base_record, "code_generation_failure_reason": "Agent2 LLM client is unavailable"}

    attempts: list[dict[str, Any]] = []
    select_call, selected_tasks = _select_agent2_edit_tasks_with_llm(
        llm_client=llm_client,
        raw_output_dir=raw_output_dir,
        plan=plan,
        execution_plan=execution_plan,
        locator=locator,
        code_index=code_index,
        code_dir=code_dir,
    )
    attempts.append(select_call)
    source_slices = _build_agent2_source_slices(code_dir, locator, selected_tasks, plan)

    last_failure = "LLM did not produce verified code edits."
    rejected_files: list[dict[str, str]] = []
    failure_history: list[dict[str, Any]] = []
    notes: list[str] = []
    if select_call.get("local_fallback_selected_tasks"):
        notes.append("SelectEditTask did not return usable edit_tasks; local locator fallback selected targets.")

    def accept_package(package: dict[str, Any], stage_result: dict[str, Any]) -> dict[str, Any]:
        _copy_staged_modified_files(stage_result["staging_dir"], code_dir, stage_result["modified_files"])
        _cleanup_staging_dir(stage_result["staging_dir"])
        return {
            "source": "llm_edits" if package.get("edits") else "llm_multi_file",
            "modified_files": sorted(stage_result["modified_files"]),
            "modified_functions": sorted(stage_result.get("modified_functions", [])),
            "rejected_files": rejected_files,
            "fallback_used": False,
            "editable_files": sorted(editable_files),
            "notes": notes,
            "source_locator_path": (code_dir / "agent2_source_locator.yaml").as_posix(),
            "code_index_path": (code_dir / "agent2_code_index.yaml").as_posix(),
            "selected_edit_tasks": selected_tasks,
            "attempts": attempts,
            "accepted_schema": "edits" if package.get("edits") else "files",
            "code_generation_success": True,
            "feature_code_success": True,
            "code_generation_failure_reason": "",
            "failure_categories": [],
            "normalized_rejections": [],
            "repair_guidance_history": [_sanitize_agent2_failure(item) for item in failure_history],
            **_agent2_train_py_reason_fields(package, stage_result["modified_files"]),
        }

    for attempt_index in range(1, AGENT2_CODEGEN_MAX_ATTEMPTS + 1):
        prompt = _agent2_slice_codegen_prompt(
            plan=plan,
            execution_plan=execution_plan,
            locator=locator,
            code_index=code_index,
            source_slices=source_slices,
            selected_tasks=selected_tasks,
            repair_guidance=_agent2_repair_guidance(failure_history, locator, plan) if failure_history else {},
        )
        result_info = _call_agent2_llm(
            llm_client,
            "你是 ComboScope Agent2。请只返回 YAML edits 包，不要 Markdown，不要解释。",
            prompt,
            step="GenerateCodeEdits",
            attempt_index=attempt_index,
            stream=False,
            raw_output_dir=raw_output_dir,
        )
        attempts.append(result_info)
        package = _parse_agent2_code_package(result_info.get("content", ""))
        if not package:
            last_failure = result_info.get("error") or "LLM did not return a valid YAML edits/files package."
            stage_result = _attach_agent2_failure_context(
                {
                    "success": False,
                    "failure_reason": last_failure,
                    "rejected_files": [{"path": "", "reason": last_failure}],
                }
            )
            failure_history.append(stage_result)
        else:
            stage_result = _stage_agent2_code_package(
                code_dir=code_dir,
                wrapper=wrapper,
                plan=plan,
                execution_plan=execution_plan,
                resource_context=resource_context,
                editable_files=editable_files,
                package=package,
                allow_full_file_replacement=False,
                accepted_edit_types=set(AGENT2_FIRST_PASS_EDIT_TYPES),
            )
            rejected_files.extend(stage_result.get("rejected_files", []))
            notes.extend(_package_notes(package))
            if stage_result.get("success"):
                return accept_package(package, stage_result)
            last_failure = stage_result.get("failure_reason") or last_failure
            failure_history.append(stage_result)
            _cleanup_staging_dir(stage_result.get("staging_dir", code_dir / "_agent2_staging"))

        repair_stage_result = stage_result
        for repair_attempt_index in range(1, AGENT2_REPAIR_MAX_ATTEMPTS + 1):
            repair_slices = _build_agent2_repair_source_slices(code_dir, locator, selected_tasks, stage_result, plan)
            repair_prompt = _agent2_slice_repair_prompt(
                plan=plan,
                execution_plan=execution_plan,
                locator=locator,
                code_index=code_index,
                source_slices=repair_slices,
                stage_result=stage_result,
                failure_history=failure_history,
            )
            repair_info = _call_agent2_llm(
                llm_client,
                "你是 ComboScope Agent2。请只返回修复后的 YAML edits 包。",
                repair_prompt,
                step="RepairCodeEdits",
                attempt_index=repair_attempt_index,
                stream=False,
                raw_output_dir=raw_output_dir,
            )
            attempts.append(repair_info)
            repair_package = _parse_agent2_code_package(repair_info.get("content", ""))
            if not repair_package:
                last_failure = repair_info.get("error") or "LLM did not return a valid YAML edits/files package."
                repair_failure = _attach_agent2_failure_context(
                    {
                        "success": False,
                        "failure_reason": last_failure,
                        "rejected_files": [{"path": "", "reason": repair_info.get("error") or "invalid YAML edits/files package"}],
                    }
                )
                failure_history.append(repair_failure)
                stage_result = repair_failure
                continue
            repair_stage_result = _stage_agent2_code_package(
                code_dir=code_dir,
                wrapper=wrapper,
                plan=plan,
                execution_plan=execution_plan,
                resource_context=resource_context,
                editable_files=editable_files,
                package=repair_package,
                allow_full_file_replacement=False,
            )
            rejected_files.extend(repair_stage_result.get("rejected_files", []))
            notes.extend(_package_notes(repair_package))
            if repair_stage_result.get("success"):
                return accept_package(repair_package, repair_stage_result)
            last_failure = repair_stage_result.get("failure_reason") or last_failure
            failure_history.append(repair_stage_result)
            _cleanup_staging_dir(repair_stage_result.get("staging_dir", code_dir / "_agent2_staging"))
            stage_result = repair_stage_result

    return {
        **base_record,
        "rejected_files": rejected_files,
        "notes": notes or [last_failure],
        "attempts": attempts,
        "code_index_path": (code_dir / "agent2_code_index.yaml").as_posix(),
        "selected_edit_tasks": selected_tasks,
        "code_generation_success": False,
            "code_generation_failure_reason": last_failure,
            "failure_categories": _failure_categories_from_history(failure_history),
            "normalized_rejections": _normalized_rejections_from_history(failure_history),
            "repair_guidance_history": [_agent2_repair_guidance([item], locator, plan) for item in failure_history],
            "required_corrections": _required_corrections_from_history(failure_history),
        "original_signatures": _original_signatures_from_history(failure_history, locator, plan),
        "entrypoint_import_chain": _entrypoint_import_chain_from_history(failure_history),
        "failed_stage_package_summary": _failed_stage_package_summaries(failure_history),
            "train_py_changed_reason": "",
            "train_py_unchanged_reason": "",
            "entrypoint_import_chain_checked": False,
            "feature_resolution": "needs_code_patch",
            "command_contract_status": "unknown",
        }


def _existing_cli_fast_path(code_dir: Path, wrapper: Path, plan: dict[str, Any]) -> dict[str, Any] | None:
    changes = [change for change in plan.get("changes", []) if isinstance(change, dict)]
    cli_changes = [change for change in changes if any(str(item).startswith("--") for item in _change_cli_args(change))]
    if not cli_changes or len(cli_changes) != len(changes):
        return None
    if _entrypoint_unresolved_global_calls(wrapper):
        return None
    code_files = sorted(path for path in code_dir.glob("*.py") if path.is_file())
    if wrapper.exists() and wrapper not in code_files:
        code_files.insert(0, wrapper)
    enabled_features: list[str] = []
    for change in cli_changes:
        cli_args = [str(item) for item in _change_cli_args(change) if str(item).startswith("--")]
        if not cli_args:
            return None
        feature = _audit_single_feature_application(change, code_files, wrapper)
        _attach_entrypoint_reachability(feature, wrapper, code_files)
        _attach_feature_activation_checks(change, feature, wrapper, code_files, train_command=None)
        if not feature.get("applied"):
            return None
        if not any(item.get("kind") == "cli_arg_consumed" for item in feature.get("evidence", []) or []):
            return None
        enabled_features.append(str(change.get("feature_slug") or change.get("feature_name") or "cli_feature"))
    if not enabled_features:
        return None
    return {
        "source": "existing_cli_fast_path",
        "modified_files": [],
        "rejected_files": [],
        "fallback_used": False,
        "notes": [
            "Enabled already parsed and consumed CLI feature path without code edits: "
            + ", ".join(list(dict.fromkeys(enabled_features)))
        ],
        "accepted_schema": "train_command_args",
        "code_generation_success": True,
        "code_generation_failure_reason": "",
        "failure_categories": [],
        "normalized_rejections": [],
        "repair_guidance_history": [],
        "train_py_changed_reason": "",
        "train_py_unchanged_reason": "train.py already exposes, consumes, and reaches the requested feature template through existing CLI wiring.",
        "entrypoint_import_chain_checked": True,
    }


def _entrypoint_unresolved_global_calls(wrapper: Path) -> list[str]:
    if not wrapper.exists():
        return []
    try:
        tree = ast.parse(wrapper.read_text(encoding="utf-8", errors="ignore"))
    except SyntaxError:
        return []
    module_names = _module_defined_or_imported_names(tree)
    builtin_names = set(dir(builtins))
    unresolved: set[str] = set()

    class Visitor(ast.NodeVisitor):
        def __init__(self) -> None:
            self.local_scopes: list[set[str]] = []

        def visit_FunctionDef(self, node: ast.FunctionDef) -> None:  # noqa: N802 - ast visitor API
            local = set(_all_param_names(node))
            for child in ast.walk(node):
                if isinstance(child, ast.Name) and isinstance(child.ctx, (ast.Store, ast.Del)):
                    local.add(child.id)
                elif isinstance(child, ast.ExceptHandler) and child.name:
                    local.add(child.name)
            self.local_scopes.append(local)
            self.generic_visit(node)
            self.local_scopes.pop()

        def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:  # noqa: N802 - ast visitor API
            self.visit_FunctionDef(node)

        def visit_Call(self, node: ast.Call) -> None:  # noqa: N802 - ast visitor API
            if isinstance(node.func, ast.Name):
                name = node.func.id
                local = self.local_scopes[-1] if self.local_scopes else set()
                if name not in local and name not in module_names and name not in builtin_names:
                    unresolved.add(name)
            self.generic_visit(node)

    Visitor().visit(tree)
    return sorted(unresolved)


def _module_defined_or_imported_names(tree: ast.AST) -> set[str]:
    names: set[str] = set()
    for node in ast.iter_child_nodes(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            names.add(node.name)
        elif isinstance(node, ast.Import):
            for alias in node.names:
                names.add(alias.asname or alias.name.split(".")[0])
        elif isinstance(node, ast.ImportFrom):
            for alias in node.names:
                if alias.name == "*":
                    continue
                names.add(alias.asname or alias.name)
        elif isinstance(node, (ast.Assign, ast.AnnAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            for target in targets:
                names.update(_assigned_name_targets(target))
    return names


def _assigned_name_targets(node: ast.AST) -> set[str]:
    if isinstance(node, ast.Name):
        return {node.id}
    if isinstance(node, (ast.Tuple, ast.List)):
        names: set[str] = set()
        for item in node.elts:
            names.update(_assigned_name_targets(item))
        return names
    return set()


def _change_matches_type(change: dict[str, Any], expected: str) -> bool:
    values = [
        str(change.get("feature_type") or ""),
        str(change.get("feature_slug") or ""),
        str(change.get("feature_name") or ""),
        str(change.get("construction") or ""),
        " ".join(str(item) for item in change.get("audit_tokens", []) or []),
    ]
    combined = " ".join(values).lower()
    return expected.lower() in combined


def _feature_cols_assignment_line(lines: list[str], start: int, end: int) -> int | None:
    for index in range(start, end):
        stripped = lines[index].strip()
        if stripped.startswith("feature_cols") and "=" in stripped:
            return index
    return None


def _first_return_line(lines: list[str], start: int, end: int) -> int | None:
    for index in range(start, end):
        if lines[index].lstrip().startswith("return "):
            return index
    return None


def _python_call_block_end(lines: list[str], start: int) -> int:
    balance = 0
    for index in range(start, len(lines)):
        line = lines[index]
        balance += line.count("(") - line.count(")")
        if index > start and balance <= 0:
            return index
        if index == start and balance <= 0:
            return index
    return start


def _call_block_keyword_line(lines: list[str], start: int, end: int, keyword: str) -> int | None:
    needle = f"{keyword}="
    for index in range(start, end + 1):
        if needle in lines[index]:
            return index
    return None


def _build_agent2_source_locator(code_dir: Path, plan: dict[str, Any], execution_plan: dict[str, Any], wrapper: Path) -> dict[str, Any]:
    feature_changes = [change for change in plan.get("changes", []) if isinstance(change, dict)]
    feature_tokens = {
        token.lower()
        for change in feature_changes
        for token in _change_feature_tokens(change)
    }
    field_tokens = {
        str(field).lower()
        for change in feature_changes
        for field in change.get("field_sources", []) or []
    }
    cli_dests = {
        dest.lower()
        for change in feature_changes
        for dest in _feature_cli_dest_names([str(item) for item in _change_cli_args(change) if str(item).startswith("--")])
    }
    plan_locations = {
        Path(str(location)).name
        for change in feature_changes
        for location in change.get("code_locations", []) or []
        if str(location).endswith(".py")
    }
    preferred_functions = {
        "parse_args",
        "build_features",
        "run_online_pipeline",
        "train_backtest_and_refit_model",
        "main",
    }
    files: list[dict[str, Any]] = []
    function_index_by_file: dict[str, list[dict[str, Any]]] = {}
    for path in sorted(code_dir.glob("*.py")):
        if not path.is_file():
            continue
        source = path.read_text(encoding="utf-8", errors="ignore")
        functions = _python_function_index(path, source, code_dir)
        function_index_by_file[path.name] = functions
        candidate_functions: list[dict[str, Any]] = []
        for info in functions:
            lowered_name = info["name"].lower()
            snippet = _slice_lines(source, int(info["line"]), int(info["end_line"]))
            lowered_snippet = snippet.lower()
            reasons: list[str] = []
            if info["name"] in preferred_functions:
                reasons.append("known_feature_or_training_path")
            if any(token and token in lowered_name for token in feature_tokens):
                reasons.append("function_name_matches_feature")
            if any(token and token in lowered_snippet for token in feature_tokens):
                reasons.append("function_body_mentions_feature")
            if any(token and token in lowered_snippet for token in field_tokens):
                reasons.append("function_body_mentions_plan_fields")
            if any(token and token in lowered_snippet for token in cli_dests):
                reasons.append("function_body_mentions_cli_dest")
            if "build_features" in lowered_snippet and info["name"] not in {"build_features"}:
                reasons.append("calls_build_features")
            if "feature_cols" in lowered_snippet:
                reasons.append("feature_cols_path")
            if reasons:
                candidate_functions.append({**info, "reasons": list(dict.fromkeys(reasons))})
        insert_points = _infer_insert_points(path, source)
        files.append(
            {
                "path": path.name,
                "is_plan_location": path.name in plan_locations or path.name == wrapper.name,
                "function_count": len(functions),
                "candidate_functions": candidate_functions[:12],
                "insert_points": insert_points,
                "outline": [{"name": item["name"], "line": item["line"], "end_line": item["end_line"]} for item in functions[:80]],
            }
        )
    return {
        "trial_code_dir": code_dir.as_posix(),
        "feature_changes": feature_changes,
        "agent2_execution_plan_mode": execution_plan.get("mode"),
        "candidate_files": files,
        "feature_cols_builders": _locator_feature_cols_builders(files),
        "train_entrypoints": _locator_train_entrypoints(function_index_by_file, wrapper),
        "reachable_dependency_functions": _locator_reachable_dependency_functions(code_dir, wrapper),
        "signature_constraints": _locator_signature_constraints(function_index_by_file),
        "existing_cli_args": _locator_existing_cli_args(code_dir),
        "cli_consumption_evidence": _locator_cli_consumption_evidence(code_dir, feature_changes),
        "fast_path_candidates": _locator_fast_path_candidates(code_dir, feature_changes),
        "notes": [
            "Locator is deterministic and only points at copied trial code files.",
            "Agent2 must modify trial code, not original experiment source.",
        ],
    }


def _locator_feature_cols_builders(files: list[dict[str, Any]]) -> list[dict[str, Any]]:
    builders: list[dict[str, Any]] = []
    for file_info in files:
        for func in file_info.get("candidate_functions", []) or []:
            reasons = set(func.get("reasons", []) or [])
            if func.get("name") == "build_features" or "feature_cols_path" in reasons:
                builders.append(
                    {
                        "path": file_info.get("path"),
                        "function": func.get("name"),
                        "line": func.get("line"),
                        "signature": func.get("signature"),
                        "reasons": sorted(reasons),
                    }
                )
    return builders


def _locator_train_entrypoints(function_index_by_file: dict[str, list[dict[str, Any]]], wrapper: Path) -> list[dict[str, Any]]:
    names = {"main", "run_online_pipeline", "train_backtest_and_refit_model", "train_package_model"}
    entrypoints: list[dict[str, Any]] = []
    for path, functions in function_index_by_file.items():
        for func in functions:
            if path == wrapper.name or func.get("name") in names:
                if func.get("name") in names or path == wrapper.name:
                    entrypoints.append(
                        {
                            "path": path,
                            "function": func.get("name"),
                            "line": func.get("line"),
                            "signature": func.get("signature"),
                        }
                    )
    return entrypoints[:20]


def _locator_reachable_dependency_functions(code_dir: Path, wrapper: Path) -> list[dict[str, Any]]:
    reachable: list[dict[str, Any]] = []
    for path in sorted(code_dir.glob("*.py")):
        if path.name == wrapper.name or not path.is_file():
            continue
        funcs = sorted(_reachable_dependency_functions_from_wrapper(wrapper, path))
        reachable.append({"path": path.name, "reachable_functions": funcs})
    return reachable


def _locator_signature_constraints(function_index_by_file: dict[str, list[dict[str, Any]]]) -> list[dict[str, Any]]:
    constraints: list[dict[str, Any]] = []
    important_names = {"parse_args", "build_features", "train_package_model", "run_online_pipeline", "main"}
    for path, functions in function_index_by_file.items():
        for func in functions:
            if func.get("name") in important_names or func.get("call_keywords") or func.get("return_arities"):
                constraints.append(
                    {
                        "path": path,
                        "function": func.get("name"),
                        "signature": func.get("signature"),
                        "accepted_keywords": func.get("accepted_keywords", []),
                        "call_keywords": func.get("call_keywords", []),
                        "return_arities": func.get("return_arities", []),
                    }
                )
    return constraints[:40]


def _locator_existing_cli_args(code_dir: Path) -> list[dict[str, Any]]:
    args: list[dict[str, Any]] = []
    for path in sorted(code_dir.glob("*.py")):
        if not path.is_file():
            continue
        args.extend({"path": path.name, **item} for item in _argparse_args_index(path.read_text(encoding="utf-8", errors="ignore")))
    return args[:80]


def _argparse_args_index(source: str) -> list[dict[str, Any]]:
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return []
    args: list[dict[str, Any]] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        if _call_name(node.func).split(".")[-1] != "add_argument":
            continue
        flags = [
            str(arg.value)
            for arg in node.args
            if isinstance(arg, ast.Constant) and isinstance(arg.value, str) and str(arg.value).startswith("--")
        ]
        if not flags:
            continue
        default = None
        choices: list[str] = []
        for keyword in node.keywords:
            if keyword.arg == "default":
                try:
                    default = ast.literal_eval(keyword.value)
                except Exception:
                    default = ast.unparse(keyword.value)
            elif keyword.arg == "choices":
                choices = sorted(_literal_choices_from_call(node))
        args.append({"flags": flags, "dest_names": _feature_cli_dest_names(flags), "default": default, "choices": choices, "line": node.lineno})
    return args


def _locator_cli_consumption_evidence(code_dir: Path, feature_changes: list[dict[str, Any]]) -> list[dict[str, Any]]:
    cli_dests = [
        dest
        for change in feature_changes
        for dest in _feature_cli_dest_names([str(item) for item in _change_cli_args(change) if str(item).startswith("--")])
    ]
    if not cli_dests:
        return []
    evidence: list[dict[str, Any]] = []
    for path in sorted(code_dir.glob("*.py")):
        if not path.is_file():
            continue
        evidence.extend(_arg_usage_evidence(path, cli_dests))
    return evidence[:40]


def _locator_fast_path_candidates(code_dir: Path, feature_changes: list[dict[str, Any]]) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    for change in feature_changes:
        if _is_rolling_sum_change(change):
            candidates.append(
                {
                    "feature_name": change.get("feature_name"),
                    "fast_path": "rolling_stat_007_guardrail",
                    "available": _code_dir_mentions(code_dir, ["generate_rolling_features", "build_features"]),
                }
            )
        if any(str(item).startswith("--") for item in _change_cli_args(change)):
            candidates.append(
                {
                    "feature_name": change.get("feature_name"),
                    "fast_path": "existing_cli",
                    "available": bool(_locator_cli_consumption_evidence(code_dir, [change])),
                }
            )
        if _is_sample_weight_wiring_change(change):
            candidates.append(
                {
                    "feature_name": change.get("feature_name"),
                    "fast_path": "sample_weight_wiring",
                    "available": _code_dir_mentions(code_dir, ["sample_weight", "fit"]),
                }
            )
    return candidates


def _code_dir_mentions(code_dir: Path, tokens: list[str]) -> bool:
    lowered_tokens = [token.lower() for token in tokens]
    for path in sorted(code_dir.glob("*.py")):
        if not path.is_file():
            continue
        text = path.read_text(encoding="utf-8", errors="ignore").lower()
        if all(token in text for token in lowered_tokens):
            return True
    return False


def _python_function_index(path: Path, source: str, code_dir: Path | None = None) -> list[dict[str, Any]]:
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return []
    functions: list[dict[str, Any]] = []
    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        start_line = min([node.lineno] + [decorator.lineno for decorator in node.decorator_list])
        signature = _function_signature_source(node)
        functions.append(
            {
                "name": node.name,
                "line": int(start_line),
                "end_line": int(getattr(node, "end_lineno", node.lineno)),
                "calls": sorted({_call_name(child.func) for child in ast.walk(node) if isinstance(child, ast.Call) and _call_name(child.func)})[:30],
                "signature": signature,
                "accepted_keywords": _accepted_keyword_params(node),
                "call_keywords": _call_keywords_for_function(code_dir, node.name) if code_dir is not None else [],
                "return_arities": sorted(_tuple_return_arities(node)),
            }
        )
    return sorted(functions, key=lambda item: (item["line"], item["name"]))


def _function_signature_source(node: ast.FunctionDef | ast.AsyncFunctionDef) -> str:
    args = ast.unparse(node.args)
    prefix = "async def" if isinstance(node, ast.AsyncFunctionDef) else "def"
    returns = f" -> {ast.unparse(node.returns)}" if node.returns is not None else ""
    return f"{prefix} {node.name}({args}){returns}"


def _infer_insert_points(path: Path, source: str) -> list[dict[str, Any]]:
    points: list[dict[str, Any]] = []
    for index, line in enumerate(source.splitlines(), start=1):
        stripped = line.strip()
        if "add_argument(" in stripped:
            points.append({"line": index, "kind": "argparse_argument", "purpose": "add or confirm feature CLI flag"})
        if "feature_cols" in stripped and ("=" in stripped or "return" in stripped):
            points.append({"line": index, "kind": "feature_cols", "purpose": "ensure constructed feature reaches model feature columns"})
        if "build_features(" in stripped:
            points.append({"line": index, "kind": "build_features_call", "purpose": "ensure train path uses modified feature builder"})
        if "train" in stripped.lower() and "(" in stripped:
            points.append({"line": index, "kind": "training_entrypoint", "purpose": "ensure the training path uses the feature"})
    return points[:40]


def _build_agent2_code_index(
    code_dir: Path,
    locator: dict[str, Any],
    plan: dict[str, Any],
    execution_plan: dict[str, Any] | None = None,
    wrapper: Path | None = None,
) -> dict[str, Any]:
    files: list[dict[str, Any]] = []
    total_source_chars = 0
    for path in sorted(code_dir.glob("*.py")):
        if not path.exists() or not path.is_file():
            continue
        source = path.read_text(encoding="utf-8", errors="ignore")
        total_source_chars += len(source)
        file_info = next(
            (item for item in locator.get("candidate_files", []) if str(item.get("path") or "") == path.name),
            {},
        )
        files.append(
            {
                "path": path.name,
                "source_chars": len(source),
                "imports": _python_imports_index(source),
                "functions": _python_function_index(path, source, code_dir),
                "call_graph": _python_function_call_graph(source),
                "argparse_args": _argparse_args_index(source),
                "output_write_hints": _output_write_hints(source),
                "outline": file_info.get("outline", []),
                "insert_points": file_info.get("insert_points", []),
                "candidate_functions": file_info.get("candidate_functions", []),
            }
        )
    return {
        "index_policy": {
            "full_trial_python_sources": False,
            "source_fields_omitted": True,
            "do_not_modify_original_experiment_source": True,
        },
        "feature_changes": plan.get("changes", []),
        "agent2_execution_plan_mode": (execution_plan or {}).get("mode"),
        "wrapper": (wrapper.name if wrapper else None),
        "total_source_chars": total_source_chars,
        "train_entrypoints": locator.get("train_entrypoints", []),
        "reachable_dependency_functions": locator.get("reachable_dependency_functions", []),
        "signature_constraints": locator.get("signature_constraints", []),
        "existing_cli_args": locator.get("existing_cli_args", []),
        "cli_consumption_evidence": locator.get("cli_consumption_evidence", []),
        "fast_path_candidates": locator.get("fast_path_candidates", []),
        "files": files,
    }


def _build_agent2_context_pack(code_dir: Path, locator: dict[str, Any], plan: dict[str, Any], *, compact: bool) -> dict[str, Any]:
    code_index = _build_agent2_code_index(code_dir, locator, plan)
    return {
        "prompt_policy": {
            "compact": True,
            "max_prompt_tokens": AGENT2_PROMPT_TOKEN_LIMIT,
            "full_trial_python_sources": False,
            "do_not_modify_original_experiment_source": True,
        },
        "feature_changes": code_index.get("feature_changes", []),
        "total_source_chars": code_index.get("total_source_chars", 0),
        "train_entrypoints": code_index.get("train_entrypoints", []),
        "reachable_dependency_functions": code_index.get("reachable_dependency_functions", []),
        "signature_constraints": code_index.get("signature_constraints", []),
        "existing_cli_args": code_index.get("existing_cli_args", []),
        "cli_consumption_evidence": code_index.get("cli_consumption_evidence", []),
        "fast_path_candidates": code_index.get("fast_path_candidates", []),
        "files": code_index.get("files", []),
    }


def _build_agent2_source_slices(
    code_dir: Path,
    locator: dict[str, Any],
    selected_tasks: list[dict[str, Any]] | None,
    plan: dict[str, Any],
    *,
    context_lines: int = AGENT2_SOURCE_SLICE_CONTEXT_LINES,
    max_slices: int = AGENT2_MAX_SOURCE_SLICES,
) -> dict[str, Any]:
    tasks = _normalize_agent2_selected_tasks(selected_tasks or [], code_dir)
    if not tasks:
        tasks = _agent2_default_edit_tasks(locator, code_dir)
    slices: list[dict[str, Any]] = []
    seen: set[tuple[str, int, int, str]] = set()
    for task in tasks:
        if len(slices) >= max_slices:
            break
        path_name = str(task.get("path") or "")
        path = code_dir / path_name
        if not path.exists() or not path.is_file():
            continue
        source = path.read_text(encoding="utf-8", errors="ignore")
        function_name = str(task.get("function") or "")
        line = _positive_int_field(task, "line")
        source_slice = _source_slice_for_target(path, source, function_name, line, context_lines)
        if not source_slice:
            continue
        key = (
            str(source_slice.get("path") or ""),
            int(source_slice.get("start_line") or 0),
            int(source_slice.get("end_line") or 0),
            str(source_slice.get("function") or ""),
        )
        if key in seen:
            continue
        seen.add(key)
        source_slice["task_reason"] = task.get("reason", "")
        slices.append(source_slice)
    return {
        "slice_policy": {
            "full_trial_python_sources": False,
            "context_lines": context_lines,
            "max_slices": max_slices,
        },
        "selected_edit_tasks": tasks[:max_slices],
        "slices": slices,
    }


def _build_agent2_repair_source_slices(
    code_dir: Path,
    locator: dict[str, Any],
    selected_tasks: list[dict[str, Any]] | None,
    stage_result: dict[str, Any],
    plan: dict[str, Any],
) -> dict[str, Any]:
    repair_tasks = _agent2_failure_edit_tasks(stage_result)
    repair_tasks.extend(selected_tasks or [])
    return _build_agent2_source_slices(
        code_dir,
        locator,
        repair_tasks,
        plan,
        context_lines=30,
        max_slices=4,
    )


def _normalize_agent2_selected_tasks(tasks: list[dict[str, Any]], code_dir: Path) -> list[dict[str, Any]]:
    normalized: list[dict[str, Any]] = []
    editable_files = {path.name for path in code_dir.glob("*.py") if path.is_file()}
    for item in tasks:
        if not isinstance(item, dict):
            continue
        raw_path = str(item.get("path") or "")
        path_name = Path(raw_path).name if raw_path else ""
        if path_name not in editable_files:
            continue
        task = {
            "path": path_name,
            "function": str(item.get("function") or item.get("target_function") or ""),
            "reason": str(item.get("reason") or item.get("purpose") or ""),
        }
        line = _positive_int_field(item, "line") or _positive_int_field(item, "insert_after_line")
        if line:
            task["line"] = line
        normalized.append(task)
    return normalized[:3]


def _agent2_default_edit_tasks(locator: dict[str, Any], code_dir: Path) -> list[dict[str, Any]]:
    tasks: list[dict[str, Any]] = []
    for item in locator.get("feature_cols_builders", []) or []:
        tasks.append(
            {
                "path": str(item.get("path") or ""),
                "function": str(item.get("function") or ""),
                "reason": "local_locator_feature_cols_builder",
            }
        )
    for file_info in locator.get("candidate_files", []) or []:
        for func in file_info.get("candidate_functions", []) or []:
            tasks.append(
                {
                    "path": str(file_info.get("path") or ""),
                    "function": str(func.get("name") or ""),
                    "reason": "local_locator_candidate_function",
                }
            )
    for entrypoint in locator.get("train_entrypoints", []) or []:
        tasks.append(
            {
                "path": str(entrypoint.get("path") or ""),
                "function": str(entrypoint.get("function") or ""),
                "reason": "local_locator_train_entrypoint",
            }
        )
    return _normalize_agent2_selected_tasks(tasks, code_dir)


def _agent2_failure_edit_tasks(stage_result: dict[str, Any]) -> list[dict[str, Any]]:
    tasks: list[dict[str, Any]] = []
    for item in stage_result.get("normalized_rejections", []) or []:
        path = str(item.get("path") or "")
        if path:
            tasks.append({"path": path, "reason": str(item.get("category") or "previous_failure")})
    for item in stage_result.get("compile_errors", []) or []:
        tasks.append({"path": str(item.get("path") or ""), "reason": "compile_error"})
    for item in stage_result.get("contract_errors", []) or []:
        task: dict[str, Any] = {
            "path": str(item.get("path") or ""),
            "function": str(item.get("function") or ""),
            "reason": str(item.get("category") or "runtime_contract_error"),
        }
        line = _positive_int_field(item, "line")
        if line:
            task["line"] = line
        tasks.append(task)
    return tasks


def _source_slice_for_target(
    path: Path,
    source: str,
    function_name: str,
    line: int | None,
    context_lines: int,
) -> dict[str, Any] | None:
    lines = source.splitlines()
    if not lines:
        return None
    target_node: ast.FunctionDef | ast.AsyncFunctionDef | None = None
    try:
        tree = ast.parse(source)
    except SyntaxError:
        tree = None
    if tree is not None and function_name:
        target_node = _find_function_node(tree, function_name)
    if target_node is not None:
        function_start = min([target_node.lineno] + [decorator.lineno for decorator in target_node.decorator_list])
        function_end = int(getattr(target_node, "end_lineno", target_node.lineno))
        start_line = max(1, function_start - context_lines)
        end_line = min(len(lines), function_end + context_lines)
        signature = _function_signature_source(target_node)
    else:
        anchor = line or 1
        start_line = max(1, anchor - context_lines)
        end_line = min(len(lines), anchor + context_lines)
        signature = ""
    return {
        "path": path.name,
        "function": function_name,
        "signature": signature,
        "start_line": start_line,
        "end_line": end_line,
        "source_chars": len("\n".join(lines[start_line - 1 : end_line])),
        "source": "\n".join(lines[start_line - 1 : end_line]),
        "call_points": _function_call_points(source, function_name) if function_name else [],
    }


def _function_call_points(source: str, function_name: str) -> list[dict[str, Any]]:
    if not function_name:
        return []
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return []
    points: list[dict[str, Any]] = []
    lines = source.splitlines()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        called = ""
        if isinstance(node.func, ast.Name):
            called = node.func.id
        elif isinstance(node.func, ast.Attribute):
            called = node.func.attr
        if called != function_name:
            continue
        lineno = int(getattr(node, "lineno", 0))
        snippet = lines[lineno - 1].strip() if 0 < lineno <= len(lines) else ""
        points.append({"line": lineno, "snippet": snippet})
    return points[:8]


def _python_imports_index(source: str) -> list[dict[str, Any]]:
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return []
    imports: list[dict[str, Any]] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports.append(
                {
                    "line": node.lineno,
                    "type": "import",
                    "names": [alias.asname or alias.name for alias in node.names],
                }
            )
        elif isinstance(node, ast.ImportFrom):
            imports.append(
                {
                    "line": node.lineno,
                    "type": "from_import",
                    "module": node.module or "",
                    "names": [alias.asname or alias.name for alias in node.names],
                }
            )
    return imports


def _python_function_call_graph(source: str) -> dict[str, list[str]]:
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return {}
    graph: dict[str, list[str]] = {}
    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        calls = sorted(
            {
                _call_name(child.func)
                for child in ast.walk(node)
                if isinstance(child, ast.Call) and _call_name(child.func)
            }
        )
        graph[node.name] = calls[:80]
    return graph


def _output_write_hints(source: str) -> list[dict[str, Any]]:
    hints: list[dict[str, Any]] = []
    for index, line in enumerate(source.splitlines(), start=1):
        lowered = line.lower()
        if any(token in lowered for token in ("to_csv", "csv.writer", "open(", "output_dir", "real_output_dir")):
            hints.append({"line": index, "snippet": line.strip()})
    return hints[:80]


def _select_agent2_edit_tasks_with_llm(
    *,
    llm_client: Any,
    raw_output_dir: Path,
    plan: dict[str, Any],
    execution_plan: dict[str, Any],
    locator: dict[str, Any],
    code_index: dict[str, Any],
    code_dir: Path,
    attempt_index: int = 1,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    prompt = _agent2_select_edit_task_prompt(plan, execution_plan, locator, code_index)
    result_info = _call_agent2_llm(
        llm_client,
        "你是 ComboScope Agent2。请只基于本地 code_index 选择编辑目标，并只返回 YAML。",
        prompt,
        step="SelectEditTask",
        attempt_index=attempt_index,
        stream=False,
        raw_output_dir=raw_output_dir,
    )
    selected_tasks = _parse_agent2_edit_task_selection(result_info.get("content", ""))
    selected_tasks = _normalize_agent2_selected_tasks(selected_tasks, code_dir)
    if not selected_tasks:
        selected_tasks = _agent2_default_edit_tasks(locator, code_dir)
        result_info["local_fallback_selected_tasks"] = selected_tasks
    return result_info, selected_tasks


def _agent2_select_edit_task_prompt(
    plan: dict[str, Any],
    execution_plan: dict[str, Any],
    locator: dict[str, Any],
    code_index: dict[str, Any],
) -> str:
    primary_changes = _agent2_primary_feature_changes(plan)
    payload = {
        "task": "Select 1-3 copied trial code functions or line insertion points for implementing selected_feature_actions. Do not write code.",
        "required_yaml_schema": {
            "edit_tasks": [
                {
                    "path": "train.py or util.py",
                    "function": "main or build_features",
                    "line": "optional insertion line when function is not enough",
                    "reason": "why this is the smallest executable edit target",
                }
            ],
            "train_py_decision": "modify train.py or leave unchanged with import/call-chain reason",
            "notes": ["short rationale"],
        },
        "hard_boundaries": [
            "Choose only files already copied under trial code/.",
            "Use code_index outlines, signatures, insert_points, feature_cols hints, CLI args, and call graph summaries.",
            "Do not ask for full source; full function slices will be fetched locally after this step.",
            "Do not choose evaluation split, label, metric, output standardization, or protected evaluation functions.",
        ],
        "selected_feature_actions": primary_changes,
        "primary_feature_change": primary_changes[0] if primary_changes else {},
        "experiment_plan": _compact_experiment_plan(plan),
        "agent2_execution_plan": _compact_agent2_execution_plan(execution_plan),
        "source_locator": _compact_agent2_source_locator(locator),
        "code_index": _compact_agent2_code_index(code_index),
    }
    return _budgeted_agent2_yaml_prompt(payload)


def _parse_agent2_edit_task_selection(content: str) -> list[dict[str, Any]]:
    if not clean_llm_yaml_text(content):
        return []
    try:
        parsed = safe_load_yaml_mapping(content, agent="Agent2", step="SelectEditTasks")
    except ValueError:
        return []
    raw_tasks = parsed.get("edit_tasks")
    if raw_tasks is None:
        raw_tasks = parsed.get("target_functions")
    if raw_tasks is None:
        raw_tasks = parsed.get("change_plan")
    if not isinstance(raw_tasks, list):
        return []
    tasks: list[dict[str, Any]] = []
    for item in raw_tasks:
        if not isinstance(item, dict):
            continue
        functions = item.get("functions")
        if isinstance(functions, list) and functions:
            for function in functions[:3]:
                tasks.append({**item, "function": str(function)})
        else:
            tasks.append(item)
    return tasks[:3]


def _agent2_codegen_prompt(
    plan: dict[str, Any],
    execution_plan: dict[str, Any],
    locator: dict[str, Any],
    context_pack: dict[str, Any],
    llm_change_plan: str,
    *,
    repair_guidance: dict[str, Any] | None = None,
    compact: bool,
) -> str:
    code_index = context_pack.get("code_index") if isinstance(context_pack, dict) else {}
    if not isinstance(code_index, dict):
        code_index = context_pack if isinstance(context_pack, dict) else {}
    source_slices = context_pack.get("source_slices") if isinstance(context_pack, dict) else {}
    if not isinstance(source_slices, dict):
        source_slices = {"slices": []}
    return _agent2_slice_codegen_prompt(
        plan=plan,
        execution_plan=execution_plan,
        locator=locator,
        code_index=code_index,
        source_slices=source_slices,
        selected_tasks=source_slices.get("selected_edit_tasks", []),
        repair_guidance=repair_guidance or {},
    )


def _agent2_slice_codegen_prompt(
    *,
    plan: dict[str, Any],
    execution_plan: dict[str, Any],
    locator: dict[str, Any],
    code_index: dict[str, Any],
    source_slices: dict[str, Any],
    selected_tasks: list[dict[str, Any]],
    repair_guidance: dict[str, Any] | None = None,
) -> str:
    primary_changes = _agent2_primary_feature_changes(plan)
    payload = {
        "task": (
            "Implement selected_feature_actions by replacing copied trial-code functions from provided source_slices. "
            "Prefer one replace_function edit for each touched function; do not use raw line-number edits in this first pass."
        ),
        "preferred_yaml_schema": {
            "edits": [
                {
                    "path": "train.py",
                    "type": "replace_function | add_feature_column_in_function | insert_before_return | append_module_code",
                    "function": "required for replace_function/add_feature_column_in_function/insert_before_return",
                    "content": "<only the replacement function or inserted lines>",
                }
            ],
            "change_summary": "what changed and why it implements the requested feature",
            "expected_training_path": "how train.py reaches the modified code and why the current train_command should run",
            "train_command": "optional corrected argv token list if the current command must change",
            "implementation_assertions": [
                "machine-checkable statement such as util.py::build_features calls time_series_continuization before feature_cols"
            ],
            "train_py_changed_reason": "required if train.py is edited; describe wiring/default/preset/output-path change",
            "train_py_unchanged_reason": "required if train.py is not edited; explain how train.py imports/calls the changed dependency",
            "entrypoint_import_chain_checked": True,
            "notes": ["what changed"],
        },
        "valid_edit_type_examples": [
            {"path": "util.py", "type": "replace_function", "function": "build_features", "content": "def build_features(...):\n    ..."},
            {"path": "train.py", "type": "replace_function", "function": "parse_args", "content": "def parse_args(...):\n    ..."},
            {"path": "util.py", "type": "add_feature_column_in_function", "function": "build_features", "content": "feature_col = 'new_feature'\ndf[feature_col] = ..."},
            {"path": "util.py", "type": "insert_before_return", "function": "build_features", "content": "df['new_feature'] = ..."},
            {"path": "util.py", "type": "append_module_code", "content": "def helper(...):\n    ..."},
        ],
        "accepted_edit_types": AGENT2_FIRST_PASS_EDIT_TYPES,
        "forbidden_edit_types": ["", "modify_function", "insert_code", "patch", "diff", "insert_after_line", "replace_lines"],
        "forbidden_default_schema": "files",
        "guardrails": [
            "path must be basename under trial code/ only.",
            "Every edits item must include one accepted type.",
            "Each selected source_slice contains the complete target function plus nearby context; copy the original signature exactly when using replace_function.",
            "Use source_slices plus code_index summaries; do not rely on or request full trial source.",
            "First-pass edits must be function-scoped. Do not return insert_after_line, replace_lines, unified diffs, or line-number patches.",
            "If a selected function is the feature-building path, replace that function rather than splicing unindented lines into it.",
            "fast_path_candidates are hints only; do not invent new deterministic business templates from them.",
            "Make the smallest source change that is likely to pass the current train_command and output_contract.",
            "Implement every action in selected_feature_actions; ignore lower-priority candidate_experiments and extra suggested features.",
            "If selected_feature_actions contains multiple coordinated actions, wire all of them into the reachable training path or explain only through executable code evidence why a source-supported CLI action needs no code edit.",
            "If the effective train_command introduces a new CLI argument, either modify copied train.py so argparse defines and reachable training code consumes that argument, or return a corrected train_command that removes the unused argument and explain why the feature does not need a command switch.",
            "If a feature CLI gates code inside a dependency function, train.py call sites must pass args=args into that dependency function; parsing the flag and consuming it only in util.py is not sufficient.",
            "If a guardrail feature relies on apply_recent_trend_guardrail or similar helper code, ensure train_command or copied argparse/defaults activate the guardrail instead of leaving trend_guardrail as none.",
            "Do not leave train_command with unsupported argparse flags or missing values; value-taking flags must receive separate argv value tokens.",
            "Use existing function parameters such as label_col, target_col, date_col, id_cols, group_cols, or args instead of hardcoding evaluation-output column names when source code already provides these parameters.",
            "Do not map natural-language recommendations to broad function rewrites; make the smallest executable edit that implements the requested behavior.",
            "Do not rewrite the full training pipeline.",
            "Do not modify original experiment source.",
            "The requested feature must be visible in executable code, not only COMBOSCOPE metadata.",
            "Feature columns should be assigned with a literal/audit-traceable name, e.g. feature_col = f'{feature_name}_...' then df[feature_col] = ...",
            "If construction names explicit output columns, those exact columns must be constructed in reachable code and included before feature_cols/model input is built; similar old columns are not sufficient evidence.",
            "If adding a helper, ensure an existing training/feature path calls it.",
            "If adding a feature column, ensure it exists before feature_cols is returned or used by model training.",
            "If editing only util.py or another dependency, verify train.py imports and calls the changed function path.",
            "If editing only util.py while selected_feature_actions declares cli_args, train.py must parse and consume only the feature CLI args that remain in the effective train_command; declared-only CLI intent can be satisfied by reachable always-on feature code.",
            "train.py is the only process entrypoint; util.py and other dependencies must be reached by local imports/calls, not executed as separate scripts.",
            "Only modify train.py for feature wiring, trial default paths, argparse/preset/default switches, or output path handling.",
            "Do not change evaluation dates/windows, split_data, train_eval_end/defaults, valid_days/test_days, labels, metrics, output filtering, standardization, output_contract semantics, category maps, output writing, or logging unless the user explicitly approved it. Model family, loss/objective, cleaning, and training strategy may be adjusted when supported by clear evidence and documented with risks, expected gains, and rollback steps.",
            "replace_function content must preserve the original function signature, existing keyword calls, defaults, and return arity.",
            "Prefer inserting the new feature near the existing feature construction path instead of rewriting unrelated training logic.",
            "Feature column or helper names should include feature_name or a clear token derived from it so audit can find executable evidence.",
            "implementation_assertions must describe the exact feature calls/columns/wiring that your edits implement; assertions must be true in the modified functions, not merely elsewhere in old copied code.",
            "The generated code must run through the current train_command and produce files matching output_contract prediction/actual fields.",
        ],
        "trial_codegen_rules": _agent2_codegen_skill_rules(
            (repair_guidance or {}).get("failure_categories", []) if isinstance(repair_guidance, dict) else []
        ),
        "selected_feature_actions": primary_changes,
        "primary_feature_change": primary_changes[0] if primary_changes else {},
        "experiment_plan": _compact_experiment_plan(plan),
        "agent2_execution_plan": _compact_agent2_execution_plan(execution_plan),
        "source_evaluation_context": execution_plan.get("source_evaluation_context", {}),
        "selected_edit_tasks": selected_tasks,
        "repair_guidance_from_previous_failures": repair_guidance or {},
        "source_locator": _compact_agent2_source_locator(locator),
        "code_index": _compact_agent2_code_index(code_index),
        "source_slices": _compact_source_slices(source_slices),
    }
    return _budgeted_agent2_yaml_prompt(payload)


def _agent2_repair_prompt(
    plan: dict[str, Any],
    locator: dict[str, Any],
    context_pack: dict[str, Any],
    stage_result: dict[str, Any],
    failure_history: list[dict[str, Any]],
) -> str:
    code_index = context_pack.get("code_index") if isinstance(context_pack, dict) else {}
    if not isinstance(code_index, dict):
        code_index = context_pack if isinstance(context_pack, dict) else {}
    source_slices = context_pack.get("source_slices") if isinstance(context_pack, dict) else {}
    if not isinstance(source_slices, dict):
        source_slices = {"slices": []}
    return _agent2_slice_repair_prompt(
        plan=plan,
        execution_plan={},
        locator=locator,
        code_index=code_index,
        source_slices=source_slices,
        stage_result=stage_result,
        failure_history=failure_history,
    )


def _agent2_slice_repair_prompt(
    *,
    plan: dict[str, Any],
    execution_plan: dict[str, Any],
    locator: dict[str, Any],
    code_index: dict[str, Any],
    source_slices: dict[str, Any],
    stage_result: dict[str, Any],
    failure_history: list[dict[str, Any]],
) -> str:
    repair_guidance = _agent2_repair_guidance(failure_history, locator, plan)
    primary_changes = _agent2_primary_feature_changes(plan)
    payload = {
        "task": "上一版代码包未通过路径/签名/编译/运行时调用契约/特征应用审计。优先用 replace_function 修复复制代码中的目标函数；只修复失败编辑项，返回窄 YAML edits，不要返回完整 files。",
        "repair_priorities": [
            "If failure mentions unsupported edit type, every edits item must include one accepted edit type; prefer narrow edits.",
            "If failure mentions signature incompatible, preserve every original parameter and existing keyword call.",
            "If failure mentions runtime call contract, fix the exact helper/function call so required parameters are passed and unknown keywords are removed.",
            "If failure mentions no executable evidence, construct or consume a real feature column/helper/CLI path containing the requested feature token.",
            "If failure mentions implementation assertions, repair the feature function itself; do not only add argparse/CLI wiring.",
            "If failure mentions evaluation_scope_modified, revert any change to split/evaluation windows/label/metric/output standardization and only edit feature/model-input code.",
            "If failure says evidence is not reachable from train.py, modify train.py or the dependency call chain so train.py imports and calls the changed path.",
            "If failure mentions effective feature CLI args are not parsed or not consumed, modify copied train.py parse_args and reachable training code, or remove the unused CLI from train_command if the feature is always-on by code.",
            "If failure mentions dependency call sites or conditional CLI wiring, pass args=args from train.py into the dependency feature builder/helper that consumes the feature CLI.",
            "If failure mentions guardrail activation, add a valid train_command value such as --trend_guardrail downtrend_clip or change the copied trial default/preset so the guardrail helper is not left at none.",
            "If failure mentions declared feature columns, construct every named output column from the plan exactly, wire the helper through an active train.py path, and include the columns before feature_cols/model input is built.",
            "If repairing a train_command issue, you may return a corrected train_command list together with files/edits.",
            "If audit did not recognize a new feature column, use a direct or traceable column name containing the feature token, such as feature_col = f'{feature_name}_...' followed by df[feature_col] = ...",
            "If a helper was added, call it from an existing train or feature-building path.",
            "If required_corrections includes original_signature, copy that signature exactly before changing function body.",
            "Prefer replace_function over raw line-number edits when repairing Python function-body failures.",
            "Use source code parameter names such as label_col/target_col/date_col/id_cols/group_cols/args when available; do not hardcode evaluation artifact columns unless the copied training source uses them.",
            "Use source_locator.signature_constraints and entrypoint_import_chain before editing; do not switch to full-file replacement unless unavoidable.",
            "Repair the concrete failure; do not regenerate an unrelated full training pipeline.",
        ],
        "preferred_yaml_schema": {
            "edits": [{"path": "train.py", "type": "replace_function", "function": "main", "content": "<replacement function only>"}],
            "train_command": "optional corrected argv token list when the command contract must change",
            "change_summary": "what was fixed",
            "expected_training_path": "why the repaired code should run",
            "implementation_assertions": ["machine-checkable statements satisfied by the repaired modified functions"],
        },
        "accepted_edit_types": _agent2_edit_types_for_prompt(),
        "forbidden_edit_types": ["", "modify_function", "insert_code", "patch", "diff"],
        "forbidden_default_schema": "files",
        "trial_codegen_rules": _agent2_codegen_skill_rules(repair_guidance.get("failure_categories", [])),
        "category_repair_rules": _agent2_category_repair_rules(repair_guidance.get("failure_categories", [])),
        "source_evaluation_context": execution_plan.get("source_evaluation_context", {}),
        "selected_feature_actions": primary_changes,
        "primary_feature_change": primary_changes[0] if primary_changes else {},
        "experiment_plan": _compact_experiment_plan(plan),
        "source_locator": _compact_agent2_source_locator(locator),
        "repair_guidance": repair_guidance,
        "failure": {
            "failure_reason": stage_result.get("failure_reason"),
            "failure_categories": stage_result.get("failure_categories", []),
            "normalized_rejections": stage_result.get("normalized_rejections", []),
            "required_corrections": stage_result.get("required_corrections", []),
            "compile_errors": stage_result.get("compile_errors", []),
            "contract_errors": stage_result.get("contract_errors", []),
            "audit": stage_result.get("audit", {}),
            "rejected_files": stage_result.get("rejected_files", []),
        },
        "all_previous_failures": [_sanitize_agent2_failure(item) for item in failure_history],
        "code_index": _compact_agent2_code_index(code_index),
        "source_slices": _compact_source_slices(source_slices),
    }
    return _budgeted_agent2_yaml_prompt(payload)


def _agent2_codegen_skill_rules(failure_categories: list[str] | None = None) -> str:
    skill_dir = Path(__file__).resolve().parents[2] / "skills" / "forecast-trial-codegen"
    skill_path = skill_dir / "SKILL.md"
    if not skill_path.exists():
        return ""
    sections = [_truncate(skill_path.read_text(encoding="utf-8", errors="ignore"), 2800)]
    categories = {str(item) for item in failure_categories or []}
    reference_map = {
        "signature_incompatible": "signature-repair.md",
        "runtime_contract_error": "runtime-contract-repair.md",
        "no_executable_evidence": "feature-audit-repair.md",
    }
    for category, filename in reference_map.items():
        if category not in categories:
            continue
        reference_path = skill_dir / "references" / filename
        if reference_path.exists():
            sections.append(_truncate(reference_path.read_text(encoding="utf-8", errors="ignore"), 1200))
    if "evaluation_scope_modified" in categories:
        sections.append(
            "Evaluation scope repair: restore source split/evaluation windows/train_eval_end/valid_days/test_days/"
            "label/metric/output standardization exactly, then implement only the requested feature/model-input change."
        )
    if "conditional_cli_not_wired" in categories:
        sections.append(
            "Conditional CLI wiring repair: when a dependency function reads args.<feature_flag>, every train.py call site "
            "for that dependency feature path must pass args=args, or the feature must be changed to an always-on path and "
            "the unused CLI removed from train_command."
        )
    if "guardrail_not_activated" in categories:
        sections.append(
            "Guardrail activation repair: do not only call the guardrail helper; ensure train_command or copied argparse/"
            "preset defaults activate a non-none guardrail mode such as --trend_guardrail downtrend_clip."
        )
    if "declared_feature_columns_missing" in categories:
        sections.append(
            "Declared feature column repair: when the plan construction names exact output columns, construct those exact "
            "columns in reachable feature-building code, ensure the helper call is not disabled by the current train_command/"
            "preset, and include the columns before feature_cols/model input is built. Existing related columns do not count."
        )
    if "inactive_feature_gate" in categories:
        sections.append(
            "Inactive feature gate repair: the feature code exists, but the current train_command/preset disables the helper "
            "call. Activate the flag through train_command or preset defaults, or move the feature construction to an always-on "
            "reachable path before feature_cols/model input is built."
        )
    return _truncate("\n\n".join(sections), 5200)


def _agent2_category_repair_rules(failure_categories: list[str] | None = None) -> list[str]:
    categories = {str(item) for item in failure_categories or []}
    rules: list[str] = []
    if "signature_incompatible" in categories:
        rules.append("Copy the original function signature exactly and preserve return arity.")
    if "runtime_contract_error" in categories:
        rules.append("Read the callee signature from contract_errors/source_locator before editing the failing call.")
    if "no_executable_evidence" in categories:
        rules.append("Add executable evidence through a real feature column/helper call or consumed CLI path.")
    if "evaluation_scope_modified" in categories:
        rules.append("Restore protected split/evaluation/label/metric/output code and edit only feature/model-input logic.")
    if "conditional_cli_not_wired" in categories:
        rules.append("Pass args=args from train.py into dependency feature-builder/helper call sites that consume feature CLI args.")
    if "guardrail_not_activated" in categories:
        rules.append("Activate guardrail mode via train_command or copied argparse/defaults; do not leave trend_guardrail as none.")
    if "declared_feature_columns_missing" in categories:
        rules.append("Construct every explicitly declared feature column and include it in the active feature_cols/model-input path.")
    if "inactive_feature_gate" in categories:
        rules.append("Do not leave the requested feature behind an inactive args flag or preset branch for the final train_command.")
    return rules


def _agent2_edit_types_for_prompt() -> list[str]:
    return [
        *AGENT2_NARROW_EDIT_TYPES,
        *[edit_type for edit_type in sorted(AGENT2_ACCEPTED_EDIT_TYPES) if edit_type not in AGENT2_NARROW_EDIT_TYPES],
    ]


def _compact_experiment_plan(plan: dict[str, Any]) -> dict[str, Any]:
    primary_changes = _agent2_primary_feature_changes(plan)
    return {
        "trial_id": plan.get("trial_id"),
        "scenario": plan.get("scenario"),
        "model_family": plan.get("model_family"),
        "objective": plan.get("objective"),
        "source_entrypoint": plan.get("source_entrypoint"),
        "changes": primary_changes,
        "editable_files": plan.get("editable_files", []),
        "expected_effect": plan.get("expected_effect"),
        "risk": plan.get("risk"),
    }


def _compact_agent2_source_locator(locator: dict[str, Any]) -> dict[str, Any]:
    files: list[dict[str, Any]] = []
    for file_info in locator.get("candidate_files", []) or []:
        candidate_functions = []
        for func in file_info.get("candidate_functions", []) or []:
            candidate_functions.append(
                {
                    "name": func.get("name"),
                    "line": func.get("line"),
                    "end_line": func.get("end_line"),
                    "signature": func.get("signature"),
                    "accepted_keywords": func.get("accepted_keywords", []),
                    "call_keywords": func.get("call_keywords", []),
                    "return_arities": func.get("return_arities", []),
                    "reasons": func.get("reasons", []),
                }
            )
        files.append(
            {
                "path": file_info.get("path"),
                "is_plan_location": file_info.get("is_plan_location"),
                "candidate_functions": candidate_functions[:4],
                "insert_points": (file_info.get("insert_points", []) or [])[:8],
                "outline": (file_info.get("outline", []) or [])[:25],
            }
        )
    return {
        "feature_changes": _agent2_primary_feature_changes({"changes": locator.get("feature_changes", [])}),
        "candidate_files": files[:3],
        "feature_cols_builders": (locator.get("feature_cols_builders", []) or [])[:6],
        "train_entrypoints": (locator.get("train_entrypoints", []) or [])[:6],
        "reachable_dependency_functions": (locator.get("reachable_dependency_functions", []) or [])[:4],
        "signature_constraints": (locator.get("signature_constraints", []) or [])[:12],
        "existing_cli_args": (locator.get("existing_cli_args", []) or [])[:12],
        "cli_consumption_evidence": (locator.get("cli_consumption_evidence", []) or [])[:8],
        "fast_path_candidates": (locator.get("fast_path_candidates", []) or [])[:4],
    }


def _compact_agent2_code_index(code_index: dict[str, Any]) -> dict[str, Any]:
    files: list[dict[str, Any]] = []
    for file_info in code_index.get("files", []) or []:
        compact_functions = []
        for func in file_info.get("functions", []) or []:
            compact_functions.append(
                {
                    "name": func.get("name"),
                    "line": func.get("line"),
                    "end_line": func.get("end_line"),
                    "signature": func.get("signature"),
                    "accepted_keywords": func.get("accepted_keywords", []),
                    "call_keywords": func.get("call_keywords", []),
                    "return_arities": func.get("return_arities", []),
                    "calls": (func.get("calls", []) or [])[:12],
                }
            )
        files.append(
            {
                "path": file_info.get("path"),
                "source_chars": file_info.get("source_chars"),
                "imports": (file_info.get("imports", []) or [])[:12],
                "functions": compact_functions[:25],
                "call_graph": {
                    str(name): calls[:20] if isinstance(calls, list) else calls
                    for name, calls in list((file_info.get("call_graph") or {}).items())[:20]
                },
                "argparse_args": (file_info.get("argparse_args", []) or [])[:20],
                "output_write_hints": (file_info.get("output_write_hints", []) or [])[:12],
                "outline": (file_info.get("outline", []) or [])[:35],
                "insert_points": (file_info.get("insert_points", []) or [])[:16],
                "candidate_functions": (file_info.get("candidate_functions", []) or [])[:8],
            }
        )
    return {
        "index_policy": code_index.get("index_policy", {}),
        "total_source_chars": code_index.get("total_source_chars", 0),
        "train_entrypoints": (code_index.get("train_entrypoints", []) or [])[:8],
        "reachable_dependency_functions": (code_index.get("reachable_dependency_functions", []) or [])[:6],
        "signature_constraints": (code_index.get("signature_constraints", []) or [])[:20],
        "existing_cli_args": (code_index.get("existing_cli_args", []) or [])[:20],
        "cli_consumption_evidence": (code_index.get("cli_consumption_evidence", []) or [])[:12],
        "fast_path_candidates": (code_index.get("fast_path_candidates", []) or [])[:6],
        "files": files[:6],
    }


def _compact_source_slices(source_slices: dict[str, Any]) -> dict[str, Any]:
    slices: list[dict[str, Any]] = []
    for item in source_slices.get("slices", []) or []:
        slices.append(
            {
                "path": item.get("path"),
                "function": item.get("function"),
                "signature": item.get("signature"),
                "start_line": item.get("start_line"),
                "end_line": item.get("end_line"),
                "source_chars": item.get("source_chars"),
                "task_reason": item.get("task_reason", ""),
                "call_points": (item.get("call_points", []) or [])[:6],
                "source": str(item.get("source") or ""),
            }
        )
    return {
        "slice_policy": source_slices.get("slice_policy", {}),
        "selected_edit_tasks": (source_slices.get("selected_edit_tasks", []) or [])[:4],
        "slices": slices[:AGENT2_MAX_SOURCE_SLICES],
    }


def _compact_agent2_execution_plan(execution_plan: dict[str, Any]) -> dict[str, Any]:
    return {
        "agent": execution_plan.get("agent"),
        "trial_id": execution_plan.get("trial_id"),
        "mode": execution_plan.get("mode"),
        "source_entrypoint": execution_plan.get("source_entrypoint"),
        "python_dependencies": execution_plan.get("python_dependencies", []),
        "train_command": execution_plan.get("train_command", []),
        "output_contract": execution_plan.get("output_contract", {}),
        "source_evaluation_context": execution_plan.get("source_evaluation_context", {}),
    }


def _call_agent2_llm(
    llm_client: Any,
    system_prompt: str,
    user_prompt: str,
    *,
    step: str,
    attempt_index: int,
    stream: bool,
    raw_output_dir: Path | None = None,
    timeout: int | float | tuple[int | float, int | float] | None = None,
) -> dict[str, Any]:
    started = time.perf_counter()
    request_timeout = timeout if timeout is not None else AGENT2_CODEGEN_TIMEOUT
    system_prompt, user_prompt = append_yaml_output_contract(system_prompt, user_prompt)
    model_hint = str(getattr(llm_client, "model", "") or getattr(llm_client, "model_name", "") or "")
    prompt_token_estimate = _estimate_prompt_tokens(user_prompt, model_hint)
    budget_limit = AGENT2_PROMPT_TOKEN_LIMIT
    budget_action = "within_budget" if prompt_token_estimate <= budget_limit else "prompt_budget_exceeded"
    if prompt_token_estimate > budget_limit:
        call_record = {
            "step": step,
            "attempt_index": attempt_index,
            "prompt_chars": len(user_prompt),
            "prompt_token_estimate": prompt_token_estimate,
            "budget_limit": budget_limit,
            "budget_action": budget_action,
            "timeout_seconds": request_timeout,
            "streaming_used": bool(stream),
            "duration_seconds": time.perf_counter() - started,
            "success": False,
            "error": f"prompt_budget_exceeded: estimate {prompt_token_estimate} > limit {budget_limit}",
            "content": "",
            "output_chars": 0,
            "attempt_count": 0,
            "retry_count": 0,
            "retry_errors": [],
        }
        raw_path = _write_agent2_raw_llm_output(
            raw_output_dir,
            step=step,
            attempt_index=attempt_index,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            call_record=call_record,
        )
        if raw_path:
            call_record["raw_output_path"] = raw_path
        _append_agent2_llm_call_jsonl(raw_output_dir, call_record)
        return call_record
    try:
        result = llm_client.complete_with_usage(
            system_prompt,
            user_prompt,
            agent="Agent2",
            step=step,
            timeout=request_timeout,
            stream=stream,
        )
    except TypeError:
        result = llm_client.complete_with_usage(system_prompt, user_prompt, agent="Agent2", step=step)
    duration = getattr(result, "duration_seconds", time.perf_counter() - started)
    content = getattr(result, "content", "") or ""
    call_record = {
        "step": step,
        "attempt_index": attempt_index,
        "prompt_chars": len(user_prompt),
        "prompt_token_estimate": prompt_token_estimate,
        "budget_limit": budget_limit,
        "budget_action": budget_action,
        "timeout_seconds": getattr(result, "timeout_seconds", AGENT2_CODEGEN_TIMEOUT),
        "streaming_used": bool(getattr(result, "streaming_used", stream)),
        "duration_seconds": duration,
        "success": bool(getattr(result, "success", False)) and bool(content),
        "error": getattr(result, "error", None),
        "content": content,
        "output_chars": len(content),
        "attempt_count": getattr(result, "attempt_count", 0),
        "retry_count": getattr(result, "retry_count", 0),
        "retry_errors": getattr(result, "retry_errors", []),
    }
    raw_path = _write_agent2_raw_llm_output(
        raw_output_dir,
        step=step,
        attempt_index=attempt_index,
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        call_record=call_record,
    )
    if raw_path:
        call_record["raw_output_path"] = raw_path
    _append_agent2_llm_call_jsonl(raw_output_dir, call_record)
    return call_record


def _write_agent2_raw_llm_output(
    raw_output_dir: Path | None,
    *,
    step: str,
    attempt_index: int,
    system_prompt: str,
    user_prompt: str,
    call_record: dict[str, Any],
) -> str:
    if raw_output_dir is None:
        return ""
    raw_output_dir.mkdir(parents=True, exist_ok=True)
    sequence = len(list(raw_output_dir.glob("*.yaml"))) + 1
    safe_step = re.sub(r"[^0-9A-Za-z_]+", "_", step).strip("_") or "Agent2Call"
    path = raw_output_dir / f"{sequence:02d}_{safe_step}_attempt{attempt_index}.yaml"
    payload = {
        "step": step,
        "attempt_index": attempt_index,
        "success": bool(call_record.get("success")),
        "error": call_record.get("error"),
        "prompt_chars": call_record.get("prompt_chars"),
        "prompt_token_estimate": call_record.get("prompt_token_estimate"),
        "budget_limit": call_record.get("budget_limit"),
        "budget_action": call_record.get("budget_action"),
        "output_chars": call_record.get("output_chars"),
        "attempt_count": call_record.get("attempt_count"),
        "retry_count": call_record.get("retry_count"),
        "retry_errors": call_record.get("retry_errors", []),
        "system_prompt": system_prompt,
        "user_prompt": user_prompt,
        "content": call_record.get("content", ""),
    }
    path.write_text(yaml.safe_dump(payload, allow_unicode=True, sort_keys=False), encoding="utf-8")
    return path.as_posix()


def _append_agent2_llm_call_jsonl(raw_output_dir: Path | None, call_record: dict[str, Any]) -> None:
    if raw_output_dir is None:
        return
    raw_output_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "agent": "Agent2",
        "step": call_record.get("step"),
        "attempt_index": call_record.get("attempt_index"),
        "success": bool(call_record.get("success")),
        "error": call_record.get("error"),
        "prompt_chars": call_record.get("prompt_chars"),
        "prompt_token_estimate": call_record.get("prompt_token_estimate"),
        "budget_limit": call_record.get("budget_limit"),
        "budget_action": call_record.get("budget_action"),
        "output_chars": call_record.get("output_chars"),
        "raw_output_path": call_record.get("raw_output_path", ""),
    }
    with (raw_output_dir / "llm_calls.jsonl").open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False) + "\n")


def _parse_agent2_code_package(content: str) -> dict[str, Any] | None:
    if not clean_llm_yaml_text(content):
        return None
    try:
        parsed = safe_load_yaml_mapping(content, agent="Agent2", step="CodePackage")
    except ValueError:
        return None
    if isinstance(parsed.get("edits"), list) or isinstance(parsed.get("files"), list) or isinstance(parsed.get("train_command"), list):
        return parsed
    return None


def _attach_agent2_failure_context(stage_result: dict[str, Any]) -> dict[str, Any]:
    normalized_rejections = _normalized_rejections(stage_result)
    categories = _failure_categories(stage_result, normalized_rejections)
    return {
        **stage_result,
        "failure_categories": categories,
        "normalized_rejections": normalized_rejections,
        "required_corrections": _required_corrections_for_failure(stage_result, categories, normalized_rejections),
    }


def _normalized_rejections(stage_result: dict[str, Any]) -> list[dict[str, str]]:
    normalized: list[dict[str, str]] = []
    for item in stage_result.get("rejected_files", []) or []:
        reason = str(item.get("reason") or "")
        normalized.append(
            {
                "path": str(item.get("path") or ""),
                "reason": reason,
                "category": _failure_category_from_reason(reason),
            }
        )
    for item in stage_result.get("compile_errors", []) or []:
        normalized.append(
            {
                "path": str(item.get("path") or ""),
                "reason": str(item.get("reason") or "compile error"),
                "category": "compile_error",
            }
        )
    for item in stage_result.get("contract_errors", []) or []:
        category = str(item.get("category") or "") or _failure_category_from_reason(str(item.get("reason") or ""))
        normalized.append(
            {
                "path": str(item.get("path") or ""),
                "reason": str(item.get("reason") or "runtime call contract error"),
                "category": category if category != "unknown" else "runtime_contract_error",
            }
        )
    audit = stage_result.get("audit") or {}
    for feature in audit.get("features", []) or []:
        if feature.get("applied"):
            continue
        reason = str(feature.get("failure_reason") or "no executable evidence")
        normalized.append(
            {
                "path": "",
                "reason": reason,
                "category": str(feature.get("failure_category") or "") or _failure_category_from_reason(reason),
            }
        )
    if not normalized and stage_result.get("failure_reason"):
        reason = str(stage_result.get("failure_reason"))
        normalized.append({"path": "", "reason": reason, "category": _failure_category_from_reason(reason)})
    return normalized


def _failure_categories(stage_result: dict[str, Any], normalized_rejections: list[dict[str, str]]) -> list[str]:
    categories = [str(item.get("category")) for item in normalized_rejections if item.get("category")]
    if stage_result.get("compile_errors"):
        categories.append("compile_error")
    contract_categories = {
        str(item.get("category") or "") or _failure_category_from_reason(str(item.get("reason") or ""))
        for item in stage_result.get("contract_errors", []) or []
    }
    if "evaluation_scope_modified" in contract_categories:
        categories.append("evaluation_scope_modified")
    elif stage_result.get("contract_errors"):
        categories.append("runtime_contract_error")
    if stage_result.get("audit") and not stage_result.get("audit", {}).get("success", False):
        categories.append("no_executable_evidence")
    if not categories:
        categories.append(_failure_category_from_reason(str(stage_result.get("failure_reason") or "")))
    return list(dict.fromkeys(categories))


def _failure_category_from_reason(reason: str) -> str:
    lowered = reason.lower()
    if "unsupported edit type" in lowered:
        return "invalid_edit_type"
    if "signature incompatible" in lowered or "return arity incompatible" in lowered:
        return "signature_incompatible"
    if "path" in lowered or "basename" in lowered or "directory traversal" in lowered or "file is not" in lowered:
        return "path_format_error"
    if "compile" in lowered or "syntax error" in lowered:
        return "compile_error"
    if "runtime call contract" in lowered or "missing required parameter" in lowered or "too many positional" in lowered:
        return "runtime_contract_error"
    if (
        "train_command" in lowered
        and (
            "unsupported argument" in lowered
            or "missing value" in lowered
            or "unexpected positional" in lowered
            or "invalid value" in lowered
        )
    ):
        return "train_command_contract_error"
    if "evaluation scope" in lowered or "protected evaluation" in lowered or "protected argparse" in lowered:
        return "evaluation_scope_modified"
    if "dependency code" in lowered and "not passed through train.py call sites" in lowered:
        return "conditional_cli_not_wired"
    if "guardrail" in lowered and "not activated" in lowered:
        return "guardrail_not_activated"
    if "declared feature columns" in lowered:
        return "declared_feature_columns_missing"
    if "inactive feature gate" in lowered or "inactive args flag" in lowered or "disabled by the current train_command" in lowered:
        return "inactive_feature_gate"
    if "context_too_large" in lowered:
        return "context_too_large"
    if "prompt_budget_exceeded" in lowered:
        return "prompt_budget_exceeded"
    if "full-file replacement is not allowed" in lowered:
        return "full_file_replacement_disallowed"
    if _is_llm_network_error_reason(lowered):
        return "llm_network_error"
    if (
        "no executable evidence" in lowered
        or "feature application audit" in lowered
        or "not reachable from train.py" in lowered
        or "feature cli args" in lowered
    ):
        return "no_executable_evidence"
    if "valid yaml" in lowered or "no accepted files" in lowered or "empty" in lowered or "mapping" in lowered:
        return "invalid_yaml_package"
    return "unknown"


def _is_llm_network_error_reason(lowered_reason: str) -> bool:
    markers = [
        "proxyerror",
        "connectionerror",
        "connecttimeout",
        "readtimeout",
        "timeout",
        "connectionpool",
        "max retries exceeded",
        "unable to connect to proxy",
        "remote end closed connection",
        "remote disconnected",
        "connection aborted",
        "service unavailable",
        "too many requests",
        "http 429",
        "http 500",
        "http 502",
        "http 503",
        "http 504",
    ]
    return any(marker in lowered_reason for marker in markers)


def _required_corrections_for_failure(
    stage_result: dict[str, Any],
    categories: list[str],
    normalized_rejections: list[dict[str, str]],
) -> list[str]:
    corrections: list[str] = []
    if "path_format_error" in categories:
        corrections.append("Return only the copied trial code basename in path, such as train.py or util.py.")
    if "signature_incompatible" in categories:
        corrections.append("For replace_function, copy the original function signature exactly and preserve existing keyword calls.")
    if "compile_error" in categories:
        corrections.append("Fix syntax/import errors before changing strategy; returned files must compile with python compile().")
    if "runtime_contract_error" in categories:
        corrections.append("Fix local function calls so every required argument is passed and no unknown keyword or extra positional argument is used.")
    if "train_command_contract_error" in categories:
        corrections.append("Repair train_command so every argv flag is defined by copied train.py argparse, or remove unused feature CLI args when the feature is already active in code.")
    if "evaluation_scope_modified" in categories:
        corrections.append(
            "Revert changes to split/evaluation windows/train_eval_end/valid_days/test_days/label/metric/output standardization and edit only feature or model-input code."
        )
    if "invalid_edit_type" in categories:
        corrections.append(
            "Every edit item must include a supported type: "
            + ", ".join(sorted(AGENT2_ACCEPTED_EDIT_TYPES))
            + "; do not use empty type, modify_function, insert_code, patch, or diff."
        )
    if "no_executable_evidence" in categories:
        corrections.append(
            "Construct or consume a real feature column/helper/CLI path containing the requested feature token; prefer direct or traceable feature column names."
        )
        if any("not reachable from train.py" in str(item.get("reason") or "") for item in normalized_rejections):
            corrections.append("Modify train.py or the dependency call chain so train.py imports and calls the changed feature path.")
    if "conditional_cli_not_wired" in categories:
        corrections.append(
            "When dependency feature code consumes a feature CLI via args, modify train.py call sites to pass args=args into that dependency function."
        )
    if "guardrail_not_activated" in categories:
        corrections.append(
            "Activate the guardrail in the trial command or copied argparse defaults, for example --trend_guardrail downtrend_clip."
        )
    if "declared_feature_columns_missing" in categories:
        corrections.append(
            "Construct every explicitly declared output feature column, ensure its helper path is active for the current train_command/preset, and include it before feature_cols/model input is built."
        )
    if "inactive_feature_gate" in categories:
        corrections.append(
            "Activate the feature gate for the final train_command/preset or move feature construction into an always-on reachable feature-building path."
        )
    if "invalid_package" in categories or "invalid_yaml_package" in categories:
        corrections.append("Return a valid YAML mapping with edits or files; do not return Markdown or commentary.")
    if "context_too_large" in categories:
        corrections.append("Reduce copied trial Python dependencies or split the trial before asking Agent2 to generate code.")
    if "prompt_budget_exceeded" in categories:
        corrections.append("Reduce Agent2 source_slices/code_index or split code generation into smaller edit tasks before retrying.")
    if "full_file_replacement_disallowed" in categories:
        corrections.append("Return narrow YAML edits instead of files; use replace_function or line-level edits for the selected slices.")
    if "llm_network_error" in categories:
        corrections.append("Check LLM API key, provider base URL, proxy settings, and transient gateway health; then retry Agent2 code generation.")
    for item in normalized_rejections:
        reason = str(item.get("reason") or "")
        if "missing accepted keyword:" in reason:
            missing = reason.rsplit(":", 1)[-1].strip()
            corrections.append(f"Restore missing accepted keyword parameter: {missing}.")
    return list(dict.fromkeys(corrections))


def _failure_categories_from_history(failure_history: list[dict[str, Any]]) -> list[str]:
    categories: list[str] = []
    for item in failure_history:
        categories.extend(str(category) for category in item.get("failure_categories", []) or [])
    return list(dict.fromkeys(categories))


def _normalized_rejections_from_history(failure_history: list[dict[str, Any]]) -> list[dict[str, str]]:
    rejections: list[dict[str, str]] = []
    seen: set[tuple[str, str, str]] = set()
    for item in failure_history:
        for rejection in item.get("normalized_rejections", []) or []:
            key = (
                str(rejection.get("path") or ""),
                str(rejection.get("reason") or ""),
                str(rejection.get("category") or ""),
            )
            if key in seen:
                continue
            seen.add(key)
            rejections.append({"path": key[0], "reason": key[1], "category": key[2]})
    return rejections


def _agent2_repair_guidance(failure_history: list[dict[str, Any]], locator: dict[str, Any], plan: dict[str, Any]) -> dict[str, Any]:
    categories = _failure_categories_from_history(failure_history)
    normalized_rejections = _normalized_rejections_from_history(failure_history)
    required_corrections: list[str] = []
    for item in failure_history:
        required_corrections.extend(str(correction) for correction in item.get("required_corrections", []) or [])
    signature_targets = _signature_targets_for_rejections(normalized_rejections, locator)
    for target in signature_targets:
        required_corrections.append(
            f"Copy original_signature exactly for {target['path']}::{target['function']}: {target['original_signature']}"
        )
    return {
        "failure_categories": categories,
        "normalized_rejections": normalized_rejections,
        "required_corrections": list(dict.fromkeys(required_corrections)),
        "must_not_repeat": _must_not_repeat_for_categories(categories),
        "original_signatures": signature_targets,
        "feature_tokens": [
            token
            for change in plan.get("changes", [])
            if isinstance(change, dict)
            for token in _change_feature_tokens(change)
        ],
    }


def _required_corrections_from_history(failure_history: list[dict[str, Any]]) -> list[str]:
    corrections: list[str] = []
    for item in failure_history:
        corrections.extend(str(correction) for correction in item.get("required_corrections", []) or [])
    return list(dict.fromkeys(corrections))


def _original_signatures_from_history(failure_history: list[dict[str, Any]], locator: dict[str, Any], plan: dict[str, Any]) -> list[dict[str, Any]]:
    guidance = _agent2_repair_guidance(failure_history, locator, plan)
    return list(guidance.get("original_signatures", []) or [])


def _entrypoint_import_chain_from_history(failure_history: list[dict[str, Any]]) -> list[dict[str, Any]]:
    chains: list[dict[str, Any]] = []
    for item in failure_history:
        audit = item.get("audit") or {}
        for feature in audit.get("features", []) or []:
            chain = feature.get("entrypoint_import_chain")
            if isinstance(chain, dict):
                chains.append(chain)
    return chains


def _failed_stage_package_summaries(failure_history: list[dict[str, Any]]) -> list[dict[str, Any]]:
    summaries: list[dict[str, Any]] = []
    for item in failure_history:
        summary = item.get("failed_stage_package_summary")
        if isinstance(summary, dict):
            summaries.append(summary)
    return summaries


def _signature_targets_for_rejections(rejections: list[dict[str, str]], locator: dict[str, Any]) -> list[dict[str, Any]]:
    if not any(item.get("category") in {"signature_incompatible", "runtime_contract_error"} for item in rejections):
        return []
    targets: list[dict[str, Any]] = []
    for file_info in locator.get("candidate_files", []) or []:
        for func in file_info.get("candidate_functions", []) or []:
            if not func.get("signature"):
                continue
            targets.append(
                {
                    "path": file_info.get("path"),
                    "function": func.get("name"),
                    "original_signature": func.get("signature"),
                    "accepted_keywords": func.get("accepted_keywords", []),
                    "call_keywords": func.get("call_keywords", []),
                }
            )
    return targets[:8]


def _must_not_repeat_for_categories(categories: list[str]) -> list[str]:
    rules: list[str] = []
    if "path_format_error" in categories:
        rules.append("Do not return runs/<trial>/code/... or absolute paths; use basename only.")
    if "signature_incompatible" in categories:
        rules.append("Do not delete original parameters or keyword-compatible arguments.")
    if "invalid_edit_type" in categories:
        rules.append(
            "Do not omit edit type or return unsupported edit types such as modify_function, insert_code, patch, or diff; use one of "
            + ", ".join(sorted(AGENT2_ACCEPTED_EDIT_TYPES))
            + "."
        )
    if "full_file_replacement_disallowed" in categories:
        rules.append("Do not return files; return YAML edits with accepted edit types only.")
    if "no_executable_evidence" in categories:
        rules.append("Do not add metadata-only comments, unused helpers, unconsumed feature columns, or feature names only in comments.")
    if "conditional_cli_not_wired" in categories:
        rules.append("Do not leave feature CLI consumption only inside a dependency; train.py must pass args=args into that dependency call path.")
    if "guardrail_not_activated" in categories:
        rules.append("Do not leave guardrail helpers disabled by the default none mode.")
    if "declared_feature_columns_missing" in categories:
        rules.append("Do not satisfy the audit with old related columns; implement the exact declared output columns and active feature_cols wiring.")
    if "inactive_feature_gate" in categories:
        rules.append("Do not leave exact feature columns behind an inactive args flag or preset branch.")
    if "runtime_contract_error" in categories:
        rules.append("Do not call local helpers without their required parameters, especially date_col/label_col/id_cols-style feature builder arguments.")
    return rules


def _sanitize_agent2_failure(stage_result: dict[str, Any]) -> dict[str, Any]:
    return {
        "failure_reason": stage_result.get("failure_reason"),
        "modified_files": stage_result.get("modified_files", []),
        "failure_categories": stage_result.get("failure_categories", []),
        "normalized_rejections": stage_result.get("normalized_rejections", []),
        "required_corrections": stage_result.get("required_corrections", []),
        "compile_errors": stage_result.get("compile_errors", []),
        "contract_errors": stage_result.get("contract_errors", []),
        "audit": stage_result.get("audit", {}),
        "entrypoint_import_chain": _entrypoint_import_chain_from_history([stage_result]),
        "failed_stage_package_summary": stage_result.get("failed_stage_package_summary", {}),
    }


def _stage_agent2_code_package(
    *,
    code_dir: Path,
    wrapper: Path,
    plan: dict[str, Any],
    resource_context: dict[str, Any],
    editable_files: set[str],
    package: dict[str, Any],
    execution_plan: dict[str, Any] | None = None,
    allow_full_file_replacement: bool = False,
    accepted_edit_types: set[str] | None = None,
) -> dict[str, Any]:
    staging_dir = code_dir / "_agent2_staging"
    _cleanup_staging_dir(staging_dir)
    staging_dir.mkdir(parents=True, exist_ok=True)
    for path in code_dir.glob("*.py"):
        shutil.copy2(path, staging_dir / path.name)

    rejected: list[dict[str, str]] = []
    modified_files: set[str] = set()
    modified_functions: set[str] = set()
    accepted_files, file_rejections = _validate_agent2_file_package(
        package,
        editable_files,
        code_dir,
        allow_full_file_replacement=allow_full_file_replacement,
    )
    rejected.extend(file_rejections)
    for rel_path, content in accepted_files.items():
        target_path = staging_dir / rel_path
        target_path.write_text(_rewrite_data_paths(content, resource_context), encoding="utf-8")
        modified_files.add(rel_path)
        modified_functions.update(f"{rel_path}::{name}" for name in _function_names_in_file(target_path))

    edits, edit_rejections = _validate_agent2_edits_package(
        package,
        editable_files,
        code_dir,
        accepted_edit_types=accepted_edit_types,
    )
    rejected.extend(edit_rejections)
    for edit in edits:
        path = staging_dir / edit["path"]
        if edit["type"] == "append_module_code":
            current = path.read_text(encoding="utf-8", errors="ignore")
            path.write_text(current.rstrip() + "\n\n" + _rewrite_data_paths(edit["content"], resource_context).strip() + "\n", encoding="utf-8")
            modified_files.add(edit["path"])
        elif edit["type"] == "replace_function":
            ok, reason = _replace_function_source(
                path,
                edit["function"],
                _rewrite_data_paths(edit["content"], resource_context),
                staging_dir,
            )
            if not ok:
                rejected.append({"path": edit["path"], "reason": reason})
            else:
                modified_files.add(edit["path"])
                modified_functions.add(f"{edit['path']}::{edit['function']}")
        elif edit["type"] == "insert_after_line":
            function_name = _function_name_at_line(path, int(edit["line"]))
            ok, reason = _insert_after_line_source(
                path,
                int(edit["line"]),
                _rewrite_data_paths(edit["content"], resource_context),
            )
            if not ok:
                rejected.append({"path": edit["path"], "reason": reason})
            else:
                modified_files.add(edit["path"])
                if function_name:
                    modified_functions.add(f"{edit['path']}::{function_name}")
        elif edit["type"] == "insert_before_return":
            ok, reason = _insert_before_return_source(
                path,
                edit["function"],
                _rewrite_data_paths(edit["content"], resource_context),
            )
            if not ok:
                rejected.append({"path": edit["path"], "reason": reason})
            else:
                modified_files.add(edit["path"])
                modified_functions.add(f"{edit['path']}::{edit['function']}")
        elif edit["type"] == "replace_lines":
            function_name = _function_name_at_line(path, int(edit["start_line"]))
            ok, reason = _replace_lines_source(
                path,
                int(edit["start_line"]),
                int(edit["end_line"]),
                _rewrite_data_paths(edit["content"], resource_context),
            )
            if not ok:
                rejected.append({"path": edit["path"], "reason": reason})
            else:
                modified_files.add(edit["path"])
                if function_name:
                    modified_functions.add(f"{edit['path']}::{function_name}")
        elif edit["type"] == "add_feature_column_in_function":
            ok, reason = _add_feature_column_in_function_source(
                path,
                edit["function"],
                _rewrite_data_paths(edit["content"], resource_context),
            )
            if not ok:
                rejected.append({"path": edit["path"], "reason": reason})
            else:
                modified_files.add(edit["path"])
                modified_functions.add(f"{edit['path']}::{edit['function']}")

    if not modified_files:
        return _attach_agent2_failure_context({
            "success": False,
            "staging_dir": staging_dir,
            "modified_files": [],
            "rejected_files": rejected,
            "failure_reason": "no accepted files or edits were produced",
            "failed_stage_package_summary": _agent2_package_summary(package),
        })

    compile_errors = _compile_code_dir(staging_dir)
    if compile_errors:
        return _attach_agent2_failure_context({
            "success": False,
            "staging_dir": staging_dir,
            "modified_files": sorted(modified_files),
            "rejected_files": rejected,
            "compile_errors": compile_errors,
            "failure_reason": "generated code did not compile",
            "failed_stage_package_summary": _agent2_package_summary(package),
        })

    contract_errors = _validate_agent2_call_contracts(staging_dir)
    if contract_errors:
        return _attach_agent2_failure_context({
            "success": False,
            "staging_dir": staging_dir,
            "modified_files": sorted(modified_files),
            "rejected_files": rejected,
            "contract_errors": contract_errors,
            "failure_reason": "generated code failed runtime call contract validation",
            "failed_stage_package_summary": _agent2_package_summary(package),
        })

    evaluation_scope_errors = _validate_evaluation_scope_preserved(code_dir, staging_dir, sorted(modified_files))
    if evaluation_scope_errors:
        return _attach_agent2_failure_context({
            "success": False,
            "staging_dir": staging_dir,
            "modified_files": sorted(modified_files),
            "rejected_files": rejected,
            "contract_errors": evaluation_scope_errors,
            "failure_reason": "generated code modified preserved evaluation scope",
            "failed_stage_package_summary": _agent2_package_summary(package),
        })

    (staging_dir / "agent2_code_modification.yaml").write_text(
        yaml.safe_dump({"modified_files": sorted(modified_files)}, allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )
    package_train_command = package.get("train_command") if isinstance(package.get("train_command"), list) else None
    audit_train_command = _audit_train_command_for_code(
        plan=plan,
        wrapper=staging_dir / wrapper.name,
        execution_plan=execution_plan,
        resource_context=resource_context,
        package_train_command=[str(item) for item in package_train_command] if package_train_command else None,
    )
    audit = _audit_feature_application(
        plan,
        staging_dir,
        staging_dir / wrapper.name,
        train_command=audit_train_command,
        modified_functions=sorted(modified_functions),
    )
    if not audit.get("success", False):
        return _attach_agent2_failure_context({
            "success": False,
            "staging_dir": staging_dir,
            "modified_files": sorted(modified_files),
            "rejected_files": rejected,
            "audit": audit,
            "failure_reason": "generated code failed feature application audit",
            "failed_stage_package_summary": _agent2_package_summary(package),
        })
    return {
        "success": True,
        "staging_dir": staging_dir,
        "modified_files": sorted(modified_files),
        "modified_functions": sorted(modified_functions),
        "rejected_files": rejected,
        "audit": audit,
    }


def _agent2_package_summary(package: dict[str, Any]) -> dict[str, Any]:
    files = package.get("files", []) if isinstance(package.get("files", []), list) else []
    edits = package.get("edits", []) if isinstance(package.get("edits", []), list) else []
    return {
        "schema": "edits" if edits else "files" if files else "unknown",
        "file_count": len(files),
        "edit_count": len(edits),
        "files": [
            {
                "path": str(item.get("path") or "") if isinstance(item, dict) else "",
                "content_chars": len(str(item.get("content") or "")) if isinstance(item, dict) else 0,
            }
            for item in files[:8]
        ],
        "edits": [
            {
                "path": str(item.get("path") or "") if isinstance(item, dict) else "",
                "type": str(item.get("type") or "") if isinstance(item, dict) else "",
                "function": str(item.get("function") or "") if isinstance(item, dict) else "",
                "line": item.get("line") if isinstance(item, dict) else None,
                "start_line": item.get("start_line") if isinstance(item, dict) else None,
                "end_line": item.get("end_line") if isinstance(item, dict) else None,
                "content_chars": len(str(item.get("content") or "")) if isinstance(item, dict) else 0,
            }
            for item in edits[:8]
        ],
    }


def _validate_agent2_edits_package(
    package: dict[str, Any],
    editable_files: set[str],
    code_dir: Path,
    *,
    accepted_edit_types: set[str] | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, str]]]:
    edits: list[dict[str, Any]] = []
    rejected: list[dict[str, str]] = []
    for item in package.get("edits", []) or []:
        if not isinstance(item, dict):
            rejected.append({"path": "", "reason": "edit item is not a mapping"})
            continue
        raw_path = str(item.get("path") or "")
        normalized_path, reason = _normalize_agent2_file_path(raw_path, editable_files, code_dir)
        if reason:
            rejected.append({"path": raw_path, "reason": reason})
            continue
        edit_type = str(item.get("type") or "")
        content = item.get("content")
        allowed_edit_types = accepted_edit_types or AGENT2_ACCEPTED_EDIT_TYPES
        if edit_type not in allowed_edit_types:
            rejected.append({"path": raw_path, "reason": f"unsupported edit type: {edit_type}"})
            continue
        if not isinstance(content, str) or not content.strip():
            rejected.append({"path": raw_path, "reason": "content is empty or not a string"})
            continue
        edit: dict[str, Any] = {"path": normalized_path, "type": edit_type, "content": content}
        if edit_type in {"replace_function", "insert_before_return", "add_feature_column_in_function"}:
            function_name = str(item.get("function") or "")
            if not function_name:
                rejected.append({"path": raw_path, "reason": f"{edit_type} requires function"})
                continue
            edit["function"] = function_name
        if edit_type == "insert_after_line":
            line = _positive_int_field(item, "line")
            if line is None:
                rejected.append({"path": raw_path, "reason": "insert_after_line requires positive integer line"})
                continue
            edit["line"] = line
        if edit_type == "replace_lines":
            start_line = _positive_int_field(item, "start_line")
            end_line = _positive_int_field(item, "end_line")
            if start_line is None or end_line is None:
                rejected.append({"path": raw_path, "reason": "replace_lines requires positive integer start_line and end_line"})
                continue
            if end_line < start_line:
                rejected.append({"path": raw_path, "reason": "replace_lines end_line must be >= start_line"})
                continue
            edit["start_line"] = start_line
            edit["end_line"] = end_line
        edits.append(edit)
    return edits, rejected


def _positive_int_field(item: dict[str, Any], key: str) -> int | None:
    value = item.get(key)
    if isinstance(value, bool):
        return None
    if isinstance(value, int) and value > 0:
        return value
    if isinstance(value, str) and value.isdigit() and int(value) > 0:
        return int(value)
    return None


def _insert_after_line_source(path: Path, line: int, content: str) -> tuple[bool, str]:
    source = path.read_text(encoding="utf-8", errors="ignore")
    lines = source.splitlines()
    if line < 1 or line > len(lines):
        return False, f"insert_after_line line out of range: {line}"
    insert_lines = _indent_edit_content(content, _line_insertion_indent(lines, line - 1))
    if not insert_lines:
        return False, "insert_after_line content is empty"
    lines[line:line] = insert_lines
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return True, ""


def _insert_before_return_source(path: Path, function_name: str, content: str) -> tuple[bool, str]:
    return _insert_in_function_source(path, function_name, content, mode="before_return")


def _add_feature_column_in_function_source(path: Path, function_name: str, content: str) -> tuple[bool, str]:
    return _insert_in_function_source(path, function_name, content, mode="before_feature_cols")


def _insert_in_function_source(path: Path, function_name: str, content: str, *, mode: str) -> tuple[bool, str]:
    source = path.read_text(encoding="utf-8", errors="ignore")
    try:
        tree = ast.parse(source)
    except SyntaxError as exc:
        return False, f"existing file has syntax error: {exc.msg}"
    target = _find_function_node(tree, function_name)
    if target is None:
        return False, f"function not found: {function_name}"
    lines = source.splitlines()
    start = int(getattr(target, "lineno", 1)) - 1
    end = int(getattr(target, "end_lineno", len(lines)))
    insert_at = _feature_cols_assignment_line(lines, start, end) if mode == "before_feature_cols" else None
    if insert_at is None:
        insert_at = _first_return_line(lines, start, end)
    if insert_at is None:
        return False, f"function has no return/feature_cols insertion point: {function_name}"
    indent = lines[insert_at][: len(lines[insert_at]) - len(lines[insert_at].lstrip())]
    insert_lines = _indent_edit_content(content, indent)
    if not insert_lines:
        return False, f"{mode} content is empty"
    lines[insert_at:insert_at] = insert_lines
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return True, ""


def _replace_lines_source(path: Path, start_line: int, end_line: int, content: str) -> tuple[bool, str]:
    lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()
    if start_line < 1 or end_line < start_line or end_line > len(lines):
        return False, f"replace_lines range out of range: {start_line}-{end_line}"
    replacement = _indent_edit_content(content, _line_indent(lines[start_line - 1]))
    if not replacement:
        return False, "replace_lines content is empty"
    lines[start_line - 1 : end_line] = replacement
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return True, ""


def _line_indent(line: str) -> str:
    return line[: len(line) - len(line.lstrip())]


def _line_insertion_indent(lines: list[str], index: int) -> str:
    line = lines[index] if 0 <= index < len(lines) else ""
    indent = _line_indent(line)
    if line.rstrip().endswith(":"):
        return indent + "    "
    return indent


def _indent_edit_content(content: str, indent: str) -> list[str]:
    raw_lines = content.strip("\n").splitlines()
    if not raw_lines:
        return []
    first_nonblank = next((line for line in raw_lines if line.strip()), "")
    if first_nonblank.startswith((" ", "\t")):
        return raw_lines
    return [indent + line if line.strip() else "" for line in raw_lines]


def _replace_function_source(path: Path, function_name: str, replacement: str, code_dir: Path | None = None) -> tuple[bool, str]:
    source = path.read_text(encoding="utf-8", errors="ignore")
    try:
        tree = ast.parse(source)
    except SyntaxError as exc:
        return False, f"existing file has syntax error: {exc.msg}"
    target = _find_function_node(tree, function_name)
    if target is None:
        return False, f"function not found: {function_name}"
    try:
        replacement_tree = ast.parse(replacement.strip() + "\n")
    except SyntaxError as exc:
        return False, f"replacement function has syntax error: {exc.msg}"
    replacement_target = _find_function_node(replacement_tree, function_name)
    if replacement_target is None:
        return False, f"replacement function not found: {function_name}"
    ok, reason = _validate_replacement_function_compatibility(target, replacement_target, function_name, code_dir)
    if not ok:
        return False, reason
    start_line = min([target.lineno] + [decorator.lineno for decorator in target.decorator_list])
    end_line = int(getattr(target, "end_lineno", target.lineno))
    lines = source.splitlines()
    replacement_lines = replacement.strip("\n").splitlines()
    lines[start_line - 1 : end_line] = replacement_lines
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return True, ""


def _find_function_node(tree: ast.AST, function_name: str) -> ast.FunctionDef | ast.AsyncFunctionDef | None:
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == function_name:
            return node
    return None


def _validate_replacement_function_compatibility(
    original: ast.FunctionDef | ast.AsyncFunctionDef,
    replacement: ast.FunctionDef | ast.AsyncFunctionDef,
    function_name: str,
    code_dir: Path | None,
) -> tuple[bool, str]:
    original_params = _accepted_keyword_params(original)
    replacement_params = _accepted_keyword_params(replacement)
    missing_params = [name for name in original_params if name not in replacement_params]
    if missing_params:
        return False, f"signature incompatible: missing accepted keyword: {missing_params[0]}"

    original_all_params = set(_all_param_names(original))
    added_required = [name for name in _required_param_names(replacement) if name not in original_all_params]
    if added_required:
        return False, f"signature incompatible: added required parameter: {added_required[0]}"

    replacement_accepts_kwargs = replacement.args.kwarg is not None
    replacement_keywords = set(replacement_params)
    for keyword in _call_keywords_for_function(code_dir, function_name):
        if keyword not in replacement_keywords and not replacement_accepts_kwargs:
            return False, f"signature incompatible: missing accepted keyword: {keyword}"

    original_arities = _tuple_return_arities(original)
    replacement_arities = _tuple_return_arities(replacement)
    if original_arities and replacement_arities and max(replacement_arities) < max(original_arities):
        return False, f"return arity incompatible: expected tuple arity {max(original_arities)}"
    return True, ""


def _accepted_keyword_params(node: ast.FunctionDef | ast.AsyncFunctionDef) -> list[str]:
    return [arg.arg for arg in [*node.args.args, *node.args.kwonlyargs]]


def _all_param_names(node: ast.FunctionDef | ast.AsyncFunctionDef) -> list[str]:
    return [arg.arg for arg in [*node.args.posonlyargs, *node.args.args, *node.args.kwonlyargs]]


def _required_param_names(node: ast.FunctionDef | ast.AsyncFunctionDef) -> list[str]:
    positional = [*node.args.posonlyargs, *node.args.args]
    defaults = list(node.args.defaults)
    required_positional_count = max(len(positional) - len(defaults), 0)
    required = [arg.arg for arg in positional[:required_positional_count]]
    required.extend(arg.arg for arg, default in zip(node.args.kwonlyargs, node.args.kw_defaults) if default is None)
    return required


def _call_keywords_for_function(code_dir: Path | None, function_name: str) -> list[str]:
    if code_dir is None:
        return []
    keywords: list[str] = []
    for path in sorted(code_dir.glob("*.py")):
        try:
            tree = ast.parse(path.read_text(encoding="utf-8", errors="ignore"))
        except SyntaxError:
            continue
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            func = node.func
            if isinstance(func, ast.Name):
                called_name = func.id
            elif isinstance(func, ast.Attribute):
                called_name = func.attr
            else:
                continue
            if called_name != function_name:
                continue
            keywords.extend(keyword.arg for keyword in node.keywords if keyword.arg)
    return list(dict.fromkeys(keywords))


def _tuple_return_arities(node: ast.FunctionDef | ast.AsyncFunctionDef) -> set[int]:
    arities: set[int] = set()
    for child in ast.walk(node):
        if isinstance(child, ast.Return) and isinstance(child.value, ast.Tuple):
            arities.add(len(child.value.elts))
    return arities


def _compile_code_dir(code_dir: Path) -> list[dict[str, str]]:
    errors: list[dict[str, str]] = []
    for path in sorted(code_dir.glob("*.py")):
        try:
            compile(path.read_text(encoding="utf-8", errors="ignore"), path.name, "exec")
        except SyntaxError as exc:
            errors.append({"path": path.name, "reason": f"syntax error line {exc.lineno}: {exc.msg}"})
    return errors


def _validate_agent2_call_contracts(code_dir: Path) -> list[dict[str, str]]:
    errors: list[dict[str, str]] = []
    for path in sorted(code_dir.glob("*.py")):
        try:
            tree = ast.parse(path.read_text(encoding="utf-8", errors="ignore"))
        except SyntaxError:
            continue
        functions = {
            node.name: node
            for node in ast.walk(tree)
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        }
        if not functions:
            continue
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Name):
                continue
            target = functions.get(node.func.id)
            if target is None:
                continue
            reason = _call_contract_error_reason(target, node)
            if reason:
                signature = _function_signature_source(target)
                errors.append(
                    {
                        "path": path.name,
                        "line": str(getattr(node, "lineno", "")),
                        "function": node.func.id,
                        "signature": signature,
                        "required_params": _required_param_names(target),
                        "accepted_keywords": _accepted_keyword_params(target),
                        "reason": (
                            f"{path.name}:{getattr(node, 'lineno', '?')} call {node.func.id} {reason}; "
                            f"callee_signature: {signature}"
                        ),
                    }
                )
    return errors


def _validate_evaluation_scope_preserved(original_dir: Path, staging_dir: Path, modified_files: list[str]) -> list[dict[str, str]]:
    errors: list[dict[str, str]] = []
    for name in modified_files:
        original_path = original_dir / name
        staged_path = staging_dir / name
        if not original_path.exists() or not staged_path.exists():
            continue
        original_source = original_path.read_text(encoding="utf-8", errors="ignore")
        staged_source = staged_path.read_text(encoding="utf-8", errors="ignore")
        try:
            original_tree = ast.parse(original_source)
            staged_tree = ast.parse(staged_source)
        except SyntaxError:
            continue
        errors.extend(_protected_function_change_errors(name, original_source, staged_source, original_tree, staged_tree))
        errors.extend(_protected_argparse_change_errors(name, original_tree, staged_tree))
        errors.extend(_protected_assignment_change_errors(name, original_tree, staged_tree))
    return errors


def _protected_function_change_errors(
    path_name: str,
    original_source: str,
    staged_source: str,
    original_tree: ast.AST,
    staged_tree: ast.AST,
) -> list[dict[str, str]]:
    original_functions = {
        node.name: node
        for node in ast.walk(original_tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    staged_functions = {
        node.name: node
        for node in ast.walk(staged_tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    errors: list[dict[str, str]] = []
    for name, original in original_functions.items():
        if not _protected_evaluation_function_name(name):
            continue
        staged = staged_functions.get(name)
        if staged is None:
            errors.append(_evaluation_scope_error(path_name, name, f"removed protected evaluation function {name}"))
            continue
        if _function_source(original_source, original) != _function_source(staged_source, staged):
            errors.append(_evaluation_scope_error(path_name, name, f"modified protected evaluation function {name}"))
    return errors


def _protected_argparse_change_errors(path_name: str, original_tree: ast.AST, staged_tree: ast.AST) -> list[dict[str, str]]:
    original_args = _protected_argparse_defs(original_tree)
    staged_args = _protected_argparse_defs(staged_tree)
    errors: list[dict[str, str]] = []
    for flag, original_def in original_args.items():
        staged_def = staged_args.get(flag)
        if staged_def is None:
            errors.append(_evaluation_scope_error(path_name, flag, f"removed protected argparse flag {flag}"))
            continue
        if staged_def != original_def:
            errors.append(_evaluation_scope_error(path_name, flag, f"changed protected argparse flag {flag}"))
    for flag in sorted(set(staged_args) - set(original_args)):
        errors.append(_evaluation_scope_error(path_name, flag, f"added protected argparse flag {flag}"))
    return errors


def _protected_assignment_change_errors(path_name: str, original_tree: ast.AST, staged_tree: ast.AST) -> list[dict[str, str]]:
    original_assignments = _protected_assignment_signatures(original_tree)
    staged_assignments = _protected_assignment_signatures(staged_tree)
    errors: list[dict[str, str]] = []
    for name, original_value in original_assignments.items():
        staged_value = staged_assignments.get(name)
        if staged_value is not None and staged_value != original_value:
            errors.append(_evaluation_scope_error(path_name, name, f"changed protected evaluation variable {name}"))
    return errors


def _protected_evaluation_function_name(name: str) -> bool:
    lowered = name.lower()
    protected_tokens = (
        "split_data",
        "fixed_split",
        "train_valid_test",
        "train_test_split",
        "calculate_metric",
        "calculate_metrics",
        "compute_metric",
        "evaluate",
        "standardize",
        "standardized",
        "normalize_output",
        "output_contract",
    )
    return any(token in lowered for token in protected_tokens)


def _protected_argparse_defs(tree: ast.AST) -> dict[str, str]:
    protected_flags = {
        "--train_eval_start",
        "--train-eval-start",
        "--train_eval_end",
        "--train-eval-end",
        "--valid_days",
        "--valid-days",
        "--test_days",
        "--test-days",
        "--label",
        "--label_col",
        "--label-col",
        "--target",
        "--target_col",
        "--target-col",
        "--metric",
    }
    defs: dict[str, str] = {}
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call) or _call_name(node.func).split(".")[-1] != "add_argument":
            continue
        flags = [
            str(arg.value)
            for arg in node.args
            if isinstance(arg, ast.Constant) and isinstance(arg.value, str) and str(arg.value).startswith("--")
        ]
        if not any(flag in protected_flags for flag in flags):
            continue
        signature = ast.dump(node, include_attributes=False)
        for flag in flags:
            if flag in protected_flags:
                defs[flag] = signature
    return defs


def _protected_assignment_signatures(tree: ast.AST) -> dict[str, str]:
    names: dict[str, str] = {}
    protected_names = {
        "label",
        "label_col",
        "target",
        "target_col",
        "metric",
        "metric_name",
        "train_eval_end",
        "valid_days",
        "test_days",
    }
    for node in ast.walk(tree):
        if not isinstance(node, (ast.Assign, ast.AnnAssign)):
            continue
        targets = node.targets if isinstance(node, ast.Assign) else [node.target]
        value = node.value if isinstance(node, ast.AnnAssign) else node.value
        for target in targets:
            for name in _assigned_name_targets(target):
                if name in protected_names:
                    names[name] = ast.dump(value, include_attributes=False) if value is not None else ""
    return names


def _function_source(source: str, node: ast.FunctionDef | ast.AsyncFunctionDef) -> str:
    lines = source.splitlines()
    start_line = min([node.lineno] + [decorator.lineno for decorator in getattr(node, "decorator_list", [])])
    end_line = int(getattr(node, "end_lineno", node.lineno))
    return "\n".join(lines[start_line - 1 : end_line]).strip()


def _evaluation_scope_error(path_name: str, item: str, reason: str) -> dict[str, str]:
    return {
        "path": path_name,
        "function": item,
        "reason": reason,
        "category": "evaluation_scope_modified",
    }


def _function_signature_source(node: ast.FunctionDef | ast.AsyncFunctionDef) -> str:
    prefix = "async def" if isinstance(node, ast.AsyncFunctionDef) else "def"
    signature = f"{prefix} {node.name}({ast.unparse(node.args)})"
    if node.returns is not None:
        signature += f" -> {ast.unparse(node.returns)}"
    return signature


def _call_contract_error_reason(target: ast.FunctionDef | ast.AsyncFunctionDef, call: ast.Call) -> str:
    has_star_args = any(isinstance(arg, ast.Starred) for arg in call.args)
    has_dynamic_kwargs = any(keyword.arg is None for keyword in call.keywords)
    positional_params = [*target.args.posonlyargs, *target.args.args]
    positional_names = [arg.arg for arg in positional_params]
    accepted_keywords = set(_accepted_keyword_params(target))
    keyword_names = [keyword.arg for keyword in call.keywords if keyword.arg]
    positional_count = len([arg for arg in call.args if not isinstance(arg, ast.Starred)])

    if not target.args.vararg and not has_star_args and positional_count > len(positional_params):
        return f"passes too many positional arguments: expected at most {len(positional_params)}, got {positional_count}"

    if not target.args.kwarg and not has_dynamic_kwargs:
        for keyword in keyword_names:
            if keyword not in accepted_keywords:
                return f"passes unknown keyword argument {keyword}"

    if not has_dynamic_kwargs:
        for keyword in keyword_names:
            if keyword in positional_names[:positional_count]:
                return f"passes multiple values for parameter {keyword}"

    if has_star_args or has_dynamic_kwargs:
        return ""

    provided = set(keyword_names)
    provided.update(positional_names[: min(positional_count, len(positional_names))])
    for required in _required_param_names(target):
        if required not in provided:
            return f"missing required parameter {required}"
    return ""


def _copy_staged_modified_files(staging_dir: Path, code_dir: Path, modified_files: list[str]) -> None:
    for name in modified_files:
        shutil.copy2(staging_dir / name, code_dir / name)


def _cleanup_staging_dir(staging_dir: str | Path) -> None:
    path = Path(staging_dir)
    if path.exists():
        shutil.rmtree(path)


def _package_notes(package: dict[str, Any]) -> list[str]:
    notes = package.get("notes", [])
    return [str(item) for item in notes] if isinstance(notes, list) else []


def _agent2_train_py_reason_fields(package: dict[str, Any], modified_files: list[str] | set[str]) -> dict[str, Any]:
    modified = {str(item) for item in modified_files}
    changed_reason = str(package.get("train_py_changed_reason") or "").strip()
    unchanged_reason = str(package.get("train_py_unchanged_reason") or "").strip()
    if "train.py" in modified:
        changed_reason = changed_reason or "Agent2 modified train.py to wire trial defaults, CLI/preset, or feature call path."
        unchanged_reason = ""
    else:
        changed_reason = ""
        unchanged_reason = unchanged_reason or (
            "Agent2 left train.py unchanged because dependency edits are expected to be reached through train.py imports/calls."
        )
    return {
        "train_py_changed_reason": changed_reason,
        "train_py_unchanged_reason": unchanged_reason,
        "entrypoint_import_chain_checked": bool(package.get("entrypoint_import_chain_checked", False)),
    }


def _slice_lines(source: str, start_line: int, end_line: int) -> str:
    lines = source.splitlines()
    return "\n".join(lines[max(start_line - 1, 0) : max(end_line, start_line)])


def _truncate(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    return text[: max(limit - 80, 0)] + f"\n# ... truncated {len(text) - limit} chars ..."


def _budgeted_agent2_yaml_prompt(payload: dict[str, Any], max_prompt_tokens: int = AGENT2_PROMPT_TOKEN_LIMIT) -> str:
    prompt, _ = _enforce_prompt_token_budget(payload, max_prompt_tokens=max_prompt_tokens)
    return prompt


def _enforce_prompt_token_budget(
    payload: dict[str, Any],
    *,
    max_prompt_tokens: int = AGENT2_PROMPT_TOKEN_LIMIT,
) -> tuple[str, dict[str, Any]]:
    target_prompt_tokens = max(1, max_prompt_tokens - 500)
    working = dict(payload)
    working["prompt_budget"] = {
        "budget_limit": max_prompt_tokens,
        "budget_action": "initial",
    }
    text = yaml.safe_dump(working, allow_unicode=True, sort_keys=False)
    estimate = _estimate_prompt_tokens(text)
    if estimate <= target_prompt_tokens:
        working["prompt_budget"] = {
            "prompt_token_estimate": estimate,
            "prompt_chars": len(text),
            "budget_limit": max_prompt_tokens,
            "budget_action": "within_budget",
        }
        text = yaml.safe_dump(working, allow_unicode=True, sort_keys=False)
        return text, working["prompt_budget"]

    budget_action = "compacted"
    for slice_limit in (6000, 3500, 2000, 1200, 700):
        working = _compact_agent2_prompt_payload(payload, source_char_limit=slice_limit)
        working["prompt_budget"] = {
            "budget_limit": max_prompt_tokens,
            "budget_action": budget_action,
            "source_slice_char_limit": slice_limit,
        }
        text = yaml.safe_dump(working, allow_unicode=True, sort_keys=False)
        estimate = _estimate_prompt_tokens(text)
        if estimate <= target_prompt_tokens:
            working["prompt_budget"] = {
                "prompt_token_estimate": estimate,
                "prompt_chars": len(text),
                "budget_limit": max_prompt_tokens,
                "budget_action": budget_action,
                "source_slice_char_limit": slice_limit,
            }
            text = yaml.safe_dump(working, allow_unicode=True, sort_keys=False)
            return text, working["prompt_budget"]
        budget_action = "aggressively_compacted"

    working = _compact_agent2_prompt_payload(payload, source_char_limit=400, aggressive=True)
    working["prompt_budget"] = {
        "budget_limit": max_prompt_tokens,
        "budget_action": "prompt_budget_exceeded",
    }
    text = yaml.safe_dump(working, allow_unicode=True, sort_keys=False)
    estimate = _estimate_prompt_tokens(text)
    working["prompt_budget"] = {
        "prompt_token_estimate": estimate,
        "prompt_chars": len(text),
        "budget_limit": max_prompt_tokens,
        "budget_action": "prompt_budget_exceeded" if estimate > max_prompt_tokens else "aggressively_compacted",
    }
    text = yaml.safe_dump(working, allow_unicode=True, sort_keys=False)
    return text, working["prompt_budget"]


def _compact_agent2_prompt_payload(
    payload: dict[str, Any],
    *,
    source_char_limit: int,
    aggressive: bool = False,
) -> dict[str, Any]:
    compact = dict(payload)
    if "trial_codegen_rules" in compact:
        compact["trial_codegen_rules"] = _truncate(str(compact.get("trial_codegen_rules") or ""), 1800 if not aggressive else 800)
    if isinstance(compact.get("agent2_execution_plan"), dict):
        compact["agent2_execution_plan"] = _compact_agent2_execution_plan(compact["agent2_execution_plan"])
    if isinstance(compact.get("code_index"), dict):
        code_index = _compact_agent2_code_index(compact["code_index"])
        if aggressive:
            code_index["files"] = [
                {
                    **file_info,
                    "functions": (file_info.get("functions", []) or [])[:8],
                    "outline": (file_info.get("outline", []) or [])[:12],
                    "insert_points": (file_info.get("insert_points", []) or [])[:6],
                    "candidate_functions": (file_info.get("candidate_functions", []) or [])[:4],
                    "call_graph": {},
                }
                for file_info in (code_index.get("files", []) or [])[:3]
            ]
            code_index["signature_constraints"] = (code_index.get("signature_constraints", []) or [])[:8]
        compact["code_index"] = code_index
    if isinstance(compact.get("source_locator"), dict) and aggressive:
        locator = dict(compact["source_locator"])
        locator["candidate_files"] = (locator.get("candidate_files", []) or [])[:2]
        locator["signature_constraints"] = (locator.get("signature_constraints", []) or [])[:8]
        compact["source_locator"] = locator
    if isinstance(compact.get("source_slices"), dict):
        compact["source_slices"] = _truncate_agent2_source_slices(
            compact["source_slices"],
            source_char_limit=source_char_limit,
            max_slices=3 if aggressive else AGENT2_MAX_SOURCE_SLICES,
        )
    return compact


def _truncate_agent2_source_slices(
    source_slices: dict[str, Any],
    *,
    source_char_limit: int,
    max_slices: int,
) -> dict[str, Any]:
    compact = _compact_source_slices(source_slices)
    trimmed: list[dict[str, Any]] = []
    for item in (compact.get("slices", []) or [])[:max_slices]:
        source = str(item.get("source") or "")
        trimmed.append({**item, "source": _truncate(source, source_char_limit), "source_chars": len(source)})
    return {**compact, "slices": trimmed}


def _estimate_prompt_tokens(text: str, model_hint: str | None = None) -> int:
    try:
        import tiktoken  # type: ignore

        encoding = tiktoken.encoding_for_model(model_hint or "gpt-4o")
        return len(encoding.encode(text))
    except Exception:
        ascii_chars = sum(1 for char in text if ord(char) < 128)
        non_ascii_chars = len(text) - ascii_chars
        return max(1, int((ascii_chars / 3.5) + (non_ascii_chars / 1.5)) + 1)


def _bounded_yaml_dump(payload: dict[str, Any], limit: int) -> str:
    text = yaml.safe_dump(payload, allow_unicode=True, sort_keys=False)
    if len(text) <= limit:
        return text
    compact_payload = dict(payload)
    if isinstance(compact_payload.get("context_pack"), dict):
        compact_payload["context_pack"] = _compact_context_for_limit(compact_payload["context_pack"])
    text = yaml.safe_dump(compact_payload, allow_unicode=True, sort_keys=False)
    return _truncate(text, limit)


def _compact_context_for_limit(context_pack: dict[str, Any]) -> dict[str, Any]:
    files: list[dict[str, Any]] = []
    for file_info in context_pack.get("files", []) or []:
        compact_snippets = []
        for snippet in file_info.get("snippets", [])[:2]:
            compact_snippets.append({**snippet, "source": _truncate(str(snippet.get("source", "")), 800)})
        files.append(
            {
                "path": file_info.get("path"),
                "outline": file_info.get("outline", [])[:25],
                "insert_points": file_info.get("insert_points", [])[:8],
                "candidate_signatures": file_info.get("candidate_signatures", [])[:4],
                "snippets": compact_snippets,
            }
        )
    return {**context_pack, "files": files}


def _editable_code_files(code_dir: Path, wrapper: Path) -> set[str]:
    names = {path.name for path in code_dir.glob("*.py") if path.is_file()}
    names.add(wrapper.name)
    return names


def _read_editable_code_files(code_dir: Path, editable_files: set[str]) -> list[dict[str, str]]:
    files: list[dict[str, str]] = []
    for name in sorted(editable_files):
        path = code_dir / name
        if path.exists():
            files.append({"path": name, "content": path.read_text(encoding="utf-8", errors="ignore")})
    return files


def _parse_agent2_file_package(content: str) -> dict[str, Any] | None:
    if not clean_llm_yaml_text(content):
        return None
    try:
        parsed = safe_load_yaml_mapping(content, agent="Agent2", step="FilePackage")
    except ValueError:
        return None
    if not isinstance(parsed.get("files"), list):
        return None
    return parsed


def _validate_agent2_file_package(
    package: dict[str, Any],
    editable_files: set[str],
    code_dir: Path,
    *,
    allow_full_file_replacement: bool = False,
) -> tuple[dict[str, str], list[dict[str, str]]]:
    accepted: dict[str, str] = {}
    rejected: list[dict[str, str]] = []
    for item in package.get("files", []):
        if not isinstance(item, dict):
            rejected.append({"path": "", "reason": "file item is not a mapping"})
            continue
        raw_path = str(item.get("path") or "")
        content = item.get("content")
        normalized_path, reason = _normalize_agent2_file_path(raw_path, editable_files, code_dir)
        if reason:
            rejected.append({"path": raw_path, "reason": reason})
            continue
        if not isinstance(content, str) or not content.strip():
            rejected.append({"path": raw_path, "reason": "content is empty or not a string"})
            continue
        if not allow_full_file_replacement:
            rejected.append(
                {
                    "path": raw_path,
                    "reason": "full-file replacement is not allowed for this Agent2 step; return narrow YAML edits instead",
                }
            )
            continue
        token_estimate = _estimate_prompt_tokens(content)
        if token_estimate > AGENT2_PROMPT_TOKEN_LIMIT:
            rejected.append(
                {
                    "path": raw_path,
                    "reason": (
                        "full-file replacement exceeds token budget: "
                        f"estimate {token_estimate} > limit {AGENT2_PROMPT_TOKEN_LIMIT}"
                    ),
                }
            )
            continue
        full_file_reason = _full_file_replacement_guardrail_reason(code_dir / normalized_path, content)
        if full_file_reason:
            rejected.append({"path": raw_path, "reason": full_file_reason})
            continue
        accepted[normalized_path] = content
    return accepted, rejected


def _full_file_replacement_guardrail_reason(existing_path: Path, replacement: str) -> str:
    if not existing_path.exists():
        return ""
    try:
        original_tree = ast.parse(existing_path.read_text(encoding="utf-8", errors="ignore"))
        replacement_tree = ast.parse(replacement)
    except SyntaxError:
        return ""
    important = {
        node.name
        for node in ast.walk(original_tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name in {"main", "parse_args", "build_features", "run_online_pipeline", "train_backtest_and_refit_model", "train_package_model"}
    }
    if not important:
        return ""
    replacement_names = {
        node.name
        for node in ast.walk(replacement_tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    missing = sorted(important - replacement_names)
    if missing:
        return "full-file replacement would remove key function: " + missing[0]
    return ""


def _reject_agent2_file_path_reason(raw_path: str, editable_files: set[str]) -> str:
    _, reason = _normalize_agent2_file_path(raw_path, editable_files, Path.cwd())
    return reason


def _normalize_agent2_file_path(raw_path: str, editable_files: set[str], code_dir: Path) -> tuple[str, str]:
    if not raw_path:
        return "", "path is empty"
    path = Path(raw_path)
    if raw_path in {".", ".."} or ".." in path.parts:
        return "", "directory traversal is not allowed"
    code_root = code_dir.resolve()
    normalized = raw_path
    if path.name != raw_path or path.is_absolute():
        candidate = path.resolve() if path.is_absolute() else (Path.cwd() / path).resolve()
        try:
            candidate.relative_to(code_root)
        except ValueError:
            return "", "path must resolve inside current trial code/ directory"
        normalized = candidate.name
    if Path(normalized).suffix != ".py":
        return "", "only .py files are editable"
    if normalized not in editable_files:
        return "", "file is not an existing copied trial code file"
    return normalized, ""


def _write_code_modification_record(
    code_dir: Path,
    llm_client: Any | None,
    resource_context: dict[str, Any],
    modification: dict[str, Any],
) -> None:
    last_call = getattr(llm_client, "last_call", None)
    record = {
        "agent": "Agent2",
        "step": "ModifyTrialTrainPy",
        "source": modification.get("source", "llm_required"),
        "llm_available": bool(last_call.available) if last_call else False,
        "llm_success": bool(last_call.success) if last_call else False,
        "modified_files": modification.get("modified_files", []),
        "modified_functions": modification.get("modified_functions", []),
        "rejected_files": modification.get("rejected_files", []),
        "fallback_used": bool(modification.get("fallback_used", False)),
        "editable_files": modification.get("editable_files", []),
        "notes": modification.get("notes", []),
        "source_locator_path": modification.get("source_locator_path"),
        "code_index_path": modification.get("code_index_path"),
        "selected_edit_tasks": modification.get("selected_edit_tasks", []),
        "attempts": [_sanitize_attempt_record(item) for item in modification.get("attempts", [])],
        "accepted_schema": modification.get("accepted_schema"),
        "code_generation_success": modification.get("code_generation_success"),
        "feature_code_success": modification.get("feature_code_success"),
        "code_generation_failure_reason": modification.get("code_generation_failure_reason", ""),
        "failure_categories": modification.get("failure_categories", []),
        "normalized_rejections": modification.get("normalized_rejections", []),
        "required_corrections": modification.get("required_corrections", []),
        "original_signatures": modification.get("original_signatures", []),
        "entrypoint_import_chain": modification.get("entrypoint_import_chain", []),
        "failed_stage_package_summary": modification.get("failed_stage_package_summary", []),
        "repair_guidance_history": modification.get("repair_guidance_history", []),
        "train_py_changed_reason": modification.get("train_py_changed_reason", ""),
        "train_py_unchanged_reason": modification.get("train_py_unchanged_reason", ""),
        "entrypoint_import_chain_checked": bool(modification.get("entrypoint_import_chain_checked", False)),
        "final_train_command": modification.get("final_train_command", []),
        "train_command_normalizations": modification.get("train_command_normalizations", []),
        "train_command_repair_attempts": [_sanitize_attempt_record(item) for item in modification.get("train_command_repair_attempts", [])],
        "train_command_repair_success": modification.get("train_command_repair_success"),
        "train_command_repair_error": modification.get("train_command_repair_error", ""),
        "train_command_contract_error_before_repair": modification.get("train_command_contract_error_before_repair", ""),
        "repaired_train_command": modification.get("repaired_train_command", []),
        "feature_resolution": modification.get("feature_resolution", ""),
        "command_contract_status": modification.get("command_contract_status", ""),
        "copied_data_files": resource_context.get("copied_files", {}),
        "agent2_backtest_repair": modification.get("source") == "llm_backtest_repair",
        "note": "这些文件都是 trial/code 中的训练副本；原实验代码未被覆盖。",
    }
    (code_dir / "agent2_code_modification.yaml").write_text(yaml.safe_dump(record, allow_unicode=True, sort_keys=False), encoding="utf-8")


def _update_code_modification_runtime_fields(
    code_dir: Path,
    train_command: list[str],
    feature_audit: dict[str, Any],
    train_command_normalizations: list[dict[str, Any]] | None = None,
) -> None:
    path = code_dir / "agent2_code_modification.yaml"
    record = _read_agent2_code_modification(code_dir)
    if not record:
        return
    record["final_train_command"] = train_command
    record["train_command_normalizations"] = train_command_normalizations or []
    if not record.get("feature_resolution"):
        record["feature_resolution"] = "needs_code_patch"
    if not record.get("command_contract_status"):
        record["command_contract_status"] = "valid"
    record["entrypoint_import_chain_checked"] = bool(feature_audit.get("entrypoint_import_chain_checked", False))
    if not feature_audit.get("success", False):
        stage_result = {
            "audit": feature_audit,
            "failure_reason": "feature application audit failed before training",
        }
        normalized = _normalized_rejections(stage_result)
        categories = _failure_categories(stage_result, normalized)
        existing_categories = [str(item) for item in record.get("failure_categories", []) or [] if item]
        existing_rejections = [
            item for item in record.get("normalized_rejections", []) or [] if isinstance(item, dict)
        ]
        record["code_generation_success"] = False
        record["feature_code_success"] = False
        record["code_generation_failure_reason"] = (
            record.get("code_generation_failure_reason")
            or "feature application audit failed before training"
        )
        record["failure_categories"] = list(dict.fromkeys([*existing_categories, *categories]))
        record["normalized_rejections"] = [*existing_rejections, *normalized]
        record["required_corrections"] = _required_corrections_for_failure(
            stage_result,
            record["failure_categories"],
            record["normalized_rejections"],
        )
    elif record.get("code_generation_success") is True and record.get("feature_code_success") is None:
        record["feature_code_success"] = True
    modified_files = {str(item) for item in record.get("modified_files", []) or []}
    if "train.py" in modified_files and not record.get("train_py_changed_reason"):
        record["train_py_changed_reason"] = "Agent2 modified train.py to wire trial defaults, CLI/preset, or feature call path."
    if "train.py" not in modified_files:
        if not feature_audit.get("success", False):
            record["train_py_unchanged_reason"] = (
                "Agent2 left train.py unchanged; feature application audit did not verify a reachable train.py import/call chain."
            )
        elif not record.get("train_py_unchanged_reason"):
            record["train_py_unchanged_reason"] = (
                "Agent2 left train.py unchanged after checking that dependency changes are reached through the train.py import/call chain."
            )
    path.write_text(yaml.safe_dump(record, allow_unicode=True, sort_keys=False), encoding="utf-8")


def _sanitize_attempt_record(attempt: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in attempt.items() if key != "content"}


def _strip_code_fence(content: str) -> str:
    return strip_code_fence(content)


def _patch_trial_train_source(
    source_text: str,
    arg_overrides: dict[str, bool],
    plan: dict[str, Any],
    experiment: Path,
    resource_context: dict[str, Any] | None = None,
) -> str:
    header = _trial_header(arg_overrides, plan, experiment)
    patched = _insert_trial_header(source_text, header)
    patched = _rewrite_data_paths(patched, resource_context or {})
    patched = _insert_output_override_hook(patched)
    old_main = "if __name__ == \"__main__\":\n    setup_logger()\n    run_online_pipeline(parse_args())"
    new_main = "\n".join(
        [
            "if __name__ == \"__main__\":",
            "    setup_logger()",
            "    args = parse_args()",
            "    for key, value in COMBOSCOPE_ARG_OVERRIDES.items():",
            "        if hasattr(args, key):",
            "            setattr(args, key, value)",
            "    _comboscope_apply_cli_output_overrides(args)",
            "    run_online_pipeline(args)",
        ]
    )
    if old_main in patched:
        patched = patched.replace(old_main, new_main)
    return patched


def _trial_header(arg_overrides: dict[str, bool], plan: dict[str, Any], experiment: Path) -> str:
    return "\n".join(
        [
            "# Generated by ComboScope Agent2 from the original experiment entrypoint.",
            "# This trial copy is safe to edit; the original experiment source is not modified.",
            "import sys as _comboscope_sys",
            "from pathlib import Path as _ComboScopePath",
            f"COMBOSCOPE_ORIGINAL_EXPERIMENT_DIR = _ComboScopePath({experiment.as_posix()!r})",
            "COMBOSCOPE_ORIGINAL_PYTHON_PACKAGE_DIR = COMBOSCOPE_ORIGINAL_EXPERIMENT_DIR / '.python_packages'",
            "for _comboscope_path in [COMBOSCOPE_ORIGINAL_PYTHON_PACKAGE_DIR, COMBOSCOPE_ORIGINAL_EXPERIMENT_DIR]:",
            "    if _comboscope_path.exists() and _comboscope_path.as_posix() not in _comboscope_sys.path:",
            "        _comboscope_sys.path.insert(0, _comboscope_path.as_posix())",
            f"COMBOSCOPE_ARG_OVERRIDES = {arg_overrides!r}",
            f"COMBOSCOPE_FEATURE_CHANGES = {plan.get('changes', [])!r}",
            "def _comboscope_cli_items():",
            "    _items = {}",
            "    _index = 1",
            "    while _index < len(_comboscope_sys.argv):",
            "        _item = _comboscope_sys.argv[_index]",
            "        if not str(_item).startswith('--'):",
            "            _index += 1",
            "            continue",
            "        _flag = str(_item)",
            "        _name = _flag[2:].replace('-', '_')",
            "        _index += 1",
            "        _values = []",
            "        while _index < len(_comboscope_sys.argv) and not str(_comboscope_sys.argv[_index]).startswith('--'):",
            "            _values.append(_comboscope_sys.argv[_index])",
            "            _index += 1",
            "        if _flag.startswith('--no-'):",
            "            _items[_flag[5:].replace('-', '_')] = False",
            "        elif not _values:",
            "            _items[_name] = True",
            "        elif len(_values) == 1:",
            "            _items[_name] = _values[0]",
            "        else:",
            "            _items[_name] = _values",
            "    return _items",
            "def _comboscope_cast_cli_value(current, value):",
            "    if isinstance(current, bool):",
            "        if isinstance(value, bool):",
            "            return value",
            "        return str(value).lower() not in {'0', 'false', 'no', 'off'}",
            "    if isinstance(current, list):",
            "        values = value if isinstance(value, list) else [value]",
            "        caster = type(current[0]) if current else str",
            "        converted = []",
            "        for item in values:",
            "            try:",
            "                converted.append(caster(item))",
            "            except Exception:",
            "                converted.append(item)",
            "        return converted",
            "    if current is None:",
            "        return value",
            "    try:",
            "        return type(current)(value)",
            "    except Exception:",
            "        return value",
            "def _comboscope_apply_cli_output_overrides(args):",
            "    for _name, _value in _comboscope_cli_items().items():",
            "        if hasattr(args, _name):",
            "            setattr(args, _name, _comboscope_cast_cli_value(getattr(args, _name), _value))",
            "    return args",
            "",
        ]
    )


def _rewrite_data_paths(source_text: str, resource_context: dict[str, Any]) -> str:
    rewritten = source_text
    for old, new in resource_context.get("path_rewrites", {}).items():
        rewritten = rewritten.replace(str(old), str(new))
    return rewritten


def _insert_trial_header(source_text: str, header: str) -> str:
    lines = source_text.splitlines()
    insert_at = 0
    for index, line in enumerate(lines):
        stripped = line.strip()
        if line.startswith("from __future__ import "):
            insert_at = index + 1
            continue
        if not stripped or stripped.startswith("#"):
            continue
        if insert_at:
            break
        break
    lines.insert(insert_at, header.rstrip("\n"))
    return "\n".join(lines) + "\n"


def _insert_output_override_hook(source_text: str) -> str:
    marker = "    output_dir = Path(args.output_dir)"
    hook = "    _comboscope_apply_cli_output_overrides(args)\n" + marker
    if marker in source_text and hook not in source_text:
        return source_text.replace(marker, hook, 1)
    return source_text


def _output_contract_from_execution_plan(execution_plan: dict[str, Any]) -> dict[str, Any]:
    contract = execution_plan.get("output_contract")
    if not isinstance(contract, dict):
        raise ValueError("Agent2 execution_plan.output_contract must be a mapping")
    for key in ("prediction_path", "actual_path", "prediction_column", "actual_column"):
        if not isinstance(contract.get(key), str) or not str(contract[key]).strip():
            raise ValueError(f"Agent2 output_contract missing required field: {key}")
    contract.setdefault("id_columns", [])
    contract.setdefault("passthrough_columns", [])
    return contract


def load_plan(path: str | Path) -> dict[str, Any]:
    return yaml.safe_load(Path(path).read_text(encoding="utf-8"))
