from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

import comboscope.orchestrators.langgraph_flow as langgraph_flow
from comboscope.orchestrators.langgraph_flow import _normalize_report_markdown
from comboscope.orchestrators.langgraph_flow import _parse_llm_yaml_mapping
from comboscope.orchestrators.langgraph_flow import _validated_agent2_execution_plan
from comboscope.orchestrators.langgraph_flow import run_forecast_evaluation, run_once
from comboscope.runtime.doubao_client import LLMCallResult


class SequencedLLM:
    def __init__(self, by_step: dict[str, list[str] | str]):
        self.by_step = {key: list(value) if isinstance(value, list) else [value] for key, value in by_step.items()}
        self.last_call = None
        self.calls = []
        self.call_results = []

    def available(self) -> bool:
        return True

    def complete_with_usage(
        self,
        system_prompt: str,
        user_prompt: str,
        *,
        agent: str,
        step: str,
        timeout=None,
        stream: bool = False,
        max_tokens=None,
    ) -> LLMCallResult:
        self.calls.append(
            {
                "agent": agent,
                "step": step,
                "stream": stream,
                "timeout": timeout,
                "max_tokens": max_tokens,
                "user_prompt": user_prompt,
            }
        )
        queue = self.by_step.get(step, [])
        content = queue.pop(0) if queue else ""
        self.last_call = LLMCallResult(
            content=content,
            model="fake",
            agent=agent,
            step=step,
            duration_seconds=0.01,
            usage={"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
            success=bool(content),
            error=None if content else f"no fake response for {step}",
            timeout_seconds=timeout,
            streaming_used=stream,
        )
        self.call_results.append(self.last_call)
        return self.last_call


def _artifact_contract() -> str:
    return yaml.safe_dump(
        {
            "prediction_path": "benchmark/data/prediction_seed.csv",
            "actual_path": "benchmark/data/actual.csv",
            "prediction_column": "prediction",
            "actual_column": "actual",
            "date_column": "date",
            "id_columns": ["combo_id"],
            "passthrough_columns": [],
            "source_entrypoint": "benchmark/train.py",
            "model_family": "unknown",
            "objective": "unknown",
            "output_contract": "forecast_outputs_to_standard_metrics",
        },
        sort_keys=False,
    )


def _invalid_artifact_contract() -> str:
    value = yaml.safe_load(_artifact_contract())
    value["prediction_column"] = "actual"
    return yaml.safe_dump(value, sort_keys=False)


def _diagnosis() -> str:
    return yaml.safe_dump(
        {
            "main_problem": "high_target_underestimate",
            "affected_scenes": ["high_target", "underestimate"],
            "error_direction": "underestimate",
            "evidence": ["overall_bias is negative", "badcases contain high target underestimation"],
            "candidate_causes": ["recent target momentum may be underrepresented"],
            "feature_opportunities": ["recent trend ratio"],
            "available_fields": ["date", "combo_id", "prediction", "actual"],
            "available_feature_functions": ["main"],
            "available_cli_args": ["--output"],
            "available_code_locations": ["benchmark/train.py"],
            "badcase_diagnosis_markdown": "# Badcase Diagnosis\n\nHigh target rows are underestimated.\n",
        },
        sort_keys=False,
    )


def _diagnosis_with_unindented_markdown_fence() -> str:
    return """main_problem: high_target_underestimate
affected_scenes:
- high_target
- underestimate
error_direction: underestimate
evidence:
- overall_bias is negative
- badcases contain high target underestimation
candidate_causes:
- recent target momentum may be underrepresented
feature_opportunities:
- recent trend ratio
available_fields:
- date
- combo_id
- prediction
- actual
available_feature_functions:
- main
available_cli_args:
- --output
available_code_locations:
- benchmark/train.py
badcase_diagnosis_markdown:
```
# Badcase Diagnosis

High target rows are underestimated.
```
"""


def _feature_hypothesis() -> str:
    return yaml.safe_dump(
        {
            "target_problem": "high_target_underestimate",
            "hypothesis": "A recent trend feature can reduce underestimation.",
            "evidence": ["overall_bias is negative", "badcases contain high target underestimation"],
            "proposed_features": [
                {
                    "action": "add_feature",
                    "feature_name": "recent_trend_ratio",
                    "feature_type": "trend_ratio",
                    "cli_args": ["--enable-recent-trend-ratio"],
                    "code_locations": ["benchmark/train.py"],
                    "field_sources": ["date", "combo_id", "actual"],
                    "construction": "recent average divided by longer average.",
                    "validation_metrics": ["wape", "bias"],
                }
            ],
        },
        sort_keys=False,
    )


def _agent2_execution_plan() -> str:
    return yaml.safe_dump(
        {
            "agent": "Agent2",
            "trial_id": "trial_001",
            "source_entrypoint": "benchmark/train.py",
            "python_dependencies": [],
            "required_data_files": [{"path": "benchmark/data/actual.csv"}],
            "train_command": ["{python}", "{train_py}", "--output", "{real_output_dir}"],
            "output_contract": {
                "prediction_path": "prediction.csv",
                "actual_path": "data/actual.csv",
                "prediction_column": "prediction",
                "actual_column": "actual",
                "date_column": "date",
                "id_columns": ["combo_id"],
                "passthrough_columns": [],
            },
        },
        sort_keys=False,
    )


def _invalid_agent2_execution_plan() -> str:
    return yaml.safe_dump(
        {
            "agent": "Agent2",
            "trial_id": "trial_001",
            "source_entrypoint": "benchmark/train.py",
            "python_dependencies": ["benchmark/feature_policy.py"],
            "required_data_files": [{"path": "data.csv", "cli_arg": "--data_path"}],
            "train_command": ["{python}", "{train_py}", "--output_dir", "{real_output_dir}", "--data_path", "{data_path}"],
            "output_contract": {
                "prediction_path": "predictions.csv",
                "actual_path": "actuals.csv",
                "prediction_column": "prediction",
                "actual_column": "actual",
                "date_column": "date",
                "id_columns": ["combo_id"],
                "passthrough_columns": [],
            },
        },
        sort_keys=False,
    )


def _unparseable_agent2_execution_plan() -> str:
    return "agent: Agent2\ntrial_id: trial_001\nsource_entrypoint: [\n"


def _agent2_code_package() -> str:
    return yaml.safe_dump(
        {
            "files": [
                {
                    "path": "train.py",
                    "content": """
from __future__ import annotations

import argparse
import csv
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    recent_trend_ratio = 1.0
    rows = [
        {"date": "2026-05-01", "combo_id": "c001", "prediction": 118 if recent_trend_ratio else 95},
        {"date": "2026-05-02", "combo_id": "c001", "prediction": 123 if recent_trend_ratio else 100},
    ]
    with (output / "prediction.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["date", "combo_id", "prediction"])
        writer.writeheader()
        writer.writerows(rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
""",
                }
            ],
            "notes": ["implemented recent_trend_ratio in copied train.py"],
        },
        sort_keys=False,
    )


def test_llm_yaml_mapping_filters_invalid_control_characters() -> None:
    parsed = _parse_llm_yaml_mapping("agent\x95: Agent2\ntrial_id: trial_001\n", agent="Agent2", step="BuildExecutionPlan")

    assert parsed == {"agent": "Agent2", "trial_id": "trial_001"}


def test_report_markdown_repairs_utf8_mojibake() -> None:
    report = "# 预测实验评测报告\n\n## 1. 一句话结论\n- WAPE 偏高。\n"
    garbled = report.encode("utf-8").decode("latin1")

    repaired = _normalize_report_markdown(garbled, step="WriteForecastReport")

    assert repaired == report


def _fake_llm() -> SequencedLLM:
    return SequencedLLM(
        {
            "SelectArtifacts": _artifact_contract(),
            "DiagnoseBadcases": _diagnosis(),
            "GenerateExperimentPlan": _feature_hypothesis(),
            "BuildExecutionPlan": _agent2_execution_plan(),
            "LocateCodeChangePlan": "change_plan:\n- path: train.py\n  function: main\n  reason: implement feature\n",
            "GenerateCodeEdits": _agent2_code_package(),
            "WriteForecastReport": "# 预测实验评测报告\n\n## 1. 一句话结论\n- 基于证据定位到高目标值低估。\n",
            "WriteFinalReport": "# ComboScope 实验验证结论报告\n\n## 1. 结论\n- 最终报告由 LLM 生成。\n",
        }
    )



def test_langgraph_flow_fails_fast_without_llm(monkeypatch, fixture_exp: Path, tmp_path: Path, repo_root: Path) -> None:
    monkeypatch.delenv("DOUBAO_API_KEY", raising=False)
    monkeypatch.delenv("DOUBAO_BASE_URL", raising=False)
    monkeypatch.delenv("DOUBAO_MODEL", raising=False)
    output_dir = tmp_path / "trial_001"

    with pytest.raises(RuntimeError, match="SelectArtifacts"):
        run_forecast_evaluation(
            {
                "experiment_dir": fixture_exp.as_posix(),
                "output_dir": output_dir.as_posix(),
                "ask": "分析预测误差，定位 badcase 并给出建议",
                "trial_id": "trial_001",
                "repo_root": repo_root.as_posix(),
                "model": "doubao",
                "errors": [],
            }
        )

    assert (output_dir / "error_report.md").exists()
    assert (output_dir / "llm_calls.jsonl").exists()
    assert "SelectArtifacts" in (output_dir / "llm_calls.jsonl").read_text(encoding="utf-8")


def test_forecast_evaluation_subgraph_uses_llm_artifact_contract(
    monkeypatch, fixture_exp: Path, tmp_path: Path, repo_root: Path
) -> None:
    client = _fake_llm()
    monkeypatch.setattr(langgraph_flow, "create_llm_client", lambda provider=None, model=None: client)
    output_dir = tmp_path / "trial_001"

    result = run_forecast_evaluation(
        {
            "experiment_dir": fixture_exp.as_posix(),
            "output_dir": output_dir.as_posix(),
            "ask": "分析预测误差，定位 badcase 并给出建议",
            "trial_id": "trial_001",
            "repo_root": repo_root.as_posix(),
            "model": "fake",
            "errors": [],
        }
    )

    assert result["real_mode"] is True
    contract = json.loads((output_dir / "artifact_contract.json").read_text(encoding="utf-8"))
    assert contract["prediction_path"] == "benchmark/data/prediction_seed.csv"
    assert (output_dir / "badcase_diagnosis.md").exists()
    assert (output_dir / "metrics.json").exists()
    assert [call["step"] for call in client.calls] == ["SelectArtifacts", "DiagnoseBadcases"]
    diagnose_prompt = next(call["user_prompt"] for call in client.calls if call["step"] == "DiagnoseBadcases")
    assert "badcase_diagnosis_markdown: |" in diagnose_prompt
    assert "do not wrap the response in Markdown fences" in diagnose_prompt


def test_forecast_evaluation_repairs_invalid_artifact_contract_columns(
    monkeypatch, fixture_exp: Path, tmp_path: Path, repo_root: Path
) -> None:
    client = SequencedLLM(
        {
            "SelectArtifacts": [_invalid_artifact_contract(), _artifact_contract()],
            "DiagnoseBadcases": _diagnosis(),
        }
    )
    monkeypatch.setattr(langgraph_flow, "create_llm_client", lambda provider=None, model=None: client)
    output_dir = tmp_path / "trial_001"

    run_forecast_evaluation(
        {
            "experiment_dir": fixture_exp.as_posix(),
            "output_dir": output_dir.as_posix(),
            "ask": "分析预测误差，定位 badcase 并给出建议",
            "trial_id": "trial_001",
            "repo_root": repo_root.as_posix(),
            "model": "fake",
            "errors": [],
        }
    )

    contract = json.loads((output_dir / "artifact_contract.json").read_text(encoding="utf-8"))
    assert contract["prediction_column"] == "prediction"
    assert [call["step"] for call in client.calls] == ["SelectArtifacts", "SelectArtifacts", "DiagnoseBadcases"]
    assert "prediction_column must exist" in client.calls[1]["user_prompt"]


def test_forecast_evaluation_repairs_invalid_badcase_diagnosis_yaml(
    monkeypatch, fixture_exp: Path, tmp_path: Path, repo_root: Path
) -> None:
    client = SequencedLLM(
        {
            "SelectArtifacts": _artifact_contract(),
            "DiagnoseBadcases": [_diagnosis_with_unindented_markdown_fence(), _diagnosis()],
        }
    )
    monkeypatch.setattr(langgraph_flow, "create_llm_client", lambda provider=None, model=None: client)
    output_dir = tmp_path / "trial_001"

    run_forecast_evaluation(
        {
            "experiment_dir": fixture_exp.as_posix(),
            "output_dir": output_dir.as_posix(),
            "ask": "分析预测误差，定位 badcase 并给出建议",
            "trial_id": "trial_001",
            "repo_root": repo_root.as_posix(),
            "model": "fake",
            "errors": [],
        }
    )

    context = json.loads((output_dir / "problem_context.json").read_text(encoding="utf-8"))
    assert context["main_problem"] == "high_target_underestimate"
    assert "High target rows are underestimated" in (output_dir / "badcase_diagnosis.md").read_text(encoding="utf-8")
    diagnose_calls = [call for call in client.calls if call["step"] == "DiagnoseBadcases"]
    assert len(diagnose_calls) == 2
    assert "validation_error" in diagnose_calls[1]["user_prompt"]
    assert "invalid_response" in diagnose_calls[1]["user_prompt"]
    assert "block scalar" in diagnose_calls[1]["user_prompt"]


def test_run_once_invokes_llm_report_and_records_calls(monkeypatch, fixture_exp: Path, tmp_path: Path, repo_root: Path) -> None:
    client = _fake_llm()
    monkeypatch.setattr(langgraph_flow, "create_llm_client", lambda provider=None, model=None: client)
    output_dir = tmp_path / "trial_001"

    result = run_once(
        {
            "experiment_dir": fixture_exp.as_posix(),
            "output_dir": output_dir.as_posix(),
            "ask": "分析预测误差，提出一个特征实验并验证效果",
            "trial_id": "trial_001",
            "repo_root": repo_root.as_posix(),
            "model": "fake",
        }
    )

    assert result["decision"] in {"keep", "rollback"}
    assert (output_dir / "agent1" / "artifact_contract.json").exists()
    assert (output_dir / "agent1" / "badcase_diagnosis.md").exists()
    assert (output_dir / "agent2" / "output_contract.json").exists()
    assert (output_dir / "reports" / "forecast_report.md").exists()
    assert (output_dir / "final_report.md").exists()
    calls_text = (output_dir / "audit" / "llm_calls.jsonl").read_text(encoding="utf-8")
    assert "SelectArtifacts" in calls_text
    assert "BuildExecutionPlan" in calls_text
    assert "WriteForecastReport" in calls_text
    assert "WriteFinalReport" in calls_text
    assert "LocateCodeChangePlan" in calls_text or "GenerateCodeEdits" in calls_text or "RepairCodeEdits" in calls_text
    steps = [call["step"] for call in client.calls]
    assert steps.index("WriteForecastReport") < steps.index("BuildExecutionPlan")
    plan_prompt = next(call["user_prompt"] for call in client.calls if call["step"] == "GenerateExperimentPlan")
    assert "forecast-optimization-case-reference" in plan_prompt
    assert "package-lgbm-optimization-case" in plan_prompt
    report_prompt = next(call["user_prompt"] for call in client.calls if call["step"] == "WriteForecastReport")
    report_call = next(call for call in client.calls if call["step"] == "WriteForecastReport")
    assert report_call["stream"] is True
    assert report_call["timeout"] == langgraph_flow.AGENT1_REPORT_TIMEOUT
    assert "forecast-optimization-case-reference" in report_prompt
    build_prompt = next(call["user_prompt"] for call in client.calls if call["step"] == "BuildExecutionPlan")
    build_call = next(call for call in client.calls if call["step"] == "BuildExecutionPlan")
    assert build_call["stream"] is False
    assert build_call["timeout"] == langgraph_flow.AGENT2_PLANNING_TIMEOUT
    assert "agent1_forecast_report_excerpt" in build_prompt
    assert "train_command may include feature CLI args only for experiment_plan.changes" in build_prompt
    route = yaml.safe_load((output_dir / "agent1" / "agent1_skill_route.yaml").read_text(encoding="utf-8"))
    assert "forecast-optimization-case-reference" in route["selected_skills"]
    assert not (output_dir / "seed_baseline").exists()


def test_run_once_without_model_leaves_llm_resolution_to_env(
    monkeypatch, fixture_exp: Path, tmp_path: Path, repo_root: Path
) -> None:
    client = _fake_llm()
    requests: list[tuple[str | None, str | None]] = []

    def fake_create_llm_client(provider=None, model=None):
        requests.append((provider, model))
        return client

    monkeypatch.setattr(langgraph_flow, "create_llm_client", fake_create_llm_client)
    output_dir = tmp_path / "trial_001"

    run_once(
        {
            "experiment_dir": fixture_exp.as_posix(),
            "output_dir": output_dir.as_posix(),
            "ask": "分析预测误差，提出一个特征实验并验证效果",
            "trial_id": "trial_001",
            "repo_root": repo_root.as_posix(),
        }
    )

    assert requests
    assert all(model is None for _, model in requests)


def test_run_once_repairs_invalid_agent2_execution_plan(
    monkeypatch, fixture_exp: Path, tmp_path: Path, repo_root: Path
) -> None:
    client = SequencedLLM(
        {
            "SelectArtifacts": _artifact_contract(),
            "DiagnoseBadcases": _diagnosis(),
            "GenerateExperimentPlan": _feature_hypothesis(),
            "BuildExecutionPlan": [_invalid_agent2_execution_plan(), _agent2_execution_plan()],
            "LocateCodeChangePlan": "change_plan:\n- path: train.py\n  function: main\n  reason: implement feature\n",
            "GenerateCodeEdits": _agent2_code_package(),
            "WriteForecastReport": "# 预测实验评测报告\n\n## 1. 一句话结论\n- 已完成执行计划修复。\n",
            "WriteFinalReport": "# ComboScope 实验验证结论报告\n\n## 1. 结论\n- 已完成最终验证。\n",
        }
    )
    monkeypatch.setattr(langgraph_flow, "create_llm_client", lambda provider=None, model=None: client)
    output_dir = tmp_path / "trial_001"

    run_once(
        {
            "experiment_dir": fixture_exp.as_posix(),
            "output_dir": output_dir.as_posix(),
            "ask": "分析预测误差，提出一个特征实验并验证效果",
            "trial_id": "trial_001",
            "repo_root": repo_root.as_posix(),
            "model": "fake",
        }
    )

    execution_plan = yaml.safe_load((output_dir / "agent2" / "agent2_execution_plan.yaml").read_text(encoding="utf-8"))
    assert execution_plan["required_data_files"] == [{"path": "benchmark/data/actual.csv"}]
    assert "{data_path}" not in yaml.safe_dump(execution_plan)
    build_calls = [call for call in client.calls if call["step"] == "BuildExecutionPlan"]
    assert len(build_calls) == 2
    assert all(call["stream"] is False for call in build_calls)
    assert all(call["timeout"] == langgraph_flow.AGENT2_PLANNING_TIMEOUT for call in build_calls)
    assert "unsupported placeholders" in build_calls[1]["user_prompt"] or "required data file" in build_calls[1]["user_prompt"]


def test_run_once_repairs_unparseable_agent2_execution_plan_yaml(
    monkeypatch, fixture_exp: Path, tmp_path: Path, repo_root: Path
) -> None:
    client = SequencedLLM(
        {
            "SelectArtifacts": _artifact_contract(),
            "DiagnoseBadcases": _diagnosis(),
            "GenerateExperimentPlan": _feature_hypothesis(),
            "BuildExecutionPlan": [_unparseable_agent2_execution_plan(), _agent2_execution_plan()],
            "LocateCodeChangePlan": "change_plan:\n- path: train.py\n  function: main\n  reason: implement feature\n",
            "GenerateCodeEdits": _agent2_code_package(),
            "WriteForecastReport": "# 预测实验评测报告\n\n## 1. 一句话结论\n- 已完成执行计划修复。\n",
            "WriteFinalReport": "# ComboScope 实验验证结论报告\n\n## 1. 结论\n- 已完成最终验证。\n",
        }
    )
    monkeypatch.setattr(langgraph_flow, "create_llm_client", lambda provider=None, model=None: client)
    output_dir = tmp_path / "trial_001"

    run_once(
        {
            "experiment_dir": fixture_exp.as_posix(),
            "output_dir": output_dir.as_posix(),
            "ask": "分析预测误差，提出一个特征实验并验证效果",
            "trial_id": "trial_001",
            "repo_root": repo_root.as_posix(),
            "model": "fake",
        }
    )

    assert (output_dir / "agent2" / "agent2_execution_plan.yaml").exists()
    build_calls = [call for call in client.calls if call["step"] == "BuildExecutionPlan"]
    assert len(build_calls) == 2
    assert "validation_error" in build_calls[1]["user_prompt"]
    audit_calls = [
        json.loads(line)
        for line in (output_dir / "audit" / "llm_calls.jsonl").read_text(encoding="utf-8").splitlines()
        if '"BuildExecutionPlan"' in line
    ]
    assert audit_calls[0]["yaml_parse_status"] == "failed"
    assert audit_calls[0]["yaml_repair_attempted"] is True
    assert audit_calls[0]["yaml_repair_success"] is True
    assert audit_calls[0]["yaml_failure_raw_output_path"]
    assert Path(audit_calls[0]["yaml_failure_raw_output_path"]).exists()


def test_agent2_execution_plan_rejects_trial_id_as_argparse_choice(tmp_path: Path) -> None:
    experiment = tmp_path / "choice_exp"
    src = experiment / "src"
    src.mkdir(parents=True)
    (src / "train.py").write_text(
        """
import argparse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", choices=["baseline", "exp_02_tweedie"], default="baseline")
    parser.add_argument("--output_dir")
    parser.parse_args()
""",
        encoding="utf-8",
    )
    execution_plan = {
        "source_entrypoint": "src/train.py",
        "python_dependencies": [],
        "required_data_files": [],
        "train_command": ["{python}", "{train_py}", "--output_dir", "{real_output_dir}", "--experiment", "{trial_id}"],
        "output_contract": {
            "prediction_path": "prediction.csv",
            "actual_path": "actual.csv",
            "prediction_column": "prediction",
            "actual_column": "actual",
        },
    }

    with pytest.raises(ValueError, match="invalid value '\\{trial_id\\}' to --experiment"):
        _validated_agent2_execution_plan(
            execution_plan,
            experiment,
            {"trial_id": "baseline_trial_06"},
            {"trial_id": "baseline_trial_06", "output_dir": tmp_path.as_posix()},
        )


def test_source_evaluation_context_infers_window_and_train_eval_end_from_artifact(tmp_path: Path) -> None:
    experiment = tmp_path / "exp"
    src = experiment / "src"
    src.mkdir(parents=True)
    (src / "train.py").write_text(
        """
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--train_eval_end", "--train-eval-end")
""",
        encoding="utf-8",
    )
    prediction = tmp_path / "standardized_prediction.csv"
    prediction.write_text(
        "ds,id,prediction\n"
        "2026-05-05,a,1\n"
        "2026-05-11,b,2\n",
        encoding="utf-8",
    )
    actual = tmp_path / "standardized_actual.csv"
    actual.write_text(
        "ds,id,actual\n"
        "2026-05-05,a,1\n"
        "2026-05-11,b,2\n",
        encoding="utf-8",
    )

    context = langgraph_flow._build_source_evaluation_context(
        experiment=experiment,
        artifact_contract={
            "date_column": "ds",
            "split_column": "split",
            "split_value": "test",
            "source_entrypoint": "src/train.py",
        },
        standardized_prediction=prediction,
        standardized_actual=actual,
        original_command="",
    )

    assert context["context_status"] == "resolved"
    assert context["test_start"] == "2026-05-05"
    assert context["test_end"] == "2026-05-11"
    assert context["sample_count"] == 2
    assert context["preserved_train_args"] == {"--train_eval_end": "20260511"}


def test_agent2_execution_plan_inherits_source_evaluation_args(tmp_path: Path) -> None:
    experiment = tmp_path / "exp"
    src = experiment / "src"
    src.mkdir(parents=True)
    (src / "train.py").write_text(
        """
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--output_dir")
parser.add_argument("--train_eval_end", "--train-eval-end")
""",
        encoding="utf-8",
    )
    execution_plan = {
        "source_entrypoint": "src/train.py",
        "python_dependencies": [],
        "required_data_files": [],
        "train_command": ["{python}", "{train_py}", "--output_dir", "{real_output_dir}"],
        "output_contract": {
            "prediction_path": "prediction.csv",
            "actual_path": "actual.csv",
            "prediction_column": "prediction",
            "actual_column": "actual",
        },
    }

    validated = _validated_agent2_execution_plan(
        execution_plan,
        experiment,
        {"trial_id": "trial_001"},
        {
            "trial_id": "trial_001",
            "output_dir": tmp_path.as_posix(),
            "source_evaluation_context": {
                "context_status": "resolved",
                "test_start": "2026-05-05",
                "test_end": "2026-05-11",
                "sample_count": 2,
                "preserved_train_args": {"--train_eval_end": "20260511"},
            },
        },
    )

    assert validated["train_command"][-2:] == ["--train_eval_end", "20260511"]
    assert validated["source_evaluation_context"]["test_end"] == "2026-05-11"
