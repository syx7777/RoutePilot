from __future__ import annotations

import yaml
import pytest
from pydantic import ValidationError

from routepilot.agents.evaluation_hypothesis_agent import (
    generate_experiment_plan,
    generate_feature_hypothesis,
    generate_real_experiment_plan,
    generate_real_feature_hypothesis,
)
from routepilot.core.schemas import ExperimentChange, ExperimentPlan
from routepilot.runtime.doubao_client import LLMCallResult


class FakeLLM:
    def __init__(self, content: str):
        self.content = content
        self.last_call = None
        self.calls = []

    def complete_with_usage(self, system_prompt: str, user_prompt: str, *, agent: str, step: str) -> LLMCallResult:
        self.calls.append({"system_prompt": system_prompt, "user_prompt": user_prompt, "agent": agent, "step": step})
        self.last_call = LLMCallResult(
            content=self.content,
            model="fake",
            agent=agent,
            step=step,
            duration_seconds=0.01,
            usage={"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
            success=bool(self.content),
            error=None if self.content else "empty",
        )
        return self.last_call


class SequencedFakeLLM(FakeLLM):
    def __init__(self, contents: list[str]):
        super().__init__("")
        self.contents = list(contents)

    def complete_with_usage(self, system_prompt: str, user_prompt: str, *, agent: str, step: str) -> LLMCallResult:
        self.content = self.contents.pop(0) if self.contents else ""
        return super().complete_with_usage(system_prompt, user_prompt, agent=agent, step=step)


def _feature_hypothesis_yaml(feature_name: str = "entity_recent_trend_ratio") -> str:
    return yaml.safe_dump(
        {
            "target_problem": "segment_underestimate",
            "hypothesis": "Recent entity trend features may reduce the observed underestimation.",
            "evidence": ["overall_bias=-0.12", "top_underestimate badcases are concentrated in high target rows"],
            "proposed_features": [
                {
                    "action": "add_feature",
                    "feature_name": feature_name,
                    "feature_type": "trend_ratio",
                    "cli_args": ["--enable-recent-trend-ratio"],
                    "field_sources": ["date", "entity_id", "actual"],
                    "construction": "compare recent target mean with a longer historical mean by entity.",
                    "code_locations": ["src/train_forecast.py", "src/features.py"],
                    "validation_metrics": ["wape", "bias"],
                }
            ],
            "confidence": "medium",
        },
        sort_keys=False,
    )


def _wrapped_feature_hypothesis_yaml(feature_name: str = "entity_recent_trend_ratio") -> str:
    return yaml.safe_dump({"feature_hypothesis": yaml.safe_load(_feature_hypothesis_yaml(feature_name))}, sort_keys=False)


def test_feature_hypothesis_is_validated_from_llm_output() -> None:
    hypothesis = generate_feature_hypothesis(
        {"main_problem": "segment_underestimate", "evidence": ["bias evidence"]},
        "feature suggestions must be evidence-bound",
        FakeLLM(_feature_hypothesis_yaml("entity_rolling_mean")),
    )
    plan = generate_experiment_plan(hypothesis, "trial_001")
    validated = ExperimentPlan.model_validate(plan)

    assert validated.trial_id == "trial_001"
    assert validated.changes[0].feature_name == "entity_rolling_mean"
    assert "benchmark/train.py" in validated.editable_files
    assert yaml.safe_dump(plan, sort_keys=False)


def test_experiment_change_rejects_dirty_cli_arg_tokens() -> None:
    with pytest.raises(ValidationError, match="single argv tokens"):
        ExperimentChange.model_validate(
            {
                "action": "add_feature",
                "feature_name": "sparsity_rolling",
                "cli_args": ["--enable_sparsity_rolling  (optional boolean flag)"],
            }
        )


def test_feature_hypothesis_repairs_dirty_cli_arg_tokens_once() -> None:
    dirty = yaml.safe_load(_feature_hypothesis_yaml("sparsity_rolling"))
    dirty["proposed_features"][0]["cli_args"] = ["--enable_sparsity_rolling  (optional boolean flag)"]
    clean = yaml.safe_load(_feature_hypothesis_yaml("sparsity_rolling"))
    clean["proposed_features"][0]["cli_args"] = ["--enable_sparsity_rolling"]
    client = SequencedFakeLLM([yaml.safe_dump(dirty, sort_keys=False), yaml.safe_dump(clean, sort_keys=False)])

    hypothesis = generate_feature_hypothesis(
        {"main_problem": "segment_overestimate", "evidence": ["bias evidence"]},
        "feature suggestions must be evidence-bound",
        client,
    )

    assert hypothesis["proposed_features"][0]["cli_args"] == ["--enable_sparsity_rolling"]
    assert [call["step"] for call in client.calls] == ["GenerateExperimentPlan", "GenerateExperimentPlan"]
    assert "cli_args must be a list of real argv tokens only" in client.calls[1]["user_prompt"]


def test_feature_hypothesis_unwraps_common_model_wrapper() -> None:
    hypothesis = generate_real_feature_hypothesis(
        {"main_problem": "segment_underestimate", "evidence": ["bias evidence"]},
        "rules",
        FakeLLM(_wrapped_feature_hypothesis_yaml("wrapped_recent_trend")),
    )

    assert hypothesis["target_problem"] == "segment_underestimate"
    assert hypothesis["proposed_features"][0]["feature_name"] == "wrapped_recent_trend"


def test_feature_hypothesis_prompt_includes_yaml_output_contract() -> None:
    llm = FakeLLM(_feature_hypothesis_yaml("contracted_recent_trend"))

    generate_real_feature_hypothesis(
        {"main_problem": "segment_underestimate", "evidence": ["bias evidence"]},
        "rules",
        llm,
    )

    assert "RoutePilot YAML output contract" in llm.calls[0]["system_prompt"]
    assert "Return exactly one YAML mapping" in llm.calls[0]["user_prompt"]


def test_feature_hypothesis_normalizes_yaml_mapping_evidence_items() -> None:
    raw = """
target_problem: package_granularity_high_wape_overestimation_dominant
hypothesis: Zero-demand evidence suggests a sparsity feature may reduce overestimation.
evidence:
- Overall WAPE=0.70 and positive bias=0.17
- low_target;overestimate;not_holiday: 14906 rows with actual_sum=0.0 but abs_error_sum=20120.8
- Top badcase: store 109540 actual=25 vs pred=99.99
proposed_features:
- action: add_feature
  feature_name: recent_demand_is_zero
  feature_type: sparsity_indicator
  field_sources:
  - ds
  - store_code
  - package_dish_code
  construction: Add leakage-safe recent zero-demand indicators.
  code_locations:
  - src/lgb_package_to_dish_online_0319.py
  validation_metrics:
  - wape
  - bias
"""

    hypothesis = generate_real_feature_hypothesis(
        {"main_problem": "segment_overestimate", "evidence": ["bias evidence"]},
        "rules",
        FakeLLM(raw),
    )

    assert hypothesis["evidence"] == [
        "Overall WAPE=0.70 and positive bias=0.17",
        "low_target;overestimate;not_holiday: 14906 rows with actual_sum=0.0 but abs_error_sum=20120.8",
        "Top badcase: store 109540 actual=25 vs pred=99.99",
    ]


def test_feature_hypothesis_normalizes_feature_level_yaml_mapping_evidence_items() -> None:
    raw = """
target_problem: package_granularity_high_wape_overestimation_dominant
hypothesis: Recent zero-demand indicators may reduce overestimation.
evidence:
- overall_bias=0.17
proposed_features:
- action: add_feature
  feature_name: recent_demand_is_zero
  feature_type: sparsity_indicator
  evidence:
  - low_target;overestimate;not_holiday: zero-actual rows are frequently overpredicted
  field_sources:
  - ds
  - store_code
  - package_dish_code
  construction: Add leakage-safe recent zero-demand indicators.
  code_locations:
  - src/lgb_package_to_dish_online_0319.py
  validation_metrics:
  - wape
"""

    hypothesis = generate_real_feature_hypothesis(
        {"main_problem": "segment_overestimate", "evidence": ["bias evidence"]},
        "rules",
        FakeLLM(raw),
    )

    assert hypothesis["proposed_features"][0]["evidence"] == [
        "low_target;overestimate;not_holiday: zero-actual rows are frequently overpredicted"
    ]


def test_feature_hypothesis_fails_when_llm_output_is_invalid() -> None:
    with pytest.raises(ValueError):
        generate_feature_hypothesis(
            {"main_problem": "segment_underestimate", "evidence": ["bias evidence"]},
            "rules",
            FakeLLM("not: [valid"),
        )

    with pytest.raises(RuntimeError):
        generate_real_feature_hypothesis(
            {"main_problem": "segment_underestimate", "evidence": ["bias evidence"]},
            "rules",
            FakeLLM(""),
        )


def test_wrapped_feature_hypothesis_still_fails_when_inner_schema_is_invalid() -> None:
    invalid = yaml.safe_dump({"feature_hypothesis": {"hypothesis": "missing required fields"}}, sort_keys=False)

    with pytest.raises(ValueError):
        generate_real_feature_hypothesis(
            {"main_problem": "segment_underestimate", "evidence": ["bias evidence"]},
            "rules",
            FakeLLM(invalid),
        )


def test_feature_hypothesis_repairs_schema_validation_failure_once() -> None:
    client = SequencedFakeLLM(
        [
            yaml.safe_dump({"not_feature_hypothesis": {"hypothesis": "wrong top-level shape"}}, sort_keys=False),
            _feature_hypothesis_yaml("repaired_recent_trend"),
        ]
    )

    hypothesis = generate_real_feature_hypothesis(
        {"main_problem": "segment_underestimate", "evidence": ["bias evidence"]},
        "rules",
        client,
    )

    assert hypothesis["proposed_features"][0]["feature_name"] == "repaired_recent_trend"
    assert [call["step"] for call in client.calls] == ["GenerateExperimentPlan", "GenerateExperimentPlan"]
    assert "Do not wrap the result under feature_hypothesis" in client.calls[1]["user_prompt"]


def test_feature_hypothesis_repairs_non_yaml_response_once() -> None:
    client = SequencedFakeLLM(
        [
            "- **证据**：模型输出了 Markdown，不是 YAML\n- proposed feature: recent trend\n",
            _feature_hypothesis_yaml("markdown_repaired_recent_trend"),
        ]
    )

    hypothesis = generate_real_feature_hypothesis(
        {"main_problem": "segment_underestimate", "evidence": ["bias evidence"]},
        "rules",
        client,
    )

    assert hypothesis["proposed_features"][0]["feature_name"] == "markdown_repaired_recent_trend"
    assert len(client.calls) == 2
    assert "raw_response" in client.calls[1]["user_prompt"]


def test_real_feature_hypothesis_uses_llm_generated_candidates_only() -> None:
    context = {
        "main_problem": "high_target_underestimate",
        "error_direction": "underestimate",
        "metric_definition": {
            "objective_label": "bias",
            "decision_metric": "bias",
            "direction": "minimize_abs",
            "metric_formula": "sum(pred - true) / sum(true)",
            "metric_definition_source": ["src/evaluate.py:42"],
        },
        "evidence": [
            "overall_bias=-0.120456",
            "high_target_underestimate_scene={'scene': 'weekend;high_target;underestimate;not_holiday'}",
        ],
        "available_fields": ["entity_id", "date", "actual", "prediction"],
        "available_feature_functions": ["build_features", "generate_rolling_features"],
        "available_cli_args": ["--rolling-windows"],
        "available_code_locations": ["src/train_forecast.py", "src/features.py"],
        "source_entrypoint": "src/train_forecast.py",
    }

    hypothesis = generate_real_feature_hypothesis(context, "rules", FakeLLM(_feature_hypothesis_yaml()))

    assert [feature["feature_name"] for feature in hypothesis["proposed_features"]] == ["entity_recent_trend_ratio"]
    feature = hypothesis["proposed_features"][0]
    assert feature["cli_args"] == ["--enable-recent-trend-ratio"]
    assert feature["code_locations"] == ["src/train_forecast.py", "src/features.py"]
    assert feature["validation_metrics"] == ["wape", "bias"]


def test_real_feature_hypothesis_prompt_declares_top_feature_execution_contract() -> None:
    client = FakeLLM(_feature_hypothesis_yaml())

    generate_real_feature_hypothesis(
        {"main_problem": "segment_underestimate", "evidence": ["bias evidence"]},
        "rules",
        client,
    )

    assert "proposed_features[0]" in client.calls[0]["user_prompt"]
    assert "highest-priority" in client.calls[0]["user_prompt"]


def test_real_experiment_plan_selects_highest_priority_candidate_for_any_trial_id() -> None:
    hypothesis = yaml.safe_load(_feature_hypothesis_yaml("first_priority_feature"))
    hypothesis["proposed_features"].append(
        {
            "action": "add_feature",
            "feature_name": "lower_priority_feature",
            "feature_type": "numeric",
            "cli_args": [],
            "field_sources": ["date"],
            "construction": "lower priority feature",
            "code_locations": ["src/train_forecast.py"],
            "validation_metrics": ["wape"],
        }
    )

    plan = generate_real_experiment_plan(
        hypothesis,
        "trial_003",
        "runs/trial_003/code/train.py",
        {"source_entrypoint": "src/train_forecast.py"},
    )

    assert plan["changes"][0]["feature_name"] == "first_priority_feature"


def test_real_experiment_plan_hypothesis_describes_selected_candidate_only() -> None:
    hypothesis = yaml.safe_load(_feature_hypothesis_yaml("disable_recent_trend_guardrail"))
    hypothesis["hypothesis"] = (
        "Disabling the recent trend guardrail and adding intermittency features will reduce overestimation."
    )
    hypothesis["proposed_features"][0]["action"] = "remove_feature"
    hypothesis["proposed_features"][0]["feature_type"] = "guardrail"
    hypothesis["proposed_features"][0]["cli_args"] = ["--disable_trend_guardrail"]
    hypothesis["proposed_features"][0]["construction"] = "Disable the recent trend guardrail."
    hypothesis["proposed_features"].append(
        {
            "action": "add_feature",
            "feature_name": "intermittency_features",
            "feature_type": "sparsity",
            "cli_args": ["--enable_intermittency_features"],
            "field_sources": ["date", "actual"],
            "construction": "Add days since last sale, zero streak, and nonzero ratio features.",
            "code_locations": ["src/train_forecast.py"],
            "validation_metrics": ["wape"],
        }
    )

    plan = generate_real_experiment_plan(
        hypothesis,
        "trial_003",
        "runs/trial_003/code/train.py",
        {"source_entrypoint": "src/train_forecast.py"},
    )

    description = plan["hypothesis"]["description"]
    assert plan["changes"][0]["feature_name"] == "disable_recent_trend_guardrail"
    assert "--disable_trend_guardrail" in description
    assert "intermittency" not in description.lower()
    assert "--enable_intermittency_features" not in description


def test_real_experiment_plan_carries_source_defined_ask_metric() -> None:
    context = {
        "scenario": "generic_forecast",
        "source_entrypoint": "src/train_forecast.py",
        "metric_definition": {
            "objective_label": "signed_bias",
            "decision_metric": "bias",
            "direction": "minimize_abs",
            "metric_formula": "sum(pred - true) / sum(true)",
            "metric_definition_source": ["src/evaluate.py:42"],
        },
    }
    hypothesis = yaml.safe_load(_feature_hypothesis_yaml("entity_recent_target_momentum"))

    plan = generate_real_experiment_plan(hypothesis, "trial_001", "runs/trial_001/code/train.py", context)

    assert plan["evaluation_metric"] == context["metric_definition"]
    assert plan["source_entrypoint"] == "src/train_forecast.py"
    assert plan["changes"][0]["validation_metrics"] == ["wape", "bias"]
    assert "bias" in yaml.safe_dump(plan, sort_keys=False, allow_unicode=True)


def test_real_editable_files_allow_trial_code_python_dependencies_only() -> None:
    base_plan = {
        "trial_id": "trial_001",
        "target_problem": "high_target_underestimate",
        "hypothesis": {"description": "test", "evidence": ["evidence"]},
        "changes": [{"action": "add_feature", "feature_name": "x", "feature_type": "rolling_stat"}],
        "expected_effect": "reduce error",
        "risk": "low",
    }

    validated = ExperimentPlan.model_validate(
        {
            **base_plan,
            "editable_files": ["runs/trial_001/code/train.py", "runs/trial_001/code/features.py"],
        }
    )
    assert "runs/trial_001/code/features.py" in validated.editable_files

    with pytest.raises(ValueError, match="editable file not allowed"):
        ExperimentPlan.model_validate({**base_plan, "editable_files": ["src/features.py"]})

    with pytest.raises(ValueError, match="editable file not allowed"):
        ExperimentPlan.model_validate({**base_plan, "editable_files": ["tmp/runs/trial_001/code/features.py"]})


def test_real_experiment_plan_maps_dependency_code_locations_to_trial_copies() -> None:
    context = {"source_entrypoint": "src/train_forecast.py"}
    hypothesis = yaml.safe_load(_feature_hypothesis_yaml("entity_recent_trend_ratio"))

    plan = generate_real_experiment_plan(hypothesis, "trial_001", "runs/trial_001/code/train.py", context)

    assert plan["editable_files"] == ["runs/trial_001/code/train.py", "runs/trial_001/code/features.py"]


def test_real_experiment_plan_converts_duplicate_rolling_default_to_cli_variation() -> None:
    context = {
        "source_entrypoint": "src/train_forecast.py",
        "argparse_defaults": {"--rolling_windows": [3, 7]},
    }
    hypothesis = yaml.safe_load(_feature_hypothesis_yaml("rolling_3d_7d_avg"))
    hypothesis["proposed_features"][0]["cli_args"] = []

    plan = generate_real_experiment_plan(hypothesis, "trial_001", "runs/trial_001/code/train.py", context)

    assert plan["changes"][0]["feature_name"] == "extend_rolling_windows_1_3_7"
    assert plan["changes"][0]["cli_args"] == ["--rolling_windows", "1", "3", "7"]
    assert plan["candidate_experiments"][0]["experiment_id"] == "exp_extend_rolling_windows_1_3_7"
