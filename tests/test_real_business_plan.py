from __future__ import annotations

import yaml

from comboscope.agents.evaluation_hypothesis_agent import generate_real_experiment_plan, generate_real_feature_hypothesis
from comboscope.runtime.doubao_client import LLMCallResult


class FakeLLM:
    last_call = None

    def complete_with_usage(self, system_prompt: str, user_prompt: str, *, agent: str, step: str) -> LLMCallResult:
        content = yaml.safe_dump(
            {
                "target_problem": "segment_underestimate",
                "hypothesis": "Evidence suggests recent target momentum is underrepresented.",
                "evidence": ["segment_underestimate_scene", "recent demand feature gap"],
                "proposed_features": [
                    {
                        "action": "add_feature",
                        "feature_name": "recent_rolling_mean",
                        "feature_type": "rolling_stat",
                        "cli_args": ["--enable-recent-rolling-mean"],
                        "code_locations": ["src/train_forecast.py"],
                        "field_sources": ["date", "entity_id", "target"],
                        "construction": "rolling target mean by entity over recent history.",
                        "validation_metrics": ["wape", "bias"],
                    }
                ],
            },
            sort_keys=False,
        )
        self.last_call = LLMCallResult(
            content=content,
            model="fake",
            agent=agent,
            step=step,
            duration_seconds=0.01,
            success=True,
        )
        return self.last_call


def test_real_business_plan_generates_evidence_backed_candidates() -> None:
    problem_context = {
        "main_problem": "segment_underestimate",
        "evidence": ["segment_underestimate_scene", "recent demand feature gap"],
        "scenario": "generic_forecast",
        "model_family": "unknown",
        "objective": "unknown",
        "entrypoint_candidates": ["src/train_forecast.py"],
    }

    hypothesis = generate_real_feature_hypothesis(problem_context, "rules", FakeLLM())
    plan = generate_real_experiment_plan(hypothesis, "trial_001", "runs/trial_001/code/train.py", problem_context)

    assert 1 <= len(plan["candidate_experiments"]) <= 3
    assert plan["scenario"] == "generic_forecast"
    assert plan["model_family"] == "unknown"
    assert plan["objective"] == "unknown"
    assert plan["source_entrypoint"] == "src/train_forecast.py"
    assert plan["changes"]
    assert all(candidate["evidence"] for candidate in plan["candidate_experiments"])
    dumped = str(plan).lower()
    assert "takeaway" not in dumped
    assert "single item" not in dumped
