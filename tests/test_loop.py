from __future__ import annotations

import json
from pathlib import Path

import yaml

import loop
from loop import run_loop
from routepilot.runtime.doubao_client import LLMCallResult


class FakeBestTrialClient:
    last_call = None

    def complete_with_usage(self, system_prompt: str, user_prompt: str, *, agent: str, step: str) -> LLMCallResult:
        self.last_call = LLMCallResult(
            content="best_trial_id: trial_001\ndecision_basis: lowest objective\nrecommended_action: keep reviewing\n",
            model="fake",
            agent=agent,
            step=step,
            duration_seconds=0.01,
            usage={"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
            success=True,
        )
        return self.last_call


def _fake_trial_runner(candidate_count: int | None = None):
    def fake_run_once(request):
        trial_dir = Path(request["output_dir"])
        (trial_dir / "agent2").mkdir(parents=True, exist_ok=True)
        (trial_dir / "evaluation").mkdir(parents=True, exist_ok=True)
        (trial_dir / "agent1").mkdir(parents=True, exist_ok=True)
        (trial_dir / "audit").mkdir(parents=True, exist_ok=True)
        (trial_dir / "agent2" / "review_result.json").write_text(
            json.dumps({"decision": "rollback", "reason": "wape improvement is below threshold"}, ensure_ascii=False),
            encoding="utf-8",
        )
        (trial_dir / "evaluation" / "metric_comparison.json").write_text(
            json.dumps(
                {
                    "primary_metric_label": "wape",
                    "primary_delta": 0.01,
                    "new_primary": 0.42,
                    "wape_delta": 0.01,
                },
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        (trial_dir / "audit" / "agent_status.json").write_text(
            json.dumps({"agents": {"Agent1": {"duration_seconds": 0.1}, "Agent2": {"duration_seconds": 0.2}}}),
            encoding="utf-8",
        )
        (trial_dir / "audit" / "token_usage.json").write_text(json.dumps({"total_tokens": 10}), encoding="utf-8")
        if candidate_count:
            candidates = [
                {
                    "experiment_id": f"exp_{index}",
                    "title": f"Experiment {index}",
                    "priority": index,
                    "feature_actions": [{"action": "add_feature", "feature_name": f"feature_{index}"}],
                    "evidence": ["evidence"],
                    "expected_effect": "validate",
                    "risk": "low",
                }
                for index in range(1, candidate_count + 1)
            ]
            (trial_dir / "agent1" / "candidate_experiments.yaml").write_text(
                yaml.safe_dump({"candidate_experiments": candidates}, sort_keys=False),
                encoding="utf-8",
            )
        (trial_dir / "final_report.md").write_text("# trial report\n", encoding="utf-8")
        return {"final_report_path": (trial_dir / "final_report.md").as_posix()}

    return fake_run_once


def test_loop_creates_history(monkeypatch, fixture_exp: Path, tmp_path: Path, repo_root: Path) -> None:
    monkeypatch.setattr(loop, "run_once", _fake_trial_runner())
    monkeypatch.setattr(loop, "create_llm_client", lambda provider=None, model=None: FakeBestTrialClient())

    run_loop(
        experiment=fixture_exp,
        ask="基于当前评测结果自动尝试特征构建优化并输出最终效果",
        max_trials=2,
        model="doubao",
        orchestrator="langgraph",
        output=tmp_path,
        repo_root=repo_root,
    )

    assert (tmp_path / "run_history.csv").exists()
    assert (tmp_path / "final_report.md").exists()
    assert (tmp_path / "best_trial_review.json").exists()
    assert (tmp_path / "best_trial_review.md").exists()
    assert (tmp_path / "trial_001" / "agent2" / "review_result.json").exists()


def test_loop_stops_at_llm_generated_candidate_experiment_count(
    monkeypatch, fixture_exp: Path, tmp_path: Path, repo_root: Path
) -> None:
    monkeypatch.setattr(loop, "run_once", _fake_trial_runner(candidate_count=2))
    monkeypatch.setattr(loop, "create_llm_client", lambda provider=None, model=None: FakeBestTrialClient())

    history = run_loop(
        experiment=fixture_exp,
        ask="分析预测误差，提出n个有效特征实验并验证结果",
        max_trials=5,
        model="doubao",
        orchestrator="langgraph",
        output=tmp_path,
        repo_root=repo_root,
    )

    assert len(history) == 2
    assert not (tmp_path / "trial_003").exists()
