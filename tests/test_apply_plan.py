from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from comboscope.core.experiment_plan import apply_experiment_plan


def test_apply_plan_updates_allowed_config_and_train_policy(fixture_exp: Path, tmp_path: Path) -> None:
    plan_path = tmp_path / "experiment_plan.yaml"
    plan_path.write_text(
        yaml.safe_dump(
            {
                "trial_id": "trial_001",
                "target_problem": "high_sales_underestimate",
                "hypothesis": {
                    "description": "high sales underestimate",
                    "evidence": ["scene metrics"],
                },
                "editable_files": ["benchmark/feature_config.yaml", "benchmark/train.py"],
                "changes": [
                    {
                        "action": "add_feature",
                        "feature_name": "combo_rolling_7d_mean",
                        "feature_type": "rolling_stat",
                        "group_by": "combo_id",
                        "window": 7,
                    }
                ],
                "expected_effect": "reduce high_sales underestimation",
                "risk": "may overfit recent demand spikes",
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )

    result = apply_experiment_plan(plan_path, fixture_exp)

    assert result["success"] is True
    assert "combo_rolling_7d_mean" in (fixture_exp / "benchmark" / "feature_config.yaml").read_text(encoding="utf-8")
    assert "combo_rolling_7d_mean" in (fixture_exp / "benchmark" / "train.py").read_text(encoding="utf-8")
    assert result["backup_info"]["files"]


def test_apply_plan_rejects_evaluate_py(fixture_exp: Path, tmp_path: Path) -> None:
    plan_path = tmp_path / "bad_plan.yaml"
    plan_path.write_text(
        yaml.safe_dump(
            {
                "trial_id": "trial_001",
                "target_problem": "high_sales_underestimate",
                "hypothesis": {"description": "bad", "evidence": ["none"]},
                "editable_files": ["benchmark/evaluate.py"],
                "changes": [
                    {
                        "action": "add_feature",
                        "feature_name": "bad",
                        "feature_type": "rolling_stat",
                    }
                ],
                "expected_effect": "none",
                "risk": "invalid",
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="not allowed"):
        apply_experiment_plan(plan_path, fixture_exp)


def test_apply_plan_accepts_trial_code_python_copy(tmp_path: Path) -> None:
    project = tmp_path / "project"
    trial_code = project / "runs" / "trial_001" / "code"
    trial_code.mkdir(parents=True)
    (trial_code / "train.py").write_text("print('trial copy')\n", encoding="utf-8")
    (trial_code / "util.py").write_text("VALUE = 1\n", encoding="utf-8")
    plan_path = tmp_path / "trial_plan.yaml"
    plan_path.write_text(
        yaml.safe_dump(
            {
                "trial_id": "trial_001",
                "target_problem": "high_sales_underestimate",
                "hypothesis": {"description": "modify copied trial code", "evidence": ["agent1 report"]},
                "editable_files": ["runs/trial_001/code/train.py", "runs/trial_001/code/util.py"],
                "changes": [{"action": "add_feature", "feature_name": "copied_trial_feature"}],
                "expected_effect": "reduce error",
                "risk": "low",
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )

    result = apply_experiment_plan(plan_path, project)

    assert result["success"] is True
    assert len(result["backup_info"]["files"]) == 2
