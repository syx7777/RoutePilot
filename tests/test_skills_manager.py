from __future__ import annotations

import pytest

from routepilot.runtime.skills_manager import SkillsManager


def test_skills_manager_discovers_forecast_skill_metadata(repo_root):
    manager = SkillsManager(repo_root / "skills")

    skills = {skill.name: skill for skill in manager.discover_skills()}

    assert "forecast-evaluation-analyzer" in skills
    assert "forecast-optimization-case-reference" in skills
    assert "forecast-trial-codegen" in skills
    assert "prediction 和 actual" in skills["forecast-evaluation-analyzer"].description
    assert "真实预测实验修改案例" in skills["forecast-optimization-case-reference"].description
    assert "Agent2" in skills["forecast-trial-codegen"].description
    assert skills["forecast-evaluation-analyzer"].path.name == "SKILL.md"


def test_skills_manager_loads_only_requested_skill_body(repo_root):
    manager = SkillsManager(repo_root / "skills")

    skill = manager.load_skill("forecast-optimization-advisor")

    assert skill.name == "forecast-optimization-advisor"
    assert "每条建议必须引用证据" in skill.content
    assert "销量预测实验评测报告" not in skill.content


def test_skills_manager_blocks_reference_path_traversal(repo_root):
    manager = SkillsManager(repo_root / "skills")

    with pytest.raises(ValueError, match="outside skill directory"):
        manager.load_reference("forecast-evaluation-analyzer", "../using-forecast/SKILL.md")


def test_skills_manager_runs_skill_script(repo_root, fixture_exp, tmp_path):
    manager = SkillsManager(repo_root / "skills")

    result = manager.run_script(
        "forecast-experiment-scanner",
        "scan_experiment.py",
        ["--experiment", fixture_exp.as_posix()],
    )

    assert result.returncode == 0
    assert "prediction_seed.csv" in result.stdout
