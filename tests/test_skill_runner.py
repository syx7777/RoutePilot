from __future__ import annotations

import json
import importlib.util
import sys
from pathlib import Path

from routepilot.runtime.artifact_adapter import metrics_csv_to_json
from routepilot.runtime.skill_runner import SkillRunner
from routepilot.runtime.skills_manager import SkillsManager


def test_skill_runner_generates_metric_and_badcase_artifacts(repo_root: Path, fixture_exp: Path, tmp_path: Path) -> None:
    runner = SkillRunner(repo_root / "skills")
    prediction = fixture_exp / "benchmark" / "data" / "prediction_seed.csv"
    actual = fixture_exp / "benchmark" / "data" / "actual.csv"

    metrics_csv = tmp_path / "metrics_summary.csv"
    scene_csv = tmp_path / "scene_metrics.csv"
    badcases_csv = tmp_path / "badcases.csv"

    metrics = runner.calculate_metrics(prediction, actual, metrics_csv)
    runner.tag_scenes(prediction, actual, scene_csv)
    runner.mine_badcases(prediction, actual, badcases_csv)

    metrics_json = metrics_csv_to_json(metrics_csv, tmp_path / "metrics.json")

    assert metrics["wape"] > 0
    assert metrics_json["sample_count"] == metrics["rows"]
    assert scene_csv.exists()
    assert badcases_csv.read_text(encoding="utf-8").count("top_underestimate") >= 1
    assert json.loads((tmp_path / "metrics.json").read_text(encoding="utf-8"))["wape"] == metrics["wape"]


def test_skill_runner_uses_skills_manager(repo_root: Path) -> None:
    runner = SkillRunner(repo_root / "skills")

    assert isinstance(runner.manager, SkillsManager)
    assert "forecast-report-writer" in {skill.name for skill in runner.manager.discover_skills()}


def test_scanner_uses_generic_header_and_directory_clues(repo_root: Path, tmp_path: Path) -> None:
    experiment = tmp_path / "exp"
    (experiment / "src").mkdir(parents=True)
    (experiment / "outputs").mkdir()
    (experiment / "archive").mkdir()
    (experiment / ".python_packages" / "vendor").mkdir(parents=True)
    (experiment / "src" / "model_pipeline.py").write_text(
        '"""销量预测训练主入口。"""\nprint("train")\n',
        encoding="utf-8",
    )
    (experiment / "outputs" / "weekly_forecast_prediction.csv").write_text("a,b\n1,2\n", encoding="utf-8")
    (experiment / "outputs" / "weekly_truth.csv").write_text("a,b\n1,2\n", encoding="utf-8")
    (experiment / "outputs" / "training.log").write_text("ok\n", encoding="utf-8")
    (experiment / "archive" / "old_prediction.csv").write_text("a,b\n1,2\n", encoding="utf-8")
    (experiment / ".python_packages" / "vendor" / "noise.py").write_text("print('noise')\n", encoding="utf-8")

    scan = SkillRunner(repo_root / "skills").scan_experiment(experiment)

    assert "src/model_pipeline.py" in scan["possible_entrypoints"]
    assert "outputs/weekly_forecast_prediction.csv" in scan["candidate_prediction_files"]
    assert "outputs/weekly_truth.csv" in scan["actual_files"]
    assert "outputs/training.log" in scan["log_files"]
    assert not any(".python_packages" in path for path in scan["code_files"])
    assert not any(path.startswith("archive/") for path in scan["candidate_prediction_files"])


def test_scanner_falls_back_to_parent_when_subdir_is_too_narrow(repo_root: Path, tmp_path: Path) -> None:
    experiment = tmp_path / "exp"
    (experiment / "src").mkdir(parents=True)
    (experiment / "outputs").mkdir()
    (experiment / "logs").mkdir()
    (experiment / "src" / "pipeline.py").write_text(
        "# sales forecast training\nprint('train')\n",
        encoding="utf-8",
    )
    (experiment / "outputs" / "prediction.csv").write_text("a,b\n1,2\n", encoding="utf-8")
    (experiment / "outputs" / "actual.csv").write_text("a,b\n1,2\n", encoding="utf-8")
    (experiment / "logs" / "train.log").write_text("ok\n", encoding="utf-8")

    scan = SkillRunner(repo_root / "skills").scan_experiment(experiment / "src")

    assert scan["experiment_dir"] == experiment.as_posix()
    assert scan["requested_dir"] == (experiment / "src").as_posix()
    assert scan["fallback_used"] is True
    assert "outputs/prediction.csv" in scan["candidate_prediction_files"]
    assert "outputs/actual.csv" in scan["actual_files"]
    assert "logs/train.log" in scan["log_files"]


def test_code_analyzer_extracts_functions_and_argparse_flags(repo_root: Path, tmp_path: Path) -> None:
    experiment = tmp_path / "exp"
    (experiment / "src").mkdir(parents=True)
    train = experiment / "src" / "train.py"
    train.write_text(
        """
import argparse


def build_features(df):
    return df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rolling_windows", "--rolling-windows", type=int, nargs="*", default=[3, 7])
    parser.add_argument("--enable_existing_feature", action="store_true")
    return parser.parse_args()
""",
        encoding="utf-8",
    )
    path = repo_root / "skills" / "forecast-code-log-analyzer" / "scripts" / "analyze_code.py"
    sys.path.insert(0, path.parent.as_posix())
    spec = importlib.util.spec_from_file_location("forecast_code_analyzer", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    analysis = module.analyze_code(experiment, ["src/train.py"])

    assert "build_features" in analysis["function_defs"]
    assert "rolling_windows" in analysis["code_identifiers"]
    assert "--rolling_windows" in analysis["argparse_args"]
    assert "--rolling-windows" in analysis["argparse_args"]
