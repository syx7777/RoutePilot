from __future__ import annotations

import sys
from pathlib import Path

import main
import loop
from loop import run_loop


def test_main_run_fails_fast_without_llm(monkeypatch, capsys, fixture_exp: Path, tmp_path: Path) -> None:
    monkeypatch.delenv("DOUBAO_API_KEY", raising=False)
    output_dir = tmp_path / "trial_001"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "main.py",
            "run",
            "--experiment",
            fixture_exp.as_posix(),
            "--ask",
            "分析套餐预测误差，提出一个特征实验并验证效果",
            "--model",
            "doubao",
            "--orchestrator",
            "langgraph",
            "--output",
            output_dir.as_posix(),
        ],
    )

    assert main.main() == 1

    captured = capsys.readouterr().out
    assert "RoutePilot run failed" in captured
    assert "SelectArtifacts" in captured
    assert "error_report:" in captured
    assert (output_dir / "error_report.md").exists()


def test_main_run_can_read_experiment_path_from_ask_before_llm_failure(monkeypatch, capsys, fixture_exp: Path, tmp_path: Path) -> None:
    monkeypatch.delenv("DOUBAO_API_KEY", raising=False)
    output_dir = tmp_path / "trial_001"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "main.py",
            "run",
            "--ask",
            f"项目地址 {fixture_exp.as_posix()}，请分析通用预测误差并提出一个特征实验",
            "--model",
            "doubao",
            "--orchestrator",
            "langgraph",
            "--output",
            output_dir.as_posix(),
        ],
    )

    assert main.main() == 1

    captured = capsys.readouterr().out
    assert "RoutePilot run failed" in captured
    assert "error_report:" in captured
    assert (output_dir / "error_report.md").exists()


def test_loop_prints_trial_and_final_summary(monkeypatch, capsys, fixture_exp: Path, tmp_path: Path, repo_root: Path) -> None:
    monkeypatch.delenv("DOUBAO_API_KEY", raising=False)

    try:
        run_loop(
            experiment=fixture_exp,
            ask="基于当前评测结果自动尝试特征构建优化并输出最终效果",
            max_trials=2,
            model="doubao",
            orchestrator="langgraph",
            output=tmp_path,
            repo_root=repo_root,
        )
    except RuntimeError:
        pass

    captured = capsys.readouterr().out
    assert "trial_001: failed=" in captured
    assert "SelectArtifacts" in captured
    assert "trial_001: error_report=" in captured
    assert not (tmp_path / "trial_002").exists()


def test_main_run_failure_prints_error_report(monkeypatch, capsys, fixture_exp: Path, tmp_path: Path) -> None:
    output_dir = tmp_path / "trial_001"

    def fake_run_once(request):
        Path(request["output_dir"]).mkdir(parents=True, exist_ok=True)
        (Path(request["output_dir"]) / "error_report.md").write_text("# error\n", encoding="utf-8")
        raise RuntimeError("planned failure")

    monkeypatch.setattr(main, "run_once", fake_run_once)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "main.py",
            "run",
            "--experiment",
            fixture_exp.as_posix(),
            "--ask",
            "分析预测误差",
            "--model",
            "doubao",
            "--orchestrator",
            "langgraph",
            "--output",
            output_dir.as_posix(),
        ],
    )

    assert main.main() == 1

    captured = capsys.readouterr().out
    assert "RoutePilot run failed" in captured
    assert "error_report:" in captured


def test_loop_failure_prints_trial_error_report(monkeypatch, capsys, fixture_exp: Path, tmp_path: Path, repo_root: Path) -> None:
    def fake_run_once(request):
        trial_dir = Path(request["output_dir"])
        trial_dir.mkdir(parents=True, exist_ok=True)
        (trial_dir / "error_report.md").write_text("# error\n", encoding="utf-8")
        raise RuntimeError("trial failed")

    monkeypatch.setattr(loop, "run_once", fake_run_once)

    try:
        run_loop(
            experiment=fixture_exp,
            ask="分析预测误差",
            max_trials=1,
            model="doubao",
            orchestrator="langgraph",
            output=tmp_path,
            repo_root=repo_root,
        )
    except RuntimeError:
        pass

    captured = capsys.readouterr().out
    assert "trial_001: failed=trial failed" in captured
    assert "trial_001: error_report=" in captured
