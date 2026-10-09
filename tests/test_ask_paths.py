from __future__ import annotations

from pathlib import Path

from routepilot.runtime.ask_paths import resolve_experiment_dir


def test_resolve_experiment_dir_uses_cli_path_first(tmp_path: Path) -> None:
    cli_dir = tmp_path / "cli_exp"
    ask_dir = tmp_path / "ask_exp"
    cli_dir.mkdir()
    ask_dir.mkdir()

    result = resolve_experiment_dir(cli_dir.as_posix(), f"请分析项目 {ask_dir.as_posix()} 的预测误差")

    assert result == cli_dir.resolve()


def test_resolve_experiment_dir_extracts_existing_path_from_ask(tmp_path: Path) -> None:
    ask_dir = tmp_path / "forecast_exp"
    ask_dir.mkdir()

    result = resolve_experiment_dir(None, f"分析预测误差，项目地址：{ask_dir.as_posix()}，输出报告")

    assert result == ask_dir.resolve()
