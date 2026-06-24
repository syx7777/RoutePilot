from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from comboscope.runtime.skills_manager import SkillsManager


class SkillRunner:
    def __init__(self, skills_dir: str | Path, manager: SkillsManager | None = None):
        self.manager = manager or SkillsManager(skills_dir)
        self.skills_dir = self.manager.skills_root

    def _run_json(self, skill: str, script: str, args: list[str]) -> dict[str, Any]:
        result = self.manager.run_script(skill, script, args)
        return json.loads(result.stdout)

    def _run_text(self, skill: str, script: str, args: list[str]) -> str:
        result = self.manager.run_script(skill, script, args)
        return result.stdout

    def scan_experiment(self, experiment_dir: str | Path) -> dict[str, Any]:
        return self._run_json("forecast-experiment-scanner", "scan_experiment.py", ["--experiment", Path(experiment_dir).as_posix()])

    def discover_artifacts(self, experiment_dir: str | Path, ask: str) -> dict[str, Any]:
        return self._run_json(
            "forecast-experiment-scanner",
            "discover_artifacts.py",
            ["--experiment", Path(experiment_dir).as_posix(), "--ask", ask],
        )

    def analyze_code(self, experiment_dir: str | Path) -> dict[str, Any]:
        return self._run_json("forecast-code-log-analyzer", "analyze_code.py", ["--experiment", Path(experiment_dir).as_posix()])

    def parse_log(self, log_path: str | Path | None) -> dict[str, Any]:
        args = ["--log", Path(log_path).as_posix()] if log_path else []
        return self._run_json("forecast-code-log-analyzer", "parse_log.py", args)

    def calculate_metrics(self, prediction: str | Path, actual: str | Path, output: str | Path) -> dict[str, Any]:
        return self._run_json(
            "forecast-evaluation-analyzer",
            "calculate_metrics.py",
            ["--prediction", Path(prediction).as_posix(), "--actual", Path(actual).as_posix(), "--output", Path(output).as_posix()],
        )

    def tag_scenes(self, prediction: str | Path, actual: str | Path, output: str | Path) -> None:
        self._run_text(
            "forecast-evaluation-analyzer",
            "tag_scenes.py",
            ["--prediction", Path(prediction).as_posix(), "--actual", Path(actual).as_posix(), "--output", Path(output).as_posix()],
        )

    def detect_anomalies(self, metrics: str | Path, scene_metrics: str | Path, log_summary: str | Path, output: str | Path) -> dict[str, Any]:
        return self._run_json(
            "forecast-evaluation-analyzer",
            "detect_anomalies.py",
            [
                "--metrics",
                Path(metrics).as_posix(),
                "--scene-metrics",
                Path(scene_metrics).as_posix(),
                "--log-summary",
                Path(log_summary).as_posix(),
                "--output",
                Path(output).as_posix(),
            ],
        )

    def mine_badcases(self, prediction: str | Path, actual: str | Path, output: str | Path) -> None:
        self._run_text(
            "forecast-badcase-locator",
            "mine_badcases.py",
            ["--prediction", Path(prediction).as_posix(), "--actual", Path(actual).as_posix(), "--output", Path(output).as_posix()],
        )

    def build_report_context(
        self,
        *,
        ask: str,
        mode: str,
        scan_result: str | Path,
        artifact_summary: str | Path,
        code_analysis: str | Path,
        log_summary: str | Path,
        metrics: str | Path,
        scene_metrics: str | Path,
        badcases: str | Path,
        anomaly_summary: str | Path,
        output: str | Path,
    ) -> dict[str, Any]:
        return self._run_json(
            "forecast-report-writer",
            "build_report_context.py",
            [
                "--ask",
                ask,
                "--mode",
                mode,
                "--scan-result",
                Path(scan_result).as_posix(),
                "--artifact-summary",
                Path(artifact_summary).as_posix(),
                "--code-analysis",
                Path(code_analysis).as_posix(),
                "--log-summary",
                Path(log_summary).as_posix(),
                "--metrics",
                Path(metrics).as_posix(),
                "--scene-metrics",
                Path(scene_metrics).as_posix(),
                "--badcases",
                Path(badcases).as_posix(),
                "--anomaly-summary",
                Path(anomaly_summary).as_posix(),
                "--output",
                Path(output).as_posix(),
            ],
        )

    def write_report(self, context: str | Path, report: str | Path, suggestions_output: str | Path | None = None) -> None:
        raise RuntimeError("Forecast reports are LLM-led; deterministic write_report.py is disabled for runtime use.")

    def write_suggestions(
        self,
        *,
        artifact_summary: str | Path,
        log_summary: str | Path,
        code_analysis: str | Path,
        metrics: str | Path,
        scene_metrics: str | Path,
        badcases: str | Path,
        anomaly_summary: str | Path,
        output: str | Path,
    ) -> None:
        self._run_text(
            "forecast-optimization-advisor",
            "write_suggestions.py",
            [
                "--artifact-summary",
                Path(artifact_summary).as_posix(),
                "--log-summary",
                Path(log_summary).as_posix(),
                "--code-analysis",
                Path(code_analysis).as_posix(),
                "--metrics",
                Path(metrics).as_posix(),
                "--scene-metrics",
                Path(scene_metrics).as_posix(),
                "--badcases",
                Path(badcases).as_posix(),
                "--anomaly-summary",
                Path(anomaly_summary).as_posix(),
                "--output",
                Path(output).as_posix(),
            ],
        )
