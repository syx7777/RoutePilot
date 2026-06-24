from __future__ import annotations

import json
import re
import time
import traceback
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

from comboscope.runtime.doubao_client import LLMCallResult


class AgentRunRecorder:
    def __init__(self, output_dir: str | Path):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.status_path = self.output_dir / "agent_status.json"
        self.timeline_path = self.output_dir / "agent_timeline.jsonl"
        self.llm_calls_path = self.output_dir / "llm_calls.jsonl"
        self.token_usage_path = self.output_dir / "token_usage.json"
        self.artifact_index_path = self.output_dir / "artifact_index.md"
        self.error_json_path = self.output_dir / "error_report.json"
        self.error_md_path = self.output_dir / "error_report.md"
        self._ensure_files()

    @contextmanager
    def step(self, agent: str, step: str, artifacts: dict[str, str | Path] | None = None) -> Iterator[None]:
        started = time.perf_counter()
        started_at = _now()
        self._mark_step(agent, step, "running", started_at, artifacts or {})
        self._append_timeline({"event": "start", "agent": agent, "step": step, "ts": started_at, "artifacts": _stringify_paths(artifacts or {})})
        try:
            yield
        except Exception as exc:
            duration = time.perf_counter() - started
            self._finish_step(agent, step, "failed", duration, str(exc), artifacts or {})
            self.write_error_report(agent, step, exc)
            self.write_artifact_index()
            raise
        else:
            duration = time.perf_counter() - started
            self._finish_step(agent, step, "success", duration, None, artifacts or {})
            self.write_artifact_index()

    def record_llm_call(self, result: LLMCallResult | None) -> None:
        if result is None:
            return
        yaml_failure_path = self._write_yaml_failure_raw_output(result)
        raw_response_path = self._write_llm_raw_response(result)
        record = {
            "ts": _now(),
            "agent": result.agent,
            "step": result.step,
            "model": result.model,
            "duration_seconds": result.duration_seconds,
            "usage": result.usage,
            "estimated": result.estimated,
            "available": result.available,
            "success": result.success,
            "summary": result.summary,
            "error": result.error,
            "attempt_count": getattr(result, "attempt_count", 0),
            "retry_count": getattr(result, "retry_count", 0),
            "retry_errors": getattr(result, "retry_errors", []),
            "api_mode": getattr(result, "api_mode", ""),
            "request_url": getattr(result, "request_url", ""),
            "response_status_code": getattr(result, "response_status_code", None),
            "response_url": getattr(result, "response_url", ""),
            "empty_content_reason": getattr(result, "empty_content_reason", ""),
            "raw_response_path": raw_response_path,
            "yaml_parse_status": getattr(result, "yaml_parse_status", ""),
            "yaml_parse_error": getattr(result, "yaml_parse_error", None),
            "yaml_repair_attempted": bool(getattr(result, "yaml_repair_attempted", False)),
            "yaml_repair_success": bool(getattr(result, "yaml_repair_success", False)),
            "yaml_control_chars": getattr(result, "yaml_control_chars", {}),
            "yaml_failure_raw_output_path": yaml_failure_path,
        }
        self._append_jsonl(self.llm_calls_path, record)
        self._update_token_usage(result)

    def write_artifact_index(self) -> None:
        lines = [
            "# ComboScope Artifact Index",
            "",
            "## Agent1 结果",
        ]
        lines.extend(self._artifact_lines(
            [
                "audit/scan_result.json",
                "audit/artifact_summary.json",
                "agent1/artifact_contract.json",
                "agent1/badcase_diagnosis.md",
                "audit/code_analysis.json",
                "audit/log_summary.json",
                "evaluation/metrics.json",
                "evaluation/scene_metrics.csv",
                "evaluation/badcases.csv",
                "agent1/problem_context.json",
                "agent1/analysis_report.md",
                "agent1/feature_hypothesis.yaml",
                "agent1/experiment_plan.yaml",
                "agent1/agent1_program.md",
                "reports/forecast_report.md",
                "reports/optimization_suggestions.md",
            ]
        ))
        lines.extend(["", "## Agent2 结果"])
        lines.extend(self._artifact_lines(
            [
                "agent2/agent2_execution_plan.yaml",
                "agent2/source_evaluation_context.json",
                "agent2/output_contract.json",
                "code/train.py",
                "code/agent2_code_modification.yaml",
                "code/agent2_feature_application_audit.yaml",
                "data/input_manifest.json",
                "logs/train.log",
                "logs/eval.log",
                "outputs/real_outputs",
                "standardized/prediction.csv",
                "standardized/actual.csv",
                "agent2/run_status.json",
                "evaluation/new_metrics.json",
                "evaluation/metric_comparison.json",
                "agent2/review_result.json",
                "agent2/experiment_review.md",
                "reports/final_report_context.json",
                "final_report.md",
            ]
        ))
        lines.extend(
            [
                "",
                "## 归档视图",
                f"- agent1/: `{(self.output_dir / 'agent1').as_posix()}`",
                f"- agent2/: `{(self.output_dir / 'agent2').as_posix()}`",
                f"- evaluation/: `{(self.output_dir / 'evaluation').as_posix()}`",
                f"- reports/: `{(self.output_dir / 'reports').as_posix()}`",
                f"- audit/: `{(self.output_dir / 'audit').as_posix()}`",
                f"- archive/: `{(self.output_dir / 'archive').as_posix()}`",
                f"- archive/archive_manifest.json: `{(self.output_dir / 'archive' / 'archive_manifest.json').as_posix()}`",
                "",
                "## 运行审计",
                f"- audit/agent_status.json: `{(self.output_dir / 'audit' / 'agent_status.json').as_posix()}`",
                f"- audit/agent_timeline.jsonl: `{(self.output_dir / 'audit' / 'agent_timeline.jsonl').as_posix()}`",
                f"- audit/llm_calls.jsonl: `{(self.output_dir / 'audit' / 'llm_calls.jsonl').as_posix()}`",
                f"- audit/llm_yaml_failures/: `{(self.output_dir / 'audit' / 'llm_yaml_failures').as_posix()}`",
                f"- audit/llm_raw_responses/: `{(self.output_dir / 'audit' / 'llm_raw_responses').as_posix()}`",
                f"- audit/token_usage.json: `{(self.output_dir / 'audit' / 'token_usage.json').as_posix()}`",
            ]
        )
        if self.error_json_path.exists():
            lines.extend(["", "## 异常报告", f"- error_report.md: `{self.error_md_path.as_posix()}`"])
        self.artifact_index_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    def write_error_report(self, agent: str, step: str, exc: BaseException) -> None:
        completed = sorted(path.name for path in self.output_dir.iterdir() if path.is_file())
        error = {
            "agent": agent,
            "step": step,
            "error": str(exc),
            "error_type": type(exc).__name__,
            "traceback": traceback.format_exc(limit=8),
            "completed_artifacts": completed,
            "human_action": "查看 agent_timeline.jsonl、trace.jsonl 和相关日志后决定重试、补充产物或修复实验配置。",
        }
        self.error_json_path.write_text(json.dumps(error, ensure_ascii=False, indent=2), encoding="utf-8")
        self.error_md_path.write_text(
            "\n".join(
                [
                    "# ComboScope Error Report",
                    "",
                    "## 需要人工介入",
                    f"- Agent: {agent}",
                    f"- Step: {step}",
                    f"- Error: {error['error']}",
                    "",
                    "## 已完成产物",
                    *[f"- {item}" for item in completed],
                    "",
                    "## 建议",
                    f"- {error['human_action']}",
                ]
            )
            + "\n",
            encoding="utf-8",
        )

    def _ensure_files(self) -> None:
        if not self.status_path.exists():
            self.status_path.write_text(
                json.dumps({"agents": {}, "updated_at": _now()}, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
        if not self.token_usage_path.exists():
            self.token_usage_path.write_text(
                json.dumps(
                    {
                        "prompt_tokens": 0,
                        "completion_tokens": 0,
                        "total_tokens": 0,
                        "estimated_calls": 0,
                        "real_usage_calls": 0,
                        "by_agent": {},
                        "by_step": {},
                    },
                    ensure_ascii=False,
                    indent=2,
                ),
                encoding="utf-8",
            )
        self.timeline_path.touch(exist_ok=True)
        self.llm_calls_path.touch(exist_ok=True)

    def _mark_step(self, agent: str, step: str, status: str, started_at: str, artifacts: dict[str, str | Path]) -> None:
        data = self._read_status()
        agent_data = data["agents"].setdefault(agent, {"status": "pending", "duration_seconds": 0.0, "steps": {}})
        agent_data["status"] = status
        agent_data["current_step"] = step
        agent_data.setdefault("started_at", started_at)
        agent_data["steps"][step] = {
            "status": status,
            "started_at": started_at,
            "artifacts": _stringify_paths(artifacts),
        }
        data["updated_at"] = _now()
        self._write_status(data)

    def _finish_step(self, agent: str, step: str, status: str, duration: float, error: str | None, artifacts: dict[str, str | Path]) -> None:
        data = self._read_status()
        agent_data = data["agents"].setdefault(agent, {"status": "pending", "duration_seconds": 0.0, "steps": {}})
        agent_data["status"] = "failed" if status == "failed" else "success"
        agent_data["current_step"] = step
        agent_data["duration_seconds"] = float(agent_data.get("duration_seconds", 0.0)) + duration
        agent_data["finished_at"] = _now()
        step_data = agent_data["steps"].setdefault(step, {})
        step_data.update(
            {
                "status": status,
                "duration_seconds": duration,
                "finished_at": _now(),
                "artifacts": _stringify_paths(artifacts),
            }
        )
        if error:
            step_data["error"] = error
            agent_data["error"] = error
        data["updated_at"] = _now()
        self._write_status(data)
        self._append_timeline(
            {
                "event": status,
                "agent": agent,
                "step": step,
                "duration_seconds": duration,
                "ts": _now(),
                "error": error,
                "artifacts": _stringify_paths(artifacts),
            }
        )

    def _update_token_usage(self, result: LLMCallResult) -> None:
        usage = self._read_json(self.token_usage_path)
        prompt = int(result.usage.get("prompt_tokens", 0))
        completion = int(result.usage.get("completion_tokens", 0))
        total = int(result.usage.get("total_tokens", prompt + completion))
        usage["prompt_tokens"] = int(usage.get("prompt_tokens", 0)) + prompt
        usage["completion_tokens"] = int(usage.get("completion_tokens", 0)) + completion
        usage["total_tokens"] = int(usage.get("total_tokens", 0)) + total
        if result.estimated:
            usage["estimated_calls"] = int(usage.get("estimated_calls", 0)) + 1
        elif result.success and result.available:
            usage["real_usage_calls"] = int(usage.get("real_usage_calls", 0)) + 1
        self._bump_bucket(usage.setdefault("by_agent", {}), result.agent, prompt, completion, total)
        self._bump_bucket(usage.setdefault("by_step", {}), f"{result.agent}.{result.step}", prompt, completion, total)
        self.token_usage_path.write_text(json.dumps(usage, ensure_ascii=False, indent=2), encoding="utf-8")

    def _write_yaml_failure_raw_output(self, result: LLMCallResult) -> str:
        if getattr(result, "yaml_parse_status", "") != "failed":
            return ""
        content = getattr(result, "content", "") or ""
        if not content:
            return ""
        failure_dir = self.output_dir / "audit" / "llm_yaml_failures"
        failure_dir.mkdir(parents=True, exist_ok=True)
        safe_agent = re.sub(r"[^0-9A-Za-z_]+", "_", str(result.agent)).strip("_") or "agent"
        safe_step = re.sub(r"[^0-9A-Za-z_]+", "_", str(result.step)).strip("_") or "step"
        sequence = len(list(failure_dir.glob("*.txt"))) + 1
        path = failure_dir / f"{sequence:02d}_{safe_agent}_{safe_step}.txt"
        path.write_text(content, encoding="utf-8", errors="replace")
        return path.as_posix()

    def _write_llm_raw_response(self, result: LLMCallResult) -> str:
        raw_response = getattr(result, "raw_response", None)
        if not raw_response:
            return ""
        should_write = (
            not bool(getattr(result, "success", False))
            or not (getattr(result, "content", "") or "")
            or getattr(result, "yaml_parse_status", "") == "failed"
        )
        if not should_write:
            return ""
        response_dir = self.output_dir / "audit" / "llm_raw_responses"
        response_dir.mkdir(parents=True, exist_ok=True)
        safe_agent = re.sub(r"[^0-9A-Za-z_]+", "_", str(result.agent)).strip("_") or "agent"
        safe_step = re.sub(r"[^0-9A-Za-z_]+", "_", str(result.step)).strip("_") or "step"
        sequence = len(list(response_dir.glob("*.json"))) + 1
        path = response_dir / f"{sequence:02d}_{safe_agent}_{safe_step}.json"
        path.write_text(json.dumps(raw_response, ensure_ascii=False, indent=2), encoding="utf-8")
        return path.as_posix()

    def _bump_bucket(self, buckets: dict[str, Any], key: str, prompt: int, completion: int, total: int) -> None:
        bucket = buckets.setdefault(key, {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0})
        bucket["prompt_tokens"] += prompt
        bucket["completion_tokens"] += completion
        bucket["total_tokens"] += total

    def _artifact_lines(self, rel_paths: list[str]) -> list[str]:
        lines = []
        for rel in rel_paths:
            path = self.output_dir / rel
            if not path.exists():
                legacy = self.output_dir / Path(rel).name
                if legacy.exists():
                    path = legacy
            if path.exists():
                display = path.relative_to(self.output_dir).as_posix()
                lines.append(f"- {display}: `{path.as_posix()}`")
        if not lines:
            lines.append("- 暂无产物。")
        return lines

    def _append_timeline(self, record: dict[str, Any]) -> None:
        self._append_jsonl(self.timeline_path, record)

    def _append_jsonl(self, path: Path, record: dict[str, Any]) -> None:
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    def _read_status(self) -> dict[str, Any]:
        return self._read_json(self.status_path)

    def _write_status(self, data: dict[str, Any]) -> None:
        self.status_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")

    def _read_json(self, path: Path) -> dict[str, Any]:
        if not path.exists():
            return {}
        return json.loads(path.read_text(encoding="utf-8"))


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _stringify_paths(values: dict[str, str | Path]) -> dict[str, str]:
    return {key: Path(value).as_posix() for key, value in values.items()}
