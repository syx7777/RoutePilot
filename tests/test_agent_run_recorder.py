from __future__ import annotations

import json
from pathlib import Path

from routepilot.runtime.agent_run_recorder import AgentRunRecorder
from routepilot.runtime.doubao_client import LLMCallResult


def test_agent_run_recorder_writes_status_timeline_tokens_and_index(tmp_path: Path) -> None:
    recorder = AgentRunRecorder(tmp_path)

    with recorder.step("Agent1", "EvaluateAndDiagnose", artifacts={"problem": tmp_path / "problem_context.json"}):
        (tmp_path / "problem_context.json").write_text("{}", encoding="utf-8")
        recorder.record_llm_call(
            LLMCallResult(
                content="ok",
                model="m",
                agent="Agent1",
                step="Generate",
                duration_seconds=0.1,
                usage={"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
                estimated=False,
                available=True,
                success=True,
                summary="safe summary",
            )
        )

    recorder.write_artifact_index()

    status = json.loads((tmp_path / "agent_status.json").read_text(encoding="utf-8"))
    tokens = json.loads((tmp_path / "token_usage.json").read_text(encoding="utf-8"))
    timeline = (tmp_path / "agent_timeline.jsonl").read_text(encoding="utf-8")
    index = (tmp_path / "artifact_index.md").read_text(encoding="utf-8")

    assert status["agents"]["Agent1"]["status"] == "success"
    assert status["agents"]["Agent1"]["duration_seconds"] >= 0
    assert tokens["total_tokens"] == 15
    assert tokens["by_agent"]["Agent1"]["total_tokens"] == 15
    assert "EvaluateAndDiagnose" in timeline
    assert "Agent1 结果" in index
    assert "problem_context.json" in index


def test_agent_run_recorder_writes_error_report_on_failure(tmp_path: Path) -> None:
    recorder = AgentRunRecorder(tmp_path)

    try:
        with recorder.step("Agent2", "RunExperiment"):
            raise RuntimeError("boom")
    except RuntimeError:
        pass

    error = json.loads((tmp_path / "error_report.json").read_text(encoding="utf-8"))
    text = (tmp_path / "error_report.md").read_text(encoding="utf-8")

    assert error["agent"] == "Agent2"
    assert error["step"] == "RunExperiment"
    assert "boom" in error["error"]
    assert "需要人工介入" in text


def test_agent_run_recorder_writes_raw_response_for_empty_llm_content(tmp_path: Path) -> None:
    recorder = AgentRunRecorder(tmp_path)

    recorder.record_llm_call(
        LLMCallResult(
            content="",
            model="glm-5.1",
            agent="Agent1",
            step="SelectArtifacts",
            duration_seconds=0.1,
            usage={"prompt_tokens": 5, "completion_tokens": 0, "total_tokens": 5},
            estimated=True,
            available=True,
            success=False,
            summary="safe summary",
            error="empty LLM response content",
            api_mode="responses",
            request_url="https://llm.example.com/v1/responses",
            response_status_code=200,
            raw_response={"output_text": "", "usage": {"input_tokens": 5, "output_tokens": 0}},
            empty_content_reason="responses output_text was empty",
        )
    )

    records = [json.loads(line) for line in (tmp_path / "llm_calls.jsonl").read_text(encoding="utf-8").splitlines()]
    record = records[0]
    raw_path = Path(record["raw_response_path"])

    assert record["request_url"] == "https://llm.example.com/v1/responses"
    assert record["response_status_code"] == 200
    assert record["empty_content_reason"] == "responses output_text was empty"
    assert raw_path.exists()
    assert json.loads(raw_path.read_text(encoding="utf-8"))["output_text"] == ""
