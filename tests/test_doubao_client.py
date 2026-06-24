from __future__ import annotations

from comboscope.runtime.doubao_client import DoubaoClient


class FakeResponse:
    def raise_for_status(self) -> None:
        return None

    def json(self) -> dict:
        return {"choices": [{"message": {"content": "ok"}}]}


class FakeUsageResponse:
    def __init__(self, payload: dict):
        self.payload = payload

    def raise_for_status(self) -> None:
        return None

    def json(self) -> dict:
        return self.payload


def test_doubao_client_uses_ark_compatible_env(monkeypatch) -> None:
    calls = []

    def fake_post(url, headers, json, timeout):
        calls.append({"url": url, "headers": headers, "json": json, "timeout": timeout})
        return FakeResponse()

    monkeypatch.setenv("DOUBAO_API_KEY", "test-key")
    monkeypatch.setenv("DOUBAO_BASE_URL", "https://ark.cn-beijing.volces.com/api/v3")
    monkeypatch.setenv("DOUBAO_MODEL", "test-endpoint")
    monkeypatch.setattr("comboscope.runtime.llm_client.requests.post", fake_post)

    client = DoubaoClient()

    assert client.available() is True
    assert client.skylark("hello") == "ok"
    assert calls[0]["url"] == "https://ark.cn-beijing.volces.com/api/v3/chat/completions"
    assert calls[0]["headers"]["Authorization"] == "Bearer test-key"
    assert calls[0]["json"]["model"] == "test-endpoint"
    assert calls[0]["json"]["messages"] == [{"role": "user", "content": "hello"}]


def test_doubao_client_records_real_usage(monkeypatch) -> None:
    def fake_post(url, headers, json, timeout):
        return FakeUsageResponse(
            {
                "choices": [{"message": {"content": "ok"}}],
                "usage": {"prompt_tokens": 11, "completion_tokens": 3, "total_tokens": 14},
            }
        )

    monkeypatch.setenv("DOUBAO_API_KEY", "test-key")
    monkeypatch.setenv("DOUBAO_BASE_URL", "https://ark.cn-beijing.volces.com/api/v3")
    monkeypatch.setenv("DOUBAO_MODEL", "test-endpoint")
    monkeypatch.setattr("comboscope.runtime.llm_client.requests.post", fake_post)

    result = DoubaoClient().complete_with_usage("system", "user prompt", agent="Agent1", step="Generate")

    assert result.content == "ok"
    assert result.usage["total_tokens"] == 14
    assert result.estimated is False
    assert result.agent == "Agent1"
    assert result.step == "Generate"


def test_doubao_client_estimates_usage_when_api_omits_usage(monkeypatch) -> None:
    def fake_post(url, headers, json, timeout):
        return FakeUsageResponse({"choices": [{"message": {"content": "ok"}}]})

    monkeypatch.setenv("DOUBAO_API_KEY", "test-key")
    monkeypatch.setenv("DOUBAO_BASE_URL", "https://ark.cn-beijing.volces.com/api/v3")
    monkeypatch.setenv("DOUBAO_MODEL", "test-endpoint")
    monkeypatch.setattr("comboscope.runtime.llm_client.requests.post", fake_post)

    result = DoubaoClient().complete_with_usage("", "hello world", agent="Agent2", step="Plan")

    assert result.content == "ok"
    assert result.estimated is True
    assert result.usage["total_tokens"] >= 1
    assert "test-key" not in result.summary
