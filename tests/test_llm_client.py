from __future__ import annotations

import importlib

import requests

from routepilot.runtime.llm_client import LLMClient, create_llm_client


class FakeResponse:
    def raise_for_status(self) -> None:
        return None

    def json(self) -> dict:
        return {
            "choices": [{"message": {"content": "ok"}}],
            "usage": {"prompt_tokens": 5, "completion_tokens": 2, "total_tokens": 7},
        }


class FakeResponsesResponse:
    status_code = 200
    url = "https://llm.example.com/v1/responses"

    def raise_for_status(self) -> None:
        return None

    def json(self) -> dict:
        return {
            "output_text": "ok",
            "usage": {"input_tokens": 5, "output_tokens": 2, "total_tokens": 7},
        }


class FakeEmptyResponsesResponse:
    status_code = 200
    url = "https://llm.example.com/v1/responses"

    def raise_for_status(self) -> None:
        return None

    def json(self) -> dict:
        return {
            "output_text": "",
            "usage": {"input_tokens": 5, "output_tokens": 0, "total_tokens": 5},
        }


class FakeStreamResponse:
    def raise_for_status(self) -> None:
        return None

    def iter_lines(self, decode_unicode: bool = False):
        lines = [
            'data: {"choices":[{"delta":{"content":"hel"}}]}',
            'data: {"choices":[{"delta":{"content":"lo"}}]}',
            "data: [DONE]",
        ]
        yield from lines


class FakeResponsesStreamResponse:
    def raise_for_status(self) -> None:
        return None

    def iter_lines(self, decode_unicode: bool = False):
        lines = [
            'data: {"type":"response.output_text.delta","delta":"hel"}',
            'data: {"type":"response.output_text.delta","delta":"lo"}',
            'data: {"type":"response.output_text.done","text":"hello"}',
            'data: {"type":"response.completed","response":{"usage":{"input_tokens":5,"output_tokens":2,"total_tokens":7}}}',
            "data: [DONE]",
        ]
        yield from lines


class FakeResponsesDoneOnlyStreamResponse:
    def raise_for_status(self) -> None:
        return None

    def iter_lines(self, decode_unicode: bool = False):
        lines = [
            'data: {"type":"response.output_text.done","text":"hello"}',
            'data: {"type":"response.completed","response":{"usage":{"input_tokens":5,"output_tokens":2,"total_tokens":7}}}',
            "data: [DONE]",
        ]
        yield from lines


class FakeUtf8StreamResponse:
    def raise_for_status(self) -> None:
        return None

    def iter_lines(self, decode_unicode: bool = False):
        assert decode_unicode is False
        lines = [
            'data: {"choices":[{"delta":{"content":"预测"}}]}'.encode("utf-8"),
            'data: {"choices":[{"delta":{"content":"报告"}}]}'.encode("utf-8"),
            b"data: [DONE]",
        ]
        yield from lines


class FakeStatusResponse:
    def __init__(self, status_code: int, message: str):
        self.status_code = status_code
        self.message = message

    def raise_for_status(self) -> None:
        error = requests.exceptions.HTTPError(self.message)
        error.response = self
        raise error


def test_create_llm_client_uses_deepseek_provider_env(monkeypatch) -> None:
    calls = []

    def fake_post(url, headers, json, timeout):
        calls.append({"url": url, "headers": headers, "json": json, "timeout": timeout})
        return FakeResponse()

    monkeypatch.setenv("DEEPSEEK_API_KEY", "deepseek-key")
    monkeypatch.setenv("DEEPSEEK_MODEL", "deepseek-chat")
    monkeypatch.setattr("routepilot.runtime.llm_client.requests.post", fake_post)

    client = create_llm_client(provider="deepseek")
    result = client.complete_with_usage("system", "hello", agent="Agent1", step="Plan")

    assert isinstance(client, LLMClient)
    assert client.provider == "deepseek"
    assert client.available() is True
    assert result.content == "ok"
    assert result.model == "deepseek-chat"
    assert result.api_mode == "chat_completions"
    assert calls[0]["url"] == "https://api.deepseek.com/chat/completions"
    assert calls[0]["headers"]["Authorization"] == "Bearer deepseek-key"
    assert calls[0]["json"]["model"] == "deepseek-chat"


def test_llm_client_retries_proxy_error_and_records_attempts(monkeypatch) -> None:
    calls = []

    def fake_post(url, headers, json, timeout):
        calls.append({"url": url, "headers": headers, "json": json, "timeout": timeout})
        if len(calls) < 3:
            raise requests.exceptions.ProxyError("proxy disconnected")
        return FakeResponse()

    monkeypatch.setenv("DEEPSEEK_API_KEY", "deepseek-key")
    monkeypatch.setenv("DEEPSEEK_MODEL", "deepseek-chat")
    monkeypatch.setenv("ROUTEPILOT_LLM_RETRIES", "3")
    monkeypatch.setenv("ROUTEPILOT_LLM_RETRY_BACKOFF_SECONDS", "0")
    monkeypatch.setattr("routepilot.runtime.llm_client.requests.post", fake_post)

    client = create_llm_client(provider="deepseek")
    result = client.complete_with_usage("system", "hello", agent="Agent2", step="GenerateCodeEdits")

    assert result.success is True
    assert result.content == "ok"
    assert len(calls) == 3
    assert result.attempt_count == 3
    assert result.retry_count == 2
    assert len(result.retry_errors) == 2


def test_llm_client_retries_transient_http_status(monkeypatch) -> None:
    calls = []

    def fake_post(url, headers, json, timeout):
        calls.append(url)
        if len(calls) == 1:
            return FakeStatusResponse(503, "service unavailable")
        return FakeResponse()

    monkeypatch.setenv("DEEPSEEK_API_KEY", "deepseek-key")
    monkeypatch.setenv("DEEPSEEK_MODEL", "deepseek-chat")
    monkeypatch.setenv("ROUTEPILOT_LLM_RETRIES", "2")
    monkeypatch.setenv("ROUTEPILOT_LLM_RETRY_BACKOFF_SECONDS", "0")
    monkeypatch.setattr("routepilot.runtime.llm_client.requests.post", fake_post)

    result = create_llm_client(provider="deepseek").complete_with_usage("", "hello")

    assert result.success is True
    assert len(calls) == 2
    assert result.retry_count == 1
    assert result.retry_errors[0].startswith("HTTP 503:")


def test_llm_client_does_not_retry_auth_error(monkeypatch) -> None:
    calls = []

    def fake_post(url, headers, json, timeout):
        calls.append(url)
        return FakeStatusResponse(401, "unauthorized")

    monkeypatch.setenv("DEEPSEEK_API_KEY", "deepseek-key")
    monkeypatch.setenv("DEEPSEEK_MODEL", "deepseek-chat")
    monkeypatch.setenv("ROUTEPILOT_LLM_RETRIES", "3")
    monkeypatch.setenv("ROUTEPILOT_LLM_RETRY_BACKOFF_SECONDS", "0")
    monkeypatch.setattr("routepilot.runtime.llm_client.requests.post", fake_post)

    result = create_llm_client(provider="deepseek").complete_with_usage("", "hello")

    assert result.success is False
    assert len(calls) == 1
    assert result.attempt_count == 1
    assert result.retry_count == 0
    assert result.retry_errors == []


def test_create_llm_client_infers_deepseek_provider_from_model(monkeypatch) -> None:
    monkeypatch.setenv("DEEPSEEK_API_KEY", "deepseek-key")

    client = create_llm_client(model="deepseek-reasoner")

    assert client.provider == "deepseek"
    assert client.base_url == "https://api.deepseek.com"
    assert client.model == "deepseek-reasoner"
    assert client.available() is True


def test_deepseek_v4_pro_enables_thinking_payload(monkeypatch) -> None:
    calls = []

    def fake_post(url, headers, json, timeout):
        calls.append({"url": url, "headers": headers, "json": json, "timeout": timeout})
        return FakeResponse()

    monkeypatch.setenv("DEEPSEEK_API_KEY", "deepseek-key")
    monkeypatch.setenv("DEEPSEEK_MODEL", "deepseek-v4-pro")
    monkeypatch.setenv("DEEPSEEK_THINKING", "enabled")
    monkeypatch.setenv("DEEPSEEK_REASONING_EFFORT", "high")
    monkeypatch.setattr("routepilot.runtime.llm_client.requests.post", fake_post)

    client = create_llm_client(provider="deepseek")
    result = client.complete_with_usage("", "hello")

    assert result.content == "ok"
    payload = calls[0]["json"]
    assert payload["model"] == "deepseek-v4-pro"
    assert payload["thinking"] == {"type": "enabled"}
    assert payload["reasoning_effort"] == "high"
    assert "temperature" not in payload


def test_deepseek_flash_sends_explicit_thinking_disabled(monkeypatch) -> None:
    """flash 默认开启思考，必须显式关闭，否则推理 token 会吃满 max_tokens 导致空响应。"""
    calls = []

    def fake_post(url, headers, json, timeout):
        calls.append({"json": json})
        return FakeResponse()

    monkeypatch.setenv("DEEPSEEK_API_KEY", "deepseek-key")
    monkeypatch.setenv("DEEPSEEK_MODEL", "deepseek-flash")
    monkeypatch.setenv("DEEPSEEK_THINKING", "disabled")
    monkeypatch.setattr("routepilot.runtime.llm_client.requests.post", fake_post)

    client = create_llm_client(provider="deepseek")
    client.complete_with_usage("", "hello")

    payload = calls[0]["json"]
    assert payload["thinking"] == {"type": "disabled"}
    assert "reasoning_effort" not in payload
    assert "temperature" in payload


def test_non_v4_deepseek_model_does_not_inject_thinking(monkeypatch) -> None:
    calls = []

    def fake_post(url, headers, json, timeout):
        calls.append({"json": json})
        return FakeResponse()

    monkeypatch.setenv("DEEPSEEK_API_KEY", "deepseek-key")
    monkeypatch.setenv("DEEPSEEK_MODEL", "deepseek-chat")
    monkeypatch.setenv("DEEPSEEK_THINKING", "disabled")
    monkeypatch.setattr("routepilot.runtime.llm_client.requests.post", fake_post)

    client = create_llm_client(provider="deepseek")
    client.complete_with_usage("", "hello")

    assert "thinking" not in calls[0]["json"]


def test_llm_client_streams_openai_compatible_chunks(monkeypatch) -> None:
    calls = []

    def fake_post(url, headers, json, timeout, stream=False):
        calls.append({"url": url, "headers": headers, "json": json, "timeout": timeout, "stream": stream})
        return FakeStreamResponse()

    monkeypatch.setenv("DEEPSEEK_API_KEY", "deepseek-key")
    monkeypatch.setenv("DEEPSEEK_MODEL", "deepseek-v4-pro")
    monkeypatch.setattr("routepilot.runtime.llm_client.requests.post", fake_post)

    client = create_llm_client(provider="deepseek")
    result = client.complete_with_usage("system", "hello", agent="Agent2", step="GenerateCodeEdits", timeout=(30, 900), stream=True)

    assert result.content == "hello"
    assert result.streaming_used is True
    assert result.timeout_seconds == (30, 900)
    assert calls[0]["stream"] is True
    assert calls[0]["json"]["stream"] is True


def test_llm_client_streams_utf8_bytes_without_mojibake(monkeypatch) -> None:
    def fake_post(url, headers, json, timeout, stream=False):
        return FakeUtf8StreamResponse()

    monkeypatch.setenv("DEEPSEEK_API_KEY", "deepseek-key")
    monkeypatch.setenv("DEEPSEEK_MODEL", "deepseek-v4-pro")
    monkeypatch.setattr("routepilot.runtime.llm_client.requests.post", fake_post)

    result = create_llm_client(provider="deepseek").complete_with_usage("system", "hello", stream=True)

    assert result.content == "预测报告"


def test_create_llm_client_defaults_non_gpt_openai_compatible_provider_to_chat(monkeypatch) -> None:
    calls = []

    def fake_post(url, headers, json, timeout):
        calls.append({"url": url, "headers": headers, "json": json, "timeout": timeout})
        return FakeResponse()

    monkeypatch.setenv("LLM_API_KEY", "generic-key")
    monkeypatch.setenv("LLM_BASE_URL", "https://llm.example.com/v1")
    monkeypatch.setenv("LLM_MODEL", "vendor-model")
    monkeypatch.setattr("routepilot.runtime.llm_client.requests.post", fake_post)

    client = create_llm_client(provider="openai-compatible")
    result = client.complete_with_usage("", "hello")

    assert client.provider == "openai-compatible"
    assert client.api_mode == "chat_completions"
    assert result.content == "ok"
    assert result.api_mode == "chat_completions"
    assert calls[0]["url"] == "https://llm.example.com/v1/chat/completions"
    assert calls[0]["json"]["model"] == "vendor-model"
    assert calls[0]["json"]["messages"] == [{"role": "user", "content": "hello"}]


def test_glm_model_uses_chat_completions_by_default(monkeypatch) -> None:
    calls = []

    def fake_post(url, headers, json, timeout):
        calls.append({"url": url, "headers": headers, "json": json, "timeout": timeout})
        return FakeResponse()

    monkeypatch.setenv("LLM_API_KEY", "generic-key")
    monkeypatch.setenv("LLM_BASE_URL", "https://llm.example.com/v1")
    monkeypatch.setenv("ROUTEPILOT_LLM_PROVIDER", "openai-compatible")
    monkeypatch.setattr("routepilot.runtime.llm_client.requests.post", fake_post)

    client = create_llm_client(model="glm-5.1")
    result = client.complete_with_usage("system", "hello")

    assert client.provider == "openai-compatible"
    assert client.api_mode == "chat_completions"
    assert result.content == "ok"
    assert calls[0]["url"] == "https://llm.example.com/v1/chat/completions"
    assert calls[0]["json"]["model"] == "glm-5.1"


def test_openai_compatible_provider_can_force_chat_completions(monkeypatch) -> None:
    calls = []

    def fake_post(url, headers, json, timeout):
        calls.append({"url": url, "headers": headers, "json": json, "timeout": timeout})
        return FakeResponse()

    monkeypatch.setenv("LLM_API_KEY", "generic-key")
    monkeypatch.setenv("LLM_BASE_URL", "https://llm.example.com/v1")
    monkeypatch.setenv("LLM_MODEL", "vendor-model")
    monkeypatch.setenv("ROUTEPILOT_LLM_API_MODE", "chat")
    monkeypatch.setattr("routepilot.runtime.llm_client.requests.post", fake_post)

    client = create_llm_client(provider="openai-compatible")
    result = client.complete_with_usage("system", "hello")

    assert client.api_mode == "chat_completions"
    assert result.content == "ok"
    assert calls[0]["url"] == "https://llm.example.com/v1/chat/completions"
    assert calls[0]["json"]["messages"] == [
        {"role": "system", "content": "system"},
        {"role": "user", "content": "hello"},
    ]


def test_codex_model_uses_responses_api_by_default(monkeypatch) -> None:
    calls = []

    def fake_post(url, headers, json, timeout):
        calls.append({"url": url, "headers": headers, "json": json, "timeout": timeout})
        return FakeResponse()

    monkeypatch.setenv("LLM_API_KEY", "generic-key")
    monkeypatch.setenv("LLM_BASE_URL", "https://llm.example.com/v1")
    monkeypatch.setattr("routepilot.runtime.llm_client.requests.post", fake_post)

    client = create_llm_client(model="gpt-5.3-codex")
    result = client.complete_with_usage("system", "hello", max_tokens=32)

    assert client.provider == "openai-compatible"
    assert client.api_mode == "responses"
    assert result.content == "ok"
    assert calls[0]["url"] == "https://llm.example.com/v1/responses"
    assert calls[0]["json"]["model"] == "gpt-5.3-codex"
    assert calls[0]["json"]["instructions"] == "system"
    assert calls[0]["json"]["input"] == "hello"
    assert calls[0]["json"]["max_output_tokens"] == 32


def test_responses_api_empty_content_is_failure_with_raw_response(monkeypatch) -> None:
    def fake_post(url, headers, json, timeout):
        return FakeEmptyResponsesResponse()

    monkeypatch.setenv("LLM_API_KEY", "generic-key")
    monkeypatch.setenv("LLM_BASE_URL", "https://llm.example.com/v1")
    monkeypatch.setattr("routepilot.runtime.llm_client.requests.post", fake_post)

    result = create_llm_client(model="gpt-5.3-codex").complete_with_usage("system", "hello")

    assert result.success is False
    assert result.content == ""
    assert result.error == "empty LLM response content"
    assert result.empty_content_reason == "responses output_text was empty"
    assert result.request_url == "https://llm.example.com/v1/responses"
    assert result.response_status_code == 200
    assert result.raw_response == {
        "output_text": "",
        "usage": {"input_tokens": 5, "output_tokens": 0, "total_tokens": 5},
    }


def test_create_llm_client_uses_temperature_env(monkeypatch) -> None:
    calls = []

    def fake_post(url, headers, json, timeout):
        calls.append({"url": url, "headers": headers, "json": json, "timeout": timeout})
        return FakeResponse()

    monkeypatch.setenv("LLM_API_KEY", "generic-key")
    monkeypatch.setenv("LLM_BASE_URL", "https://llm.example.com/v1")
    monkeypatch.setenv("LLM_MODEL", "vendor-model")
    monkeypatch.setenv("ROUTEPILOT_LLM_TEMPERATURE", "1")
    monkeypatch.setattr("routepilot.runtime.llm_client.requests.post", fake_post)

    result = create_llm_client(provider="openai-compatible").complete_with_usage("", "hello")

    assert result.content == "ok"
    assert calls[0]["json"]["temperature"] == 1.0


def test_responses_api_streams_chunks(monkeypatch) -> None:
    calls = []

    def fake_post(url, headers, json, timeout, stream=False):
        calls.append({"url": url, "headers": headers, "json": json, "timeout": timeout, "stream": stream})
        return FakeResponsesStreamResponse()

    monkeypatch.setenv("LLM_API_KEY", "generic-key")
    monkeypatch.setenv("LLM_BASE_URL", "https://llm.example.com/v1")
    monkeypatch.setattr("routepilot.runtime.llm_client.requests.post", fake_post)

    result = create_llm_client(model="gpt-5.3-codex").complete_with_usage("system", "hello", stream=True)

    assert result.content == "hello"
    assert result.streaming_used is True
    assert result.usage["total_tokens"] == 7
    assert calls[0]["url"] == "https://llm.example.com/v1/responses"
    assert calls[0]["stream"] is True
    assert calls[0]["json"]["stream"] is True


def test_responses_api_stream_uses_done_text_as_fallback(monkeypatch) -> None:
    def fake_post(url, headers, json, timeout, stream=False):
        return FakeResponsesDoneOnlyStreamResponse()

    monkeypatch.setenv("LLM_API_KEY", "generic-key")
    monkeypatch.setenv("LLM_BASE_URL", "https://llm.example.com/v1")
    monkeypatch.setattr("routepilot.runtime.llm_client.requests.post", fake_post)

    result = create_llm_client(model="gpt-5.3-codex").complete_with_usage("system", "hello", stream=True)

    assert result.content == "hello"
    assert result.usage["total_tokens"] == 7


def test_create_llm_client_keeps_doubao_backward_compatibility(monkeypatch) -> None:
    monkeypatch.setenv("DOUBAO_API_KEY", "doubao-key")
    monkeypatch.setenv("DOUBAO_BASE_URL", "https://ark.cn-beijing.volces.com/api/v3")
    monkeypatch.setenv("DOUBAO_MODEL", "ark-endpoint")

    client = create_llm_client(model="doubao")

    assert client.provider == "doubao"
    assert client.model == "ark-endpoint"
    assert client.available() is True


def test_llm_client_loads_dotenv_without_overriding_existing_env(monkeypatch, tmp_path) -> None:
    env_path = tmp_path / ".env"
    env_path.write_text(
        "\n".join(
            [
                "DEEPSEEK_API_KEY=from-dotenv",
                "DEEPSEEK_MODEL=deepseek-v4-pro",
                "DEEPSEEK_REASONING_EFFORT=max",
                "DOUBAO_MODEL=from-dotenv-doubao",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    monkeypatch.delenv("DEEPSEEK_MODEL", raising=False)
    monkeypatch.delenv("DEEPSEEK_REASONING_EFFORT", raising=False)
    monkeypatch.setenv("DOUBAO_MODEL", "from-shell")

    import routepilot.runtime.llm_client as llm_client

    importlib.reload(llm_client)

    client = llm_client.create_llm_client(provider="deepseek")
    assert client.available() is True
    assert client.model == "deepseek-v4-pro"
    assert llm_client._env_value("deepseek", "REASONING_EFFORT") == "max"
    assert llm_client._env_value("doubao", "MODEL") == "from-shell"
