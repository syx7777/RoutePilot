from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import requests

_DOTENV_LOADED = False


@dataclass
class LLMCallResult:
    content: str
    model: str | None
    agent: str
    step: str
    duration_seconds: float
    usage: dict[str, int] = field(default_factory=dict)
    estimated: bool = False
    available: bool = True
    success: bool = True
    summary: str = ""
    error: str | None = None
    timeout_seconds: int | float | tuple[int | float, int | float] | None = None
    streaming_used: bool = False
    attempt_count: int = 0
    retry_count: int = 0
    retry_errors: list[str] = field(default_factory=list)
    api_mode: str = ""
    request_url: str = ""
    response_status_code: int | None = None
    response_url: str = ""
    raw_response: dict[str, Any] | None = None
    empty_content_reason: str = ""


class LLMClient:
    def __init__(
        self,
        *,
        provider: str = "doubao",
        model: str | None = None,
        api_key: str | None = None,
        base_url: str | None = None,
        temperature: float = 1.0,
        api_mode: str | None = None,
    ):
        self.provider = _normalize_provider(provider)
        self.api_key = api_key or _env_value(self.provider, "API_KEY")
        self.base_url = (base_url or _env_value(self.provider, "BASE_URL") or _default_base_url(self.provider)).rstrip("/")
        self.model = _resolve_model(self.provider, model)
        self.temperature = temperature
        self.api_mode = _resolve_api_mode(self.provider, self.model, api_mode)
        self.last_call: LLMCallResult | None = None
        self.call_results: list[LLMCallResult] = []

    def available(self) -> bool:
        return bool(self.api_key and self.base_url and self.model)

    def complete(self, system_prompt: str, user_prompt: str) -> str:
        return self.complete_with_usage(system_prompt, user_prompt).content

    def complete_with_usage(
        self,
        system_prompt: str,
        user_prompt: str,
        *,
        agent: str = "unknown",
        step: str = "complete",
        timeout: int | float | tuple[int | float, int | float] | None = None,
        stream: bool = False,
        max_tokens: int | None = None,
    ) -> LLMCallResult:
        started = time.perf_counter()
        request_timeout = timeout if timeout is not None else 120
        if not self.available():
            result = LLMCallResult(
                content="",
                model=self.model,
                agent=agent,
                step=step,
                duration_seconds=time.perf_counter() - started,
                usage={"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
                estimated=False,
                available=False,
                success=False,
                summary=f"LLM unavailable for provider={self.provider}; RoutePilot will fail fast for this LLM-led step.",
                timeout_seconds=request_timeout,
                streaming_used=False,
                attempt_count=0,
                retry_count=0,
                retry_errors=[],
                api_mode=self.api_mode,
            )
            self.last_call = result
            self.call_results.append(result)
            return result
        payload = _build_payload(
            self.provider,
            self.model,
            self.api_mode,
            system_prompt,
            user_prompt,
            temperature=self.temperature,
            stream=stream,
            max_tokens=max_tokens,
        )
        retry_limit = _llm_retry_limit()
        retry_errors: list[str] = []
        attempt_count = 0
        request_url = f"{self.base_url}/{_api_path(self.api_mode)}"
        try:
            response = None
            data: dict[str, Any] = {}
            content = ""
            for attempt_index in range(1, retry_limit + 2):
                attempt_count = attempt_index
                try:
                    request_kwargs: dict[str, Any] = {
                        "headers": {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"},
                        "json": payload,
                        "timeout": request_timeout,
                    }
                    if stream:
                        request_kwargs["stream"] = True
                    response = requests.post(request_url, **request_kwargs)
                    response.raise_for_status()
                    if stream and self.api_mode == "responses":
                        content, data = _content_from_responses_stream_response(response)
                    elif stream:
                        content, data = _content_from_stream_response(response)
                    elif self.api_mode == "responses":
                        data = response.json()
                        content = _content_from_responses_response(data)
                    else:
                        data = response.json()
                        content = data["choices"][0]["message"]["content"]
                    break
                except Exception as exc:  # noqa: BLE001 - retry policy decides whether to try again.
                    if attempt_index > retry_limit or not _should_retry_llm_exception(exc):
                        raise
                    retry_errors.append(_exception_summary(exc))
                    time.sleep(_llm_retry_backoff_seconds(attempt_index))
            else:  # pragma: no cover - loop always breaks or raises.
                raise RuntimeError("LLM request retry loop exited without a response")
            usage, estimated = _usage_from_response(data, system_prompt, user_prompt, content)
            response_status_code = _response_status_code(response)
            response_url = _response_url(response)
            raw_response = _bounded_response_data(data)
            if not content:
                result = LLMCallResult(
                    content="",
                    model=self.model,
                    agent=agent,
                    step=step,
                    duration_seconds=time.perf_counter() - started,
                    usage=usage,
                    estimated=estimated,
                    available=True,
                    success=False,
                    summary=_summarize_call(system_prompt, user_prompt, ""),
                    error="empty LLM response content",
                    timeout_seconds=request_timeout,
                    streaming_used=stream,
                    attempt_count=attempt_count,
                    retry_count=max(0, attempt_count - 1),
                    retry_errors=retry_errors,
                    api_mode=self.api_mode,
                    request_url=request_url,
                    response_status_code=response_status_code,
                    response_url=response_url,
                    raw_response=raw_response,
                    empty_content_reason=_empty_content_reason(data, self.api_mode),
                )
                self.last_call = result
                self.call_results.append(result)
                return result
            result = LLMCallResult(
                content=content,
                model=self.model,
                agent=agent,
                step=step,
                duration_seconds=time.perf_counter() - started,
                usage=usage,
                estimated=estimated,
                available=True,
                success=True,
                summary=_summarize_call(system_prompt, user_prompt, content),
                timeout_seconds=request_timeout,
                streaming_used=stream,
                attempt_count=attempt_count,
                retry_count=max(0, attempt_count - 1),
                retry_errors=retry_errors,
                api_mode=self.api_mode,
                request_url=request_url,
                response_status_code=response_status_code,
                response_url=response_url,
                raw_response=raw_response,
            )
        except Exception as exc:  # noqa: BLE001 - callers decide whether to fail fast after recording evidence.
            if attempt_count == 0:
                attempt_count = 1
            result = LLMCallResult(
                content="",
                model=self.model,
                agent=agent,
                step=step,
                duration_seconds=time.perf_counter() - started,
                usage={"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
                estimated=False,
                available=True,
                success=False,
                summary=_summarize_call(system_prompt, user_prompt, ""),
                error=str(exc),
                timeout_seconds=request_timeout,
                streaming_used=stream,
                attempt_count=attempt_count,
                retry_count=max(0, attempt_count - 1),
                retry_errors=retry_errors,
                api_mode=self.api_mode,
                request_url=request_url,
            )
        self.last_call = result
        self.call_results.append(result)
        return result

    def skylark(self, prompt: str) -> str:
        return self.complete("", prompt)


class DoubaoClient(LLMClient):
    def __init__(self, model: str | None = None):
        super().__init__(provider="doubao", model=model)


def create_llm_client(provider: str | None = None, model: str | None = None) -> LLMClient:
    _load_dotenv_once()
    resolved_provider = provider or _provider_from_model(model) or os.environ.get("ROUTEPILOT_LLM_PROVIDER") or "doubao"
    return LLMClient(provider=resolved_provider, model=None if model == resolved_provider else model)


def _normalize_provider(provider: str) -> str:
    value = provider.strip().lower().replace("_", "-")
    if value in {"openai", "generic", "llm"}:
        return "openai-compatible"
    return value


def _provider_from_model(model: str | None) -> str | None:
    if not model:
        return None
    normalized = model.strip().lower()
    if normalized in {"doubao", "deepseek", "openai-compatible"}:
        return normalized
    if normalized.startswith("deepseek-"):
        return "deepseek"
    if normalized.startswith("gpt-"):
        return "openai-compatible"
    return None


def _resolve_model(provider: str, model: str | None) -> str | None:
    if model and model != provider:
        return model
    return _env_value(provider, "MODEL") or _default_model(provider)


def _resolve_api_mode(provider: str, model: str | None, api_mode: str | None = None) -> str:
    configured = (
        _normalize_api_mode(api_mode)
        or _normalize_api_mode(os.environ.get("ROUTEPILOT_LLM_API_MODE"))
        or _normalize_api_mode(_env_value(provider, "API_MODE"))
        or _normalize_api_mode(os.environ.get("ROUTEPILOT_LLM_API"))
        or _normalize_api_mode(_env_value(provider, "API"))
    )
    if configured:
        return configured
    if _prefers_responses_api(provider, model):
        return "responses"
    return "chat_completions"


def _normalize_api_mode(value: str | None) -> str | None:
    if not value:
        return None
    normalized = value.strip().lower().replace("-", "_").replace(" ", "_")
    if normalized in {"response", "responses", "responses_api"}:
        return "responses"
    if normalized in {"chat", "chat_completion", "chat_completions", "chat_completions_api"}:
        return "chat_completions"
    return None


def _prefers_responses_api(provider: str, model: str | None) -> bool:
    normalized_model = (model or "").strip().lower()
    return normalized_model.startswith("gpt-") or "codex" in normalized_model


def _api_path(api_mode: str) -> str:
    if api_mode == "responses":
        return "responses"
    return "chat/completions"


def _build_payload(
    provider: str,
    model: str | None,
    api_mode: str,
    system_prompt: str,
    user_prompt: str,
    *,
    temperature: float,
    stream: bool,
    max_tokens: int | None,
) -> dict[str, Any]:
    if api_mode == "responses":
        payload: dict[str, Any] = {"model": model, "input": user_prompt}
        if system_prompt:
            payload["instructions"] = system_prompt
        if max_tokens is not None:
            payload["max_output_tokens"] = max_tokens
        if stream:
            payload["stream"] = True
        if _should_send_temperature(provider, model):
            payload["temperature"] = temperature
        return payload

    messages: list[dict[str, str]] = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": user_prompt})
    payload = {"model": model, "messages": messages}
    if max_tokens is not None:
        payload["max_tokens"] = max_tokens
    if stream:
        payload["stream"] = True
    thinking = _deepseek_thinking_setting(provider, model)
    if thinking == "enabled":
        payload["thinking"] = {"type": "enabled"}
        payload["reasoning_effort"] = _env_value(provider, "REASONING_EFFORT") or "high"
    elif thinking == "disabled":
        # 必须显式关闭：DeepSeek 默认开启思考，推理 token 会吃掉 max_tokens，
        # 表现为 completion_tokens 打满而 content 为空。
        payload["thinking"] = {"type": "disabled"}
        payload["temperature"] = temperature
    elif _should_send_temperature(provider, model):
        payload["temperature"] = temperature
    return payload


def _should_send_temperature(provider: str, model: str | None) -> bool:
    return _deepseek_thinking_setting(provider, model) != "enabled"


def _deepseek_thinking_setting(provider: str, model: str | None) -> str | None:
    """返回 DeepSeek V4 系列的思考模式开关；其它模型返回 None（不干预）。"""
    if provider != "deepseek" or not model:
        return None
    normalized = model.strip().lower()
    if not (normalized.startswith("deepseek-v4") or normalized.startswith("deepseek-flash")):
        return None
    setting = (_env_value(provider, "THINKING") or "enabled").strip().lower()
    return "disabled" if setting == "disabled" else "enabled"


def _env_value(provider: str, suffix: str) -> str | None:
    prefix = _env_prefix(provider)
    value = os.environ.get(f"{prefix}_{suffix}")
    if value:
        return value
    if provider == "openai-compatible":
        return os.environ.get(f"LLM_{suffix}")
    return None


def _env_prefix(provider: str) -> str:
    if provider == "openai-compatible":
        return "LLM"
    return provider.upper().replace("-", "_")


def _default_base_url(provider: str) -> str:
    if provider == "deepseek":
        return "https://api.deepseek.com"
    return ""


def _default_model(provider: str) -> str | None:
    if provider == "deepseek":
        return "deepseek-v4-pro"
    return None


def _deepseek_thinking_enabled(provider: str, model: str | None) -> bool:
    return _deepseek_thinking_setting(provider, model) == "enabled"


def _llm_retry_limit() -> int:
    return max(0, _env_int("ROUTEPILOT_LLM_RETRIES", 3))


def _llm_retry_backoff_seconds(attempt_index: int) -> float:
    base = max(0.0, _env_float("ROUTEPILOT_LLM_RETRY_BACKOFF_SECONDS", 5.0))
    max_backoff = max(0.0, _env_float("ROUTEPILOT_LLM_RETRY_MAX_BACKOFF_SECONDS", 60.0))
    delay = base * (2 ** max(0, attempt_index - 1))
    return min(delay, max_backoff)


def _env_int(key: str, default: int) -> int:
    value = os.environ.get(key)
    if value is None or value.strip() == "":
        return default
    try:
        return int(value)
    except ValueError:
        return default


def _env_float(key: str, default: float) -> float:
    value = os.environ.get(key)
    if value is None or value.strip() == "":
        return default
    try:
        return float(value)
    except ValueError:
        return default


def _should_retry_llm_exception(exc: Exception) -> bool:
    if isinstance(
        exc,
        (
            requests.exceptions.ProxyError,
            requests.exceptions.ConnectionError,
            requests.exceptions.Timeout,
        ),
    ):
        return True
    if isinstance(exc, requests.exceptions.HTTPError):
        response = getattr(exc, "response", None)
        status_code = getattr(response, "status_code", None)
        return status_code in {429, 500, 502, 503, 504}
    return False


def _exception_summary(exc: Exception) -> str:
    response = getattr(exc, "response", None)
    status_code = getattr(response, "status_code", None)
    prefix = f"HTTP {status_code}: " if status_code else ""
    return prefix + str(exc)


def _usage_from_response(data: dict[str, Any], system_prompt: str, user_prompt: str, content: str) -> tuple[dict[str, int], bool]:
    usage = data.get("usage")
    if isinstance(usage, dict):
        prompt_tokens = int(usage.get("prompt_tokens") or usage.get("input_tokens") or 0)
        completion_tokens = int(usage.get("completion_tokens") or usage.get("output_tokens") or 0)
        total_tokens = int(usage.get("total_tokens") or prompt_tokens + completion_tokens)
        return {"prompt_tokens": prompt_tokens, "completion_tokens": completion_tokens, "total_tokens": total_tokens}, False
    prompt_tokens = _estimate_tokens(system_prompt + "\n" + user_prompt)
    completion_tokens = _estimate_tokens(content)
    return {"prompt_tokens": prompt_tokens, "completion_tokens": completion_tokens, "total_tokens": prompt_tokens + completion_tokens}, True


def _response_status_code(response: Any) -> int | None:
    value = getattr(response, "status_code", None)
    return int(value) if isinstance(value, int) else None


def _response_url(response: Any) -> str:
    value = getattr(response, "url", "")
    return value if isinstance(value, str) else ""


def _bounded_response_data(data: dict[str, Any]) -> dict[str, Any]:
    return _bounded_value(data, max_string=4000, max_items=40, depth=0)


def _bounded_value(value: Any, *, max_string: int, max_items: int, depth: int) -> Any:
    if depth >= 6:
        return "<truncated: depth>"
    if isinstance(value, dict):
        items = list(value.items())
        bounded = {
            str(key): _bounded_value(item, max_string=max_string, max_items=max_items, depth=depth + 1)
            for key, item in items[:max_items]
        }
        if len(items) > max_items:
            bounded["<truncated_items>"] = len(items) - max_items
        return bounded
    if isinstance(value, list):
        bounded_list = [_bounded_value(item, max_string=max_string, max_items=max_items, depth=depth + 1) for item in value[:max_items]]
        if len(value) > max_items:
            bounded_list.append({"<truncated_items>": len(value) - max_items})
        return bounded_list
    if isinstance(value, str):
        if len(value) <= max_string:
            return value
        return value[:max_string] + f"\n<truncated {len(value) - max_string} chars>"
    return value


def _empty_content_reason(data: dict[str, Any], api_mode: str) -> str:
    if not data:
        return f"{api_mode} response JSON was empty or unavailable"
    if api_mode == "responses":
        if isinstance(data.get("error"), dict) or isinstance(data.get("error"), str):
            return "responses response contained error but no output text"
        if data.get("output_text") == "":
            return "responses output_text was empty"
        if data.get("output") == []:
            return "responses output list was empty"
        return "responses response had no extractable output text"
    return "chat completion response had no message content"


def _content_from_responses_response(data: dict[str, Any]) -> str:
    output_text = data.get("output_text")
    if isinstance(output_text, str):
        return output_text
    choices = data.get("choices")
    if isinstance(choices, list) and choices:
        message = choices[0].get("message") if isinstance(choices[0], dict) else None
        if isinstance(message, dict) and isinstance(message.get("content"), str):
            return message["content"]
    chunks: list[str] = []
    for item in data.get("output") or []:
        if not isinstance(item, dict):
            continue
        text = item.get("text")
        if isinstance(text, str):
            chunks.append(text)
        content = item.get("content")
        if isinstance(content, str):
            chunks.append(content)
            continue
        if not isinstance(content, list):
            continue
        for content_item in content:
            if not isinstance(content_item, dict):
                continue
            text = content_item.get("text")
            if isinstance(text, str):
                chunks.append(text)
            elif isinstance(content_item.get("content"), str):
                chunks.append(str(content_item["content"]))
    return "".join(chunks)


def _content_from_responses_stream_response(response: Any) -> tuple[str, dict[str, Any]]:
    chunks: list[str] = []
    final_data: dict[str, Any] = {}
    completed_text: str | None = None
    for raw_line in response.iter_lines(decode_unicode=False):
        if not raw_line:
            continue
        line = _decode_stream_line(raw_line).strip()
        if line.startswith("data:"):
            line = line[len("data:") :].strip()
        if line == "[DONE]":
            break
        try:
            data = json_loads(line)
        except Exception:
            continue
        event_type = str(data.get("type") or "")
        if event_type == "response.completed" and isinstance(data.get("response"), dict):
            final_data = data["response"]
            continue
        if isinstance(data.get("response"), dict):
            final_data = data["response"]
        if event_type == "response.output_text.done" and isinstance(data.get("text"), str):
            completed_text = data["text"]
            continue
        delta = data.get("delta")
        if event_type == "response.output_text.delta" and isinstance(delta, str):
            chunks.append(delta)
            continue
        if not event_type and isinstance(delta, str):
            chunks.append(delta)
            continue
        if not event_type and isinstance(data.get("text"), str) and not chunks:
            completed_text = str(data["text"])
    content = "".join(chunks)
    if not final_data:
        final_data = {"output_text": content or completed_text or ""}
    elif not content:
        content = completed_text or _content_from_responses_response(final_data)
    return content, final_data


def _content_from_stream_response(response: Any) -> tuple[str, dict[str, Any]]:
    chunks: list[str] = []
    final_data: dict[str, Any] = {}
    for raw_line in response.iter_lines(decode_unicode=False):
        if not raw_line:
            continue
        line = _decode_stream_line(raw_line).strip()
        if line.startswith("data:"):
            line = line[len("data:") :].strip()
        if line == "[DONE]":
            break
        try:
            data = json_loads(line)
        except Exception:
            continue
        final_data = data
        choices = data.get("choices") or []
        if not choices:
            continue
        delta = choices[0].get("delta") or {}
        message = choices[0].get("message") or {}
        text = delta.get("content") if isinstance(delta, dict) else None
        if text is None and isinstance(message, dict):
            text = message.get("content")
        if text:
            chunks.append(str(text))
    content = "".join(chunks)
    if not final_data:
        final_data = {"choices": [{"message": {"content": content}}]}
    return content, final_data


def _decode_stream_line(raw_line: Any) -> str:
    if isinstance(raw_line, bytes):
        return raw_line.decode("utf-8", errors="replace")
    return str(raw_line)


def json_loads(text: str) -> dict[str, Any]:
    import json

    value = json.loads(text)
    return value if isinstance(value, dict) else {}


def _load_dotenv_once() -> None:
    global _DOTENV_LOADED
    if _DOTENV_LOADED:
        return
    _DOTENV_LOADED = True
    env_path = Path.cwd() / ".env"
    if not env_path.exists():
        return
    for raw_line in env_path.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip("\"'")
        if key and key not in os.environ:
            os.environ[key] = value


def _estimate_tokens(text: str) -> int:
    if not text:
        return 0
    return max(1, (len(text) + 3) // 4)


def _summarize_call(system_prompt: str, user_prompt: str, content: str) -> str:
    return (
        f"system_chars={len(system_prompt)}, "
        f"user_chars={len(user_prompt)}, "
        f"output_chars={len(content)}"
    )
