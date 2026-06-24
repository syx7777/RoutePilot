from __future__ import annotations

import unicodedata
from typing import Any

import yaml


YAML_OUTPUT_CONTRACT_MARKER = "ComboScope YAML output contract"

YAML_OUTPUT_CONTRACT_RULES = [
    "Return exactly one YAML mapping as the whole response.",
    "Do not wrap the response in Markdown fences and do not add explanations before or after YAML.",
    "Do not return a JSON envelope; use plain YAML keys at the top level.",
    "Quote strings that contain ':', ';', '#', braces, brackets, leading dashes, or Chinese punctuation.",
    "Use block scalars with | for any multiline text, Markdown, code, or prose paragraphs.",
    "Write lists as YAML lists with one item per line.",
    "Do not emit invisible control characters; only tab, newline, and carriage return are allowed as control whitespace.",
]


def append_yaml_output_contract(system_prompt: str, prompt: str) -> tuple[str, str]:
    contract = yaml_output_contract()
    if YAML_OUTPUT_CONTRACT_MARKER in system_prompt or YAML_OUTPUT_CONTRACT_MARKER in prompt:
        return system_prompt, prompt
    system = (system_prompt or "").rstrip()
    user = (prompt or "").rstrip()
    return f"{system}\n\n{contract}".strip(), f"{user}\n\n{contract}".strip()


def yaml_output_contract() -> str:
    lines = [f"{YAML_OUTPUT_CONTRACT_MARKER}:"]
    lines.extend(f"- {rule}" for rule in YAML_OUTPUT_CONTRACT_RULES)
    return "\n".join(lines)


def strip_code_fence(text: str) -> str:
    stripped = str(text or "").strip()
    if not stripped.startswith("```"):
        return stripped
    lines = stripped.splitlines()
    if lines and lines[0].startswith("```"):
        lines = lines[1:]
    if lines and lines[-1].strip().startswith("```"):
        lines = lines[:-1]
    return "\n".join(lines).strip()


def sanitize_yaml_text(text: str) -> str:
    return "".join(ch for ch in str(text or "") if _is_yaml_printable(ch))


def clean_llm_yaml_text(text: str) -> str:
    return sanitize_yaml_text(strip_code_fence(text))


def yaml_control_char_summary(text: str, *, limit: int = 10) -> dict[str, Any]:
    invalid: list[dict[str, Any]] = []
    total = 0
    for index, ch in enumerate(str(text or "")):
        if _is_yaml_printable(ch):
            continue
        total += 1
        if len(invalid) < limit:
            invalid.append(
                {
                    "position": index,
                    "codepoint": f"U+{ord(ch):04X}",
                    "category": unicodedata.category(ch),
                }
            )
    return {"count": total, "examples": invalid}


def safe_load_yaml_mapping(raw: str, *, agent: str, step: str) -> dict[str, Any]:
    text = clean_llm_yaml_text(raw)
    try:
        parsed = yaml.safe_load(text) or {}
    except Exception as exc:  # noqa: BLE001 - normalize parser failures for LLM contract handling.
        summary = yaml_control_char_summary(raw)
        detail = f"; removed_control_chars={summary['count']}" if summary["count"] else ""
        raise ValueError(f"{agent} {step} did not return valid YAML{detail}") from exc
    if not isinstance(parsed, dict):
        raise ValueError(f"{agent} {step} must return a YAML mapping")
    return parsed


def _is_yaml_printable(ch: str) -> bool:
    if ch in "\t\n\r":
        return True
    category = unicodedata.category(ch)
    return category[0] != "C"
