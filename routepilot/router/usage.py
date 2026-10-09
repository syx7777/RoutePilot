"""LLM 调用台账：token / 成本 / 延迟，按档位与 step 聚合。"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class CallRecord:
    agent: str
    step: str
    tier: str
    model: str
    prompt_tokens: int
    completion_tokens: int
    cost_usd: float
    duration_sec: float
    success: bool
    attempt_count: int = 1


def percentile(values: list[float], ratio: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = ratio * (len(ordered) - 1)
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


@dataclass
class UsageLedger:
    records: list[CallRecord] = field(default_factory=list)

    def record(self, result: Any, tier: Any) -> CallRecord:
        usage = dict(getattr(result, "usage", None) or {})
        prompt_tokens = int(usage.get("prompt_tokens") or 0)
        completion_tokens = int(usage.get("completion_tokens") or 0)
        entry = CallRecord(
            agent=str(getattr(result, "agent", "") or ""),
            step=str(getattr(result, "step", "") or ""),
            tier=tier.name,
            model=str(getattr(result, "model", "") or tier.model),
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            cost_usd=tier.cost_usd(prompt_tokens, completion_tokens),
            duration_sec=float(getattr(result, "duration_seconds", 0.0) or 0.0),
            success=bool(getattr(result, "success", False)),
            attempt_count=int(getattr(result, "attempt_count", 1) or 1),
        )
        self.records.append(entry)
        return entry

    @property
    def total_cost_usd(self) -> float:
        return sum(item.cost_usd for item in self.records)

    @property
    def total_tokens(self) -> int:
        return sum(item.prompt_tokens + item.completion_tokens for item in self.records)

    def summary(self) -> dict[str, Any]:
        durations = [item.duration_sec for item in self.records]
        return {
            "calls": len(self.records),
            "cost_usd": round(self.total_cost_usd, 6),
            "tokens": self.total_tokens,
            "latency_p50_sec": round(percentile(durations, 0.5), 3),
            "latency_p95_sec": round(percentile(durations, 0.95), 3),
            "success_rate": (
                round(sum(1 for item in self.records if item.success) / len(self.records), 4)
                if self.records
                else 0.0
            ),
            "by_tier": self._group(lambda item: item.tier),
            "by_step": self._group(lambda item: item.step),
        }

    def _group(self, key) -> dict[str, Any]:
        buckets: dict[str, list[CallRecord]] = {}
        for item in self.records:
            buckets.setdefault(key(item), []).append(item)
        grouped: dict[str, Any] = {}
        for name, items in sorted(buckets.items()):
            grouped[name] = {
                "calls": len(items),
                "cost_usd": round(sum(item.cost_usd for item in items), 6),
                "tokens": sum(item.prompt_tokens + item.completion_tokens for item in items),
                "latency_p95_sec": round(
                    percentile([item.duration_sec for item in items], 0.95), 3
                ),
                "success_rate": round(
                    sum(1 for item in items if item.success) / len(items), 4
                ),
            }
        return grouped

    def to_list(self) -> list[dict[str, Any]]:
        return [asdict(item) for item in self.records]
