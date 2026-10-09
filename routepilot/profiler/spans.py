"""链路埋点：按 category 归集 span 耗时，用于瓶颈归因。"""

from __future__ import annotations

import time
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from typing import Any, Iterator

PROJECT_RUN = "project_run"
LLM_INFERENCE = "llm_inference"
EVALUATE = "evaluate"

CATEGORIES = (PROJECT_RUN, LLM_INFERENCE, EVALUATE)


@dataclass
class Span:
    name: str
    category: str
    duration_sec: float
    meta: dict[str, Any] = field(default_factory=dict)


@dataclass
class Profiler:
    spans: list[Span] = field(default_factory=list)

    @contextmanager
    def span(self, name: str, category: str, **meta: Any) -> Iterator[Span]:
        started = time.perf_counter()
        record = Span(name=name, category=category, duration_sec=0.0, meta=dict(meta))
        try:
            yield record
        finally:
            record.duration_sec = time.perf_counter() - started
            self.spans.append(record)

    def attribution(self) -> dict[str, float]:
        """各 category 的累计耗时（秒），按耗时降序。"""
        totals: dict[str, float] = {}
        for item in self.spans:
            totals[item.category] = totals.get(item.category, 0.0) + item.duration_sec
        return dict(sorted(totals.items(), key=lambda pair: pair[1], reverse=True))

    def by_name(self) -> dict[str, float]:
        totals: dict[str, float] = {}
        for item in self.spans:
            totals[item.name] = totals.get(item.name, 0.0) + item.duration_sec
        return dict(sorted(totals.items(), key=lambda pair: pair[1], reverse=True))

    def wall_seconds(self) -> float:
        return sum(item.duration_sec for item in self.spans)

    def to_list(self) -> list[dict[str, Any]]:
        return [asdict(item) for item in self.spans]
