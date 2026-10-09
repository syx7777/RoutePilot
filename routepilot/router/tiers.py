"""模型档位与价目表。

价格取 DeepSeek 官方**空闲时段**刊例价（USD / 百万 tokens），
可用 `ROUTEPILOT_PRICE_<TIER>_IN/OUT` 覆盖。价格会变，实验前请核对官方价目表。
"""

from __future__ import annotations

import os
from dataclasses import dataclass

WEAK = "weak"
STRONG = "strong"


@dataclass(frozen=True)
class ModelTier:
    name: str
    provider: str
    model: str
    price_input_per_mtok: float
    price_output_per_mtok: float

    def cost_usd(self, prompt_tokens: int, completion_tokens: int) -> float:
        return (
            prompt_tokens * self.price_input_per_mtok
            + completion_tokens * self.price_output_per_mtok
        ) / 1_000_000


def _override(tier_name: str, suffix: str, default: float) -> float:
    raw = os.environ.get(f"ROUTEPILOT_PRICE_{tier_name.upper()}_{suffix}")
    if raw is None or not raw.strip():
        return default
    try:
        return float(raw)
    except ValueError:
        return default


def default_tiers() -> tuple[ModelTier, ...]:
    """两档：deepseek-flash（弱）与 deepseek-v4-pro（强），约 4 倍价差。"""
    return (
        ModelTier(
            WEAK,
            "deepseek",
            "deepseek-flash",
            _override(WEAK, "IN", 0.15),
            _override(WEAK, "OUT", 0.60),
        ),
        ModelTier(
            STRONG,
            "deepseek",
            "deepseek-v4-pro",
            _override(STRONG, "IN", 0.66),
            _override(STRONG, "OUT", 1.98),
        ),
    )


def tier_by_name(tiers: tuple[ModelTier, ...], name: str) -> ModelTier:
    for tier in tiers:
        if tier.name == name:
            return tier
    raise KeyError(f"unknown tier: {name}")
