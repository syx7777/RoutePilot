"""step 级动态路由。

决策目标（设计文档 §4.1 的按 step 分解）：
    score(tier) = 收益(难度, 历史成功率) − λ·成本 − μ·延迟
其中 λ 由预算对偶上升在线抬高，历史成功率来自 Thompson 采样的 Beta 后验。
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable

from routepilot.router.tiers import STRONG, WEAK, ModelTier, default_tiers, tier_by_name
from routepilot.router.usage import UsageLedger, percentile


class RouterMode(str, Enum):
    STRONG = "strong"    # Baseline A：全强模型
    WEAK = "weak"        # Baseline B：全弱模型
    STATIC = "static"    # Baseline C：固定 step→档位 映射
    DYNAMIC = "dynamic"  # Ours：难度先验 + Thompson 后验 + 预算对偶


# 固定路由的基线映射：可判定的步骤给弱模型，生成类步骤给强模型
STATIC_STEP_TIERS = {
    "Diagnose": WEAK,
    "ProposeExperiment": STRONG,
    "RepairProposal": STRONG,
}

# 各 step 的固有难度先验（0=极简，1=极难）
STEP_BASE_DIFFICULTY = {
    "Diagnose": 0.25,
    "ProposeExperiment": 0.75,
    "RepairProposal": 0.85,
}
DEFAULT_DIFFICULTY = 0.5

# 未见过的档位延迟先验（秒），仅用于首次决策；之后被实测 p50 取代
LATENCY_PRIOR = {WEAK: 1.2, STRONG: 3.0}


@dataclass
class StepFeatures:
    step: str
    prompt_tokens: int
    editable_files: int = 0
    prior_failures: int = 0


def estimate_difficulty(features: StepFeatures) -> float:
    """难度先验：step 类型 + 上下文规模 + 可改面大小 + 同 step 历史失败次数。"""
    base = STEP_BASE_DIFFICULTY.get(features.step, DEFAULT_DIFFICULTY)
    context_term = min(features.prompt_tokens / 6000.0, 1.0) * 0.15
    editable_term = min(features.editable_files / 5.0, 1.0) * 0.10
    failure_term = min(features.prior_failures / 3.0, 1.0) * 0.20
    return max(0.0, min(1.0, base + context_term + editable_term + failure_term))


def difficulty_bucket(value: float) -> str:
    if value < 0.4:
        return "easy"
    if value < 0.7:
        return "medium"
    return "hard"


class BetaPosterior:
    """按 (step, 难度桶, 档位) 维护成功率 Beta 后验，用 Thompson 采样引入探索。"""

    def __init__(self, rng: random.Random | None = None):
        self._stats: dict[tuple[str, str, str], list[float]] = {}
        self._rng = rng or random.Random(20261009)

    def sample(self, key: tuple[str, str, str]) -> float:
        alpha, beta = self._stats.get(key, [1.0, 1.0])
        return self._rng.betavariate(alpha, beta)

    def update(self, key: tuple[str, str, str], success: bool) -> None:
        stats = self._stats.setdefault(key, [1.0, 1.0])
        stats[0 if success else 1] += 1.0

    def mean(self, key: tuple[str, str, str]) -> float:
        alpha, beta = self._stats.get(key, [1.0, 1.0])
        return alpha / (alpha + beta)

    def observations(self, key: tuple[str, str, str]) -> float:
        alpha, beta = self._stats.get(key, [1.0, 1.0])
        return alpha + beta - 2.0

    def sample_blended(
        self, key: tuple[str, str, str], *, prior_mean: float = 0.5, strength: float = 2.0
    ) -> float:
        """按观测数把 Thompson 采样收缩到无信息先验。

        冷启动时没有任何数据，直接采样等于掷骰子，会让路由变成随机；
        观测越多，后验采样越占主导，探索行为才真正生效。
        """
        count = self.observations(key)
        trust = count / (count + strength)
        return trust * self.sample(key) + (1.0 - trust) * prior_mean

    def snapshot(self) -> dict[str, dict[str, float]]:
        return {
            "|".join(key): {
                "mean": round(self.mean(key), 4),
                "observations": self._stats[key][0] + self._stats[key][1] - 2.0,
            }
            for key in sorted(self._stats)
        }


@dataclass
class RouterConfig:
    mode: RouterMode = RouterMode.DYNAMIC
    # 成本厌恶系数：在“收益(0~1)”与“归一化成本(0~1)”之间做权衡
    lambda_cost: float = 0.25
    mu_latency: float = 0.05
    budget_usd: float | None = None
    expected_completion_tokens: int = 300


class Router:
    def __init__(
        self,
        *,
        tiers: tuple[ModelTier, ...] | None = None,
        config: RouterConfig | None = None,
        posterior: BetaPosterior | None = None,
        rng: random.Random | None = None,
        ledger: UsageLedger | None = None,
    ):
        self.tiers = tiers or default_tiers()
        self.config = config or RouterConfig()
        self.posterior = posterior or BetaPosterior(rng)
        self.ledger = ledger
        self._failures: dict[str, int] = {}
        self.decisions: list[dict[str, Any]] = []

    @property
    def mode(self) -> RouterMode:
        return self.config.mode

    def failures_for(self, step: str) -> int:
        return self._failures.get(step, 0)

    def choose(self, features: StepFeatures, *, spent_usd: float = 0.0) -> ModelTier:
        if self.mode is RouterMode.STRONG:
            tier = tier_by_name(self.tiers, STRONG)
        elif self.mode is RouterMode.WEAK:
            tier = tier_by_name(self.tiers, WEAK)
        elif self.mode is RouterMode.STATIC:
            tier = tier_by_name(self.tiers, STATIC_STEP_TIERS.get(features.step, WEAK))
        else:
            tier = self._choose_dynamic(features, spent_usd)
        self.decisions.append(
            {
                "step": features.step,
                "mode": self.mode.value,
                "tier": tier.name,
                "difficulty": round(estimate_difficulty(features), 4),
            }
        )
        return tier

    def _choose_dynamic(self, features: StepFeatures, spent_usd: float) -> ModelTier:
        difficulty = estimate_difficulty(features)
        bucket = difficulty_bucket(difficulty)
        lambda_cost = self.config.lambda_cost
        if self.config.budget_usd:
            # 对偶上升：预算消耗过半后逐步加大成本惩罚
            ratio = spent_usd / self.config.budget_usd
            lambda_cost *= 1.0 + max(0.0, ratio - 0.5) * 4.0

        # 成本与延迟都按“最贵档位”归一化到 0~1，才能与 0~1 的收益项同量纲比较
        costs = [
            tier.cost_usd(features.prompt_tokens, self.config.expected_completion_tokens)
            for tier in self.tiers
        ]
        latencies = [self._latency_prior(tier) for tier in self.tiers]
        max_cost = max(costs) or 1.0
        max_latency = max(latencies) or 1.0

        best_tier = self.tiers[0]
        best_score = float("-inf")
        for tier, cost, latency in zip(self.tiers, costs, latencies):
            # 收益：强模型能力恒为 1；弱模型在难题上按难度打折
            capability = 1.0 if tier.name == STRONG else max(0.0, 1.0 - difficulty)
            observed = self.posterior.sample_blended((features.step, bucket, tier.name))
            gain = 0.5 * capability + 0.5 * observed
            score = (
                gain
                - lambda_cost * (cost / max_cost)
                - self.config.mu_latency * (latency / max_latency)
            )
            if score > best_score:
                best_tier, best_score = tier, score
        return best_tier

    def _latency_prior(self, tier: ModelTier) -> float:
        if self.ledger and self.ledger.records:
            observed = [
                item.duration_sec for item in self.ledger.records if item.tier == tier.name
            ]
            if observed:
                return percentile(observed, 0.5)
        return LATENCY_PRIOR.get(tier.name, 2.0)

    def observe(self, features: StepFeatures, tier: ModelTier, success: bool) -> None:
        bucket = difficulty_bucket(estimate_difficulty(features))
        self.posterior.update((features.step, bucket, tier.name), success)
        if not success:
            self._failures[features.step] = self._failures.get(features.step, 0) + 1


def _estimate_tokens(text: str) -> int:
    if not text:
        return 0
    return max(1, (len(text) + 3) // 4)


@dataclass
class RoutedLLMClient:
    """把 Router 包在 LLM 客户端外面，让每个 step 的调用都被路由并记账。

    对被调用方（LlmProposer / diagnose）来说接口与普通 LLM 客户端一致。
    """

    router: Router
    ledger: UsageLedger = field(default_factory=UsageLedger)
    client_factory: Callable[..., Any] | None = None
    _clients: dict[str, Any] = field(default_factory=dict, repr=False)

    def available(self) -> bool:
        return any(client.available() for client in self._all_clients())

    def _all_clients(self) -> list[Any]:
        return [self._client(tier) for tier in self.router.tiers]

    def _client(self, tier: ModelTier) -> Any:
        if tier.name not in self._clients:
            factory = self.client_factory or _default_client_factory
            self._clients[tier.name] = factory(provider=tier.provider, model=tier.model)
        return self._clients[tier.name]

    def complete_with_usage(
        self,
        system_prompt: str,
        user_prompt: str,
        *,
        agent: str = "unknown",
        step: str = "complete",
        editable_files: int = 0,
        **kwargs: Any,
    ) -> Any:
        features = StepFeatures(
            step=step,
            prompt_tokens=_estimate_tokens(f"{system_prompt}\n{user_prompt}"),
            editable_files=editable_files,
            prior_failures=self.router.failures_for(step),
        )
        tier = self.router.choose(features, spent_usd=self.ledger.total_cost_usd)
        result = self._client(tier).complete_with_usage(
            system_prompt, user_prompt, agent=agent, step=step, **kwargs
        )
        self.ledger.record(result, tier)
        self.router.observe(features, tier, bool(getattr(result, "success", False)))
        return result


def _default_client_factory(**kwargs: Any) -> Any:
    from routepilot.runtime.llm_client import create_llm_client

    return create_llm_client(**kwargs)
