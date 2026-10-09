from __future__ import annotations

from types import SimpleNamespace

import pytest

from routepilot.router import (
    BetaPosterior,
    RoutedLLMClient,
    Router,
    RouterConfig,
    RouterMode,
    StepFeatures,
    UsageLedger,
    default_tiers,
    difficulty_bucket,
    estimate_difficulty,
    percentile,
    tier_by_name,
)
from routepilot.router.tiers import STRONG, WEAK


class FakeLLM:
    def __init__(self, provider: str | None = None, model: str | None = None, *, success: bool = True):
        self.provider = provider
        self.model = model
        self.success = success
        self.calls: list[str] = []

    def available(self) -> bool:
        return True

    def complete_with_usage(self, system_prompt, user_prompt, *, agent="unknown", step="complete", **kwargs):
        self.calls.append(step)
        usage = {"prompt_tokens": 1000, "completion_tokens": 200, "total_tokens": 1200}
        return SimpleNamespace(
            content="ok" if self.success else "",
            model=self.model,
            agent=agent,
            step=step,
            duration_seconds=2.0,
            usage=usage,
            success=self.success,
            attempt_count=1,
        )


def _router(mode: RouterMode, **kwargs) -> Router:
    return Router(config=RouterConfig(mode=mode, **kwargs))


def test_tier_cost_uses_token_prices() -> None:
    tiers = default_tiers()
    weak = tier_by_name(tiers, WEAK)
    strong = tier_by_name(tiers, STRONG)
    assert weak.cost_usd(1_000_000, 1_000_000) == pytest.approx(0.75)
    assert strong.cost_usd(1_000_000, 1_000_000) == pytest.approx(2.64)
    assert strong.cost_usd(0, 0) == 0.0


def test_percentile_interpolates() -> None:
    assert percentile([], 0.5) == 0.0
    assert percentile([3.0], 0.95) == 3.0
    assert percentile([1.0, 2.0, 3.0], 0.5) == pytest.approx(2.0)
    assert percentile([0.0, 10.0], 0.95) == pytest.approx(9.5)


def test_usage_ledger_groups_by_tier_and_step() -> None:
    tiers = default_tiers()
    ledger = UsageLedger()
    ledger.record(FakeLLM().complete_with_usage("", "", agent="A", step="Diagnose"), tier_by_name(tiers, WEAK))
    ledger.record(FakeLLM().complete_with_usage("", "", agent="A", step="ProposeExperiment"), tier_by_name(tiers, STRONG))

    summary = ledger.summary()
    assert summary["calls"] == 2
    assert summary["tokens"] == 2400
    assert set(summary["by_tier"]) == {WEAK, STRONG}
    assert summary["by_step"]["Diagnose"]["calls"] == 1
    assert summary["success_rate"] == 1.0
    assert summary["cost_usd"] > 0


def test_difficulty_orders_step_types() -> None:
    diagnose = estimate_difficulty(StepFeatures(step="Diagnose", prompt_tokens=0))
    propose = estimate_difficulty(StepFeatures(step="ProposeExperiment", prompt_tokens=0))
    assert diagnose < propose
    assert difficulty_bucket(diagnose) == "easy"
    assert difficulty_bucket(propose) == "hard"


def test_difficulty_grows_with_context_and_failures() -> None:
    base = estimate_difficulty(StepFeatures(step="Diagnose", prompt_tokens=0))
    bigger = estimate_difficulty(StepFeatures(step="Diagnose", prompt_tokens=8000))
    failing = estimate_difficulty(
        StepFeatures(step="Diagnose", prompt_tokens=0, prior_failures=3)
    )
    assert base < bigger
    assert base < failing


def test_static_and_strong_weak_modes_are_fixed() -> None:
    features = StepFeatures(step="Diagnose", prompt_tokens=0)
    assert _router(RouterMode.STRONG).choose(features).name == STRONG
    assert _router(RouterMode.WEAK).choose(features).name == WEAK
    static = _router(RouterMode.STATIC)
    assert static.choose(features).name == WEAK
    assert static.choose(StepFeatures(step="ProposeExperiment", prompt_tokens=0)).name == STRONG


def test_dynamic_router_matches_difficulty_to_tier() -> None:
    dynamic = _router(RouterMode.DYNAMIC)
    easy = dynamic.choose(StepFeatures(step="Diagnose", prompt_tokens=0))
    hard = dynamic.choose(StepFeatures(step="ProposeExperiment", prompt_tokens=0))
    assert easy.name == WEAK
    assert hard.name == STRONG


def test_dynamic_router_escalates_after_repeated_failures() -> None:
    dynamic = _router(RouterMode.DYNAMIC)
    features = StepFeatures(step="Diagnose", prompt_tokens=0)
    weak_tier = tier_by_name(dynamic.tiers, WEAK)
    assert dynamic.choose(features).name == WEAK

    for _ in range(3):
        dynamic.observe(features, weak_tier, success=False)

    assert dynamic.choose(features).name == STRONG


def test_budget_dual_penalises_cost_as_budget_is_consumed() -> None:
    features = StepFeatures(step="ProposeExperiment", prompt_tokens=0)
    no_budget = _router(RouterMode.DYNAMIC, lambda_cost=0.0).choose(features)
    tight = _router(RouterMode.DYNAMIC, lambda_cost=5.0, budget_usd=0.001)
    # 预算几乎耗尽时，成本惩罚被对偶上升放大，倾向弱模型
    assert tight.choose(features, spent_usd=0.01).name == WEAK
    assert no_budget.name == STRONG


def test_beta_posterior_tracks_success_rate() -> None:
    posterior = BetaPosterior()
    key = ("Diagnose", "easy", WEAK)
    baseline = posterior.mean(key)
    posterior.update(key, success=True)
    assert posterior.mean(key) > baseline
    snapshot = posterior.snapshot()
    assert "Diagnose|easy|weak" in snapshot


def test_routed_client_records_usage_and_routes_by_step() -> None:
    ledger = UsageLedger()
    router = Router(config=RouterConfig(mode=RouterMode.DYNAMIC), ledger=ledger)
    created: dict[str, FakeLLM] = {}

    def factory(provider=None, model=None):
        created[model] = FakeLLM(provider=provider, model=model)
        return created[model]

    client = RoutedLLMClient(router=router, ledger=ledger, client_factory=factory)
    assert client.available() is True

    client.complete_with_usage("s", "u", agent="A", step="Diagnose")
    client.complete_with_usage("s", "u", agent="A", step="ProposeExperiment")

    assert created["deepseek-flash"].calls == ["Diagnose"]
    assert created["deepseek-v4-pro"].calls == ["ProposeExperiment"]
    assert [record.tier for record in ledger.records] == [WEAK, STRONG]
    assert router.failures_for("Diagnose") == 0
