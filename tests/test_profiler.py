from __future__ import annotations

import time

from routepilot.profiler import LLM_INFERENCE, PROJECT_RUN, Profiler, diagnose_profile
from routepilot.router.router import Router, RouterConfig, RouterMode, StepFeatures
from routepilot.router.tiers import STRONG, WEAK
from routepilot.router.usage import CallRecord, UsageLedger


def _ledger(step: str, duration: float, *, tier: str = STRONG) -> UsageLedger:
    ledger = UsageLedger()
    ledger.records.append(
        CallRecord(
            agent="A",
            step=step,
            tier=tier,
            model="m",
            prompt_tokens=100,
            completion_tokens=10,
            cost_usd=0.001,
            duration_sec=duration,
            success=True,
        )
    )
    return ledger


def test_profiler_attributes_spans_by_category() -> None:
    profiler = Profiler()
    with profiler.span("baseline_run", PROJECT_RUN):
        time.sleep(0.01)
    with profiler.span("trial_1_run", PROJECT_RUN):
        time.sleep(0.01)

    attribution = profiler.attribution()
    assert list(attribution) == [PROJECT_RUN]
    assert attribution[PROJECT_RUN] > 0
    assert set(profiler.by_name()) == {"baseline_run", "trial_1_run"}
    assert len(profiler.to_list()) == 2


def test_profiler_records_duration_of_block() -> None:
    profiler = Profiler()
    with profiler.span("work", PROJECT_RUN):
        total = sum(range(200_000))
    assert total > 0
    assert profiler.spans[0].duration_sec > 0


def test_diagnose_flags_dominant_project_run() -> None:
    profiler = Profiler()
    with profiler.span("baseline_run", PROJECT_RUN):
        time.sleep(0.01)
    diagnosis = diagnose_profile(profiler)

    assert diagnosis.attribution[PROJECT_RUN] > 0
    assert any(item.bottleneck == PROJECT_RUN for item in diagnosis.findings)
    assert diagnosis.findings[0].severity == "high"


def test_diagnose_flags_dominant_llm_inference() -> None:
    profiler = Profiler()
    with profiler.span("baseline_run", PROJECT_RUN):
        time.sleep(0.01)
    diagnosis = diagnose_profile(profiler, _ledger("ProposeExperiment", 30.0))

    assert LLM_INFERENCE in diagnosis.attribution
    assert diagnosis.attribution[LLM_INFERENCE] > diagnosis.attribution[PROJECT_RUN]
    assert any(item.bottleneck == LLM_INFERENCE for item in diagnosis.findings)


def test_diagnose_flags_slow_step_and_emits_directive() -> None:
    profiler = Profiler()
    with profiler.span("baseline_run", PROJECT_RUN):
        time.sleep(0.01)
    diagnosis = diagnose_profile(profiler, _ledger("ProposeExperiment", 25.0))

    slow = [item for item in diagnosis.findings if item.bottleneck == "step:ProposeExperiment"]
    assert slow, diagnosis.findings
    assert slow[0].directive == {
        "kind": "raise_step_latency_weight",
        "step": "ProposeExperiment",
        "multiplier": 2.5,
    }


def test_fast_steps_do_not_emit_step_directives() -> None:
    profiler = Profiler()
    with profiler.span("baseline_run", PROJECT_RUN):
        time.sleep(0.01)
    diagnosis = diagnose_profile(profiler, _ledger("Diagnose", 1.0))
    assert not [item for item in diagnosis.findings if item.bottleneck.startswith("step:")]


def test_router_apply_directives_raises_step_latency_weight() -> None:
    router = Router(config=RouterConfig(mode=RouterMode.DYNAMIC))
    applied = router.apply_directives(
        [{"kind": "raise_step_latency_weight", "step": "ProposeExperiment", "multiplier": 2.5}]
    )
    assert router.step_latency_multipliers["ProposeExperiment"] == 2.5
    assert applied and "ProposeExperiment" in applied[0]

    # 重复下发取较大值，不会来回抖动
    router.apply_directives(
        [{"kind": "raise_step_latency_weight", "step": "ProposeExperiment", "multiplier": 1.5}]
    )
    assert router.step_latency_multipliers["ProposeExperiment"] == 2.5


def test_latency_directive_can_push_a_step_to_the_cheaper_tier() -> None:
    """放大延迟权重后，原本走强模型的难题 step 应改走弱模型（延迟更低）。"""
    features = StepFeatures(step="ProposeExperiment", prompt_tokens=0)
    baseline = Router(config=RouterConfig(mode=RouterMode.DYNAMIC)).choose(features)
    assert baseline.name == STRONG

    pressured = Router(config=RouterConfig(mode=RouterMode.DYNAMIC, mu_latency=40.0))
    assert pressured.choose(features).name == WEAK
