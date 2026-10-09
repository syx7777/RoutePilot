"""瓶颈诊断与回流动作。

Profiling → Bottleneck Diagnosis → Optimization：把 span 归因结果翻译成
可执行的调参动作，再交回 Router 影响后续决策。
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

from routepilot.profiler.spans import EVALUATE, LLM_INFERENCE, PROJECT_RUN, Profiler
from routepilot.router.usage import UsageLedger

# 单步 LLM P95 超过该值即视为"该 step 是延迟瓶颈"
STEP_LATENCY_P95_THRESHOLD = 10.0
# 某一 category 占比超过该值即视为瓶颈
DOMINANT_SHARE = 0.5


@dataclass
class Finding:
    bottleneck: str
    share: float
    severity: str
    detail: str
    directive: dict[str, Any] = field(default_factory=dict)


@dataclass
class ProfileDiagnosis:
    instrumented_seconds: float
    attribution: dict[str, float]
    findings: list[Finding] = field(default_factory=list)
    directives: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def diagnose_profile(
    profiler: Profiler,
    ledger: UsageLedger | None = None,
) -> ProfileDiagnosis:
    attribution = dict(profiler.attribution())
    # LLM 推理耗时直接来自调用台账，避免让 router 反向依赖 profiler
    if ledger is not None and ledger.records:
        attribution[LLM_INFERENCE] = attribution.get(LLM_INFERENCE, 0.0) + sum(
            record.duration_sec for record in ledger.records
        )
        attribution = dict(
            sorted(attribution.items(), key=lambda pair: pair[1], reverse=True)
        )
    wall = sum(attribution.values()) or 1e-9
    findings: list[Finding] = []
    directives: list[dict[str, Any]] = []

    for category, seconds in attribution.items():
        share = seconds / wall
        if share < DOMINANT_SHARE or category == EVALUATE:
            continue
        severity = "high" if share >= 0.75 else "medium"
        if category == PROJECT_RUN:
            findings.append(
                Finding(
                    bottleneck=category,
                    share=round(share, 4),
                    severity=severity,
                    detail="项目自身的训练/预测耗时占主导，LLM 路由可优化空间有限",
                    directive={"kind": "note", "message": "考虑并行 trial 或缩小子采样窗口"},
                )
            )
        elif category == LLM_INFERENCE:
            findings.append(
                Finding(
                    bottleneck=category,
                    share=round(share, 4),
                    severity=severity,
                    detail="LLM 推理占主导，应优先压低延迟权重高的 step",
                    directive={"kind": "raise_latency_weight_all", "multiplier": 2.0},
                )
            )

    if ledger is not None:
        for step, item in (ledger.summary().get("by_step") or {}).items():
            if item["latency_p95_sec"] < STEP_LATENCY_P95_THRESHOLD:
                continue
            findings.append(
                Finding(
                    bottleneck=f"step:{step}",
                    share=round(item["latency_p95_sec"] / wall, 4),
                    severity="medium",
                    detail=(
                        f"{step} 的 LLM P95 延迟 {item['latency_p95_sec']:.2f}s "
                        f"≥ {STEP_LATENCY_P95_THRESHOLD:.0f}s，倾向选择更低延迟的档位"
                    ),
                    directive={
                        "kind": "raise_step_latency_weight",
                        "step": step,
                        "multiplier": 2.5,
                    },
                )
            )

    for finding in findings:
        if finding.directive:
            directives.append(finding.directive)
    return ProfileDiagnosis(
        instrumented_seconds=round(wall, 4),
        attribution={key: round(value, 4) for key, value in attribution.items()},
        findings=findings,
        directives=directives,
    )
