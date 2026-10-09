"""通用优化闭环：baseline → 提案 → 快照 → 应用 → 运行 → 评估 → keep/rollback。

整条链路只通过 `ProjectAdapter` 与被接入项目交互，不感知任何业务语义。
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable

from routepilot.adapter.base import ProjectAdapter
from routepilot.adapter.manifest import ProjectManifest
from routepilot.loop.proposer import ProposalContext, Proposer, validate_proposal
from routepilot.metrics import MetricError, compute_metrics, decide

LogFn = Callable[[str], None]


@dataclass
class TrialRecord:
    trial_index: int
    summary: str
    rationale: str
    files: list[str]
    decision: str
    reason: str
    metrics: dict[str, float] | None
    improvement: float
    run_success: bool
    duration_sec: float
    guard_violations: list[str] = field(default_factory=list)


@dataclass
class OptimizationOutcome:
    project: str
    goal: str
    primary_metric: str
    baseline_metrics: dict[str, float]
    final_metrics: dict[str, float]
    trials: list[TrialRecord]
    kept_trials: list[int]
    total_duration_sec: float
    # keep 是就地生效的，因此必须留档 editable 文件的原始内容，才能事后回滚。
    original_editable_files: dict[str, str] = field(default_factory=dict)
    routing: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["primary_improvement"] = self.baseline_metrics.get(
            self.primary_metric, float("nan")
        ) - self.final_metrics.get(self.primary_metric, float("nan"))
        return payload


def run_optimization(
    adapter: ProjectAdapter,
    manifest: ProjectManifest,
    *,
    goal: str,
    proposer: Proposer,
    diagnoser: Callable[[ProposalContext], str] | None = None,
    ledger: Any = None,
    router: Any = None,
    max_trials: int | None = None,
    log: LogFn = print,
) -> OptimizationOutcome:
    import time

    started = time.perf_counter()
    budget = max_trials if max_trials is not None else manifest.budget.max_trials

    discovery = adapter.discover()
    if discovery.confidence == 0.0:
        raise RuntimeError(
            "项目未通过接入检查，无法运行优化闭环：" + "; ".join(discovery.warnings)
        )
    for warning in discovery.warnings:
        log(f"[discover] warning: {warning}")

    log("[baseline] 运行项目原始配置")
    original_editable_files = _editable_contents(adapter)
    baseline_run = adapter.run()
    baseline_metrics = _metrics_from_run(adapter, manifest, baseline_run)
    log(f"[baseline] {_format_metrics(baseline_metrics, manifest)}")

    best_metrics = dict(baseline_metrics)
    trials: list[TrialRecord] = []
    kept: list[int] = []

    for trial_index in range(1, budget + 1):
        context = ProposalContext(
            goal=goal,
            trial_index=trial_index,
            baseline_metrics=baseline_metrics,
            best_metrics=best_metrics,
            history=[asdict(record) for record in trials],
            editable_files=_editable_contents(adapter),
        )
        if diagnoser is not None:
            context.diagnosis = diagnoser(context) or ""
            if context.diagnosis:
                log(f"[trial {trial_index}] 诊断: {context.diagnosis}")
        proposal = proposer.propose(context)
        if proposal is None:
            log(f"[trial {trial_index}] proposer 不再给出候选，提前结束")
            break

        ok, error = validate_proposal(proposal, manifest)
        if not ok:
            log(f"[trial {trial_index}] 提案被拒绝: {error}")
            trials.append(
                TrialRecord(
                    trial_index=trial_index,
                    summary=proposal.summary,
                    rationale=proposal.rationale,
                    files=[edit.path for edit in proposal.edits],
                    decision="rollback",
                    reason=f"proposal rejected: {error}",
                    metrics=None,
                    improvement=0.0,
                    run_success=False,
                    duration_sec=0.0,
                )
            )
            continue

        log(f"[trial {trial_index}] {proposal.summary}")
        snapshot = adapter.snapshot()
        trial_started = time.perf_counter()
        try:
            _apply_edits(adapter, proposal)
            run_result = adapter.run()
            candidate_metrics = (
                _metrics_from_run(adapter, manifest, run_result)
                if run_result.success
                else None
            )
            verdict = decide(
                best_metrics,
                candidate_metrics,
                metrics=manifest.metrics,
                run_success=run_result.success and candidate_metrics is not None,
                reject_reason=run_result.error or "run failed",
            )
        except (MetricError, OSError, ValueError) as exc:
            adapter.rollback(snapshot)
            verdict = {
                "decision": "rollback",
                "reason": f"{type(exc).__name__}: {exc}",
                "improvement": 0.0,
                "guard_violations": [],
            }
            candidate_metrics = None
            run_result = None
        duration = time.perf_counter() - trial_started

        if verdict["decision"] == "keep":
            best_metrics = dict(candidate_metrics or {})
            kept.append(trial_index)
        else:
            adapter.rollback(snapshot)

        log(
            f"[trial {trial_index}] {verdict['decision']}: {verdict['reason']}"
            + (f" | {_format_metrics(candidate_metrics, manifest)}" if candidate_metrics else "")
        )
        trials.append(
            TrialRecord(
                trial_index=trial_index,
                summary=proposal.summary,
                rationale=proposal.rationale,
                files=[edit.path for edit in proposal.edits],
                decision=verdict["decision"],
                reason=verdict["reason"],
                metrics=candidate_metrics,
                improvement=float(verdict.get("improvement") or 0.0),
                run_success=bool(run_result and run_result.success),
                duration_sec=duration,
                guard_violations=list(verdict.get("guard_violations") or []),
            )
        )

    return OptimizationOutcome(
        project=manifest.project.name,
        goal=goal,
        primary_metric=manifest.metrics.primary.name,
        baseline_metrics=baseline_metrics,
        final_metrics=best_metrics,
        trials=trials,
        kept_trials=kept,
        total_duration_sec=time.perf_counter() - started,
        original_editable_files=original_editable_files,
        routing=ledger.summary() if ledger is not None else {},
    )


def _metrics_from_run(adapter: ProjectAdapter, manifest: ProjectManifest, run_result: Any):
    if not run_result.success:
        raise MetricError(run_result.error or "run failed")
    if not run_result.prediction_path or not run_result.actual_path:
        raise MetricError(
            "run 未产出预测或真实值文件："
            f"prediction={run_result.prediction_path} actual={run_result.actual_path}"
        )
    return compute_metrics(
        run_result.prediction_path,
        run_result.actual_path,
        manifest.artifacts,
        manifest.metrics,
    )


def _editable_contents(adapter: ProjectAdapter) -> dict[str, str]:
    contents: dict[str, str] = {}
    for relative in adapter.editable_files():
        path = adapter.project_root / relative
        try:
            contents[relative] = path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            continue
    return contents


def _apply_edits(adapter: ProjectAdapter, proposal) -> None:
    for edit in proposal.edits:
        target: Path = adapter.project_root / edit.path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(edit.content, encoding="utf-8")


def _format_metrics(metrics: dict[str, float] | None, manifest: ProjectManifest) -> str:
    if not metrics:
        return "<no metrics>"
    names = [manifest.metrics.primary.name, *(guard.name for guard in manifest.metrics.guards)]
    parts = [f"{name}={metrics[name]:.4f}" for name in names if name in metrics]
    if "sample_count" in metrics:
        parts.append(f"rows={int(metrics['sample_count'])}")
    return " ".join(parts)
