from __future__ import annotations

from typing import Any


DEFAULT_THRESHOLDS = {
    "min_wape_improvement": 0.005,
    "max_bias_regression": 0.02,
}


def compare_metrics(
    old_metrics: dict[str, Any],
    new_metrics: dict[str, Any],
    run_status: dict[str, Any],
    thresholds: dict[str, float] | None = None,
) -> dict[str, Any]:
    limits = {**DEFAULT_THRESHOLDS, **(thresholds or {})}
    old_wape = float(old_metrics["wape"])
    new_wape = float(new_metrics["wape"])
    old_bias = float(old_metrics["bias"])
    new_bias = float(new_metrics["bias"])
    wape_delta = new_wape - old_wape
    bias_delta = abs(new_bias) - abs(old_bias)
    old_sample_count = _sample_count(old_metrics)
    new_sample_count = _sample_count(new_metrics)
    sample_count_consistent = _sample_count_consistent(old_sample_count, new_sample_count)

    train_success = bool(run_status.get("train_success"))
    eval_success = bool(run_status.get("eval_success"))
    keep = (
        new_wape <= old_wape - limits["min_wape_improvement"]
        and abs(new_bias) <= abs(old_bias) + limits["max_bias_regression"]
        and train_success
        and eval_success
        and sample_count_consistent is not False
    )
    if run_status.get("evaluation_context_consistent") is False or run_status.get("failure_stage") == "evaluation_context_drift":
        reason = "evaluation context drift"
    elif run_status.get("train_returncode") == "agent2_code_generation_failed" or run_status.get("agent2_code_generation_success") is False:
        reason = "agent2 code generation failed"
    elif run_status.get("feature_application_success") is False:
        reason = "feature change was not applied"
    elif keep:
        reason = "wape improved enough and bias stayed within threshold"
    elif not train_success or not eval_success:
        reason = "train or evaluation failed"
    elif sample_count_consistent is False:
        reason = "evaluation sample mismatch"
    elif new_wape > old_wape - limits["min_wape_improvement"]:
        reason = "wape improvement is below threshold"
    else:
        reason = "bias regression exceeds threshold"
    return {
        "decision": "keep" if keep else "rollback",
        "reason": reason,
        "wape_delta": wape_delta,
        "bias_delta": bias_delta,
        "old_wape": old_wape,
        "new_wape": new_wape,
        "old_bias": old_bias,
        "new_bias": new_bias,
        "old_sample_count": old_sample_count,
        "new_sample_count": new_sample_count,
        "sample_count_consistent": sample_count_consistent,
    }


def _sample_count(metrics: dict[str, Any]) -> int | None:
    for key in ("sample_count", "rows"):
        if key in metrics and metrics[key] is not None:
            return int(float(metrics[key]))
    return None


def _sample_count_consistent(old_count: int | None, new_count: int | None) -> bool | None:
    if old_count is None or new_count is None:
        return None
    return old_count == new_count
