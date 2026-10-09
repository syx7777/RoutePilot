"""指标计算与 keep/rollback 决策。

完全由 manifest 的 `artifacts.columns` 与 `metrics` 驱动，不绑定任何业务口径。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from routepilot.adapter.manifest import ArtifactsSpec, MetricsSpec

PSEUDO_PREDICTION = "__routepilot_prediction__"
PSEUDO_ACTUAL = "__routepilot_actual__"

METRIC_FUNCTIONS = ("wape", "bias", "mae", "rmse", "mape", "smape")


class MetricError(RuntimeError):
    pass


def align_predictions(
    prediction_path: str | Path,
    actual_path: str | Path,
    artifacts: ArtifactsSpec,
) -> pd.DataFrame:
    """按 id + date 对齐预测与真实值，返回含 prediction / actual 两列的长表。"""
    columns = artifacts.columns
    keys = [*columns.id]
    if columns.date:
        keys.append(columns.date)
    if not keys:
        raise MetricError("artifacts.columns 至少需要 id 或 date 之一才能对齐预测与真实值")

    prediction = pd.read_csv(prediction_path)
    actual = pd.read_csv(actual_path)
    missing_prediction = [key for key in keys if key not in prediction.columns]
    missing_actual = [key for key in keys if key not in actual.columns]
    if missing_prediction:
        raise MetricError(f"预测文件缺少对齐列 {missing_prediction}: {prediction_path}")
    if missing_actual:
        raise MetricError(f"真实值文件缺少对齐列 {missing_actual}: {actual_path}")
    for frame, label, column in (
        (prediction, "预测", columns.prediction),
        (actual, "真实值", columns.actual),
    ):
        if column not in frame.columns:
            raise MetricError(f"{label}文件缺少目标列 {column!r}: {list(frame.columns)}")

    merged = prediction[keys + [columns.prediction]].merge(
        actual[keys + [columns.actual]], on=keys, how="inner"
    )
    if merged.empty:
        raise MetricError("预测与真实值按对齐列 join 后为空，请检查 artifacts.columns")
    renamed = merged.rename(
        columns={columns.prediction: PSEUDO_PREDICTION, columns.actual: PSEUDO_ACTUAL}
    )
    return renamed[[*keys, PSEUDO_PREDICTION, PSEUDO_ACTUAL]].astype(
        {PSEUDO_PREDICTION: float, PSEUDO_ACTUAL: float}
    )


def compute_metrics(
    prediction_path: str | Path,
    actual_path: str | Path,
    artifacts: ArtifactsSpec,
    metrics: MetricsSpec,
) -> dict[str, float]:
    frame = align_predictions(prediction_path, actual_path, artifacts)
    prediction = frame[PSEUDO_PREDICTION]
    actual = frame[PSEUDO_ACTUAL]
    error = prediction - actual

    names = {metrics.primary.name, *(guard.name for guard in metrics.guards)}
    values: dict[str, float] = {"sample_count": float(len(frame))}
    for name in sorted(names):
        values[name] = _metric_value(name, prediction, actual, error)
    return values


def _metric_value(
    name: str, prediction: pd.Series, actual: pd.Series, error: pd.Series
) -> float:
    scale = float(actual.abs().sum())
    if name == "wape":
        return _safe_divide(float(error.abs().sum()), scale)
    if name == "bias":
        return _safe_divide(float(error.sum()), scale)
    if name == "mae":
        return float(error.abs().mean())
    if name == "rmse":
        return float((error.pow(2).mean()) ** 0.5)
    if name == "mape":
        mask = actual.abs() > 0
        return _safe_divide(float((error.abs() / actual.abs())[mask].sum()), float(mask.sum()))
    if name == "smape":
        denominator = prediction.abs() + actual.abs()
        mask = denominator > 0
        ratio = (2 * error.abs() / denominator)[mask]
        return _safe_divide(float(ratio.sum()), float(mask.sum()))
    raise MetricError(f"不支持的指标 {name!r}；可用：{', '.join(METRIC_FUNCTIONS)}")


def _safe_divide(numerator: float, denominator: float) -> float:
    if denominator == 0:
        return float("nan")
    return numerator / denominator


def decide(
    baseline: dict[str, float],
    candidate: dict[str, float] | None,
    *,
    metrics: MetricsSpec,
    run_success: bool,
    reject_reason: str = "",
) -> dict[str, Any]:
    """按 manifest 的 primary + guards 决定 keep / rollback。"""
    primary = metrics.primary
    if not run_success or candidate is None:
        return {
            "decision": "rollback",
            "reason": reject_reason or "run failed",
            "improvement": 0.0,
            "guard_violations": [],
        }

    old_count = baseline.get("sample_count")
    new_count = candidate.get("sample_count")
    if old_count is not None and new_count is not None and old_count != new_count:
        return {
            "decision": "rollback",
            "reason": f"evaluation sample mismatch: baseline={old_count:g} candidate={new_count:g}",
            "improvement": 0.0,
            "guard_violations": [],
        }

    old_value = baseline.get(primary.name)
    new_value = candidate.get(primary.name)
    if old_value is None or new_value is None:
        return {
            "decision": "rollback",
            "reason": f"primary metric {primary.name!r} missing",
            "improvement": 0.0,
            "guard_violations": [],
        }

    improvement = (
        old_value - new_value if primary.direction == "minimize" else new_value - old_value
    )
    improved = improvement >= primary.min_delta

    violations: list[str] = []
    for guard in metrics.guards:
        if guard.name not in baseline or guard.name not in candidate:
            continue
        delta = abs(candidate[guard.name]) - abs(baseline[guard.name])
        if delta > guard.max_regression:
            violations.append(
                f"{guard.name} 恶化 {delta:.4f} > {guard.max_regression:.4f}"
            )

    if improved and not violations:
        reason = f"{primary.name} 改善 {improvement:.4f} ≥ {primary.min_delta}"
        decision = "keep"
    elif violations:
        reason = "护栏指标越界: " + "; ".join(violations)
        decision = "rollback"
    else:
        reason = f"{primary.name} 改善 {improvement:.4f} < {primary.min_delta}"
        decision = "rollback"
    return {
        "decision": decision,
        "reason": reason,
        "improvement": improvement,
        "guard_violations": violations,
    }
