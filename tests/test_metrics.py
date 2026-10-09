from __future__ import annotations

from pathlib import Path

import pytest

from routepilot.adapter.manifest import ArtifactsSpec, MetricsSpec
from routepilot.metrics import MetricError, align_predictions, compute_metrics, decide

ACTUAL_ROWS = "unique_id,ds,y\nS001,2024-01-01,100\nS001,2024-01-02,100\nS002,2024-01-01,200\n"
PREDICTION_ROWS = "unique_id,ds,yhat\nS001,2024-01-01,110\nS001,2024-01-02,110\nS002,2024-01-01,180\n"


def _artifacts(**overrides) -> ArtifactsSpec:
    payload = {
        "prediction": "prediction.csv",
        "actual": "actual.csv",
        "columns": {"prediction": "yhat", "actual": "y", "date": "ds", "id": ["unique_id"]},
    }
    payload.update(overrides)
    return ArtifactsSpec.model_validate(payload)


def _metrics_spec(min_delta: float = 0.01, guards: list[dict] | None = None) -> MetricsSpec:
    return MetricsSpec.model_validate(
        {
            "primary": {"name": "wape", "direction": "minimize", "min_delta": min_delta},
            # 默认带一个 bias 护栏：compute_metrics 只会计算 primary + guards 里声明的指标。
            "guards": [{"name": "bias", "max_regression": 0.02}] if guards is None else guards,
        }
    )


def _write_pair(tmp_path: Path, prediction: str = PREDICTION_ROWS, actual: str = ACTUAL_ROWS):
    prediction_path = tmp_path / "prediction.csv"
    actual_path = tmp_path / "actual.csv"
    prediction_path.write_text(prediction, encoding="utf-8")
    actual_path.write_text(actual, encoding="utf-8")
    return prediction_path, actual_path


def test_align_predictions_joins_on_id_and_date(tmp_path: Path) -> None:
    prediction_path, actual_path = _write_pair(tmp_path)
    frame = align_predictions(prediction_path, actual_path, _artifacts())

    assert len(frame) == 3
    assert list(frame.columns)[-2:] == ["__routepilot_prediction__", "__routepilot_actual__"]


def test_compute_metrics_wape_and_bias(tmp_path: Path) -> None:
    prediction_path, actual_path = _write_pair(tmp_path)
    values = compute_metrics(prediction_path, actual_path, _artifacts(), _metrics_spec())

    # 误差 = (110-100, 110-100, 180-200) -> |e| 合计 40, |a| 合计 400
    assert values["wape"] == pytest.approx(0.1)
    assert values["bias"] == pytest.approx(0.0)
    assert values["sample_count"] == 3.0


def test_compute_metrics_rejects_missing_alignment_column(tmp_path: Path) -> None:
    prediction_path, actual_path = _write_pair(tmp_path, prediction="ds,yhat\n2024-01-01,1\n")
    with pytest.raises(MetricError, match="缺少对齐列"):
        compute_metrics(prediction_path, actual_path, _artifacts(), _metrics_spec())


def test_align_predictions_requires_at_least_one_key(tmp_path: Path) -> None:
    prediction_path, actual_path = _write_pair(tmp_path)
    artifacts = _artifacts(columns={"prediction": "yhat", "actual": "y"})
    with pytest.raises(MetricError, match="至少需要 id 或 date"):
        align_predictions(prediction_path, actual_path, artifacts)


def test_decide_keeps_on_sufficient_improvement() -> None:
    verdict = decide(
        {"wape": 0.10, "sample_count": 3},
        {"wape": 0.08, "sample_count": 3},
        metrics=_metrics_spec(min_delta=0.01),
        run_success=True,
    )
    assert verdict["decision"] == "keep"
    assert verdict["improvement"] == pytest.approx(0.02)


def test_decide_rolls_back_below_min_delta() -> None:
    verdict = decide(
        {"wape": 0.10, "sample_count": 3},
        {"wape": 0.095, "sample_count": 3},
        metrics=_metrics_spec(min_delta=0.01),
        run_success=True,
    )
    assert verdict["decision"] == "rollback"
    assert "改善" in verdict["reason"]


def test_decide_rolls_back_on_guard_violation() -> None:
    verdict = decide(
        {"wape": 0.10, "bias": 0.01, "sample_count": 3},
        {"wape": 0.05, "bias": 0.09, "sample_count": 3},
        metrics=_metrics_spec(min_delta=0.01, guards=[{"name": "bias", "max_regression": 0.02}]),
        run_success=True,
    )
    assert verdict["decision"] == "rollback"
    assert verdict["guard_violations"]


def test_decide_rolls_back_on_sample_count_mismatch() -> None:
    verdict = decide(
        {"wape": 0.10, "sample_count": 3},
        {"wape": 0.01, "sample_count": 4},
        metrics=_metrics_spec(min_delta=0.01),
        run_success=True,
    )
    assert verdict["decision"] == "rollback"
    assert "sample mismatch" in verdict["reason"]


def test_decide_rolls_back_when_run_failed() -> None:
    verdict = decide(
        {"wape": 0.10, "sample_count": 3},
        None,
        metrics=_metrics_spec(),
        run_success=False,
        reject_reason="boom",
    )
    assert verdict["decision"] == "rollback"
    assert verdict["reason"] == "boom"


def test_decide_supports_maximize_direction() -> None:
    metrics = MetricsSpec.model_validate(
        {"primary": {"name": "score", "direction": "maximize", "min_delta": 0.01}}
    )
    verdict = decide(
        {"score": 0.80, "sample_count": 3},
        {"score": 0.85, "sample_count": 3},
        metrics=metrics,
        run_success=True,
    )
    assert verdict["decision"] == "keep"
    assert verdict["improvement"] == pytest.approx(0.05)
